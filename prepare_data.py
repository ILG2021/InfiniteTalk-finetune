"""
InfiniteTalk LoRA 微调训练 - 数据预处理脚本

功能:
1. 从视频中提取 wav2vec2 音频嵌入
2. 生成 metadata.json

Usage:
    python prepare_data.py --video_dir ./raw_videos --output_dir ./training_data

数据准备要求:
- 准备多段同一人说话的视频 (.mp4)
- 视频应有清晰的音频轨道
- 建议每段视频 5-60 秒
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm

def get_crop_params(video_path, target_w, target_h):
    """Detect face and calculate crop dimensions to keep face centered using robust multi-frame sampling."""
    import cv2
    import os
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Heuristic: check multiple frames (10%, 33%, 50%) to find a face
    sample_points = [total_frames // 10, total_frames // 3, total_frames // 2]
    cx, cy = w // 2, h // 2  # Default center
    face_found = False

    # Use 'full_range' model (~2.3MB) which is better for studio/mid-range shots than 'short_range'
    model_path = os.path.join('weights', 'face_detector_full.tflite')
    if not os.path.exists(model_path):
        os.makedirs('weights', exist_ok=True)
        try:
            import urllib.request
            print(f"Downloading robust full-range face detector to {model_path}...")
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite"
            urllib.request.urlretrieve(url, model_path)
        except Exception as e:
            print(f"Auto-download failed: {e}. Falling back to default center if model missing.")

    try:
        if os.path.exists(model_path):
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceDetectorOptions(base_options=base_options)

            with vision.FaceDetector.create_from_options(options) as detector:
                for sp in sample_points:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, sp)
                    ret, frame = cap.read()
                    if not ret: continue

                    # Convert to MediaPipe Image
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    detection_result = detector.detect(mp_image)

                    if detection_result.detections:
                        bbox = detection_result.detections[0].bounding_box
                        cx = int(bbox.origin_x + bbox.width / 2)
                        cy = int(bbox.origin_y + bbox.height / 2)
                        face_found = True
                        print(f"  Face detected at frame {sp}: position ({cx}, {cy})")
                        break # Found a face, use these coordinates
    except Exception as e:
        print(f"  MediaPipe Detection Failed: {e}. Using center-crop fallback.")

    cap.release()
    if not face_found:
        print("  Warning: No face detected in sample frames. Using center-crop fallback.")

    # Calculate crop dimensions
    target_ratio = target_w / target_h
    src_ratio = w / h
    
    if src_ratio > target_ratio:
        # Source is wider, crop the width
        crop_h = h
        crop_w = int(h * target_ratio)
        crop_y = 0
        crop_x = int(cx - crop_w / 2)
        crop_x = max(0, min(crop_x, w - crop_w))
    else:
        # Source is taller, crop the height
        crop_w = w
        crop_h = int(w / target_ratio)
        crop_x = 0
        crop_y = int(cy - crop_h / 2)
        crop_y = max(0, min(crop_y, h - crop_h))
        
    return crop_w, crop_h, crop_x, crop_y


def extract_audio_from_video(video_path, target_sr=16000):
    """Extract audio from video and resample to 16kHz."""
    waveform, sr = torchaudio.load(video_path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0), target_sr


def extract_wav2vec2_embeddings(audio_path_or_waveform, feature_extractor, wav2vec_model, 
                                 video_fps=25, sr=16000, device='cpu'):
    """
    Extract wav2vec2 embeddings frame-aligned to video.
    Returns: [num_video_frames, 12, 768]
    """
    if isinstance(audio_path_or_waveform, str):
        waveform, sr = torchaudio.load(audio_path_or_waveform)
        waveform = waveform.squeeze(0)
    else:
        waveform = audio_path_or_waveform

    audio_duration = len(waveform) / sr
    num_video_frames = int(audio_duration * video_fps)

    # Process through wav2vec2
    input_values = feature_extractor(waveform.numpy(), sampling_rate=sr).input_values
    input_values = np.squeeze(input_values)
    input_values = torch.from_numpy(input_values).float().to(device)
    input_values = input_values.unsqueeze(0)

    with torch.no_grad():
        outputs = wav2vec_model(input_values, seq_len=num_video_frames, output_hidden_states=True)

    # The custom Wav2Vec2Model already performs interpolation if seq_len is provided.
    # hidden_states will be list of layers, each [B, T, D]
    hidden_states = torch.stack(outputs.hidden_states[1:], dim=1)  # B, 12, num_video_frames, 768
    hidden_states = hidden_states.squeeze(0).permute(1, 0, 2)  # num_video_frames, 12, 768

    return hidden_states.cpu()


def get_video_info(video_path):
    """Get basic video info."""
    import subprocess
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-count_packets', '-show_entries',
         'stream=nb_read_packets,r_frame_rate,width,height,duration',
         '-of', 'json', video_path],
        capture_output=True, text=True
    )
    info = json.loads(result.stdout)
    stream = info['streams'][0]
    fps_str = stream.get('r_frame_rate', '25/1')
    num, den = fps_str.split('/')
    fps = float(num) / float(den)
    duration = float(stream.get('duration', 0))
    width = int(stream.get('width', 0))
    height = int(stream.get('height', 0))
    return {
        'fps': fps,
        'duration': duration,
        'width': width,
        'height': height,
        'num_frames': int(fps * duration),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for InfiniteTalk LoRA")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing training videos")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for processed data")
    parser.add_argument("--wav2vec_model", type=str,
                        default="weights/chinese-wav2vec2-base",
                        help="wav2vec2 model name or path")
    parser.add_argument("--prompt", type=str, default="A person is talking.",
                        help="Default text prompt for all videos")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--target_h", type=int, default=832,
                        help="Target height for cropped video")
    parser.add_argument("--target_w", type=int, default=480,
                        help="Target width for cropped video")
    args = parser.parse_args()

    # Create output dirs
    os.makedirs(os.path.join(args.output_dir, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'audio_embs'), exist_ok=True)

    # Load wav2vec2 (inference style)
    print(f"Loading wav2vec2 model: {args.wav2vec_model}")
    from transformers import Wav2Vec2FeatureExtractor
    from src.audio_analysis.wav2vec2 import Wav2Vec2Model
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_model, local_files_only=True)
    wav2vec_model = Wav2Vec2Model.from_pretrained(args.wav2vec_model, local_files_only=True).to(args.device)
    wav2vec_model.feature_extractor._freeze_parameters()
    wav2vec_model.eval()
    print("wav2vec2 loaded!")

    # Process videos
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    video_files = sorted([
        f for f in os.listdir(args.video_dir)
        if Path(f).suffix.lower() in video_exts
    ])

    if not video_files:
        print(f"No video files found in {args.video_dir}")
        return

    samples = []
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(args.video_dir, video_file)
        stem = Path(video_file).stem
        out_video_name = f"{stem}-fps25.mp4"
        dst_video = os.path.join(args.output_dir, 'videos', out_video_name)

        try:
            # 1. Smart Crop Base on Face Detection & Force 25 FPS
            if not os.path.exists(dst_video):
                print(f"\n  Analyzing {video_file} for face-center crop...")
                crop_params = get_crop_params(video_path, args.target_w, args.target_h)
                
                if crop_params:
                    cw, ch, cx, cy = crop_params
                    vf_filter = f"crop={cw}:{ch}:{cx}:{cy},scale={args.target_w}:{args.target_h}"
                else:
                    vf_filter = f"scale={args.target_w}:{args.target_h}:force_original_aspect_ratio=increase,crop={args.target_w}:{args.target_h}"
                
                print(f"  Cropping to {args.target_w}x{args.target_h} & Forcing 25 FPS...")
                import subprocess
                cmd = [
                    'ffmpeg', '-y', '-i', video_path,
                    '-vf', vf_filter,
                    '-r', '25',  # Standardize all videos to 25 FPS
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                    '-c:a', 'copy',
                    dst_video
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # 2. Get info from the *processed* standardized video
            info = get_video_info(dst_video)
            print(f"\n  Processed {out_video_name}: {info['duration']:.1f}s, {info['fps']:.0f}fps, "
                  f"{info['width']}x{info['height']}")

            # 3. Extract audio from the processed video
            waveform, sr = extract_audio_from_video(dst_video)
            print(f"  Audio: {len(waveform)/sr:.1f}s, {sr}Hz")

            # 4. Extract wav2vec2 embeddings
            audio_emb = extract_wav2vec2_embeddings(
                waveform, processor, wav2vec_model,
                video_fps=25, sr=sr, device=args.device
            )
            print(f"  Embedding shape: {audio_emb.shape}")

            # Save
            emb_file = f"{stem}.pt"
            torch.save(audio_emb, os.path.join(args.output_dir, 'audio_embs', emb_file))

            samples.append({
                'video': out_video_name,
                'audio_emb': emb_file,
                'prompt': args.prompt,
                'duration': info['duration'],
                'fps': info['fps'],
                'num_frames': audio_emb.shape[0],
            })

        except Exception as e:
            print(f"  Error processing {video_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save metadata
    metadata = {
        'samples': samples,
        'wav2vec_model': args.wav2vec_model,
        'total_videos': len(samples),
        'total_duration': sum(s['duration'] for s in samples),
    }
    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Data preparation complete!")
    print(f"  Total videos: {len(samples)}")
    print(f"  Total duration: {metadata['total_duration']:.1f}s")
    print(f"  Output: {args.output_dir}")
    print(f"  Metadata: {metadata_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
