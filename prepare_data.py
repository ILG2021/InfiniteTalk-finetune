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

def get_crop_params(video_path, target_h):
    """Detect person and calculate crop dimensions to ensure 50px padding on all sides.
    
    Args:
        target_h: Target output height (e.g., 1024). Width is calculated automatically
            to maintain original video aspect ratio.
    """
    import cv2
    import os

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    target_h = target_h // 16 * 16  # must be multiple of 16 (VAE stride 8 × patch 2)

    print(f"  Source: {w}x{h}, target height: {target_h}")

    # 寻找一个包含人物的“锚点帧”（避免第一帧是全黑或渐显）
    # 一旦找到，就会计算出全局唯一的裁切框，应用到整个视频
    anchor_frames = [0, total_frames // 10, total_frames // 2]
    
    # ==========================================
    # YOLO Person Detection (Forced Dependency)
    # ==========================================
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  [Info] ultralytics not found. Force installing ultralytics...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        from ultralytics import YOLO
        
    # Load lightweight YOLOv8n model
    model = YOLO('yolov8n.pt')
    
    person_xmin, person_ymin, person_xmax, person_ymax = 0, 0, w, h
    person_found = False
    
    for check_idx in anchor_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, check_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Predict person class (class 0)
        results = model(frame, classes=[0], verbose=False)
        
        if results and len(results[0].boxes) > 0:
            largest_area = 0
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    person_xmin, person_ymin, person_xmax, person_ymax = x1, y1, x2, y2
            
            if largest_area > 0:
                print(f"  YOLO anchor frame found at idx {check_idx}: "
                      f"bbox ({int(person_xmin)},{int(person_ymin)}) → ({int(person_xmax)},{int(person_ymax)}) "
                      f"size {int(person_xmax-person_xmin)}x{int(person_ymax-person_ymin)}")
                person_found = True
                
                person_w_box = int(person_xmax - person_xmin)
                person_h_box = int(person_ymax - person_ymin)

                # Step 1: Add 50px padding on all sides in SOURCE space
                pad_src = 50
                crop_x = int(person_xmin) - pad_src
                crop_y = int(person_ymin) - pad_src
                crop_w = person_w_box + 2 * pad_src
                crop_h = person_h_box + 2 * pad_src

                # Step 2: Clamp to video boundaries
                crop_x = max(0, crop_x)
                crop_y = max(0, crop_y)
                crop_w = min(crop_w, w - crop_x)
                crop_h = min(crop_h, h - crop_y)

                # Ensure even dimensions for ffmpeg codec
                crop_w = crop_w // 2 * 2
                crop_h = crop_h // 2 * 2

                # Step 3: Scale so output height = target_h, maintain aspect ratio
                scale = target_h / crop_h
                out_w = int(crop_w * scale) // 16 * 16  # must be multiple of 16 (VAE stride 8 × patch 2)

                print(f"  Person bbox: {person_w_box}x{person_h_box} + {pad_src}px padding each side")
                print(f"  Crop region (source): {crop_w}x{crop_h} @ ({crop_x},{crop_y})")
                print(f"  Output: {out_w}x{target_h} (scale={scale:.3f})")

                cap.release()
                return crop_w, crop_h, crop_x, crop_y, out_w, target_h
                
    cap.release()
    
    raise RuntimeError(
        f"YOLO found no person in any of the anchor frames {anchor_frames} of:\n  {video_path}\n"
        f"Please check that the video contains a clearly visible person, "
        f"or delete this video from the input directory."
    )


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


def select_best_ref_frame(video_path, target_w, target_h, num_samples=20):
    """Select the best reference frame from a video.
    
    Scoring criteria (higher is better):
    - Face size: larger face = more frontal / closer
    - Mouth closed: neutral expression preferred for reference
    - Sharpness: Laplacian variance (less blur = better)
    
    Returns: best frame as numpy array (BGR, target_w x target_h), or None
    """
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return None

    # Sample frames evenly across the video
    sample_indices = np.linspace(0, total_frames - 1, min(num_samples, total_frames), dtype=int)

    # Load face detector
    model_path = os.path.join('weights', 'face_detector_full.tflite')
    
    # Load face landmarker for mouth detection
    face_landmarker_path = os.path.join('weights', 'face_landmarker.task')
    need_download_landmarker = False
    if not os.path.exists(face_landmarker_path):
        os.makedirs('weights', exist_ok=True)
        try:
            import urllib.request
            print(f"  Downloading face landmarker model for mouth detection...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, face_landmarker_path)
        except Exception as e:
            print(f"  Face landmarker download failed: {e}. Will use face size only for scoring.")
            need_download_landmarker = True

    best_score = -1
    best_frame = None

    try:
        # Setup face detector
        if os.path.exists(model_path):
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceDetectorOptions(base_options=base_options)
            
            # Setup face landmarker for mouth detection
            landmarker = None
            if os.path.exists(face_landmarker_path):
                try:
                    landmarker_base = python.BaseOptions(model_asset_path=face_landmarker_path)
                    landmarker_options = vision.FaceLandmarkerOptions(
                        base_options=landmarker_base,
                        output_face_blendshapes=True,
                        num_faces=1,
                    )
                    landmarker = vision.FaceLandmarker.create_from_options(landmarker_options)
                except Exception:
                    landmarker = None

            with vision.FaceDetector.create_from_options(options) as detector:
                for idx in sample_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    # Resize to target size for consistent scoring
                    frame_resized = cv2.resize(frame, (target_w, target_h))

                    # Convert to MediaPipe Image
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                        data=cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
                    detection_result = detector.detect(mp_image)

                    if not detection_result.detections:
                        continue

                    # Score 1: Face size (larger = better, more frontal)
                    bbox = detection_result.detections[0].bounding_box
                    face_area = bbox.width * bbox.height
                    frame_area = target_w * target_h
                    face_ratio = face_area / frame_area  # 0~1
                    face_score = min(face_ratio * 5, 1.0)  # Scale: 20% face area = max score

                    # Score 2: Mouth closed (check blendshapes if landmarker available)
                    mouth_score = 0.5  # Default neutral
                    if landmarker is not None:
                        try:
                            landmark_result = landmarker.detect(mp_image)
                            if landmark_result.face_blendshapes:
                                blends = {bs.category_name: bs.score for bs in landmark_result.face_blendshapes[0]}
                                jaw_open = blends.get('jawOpen', 0)
                                mouth_open = blends.get('mouthOpen', 0)
                                # Lower jaw/mouth open = better (closed mouth preferred)
                                mouth_score = 1.0 - min((jaw_open + mouth_open) * 3, 1.0)
                        except Exception:
                            pass

                    # Score 3: Sharpness (Laplacian variance)
                    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                    # Normalize: typical sharpness 50-500
                    sharpness_score = min(sharpness / 300.0, 1.0)

                    # Combined score
                    total_score = face_score * 0.4 + mouth_score * 0.35 + sharpness_score * 0.25

                    if total_score > best_score:
                        best_score = total_score
                        best_frame = frame_resized

            if landmarker is not None:
                landmarker.close()

        else:
            # No face detector: fallback - pick sharpest frame from center portion
            center_indices = sample_indices[len(sample_indices)//4 : 3*len(sample_indices)//4]
            for idx in center_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret:
                    continue
                frame_resized = cv2.resize(frame, (target_w, target_h))
                gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                if sharpness > best_score:
                    best_score = sharpness
                    best_frame = frame_resized

    except Exception as e:
        print(f"  Reference frame selection failed: {e}")

    cap.release()
    return best_frame


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
    parser.add_argument("--prompt", type=str, default="A news anchor is broadcasting.",
                        help="Default text prompt for all videos")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--target_h", type=int, default=1024,
                        help="Target output height (e.g., 1024). "
                             "Width is calculated automatically to maintain aspect ratio. "
                             "Person will have 50px padding on all sides in output space.")
    parser.add_argument("--force_recrop", action="store_true",
                        help="Force re-cropping even if processed video already exists.")
    args = parser.parse_args()

    # Create output dirs
    os.makedirs(os.path.join(args.output_dir, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'audio_embs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'ref_images'), exist_ok=True)

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

    # ---- Detect crop params ONCE from the first video's middle frame ----
    # All videos share the same crop box and output resolution so that
    # every processed video has identical spatial dimensions.
    crop_params_path = os.path.join(args.output_dir, 'crop_params.json')
    if os.path.exists(crop_params_path) and not args.force_recrop:
        with open(crop_params_path, 'r', encoding='utf-8') as _f:
            _cp = json.load(_f)
        global_cw, global_ch, global_cx, global_cy = _cp['cw'], _cp['ch'], _cp['cx'], _cp['cy']
        global_out_w, global_out_h = _cp['out_w'], _cp['out_h']
        print(f"Loaded existing crop params from {crop_params_path}: "
              f"crop {global_cw}x{global_ch}@({global_cx},{global_cy}) → {global_out_w}x{global_out_h}")
    else:
        first_video = os.path.join(args.video_dir, video_files[0])
        print(f"Detecting crop params from first video (middle frame): {video_files[0]}")
        global_cw, global_ch, global_cx, global_cy, global_out_w, global_out_h = \
            get_crop_params(first_video, args.target_h)
        _cp = {'cw': global_cw, 'ch': global_ch, 'cx': global_cx, 'cy': global_cy,
               'out_w': global_out_w, 'out_h': global_out_h}
        with open(crop_params_path, 'w', encoding='utf-8') as _f:
            json.dump(_cp, _f, indent=2)
        print(f"Saved crop params → {crop_params_path}")

    global_vf_filter = f"crop={global_cw}:{global_ch}:{global_cx}:{global_cy},scale={global_out_w}:{global_out_h}"
    print(f"Global ffmpeg filter: {global_vf_filter}\n")

    samples = []
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(args.video_dir, video_file)
        stem = Path(video_file).stem
        out_video_name = f"{stem}-fps25.mp4"
        dst_video = os.path.join(args.output_dir, 'videos', out_video_name)

        try:
            # 1. Crop & scale using global params detected from first video, Force 25 FPS
            if not os.path.exists(dst_video) or args.force_recrop:
                if args.force_recrop and os.path.exists(dst_video):
                    print(f"\n  [force_recrop] Removing existing video: {dst_video}")
                    os.remove(dst_video)

                print(f"\n  Processing {video_file}: crop {global_cw}x{global_ch}@({global_cx},{global_cy}) → {global_out_w}x{global_out_h} @ 25 FPS...")
                import subprocess
                cmd = [
                    'ffmpeg', '-y', '-i', video_path,
                    '-vf', global_vf_filter,
                    '-r', '25',  # Standardize all videos to 25 FPS
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                    '-c:a', 'aac', '-b:a', '192k',
                    dst_video
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"FFmpeg failed (code {result.returncode}):\n{result.stderr[-2000:]}"
                    )

            # 2. Get info from the *processed* standardized video
            info = get_video_info(dst_video)
            print(f"\n  Processed {out_video_name}: {info['duration']:.1f}s, {info['fps']:.0f}fps, "
                  f"{info['width']}x{info['height']} (height {args.target_h})")

            # 3 & 4. Extract audio + wav2vec2 embeddings (skip if already exists)
            emb_file = f"{stem}.pt"
            emb_path = os.path.join(args.output_dir, 'audio_embs', emb_file)
            if not os.path.exists(emb_path):
                # 3. Extract audio from the processed video
                waveform, sr = extract_audio_from_video(dst_video)
                print(f"  Audio: {len(waveform)/sr:.1f}s, {sr}Hz")

                # 4. Extract wav2vec2 embeddings
                audio_emb = extract_wav2vec2_embeddings(
                    waveform, processor, wav2vec_model,
                    video_fps=25, sr=sr, device=args.device
                )
                print(f"  Embedding shape: {audio_emb.shape}")
                torch.save(audio_emb, emb_path)
                print(f"  Saved embedding: {emb_file}")
            else:
                # Load existing embedding to get num_frames
                audio_emb = torch.load(emb_path, map_location='cpu', weights_only=True)
                print(f"  Embedding already exists, loaded: {emb_file} shape={audio_emb.shape}")

            # 5. Auto-select best reference frame (front-facing, mouth closed, sharp)
            # Use actual dimensions from processed video (may vary per video based on aspect ratio)
            ref_image_name = f"{stem}_ref.jpg"
            ref_image_path = os.path.join(args.output_dir, 'ref_images', ref_image_name)
            if not os.path.exists(ref_image_path):
                best_frame = select_best_ref_frame(dst_video, info['width'], info['height'])
                if best_frame is not None:
                    import cv2 as cv2_ref
                    cv2_ref.imwrite(ref_image_path, best_frame)
                    print(f"  Saved reference frame: {ref_image_name}")
                else:
                    ref_image_name = None
                    print(f"  Warning: Could not select reference frame for {video_file}")
            else:
                print(f"  Reference frame already exists: {ref_image_name}")

            sample_dict = {
                'video': out_video_name,
                'audio_emb': emb_file,
                'prompt': args.prompt,
                'duration': info['duration'],
                'fps': info['fps'],
                'num_frames': audio_emb.shape[0],
            }
            if ref_image_name:
                sample_dict['ref_image'] = ref_image_name
            samples.append(sample_dict)

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
