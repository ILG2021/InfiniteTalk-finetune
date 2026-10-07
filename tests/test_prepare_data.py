import shutil
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import prepare_data as prep


class PreparationContracts(unittest.TestCase):
    def test_windows_and_adjacent_references(self):
        plans = prep.plan_clips(250)
        self.assertEqual([(p['start_frame'], p['end_frame']) for p in plans],
                         [(0, 81), (81, 162), (162, 243)])
        self.assertEqual(plans, prep.plan_clips(250))
        for p in plans:
            s, e, r = p['start_frame'], p['end_frame'], p['reference_frame']
            self.assertTrue(max(0, s-25) <= r < s or e <= r < min(250, e+25))
        self.assertEqual(prep.plan_clips(81), [])
        self.assertEqual(prep.plan_clips(80), [])
        self.assertEqual(prep.plan_clips(82)[0]['reference_frame'], 81)

    def test_crop_unions_motion_and_pads_source_pixels(self):
        crop = prep.crop_from_boxes([[300, 100, 500, 400], [400, 120, 700, 450]], 1200, 800)
        self.assertEqual([crop[k] for k in ('x', 'y', 'width', 'height')], [100, 50, 800, 450])
        clipped = prep.crop_from_boxes([[0, 0, 1200, 800]], 1200, 800)
        self.assertEqual(clipped['width'], 1200)
        self.assertEqual(clipped['height'], 800)
        with self.assertRaises(ValueError):
            prep.crop_from_boxes([], 1200, 800)

    def test_detector_visits_every_frame(self):
        calls = []
        cap = SimpleNamespace(isOpened=lambda: True, set=lambda *a: None,
                              read=lambda: (True, len(calls)), release=lambda: None)
        def predict(frame, **kwargs):
            calls.append(frame)
            tensor = SimpleNamespace()
            tensor.detach = tensor.cpu = lambda: tensor
            tensor.tolist = lambda: [[frame, 0, frame+10, 20]]
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=tensor))]
        with patch.dict('sys.modules', cv2=SimpleNamespace(VideoCapture=lambda _: cap, CAP_PROP_POS_FRAMES=1)):
            boxes, detected = prep.detect_clip_boxes('fake', 0, 81, SimpleNamespace(predict=predict), 'cpu', .25)
        self.assertEqual(calls, list(range(81)))
        self.assertEqual((len(boxes), detected), (81, 81))

    def test_discovery_rejects_ambiguous_stems(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'scene.v1.mp4').touch()
            self.assertEqual(prep.discover_videos(root), [root/'scene.v1.mp4'])
            (root/'scene.v1.mov').touch()
            with self.assertRaises(ValueError):
                prep.discover_videos(root)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_real_ffmpeg_frame_audio_and_reference_alignment(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, normalized = root/'source.mp4', root/'normalized.mkv'
            prep.run_command(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                              'testsrc2=size=96x64:rate=30:duration=7', '-f', 'lavfi', '-i',
                              'sine=frequency=440:sample_rate=48000:duration=7',
                              '-c:v', 'libx264', '-c:a', 'aac', source])
            prep.normalize_source(source, normalized)
            info = prep.probe_video(normalized)
            self.assertEqual((info['fps'], info['num_frames']), (25, 175))
            crop = prep.crop_from_boxes([[20, 10, 70, 50]], 96, 64, 4, 2)
            for index, plan in enumerate(prep.plan_clips(info['num_frames'])):
                video, ref, audio = root/f'{index}.mp4', root/f'{index}ref.jpg', root/f'{index}.wav'
                prep.export_clip(normalized, video, ref, audio, plan, crop)
                self.assertEqual(prep.probe_video(video)['num_frames'], 81)
                with wave.open(str(audio)) as pcm:
                    self.assertEqual((pcm.getframerate(), pcm.getnframes()), (16000, 51840))
                with Image.open(ref) as image:
                    self.assertEqual(image.size, (crop['output_width'], crop['output_height']))


if __name__ == '__main__':
    unittest.main()
