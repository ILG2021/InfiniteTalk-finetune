import ast
import json
import logging
import os
from pathlib import Path
import random
import tempfile
import sys
from types import SimpleNamespace
import unittest

try:
    import torch
    import numpy as np
    from PIL import Image
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch, NumPy and Pillow required')
class DatasetTests(unittest.TestCase):
    def test_prepared_clip_audio_window_and_rejections(self):
        source = (Path(__file__).resolve().parents[1]/'train_lora.py').read_text(encoding='utf-8')
        names = {'InfiniteTalkDataset', '_prepared_reference_name', '_sample_adjacent_reference'}
        nodes = [n for n in ast.parse(source).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
        ns = dict(Dataset=torch.utils.data.Dataset, torch=torch, np=np, random=random,
                  os=os, json=json, logging=logging)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'train_lora.py', 'exec'), ns)
        fps = [25.]
        class Reader:
            def __init__(self, *a, **kw): pass
            def __len__(self): return 81
            def get_avg_fps(self): return fps[0]
            def get_batch(self, indices):
                frames = np.stack([np.full((16, 32, 3), i, dtype=np.uint8) for i in indices])
                return SimpleNamespace(asnumpy=lambda: frames)
        previous = sys.modules.get('decord')
        sys.modules['decord'] = SimpleNamespace(VideoReader=Reader, cpu=lambda _: None)
        def restore_decoder():
            if previous is None:
                sys.modules.pop('decord', None)
            else:
                sys.modules['decord'] = previous
        self.addCleanup(restore_decoder)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ('videos', 'audio_embs', 'ref_images'):
                (root/folder).mkdir()
            sample = dict(video='clip.mp4', audio_emb='clip.pt', ref_image='clipref.jpg', prompt='A person speaks.',
                          reference_policy='adjacent', num_frames=81, start_frame=81, end_frame=162, reference_frame=80)
            (root/'metadata.json').write_text(json.dumps({'samples': [sample]}))
            emb = torch.arange(81.)[:, None, None].expand(81, 12, 768).clone()
            torch.save(emb, root/'audio_embs/clip.pt')
            Image.new('RGB', (32, 16)).save(root/'ref_images/clipref.jpg')
            dataset = ns['InfiniteTalkDataset'](tmp)
            batch = dataset[0]
            self.assertEqual(batch['video_full'].shape, (3, 81, 16, 32))
            torch.testing.assert_close(batch['audio_emb_full'][0, :, 0, 0], torch.tensor([0., 0., 0., 1., 2.]))
            torch.testing.assert_close(batch['audio_emb_full'][-1, :, 0, 0], torch.tensor([78., 79., 80., 80., 80.]))
            fps[0] = 30.
            with self.assertRaisesRegex(ValueError, '25 fps'): dataset[0]
            fps[0] = 25.
            Image.new('RGB', (16, 16)).save(root/'ref_images/clipref.jpg')
            with self.assertRaisesRegex(ValueError, 'crop/resolution'): dataset[0]
            dataset.samples[0]['prompt'] = ''
            with self.assertRaisesRegex(ValueError, 'caption'): dataset[0]
            dataset.samples[0]['prompt'] = 'A person speaks.'
            emb[0, 0, 0] = float('nan')
            torch.save(emb, root/'audio_embs/clip.pt')
            with self.assertRaisesRegex(ValueError, 'audio embedding'): dataset[0]
            (root/'metadata.json').write_text(json.dumps({'samples': []}))
            with self.assertRaisesRegex(ValueError, 'no samples'): ns['InfiniteTalkDataset'](tmp)
