import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import download_weights as weights


class DownloadWeightsTests(unittest.TestCase):
    def test_training_plan_matches_loader_and_excludes_extra_variants(self):
        models = weights.selected_models('training')
        self.assertEqual([m['directory'] for m in models],
                         ['Wan2.1-I2V-14B-480P', 'InfiniteTalk', 'chinese-wav2vec2-base'])
        self.assertEqual(models[1]['files'], ('quant_models/infinitetalk_single_fp8.safetensors',
                                             'quant_models/infinitetalk_single_fp8.json'))
        self.assertEqual(sum(name.endswith('.safetensors') for name in models[0]['files']), 0)
        for name in ('Wan2.1_VAE.pth', 'models_t5_umt5-xxl-enc-bf16.pth',
                     'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth',
                     'google/umt5-xxl/tokenizer.json', 'xlm-roberta-large/tokenizer.json'):
            self.assertIn(name, models[0]['files'])
        for model in models:
            self.assertEqual(len(model['revision']), 40)
            self.assertFalse(any(name.endswith('.bin') or 'comfyui/' in name
                                 for name in model['files']))
        self.assertEqual(sum(name.endswith('.safetensors') for name in
                             weights.selected_models('training-original')[0]['files']), 7)

    def test_offline_plan_does_not_create_directories(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)/'not-created'
            self.assertEqual(weights.main(['--weights_dir', str(root)]), 0)
            self.assertFalse(root.exists())
            self.assertEqual(weights.main(['--weights_dir', str(root), '--check']), 1)

    def test_download_uses_pinned_revisions_and_preserves_unrelated_files(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            unrelated = root/'keep.txt'
            unrelated.write_text('keep')
            calls = []
            def fake_download(**kwargs):
                calls.append(kwargs)
                path = Path(kwargs['local_dir'])/kwargs['filename']
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture')
            models = weights.selected_models('audio')
            weights.download(root, models, fake_download)
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(c['revision'] == models[0]['revision'] for c in calls))
            self.assertEqual(weights.missing_files(root, models), [])
            receipt = json.loads((root/models[0]['directory']/'.download-revision.json').read_text())
            self.assertEqual(receipt['revision'], models[0]['revision'])
            self.assertEqual(unrelated.read_text(), 'keep')
            empty = root/models[0]['directory']/'model.safetensors'
            empty.write_bytes(b'')
            self.assertEqual(weights.missing_files(root, models), [empty])

    def test_failed_download_does_not_publish_completion_receipt(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            with self.assertRaises(RuntimeError):
                weights.download(root, weights.selected_models('audio'), lambda **kwargs: None)
            self.assertFalse((root/'chinese-wav2vec2-base'/'.download-revision.json').exists())


if __name__ == '__main__':
    unittest.main()
