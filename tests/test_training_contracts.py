"""CPU regressions without importing the 14B pipeline or requiring PyTorch.

Execute actual helper/control-flow ASTs from train_lora.py. NumPy substitutes
only tensor concatenation/zeros/ones for numerical conditioning checks. These
tests do not establish CUDA, autograd, quantization or offload correctness.
"""
import argparse
import ast
import logging
import math
from pathlib import Path
import random
import symtable
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


SOURCE = (Path(__file__).resolve().parents[1] / 'train_lora.py').read_text(encoding='utf-8')
TREE = ast.parse(SOURCE)
TRAIN = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'train')


def run_nodes(nodes, namespace):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 'train_lora.py', 'exec'), namespace)


def helpers():
    torch = SimpleNamespace(cat=lambda xs, dim: np.concatenate(xs, axis=dim),
                            zeros_like=np.zeros_like, ones_like=np.ones_like)
    ns = {'torch': torch, 'random': random, 'argparse': argparse}
    names = {'_flow_inputs', '_reference_cache_key', '_sample_adjacent_reference', '_sample_training_timestep', '_prepared_reference_name', 'parse_args'}
    run_nodes([n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in names], ns)
    return ns


class TrainingContracts(unittest.TestCase):
    def test_memory_options_and_invalid_combinations(self):
        args = self.args('--cpu_offload_blocks', '32', '--activation_offload', '--cpu_cache_gb', '8')
        self.validate(args)
        self.assertEqual(args.blocks_to_swap, 32)
        self.assertEqual(args.attention_backend, 'sdpa')
        self.assertTrue(args.vae_cpu_offload)
        self.assertFalse(args.offload_pin_memory)
        for options in (('--activation_offload', '--no-gradient_checkpointing'),
                        ('--cpu_cache_gb', '-1'), ('--cpu_cache_gb', 'nan'),
                        ('--blocks_to_swap', '-1')):
            with self.assertRaises(ValueError):
                self.validate(self.args(*options))

    def test_81_frame_continuation_has_18_supervised_latents(self):
        context = np.full((2, 3, 1, 1), 7., dtype=np.float32)
        clean = np.full((2, 18, 1, 1), 2., dtype=np.float32)
        noise = np.full_like(clean, -2.)
        x, target, mask = helpers()['_flow_inputs'](clean, noise, .25, context)
        self.assertEqual(x.shape[1], 21)
        np.testing.assert_array_equal(x[:, :3], context)
        np.testing.assert_array_equal(mask[:, :3], 0)
        self.assertEqual(mask.sum(), 36)

    def test_prepared_reference_preserves_caption_window(self):
        fn = helpers()['_prepared_reference_name']
        sample = dict(reference_policy='adjacent', num_frames=81, start_frame=81,
                      end_frame=162, reference_frame=75, ref_image='clipref.jpg')
        self.assertEqual(fn(sample, 81, 'adjacent'), 'clipref.jpg')
        with self.assertRaises(ValueError):
            fn(sample, 61, 'adjacent')
        with self.assertRaises(ValueError):
            fn(dict(sample, reference_frame=100), 81, 'adjacent')
        self.assertIsNone(fn({'ref_image': 'legacy.jpg'}, 81, 'adjacent'))

    def setUp(self):
        self.ns = helpers()

    def args(self, *extra):
        argv = ['train', '--ckpt_dir', 'base', '--infinitetalk_dir', 'audio', '--data_dir', 'data', *extra]
        with patch.object(sys, 'argv', argv):
            return self.ns['parse_args']()

    def validate(self, args):
        # Execute the real pre-model validation, excluding logging setup.
        nodes = []
        for node in TRAIN.body[1:]:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'device' for t in node.targets):
                break
            nodes.append(node)
        run_nodes(nodes, {'args': args, 'math': math})

    def test_defaults_match_81_frame_baseline(self):
        args = self.args()
        self.validate(args)
        self.assertEqual(args.frame_num, 81)
        self.assertEqual(args.reference_mode, 'adjacent')
        self.assertFalse(hasattr(args, 'timestep_bias'))
        self.assertFalse(hasattr(args, 'bias_switch_step'))
        self.assertFalse(hasattr(args, 'override_shift'))
        self.assertEqual((args.cfg_drop_clip_prob, args.cfg_drop_ref_prob), (0, 0))

    def test_invalid_options_fail_before_loading_models(self):
        for options in [('--quant', 'int8'), ('--frame_num', '9'), ('--frame_num', '60'),
                        ('--cfg_drop_ref_prob', '1.1'), ('--ref_neighbor_frames', '0')]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.validate(self.args(*options))
        self.validate(self.args('--quant', 'fp8'))

    def test_loss_function_has_no_conditional_local_binding(self):
        scope = next(s for s in symtable.symtable(SOURCE, 'train_lora.py', 'exec').get_children()
                     if s.get_name() == 'train')
        self.assertTrue(scope.lookup('F').is_global())

    def test_dynamic_references_cannot_share_a_cache_entry(self):
        key = self.ns['_reference_cache_key']
        self.assertIsNone(key(None, 61))
        self.assertIsNone(key('', 61))
        self.assertNotEqual(key('a.jpg', 61), key('b.jpg', 61))
        self.assertNotEqual(key('a.jpg', 61), key('a.jpg', 33))

    def test_adjacent_sampling_never_reenters_window_at_boundaries(self):
        sample = self.ns['_sample_adjacent_reference']
        with patch.object(random, 'choice', side_effect=lambda xs: xs[0]):
            for start in (0, 1, 25, 39):
                for _ in range(20):
                    frame = sample(start, 61, 100, 25)
                    self.assertTrue(0 <= frame < 100)
                    self.assertFalse(start <= frame < start + 61)
                    self.assertTrue(start - 25 <= frame <= start + 60 + 25)
        self.assertEqual(sample(0, 61, 62, 25), 61)
        self.assertEqual(sample(1, 61, 62, 25), 0)
        with self.assertRaises(ValueError):
            sample(0, 61, 61, 25)

    def test_first_clip_has_clean_initial_latent_and_no_prefix_loss(self):
        clean = np.arange(32, dtype=np.float32).reshape(2, 16, 1, 1)
        noise = np.full_like(clean, -3)
        for t in (0., .4, 1.):
            x, target, mask = self.ns['_flow_inputs'](clean, noise, t)
            np.testing.assert_array_equal(x[:, :1], clean[:, :1])
            np.testing.assert_allclose(x[:, 1:], (1-t)*clean[:, 1:] + t*noise[:, 1:])
            np.testing.assert_array_equal(target[:, 1:], noise[:, 1:] - clean[:, 1:])
            np.testing.assert_array_equal(mask[:, :1], 0)
            self.assertEqual(mask.sum(), 2 * 15)

    def test_continuation_61_frames_has_3_context_and_13_target_latents(self):
        context = np.full((2, 3, 1, 1), 7., dtype=np.float32)
        clean = np.full((2, 13, 1, 1), 2., dtype=np.float32)
        noise = np.full_like(clean, -2.)
        x, target, mask = self.ns['_flow_inputs'](clean, noise, .25, context)
        self.assertEqual(x.shape, (2, 16, 1, 1))
        np.testing.assert_array_equal(x[:, :3], context)
        np.testing.assert_array_equal(x[:, 3:], 1.)
        np.testing.assert_array_equal(target[:, 3:], -4.)
        np.testing.assert_array_equal(mask[:, :3], 0.)
        self.assertEqual(mask.sum(), 26)
        # Prefix prediction errors must not affect loss; unit target errors must
        # still average to one, independent of the number of context latents.
        pred = target + 1
        pred[:, :3] = 1e6
        loss = np.mean((pred * mask - target * mask)**2) / mask.mean()
        self.assertAlmostEqual(float(loss), 1., places=6)

    def test_reference_dropout_produces_an_absent_mask_not_random_mask(self):
        branch = next(n for n in ast.walk(TRAIN) if isinstance(n, ast.If)
                      and isinstance(n.test, ast.Name) and n.test.id == 'drop_ref')
        ns = dict(self.ns, drop_ref=True, y_cond=np.ones((20, 16, 2, 2)))
        run_nodes([branch], ns)
        np.testing.assert_array_equal(ns['y_cond'], 0)

    def test_resume_lr_override_uses_resolved_audio_lr(self):
        assigns = [n for n in ast.walk(TRAIN) if isinstance(n, ast.Assign)]
        resolve = next(n for n in assigns if any(isinstance(t, ast.Name) and t.id == 'audio_lr' for t in n.targets))
        override_value = next(n for n in assigns if any(isinstance(t, ast.Name) and t.id == '_audio_lr_cli' for t in n.targets))
        override = next(n for n in ast.walk(TRAIN) if isinstance(n, ast.If)
                        and ast.unparse(n.test) == "getattr(args, 'override_lr', False)")
        for explicit in (None, 2e-5):
            args = self.args('--override_lr', '--lr', '5e-5')
            args.audio_lr = explicit
            optimizer = SimpleNamespace(param_groups=[{'lr': 1e-6}, {'lr': 1e-6}])
            ns = {'args': args, 'optimizer': optimizer, 'current_step': 250, 'logging': logging,
                  'torch': SimpleNamespace(optim=SimpleNamespace(lr_scheduler=SimpleNamespace(
                      CosineAnnealingLR=lambda opt, **kw: kw)))}
            run_nodes([resolve, override_value, override], ns)
            self.assertEqual(optimizer.param_groups[0]['lr'], 5e-5)
            self.assertEqual(optimizer.param_groups[1]['lr'], explicit if explicit is not None else 5e-5)
            self.assertEqual(ns['scheduler']['T_max'], args.max_steps - 250)

    def test_logit_normal_sampling_has_no_scale_offset_or_shift(self):
        # Known standard-normal quantiles must map to the corresponding logistic
        # quantiles, catching the old 1.6 scale and shifted-uniform alternatives.
        calls = []
        for z, expected in ((-1., .268941421), (0., .5), (1., .731058579)):
            def randn(size, *, device, dtype):
                calls.append((size, device, dtype))
                return np.array([z], dtype=dtype)
            self.ns['torch'] = SimpleNamespace(
                randn=randn, sigmoid=lambda x: 1 / (1 + np.exp(-x)), float32=np.float32)
            sigma = self.ns['_sample_training_timestep']('cpu')
            self.assertEqual(sigma.shape, (1,))
            self.assertAlmostEqual(float(sigma[0]), expected, places=6)
        self.assertEqual(calls, [(1, 'cpu', np.float32)] * 3)


if __name__ == '__main__':
    unittest.main()
