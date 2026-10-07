"""Execute actual Wan classes at small dimensions, without importing 14B deps."""
import ast
import importlib.util
import logging
import math
import tempfile
import json
import unittest
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, cast
from types import SimpleNamespace

try:
    import torch
    import numpy as np
    from einops import rearrange, repeat
    from safetensors.torch import save_file
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


def definitions(path, namespace, names=None):
    tree = ast.parse((ROOT/path).read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and (names is None or n.name in names)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


@unittest.skipIf(torch is None, 'PyTorch, safetensors and einops required')
class ModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('integration_memory', ROOT/'wan/utils/training_memory.py')
        cls.memory = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.memory)
        ns = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, amp=torch.cuda.amp,
                  math=math, np=np, rearrange=rearrange, repeat=repeat, logging=logging,
                  lru_cache=lru_cache, ModelMixin=torch.nn.Module, ConfigMixin=type('ConfigMixin', (), {}),
                  register_to_config=lambda f: f, USE_SAGEATTN=False,
                  TRAINING_ATTENTION_BACKEND='sdpa', xformers=None, Optional=Optional, List=List, cast=cast,
                  FrozenLinear=cls.memory.FrozenLinear, uses_fp32_compute=cls.memory.uses_fp32_compute,
                  sdpa_attention=cls.memory.sdpa_attention, checkpoint_block=cls.memory.checkpoint_block)
        definitions('wan/utils/multitalk_utils.py', ns, {'normalize_and_scale', 'rotate_half',
                    'calculate_x_ref_attn_map_chunk', 'get_attn_map_with_target', 'RotaryPositionalEmbedding1D'})
        definitions('wan/modules/attention.py', ns)
        definitions('wan/modules/multitalk_model.py', ns)
        definitions('wan/multitalk.py', ns, {'to_param_dtype_fp32only'})
        definitions('wan/modules/clip.py', ns, {'comfy_clip_pixels'})
        definitions('train_lora.py', ns, {'LoRALinear', 'apply_lora_to_model', 'extract_lora_state_dict',
                                         '_validate_resume_configuration'})
        cls.ns = ns

    def test_clip_center_preserves_geometry_and_comfy_pixel_rounding(self):
        pixels = torch.full((1, 3, 8, 16), -1.)
        pixels[:, :, :, 4:12] = 1.
        center = self.ns['comfy_clip_pixels'](pixels, size=8, crop='center')
        resized = self.ns['comfy_clip_pixels'](pixels, size=8, crop='disabled')
        torch.testing.assert_close(center, torch.ones_like(center))
        self.assertLess(resized.mean().item(), .6)
        self.assertEqual(center.shape, (1, 3, 8, 8))
        torch.testing.assert_close(resized*255, (resized*255).round())

    def test_resume_rejects_changed_adapter_scaling_or_parameter_set(self):
        saved = dict(lora_rank=2, lora_alpha=7., train_audio=True)
        validate = self.ns['_validate_resume_configuration']
        validate(saved, SimpleNamespace(**saved))
        for key, value in [('lora_rank', 4), ('lora_alpha', 2.), ('train_audio', False)]:
            with self.assertRaisesRegex(ValueError, key):
                validate(saved, SimpleNamespace(**dict(saved, **{key: value})))

    def model(self):
        model = self.ns['WanModel'](in_dim=12, out_dim=4, dim=16, ffn_dim=32, freq_dim=8,
            text_dim=8, text_len=4, num_heads=2, num_layers=1, intermediate_dim=8, output_dim=8,
            context_tokens=2, weight_init=False)
        model.disable_teacache()
        return model

    def test_full_prequantized_forward_backward_and_checkpoint_match(self, amp_enabled=True):
        torch.manual_seed(4)
        source = self.model()
        state = source.state_dict()
        mapping = {}
        for name, layer in source.named_modules():
            if isinstance(layer, torch.nn.Linear) and not self.memory.uses_fp32_compute(name):
                quant = self.memory.FrozenLinear(layer.requires_grad_(False), fp8=True)
                state.pop(name+'.weight')
                state[name+'.weight._data'] = quant.weight.detach()
                state[name+'.weight._scale'] = quant.scale_weight
                mapping[name] = {'weights': 'qfloat8_e4m3fn', 'activations': 'none'}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'tiny.safetensors'
            save_file(state, str(path))
            path.with_suffix('.json').write_text(json.dumps(mapping))
            with torch.device('meta'):
                model = self.model()
            self.memory.load_prequantized_fp8(model, path, path.with_suffix('.json'))
        model.init_freqs()
        self.ns['to_param_dtype_fp32only'](model, torch.bfloat16)
        model.requires_grad_(False)
        self.ns['apply_lora_to_model'](model, rank=2, alpha=1,
            target_modules=['self_attn.q', 'audio_cross_attn.q_linear', 'audio_proj.proj3'])
        self.memory.prepare_model_memory(model, 'cpu', blocks_to_offload=1)
        params = [p for p in model.parameters() if p.requires_grad]
        inputs = dict(x=[torch.randn(4, 4, 4, 4)], y=[torch.randn(8, 4, 4, 4)],
            t=torch.tensor([450.]), context=[torch.randn(4, 8, dtype=torch.bfloat16)], seq_len=16,
            clip_fea=torch.randn(1, 257, 1280, dtype=torch.bfloat16),
            audio=torch.randn(1, 13, 5, 12, 768, dtype=torch.bfloat16),
            ref_target_masks=torch.ones(3, 4, 4))
        outputs, grads = [], []
        for checkpointing in (False, True):
            model.zero_grad(set_to_none=True)
            model.gradient_checkpointing = checkpointing
            model.activation_offload = checkpointing
            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp_enabled):
                out = model(**dict(inputs, x=[inputs['x'][0].clone()]))[0]
            self.assertEqual(tuple(out.shape), (4, 4, 4, 4))
            self.assertTrue(torch.isfinite(out).all())
            out.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params))
            self.assertGreater(sum(p.grad.abs().sum() for p in params).item(), 0)
            outputs.append(out.detach())
            grads.append([p.grad.clone() for p in params])
        torch.testing.assert_close(*outputs)
        for a, b in zip(*grads):
            torch.testing.assert_close(a, b)

    def test_full_prequantized_without_amp(self):
        self.test_full_prequantized_forward_backward_and_checkpoint_match(amp_enabled=False)

    def test_comfy_export_maps_every_audio_layer_and_preserves_scaling(self):
        model = self.model().requires_grad_(False)
        targets = ['self_attn.q', 'audio_cross_attn.q_linear', 'audio_proj.proj1', 'audio_proj.proj1_vf',
                   'audio_proj.proj2', 'audio_proj.proj3']
        self.ns['apply_lora_to_model'](model, rank=2, alpha=7, target_modules=targets)
        for module in model.modules():
            if isinstance(module, self.ns['LoRALinear']):
                torch.nn.init.normal_(module.lora_up.weight)
        official = self.ns['extract_lora_state_dict'](model)
        comfy = self.ns['extract_lora_state_dict'](model, comfyui=True)
        self.assertEqual(len(official), len(comfy))
        self.assertFalse(any(k.startswith('diffusion_model.audio_proj.') for k in comfy))
        for name, module in model.named_modules():
            if isinstance(module, self.ns['LoRALinear']):
                target = name.replace('audio_proj.', 'multitalk_audio_proj.', 1) if name.startswith('audio_proj.') else name
                prefix = 'diffusion_model.'+target
                delta = comfy[prefix+'.lora_up.weight'] @ comfy[prefix+'.lora_down.weight']
                expected = (module.lora_up.weight @ module.lora_down.weight) * module.scaling
                torch.testing.assert_close(delta, expected)
