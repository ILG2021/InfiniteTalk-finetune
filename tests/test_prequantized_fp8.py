import importlib.util
import ast
import logging
import math
from typing import cast, Optional, List
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    from safetensors.torch import save_file
except ImportError:
    torch = None

if torch is not None:
    spec = importlib.util.spec_from_file_location('fp8_loader_test',
        Path(__file__).resolve().parents[1]/'wan/utils/training_memory.py')
    memory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(memory)


@unittest.skipIf(torch is None, 'PyTorch and safetensors required')
class PrequantizedTests(unittest.TestCase):
    def test_downloaded_fp8_accepts_real_lora_and_optimizer_step(self):
        source = (Path(__file__).resolve().parents[1]/'train_lora.py').read_text(encoding='utf-8')
        nodes = [n for n in ast.parse(source).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                 and n.name in ('LoRALinear', 'apply_lora_to_model')]
        ns = dict(torch=torch, nn=torch.nn, math=math, logging=logging, cast=cast,
                  FrozenLinear=memory.FrozenLinear, Optional=Optional, List=List)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'train_lora.py', 'exec'), ns)
        with tempfile.TemporaryDirectory() as tmp:
            path, _, _ = self.fixture(tmp)
            with torch.device('meta'):
                model = self.make_model()
            memory.load_prequantized_fp8(model, path, path.with_suffix('.json'))
            model.requires_grad_(False)
            frozen = model.blocks[0][0]
            before = frozen.weight.float().clone()
            ns['apply_lora_to_model'](model, rank=2, alpha=2, target_modules=['blocks.0.0'])
            memory.prepare_model_memory(model, 'cpu', fp8=False, blocks_to_offload=1)
            params = [p for p in model.parameters() if p.requires_grad]
            self.assertEqual(len(params), 2)
            self.assertTrue(all(p.dtype == torch.float32 for p in params))
            optimizer = torch.optim.AdamW(params, lr=.01)
            initial = [p.detach().clone() for p in params]
            with torch.autocast('cpu', dtype=torch.bfloat16):
                loss = model.head(model.blocks[0](torch.randn(2, 4))).float().square().mean()
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params))
            optimizer.step()
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(params, initial)))
            torch.testing.assert_close(frozen.weight.float(), before, rtol=0, atol=0)

    def make_model(self):
        model = torch.nn.Module()
        model.blocks = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(4, 3))])
        model.head = torch.nn.Linear(3, 2)
        return model

    def fixture(self, folder):
        state = self.make_model().state_dict()
        state.pop('blocks.0.0.weight')
        data = torch.tensor([[1., 2., 3., 4.], [-1., -2., -3., -4.], [0., 0., 0., 0.]]).to(torch.float8_e4m3fn)
        scale = torch.tensor([[.01], [.0314159], [.2]])
        state.update({'blocks.0.0.weight._data': data, 'blocks.0.0.weight._scale': scale,
                      'blocks.0.0.input_scale': torch.tensor(1.), 'blocks.0.0.output_scale': torch.tensor(1.)})
        path = Path(folder)/'official.safetensors'
        save_file(state, str(path))
        mapping = {'blocks.0.0': {'weights': 'qfloat8_e4m3fn', 'activations': 'none'}}
        path.with_suffix('.json').write_text(json.dumps(mapping))
        return path, state, mapping

    def test_meta_load_preserves_source_bytes_scales_and_unquantized_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, state, _ = self.fixture(tmp)
            with torch.device('meta'):
                model = self.make_model()
            self.assertEqual(memory.load_prequantized_fp8(model, path, path.with_suffix('.json')), 1)
            layer = model.blocks[0][0]
            torch.testing.assert_close(layer.weight.float(), state['blocks.0.0.weight._data'].float())
            torch.testing.assert_close(layer.scale_weight, state['blocks.0.0.weight._scale'], rtol=0, atol=0)
            model.requires_grad_(False)
            memory.prepare_model_memory(model, 'cpu', fp8=False, blocks_to_offload=1)
            self.assertIsInstance(model.head, torch.nn.Linear)
            self.assertEqual(model.head.weight.dtype, torch.float32)
            model.bfloat16()
            self.assertEqual(layer.scale_weight.dtype, torch.float32)
            x = torch.randn(2, 4, dtype=torch.bfloat16, requires_grad=True)
            actual = layer(x)
            expected = torch.nn.functional.linear(x,
                state['blocks.0.0.weight._data'].bfloat16()*state['blocks.0.0.weight._scale'].bfloat16(),
                state['blocks.0.0.bias'].bfloat16())
            torch.testing.assert_close(actual, expected)
            ga, = torch.autograd.grad(actual.float().sum(), x)
            ge, = torch.autograd.grad(expected.float().sum(), x)
            torch.testing.assert_close(ga, ge)

    def test_rejects_activation_quantization_and_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, state, mapping = self.fixture(tmp)
            mapping['blocks.0.0']['activations'] = 'qint8'
            path.with_suffix('.json').write_text(json.dumps(mapping))
            with self.assertRaises(ValueError):
                memory.load_prequantized_fp8(self.make_model(), path, path.with_suffix('.json'))
            mapping['blocks.0.0']['activations'] = 'none'
            path.with_suffix('.json').write_text(json.dumps(mapping))
            state.pop('head.weight')
            save_file(state, str(path))
            with self.assertRaises(RuntimeError):
                memory.load_prequantized_fp8(self.make_model(), path, path.with_suffix('.json'))


if __name__ == '__main__':
    unittest.main()
