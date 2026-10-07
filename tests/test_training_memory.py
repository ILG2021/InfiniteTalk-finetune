"""Real PyTorch numerical regressions; no Wan/14B imports or model downloads."""
import copy
import ast
import importlib.util
import math
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    path = Path(__file__).resolve().parents[1]/'wan/utils/training_memory.py'
    spec = importlib.util.spec_from_file_location('training_memory_under_test', path)
    memory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(memory)


@unittest.skipIf(torch is None, 'PyTorch required for real autograd tests')
class TrainingMemoryTests(unittest.TestCase):
    def test_fp8_small_channels_survive_large_channel_outliers(self):
        base = torch.nn.Linear(4, 3, bias=False).requires_grad_(False)
        values = torch.tensor([[1000., -500., 250., -125.],
                               [1e-5, -5e-6, 2.5e-6, -1.25e-6], [0., 0., 0., 0.]])
        base.weight.copy_(values)
        layer = memory.FrozenLinear(base, fp8=True)
        actual = layer.weight.float()*layer.scale_weight
        torch.testing.assert_close(actual, values, rtol=.05, atol=1e-9)
        self.assertEqual(layer.scale_weight.shape, (3, 1))
        self.assertEqual(actual[2].count_nonzero(), 0)

    def test_fp8_storage_survives_parent_dtype_conversion(self):
        for offload in (False, True):
            for bias in (False, True):
                base = torch.nn.Linear(4, 3, bias=bias).requires_grad_(False)
                layer = memory.FrozenLinear(base, fp8=True, offload=offload)
                scale, weight = layer.scale_weight.clone(), layer.weight.float().clone()
                parent = torch.nn.Sequential(layer).bfloat16().float()
                self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
                self.assertEqual(layer.scale_weight.dtype, torch.float32)
                torch.testing.assert_close(layer.scale_weight, scale)
                torch.testing.assert_close(layer.weight.float(), weight)
                self.assertTrue(torch.isfinite(parent(torch.randn(2, 4))).all())
        layer = memory.FrozenLinear(torch.nn.Linear(4, 3).requires_grad_(False), fp8=True, offload=True)
        torch.nn.Sequential(layer).to('meta')
        self.assertEqual(layer.weight.device.type, 'cpu')
        self.assertEqual(layer.bias.device.type, 'cpu')

    def test_reject_nonfinite_weights_and_trainable_base(self):
        with self.assertRaises(ValueError):
            memory.FrozenLinear(torch.nn.Linear(2, 2), fp8=True)
        for value in (float('nan'), float('inf')):
            base = torch.nn.Linear(2, 2).requires_grad_(False)
            base.weight[0, 0] = value
            with self.assertRaises(ValueError):
                memory.FrozenLinear(base, fp8=True)

    def test_real_lora_fp8_checkpoint_optimizer_updates_match_dense_reference(self):
        # Execute the project's actual LoRALinear without importing 14B deps.
        source = (Path(__file__).resolve().parents[1]/'train_lora.py').read_text(encoding='utf-8')
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'LoRALinear')
        ns = {'torch': torch, 'nn': torch.nn, 'math': math}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'train_lora.py', 'exec'), ns)
        for amp_enabled in (False, True):
            with self.subTest(amp=amp_enabled):
                torch.manual_seed(47)
                model = torch.nn.Module()
                model.blocks = torch.nn.ModuleList([ns['LoRALinear'](
                    torch.nn.Linear(8, 8).bfloat16(), rank=3, alpha=3) for _ in range(2)])
                memory.prepare_model_memory(model, 'cpu', fp8=True, blocks_to_offload=2)
                reference = copy.deepcopy(model)
                for block in reference.blocks:
                    frozen = block.original_linear
                    dense = torch.nn.Linear(8, 8, dtype=torch.bfloat16).requires_grad_(False)
                    dense.weight.copy_(frozen.weight.bfloat16()*frozen.scale_weight.bfloat16())
                    dense.bias.copy_(frozen.bias)
                    block.original_linear = dense
                p_actual = [p for p in model.parameters() if p.requires_grad]
                p_expected = [p for p in reference.parameters() if p.requires_grad]
                optimizers = [torch.optim.AdamW(p, lr=.01) for p in (p_actual, p_expected)]
                x = torch.randn(2, 5, 8)
                target = torch.randn_like(x)
                for step in range(3):
                    outputs = []
                    for candidate, optimizer in zip((model, reference), optimizers):
                        optimizer.zero_grad(set_to_none=True)
                        out = x
                        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp_enabled):
                            for block in candidate.blocks:
                                out = memory.checkpoint_block(block, out, offload=True) if candidate is model else block(out)
                            loss = (out.float()-target).square().mean()
                        loss.backward()
                        outputs.append(out)
                    torch.testing.assert_close(*outputs)
                    for a, b in zip(p_actual, p_expected):
                        self.assertIsNotNone(a.grad)
                        torch.testing.assert_close(a.grad, b.grad)
                    self.assertGreater(sum(p.grad.abs().sum() for p in p_actual).item(), 0)
                    for optimizer in optimizers:
                        optimizer.step()
                    for a, b in zip(p_actual, p_expected):
                        torch.testing.assert_close(a, b)

    def test_frozen_linear_matches_forward_and_input_gradient(self):
        torch.manual_seed(12)
        original = torch.nn.Linear(7, 5, dtype=torch.float64).requires_grad_(False)
        layer = memory.FrozenLinear(original, offload=True)
        x = torch.randn(2, 3, 7, dtype=torch.float64, requires_grad=True)
        expected = original(x)
        actual = layer(x)
        torch.testing.assert_close(actual, expected)
        expected_grad, = torch.autograd.grad(expected.square().sum(), x)
        actual_grad, = torch.autograd.grad(actual.square().sum(), x)
        torch.testing.assert_close(actual_grad, expected_grad)
        self.assertTrue(torch.autograd.gradcheck(layer, (x,)))
        self.assertIsNone(layer.weight.grad)

    def test_fp8_matches_explicit_dequantized_reference_and_saves_no_dense_weight(self):
        original = torch.nn.Linear(9, 6).requires_grad_(False)
        layer = memory.FrozenLinear(original, fp8=True, offload=True)
        saved = []
        def pack(t):
            saved.append((t.dtype, tuple(t.shape)))
            return t
        x = torch.randn(2, 9, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            actual = layer(x)
        dense = layer.weight.float()*layer.scale_weight
        expected = torch.nn.functional.linear(x, dense, layer.bias)
        torch.testing.assert_close(actual, expected)
        ga, = torch.autograd.grad(actual.sum(), x)
        ge, = torch.autograd.grad(expected.sum(), x)
        torch.testing.assert_close(ga, ge)
        self.assertIn((torch.float8_e4m3fn, (6, 9)), saved)
        self.assertNotIn((torch.float32, (6, 9)), saved)
        self.assertNotIn((torch.float32, (2, 9)), saved)

    def test_zero_fp8_weights_remain_finite(self):
        original = torch.nn.Linear(4, 4, bias=False).requires_grad_(False)
        original.weight.zero_()
        layer = memory.FrozenLinear(original, fp8=True)
        self.assertTrue(torch.isfinite(layer(torch.ones(2, 4))).all())

    def test_autocast_restores_original_input_gradient_dtype(self):
        original = torch.nn.Linear(8, 6).requires_grad_(False)
        layer = memory.FrozenLinear(original, offload=True)
        x = torch.randn(2, 8, requires_grad=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            expected, actual = original(x), layer(x)
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected)
        ga, = torch.autograd.grad(actual.float().sum(), x)
        ge, = torch.autograd.grad(expected.float().sum(), x)
        self.assertEqual(ga.dtype, torch.float32)
        torch.testing.assert_close(ga, ge)

    def test_checkpoint_offload_matches_adapter_gradients_without_input_grads(self):
        class Adapter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Linear(8, 8).requires_grad_(False)
                self.down = torch.nn.Linear(8, 3, bias=False)
                self.up = torch.nn.Linear(3, 8, bias=False)
                self.dropout = torch.nn.Dropout(.2)
            def forward(self, x, condition):
                return self.dropout(torch.tanh(self.base(x)+self.up(self.down(x))+condition))
        reference = torch.nn.ModuleList([Adapter(), Adapter()])
        streamed = copy.deepcopy(reference)
        for layer in streamed:
            layer.base = memory.FrozenLinear(layer.base, offload=True)
        x = torch.randn(2, 4, 8)  # Deliberately no input requires_grad.
        condition = torch.randn(2, 4, 8, requires_grad=True)
        torch.manual_seed(73)
        expected = x
        for layer in reference:
            expected = layer(expected, condition)
        expected.square().mean().backward()
        condition_grad = condition.grad.clone()
        condition.grad = None
        torch.manual_seed(73)
        actual = x
        for layer in streamed:
            actual = memory.checkpoint_block(layer, actual, condition=condition, offload=True)
        actual.square().mean().backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(condition.grad, condition_grad)
        for a, b in zip(reference.parameters(), streamed.parameters()):
            if a.requires_grad:
                self.assertIsNotNone(b.grad)
                torch.testing.assert_close(a.grad, b.grad)

    def test_placement_keeps_only_frozen_selected_weights_on_cpu(self):
        model = torch.nn.Module()
        model.blocks = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Linear(4, 4).requires_grad_(False), torch.nn.Linear(4, 2)) for _ in range(3)])
        stats = memory.prepare_model_memory(model, torch.device('meta'), fp8=True, blocks_to_offload=2)
        self.assertEqual(stats['converted_linears'], 3)
        self.assertGreater(stats['cpu_weight_bytes'], 0)
        self.assertEqual(model.blocks[0][0].weight.device.type, 'meta')
        self.assertEqual(model.blocks[1][0].weight.device.type, 'cpu')
        for block in model.blocks:
            self.assertEqual(block[1].weight.device.type, 'meta')
            self.assertTrue(block[1].weight.requires_grad)
        with self.assertRaises(ValueError):
            memory.prepare_model_memory(model, 'cpu', blocks_to_offload=4)

    def test_bounded_cpu_cache_evicts_least_recently_used(self):
        cache = memory.TensorLRUCache(32)
        cache['a'] = torch.zeros(4)
        cache['b'] = [torch.zeros(4)]
        cache['a']
        cache['c'] = torch.ones(4)
        self.assertNotIn('b', cache)
        self.assertEqual(cache.bytes, 32)
        cache['oversized'] = torch.zeros(100)
        self.assertNotIn('oversized', cache)
        with self.assertRaises(ValueError):
            cache['grad'] = torch.zeros(2, requires_grad=True)

    def test_sdpa_masks_padding_and_preserves_gradients(self):
        q = torch.randn(2, 5, 2, 4, dtype=torch.float64, requires_grad=True)
        k = torch.randn(2, 6, 2, 4, dtype=torch.float64, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        actual = memory.sdpa_attention(q, k, v, q_lens=[3, 5], k_lens=[4, 6],
                                       dtype=torch.float64, softmax_scale=.7)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q[:1, :3].transpose(1, 2), k[:1, :4].transpose(1, 2), v[:1, :4].transpose(1, 2), scale=.7).transpose(1, 2)
        torch.testing.assert_close(actual[:1, :3], expected)
        self.assertEqual(actual[0, 3:].abs().sum().item(), 0)
        actual.sum().backward()
        self.assertEqual(k.grad[0, 4:].abs().sum().item(), 0)
        self.assertGreater(q.grad.abs().sum().item(), 0)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA required')
    def test_cuda_streaming_with_pinned_masters_and_checkpoint(self):
        original = torch.nn.Linear(16, 16, dtype=torch.bfloat16).requires_grad_(False)
        layer = memory.FrozenLinear(original, fp8=True, offload=True, pin_memory=True)
        x = torch.randn(2, 8, 16, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        actual = memory.checkpoint_block(layer, x, offload=True, pin_memory=True)
        expected = torch.nn.functional.linear(x, layer.weight.cuda().bfloat16()*layer.scale_weight.cuda().bfloat16(), layer.bias.cuda())
        torch.testing.assert_close(actual, expected)
        ga, = torch.autograd.grad(actual.float().square().mean(), x)
        ge, = torch.autograd.grad(expected.float().square().mean(), x)
        torch.testing.assert_close(ga, ge)
        self.assertEqual(layer.weight.device.type, 'cpu')


if __name__ == '__main__':
    unittest.main()
