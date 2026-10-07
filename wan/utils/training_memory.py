"""Single-GPU LoRA memory tools using native PyTorch (including Windows).

Inspired by SimpleTuner's frozen-weight streaming and offloaded checkpointing.
This deliberately uses the current CUDA stream, without speculative prefetch,
mutable ping-pong weights, or a dependency on RamTorch/Triton.
"""
from collections import OrderedDict
from contextlib import nullcontext
import json

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import sdpa_kernel, SDPBackend


def uses_fp32_compute(name):
    # Wan explicitly disables autocast in its timestep MLPs and output head.
    return name.split('.', 1)[0] in ('time_embedding', 'time_projection', 'head')


def _materialize(weight, scale, device, dtype):
    # Transfer compressed weights before expanding, so PCIe carries FP8 bytes.
    value = weight.to(device=device, non_blocking=weight.is_pinned())
    value = value.to(dtype=dtype)
    if scale is not None:
        value = value * scale.to(device=device, dtype=dtype)
    return value


class _FrozenLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, scale):
        dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        ctx.input_dtype, ctx.compute_dtype = x.dtype, dtype
        # Save only compressed/master weights, never input activations or an
        # expanded BF16 weight. Frozen linear backward only needs W for dX.
        ctx.save_for_backward(weight, scale)
        with torch.autocast(x.device.type, enabled=False):
            w = _materialize(weight, scale, x.device, dtype)
            b = None if bias is None else bias.to(device=x.device, dtype=dtype)
            return F.linear(x.to(dtype), w, b)

    @staticmethod
    def backward(ctx, grad_output):
        weight, scale = ctx.saved_tensors
        with torch.autocast(grad_output.device.type, enabled=False):
            w = _materialize(weight, scale, grad_output.device, ctx.compute_dtype)
            grad_input = torch.matmul(grad_output.to(ctx.compute_dtype), w)
        return grad_input.to(ctx.input_dtype), None, None, None


class FrozenLinear(nn.Module):
    """Immutable CPU or device master weights; adapter parameters stay separate."""
    def __init__(self, linear, *, fp8=False, offload=False, pin_memory=False):
        super().__init__()
        if any(p.requires_grad for p in linear.parameters()):
            raise ValueError('FrozenLinear cannot replace trainable parameters')
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.offload = offload
        weight = linear.weight.detach().cpu()
        scale = None
        if fp8:
            dense = weight.float()
            if not torch.isfinite(dense).all():
                raise ValueError('Cannot quantize non-finite base weights to FP8')
            limit = torch.finfo(torch.float8_e4m3fn).max
            # One FP32 scale per output channel. A large row must not erase
            # small-valued rows; zero rows use scale=1 and stay exactly zero.
            maximum = dense.abs().amax(dim=1, keepdim=True)
            scale = torch.where(maximum > 0,
                                (maximum / limit).clamp_min(torch.finfo(torch.float32).tiny),
                                torch.ones_like(maximum))
            weight = (dense / scale).clamp(-limit, limit).to(torch.float8_e4m3fn)
        if offload and pin_memory:
            weight = weight.pin_memory()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.register_parameter('bias', None if linear.bias is None else
                                nn.Parameter(linear.bias.detach().cpu(), requires_grad=False))
        self.register_buffer('scale_weight', scale)

    def _apply(self, fn, recurse=True):
        # Module.bfloat16()/to(dtype=...) must never expand FP8 storage or round
        # its scale. CPU masters also stay on CPU through parent device moves.
        if self.scale_weight is None and not self.offload:
            return super()._apply(fn, recurse=recurse)
        weight = self._parameters.pop('weight')
        scale = self._buffers.pop('scale_weight')
        bias = self._parameters.pop('bias') if self.offload else None
        try:
            probe = fn(torch.empty(0, dtype=torch.uint8, device=weight.device))
            super()._apply(fn, recurse=recurse)
        finally:
            self._parameters['weight'] = weight
            self._buffers['scale_weight'] = scale
            if self.offload:
                self._parameters['bias'] = bias
        destination = torch.device('cpu') if self.offload else probe.device
        self.weight = nn.Parameter(weight.to(device=destination), requires_grad=False)
        if scale is not None:
            self.scale_weight = scale.to(device=destination, dtype=torch.float32)
        return self

    def forward(self, x):
        return _FrozenLinearFunction.apply(x, self.weight, self.bias, self.scale_weight)

    @classmethod
    def from_quantized(cls, linear, weight, scale, bias, compute_dtype):
        if weight.dtype != torch.float8_e4m3fn or tuple(weight.shape) != tuple(linear.weight.shape):
            raise ValueError('Expected matching E4M3 FP8 weight')
        if scale.dtype != torch.float32 or tuple(scale.shape) != (linear.out_features, 1):
            raise ValueError('Expected FP32 per-output-channel scale')
        if not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError('FP8 scales must be finite and positive')
        if (bias is None) != (linear.bias is None) or (bias is not None and tuple(bias.shape) != (linear.out_features,)):
            raise ValueError('FP8 checkpoint bias does not match the model')
        layer = cls.__new__(cls)
        nn.Module.__init__(layer)
        layer.in_features, layer.out_features = linear.in_features, linear.out_features
        layer.offload = False
        layer.weight = nn.Parameter(weight, requires_grad=False)
        layer.register_buffer('scale_weight', scale)
        layer.register_parameter('bias', None if bias is None else
                                 nn.Parameter(bias.to(compute_dtype), requires_grad=False))
        return layer


def load_prequantized_fp8(model, checkpoint, quantization_map, compute_dtype=torch.bfloat16):
    """Load the official Quanto weight-only FP8 layout without requantization.

    This is a full merged DiT checkpoint. Keep its unquantized layers in compute
    precision; reject other layouts rather than guessing ComfyUI/INT8 formats.
    """
    from safetensors import safe_open
    with open(quantization_map, encoding='utf-8') as handle:
        mapping = json.load(handle)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError('Missing FP8 quantization map')
    state, consumed = {}, set()
    with safe_open(str(checkpoint), framework='pt', device='cpu') as handle:
        keys = set(handle.keys())
        for name, spec in mapping.items():
            if spec.get('weights') != 'qfloat8_e4m3fn' or spec.get('activations') != 'none':
                raise ValueError(f'Unsupported quantization at {name}: {spec}')
            original = model.get_submodule(name)
            if not isinstance(original, nn.Linear):
                raise ValueError(f'FP8 map must refer to a Linear: {name}')
            weight_key, scale_key = name+'.weight._data', name+'.weight._scale'
            bias_key = name+'.bias'
            weight, scale = handle.get_tensor(weight_key), handle.get_tensor(scale_key)
            bias = handle.get_tensor(bias_key) if original.bias is not None else None
            layer = FrozenLinear.from_quantized(original, weight, scale, bias, compute_dtype)
            parent_name, _, child = name.rpartition('.')
            setattr(model.get_submodule(parent_name) if parent_name else model, child, layer)
            state[name+'.weight'], state[name+'.scale_weight'] = weight, scale
            consumed.update((weight_key, scale_key))
            if bias is not None:
                state[bias_key] = layer.bias.detach()
                consumed.add(bias_key)
            # Quanto keeps activation scale placeholders even for weight-only
            # quantization. They are unused when activations='none'.
            consumed.update({name+'.input_scale', name+'.output_scale'} & keys)
        for key in keys-consumed:
            value = handle.get_tensor(key)
            if value.dtype == torch.float8_e4m3fn:
                raise ValueError(f'FP8 tensor absent from quantization map: {key}')
            dtype = torch.float32 if uses_fp32_compute(key) else compute_dtype
            state[key] = value.to(dtype) if value.is_floating_point() else value
    model.load_state_dict(state, strict=True, assign=True)
    return len(mapping)


def prepare_model_memory(model, device, *, fp8=False, blocks_to_offload=0, pin_memory=False):
    """Quantize on CPU; place each leaf directly in its final device.

    Offload frozen Linear weights in the last N blocks. No trainable tensor is
    offloaded and no whole-model GPU allocation happens before the offload.
    """
    depth = len(model.blocks)
    if not 0 <= blocks_to_offload <= depth:
        raise ValueError(f'blocks_to_offload must be between 0 and {depth}')
    replaced = streamed_bytes = 0
    for name, module in list(model.named_modules()):
        parts = name.split('.')
        offload = len(parts) > 2 and parts[0] == 'blocks' and int(parts[1]) >= depth-blocks_to_offload
        if isinstance(module, FrozenLinear):
            module.offload = offload
            if offload:
                module.cpu()
                if pin_memory:
                    module.weight = nn.Parameter(module.weight.pin_memory(), requires_grad=False)
                streamed_bytes += sum(t.numel()*t.element_size() for t in
                                      (*module.parameters(), *module.buffers()))
            continue
        if not isinstance(module, nn.Linear) or any(p.requires_grad for p in module.parameters()):
            continue
        quantize = fp8 and not uses_fp32_compute(name)
        if not quantize and not offload:
            continue
        replacement = FrozenLinear(module, fp8=quantize, offload=offload, pin_memory=pin_memory)
        parent_name, _, child = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, replacement)
        replaced += 1
        if offload:
            streamed_bytes += sum(t.numel()*t.element_size() for t in
                                  (*replacement.parameters(), *replacement.buffers()))

    def place(module):
        if isinstance(module, FrozenLinear) and module.offload:
            return
        for child in module.children():
            place(child)
        # Moving only local state avoids traversing into CPU master weights.
        module._apply(lambda tensor: tensor.to(device), recurse=False)
    place(model)
    return {'converted_linears': replaced, 'cpu_weight_bytes': streamed_bytes}


def checkpoint_block(block, x, *, offload=False, pin_memory=False, **kwargs):
    # The outer hooks pack checkpoint boundary inputs; PyTorch's inner hooks
    # still discard/recompute intermediates. kwargs are passed explicitly so
    # conditioning gradients and RNG replay retain native checkpoint semantics.
    context = torch.autograd.graph.save_on_cpu(pin_memory=pin_memory) if offload else nullcontext()
    with context:
        return checkpoint(block, x, use_reentrant=False, **kwargs)


class TensorLRUCache:
    """Bound CPU caches by tensor bytes, including nested text embedding lists."""
    def __init__(self, max_bytes):
        self.max_bytes = max(0, int(max_bytes))
        self.bytes = 0
        self.values = OrderedDict()

    @staticmethod
    def size(value):
        if isinstance(value, torch.Tensor):
            if value.device.type != 'cpu' or value.requires_grad:
                raise ValueError('Cache only detached CPU tensors')
            return value.numel()*value.element_size()
        if isinstance(value, (tuple, list)):
            return sum(TensorLRUCache.size(v) for v in value)
        raise TypeError('Expected tensor or tensor list')

    def __contains__(self, key):
        return key in self.values

    def __getitem__(self, key):
        value, size = self.values.pop(key)
        self.values[key] = (value, size)
        return value

    def __setitem__(self, key, value):
        size = self.size(value)
        if key in self.values:
            self.bytes -= self.values.pop(key)[1]
        if size > self.max_bytes:
            return
        while self.values and self.bytes+size > self.max_bytes:
            self.bytes -= self.values.popitem(last=False)[1][1]
        self.values[key] = (value, size)
        self.bytes += size


def sdpa_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
                   softmax_scale=None, q_scale=None, causal=False,
                   window_size=(-1, -1), dtype=torch.bfloat16, **ignored):
    """BLHD attention, with real sequence lengths and no quadratic pad mask.

    Slicing each batch item preserves FlashAttention's variable-length contract.
    Training uses batch size one. Native SDPA chooses an available CUDA kernel.
    """
    if tuple(window_size) != (-1, -1):
        raise ValueError('Native training SDPA currently requires global attention')
    def attend(qi, ki, vi):
        qi, ki, vi = (t.transpose(1, 2).to(dtype) for t in (qi, ki, vi))
        if q_scale is not None:
            qi = qi * q_scale
        # Never silently allocate a quadratic score matrix on CUDA when fused
        # kernels are unavailable. CPU math is useful for numerical regression.
        context = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]) \
            if qi.device.type == 'cuda' else nullcontext()
        with context:
            out = F.scaled_dot_product_attention(qi, ki, vi, dropout_p=dropout_p,
                                                 is_causal=causal, scale=softmax_scale)
        return out.transpose(1, 2).to(q.dtype)
    if q_lens is None and k_lens is None:
        return attend(q, k, v)
    result = []
    for i in range(q.shape[0]):
        nq = q.shape[1] if q_lens is None else int(q_lens[i])
        nk = k.shape[1] if k_lens is None else int(k_lens[i])
        if not 0 < nq <= q.shape[1] or not 0 < nk <= k.shape[1]:
            raise ValueError('Invalid attention sequence lengths')
        out = attend(q[i:i+1, :nq], k[i:i+1, :nk], v[i:i+1, :nk])
        result.append(F.pad(out, (0, 0, 0, 0, 0, q.shape[1]-nq)))
    return torch.cat(result, dim=0)
