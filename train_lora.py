# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# LoRA fine-tuning script for InfiniteTalk (single-person)
# Designed to run on a single RTX 5090 (32GB VRAM)

import argparse
import gc
import json
import logging
import math
import os
import random
import sys
import warnings
from datetime import datetime
from typing import Optional, Union, List, Any, Dict, cast, Tuple

from safetensors import safe_open

warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from safetensors.torch import save_file
from PIL import Image
from tqdm import tqdm

import wan
from wan.configs import WAN_CONFIGS
from wan.modules.multitalk_model import (
    WanModel, sinusoidal_embedding_1d, rope_params, rope_apply,
    AudioProjModel, WanLayerNorm, WanRMSNorm
)
from wan.utils.training_memory import prepare_model_memory, TensorLRUCache, FrozenLinear

DEFAULT_TARGET_MODULES = [
    # Identity / Visual appearance pathways
    'self_attn.q', 'self_attn.k', 'self_attn.v', 'self_attn.o',
    'cross_attn.q', 'cross_attn.k', 'cross_attn.v', 'cross_attn.k_img', 'cross_attn.v_img', 'cross_attn.o',
    'ffn.0', 'ffn.2',
    # Audio-mouth sync pathways
    'audio_cross_attn.q_linear', 'audio_cross_attn.kv_linear', 'audio_cross_attn.proj',
    'audio_proj.proj1', 'audio_proj.proj1_vf', 'audio_proj.proj2', 'audio_proj.proj3',
]

# ============================================================
# LoRA Module
# ============================================================

class LoRALinear(nn.Module):
    """LoRA adapter for a frozen Linear layer."""

    def __init__(self, original_linear: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # LoRA A (down projection) and B (up projection)
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

        # Initialize: A with kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        # Freeze original
        for param in self.original_linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Original path (frozen) + LoRA path (trainable)
        # Wan blocks may supply FP32 activations. The frozen branch computes
        # in BF16 while adapters retain FP32 parameters (AMP handles matmuls).
        result = self.original_linear(x.to(torch.bfloat16))
        lora_out = self.lora_up(self.lora_down(x.to(self.lora_down.weight.dtype))) * self.scaling
        return result + lora_out.to(result.dtype)


def apply_lora_to_model(model: nn.Module, rank: int = 16, alpha: float = 16.0,
                        target_modules: Optional[List[str]] = None):
    """
    Apply LoRA adapters to specified modules in the model.

    Args:
        model: WanModel instance
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        target_modules: list of module name patterns to apply LoRA to.
            Defaults to audio_cross_attn layers.
    """
    if target_modules is None:
        target_modules = DEFAULT_TARGET_MODULES

    lora_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, FrozenLinear)) or type(module).__name__ == "QLinear":
            # Check if the full dotted name ends with one of the target patterns
            if any(name == target or name.endswith('.' + target) for target in target_modules):
                lora_modules[name] = module

    # Replace with LoRA versions
    applied_count: int = 0
    for name, original_linear in lora_modules.items():
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)

        lora_linear = LoRALinear(original_linear, rank=rank, alpha=alpha)
        setattr(parent, parts[-1], lora_linear)
        applied_count = cast(int, applied_count + 1)

    logging.info(f"Applied LoRA (rank={rank}, alpha={alpha}) to {applied_count} layers")
    return model




def extract_lora_state_dict(model, *, comfyui=False):
    """
    Extract LoRA weights in a format compatible with wan_lora.py.

    Output key format:
        diffusion_model.{path}.lora_down.weight   (LoRA A matrix)
        diffusion_model.{path}.lora_up.weight     (LoRA B matrix)
    """
    state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if comfyui and name.startswith('audio_proj.'):
                name = 'multitalk_audio_proj.' + name[len('audio_proj.'):]
            prefix = f"diffusion_model.{name}"
            state_dict[f"{prefix}.lora_down.weight"] = module.lora_down.weight.data.clone().cpu()
            # Bake training-time LoRA scaling (alpha/rank) into lora_up so
            # merge-time delta = (B_scaled @ A) * lora_scale remains equivalent.
            state_dict[f"{prefix}.lora_up.weight"] = (module.lora_up.weight.data * module.scaling).clone().cpu()
    return state_dict


def _pixel_frames_for_latent_len(t_lat: int, vae_temporal: int = 4) -> int:
    """
    Pixel-frame count F such that VAE latent temporal length is t_lat and
    WanModel audio path can reshape latter frames: (F - 1) % vae_temporal == 0.
    Same relation as inference: T = (F - 1) // vae_temporal + 1  =>  F = (T - 1) * vae_temporal + 1.
    """
    return (t_lat - 1) * vae_temporal + 1


def _align_audio_frames_to_latent(audio_b: torch.Tensor, f_req: int) -> torch.Tensor:
    """audio_b: [B, F, window, 12, 768]. Pad (repeat last) or trim to length f_req."""
    f_cur = audio_b.shape[1]
    if f_cur == f_req:
        return audio_b
    if f_cur > f_req:
        return audio_b[:, :f_req, ...]
    pad_n = f_req - f_cur
    last = audio_b[:, -1:]
    tail = last.repeat(1, pad_n, *((1,) * (audio_b.ndim - 2)))
    return torch.cat([audio_b, tail], dim=1)


def _sample_training_timestep(device):
    """Standard logit-normal sigma: sigmoid(N(0, 1)); keep sampling in FP32."""
    return torch.sigmoid(torch.randn(1, device=device, dtype=torch.float32))


def _reference_cache_key(ref_image_name, frame_num):
    """Only metadata-backed, fixed images have a reusable identity."""
    return (ref_image_name, frame_num) if ref_image_name else None


def _prepared_reference_name(sample, frame_num, reference_mode):
    """Use references sampled outside an exact, pre-captioned source window."""
    if sample.get('reference_policy') == 'adjacent':
        if sample.get('num_frames') != frame_num:
            raise ValueError('Prepared clip length must equal --frame_num to keep captions aligned')
        start, end, ref = (sample[k] for k in ('start_frame', 'end_frame', 'reference_frame'))
        if end - start != frame_num or start <= ref < end:
            raise ValueError('Prepared adjacent reference must be outside the source clip')
        if not sample.get('ref_image'):
            raise ValueError('Prepared adjacent clip is missing ref_image')
        return sample['ref_image']
    return sample.get('ref_image') if reference_mode == 'fixed' else None


def _sample_adjacent_reference(start, length, total_frames, neighbor_frames):
    """Sample outside the current window, never clamp back inside it."""
    ranges = [(max(0, start - neighbor_frames), start - 1),
              (start + length, min(total_frames - 1, start + length - 1 + neighbor_frames))]
    ranges = [(lo, hi) for lo, hi in ranges if lo <= hi]
    if not ranges:
        raise ValueError("No adjacent reference frame; use a longer video or --reference_mode fixed with ref_image")
    lo, hi = random.choice(ranges)
    return random.randint(lo, hi)


def _flow_inputs(clean, noise, t_frac, context=None):
    """Match inference's clean prefix; only supervise generated latent frames."""
    noisy = t_frac * noise + (1 - t_frac) * clean
    velocity = noise - clean
    if context is None:
        prefix, noisy, velocity = clean[:, :1], noisy[:, 1:], velocity[:, 1:]
    else:
        prefix = context
    model_input = torch.cat([prefix, noisy], dim=1)
    target = torch.cat([torch.zeros_like(prefix), velocity], dim=1)
    mask = torch.cat([torch.zeros_like(prefix), torch.ones_like(velocity)], dim=1)
    return model_input, target, mask


def _serialize_args(args: argparse.Namespace) -> Dict[str, Any]:
    """JSON-friendly hyperparameter dict for checkpoints."""
    out: Dict[str, Any] = {}
    
    # Pre-resolve defaults for clarity in checkpoint jsons
    resolved_args = vars(args).copy()
    if resolved_args.get('quant') == 'fp8':
        resolved_args['fp8_weight_scaling'] = 'checkpoint_preserved' if resolved_args.get('fp8_checkpoint') else 'per_output_channel_v1'
    if not resolved_args.get("tensorboard_dir"):
        resolved_args["tensorboard_dir"] = "output/my_lora/tensorboard"

    for k, v in resolved_args.items():
        if v is None or isinstance(v, (bool, int, float, str)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [str(x) if not isinstance(x, (int, float, str, bool, type(None))) else x for x in v]
        else:
            out[k] = str(v)
    return out


def _get_rng_state() -> Dict[str, Any]:
    st: Dict[str, Any] = {
        'torch': torch.get_rng_state(),
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    }
    if torch.cuda.is_available():
        st['cuda'] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st: Dict[str, Any]) -> None:
    torch.set_rng_state(st['torch'])
    np.random.set_state(st['numpy'])
    random.setstate(st['python'])
    if 'cuda' in st and st['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st['cuda'])


def _save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return cast(Dict[str, Any], json.load(f))


def _is_legacy_training_pt(path: str) -> bool:
    return os.path.isfile(path) and path.lower().endswith(".pt")


def _resolve_checkpoint_dir(path: str) -> str:
    """Accept either a checkpoint directory or a file inside it."""
    if os.path.isdir(path):
        return path
    parent = os.path.dirname(path)
    if parent and os.path.isdir(parent) and os.path.isfile(os.path.join(parent, "trainer_state.json")):
        return parent
    return path


def _get_checkpoint_dir(output_dir: str, suffix: Optional[str], step: int) -> str:
    name = f"checkpoint-{suffix}" if suffix else f"checkpoint-{step}"
    return os.path.join(output_dir, name)


def _extract_trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def save_training_checkpoint(
        output_dir: str,
        step: int,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        args: argparse.Namespace,
        suffix: Optional[str] = None,
        write_latest: bool = True,
) -> Tuple[str, str]:
    """
    Saves a Transformers/PEFT-like checkpoint directory containing:
    - adapter_model.safetensors (trainable LoRA parameters only)
    - optimizer.pt / scheduler.pt / scaler.pt
    - rng_state.pth
    - trainer_state.json

    For inference, you typically only need a wan_lora-compatible safetensors. To avoid duplication,
    this function exports the inference LoRA file only when suffix is provided (e.g. "final").
    """
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = _get_checkpoint_dir(output_dir, suffix=suffix, step=step)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1) Trainable adapter weights
    adapter_sd = _extract_trainable_state_dict(model)
    adapter_path = os.path.join(ckpt_dir, "adapter_model.safetensors")
    save_file(adapter_sd, adapter_path)

    # 2) Training states
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
    torch.save(_get_rng_state(), os.path.join(ckpt_dir, "rng_state.pth"))

    trainer_state = {
        "format_version": 3,
        "step": step,
        "epoch": epoch,
        "args": _serialize_args(args),
    }
    _save_json(os.path.join(ckpt_dir, "trainer_state.json"), trainer_state)

    # 3) Optional inference LoRA export (wan_lora-compatible) to avoid duplication.
    inference_lora_path = ""
    if suffix is not None:
        lora_sd = extract_lora_state_dict(model)
        inference_lora_path = os.path.join(ckpt_dir, "lora_for_inference.safetensors")
        save_file(lora_sd, inference_lora_path)
        comfy_path = os.path.join(ckpt_dir, 'lora_for_comfyui.safetensors')
        save_file(extract_lora_state_dict(model, comfyui=True), comfy_path,
                  metadata={'format': 'WanVideoWrapper', 'scaling': 'alpha_over_rank_baked_into_up'})

    # 4) Latest pointer (Windows-friendly: a tiny json file in output root)
    if write_latest:
        latest_meta = {
            "checkpoint_dir": ckpt_dir,
            "step": step,
            "epoch": epoch,
        }
        _save_json(os.path.join(output_dir, "checkpoint_latest.json"), latest_meta)

    return adapter_path, inference_lora_path


def _trim_optimizer_state_dict(
        saved_sd: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
) -> Dict[str, Any]:
    """
    Trim a saved optimizer state_dict to fit the current optimizer's param groups.
    Handles the case where the current run has fewer trainable params (e.g. audio
    layers removed via --no-train_audio). Params that existed in the checkpoint but
    are no longer present are dropped; their momentum/state tensors are discarded.
    """
    saved_groups = saved_sd["param_groups"]
    curr_groups  = optimizer.param_groups

    new_state  = {}
    new_groups = []
    new_id = 0

    for g_idx, curr_g in enumerate(curr_groups):
        n_params = len(curr_g["params"])      # How many params current group expects

        if g_idx < len(saved_groups):
            saved_g  = saved_groups[g_idx]
            old_ids  = saved_g["params"]      # Param IDs as stored in checkpoint

            # Copy state for the first n_params entries (rest are the removed layers)
            for i, old_id in enumerate(old_ids[:n_params]):
                if old_id in saved_sd["state"]:
                    new_state[new_id + i] = saved_sd["state"][old_id]

            # Build trimmed group (same hyper-params, fresh param ID list)
            trimmed = {k: v for k, v in saved_g.items() if k != "params"}
            trimmed["params"] = list(range(new_id, new_id + n_params))

            dropped = max(0, len(old_ids) - n_params)
            if dropped:
                logging.warning(
                    f"Optimizer group {g_idx}: dropped {dropped} params "
                    f"(not present in current model, e.g. audio layers)."
                )
        else:
            # Checkpoint has fewer groups than current optimizer — use current hyper-params
            trimmed = {k: v for k, v in curr_g.items() if k != "params"}
            trimmed["params"] = list(range(new_id, new_id + n_params))

        new_groups.append(trimmed)
        new_id += n_params

    return {"state": new_state, "param_groups": new_groups}


def _validate_resume_configuration(saved, current):
    if current is None:
        return
    for key in ('lora_rank', 'lora_alpha', 'train_audio'):
        if key in saved and saved[key] != getattr(current, key):
            raise ValueError(f'Cannot resume with changed {key}: checkpoint={saved[key]}, current={getattr(current, key)}')


def load_training_checkpoint(
        path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        strict: bool = True,
        expected_args=None,
) -> Tuple[int, int, Dict[str, Any]]:
    """Loads adapter weights + optimizer/scheduler/scaler/RNG. Supports new checkpoint dirs and legacy .pt."""
    if _is_legacy_training_pt(path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        _validate_resume_configuration(ckpt.get('args', {}), expected_args)
        if ckpt.get('format_version') != 1:
            logging.warning(f"Checkpoint format_version={ckpt.get('format_version')!r}; expected 1")

        trainable_state: Dict[str, torch.Tensor] = ckpt['trainable_state']
        name_to_param = {n: p for n, p in model.named_parameters() if p.requires_grad}
        loaded = 0
        for name, tensor in trainable_state.items():
            if name not in name_to_param:
                if strict:
                    raise KeyError(f"Checkpoint has param {name!r} not found in model")
                logging.warning(f"Skipping ckpt param not in model: {name}")
                continue
            p = name_to_param[name]
            if tuple(p.shape) != tuple(tensor.shape):
                raise ValueError(f'Adapter shape mismatch: {name}')
            p.data.copy_(tensor.to(p.device, dtype=p.dtype))
            loaded += 1
        model_keys = set(name_to_param.keys())
        ckpt_keys = set(trainable_state.keys())
        if strict and model_keys != ckpt_keys:
            raise RuntimeError(
                f"Trainable keys mismatch: only_in_model={repr(model_keys - ckpt_keys)} "
                f"only_in_ckpt={repr(ckpt_keys - model_keys)}"
            )
        if not strict and model_keys != ckpt_keys:
            logging.warning(
                f"Trainable keys differ: only_in_model={repr(model_keys - ckpt_keys)} "
                f"only_in_ckpt={repr(ckpt_keys - model_keys)}"
            )

        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        _set_rng_state(ckpt['rng'])

        step = int(ckpt['step'])
        epoch = int(ckpt.get('epoch', 0))
        saved_args = ckpt.get('args', {})
        logging.info(f"Resumed (legacy) from {path}: step={step}, epoch={epoch}, trainable_tensors={loaded}")
        return step, epoch, saved_args

    ckpt_dir = _resolve_checkpoint_dir(path)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"--resume_from not found: {path}")

    trainer_state_path = os.path.join(ckpt_dir, "trainer_state.json")
    if not os.path.isfile(trainer_state_path):
        raise FileNotFoundError(f"trainer_state.json not found in checkpoint dir: {ckpt_dir}")
    trainer_state = _load_json(trainer_state_path)
    _validate_resume_configuration(trainer_state.get('args', {}), expected_args)
    if int(trainer_state.get("format_version", 0)) not in (2, 3):
        logging.warning(f"Checkpoint format_version={trainer_state.get('format_version')!r}; expected 3")

    # 1) Adapter weights
    adapter_path = os.path.join(ckpt_dir, "adapter_model.safetensors")
    if not os.path.isfile(adapter_path):
        raise FileNotFoundError(f"adapter_model.safetensors not found in checkpoint dir: {ckpt_dir}")

    name_to_param = {n: p for n, p in model.named_parameters() if p.requires_grad}
    loaded = 0
    with safe_open(adapter_path, framework="pt") as f:
        ckpt_keys = set(f.keys())
        for name in f.keys():
            if name not in name_to_param:
                if strict:
                    raise KeyError(f"Checkpoint has param {name!r} not found in model")
                logging.warning(f"Skipping adapter param not in model: {name}")
                continue
            p = name_to_param[name]
            if tuple(p.shape) != tuple(f.get_slice(name).get_shape()):
                raise ValueError(f'Adapter shape mismatch: {name}')
            p.data.copy_(f.get_tensor(name).to(p.device, dtype=p.dtype))
            loaded += 1

    model_keys = set(name_to_param.keys())
    if strict and model_keys != ckpt_keys:
        raise RuntimeError(
            f"Trainable keys mismatch: only_in_model={repr(model_keys - ckpt_keys)} "
            f"only_in_ckpt={repr(ckpt_keys - model_keys)}"
        )
    if not strict and model_keys != ckpt_keys:
        logging.warning(
            f"Trainable keys differ: only_in_model={repr(model_keys - ckpt_keys)} "
            f"only_in_ckpt={repr(ckpt_keys - model_keys)}"
        )

    # 2) Training states
    saved_opt_sd = torch.load(os.path.join(ckpt_dir, "optimizer.pt"), map_location="cpu", weights_only=False)
    opt_sd = saved_opt_sd if strict else _trim_optimizer_state_dict(saved_opt_sd, optimizer)
    optimizer.load_state_dict(opt_sd)
    scheduler.load_state_dict(torch.load(os.path.join(ckpt_dir, "scheduler.pt"), map_location="cpu", weights_only=False))
    _set_rng_state(torch.load(os.path.join(ckpt_dir, "rng_state.pth"), map_location="cpu", weights_only=False))

    step = int(trainer_state.get("step", 0))
    epoch = int(trainer_state.get("epoch", 0))
    saved_args = cast(Dict[str, Any], trainer_state.get("args", {}))
    logging.info(f"Resumed from {ckpt_dir}: step={step}, epoch={epoch}, trainable_tensors={loaded}")
    return step, epoch, saved_args


# ============================================================
# Training Dataset
# ============================================================

class InfiniteTalkDataset(Dataset):
    """
    Dataset for InfiniteTalk LoRA fine-tuning.

    Expected data directory structure:
        data_dir/
        ├── videos/          # Video files (.mp4)
        ├── audio_embs/      # Pre-extracted wav2vec2 embeddings (.pt)
        └── metadata.json    # {"samples": [{"video": "xxx.mp4", "audio_emb": "xxx.pt", "prompt": "..."}]}
    """

    def __init__(
            self,
            data_dir,
            frame_num=81,
            audio_window=5,
            ref_neighbor_frames: int = 25,
            reference_mode: str = 'adjacent',
    ):
        self.data_dir = data_dir
        self.frame_num = frame_num
        self.audio_window = audio_window
        self.ref_neighbor_frames = ref_neighbor_frames
        self.reference_mode = reference_mode

        # Load metadata
        metadata_path = os.path.join(data_dir, 'metadata.json')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        self.samples = self.metadata['samples']
        if not self.samples:
            raise ValueError('Training dataset has no samples')
        logging.info(f"Loaded {len(self.samples)} training samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def _load_video_frames(self, video_path, start_frame, num_frames):
        """Load specific frames from video using robust random-access decord."""
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)

        # Build indices handling padding explicitly
        indices = []
        for i in range(num_frames):
            idx = start_frame + i
            if idx >= total_frames:
                idx = total_frames - 1
            indices.append(idx)

        frames = vr.get_batch(indices).asnumpy()  # Returns (T, H, W, C)

        # Convert to PyTorch format expected by the model
        video = torch.from_numpy(frames).permute(0, 3, 1, 2)  # T, C, H, W
        video = video.float() / 255.0  # [0, 1]
        video = (video - 0.5) * 2  # [-1, 1]
        return video

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ref_image_name = _prepared_reference_name(sample, self.frame_num, self.reference_mode)
        video_path = os.path.join(self.data_dir, 'videos', sample['video'])
        audio_emb_path = os.path.join(self.data_dir, 'audio_embs', sample['audio_emb'])
        prompt = sample.get('prompt')
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Missing caption for {sample['video']}")

        # Load pre-computed audio embedding: [total_frames, 12, 768]
        full_audio_emb = torch.load(audio_emb_path, map_location='cpu', weights_only=True)
        if (full_audio_emb.ndim != 3 or tuple(full_audio_emb.shape[1:]) != (12, 768)
                or not torch.isfinite(full_audio_emb).all()):
            raise ValueError(f'Invalid audio embedding shape/values: {audio_emb_path}')
        total_audio_frames = full_audio_emb.shape[0]

        # We need self.frame_num frames for the entire training segment.
        # To perfectly mirror inference sliding window size, the total window is fixed.
        # For continuation mode, context (9 frames) and target (remaining frames) share this window.
        needed_frames = self.frame_num

        # Load video frames (up to needed_frames)
        # Also get total video frame count for correct reference frame boundary clamping.
        from decord import VideoReader, cpu as decord_cpu
        _vr = VideoReader(video_path, ctx=decord_cpu(0))
        total_video_frames = len(_vr)
        if abs(_vr.get_avg_fps() - 25) > 0.01:
            raise ValueError(f'{video_path}: expected 25 fps; rerun preprocessing')
        del _vr
        usable_frames = min(total_audio_frames, total_video_frames)
        if sample.get('reference_policy') == 'adjacent' and (
                total_video_frames != self.frame_num or total_audio_frames != self.frame_num):
            raise ValueError('Prepared video/audio length does not match its caption window')
        if usable_frames < needed_frames:
            raise ValueError(f"{sample['video']}: need {needed_frames} aligned frames, got {usable_frames}")
        start_frame = random.randint(0, usable_frames - needed_frames)
        video_full = self._load_video_frames(video_path, start_frame, needed_frames)  # T_full, C, H, W

        # Reference frame for identity
        if self.reference_mode == 'fixed' and not ref_image_name:
            raise ValueError(f"{sample['video']}: --reference_mode fixed requires ref_image")
        if ref_image_name:
            # Preselected adjacent source frame, or an explicitly fixed reference.
            from PIL import Image as PILImage
            ref_img_path = os.path.join(self.data_dir, 'ref_images', ref_image_name)
            ref_pil = PILImage.open(ref_img_path).convert('RGB')
            ref_frame = torch.from_numpy(np.array(ref_pil)).permute(2, 0, 1).float() / 255.0
            ref_frame = (ref_frame - 0.5) * 2  # C, H, W
        else:
            # Default: sample outside the current window from a nearby region (M3).
            ref_offset = _sample_adjacent_reference(
                start_frame, needed_frames, total_video_frames, self.ref_neighbor_frames)
            ref_video = self._load_video_frames(video_path, ref_offset, 1)
            ref_frame = ref_video.squeeze(0)  # C, H, W

        if tuple(ref_frame.shape) != tuple(video_full.shape[1:]):
            raise ValueError(f'{video_path}: reference and video must share the same crop/resolution')
        if any(size % 16 for size in video_full.shape[-2:]):
            raise ValueError(f'{video_path}: height and width must be divisible by 16')

        # Extract audio window for the FULL needed frames
        audio_window_indices = (torch.arange(self.audio_window) - self.audio_window // 2)
        total_audio_indices = torch.arange(start_frame, start_frame + needed_frames).unsqueeze(
            1) + audio_window_indices.unsqueeze(0)
        total_audio_indices = total_audio_indices.clamp(0, total_audio_frames - 1)
        full_audio_emb_segment = full_audio_emb[total_audio_indices]  # needed_frames, window, 12, 768

        return {
            'video_full': video_full.permute(1, 0, 2, 3),  # C, needed_frames, H, W
            'ref_frame': ref_frame.squeeze(0),  # C, H, W
            'audio_emb_full': full_audio_emb_segment,  # needed_frames, window, 12, 768
            'prompt': prompt,
            'ref_image_name': ref_image_name or '',  # Used as cache key for CLIP/text embeddings
            'latent_cache_key': f"{sample['video']}:{start_frame}:{needed_frames}",
        }


def count_trainable(model):
    lora_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            lora_params.append(param)

    total_trainable = sum(p.numel() for p in lora_params)
    return total_trainable


# ============================================================
# Training Loop
# ============================================================

def train(args):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)]
    )
    if not (0.0 <= args.first_clip_prob <= 1.0):
        raise ValueError(f"--first_clip_prob must be in [0, 1], got {args.first_clip_prob}")
    if args.frame_num <= 9 or (args.frame_num - 1) % 4 != 0:
        raise ValueError(f"--frame_num must be greater than 9 and satisfy 4n+1, got {args.frame_num}")
    if args.ref_neighbor_frames < 1:
        raise ValueError("--ref_neighbor_frames must be positive")
    if args.lora_rank < 1 or not math.isfinite(args.lora_alpha) or args.lora_alpha <= 0:
        raise ValueError('LoRA rank and alpha must be positive and finite')
    for key in ('lr', 'audio_lr'):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f'{key} must be positive and finite')
    if args.blocks_to_swap < 0 or not math.isfinite(args.cpu_cache_gb) or args.cpu_cache_gb < 0:
        raise ValueError('Offloaded block count and CPU cache budget must be non-negative')
    if args.log_every < 1 or args.max_steps < 1 or args.num_workers < 0:
        raise ValueError('log_every/max_steps must be positive and num_workers non-negative')
    if args.activation_offload and not args.gradient_checkpointing:
        raise ValueError('--activation_offload requires --gradient_checkpointing')
    for name in ('cfg_drop_clip_prob', 'cfg_drop_ref_prob'):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"--{name} must be in [0, 1]")
    if args.quant == 'int8':
        raise ValueError("INT8 training is not implemented. Use --quant fp8 or omit --quant for BF16.")
    if args.fp8_checkpoint:
        args.quant = 'fp8'
        if args.infinitetalk_dir:
            raise ValueError('--fp8_checkpoint already includes InfiniteTalk; omit --infinitetalk_dir')
    elif not args.infinitetalk_dir:
        raise ValueError('Provide --fp8_checkpoint or original --infinitetalk_dir')
    if args.cfg_drop_text_prob < 0 or args.cfg_drop_audio_prob < 0 or args.cfg_drop_both_prob < 0:
        raise ValueError("CFG dropout probabilities must be non-negative")
    if args.cfg_drop_text_prob + args.cfg_drop_audio_prob + args.cfg_drop_both_prob > 1.0:
        raise ValueError(
            "Sum of --cfg_drop_text_prob, --cfg_drop_audio_prob, --cfg_drop_both_prob must be <= 1.0"
        )

    device = torch.device(f'cuda:{args.device_id}')
    torch.cuda.set_device(device)
    from wan.modules import attention as attention_module
    attention_module.TRAINING_ATTENTION_BACKEND = args.attention_backend
    logging.info('GPU %s: %.1f GiB dedicated VRAM; CPU offload blocks=%d, activation offload=%s',
                 torch.cuda.get_device_name(device), torch.cuda.get_device_properties(device).total_memory / 2**30,
                 args.blocks_to_swap, args.activation_offload)

    # ---- Load model ----
    logging.info("Loading InfiniteTalk model...")
    cfg = WAN_CONFIGS['infinitetalk-14B']

    pipeline = wan.InfiniteTalkPipeline(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        quant_dir=None,
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=True,  # Keep T5 on CPU to save VRAM
        lora_dir=None,
        lora_scales=None,
        quant=None,
        init_on_cpu=True,
        auxiliary_device='cpu',
        training_fp8_path=args.fp8_checkpoint,
        dit_path=None,
        infinitetalk_dir=args.infinitetalk_dir,
    )

    model = pipeline.model
    # Auxiliary models start on CPU to avoid a transient startup GPU peak.
    vae = pipeline.vae
    vae.to('cpu' if args.vae_cpu_offload else device)
    clip_model = pipeline.clip
    clip_model.model.to('cpu').float()  # float16 not supported on CPU, cast to float32
    text_encoder = pipeline.text_encoder
    text_encoder.model.to('cpu')
    logging.info('CLIP/T5 on CPU; VAE is %s', 'loaded on demand' if args.vae_cpu_offload else 'GPU resident')

    # ---- Freeze before adding adapters; quantize frozen layers on CPU below ----
    for param in model.parameters():
        param.requires_grad = False

    # ---- Apply LoRA to Linear layers (norms remain frozen, following musubi-tuner convention) ----
    target_modules = list(DEFAULT_TARGET_MODULES)
    
    if not getattr(args, "train_audio", True):
        target_modules = [m for m in target_modules if 'audio' not in m]
        logging.info(f"--no-train_audio is set. Filtered target_modules: {target_modules}")

    model = apply_lora_to_model(model, rank=args.lora_rank, alpha=args.lora_alpha,
                                target_modules=target_modules)

    torch.cuda.empty_cache()
    gc.collect()
    memory_plan = prepare_model_memory(
        model, device, fp8=args.quant == 'fp8' and not args.fp8_checkpoint, blocks_to_offload=args.blocks_to_swap,
        pin_memory=args.offload_pin_memory)
    logging.info('Memory placement: %s; GPU allocated %.2f GiB', memory_plan,
                 torch.cuda.memory_allocated(device) / 2**30)
    model.disable_teacache()
    model.activation_offload = args.activation_offload
    model.activation_offload_pin_memory = args.offload_pin_memory
    
    # ---- Collect trainable parameters (keep in float32 for training stability) ----
    visual_params = []
    audio_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Keep LoRA adapters in float32 exactly like musubi-tuner for max stability
            if 'audio' in name:
                audio_params.append(param)
            else:
                visual_params.append(param)
            logging.info(f"  Trainable: {name} {param.shape} ({param.data.dtype})")

    total_trainable = sum(p.numel() for p in visual_params + audio_params)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable params: {total_trainable:,} / {total_params:,} "
                 f"({100 * total_trainable / total_params:.4f}%)")

    # Enable gradient checkpointing (Explicitly patch for WanModel)
    if args.gradient_checkpointing:
        # Patch the class definition so the model instance knows it's supported
        if not hasattr(type(model), '_supports_gradient_checkpointing'):
            type(model)._supports_gradient_checkpointing = True
        model.gradient_checkpointing = True
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        logging.info("Gradient checkpointing: FULLY ENABLED (Instance + Patch)")

    # ---- Optimizer ----
    audio_lr = args.audio_lr if getattr(args, "audio_lr", None) is not None else args.lr
    param_groups = [
        {"params": visual_params, "lr": args.lr},
        {"params": audio_params, "lr": audio_lr}
    ]
    logging.info(f"Optimizer param groups: visual_lr={args.lr}, audio_lr={audio_lr}")

    if getattr(args, "use_8bit_optim", False):
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(param_groups, weight_decay=args.weight_decay)
            logging.info("Using 8-bit AdamW optimizer from bitsandbytes.")
        except ImportError:
            logging.warning("bitsandbytes not installed. Falling back to regular AdamW.")
            optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # ---- LR Scheduler ----
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.lr * 0.1
    )

    tb_log_dir = args.tensorboard_dir or os.path.join(args.output_dir, "tensorboard")
    writer: Optional[Any] = None
    if args.tensorboard:
        os.makedirs(tb_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_log_dir)
        logging.info(f"TensorBoard log dir: {tb_log_dir}")
        writer.add_text("hparams", json.dumps(_serialize_args(args), indent=2, ensure_ascii=False), 0)

    current_step = 0
    epoch = 0
    if args.resume_from:
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f"--resume_from not found: {args.resume_from}")
        current_step, epoch, saved_args = load_training_checkpoint(
            args.resume_from,
            model,
            optimizer,
            scheduler,
            strict=True,
            expected_args=args,
        )
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "resume_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"resumed_from": args.resume_from, "step": current_step, "epoch": epoch}, f, indent=2)
        if saved_args:
            for k in ("lora_rank", "lora_alpha", "frame_num", "quant"):
                if k in saved_args and getattr(args, k, None) != saved_args.get(k):
                    logging.warning(
                        f"Arg {k!r} differs from checkpoint: current={getattr(args, k)!r} saved={saved_args.get(k)!r}"
                    )
        # [NEW] Detect if user explicitly requested to override the resumed learning rate
        _audio_lr_cli = audio_lr
        
        if getattr(args, "override_lr", False):
            if len(optimizer.param_groups) > 0:
                optimizer.param_groups[0]['lr'] = args.lr
                optimizer.param_groups[0]['initial_lr'] = args.lr
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = _audio_lr_cli
                optimizer.param_groups[1]['initial_lr'] = _audio_lr_cli
            
            # Since we jump the LR back up, we generate a fresh decay curve for the REMAINING steps
            remaining_steps = max(1, args.max_steps - current_step)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining_steps, eta_min=args.lr * 0.1
            )
            logging.info(f"override_lr=True: Overridden checkpoint LR with new CLI args: lr={args.lr}, audio_lr={_audio_lr_cli}")
        else:
            logging.info("Resuming with checkpoint's decayed learning rate.")

    # ---- Dataset ----
    dataset = InfiniteTalkDataset(
        data_dir=args.data_dir,
        frame_num=args.frame_num,
        audio_window=cfg.get('audio_window', 5) if hasattr(cfg, 'get') else 5,
        ref_neighbor_frames=args.ref_neighbor_frames,
        reference_mode=args.reference_mode,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Batch size 1 for VRAM
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ---- Training constants ----
    num_timesteps = 1000
    vae_stride = (4, 8, 8)
    patch_size = (1, 2, 2)
    # One fixed logit-normal distribution throughout training, without schedule shift.
    logging.info("Training timesteps: logit-normal, mean=0, std=1, no shift or stage switching")

    # ---- Training ----
    logging.info(f"Starting LoRA training for {args.max_steps} steps...")
    logging.info(f"Sampling first-clip probability: {args.first_clip_prob:.2f}")
    # Keep the entire base model frozen and in eval mode (safeguards BatchNorm, Dropout, etc.)
    model.eval()
    # Explicitly set only the trainable sub-modules to train mode
    for name, module in model.named_modules():
        if any(p.requires_grad for p in module.parameters(recurse=False)):
            module.train()
            
    # ---- Cache CLIP visual and text embeddings ----
    # Cache fixed metadata images only; adjacent and first-clip references vary.
    cache_budget = int(args.cpu_cache_gb * 2**30 / 4)
    clip_cache = TensorLRUCache(cache_budget)
    text_cache = TensorLRUCache(cache_budget)
    y_cond_cache = TensorLRUCache(cache_budget)
    latent_cache = TensorLRUCache(cache_budget)

    def get_clip_features(ref_frame_tensor: torch.Tensor, ref_image_name: str = None) -> torch.Tensor:
        """Get CLIP visual features, using cache if available."""
        if ref_image_name and ref_image_name in clip_cache:
            return clip_cache[ref_image_name].to(device).to(torch.bfloat16)
        # Compute and cache
        with torch.no_grad():
            ref_for_clip_cpu = ref_frame_tensor.unsqueeze(0).unsqueeze(2).to('cpu')
            clip_fea = clip_model.visual(ref_for_clip_cpu, comfy_crop=args.clip_crop).to(device).to(torch.bfloat16)
        if ref_image_name:
            clip_cache[ref_image_name] = clip_fea.cpu()  # Cache on CPU to save VRAM
        return clip_fea

    def get_text_features(prompt_str: str) -> List[torch.Tensor]:
        """Get text encoder output, using cache if available."""
        if prompt_str in text_cache:
            return [t.to(device) for t in text_cache[prompt_str]]
        # Compute and cache
        with torch.no_grad():
            context_list = [t.to(device) for t in text_encoder([prompt_str], torch.device('cpu'))]
        text_cache[prompt_str] = [t.cpu() for t in context_list]  # Cache on CPU
        return context_list

    def encode_video(video, key):
        if key in latent_cache:
            return latent_cache[key].to(device)
        vae.to(device)
        latent = vae.encode([video])[0]
        latent_cache[key] = latent.detach().cpu()
        return latent

    def encode_reference(ref_frame):
        # Function scope releases large padded pixel tensors before DiT runs.
        vae.to(device)
        padded = torch.zeros(3, args.frame_num, *ref_frame.shape[1:], device=device)
        padded[:, 0] = ref_frame
        latent = vae.encode([padded])[0]
        mask = torch.zeros(4, latent.shape[1], *latent.shape[2:], device=device)
        mask[:, 0] = 1
        return torch.cat([mask, latent], dim=0).to(torch.bfloat16)

    progress_bar = tqdm(total=args.max_steps, initial=current_step, desc="Training steps")

    while current_step < args.max_steps:
        epoch += 1
        for batch in dataloader:
            if current_step >= args.max_steps:
                break

            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)

            video_full = batch['video_full'].to(device)[0]  # C, needed_frames, H, W
            ref_frame = batch['ref_frame'].to(device)[0]  # C, H, W
            audio_emb_full = batch['audio_emb_full'].to(device)[0]  # needed_frames, window, 12, 768
            prompt = batch['prompt']
            if isinstance(prompt, str):
                prompt_batch: List[str] = [prompt]
            else:
                prompt_batch = list(prompt)

            C, T_full, H, W = video_full.shape
            context_frames = 9

            # ---- Decision: First Clip vs Continuation Clip ----
            # Lower first-clip probability so continuation training dominates.
            is_first_clip = random.random() < args.first_clip_prob
            is_continuation = not is_first_clip
            if is_first_clip:
                # Inference clamps the first latent to the initial reference image.
                ref_frame = video_full[:, 0]
            target_frames = args.frame_num

            with torch.no_grad():
                if not is_continuation:
                    # First clip: no previous chunk, but the initial latent is clamped below.
                    with torch.no_grad():
                        x_1 = encode_video(video_full, batch['latent_cache_key'][0])
                    x_context = None
                    total_latents = x_1.shape[1]
                    # The same initial frame supplies CLIP, VAE conditioning and the clean prefix.
                    # The audio track matches 1:1 with the pixel frames of the target video.
                    audio_input = audio_emb_full[:target_frames].unsqueeze(0).to(torch.bfloat16)
                else:
                    # Continuation clip: Eq.(3) uses explicit temporal concatenation.
                    # Total sequence length MUST be identical to inference (where it predicts frame_num - context).
                    # This implies target_frames = full window - context_frames.
                    target_frames = args.frame_num - context_frames
                    with torch.no_grad():
                        # Encode all frames together to preserve temporal receptive field and avoid boundary artifacts
                        # from the VAE's 3D convolutions at the cut point.
                        x_combined = encode_video(video_full, batch['latent_cache_key'][0])
                        
                        # In pixel space, context_frames is 9. In the temporal latent space with stride 4:
                        # latent_length = int(1 + (pixel_length - 1) // 4)
                        # So 9 pixel frames = 3 latent frames.
                        context_latent_frames = int(1 + (context_frames - 1) // 4)
                        
                        x_context = x_combined[:, :context_latent_frames]
                        x_1 = x_combined[:, context_latent_frames:]
                    

                    total_latents = x_context.shape[1] + x_1.shape[1]
                    # Audio length must identically align with the exact sequence loaded (args.frame_num)
                    audio_input = audio_emb_full[:args.frame_num].unsqueeze(0).to(torch.bfloat16)

                C_lat, _, lat_h, lat_w = x_1.shape[0], x_1.shape[1], x_1.shape[2], x_1.shape[3]

            # Extract ref_image_name here so it's available for both y_cond and CLIP caches
            ref_image_name = batch.get('ref_image_name', None)
            if isinstance(ref_image_name, (list, tuple)):
                ref_image_name = ref_image_name[0]
            if not ref_image_name:  # empty string fallback
                ref_image_name = None
            if is_first_clip:
                ref_image_name = None  # This frame changes with the sampled window.

            with torch.no_grad():
                _y_cond_key = _reference_cache_key(ref_image_name, args.frame_num)
                if _y_cond_key is not None and _y_cond_key in y_cond_cache:
                    y_cond = y_cond_cache[_y_cond_key].to(device)
                else:
                    y_cond = encode_reference(ref_frame)
                    if _y_cond_key is not None:
                        y_cond_cache[_y_cond_key] = y_cond.cpu()

            if args.vae_cpu_offload:
                vae.to('cpu')

            # ---- CLIP and Text (Shared) ----
            # ---- CFG dropout (train-time) ----
            drop_text = False
            drop_audio = False
            drop_clip = False
            drop_ref = False
            r_cfg = random.random()
            if r_cfg < args.cfg_drop_both_prob:
                drop_text = True
                drop_audio = True
            elif r_cfg < args.cfg_drop_both_prob + args.cfg_drop_text_prob:
                drop_text = True
            elif r_cfg < args.cfg_drop_both_prob + args.cfg_drop_text_prob + args.cfg_drop_audio_prob:
                drop_audio = True

            # CLIP visual dropout: forces model to rely on LoRA for identity
            if random.random() < args.cfg_drop_clip_prob:
                drop_clip = True

            # Reference frame VAE dropout: forces model to learn identity from LoRA weights
            if random.random() < args.cfg_drop_ref_prob:
                drop_ref = True

            if drop_audio:
                audio_input = torch.zeros_like(audio_input)

            with torch.no_grad():
                # Use cached CLIP features (computed once per unique ref image)
                clip_fea = get_clip_features(ref_frame, ref_image_name)

                # Drop CLIP visual features: forces LoRA to carry identity
                if drop_clip:
                    clip_fea = torch.zeros_like(clip_fea)

                # Use cached text features (computed once per unique prompt)
                prompt_list = prompt_batch if not drop_text else [""] * len(prompt_batch)
                context_list = get_text_features(prompt_list[0]) if len(set(prompt_list)) == 1 \
                    else [t.to(device) for t in text_encoder(prompt_list, torch.device('cpu'))]

                human_mask = torch.ones([lat_h, lat_w], device=device).unsqueeze(0).repeat(3, 1, 1).float()

            del video_full, ref_frame, audio_emb_full

            # Explicit absent-reference condition: no indicated frames, zero latents.
            # Opt-in experiment; disabled by default to preserve inference conditioning.
            if drop_ref:
                y_cond = torch.zeros_like(y_cond)

            # ---- Flow matching interpolation ----
            x_0 = torch.randn_like(x_1)
            t_frac = _sample_training_timestep(device)
            timestep = t_frac * num_timesteps

            x_input, target_full, loss_mask = _flow_inputs(x_1, x_0, t_frac, x_context)

            # Align audio frame count to latent temporal length (WanModel rearrange needs (F-1) % vae_scale == 0).
            vae_t = int(getattr(model, "vae_scale", 4))
            t_lat = x_input.shape[1]
            f_req = _pixel_frames_for_latent_len(t_lat, vae_t)
            if audio_input.shape[1] != f_req:
                if args.debug_assert_shapes or current_step == 0:
                    logging.info(
                        f"Aligning audio frames {audio_input.shape[1]} -> {f_req} "
                        f"(latent T={t_lat}, vae_scale={vae_t})"
                    )
                audio_input = _align_audio_frames_to_latent(audio_input, f_req)
            if args.debug_assert_shapes:
                assert audio_input.shape[1] == f_req, (audio_input.shape[1], f_req, t_lat, vae_t)
                assert x_input.shape[1] == t_lat
                if is_continuation:
                    assert x_context is not None
                    assert t_lat == x_context.shape[1] + x_1.shape[1]

            # ---- Forward pass ----
            T_total = x_input.shape[1]
            # Compute seq_len to match exactly how WanModel patchifies the latent:
            # patchify does (lat_h // p) * (lat_w // p) per frame, NOT lat_h * lat_w // p^2.
            # These differ when lat_h or lat_w is not divisible by p (off-by-one via floor).
            max_seq_len = T_total * (lat_h // patch_size[1]) * (lat_w // patch_size[2])

            # Non-reentrant checkpointing tracks adapter gradients even when
            # pixel/text conditions are frozen; no extra condition gradients.

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=args.use_amp):
                pred = model(
                    x=[x_input], t=timestep, context=context_list, seq_len=max_seq_len,
                    clip_fea=clip_fea, y=[y_cond], audio=audio_input, ref_target_masks=human_mask,
                )[0]
                loss = F.mse_loss(pred.float() * loss_mask, target_full.float() * loss_mask) / loss_mask.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at step {current_step}; optimizer was not updated')

            # ---- Backward ----
            loss.backward()
            grad_norm_val: Optional[torch.Tensor] = None
            if args.max_grad_norm > 0:
                grad_norm_val = torch.nn.utils.clip_grad_norm_(visual_params + audio_params, args.max_grad_norm,
                                                               error_if_nonfinite=True)
            elif any(p.grad is not None and not torch.isfinite(p.grad).all() for p in visual_params + audio_params):
                raise FloatingPointError('Non-finite gradients; optimizer was not updated')

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            current_step = cast(int, current_step + 1)
            progress_bar.update(1)
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}", 
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                epoch=epoch
            )

            # ---- TensorBoard ----
            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), current_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], current_step)
                writer.add_scalar("train/epoch", float(epoch), current_step)
                writer.add_scalar("train/is_continuation", 1.0 if is_continuation else 0.0, current_step)
                writer.add_scalar("train/cfg_drop_text", 1.0 if drop_text else 0.0, current_step)
                writer.add_scalar("train/cfg_drop_audio", 1.0 if drop_audio else 0.0, current_step)
                writer.add_scalar("train/cfg_drop_clip", 1.0 if drop_clip else 0.0, current_step)
                writer.add_scalar("train/cfg_drop_ref", 1.0 if drop_ref else 0.0, current_step)
                writer.add_scalar("train/timestep_frac", float(t_frac.item()), current_step)
                if grad_norm_val is not None:
                    writer.add_scalar("train/grad_norm", float(grad_norm_val), current_step)
                if current_step % args.log_every == 0:
                    writer.flush()

            # ---- Logging ----
            if current_step == 1 or current_step % args.log_every == 0:
                peak = torch.cuda.max_memory_allocated(device) / 2**30
                reserved = torch.cuda.max_memory_reserved(device) / 2**30
                cache_gb = sum(c.bytes for c in (clip_cache, text_cache, y_cond_cache, latent_cache)) / 2**30
                logging.info('Memory: peak allocated %.2f GiB, peak reserved %.2f GiB; CPU tensor cache %.2f GiB',
                             peak, reserved, cache_gb)
                if writer is not None:
                    writer.add_scalar('memory/peak_allocated_gib', peak, current_step)
                    writer.add_scalar('memory/peak_reserved_gib', reserved, current_step)

            # ---- Save checkpoint ----
            if args.save_every > 0 and current_step % args.save_every == 0:
                os.makedirs(args.output_dir, exist_ok=True)
                adapter_path, inference_lora_path = save_training_checkpoint(
                    args.output_dir,
                    current_step,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    args,
                    suffix=str(current_step),
                )
                logging.info(f"Saved checkpoint adapter: {adapter_path}")
                if inference_lora_path:
                    logging.info(f"Saved inference LoRA: {inference_lora_path}")

            # Cleanup
            del x_1, x_0, x_input, target_full, loss_mask, pred, loss
            if is_continuation and x_context is not None:
                del x_context, x_combined
            del y_cond, clip_fea, context_list, audio_input, human_mask

    progress_bar.close()
    
    # ---- Final save ----
    os.makedirs(args.output_dir, exist_ok=True)
    adapter_path, inference_lora_path = save_training_checkpoint(
        args.output_dir,
        current_step,
        epoch,
        model,
        optimizer,
        scheduler,
        args,
        suffix="final",
    )
    logging.info(f"Training complete! Final adapter: {adapter_path}")
    if inference_lora_path:
        logging.info(f"Final inference LoRA: {inference_lora_path}")
    logging.info(f"Total trainable parameters: {total_trainable:,}")
    if inference_lora_path:
        logging.info(f"Use with: --lora_dir {inference_lora_path} --lora_scale 1.0")
    logging.info(f"Resume with: --resume_from {os.path.dirname(adapter_path)}")

    if writer is not None:
        writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="InfiniteTalk LoRA Fine-tuning (Single Person)")
    parser.add_argument('--clip_crop', choices=['center', 'disabled'], default='center',
                        help='Match WanVideoWrapper CLIP preprocessing (workflow default: center)')

    # Model paths
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Path to Wan2.1-I2V-14B checkpoint directory")
    parser.add_argument("--infinitetalk_dir", type=str, default=None,
                        help="Original audio weights, only for loading unquantized Wan shards")
    parser.add_argument('--fp8_checkpoint', type=str, default=None,
                        help='Official merged single FP8 .safetensors; matching .json must be alongside it')
    parser.add_argument("--quant", type=str, default=None, choices=['int8', 'fp8', None],
                        help="Base weights: fp8 or omit for BF16. int8 is rejected (not implemented).")

    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Training data directory")
    parser.add_argument("--num_workers", type=int, default=2)

    # LoRA config
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                        help="LoRA alpha")

    # Training config
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate for visual/attention layers.")
    parser.add_argument("--audio_lr", type=float, default=None, help="Separate learning rate for audio layers. If None, uses --lr.")
    parser.add_argument("--train_audio", action=argparse.BooleanOptionalAction, default=True, help="Whether to fine-tune audio layers. Use --no-train_audio to skip audio layers.")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--frame_num", type=int, default=81,
                        help="Frames per training clip (4n+1, >9). Default 81 matches prepared caption windows.")
    parser.add_argument("--reference_mode", choices=['adjacent', 'fixed'], default='adjacent',
                        help="adjacent: sample neighboring video frames (M3); fixed: use metadata ref_image.")
    parser.add_argument(
        "--ref_neighbor_frames",
        type=int,
        default=25,
        help="Reference frame sampling window (in frames) around the current segment; used for adjacent-frame sampling.",
    )
    parser.add_argument("--use_8bit_optim", action=argparse.BooleanOptionalAction, default=True, help="Use bitsandbytes 8-bit optimizer to save VRAM on optimizer states")
    parser.add_argument("--blocks_to_swap", "--cpu_offload_blocks", dest='blocks_to_swap', type=int, default=0,
                        help="Stream frozen Linear weights from CPU in the last N blocks (0 to model depth); adapters stay on GPU")
    parser.add_argument('--activation_offload', action=argparse.BooleanOptionalAction, default=False,
                        help='Offload gradient-checkpoint boundary activations to CPU')
    parser.add_argument('--offload_pin_memory', action=argparse.BooleanOptionalAction, default=False,
                        help='Pin streamed weights and saved activations; increases locked physical RAM use')
    parser.add_argument('--vae_cpu_offload', action=argparse.BooleanOptionalAction, default=True,
                        help='Unload VAE to CPU before DiT forward/backward')
    parser.add_argument('--cpu_cache_gb', type=float, default=4.,
                        help='Total GiB limit for four CPU caches (latents, VAE reference, CLIP, text); 0 disables')
    parser.add_argument('--attention_backend', choices=['sdpa', 'auto'], default='sdpa',
                        help='Native training SDPA (Windows-friendly), or available FlashAttention kernels')
    parser.add_argument("--first_clip_prob", type=float, default=0.2,
                        help="Probability of sampling first-clip training branch. Continuation prob is 1-p.")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=True,
                        help="Use BF16 mixed precision")
    parser.add_argument("--debug_assert_shapes", action=argparse.BooleanOptionalAction, default=False,
                        help="Assert audio frame count matches latent T (WanModel audio rearrange).")
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True,
                        help="Log scalars to TensorBoard.")
    parser.add_argument("--tensorboard_dir", type=str, default=None,
                        help="TensorBoard log directory. Default: <output_dir>/tensorboard")
    parser.add_argument(
        "--cfg_drop_text_prob",
        type=float,
        default=0.1,
        help="Train-time CFG dropout: probability to drop text condition (set prompt to empty string).",
    )
    parser.add_argument(
        "--cfg_drop_audio_prob",
        type=float,
        default=0.1,
        help="Train-time CFG dropout: probability to drop audio condition (set audio embedding to zeros).",
    )
    parser.add_argument(
        "--cfg_drop_both_prob",
        type=float,
        default=0.05,
        help="Train-time CFG dropout: probability to drop both text and audio conditions.",
    )
    parser.add_argument(
        "--cfg_drop_clip_prob",
        type=float,
        default=0.0,
        help="Optional CLIP feature dropout; disabled by default.",
    )
    parser.add_argument(
        "--cfg_drop_ref_prob",
        type=float,
        default=0.0,
        help="Optional reference dropout: zero both mask and VAE latents; disabled by default.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume from a checkpoint directory (checkpoint-<step> with adapter_model.safetensors, optimizer.pt, etc) or legacy training_*.pt.",
    )

    parser.add_argument("--override_lr", action=argparse.BooleanOptionalAction, default=False,
                        help="Force override the resumed optimizer's learning rate with the CLI values.")
    # Output
    parser.add_argument("--output_dir", type=str, default="output/lora")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=200,
                        help="Save LoRA + training .pt every N steps; 0 disables periodic save.")
    parser.add_argument("--device_id", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
