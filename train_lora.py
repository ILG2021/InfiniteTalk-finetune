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
        result = self.original_linear(x)
        lora_out = self.lora_up(self.lora_down(x)) * self.scaling
        return result + lora_out


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
        target_modules = [
            'audio_cross_attn.q_linear',
            'audio_cross_attn.kv_linear',
            'audio_cross_attn.proj',
            'audio_proj.proj1',  # will also match proj1_vf, see exact match below
            'audio_proj.proj2',
            'audio_proj.proj3',
        ]

    lora_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or type(module).__name__ == "QLinear":
            # Check if the full dotted name ends with one of the target patterns
            if any(name == target or name.endswith('.' + target) for target in target_modules):
                lora_modules[name] = module
            # Special case: audio_proj.proj1 should also match audio_proj.proj1_vf
            elif 'audio_proj.proj1_vf' in name and 'audio_proj.proj1' in target_modules:
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


def unfreeze_small_norms(model):
    """
    Unfreeze small normalization layers for direct fine-tuning.
    These are too small for LoRA, so we train them directly and save as diff.

    Targets:
        - blocks.{i}.norm_x (WanLayerNorm, 10K params/block × 40 = 410K)
        - audio_proj.norm   (LayerNorm, 1.5K params)
    """
    original_weights = {}
    count: int = 0
    for name, param in model.named_parameters():
        if 'norm_x.' in name or name.startswith('audio_proj.norm.'):
            original_weights[name] = param.data.clone()
            param.requires_grad = True
            count = cast(int, count + 1)
    logging.info(f"Unfroze {count} norm parameters for direct fine-tuning")
    return original_weights


def extract_lora_state_dict(model, original_norms=None):
    """
    Extract LoRA weights + norm_x diff in a format compatible with wan_lora.py.

    Output key format:
        diffusion_model.{path}.lora_down.weight   (LoRA A matrix)
        diffusion_model.{path}.lora_up.weight     (LoRA B matrix)
        diffusion_model.{path}.diff                (direct weight diff)
        diffusion_model.{path}.diff_b              (direct bias diff)
    """
    state_dict = {}

    # Extract LoRA weights
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            prefix = f"diffusion_model.{name}"
            state_dict[f"{prefix}.lora_down.weight"] = module.lora_down.weight.data.clone().cpu()
            # Bake training-time LoRA scaling (alpha/rank) into lora_up so
            # merge-time delta = (B_scaled @ A) * lora_scale remains equivalent.
            state_dict[f"{prefix}.lora_up.weight"] = (module.lora_up.weight.data * module.scaling).clone().cpu()

    # Extract norm_x diff weights (current - original)
    if original_norms is not None:
        for name, param in model.named_parameters():
            if 'norm_x.' in name or name.startswith('audio_proj.norm.'):
                diff = (param.data - original_norms[name].to(param.device)).cpu()
                lora_key = f"diffusion_model.{name}"
                # wan_lora.py expects .diff for weight, .diff_b for bias
                if name.endswith('.weight'):
                    lora_key = lora_key.replace('.weight', '.diff')
                elif name.endswith('.bias'):
                    lora_key = lora_key.replace('.bias', '.diff_b')
                state_dict[lora_key] = diff

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


def _serialize_args(args: argparse.Namespace) -> Dict[str, Any]:
    """JSON-friendly hyperparameter dict for checkpoints."""
    out: Dict[str, Any] = {}
    for k, v in vars(args).items():
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
        original_norms: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Any,
        args: argparse.Namespace,
        suffix: Optional[str] = None,
        write_latest: bool = True,
) -> Tuple[str, str]:
    """
    Saves a Transformers/PEFT-like checkpoint directory containing:
    - adapter_model.safetensors (trainable parameters only: LoRA + small norm layers)
    - optimizer.pt / scheduler.pt / scaler.pt
    - rng_state.pth
    - original_norms.pt
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
    torch.save(scaler.state_dict(), os.path.join(ckpt_dir, "scaler.pt"))
    torch.save(_get_rng_state(), os.path.join(ckpt_dir, "rng_state.pth"))
    torch.save({k: v.cpu().clone() for k, v in original_norms.items()}, os.path.join(ckpt_dir, "original_norms.pt"))

    trainer_state = {
        "format_version": 2,
        "step": step,
        "epoch": epoch,
        "args": _serialize_args(args),
    }
    _save_json(os.path.join(ckpt_dir, "trainer_state.json"), trainer_state)

    # 3) Optional inference LoRA export (wan_lora-compatible) to avoid duplication.
    inference_lora_path = ""
    if suffix is not None:
        lora_sd = extract_lora_state_dict(model, original_norms)
        inference_lora_path = os.path.join(ckpt_dir, "lora_for_inference.safetensors")
        save_file(lora_sd, inference_lora_path)

    # 4) Latest pointer (Windows-friendly: a tiny json file in output root)
    if write_latest:
        latest_meta = {
            "checkpoint_dir": ckpt_dir,
            "step": step,
            "epoch": epoch,
        }
        _save_json(os.path.join(output_dir, "checkpoint_latest.json"), latest_meta)

    return adapter_path, inference_lora_path


def load_training_checkpoint(
        path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Any,
        strict: bool = True,
) -> Tuple[int, int, Dict[str, torch.Tensor], Dict[str, Any]]:
    """Loads adapter weights + optimizer/scheduler/scaler/RNG. Supports new checkpoint dirs and legacy .pt."""
    if _is_legacy_training_pt(path):
        ckpt = torch.load(path, map_location='cpu')
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

        original_norms = {k: v.clone() for k, v in ckpt['original_norms'].items()}
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        _set_rng_state(ckpt['rng'])

        step = int(ckpt['step'])
        epoch = int(ckpt.get('epoch', 0))
        saved_args = ckpt.get('args', {})
        logging.info(f"Resumed (legacy) from {path}: step={step}, epoch={epoch}, trainable_tensors={loaded}")
        return step, epoch, original_norms, saved_args

    ckpt_dir = _resolve_checkpoint_dir(path)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"--resume_from not found: {path}")

    trainer_state_path = os.path.join(ckpt_dir, "trainer_state.json")
    if not os.path.isfile(trainer_state_path):
        raise FileNotFoundError(f"trainer_state.json not found in checkpoint dir: {ckpt_dir}")
    trainer_state = _load_json(trainer_state_path)
    if int(trainer_state.get("format_version", 0)) != 2:
        logging.warning(f"Checkpoint format_version={trainer_state.get('format_version')!r}; expected 2")

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
    optimizer.load_state_dict(torch.load(os.path.join(ckpt_dir, "optimizer.pt"), map_location="cpu"))
    scheduler.load_state_dict(torch.load(os.path.join(ckpt_dir, "scheduler.pt"), map_location="cpu"))
    scaler.load_state_dict(torch.load(os.path.join(ckpt_dir, "scaler.pt"), map_location="cpu"))
    _set_rng_state(torch.load(os.path.join(ckpt_dir, "rng_state.pth"), map_location="cpu"))

    original_norms = torch.load(os.path.join(ckpt_dir, "original_norms.pt"), map_location="cpu")
    step = int(trainer_state.get("step", 0))
    epoch = int(trainer_state.get("epoch", 0))
    saved_args = cast(Dict[str, Any], trainer_state.get("args", {}))
    logging.info(f"Resumed from {ckpt_dir}: step={step}, epoch={epoch}, trainable_tensors={loaded}")
    return step, epoch, original_norms, saved_args


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
            frame_num=17,
            target_size=(832, 480),
            audio_window=5,
            ref_neighbor_frames: int = 25,
    ):
        self.data_dir = data_dir
        self.frame_num = frame_num
        self.target_h, self.target_w = target_size
        self.audio_window = audio_window
        self.ref_neighbor_frames = ref_neighbor_frames

        # Load metadata
        metadata_path = os.path.join(data_dir, 'metadata.json')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        self.samples = self.metadata['samples']
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
        video_path = os.path.join(self.data_dir, 'videos', sample['video'])
        audio_emb_path = os.path.join(self.data_dir, 'audio_embs', sample['audio_emb'])
        prompt = sample.get('prompt', 'A person is talking.')

        # Load pre-computed audio embedding: [total_frames, 12, 768]
        full_audio_emb = torch.load(audio_emb_path, map_location='cpu')
        total_audio_frames = full_audio_emb.shape[0]

        # We need self.frame_num frames for the main training segment.
        # For continuation mode with Eq.(3) concat, context and target should be non-overlapping:
        # context = 9 frames (indices 0..8), target starts at frame 9.
        context_frames = 9
        needed_frames = context_frames + self.frame_num
        max_start = max(0, total_audio_frames - needed_frames - 5)
        start_frame = random.randint(0, max_start) if max_start > 0 else 0

        # Load video frames (up to needed_frames)
        # Also get total video frame count for correct reference frame boundary clamping.
        from decord import VideoReader, cpu as decord_cpu
        _vr = VideoReader(video_path, ctx=decord_cpu(0))
        total_video_frames = len(_vr)
        del _vr
        video_full = self._load_video_frames(video_path, start_frame, needed_frames)
        # Resize to target size
        video_full = F.interpolate(
            video_full, size=(self.target_h, self.target_w),
            mode='bilinear', align_corners=False
        )  # T_full, C, H, W (needed_frames, 3, H, W)

        # Reference frame for identity: sample from a temporally adjacent region (paper M3).
        # Prefer a nearby-but-non-overlapping region to avoid overly strong copy behavior.
        seg_start = start_frame
        seg_end = start_frame + needed_frames - 1
        nb = max(0, int(self.ref_neighbor_frames))
        left_lo = max(0, seg_start - nb)
        left_hi = max(0, seg_start - 1)
        right_lo = min(total_video_frames - 1, seg_end + 1)
        right_hi = min(total_video_frames - 1, seg_end + nb)
        candidates: List[int] = []
        if left_lo <= left_hi:
            candidates.append(random.randint(left_lo, left_hi))
        if right_lo <= right_hi:
            candidates.append(random.randint(right_lo, right_hi))
        if candidates:
            ref_offset = random.choice(candidates)
        else:
            ref_offset = random.randint(0, total_video_frames - 1)
        ref_video = self._load_video_frames(video_path, ref_offset, 1)
        ref_frame = F.interpolate(
            ref_video, size=(self.target_h, self.target_w),
            mode='bilinear', align_corners=False
        )  # 1, C, H, W

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
    if (args.frame_num - 1) % 4 != 0:
        raise ValueError(f"--frame_num must satisfy 4n+1, got {args.frame_num}")
    if args.cfg_drop_text_prob < 0 or args.cfg_drop_audio_prob < 0 or args.cfg_drop_both_prob < 0:
        raise ValueError("CFG dropout probabilities must be non-negative")
    if args.cfg_drop_text_prob + args.cfg_drop_audio_prob + args.cfg_drop_both_prob > 1.0:
        raise ValueError(
            "Sum of --cfg_drop_text_prob, --cfg_drop_audio_prob, --cfg_drop_both_prob must be <= 1.0"
        )

    device = torch.device(f'cuda:{args.device_id}')
    torch.cuda.set_device(device)

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
        dit_path=None,
        infinitetalk_dir=args.infinitetalk_dir,
    )

    model = pipeline.model
    # Offload VAE/CLIP/T5 to CPU to save VRAM for the DiT
    vae = pipeline.vae
    vae.to('cpu')  # WanVAE.to() moves model + mean/std/scale tensors together
    clip_model = pipeline.clip
    clip_model.model.to('cpu').float()  # float16 not supported on CPU, cast to float32
    text_encoder = pipeline.text_encoder
    text_encoder.model.to('cpu')
    logging.info("Offloaded vae / clip / text_encoder to CPU")

    # ---- Quantize frozen base model to reduce VRAM ----
    if args.quant == 'int8':
        from optimum.quanto import quantize, freeze, qint8
        logging.info("Quantizing frozen base model to INT8 before applying LoRA...")
        quantize(model, weights=qint8)
        freeze(model)
        logging.info(f"Memory after model.to & freeze: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logging.info("INT8 quantization applied and base layers moved to device.")

    # ---- Freeze everything (re-verify after quantization) ----
    for param in model.parameters():
        param.requires_grad = False

    total_trainable = count_trainable(model)
    logging.info(f"2 Total trainable parameters: {total_trainable:,}")
    # ---- Apply LoRA to Linear layers ----
    target_modules = args.target_modules.split(',') if args.target_modules else None
    model = apply_lora_to_model(model, rank=args.lora_rank, alpha=args.lora_alpha,
                                target_modules=target_modules)

    total_trainable = count_trainable(model)
    logging.info(f"3 Total trainable parameters: {total_trainable:,}")
    # ---- Unfreeze small norms for direct training (too small for LoRA) ----
    original_norms = unfreeze_small_norms(model)

    total_trainable = count_trainable(model)
    logging.info(f"4 Total trainable parameters: {total_trainable:,}")

    torch.cuda.empty_cache()
    gc.collect()
    model = model.to(device)
    model.disable_teacache()
    # Move trainable params to float32 for training stability
    lora_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # [Fix] Only convert non-quantized FLOAT types (prevents crash on INT8 weights)
            if not str(param.data.dtype).startswith("torch.int") and param.data.dtype != torch.float32:
                try:
                    param.data = param.data.float()
                except:
                    pass
            lora_params.append(param)
            logging.info(f"  Trainable: {name} {param.shape} ({param.data.dtype})")

    total_trainable = sum(p.numel() for p in lora_params)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable params: {total_trainable:,} / {total_params:,} "
                 f"({100 * total_trainable / total_params:.4f}%)")

    # Enable gradient checkpointing (with Monkey Patch for WanModel)
    if args.gradient_checkpointing:
        # 1. Force diffusers to accept this model by patching the Class
        type(model)._supports_gradient_checkpointing = True

        enabled = False
        if hasattr(model, "enable_gradient_checkpointing"):
            try:
                model.enable_gradient_checkpointing()
                enabled = True
            except Exception as e:
                logging.warning(f"  Native enable_gradient_checkpointing failed: {e}")

        # 2. Direct attribute fallback (common for Wan/DiT structures)
        if not enabled and hasattr(model, "gradient_checkpointing"):
            setattr(model, "gradient_checkpointing", True)
            enabled = True

        logging.info(f"Gradient checkpointing: {'ENABLED (patched)' if enabled else 'FAILED TO ENABLE'}")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=args.weight_decay)

    # ---- LR Scheduler ----
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.lr * 0.1
    )

    scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)

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
        current_step, epoch, original_norms, saved_args = load_training_checkpoint(
            args.resume_from,
            model,
            optimizer,
            scheduler,
            scaler,
            strict=args.resume_strict,
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

    # ---- Dataset ----
    dataset = InfiniteTalkDataset(
        data_dir=args.data_dir,
        frame_num=args.frame_num,
        target_size=(args.target_h, args.target_w),
        audio_window=cfg.get('audio_window', 5) if hasattr(cfg, 'get') else 5,
        ref_neighbor_frames=args.ref_neighbor_frames,
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
    shift = 7.0  # for 480P

    # ---- Training ----
    logging.info(f"Starting LoRA training for {args.max_steps} steps...")
    logging.info(f"Sampling first-clip probability: {args.first_clip_prob:.2f}")
    # Keep the entire base model frozen and in eval mode (safeguards BatchNorm, Dropout, etc.)
    model.eval()
    # Explicitly set only the trainable sub-modules to train mode
    for name, module in model.named_modules():
        if any(p.requires_grad for p in module.parameters(recurse=False)):
            module.train()

    while current_step < args.max_steps:
        epoch += 1
        for batch in dataloader:
            if current_step >= args.max_steps:
                break

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
            target_frames = args.frame_num

            with torch.no_grad():
                if not is_continuation:
                    # First clip: no context
                    with torch.no_grad():
                        video_target_cpu = video_full[:, :target_frames].to('cpu')
                        ref_frame_cpu = ref_frame.unsqueeze(1).to('cpu')
                        
                        x_1 = vae.encode([video_target_cpu])[0].to(device)  # C_lat, T_target, lat_h, lat_w
                        x_0 = vae.encode([ref_frame_cpu])[0].to(device)    # C_lat, 1, lat_h, lat_w
                        
                    x_context = None
                    total_latents = x_0.shape[1] + x_1.shape[1]
                    audio_input = audio_emb_full[:total_latents].unsqueeze(0).to(torch.bfloat16)
                else:
                    # Continuation clip: Eq.(3) uses explicit temporal concatenation.
                    # context frames = 4*(tc-1)+1 (paper uses 9 frames when tc=3), target frames = frame_num
                    # Use non-overlapping concat as in Eq.(3): target starts after context.
                    # [VRAM Fix] VAE/CLIP are already on CPU. Move inputs to CPU, then results to GPU.
                    with torch.no_grad():
                        video_context_cpu = video_full[:, :context_frames].to('cpu')
                        video_target_cpu = video_full[:, context_frames: context_frames + target_frames].to('cpu')
                        
                        x_context = vae.encode([video_context_cpu])[0].to(device)  # C_lat, T_context, lat_h, lat_w
                        x_1 = vae.encode([video_target_cpu])[0].to(device)  # C_lat, T_target, lat_h, lat_w
                    
                    # [VRAM Maintenance]
                    torch.cuda.empty_cache()
                    gc.collect()

                    total_latents = x_context.shape[1] + x_1.shape[1]
                    # Audio length should align with concatenated z1 timeline.
                    audio_input = audio_emb_full[:context_frames + target_frames].unsqueeze(0).to(torch.bfloat16)

                C_lat, _, lat_h, lat_w = x_1.shape[0], x_1.shape[1], x_1.shape[2], x_1.shape[3]

                # Build reference condition z2 + mask m directly in latent space to match total_latents.
                with torch.no_grad():
                    ref_frame_cpu = ref_frame.unsqueeze(1).to('cpu')
                    ref_latent = vae.encode([ref_frame_cpu])[0].to(device)  # C_lat, 1, lat_h, lat_w
                
                y_latent = torch.zeros(C_lat, total_latents, lat_h, lat_w, device=device, dtype=ref_latent.dtype)
                y_latent[:, :1] = ref_latent
                msk = torch.zeros(4, total_latents, lat_h, lat_w, device=device, dtype=ref_latent.dtype)
                msk[:, :1] = 1
                y_cond = torch.cat([msk, y_latent], dim=0).to(torch.bfloat16)

            # ---- CLIP and Text (Shared) ----
            # ---- CFG dropout (train-time) ----
            drop_text = False
            drop_audio = False
            r_cfg = random.random()
            if r_cfg < args.cfg_drop_both_prob:
                drop_text = True
                drop_audio = True
            elif r_cfg < args.cfg_drop_both_prob + args.cfg_drop_text_prob:
                drop_text = True
            elif r_cfg < args.cfg_drop_both_prob + args.cfg_drop_text_prob + args.cfg_drop_audio_prob:
                drop_audio = True

            if drop_audio:
                audio_input = torch.zeros_like(audio_input)

            with torch.no_grad():
                # [VRAM Optimize] CLIP and Text encoding strictly on CPU
                ref_for_clip_cpu = ref_frame.unsqueeze(0).unsqueeze(2).to('cpu')
                # clip_model is already permanently offloaded to CPU
                clip_fea = clip_model.visual(ref_for_clip_cpu).to(device).to(torch.bfloat16)

                prompt_list = prompt_batch
                if drop_text:
                    prompt_list = [""] * len(prompt_list)
                
                # text_encoder is on CPU, just move results back
                context_list = [t.to(device) for t in text_encoder(prompt_list, torch.device('cpu'))]
                human_mask = torch.ones([lat_h, lat_w], device=device).unsqueeze(0).repeat(3, 1, 1).float()

            # ---- Flow matching interpolation ----
            x_0 = torch.randn_like(x_1)
            t_frac = torch.rand(1, device=device, dtype=x_1.dtype)
            # Shift T
            t_frac = shift * t_frac / (1 + (shift - 1) * t_frac)
            t_shifted = t_frac * num_timesteps

            x_t = cast(torch.Tensor, t_frac.view(1, 1, 1, 1) * x_0 + (1 - t_frac).view(1, 1, 1, 1) * x_1)
            target = x_1 - x_0

            if is_continuation:
                # Eq.(3): z1 = concat(x_context, x_t)
                # Note: although wan/multitalk.py contains an "add_noise" injection for motion frames,
                # it is immediately overwritten by a clean-prefix assignment in the same step.
                # We follow the effective open-source inference behavior here.
                assert x_context is not None
                x_input = torch.cat([x_context, x_t], dim=1)
                target_full = torch.cat([torch.zeros_like(x_context), target], dim=1)
                loss_mask = torch.cat([torch.zeros_like(x_context), torch.ones_like(target)], dim=1)
            else:
                x_input = x_t
                target_full = target
                loss_mask = torch.ones_like(target_full)

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
            optimizer.zero_grad()
            T_total = x_input.shape[1]
            max_seq_len = T_total * lat_h * lat_w // (patch_size[1] * patch_size[2])

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=args.use_amp):
                pred = model(
                    x=[x_input], t=t_shifted, context=context_list, seq_len=max_seq_len,
                    clip_fea=clip_fea, y=[y_cond], audio=audio_input, ref_target_masks=human_mask,
                )[0]
                loss = F.mse_loss(pred.float() * loss_mask, target_full.float() * loss_mask) / loss_mask.mean()

            # ---- Backward ----
            scaler.scale(loss).backward()
            grad_norm_val: Optional[torch.Tensor] = None
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                grad_norm_val = torch.nn.utils.clip_grad_norm_(lora_params, args.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            current_step = cast(int, current_step + 1)

            # ---- TensorBoard ----
            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), current_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], current_step)
                writer.add_scalar("train/epoch", float(epoch), current_step)
                writer.add_scalar("train/is_continuation", 1.0 if is_continuation else 0.0, current_step)
                writer.add_scalar("train/cfg_drop_text", 1.0 if drop_text else 0.0, current_step)
                writer.add_scalar("train/cfg_drop_audio", 1.0 if drop_audio else 0.0, current_step)
                writer.add_scalar("train/amp_scale", scaler.get_scale(), current_step)
                if grad_norm_val is not None:
                    writer.add_scalar("train/grad_norm", float(grad_norm_val), current_step)
                writer.flush()

            # ---- Logging ----
            if current_step % args.log_every == 0:
                lr = optimizer.param_groups[0]['lr']
                logging.info(
                    f"Step {current_step}/{args.max_steps} | "
                    f"Loss: {loss.item():.6f} | "
                    f"LR: {lr:.2e} | "
                    f"Epoch: {epoch}"
                )

            # ---- Save checkpoint ----
            if args.save_every > 0 and current_step % args.save_every == 0:
                os.makedirs(args.output_dir, exist_ok=True)
                adapter_path, inference_lora_path = save_training_checkpoint(
                    args.output_dir,
                    current_step,
                    epoch,
                    model,
                    original_norms,
                    optimizer,
                    scheduler,
                    scaler,
                    args,
                )
                logging.info(f"Saved checkpoint adapter: {adapter_path}")
                if inference_lora_path:
                    logging.info(f"Saved inference LoRA: {inference_lora_path}")

            # Cleanup
            del x_1, x_0, x_t, x_input, target, target_full, loss_mask, pred, loss
            if is_continuation and x_context is not None:
                del x_context
            torch.cuda.empty_cache()

    # ---- Final save ----
    os.makedirs(args.output_dir, exist_ok=True)
    adapter_path, inference_lora_path = save_training_checkpoint(
        args.output_dir,
        current_step,
        epoch,
        model,
        original_norms,
        optimizer,
        scheduler,
        scaler,
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

    # Model paths
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Path to Wan2.1-I2V-14B checkpoint directory")
    parser.add_argument("--infinitetalk_dir", type=str, required=True,
                        help="Path to infinitetalk.safetensors")
    parser.add_argument("--quant", type=str, default=None, choices=['int8', 'fp8', None],
                        help="Quantization for base model. Recommended: int8 for 5090")

    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Training data directory")
    parser.add_argument("--num_workers", type=int, default=2)

    # LoRA config
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                        help="LoRA alpha")
    parser.add_argument("--target_modules", type=str, default=None,
                        help="Comma-separated target module patterns. Default: audio_cross_attn layers")

    # Training config
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--frame_num", type=int, default=17,
                        help="Frames per training clip (4n+1). Use 17 for 5090.")
    parser.add_argument(
        "--ref_neighbor_frames",
        type=int,
        default=25,
        help="Reference frame sampling window (in frames) around the current segment; used for adjacent-frame sampling.",
    )
    parser.add_argument("--first_clip_prob", type=float, default=0.2,
                        help="Probability of sampling first-clip training branch. Continuation prob is 1-p.")
    parser.add_argument("--target_h", type=int, default=832,
                        help="Target height. Recommend 832 for vertical 480P bucket")
    parser.add_argument("--target_w", type=int, default=480,
                        help="Target width. Recommend 480 for vertical 480P bucket")
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
        "--resume_from",
        type=str,
        default=None,
        help="Resume from a checkpoint directory (checkpoint-<step> with adapter_model.safetensors, optimizer.pt, etc) or legacy training_*.pt.",
    )
    parser.add_argument("--resume_strict", action=argparse.BooleanOptionalAction, default=True,
                        help="Strict checkpoint key match (trainable param names must match).")

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
