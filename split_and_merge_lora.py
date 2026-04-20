"""
split_and_merge_lora.py
=======================
一键处理 InfiniteTalk LoRA 分割与合并：
  1. 从 lora_for_inference.safetensors 提取纯视觉 LoRA → 给 ComfyUI WanVideo Lora Select 用
  2. 将所有音频层 LoRA (audio_proj + audio_cross_attn) 烘焙进 InfiniteTalk 权重文件

用法：
  python split_and_merge_lora.py \\
      --lora        output/my_lora/checkpoint-final/lora_for_inference.safetensors \\
      --infinitetalk weights/InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp8.safetensors \\
      --out_visual  output/my_lora/wan_lora.safetensors \\
      --out_it      output/my_lora/infinitetalk_lora.safetensors \\
      --alpha       1.0
"""

import argparse
import sys
import torch
from safetensors.torch import load_file, save_file


def split_and_merge(lora_path, infinitetalk_path, out_visual_path, out_it_path, alpha):

    # ── 加载文件 ───────────────────────────────────────────────
    print(f"Loading LoRA          : {lora_path}")
    lora = load_file(lora_path)
    print(f"Loading InfiniteTalk  : {infinitetalk_path}")
    base = load_file(infinitetalk_path)
    print(f"  LoRA keys     : {len(lora)}")
    print(f"  IT base keys  : {len(base)}\n")

    # ── 分类 ──────────────────────────────────────────────────
    visual_lora = {}
    audio_pairs = {}          # base_name → (down_key, up_key)
    prefix = "diffusion_model."

    for key, tensor in lora.items():
        is_audio = "audio" in key

        if not is_audio:
            visual_lora[key] = tensor
        else:
            # 只收集 lora_down，配对 lora_up
            if key.endswith("lora_down.weight") and key.startswith(prefix):
                base_name = key[len(prefix):].replace("lora_down.weight", "weight")
                up_key = key.replace("lora_down.weight", "lora_up.weight")
                if up_key in lora:
                    audio_pairs[base_name] = (key, up_key)

    audio_proj_n = sum(1 for k in audio_pairs if "audio_proj" in k)
    audio_cross_n = sum(1 for k in audio_pairs if "audio_cross_attn" in k)
    print(f"Visual LoRA keys      : {len(visual_lora)}")
    print(f"Audio LoRA pairs      : {len(audio_pairs)}")
    print(f"  audio_proj          : {audio_proj_n}")
    print(f"  audio_cross_attn    : {audio_cross_n}\n")

    # ── Step 1: 保存视觉 LoRA ──────────────────────────────────
    print(f"[1/2] Saving visual LoRA → {out_visual_path}")
    save_file(visual_lora, out_visual_path)
    print(f"      Done. ({len(visual_lora)} keys)\n")

    # ── Step 2: 烘焙音频 LoRA 进 InfiniteTalk 权重 ────────────
    print(f"[2/2] Merging audio LoRA into InfiniteTalk weights (alpha={alpha})...")
    merged = 0
    missing = []

    for base_name, (down_key, up_key) in audio_pairs.items():
        if base_name not in base:
            missing.append(base_name)
            continue

        lora_down = lora[down_key].float()
        lora_up   = lora[up_key].float()
        rank = lora_down.shape[0]

        delta = (alpha / rank) * torch.matmul(lora_up, lora_down)
        orig  = base[base_name].float()
        base[base_name] = (orig + delta).to(base[base_name].dtype)

        print(f"  [OK] {base_name}  (rank={rank}, Δnorm={delta.norm().item():.5f})")
        merged += 1

    if missing:
        print(f"\n  WARNING: {len(missing)} keys not found in InfiniteTalk base weights:")
        for k in missing:
            print(f"    [MISS] {k}")

    print(f"\n      Saving merged InfiniteTalk → {out_it_path}")
    save_file(base, out_it_path)
    print(f"      Done. ({merged} audio layers merged)\n")

    # ── 总结 ─────────────────────────────────────────────────
    print("=" * 60)
    print("完成！使用方法：")
    print(f"  ComfyUI WanVideo Lora Select  ← {out_visual_path}")
    print(f"  Multi/InfiniteTalk Model Loader ← {out_it_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Split visual LoRA and merge audio LoRA into InfiniteTalk weights")
    parser.add_argument("--lora",         required=True, help="Path to lora_for_inference.safetensors")
    parser.add_argument("--infinitetalk", required=True, help="Path to the InfiniteTalk base weights .safetensors")
    parser.add_argument("--out_visual",   required=True, help="Output path for visual-only LoRA (for ComfyUI)")
    parser.add_argument("--out_it",       required=True, help="Output path for merged InfiniteTalk weights")
    parser.add_argument("--alpha", type=float, default=1.0, help="Audio LoRA merge strength (default: 1.0)")
    args = parser.parse_args()

    split_and_merge(
        lora_path=args.lora,
        infinitetalk_path=args.infinitetalk,
        out_visual_path=args.out_visual,
        out_it_path=args.out_it,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
