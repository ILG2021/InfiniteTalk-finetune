# InfiniteTalk 单人 LoRA 微调指南 (RTX 5090)

本指南专为在单张 RTX 5090 (32GB VRAM) 上对 InfiniteTalk 模型进行单人视频数据的 LoRA 微调而设计。

## 0. 环境与预训练
创建一个python 3.10虚拟环境
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P
huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base
huggingface-cli download MeiGen-AI/InfiniteTalk single/infinitetalk.safetensors --local-dir ./weights/InfiniteTalk
```


## 1. 原理与策略

由于 InfiniteTalk (14B) 的基础模型过大（BF16格式下约需 28GB 显存），无法直接在单张 32GB 显卡上进行梯度回传。因此我们采用以下并行微调策略：

1. **基础模型 FP8 量化**：加载基础权重后，使用 FP8 精度进行计算，极大降低显存占用。
2. **LoRA 微调策略**：
    - 我们采用 **纯 LoRA (Low-Rank Adaptation)** 策略。针对底座模型中新增的音频投影层 (`audio_proj`) 和所有 40 层 blocks 的音频交叉注意力层 (`audio_cross_attn`) 中的线性层进行低秩适配。
    - 底座权重保持冻结并量化，LoRA 参数以 FP32 训练并在导出时合并缩放系数 (scaling)，确保与推理侧脚本 (`wan_lora.py`) 完全匹配。

## 2. 数据准备

### 2.1 准备原始视频
在 `raw_videos` 目录下准备几段 5-60 秒的清晰、正脸无遮挡、带干净语音的 MP4 视频。

```text
raw_videos/
├── person_clip01.mp4  
├── person_clip02.mp4
└── ...
```

### 2.2 提取音频特征 (Wav2Vec2)
需使用 `prepare_data.py` 脚本提前提取视频的 `wav2vec2` 特征，以此减少训练时的前向开销：

```bash
python prepare_data.py \
    --video_dir ./raw_videos \
    --output_dir ./training_data \
    --wav2vec_model weights/chinese-wav2vec2-base \
    --prompt "A person is talking." \
    --device cuda:0 \
    --target_h 832 \
    --target_w 540
```
> **输出规范**：该脚本会在 `./training_data` 目录下生成统一的 `videos` 和处理后的 `audio_embs` 及 `metadata.json`，这就是下一步的训练数据集。

## 3. 启动训练

执行微调脚本。请确保路径与你本地的 Wan2.1 Checkpoint 及 InfiniteTalk 权重相匹配。

```bash
python train_lora.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --data_dir ./training_data \
    --quant fp8 \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lr 5e-5 \
    --max_steps 1000 \
    --frame_num 49 \
    --target_h 832 \
    --target_w 540 \
    --ref_neighbor_frames 25 \
    --cfg_drop_text_prob 0.1 \
    --cfg_drop_audio_prob 0.1 \
    --cfg_drop_both_prob 0.05 \
    --debug_assert_shapes \
    --gradient_checkpointing \
    --use_amp \
    --output_dir output/my_lora \
    --save_every 200 \
    --log_every 10 \
    --debug_assert_shapes
```

### 参数建议：
- `--frame_num`: **强烈建议设为 33 或 49**。
    - **33 (非常稳定)**: 约占 20-22GB 显存，适合后台运行。
    - **49 (推荐/画质优先)**: 约占 25-27GB 显存，涵盖近 2 秒视频，能学到更好的连贯性。
    - 注：帧数必须为 `4n+1`，如 17, 33, 49, 65...
- `--target_h / --target_w`: 训练输入会被缩放到固定分辨率（默认为竖屏 `832x480`，即 `H=832, W=480`）。建议与你用 `prepare_data.py` 预处理的分辨率保持一致。
- `--ref_neighbor_frames`: 参考帧采样窗口（单位：帧）。训练时参考帧会从当前片段左右相邻区域采样（论文 M3 思路），避免控制过强/过弱。默认 25（约 1 秒）。
- `--cfg_drop_text_prob / --cfg_drop_audio_prob / --cfg_drop_both_prob`: CFG dropout 概率，用于让推理侧 Text/Audio CFG 更稳定、可控。三者之和必须小于等于 1。
- `--debug_assert_shapes`: 首次开训建议打开，用于强制检查音频长度与 latent 时间轴对齐，避免 silent shape mismatch。
- `--lora_rank / --lora_alpha`: **推荐为 16**。如果显存充裕可以提升至 `32` 以增强人物面部还原度。
- `--quant fp8`: **必须开启**。使用 FP8 Monkey Patch 加速并极大压缩 base 模型显存，是 RTX 5090 运行 14B 模型的关键。
- `--lr`: **推荐 5e-5 ~ 1e-4**。与全参微调不同，LoRA 容许适当使用偏大一点的学习率进行初期收敛。 

## 4. 显存监控与排障

在执行过程中：
1. **显存稳定区间应为 20G ~ 26G** (视 `--frame_num` 而定)。
2. 训练初期终端会打印 Trainable 层，应看到 `lora_down.weight` 和 `lora_up.weight`。
3. 如果发生 OOM：尝试设置环境变量 `set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，或将 `--frame_num` 降至 33。

### 断点续训

训练会按 Transformers/PEFT 风格保存 checkpoint：每个断点是一个目录：

```
<output_dir>/
  checkpoint-200/
    adapter_model.safetensors    # 训练用全参 adapter，支持断点续传
    optimizer.pt
    scheduler.pt
    scaler.pt
    rng_state.pth
    trainer_state.json
  checkpoint-final/
    ...
    lora_for_inference.safetensors   # 仅 final 导出，用于推理 (wan_lora.py)
  checkpoint_latest.json
```

其中：
- `adapter_model.safetensors`：**训练用 adapter 权重**（仅保存 `requires_grad=True` 的参数：LoRA + 小归一化层等）。续训时会写回模型。
- `optimizer.pt / scheduler.pt / scaler.pt / rng_state.pth`：训练状态。
- `original_norms.pt`：用于继续计算 `.diff`。
- `lora_for_inference.safetensors`：推理用 LoRA（wan_lora.py 兼容格式）。为了避免占用空间，脚本默认只在 `checkpoint-final` 导出这一份。

如果需要断点续训：

```bash
python train_lora.py \
    ... \
    --resume_from output/my_lora/checkpoint-200
```

也可以直接读取根目录的 `checkpoint_latest.json`，找到最新的 `checkpoint_dir` 继续训练。

## 5. 使用最终模型推理

微调结束后，使用 `checkpoint-final/lora_for_inference.safetensors` 进行推理。

```bash
python generate_infinitetalk.py \
    --ckpt_dir /path/to/Wan2.1-I2V-14B-480P \
    --infinitetalk_dir /path/to/infinitetalk.safetensors \
    --lora_dir output/my_lora/checkpoint-final/lora_for_inference.safetensors \
    --lora_scale 1.0 \
    --size infinitetalk-480 \
    --input_json input.json
```

> **提示**：可以适当调节 `--lora_scale`（例如 0.8），来微调你的 LoRA 权重在生成画面中的影响力度。
