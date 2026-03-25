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

1. **基础模型 INT8 量化**：加载基础权重后，使用 `optimum.quanto` 将非训练的 `Linear` 冻结并量化为 INT8。这样底座模型仅占约 14GB 显存。
2. **混合训练策略**：
    - **策略 A (大体量投影层) → LoRA 微调**：针对 `audio_proj` 中的线性映射层 (如 `proj1`, `proj1_vf`, `proj2` 等) 和所有 40 层 blocks 的 `audio_cross_attn` 相关的 `q_linear`, `kv_linear`, `proj` 投影层采用 LoRA 进行低秩微调，维持极少的显存开销。
    - **策略 B (轻量归一化层) → 全参差异微调**：由于 `LayerNorm` (如 `norm_x` 和 `audio_proj.norm`) 参数量极小，挂载 LoRA 不具有性价比，且往往直接影响训练的收敛性，这些层将被完全解冻计算梯度，并最终以 `.diff` weight 和 bias 差值的形式保存。

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
    --quant int8 \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lr 5e-5 \
    --max_steps 1000 \
    --frame_num 17 \
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
- `--frame_num`: **强烈建议设为 17**。帧数必须为 `4n+1`（因 VAE 的时间轴压缩比为 4）。训练脚本会对 `4n+1` 进行强制校验，不满足会直接报错。
- `--target_h / --target_w`: 训练输入会被缩放到固定分辨率（默认为竖屏 `832x480`，即 `H=832, W=480`）。建议与你用 `prepare_data.py` 预处理的分辨率保持一致。
- `--ref_neighbor_frames`: 参考帧采样窗口（单位：帧）。训练时参考帧会从当前片段左右相邻区域采样（论文 M3 思路），避免控制过强/过弱。默认 25（约 1 秒）。
- `--cfg_drop_text_prob / --cfg_drop_audio_prob / --cfg_drop_both_prob`: CFG dropout 概率，用于让推理侧 Text/Audio CFG 更稳定、可控。三者之和必须小于等于 1。
- `--debug_assert_shapes`: 首次开训建议打开，用于强制检查音频长度与 latent 时间轴对齐，避免 silent shape mismatch。
- `--lora_rank / --lora_alpha`: **推荐为 16**。如果发现微调无法收敛或希望进一步加强人物关联特征，且监控到当前占用仍有余量，可以提升至 `32`。
- `--quant int8`: 开启基础模型 INT8 量化，必须激活，否则显存溢出。
- `--lr`: **推荐 5e-5 ~ 1e-4**。与全参微调不同，LoRA 容许适当使用偏大一点的学习率进行初期收敛。 

## 4. 显存监控与排障

在执行过程中：
1. **显存稳定区间应为 22G ~ 26G**，如果是多卡机器需要使用 `CUDA_VISIBLE_DEVICES=1` 等环境变量。
2. 训练初期终端会打印所有的 Trainable 层名称信息，请特别关注是否存在以 `lora_down.weight`, `lora_up.weight` 为主的日志，并确认 `norm_x.weight` 等属于正常显式微调追踪队列。
3. 如果发生 OOM：优先检查 `--frame_num` 是否依然是 17，其次尝试调小 `--lora_rank` (如 8)。

### 断点续训

训练会按 Transformers/PEFT 风格保存 checkpoint：每个断点是一个目录：

```
<output_dir>/
  checkpoint-200/
    adapter_model.safetensors
    optimizer.pt
    scheduler.pt
    scaler.pt
    rng_state.pth
    original_norms.pt
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
