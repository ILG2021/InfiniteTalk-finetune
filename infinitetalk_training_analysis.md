# InfiniteTalk 训练过程深度分析

> [!NOTE]
> 论文: [InfiniteTalk: Audio-driven Video Generation for Sparse-Frame Video Dubbing](https://arxiv.org/abs/2508.14033)
> 项目: [GitHub - MeiGen-AI/InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk)
> 项目当前仅提供推理代码，本文档基于论文内容和开源推理代码反推训练过程。

---

## 1. 项目概述

InfiniteTalk 是一个**音频驱动的无限长度说话人视频生成框架**，提出了"**稀疏帧视频配音 (Sparse-Frame Video Dubbing)**"的新范式。与传统仅编辑嘴部区域的配音方法不同，InfiniteTalk 能够在保持身份一致性、标志性手势和相机轨迹的同时，进行**全身音频同步的运动编辑**。

- **传统配音方法**: 只改嘴部区域 → 表情和体态与新音频不匹配，观感不协调
- **InfiniteTalk**: 以稀疏关键帧为参考，**重新生成整个视频**，嘴形、表情、头部、身体动作全部与音频同步

---

## 2. 模型架构

InfiniteTalk 基于 **Wan2.1-I2V-14B** 构建，是一个 14B 参数的 **Diffusion Transformer (DiT)**。

### 2.1 组件一览

| 组件 | 模型 | 用途 |
|------|------|------|
| **基础生成模型** | Wan2.1-I2V-14B (DiT) | 14B参数的扩散Transformer，视频生成骨干 |
| **文本编码器** | UMT5-XXL | 文本提示编码，最大512 tokens |
| **图像编码器** | CLIP/H | 参考图像特征提取，输出257个token |
| **音频编码器** | chinese-wav2vec2-base | 语音特征提取，使用全部12层hidden states |
| **视频VAE** | WanVAE | 视频编解码，时空压缩比 4×8×8 |
| **音频投影** | AudioProjModel (新增) | 将wav2vec2特征投影到DiT的隐空间 |
| **音频注意力** | SingleStreamAttention (新增) | 每个DiT Block中新增的音频交叉注意力 |

### 2.2 DiT 参数

来自 [wan_multitalk_14B.py](file:///d:/InfiniteTalk/wan/configs/wan_multitalk_14B.py):

| 参数 | 值 |
|------|-----|
| 隐藏维度 dim | 5120 |
| FFN 维度 | 13824 |
| 注意力头数 | 40 |
| Transformer 层数 | 40 |
| Patch Size | (1, 2, 2) |

### 2.3 DiT Block 结构

原始 Wan2.1-I2V 的每个 Block 有: Self-Attention → Text Cross-Attention → FFN。

InfiniteTalk **在 Text Cross-Attention 和 FFN 之间插入了一个 Audio Cross-Attention**：

```
DiT Block (×40):
  ┌─────────────────────────────────┐
  │  LayerNorm → Self-Attention     │  ← 3D RoPE 位置编码
  │         (+ 时间步调制残差)        │
  ├─────────────────────────────────┤
  │  LayerNorm → Text Cross-Attn   │  ← CLIP图像(257 tokens) + T5文本
  │         (残差连接)               │
  ├─────────────────────────────────┤
  │  LayerNorm → Audio Cross-Attn  │  ← 【新增】wav2vec2 音频特征   
  │         (残差连接)               │
  ├─────────────────────────────────┤
  │  LayerNorm → FFN (GELU)        │
  │         (+ 时间步调制残差)        │
  └─────────────────────────────────┘
```

对应代码 [WanAttentionBlock.forward](file:///d:/InfiniteTalk/wan/modules/multitalk_model.py#L274-L318):

```python
# self-attention
y = self.self_attn(self.norm1(x) * (1+e[1]) + e[0], ...)
x = x + y * e[2]  # 调制残差

# text cross-attention
x = x + self.cross_attn(self.norm3(x), context, context_lens)

# 【新增】audio cross-attention
x_a = self.audio_cross_attn(self.norm_x(x), encoder_hidden_states=audio_embedding, ...)
x = x + x_a

# FFN
y = self.ffn(self.norm2(x) * (1+e[4]) + e[3])
x = x + y * e[5]  # 调制残差
```

### 2.4 新增的可训练参数 (代码级分析)

通过对比 [model.py](file:///d:/InfiniteTalk/wan/modules/model.py) (原始 Wan2.1) 和 [multitalk_model.py](file:///d:/InfiniteTalk/wan/modules/multitalk_model.py) (InfiniteTalk)，可以**精确定位所有新增的参数**。

**分析方法**: 加载代码 ([multitalk.py:212-224](file:///d:/InfiniteTalk/wan/multitalk.py#L212-L224)) 先加载 Wan2.1 的 7 个基础分片，再 [update()](file:///d:/InfiniteTalk/wan/utils/multitalk_utils.py#321-324) InfiniteTalk 权重。因此 InfiniteTalk safetensors 中的 key 就是**新增或被覆盖的参数**。

#### AudioProjModel (全局，1个实例)

来自 [AudioProjModel](file:///d:/InfiniteTalk/wan/modules/multitalk_model.py#L366-L429):

| key | shape | 参数量 | 说明 |
|-----|-------|--------|------|
| `audio_proj.proj1.weight` | [512, 46080] | 23,592,960 | 首帧: `seq_len(5) × blocks(12) × 768 = 46080` → 512 |
| `audio_proj.proj1.bias` | [512] | 512 | |
| `audio_proj.proj1_vf.weight` | [512, 73728] | 37,748,736 | 后续帧: `seq_len_vf(8) × 12 × 768 = 73728` → 512 |
| `audio_proj.proj1_vf.bias` | [512] | 512 | |
| `audio_proj.proj2.weight` | [512, 512] | 262,144 | 第二层投影 |
| `audio_proj.proj2.bias` | [512] | 512 | |
| `audio_proj.proj3.weight` | [24576, 512] | 12,582,912 | `512 → context_tokens(32) × output_dim(768)` |
| `audio_proj.proj3.bias` | [24576] | 24,576 | |
| `audio_proj.norm.weight` | [768] | 768 | LayerNorm (`norm_output_audio=True`) |
| `audio_proj.norm.bias` | [768] | 768 | |
| | | **74,214,400** | AudioProjModel 合计 (~74M) |

#### SingleStreamMutiAttention (每个 DiT Block，共40个)

来自 [SingleStreamAttention.__init__](file:///d:/InfiniteTalk/wan/modules/attention.py#L191-L225)，注意 `qk_norm=False`：

| key 模式 (`blocks.{i}.`) | shape | 参数量/Block | 说明 |
|-----|-------|--------|------|
| `audio_cross_attn.q_linear.weight` | [5120, 5120] | 26,214,400 | Q 投影 (`qkv_bias=True`) |
| `audio_cross_attn.q_linear.bias` | [5120] | 5,120 | |
| `audio_cross_attn.kv_linear.weight` | [10240, 768] | 7,864,320 | 音频768维 → K,V 各5120维 |
| `audio_cross_attn.kv_linear.bias` | [10240] | 10,240 | |
| `audio_cross_attn.proj.weight` | [5120, 5120] | 26,214,400 | 输出投影 |
| `audio_cross_attn.proj.bias` | [5120] | 5,120 | |
| | | **60,313,600** | 单个 Block 的 audio_cross_attn (~60M) |

> [!TIP]
> 因为 `qk_norm=False`，`q_norm`、`k_norm`、`add_q_norm`、`add_k_norm` 全部是 `nn.Identity()`，**没有可训练参数**。`rope_1d` (RotaryPositionalEmbedding1D) 也无可训练参数（频率在推理时计算）。

#### norm_x (每个 DiT Block，共40个)

| key 模式 (`blocks.{i}.`) | shape | 参数量/Block | 说明 |
|-----|-------|--------|------|
| `norm_x.weight` | [5120] | 5,120 | WanLayerNorm (`elementwise_affine=True`) |
| `norm_x.bias` | [5120] | 5,120 | |

> [!NOTE]
> 原始 Wan2.1 的 [WanLayerNorm](file:///d:/InfiniteTalk/wan/modules/model.py#92-103) 默认 `elementwise_affine=False`（无可训练参数），而 InfiniteTalk 的 `norm_x` 设置为 `elementwise_affine=True`，PyTorch LayerNorm 在此选项下同时创建 weight 和 bias。已通过实际权重文件截图验证。

#### 参数量汇总

| 模块 | 计算 | 参数量 |
|------|------|--------|
| AudioProjModel (×1) | — | **74,214,400** |
| audio_cross_attn (×40) | 60,313,600 × 40 | **2,412,544,000** |
| norm_x (×40) | 10,240 × 40 | **409,600** |
| | | |
| **InfiniteTalk 新增总参数** | | **2,487,168,000 (~2.49B)** |
| Wan2.1-I2V-14B 基础参数 | | ~14B |
| **新增占比** | | **~17.8%** |

#### `infinitetalk.safetensors` 预期包含的完整 key 列表

```
# AudioProjModel (10个 key)
audio_proj.norm.bias
audio_proj.norm.weight
audio_proj.proj1.bias
audio_proj.proj1.weight
audio_proj.proj1_vf.bias
audio_proj.proj1_vf.weight
audio_proj.proj2.bias
audio_proj.proj2.weight
audio_proj.proj3.bias
audio_proj.proj3.weight

# 每个 Block (8个 key × 40 blocks = 320个 key)
blocks.0.audio_cross_attn.kv_linear.bias
blocks.0.audio_cross_attn.kv_linear.weight
blocks.0.audio_cross_attn.proj.bias
blocks.0.audio_cross_attn.proj.weight
blocks.0.audio_cross_attn.q_linear.bias
blocks.0.audio_cross_attn.q_linear.weight
blocks.0.norm_x.bias
blocks.0.norm_x.weight
blocks.1.audio_cross_attn.kv_linear.bias
...
blocks.39.norm_x.weight

# 总计: 10 + 320 = 330 个 key  ← 已与实际权重文件验证一致
```

> [!IMPORTANT]
> 这 **330 个 key 全部是新增参数**，不存在于原始 Wan2.1 的 7 个分片文件中。InfiniteTalk 没有覆盖或修改任何基础模型参数。这与 MultiTalk 论文的 **Partial Parameter Training** 策略完全吻合：冻结 DiT 基础参数，仅训练新增的 audio cross-attention layer 和 adapter。
>
> ✅ 已通过实际 `single/infinitetalk.safetensors` 权重文件截图验证，key 结构与代码分析完全一致。

---

## 3. 训练数据

- **数据量**: 约 **2000 小时**的说话人视频
- **关键特点**: **不需要配音视频对**，只需原始视频 + 自带音频轨即可
  - 训练时：视频的原始音频 ↔ 视频本身的运动 → 天然配对
  - 推理时：输入新音频 → 模型生成与新音频匹配的运动
- 视频帧率假定 **25 fps**（代码中 `video_length = audio_duration * 25`）
- 评估在 HDTF、CelebV-HQ、EMTD 数据集上进行

---

## 4. 训练目标

### 4.1 条件流匹配 (Conditional Flow Matching)

InfiniteTalk 使用 **Rectified Flow** 训练范式，而不是传统 DDPM：

**损失函数:**

```
L_fm = E_{t, x₀, x₁, c} ‖ v_θ(xₜ | c) - (x₁ - x₀) ‖²
```

- `x₀`: 干净视频潜变量（VAE 编码后的 GT 视频）
- `x₁`: 纯高斯噪声
- `xₜ = (1-t) · x₀ + t · x₁`: 线性插值（flow matching 的直线路径）
- `v_θ`: DiT 预测的速度场
- `c`: 所有条件信号（文本 + 音频 + 参考帧）

从推理代码 [multitalk.py:286-298](file:///d:/InfiniteTalk/wan/multitalk.py#L286-L298) 可验证插值方式：

```python
def add_noise(self, original_samples, noise, timesteps):
    # 线性插值: x_t = (1-t)*x_0 + t*noise
    timesteps = timesteps.float() / self.num_timesteps
    return (1 - timesteps) * original_samples + timesteps * noise
```

推理时的 Euler 求解也印证了 flow matching（速度场积分）：

```python
noise_pred = -noise_pred                           # 速度方向
dt = (timesteps[i] - timesteps[i+1]) / num_timesteps
latent = latent + noise_pred * dt                  # Euler 步进
```

### 4.2 时间步变换

推理时使用了 shift 变换，训练时应同样使用：

```python
def timestep_transform(t, shift=5.0, num_timesteps=1000):
    t = t / num_timesteps
    new_t = shift * t / (1 + (shift - 1) * t)   # 非线性变换，偏向更大噪声
    return new_t * num_timesteps
```

480P 默认 `shift=7`，720P 默认 `shift=11`。

---

## 5. 条件信号处理

### 5.1 参考帧处理

参考帧经过两条路径进入模型：

**路径1 — CLIP 特征 → Cross-Attention:**

```python
clip_context = self.clip.visual(cond_image)        # → 257 tokens
context_clip = self.img_emb(clip_fea)              # MLPProj: 1280 → 5120
context = concat([context_clip, text_emb], dim=1)  # 拼接后统一 cross-attn
```

**路径2 — VAE 潜变量 → 与噪声拼接:**

```python
# 参考帧 VAE 编码 + 零填充到目标帧数
y = self.vae.encode(padding_frames_pixels_values)  # → 16×T×H×W
# mask: 第一帧=1, 其余=0
msk = torch.ones(1, frame_num, lat_h, lat_w)
msk[:, 1:] = 0
# 拼接: [mask(4ch) + latent(16ch)] → 20ch 输入
y = concat([msk, y], dim=1)                        # → 20×T×H×W
# 与噪声 x 合并后进入 patch_embedding
x = concat([noise_x, y], dim=0)                    # channel 维拼接
```

### 5.2 音频处理流程

**Step 1: Wav2Vec2 特征提取**

```python
# 音频 → wav2vec2 → 12层 hidden states 全部堆叠
embeddings = audio_encoder(audio, seq_len=video_length, output_hidden_states=True)
audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1)  # → [帧数, 12, 768]
```

**Step 2: 滑窗上下文 (audio_window=5)**

每帧的音频特征包含前后各2帧的上下文，索引为 `[-2, -1, 0, +1, +2]`：

```python
indices = (torch.arange(5) - 2) * 1     # [-2, -1, 0, 1, 2]
center_indices = torch.arange(start, end).unsqueeze(1) + indices.unsqueeze(0)
audio_emb = full_audio_emb[center_indices]  # → [帧数, 5, 12, 768]
```

**Step 3: AudioProjModel 投影**

从 [AudioProjModel](file:///d:/InfiniteTalk/wan/modules/multitalk_model.py#L366-L429):

```python
# 首帧音频: 窗口5 × 12层 × 768维 = 46080 → 512
audio_embeds = relu(proj1(audio_embeds.flatten()))

# 后续帧音频(VAE时间压缩后): 窗口8 × 12层 × 768维 = 73728 → 512
audio_embeds_vf = relu(proj1_vf(audio_embeds_vf.flatten()))

# 拼接 → 第二层投影 → 生成最终token
audio_concat = concat([first_frame, latter_frames])
audio_concat = relu(proj2(audio_concat))
context_tokens = proj3(audio_concat)     # → [帧数, 32, 768]
context_tokens = norm(context_tokens)
```

最终每帧产生 **32个 token**，维度 768，送入每个 DiT Block 的 Audio Cross-Attention。

### 5.3 文本处理

```python
# T5 编码 → 线性投影
context = self.text_embedding(t5_output)   # Sequential: 4096 → 5120 → GELU → 5120
```

---

## 6. 参考帧采样策略（核心训练创新）

> [!IMPORTANT]
> 这是论文的**核心发现**：参考帧在训练数据中的采样方式决定了模型学到的控制强度。

论文通过消融实验对比了四种策略（Section 4）：

| 策略 | 做法 | 控制强度 | 问题 |
|------|------|---------|------|
| **M0** (均匀随机) | 从当前视频片段中随机选帧做参考 | 🔴 过强 | 模型学会"复制粘贴"参考帧的姿态到任意时间步，音频同步崩坏 |
| **M1** (首末帧) | 只用片段的第一帧或最后一帧 | 🔴 过刚 | 类似 FL2V 模型，边界处强制复制参考姿态，即便与音频情感矛盾 |
| **M2** (时间远距) | 采样间隔 >5秒 的帧 | 🟡 不足 | 参考帧与目标差异过大，身份保持不够，误差累积 |
| **M3** (时间相邻) ✅ | 从**相邻片段**采样（约1秒范围） | 🟢 适中 | 参考帧与目标帧足够相似→维持身份；又足够不同→允许自由运动 |

**根本原理**: I2V 模型的控制强度由**参考帧与待生成内容的相似度**决定。

- 参考帧和目标帧越像（同一片段内随机采） → 模型学会直接复制 → 控制过强
- 参考帧和目标帧差异过大（完全不相关的帧） → 模型忽略参考 → ID 丢失
- 时间相邻帧 → 恰好的相似度 → "软参考"，保身份但不限制运动

---

## 7. Classifier-Free Guidance 训练

推理时使用**双重 CFG**（Text + Audio），需要三次前向传播：

```python
# 1. 全条件
noise_pred_cond = model(latent, t, text=text, audio=audio)

# 2. 丢弃文本
noise_pred_drop_text = model(latent, t, text=null_text, audio=audio)

# 3. 全无条件
noise_pred_uncond = model(latent, t, text=null_text, audio=zeros)

# CFG 组合
noise_pred = noise_pred_uncond
           + text_scale  × (noise_pred_cond - noise_pred_drop_text)
           + audio_scale × (noise_pred_drop_text - noise_pred_uncond)
```

> [!TIP]
> 为了支持推理时的双重 CFG，**训练时必须随机 dropout 条件**：
> - 一定概率将文本替换为空/负面提示词
> - 一定概率将音频替换为全零向量
> - 一定概率同时 dropout 两者
>
> 这是标准 CFG 训练做法，使模型能学会"无条件"和"单条件"的预测。

---

## 8. 流式生成训练（Motion Frames 机制）

InfiniteTalk 支持无限长度视频，原理是**分片段生成 + 上下文衔接**：

```
时间轴:
──────────────────────────────────────────────────→

片段1 (81帧):
[参考帧][================================]
                                    ↕ 末尾9帧
片段2 (81帧):                       ↓
                          [motion][==================]
                                                ↕ 末尾9帧
片段3:                                          ↓
                                      [motion][==================]
                                      ...
最终视频: 拼接各片段（去掉重叠的motion frames）
```

**训练时的 Motion Frames 处理**:

```python
# 前一片段末尾的 GT（训练时有GT）或生成帧，作为当前片段的运动上下文
latent_motion_frames = vae.encode(cond_frame)  # 前一片段末尾9帧的潜变量

# 对 motion frames 按当前时间步重新加噪（匹配当前去噪状态）
add_latent = add_noise(latent_motion_frames, random_noise, current_timestep)
latent[:, :T_m] = add_latent  # 替换噪声前端的 T_m 帧

# 去噪循环中每步都重新注入（保持 motion frames 的主导性）
for i in range(sampling_steps):
    noise_pred = model(latent, timestep, ...)
    latent = latent + noise_pred * dt
    # 重新注入 motion frames（按更新后的时间步加噪）
    add_latent = add_noise(latent_motion_frames, new_noise, timesteps[i+1])
    latent[:, :T_m] = add_latent
```

这个机制保证了：
1. 前一片段的运动方向（惯性）传递到下一片段
2. 片段间过渡平滑，没有突变
3. 但 motion frames 不是完全固定的，通过加噪允许一定的自由度

---

## 9. 训练阶段与超参数

> [!IMPORTANT]
> 以下训练参数来自 InfiniteTalk 的前身 **MultiTalk** ("Let Them Talk", Kong et al. 2025，同一团队 MeiGen)。InfiniteTalk 构建在 MultiTalk 之上，训练方案高度一致。

### 9.1 训练硬件

- **GPU**: **64 × NVIDIA H100 80G**（论文确认）
- 使用 FSDP (Fully Sharded Data Parallel) + gradient checkpointing + 混合精度训练

### 9.2 训练策略: Partial Parameter Training（部分参数训练）

**核心策略**: 冻结 DiT 基础网络的所有原始参数，**仅训练新增的音频模块**。

冻结的参数（不更新）:
- Self-Attention (Q/K/V/O)
- Text Cross-Attention
- FFN
- Patch Embedding
- Time Embedding
- Head
- img_emb (CLIP projector)

训练的参数（更新）:
- `audio_proj` (AudioProjModel)
- `audio_cross_attn` (×40 DiT Blocks)
- `norm_x` (×40 DiT Blocks)

> [!TIP]
> 这种 partial parameter training 策略的核心目的是**保留基础模型的指令跟随能力和视频生成质量**，仅学习音频到视频运动的映射。在计算资源和数据有限时尤为关键。

### 9.3 训练超参数

| 参数 | 值 | 来源 |
|------|-----|------|
| 优化器 | **AdamW** | MultiTalk 论文确认 |
| 学习率 | **2e-5** (constant + warm-up) | MultiTalk 论文确认 |
| 训练数据 | **~2000 小时**说话人视频 | InfiniteTalk 论文确认 |
| 训练 GPU | **64 × H100 80G** | InfiniteTalk 论文确认 |
| 视频片段长度 | **81 帧** (4n+1) | 代码确认 |
| Context Frames | **9 帧** (潜空间 tc=3) | 论文确认 |
| 每步新生成帧 | **72 帧** (81-9) | 论文确认 |
| VAE 压缩比 | 4×8×8 (T×H×W) | 代码确认 |
| Flow Matching 时间步 | 1000 | 代码确认 |
| 并行策略 | FSDP + gradient checkpointing + 混合精度 | 衍生工作确认 |

### 9.4 训练阶段

| 阶段 | 内容 | 详情 |
|------|------|------|
| **Stage 0** | 基础权重 | 直接加载 Wan2.1-I2V-14B 预训练权重 |
| **Stage 1** | 单人说话训练 | 冻结 DiT 基础参数 (partial parameter training)，只训练 `audio_proj` + `audio_cross_attn` + `norm_x`，学习率 2e-5，AdamW |

---

## 10. 训练伪代码

```python
# ═══════════════════════════════════════════
# 初始化
# ═══════════════════════════════════════════
model = WanModel(**config)
model.load_pretrained("Wan2.1-I2V-14B")      # 加载预训练 DiT
# infinitetalk 的新增参数随机初始化 (Xavier)

vae = WanVAE(...)                             # 冻结
clip = CLIPModel(...)                         # 冻结
t5 = T5EncoderModel(...)                      # 冻结
# wav2vec2 在数据预处理阶段使用，训练时直接加载预计算的 embedding

# 冻结 DiT 基础参数 (Partial Parameter Training)
for name, param in model.named_parameters():
    if 'audio' not in name and 'norm_x' not in name:
        param.requires_grad = False

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-5,           # MultiTalk 论文确认
    weight_decay=0.01,
)

# ═══════════════════════════════════════════
# 数据准备 (离线)
# ═══════════════════════════════════════════
# 对每个训练视频:
#   1. VAE encode 所有帧 → 潜变量 (16×T×H×W)
#   2. Wav2Vec2 提取音频 embedding → [帧数, 12, 768]
#   3. CLIP encode 关键帧
#   4. T5 encode 文本描述

# ═══════════════════════════════════════════
# 训练循环
# ═══════════════════════════════════════════
for batch in dataloader:
    # --- 准备 GT ---
    video_latent = batch['vae_latent']            # x₀, 干净潜变量
    audio_emb    = batch['audio_embedding']       # wav2vec2 预提取
    text_emb     = batch['text_embedding']        # T5 预提取

    # --- 参考帧采样 (核心!) ---
    # 从时间相邻的片段中采样参考帧 (约1秒范围)
    ref_frame = sample_from_adjacent_chunk(batch)
    clip_emb  = clip.visual(ref_frame)
    ref_latent = vae.encode(ref_frame)

    # --- 构建输入 ---
    noise = torch.randn_like(video_latent)        # x₁
    t = torch.rand(B) * num_timesteps             # 随机时间步
    t_shifted = timestep_transform(t, shift=7)    # 时间步变换

    # Flow Matching 线性插值
    t_norm = t_shifted / num_timesteps
    x_t = (1 - t_norm) * video_latent + t_norm * noise

    # 目标速度 = 噪声 - 干净数据
    target_velocity = noise - video_latent

    # --- CFG Dropout ---
    # 随机决定 drop 哪些条件
    drop_text  = (torch.rand(B) < 0.1)
    drop_audio = (torch.rand(B) < 0.1)
    text_emb[drop_text]   = null_text_embedding
    audio_emb[drop_audio] = torch.zeros_like(audio_emb[drop_audio])

    # --- mask + 参考帧拼接 ---
    mask = torch.zeros(...)
    mask[:, :1] = 1                               # 第一帧mask=1
    y = concat([mask, ref_latent_padded], dim=1)  # 20ch input

    # --- 前向传播 ---
    pred_velocity = model(
        x=[x_t],                                  # 噪声输入
        t=t_shifted,                              # 时间步
        context=[text_emb],                       # 文本
        clip_fea=clip_emb,                        # CLIP图像
        y=y,                                      # mask+参考帧潜变量
        audio=audio_emb,                          # 音频
    )

    # --- 损失计算 ---
    loss = F.mse_loss(pred_velocity, target_velocity)

    # --- 反向传播 ---
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## 11. 训练 vs 推理 对比

| 方面 | 训练 | 推理 |
|------|------|------|
| 参考帧来源 | GT视频的相邻片段帧 | 用户输入的图像/视频 |
| 音频来源 | 视频自带的原始音频 | 用户提供的新音频 |
| Motion Frames | GT的末尾帧（加噪） | 上一片段生成结果的末尾帧（加噪） |
| CFG 条件 | 随机 dropout | 三次前向 + CFG scale 加权组合 |
| 时间步 | 随机均匀采样 t∈[0,1] | 等间距 1000→1 (40步) |
| 求解器 | 无（单步损失） | Euler 求解器（多步迭代） |
| 批量 | 多视频 batch 训练 | 单个视频逐片段生成 |

---

## 12. 关键技术总结

| 创新点 | 说明 |
|--------|------|
| **稀疏帧配音范式** | 不仅改嘴，而是以关键帧为参考重新生成整个视频，全身同步 |
| **相邻帧采样策略** | 训练时参考帧取自相邻时间段，获得"适中"控制强度，既保身份又不限制运动 |
| **流式 Motion Frames** | 通过传递末尾帧的运动惯性实现无限长度生成和片段间平滑过渡 |
| **双重 CFG** | 文本和音频分别独立做 CFG，可分别控制视觉风格和唇形同步强度 |
| **Wav2Vec2 全层特征** | 使用12层 hidden states 而非仅最后一层，获取更丰富的语音表征 |
| **滑窗音频上下文** | 每帧携带 ±2帧 的音频窗口，提升音画同步的时间精度 |

> [!NOTE]
> **已确认的训练参数** (来源: InfiniteTalk 论文 + 前身 MultiTalk 论文 + 衍生工作):
> - 优化器 AdamW, 学习率 2e-5, constant schedule + warm-up
> - 64 × H100 80G GPU 集群
> - Partial parameter training: 仅训练 audio 相关模块
> - ~2000 小时训练数据, 81帧/chunk, 9帧 context frames
>
> **未公开的细节** (论文技术报告中可能有但无法从搜索引擎确认):
> - 具体 batch size、训练总步数/epoch数
> - warm-up 步数
> - CFG dropout 概率的具体值
> - 是否在 Stage 1 之后有全参数微调的 Stage 2

---

## 13. LoRA 微调实现 (单人, RTX 5090)

> [!IMPORTANT]
> 本节提供在**单张 RTX 5090 (32GB VRAM)** 上进行 InfiniteTalk 单人 LoRA 微调的完整方案。

### 13.1 VRAM 预算分析

| 组件 | 显存占用 | 说明 |
|------|----------|------|
| DiT (BF16) | ~28 GB | ❌ 无法全放 VRAM |
| DiT (INT8 量化) | ~14 GB | ✅ 可行 |
| VAE 编解码 | ~2 GB | 临时占用，用完释放 |
| CLIP 编码 | ~1 GB | 临时占用，用完释放 |
| LoRA 参数 (rank=16) | ~50 MB | 极小 |
| LoRA 梯度 + 优化器 | ~150 MB | Adam 状态 |
| 激活值 (17帧, 480P) | ~4-8 GB | gradient checkpointing 可降低 |
| **总计 (INT8 + 17帧)** | **~22-25 GB** | ✅ **5090 可用** |

### 13.2 训练参数分布与策略

训练脚本默认将对模型进行两路并行的微调策略：

**策略 A: LoRA 微调** (应用于体积较大的全连接层)
```
blocks.{0~39}.audio_cross_attn.q_linear   ← Q 投影, [5120, 5120]
blocks.{0~39}.audio_cross_attn.kv_linear   ← KV 投影, [10240, 768]
blocks.{0~39}.audio_cross_attn.proj        ← 输出投影, [5120, 5120]
audio_proj.proj1                           ← 首帧音频投影, [5120, 46080]
audio_proj.proj1_vf                        ← 后续帧音频投影, [5120, 73728]
audio_proj.proj2                           ← 二阶段投影, [5120, 5120]
audio_proj.proj3                           ← 最终特征投影, [24576, 5120]
```

**策略 B: 全参微调差异保存 (.diff)** (应用于体积过小，不适合 LoRA 的归一化层)
```
blocks.{0~39}.norm_x    ← (WanLayerNorm, 10K params/block × 40 = 410K)
audio_proj.norm         ← (LayerNorm, 1.5K params)
```
*所有新增的这 330 个 InfiniteTalk 专属权重 (不管是 LoRA 还是全参) 都被完美覆盖，且不会污染原本的 Wan2.1 基础权重。*

| LoRA Rank | 每层参数 | 总 LoRA 参数 | 文件大小 (BF16) |
|-----------|----------|-------------|----------------|
| 4 | ~48K | ~5.7M | ~11 MB |
| 8 | ~96K | ~11.5M | ~23 MB |
| **16** (推荐) | ~192K | **~23.0M** | **~46 MB** |
| 32 | ~384K | ~46.0M | ~92 MB |
| 64 | ~768K | ~92.0M | ~184 MB |

### 13.3 数据准备

**Step 1: 准备视频**

```
raw_videos/
├── person_clip01.mp4    # 5-60秒, 含清晰语音
├── person_clip02.mp4
└── ...
```

**Step 2: 提取 wav2vec2 嵌入**

```bash
python prepare_data.py \
    --video_dir ./raw_videos \
    --output_dir ./training_data \
    --wav2vec_model TencentGameMate/chinese-wav2vec2-base \
    --prompt "A person is talking." \
    --device cuda:0
```

输出结构:
```
training_data/
├── videos/         # 视频副本
├── audio_embs/     # .pt 文件, 每个 shape [num_frames, 12, 768]
└── metadata.json   # 样本列表
```

### 13.4 训练命令

```bash
python train_lora.py \
    --ckpt_dir /path/to/Wan2.1-I2V-14B-480P \
    --infinitetalk_dir /path/to/single/infinitetalk.safetensors \
    --data_dir ./training_data \
    --quant int8 \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lr 1e-4 \
    --max_steps 1000 \
    --frame_num 17 \
    --gradient_checkpointing \
    --use_amp \
    --output_dir output/my_lora \
    --save_every 200 \
    --log_every 10
```

### 13.5 使用训练好的 LoRA 推理

```bash
python generate_infinitetalk.py \
    --ckpt_dir /path/to/Wan2.1-I2V-14B-480P \
    --infinitetalk_dir /path/to/single/infinitetalk.safetensors \
    --lora_dir output/my_lora/lora_final.safetensors \
    --lora_scale 1.0 \
    --size infinitetalk-480 \
    --input_json input.json
```

### 13.6 LoRA 权重格式

训练产出的 safetensors 兼容 [wan_lora.py](file:///d:/InfiniteTalk/wan/wan_lora.py)：

# Key 命名 (兼容 wan_lora.py):
# 1. LoRA 矩阵:
diffusion_model.blocks.0.audio_cross_attn.q_linear.lora_down.weight  # [rank, 5120]
diffusion_model.blocks.0.audio_cross_attn.q_linear.lora_up.weight    # [5120, rank]
diffusion_model.blocks.0.audio_cross_attn.kv_linear.lora_down.weight # [rank, 768]
diffusion_model.blocks.0.audio_cross_attn.kv_linear.lora_up.weight   # [10240, rank]
...
diffusion_model.audio_proj.proj1.lora_down.weight
...
# 2. 小归一化层 (全参差值):
diffusion_model.blocks.0.norm_x.diff           # weight 差值
diffusion_model.blocks.0.norm_x.diff_b         # bias 差值
diffusion_model.audio_proj.norm.diff
...
# 共 330 个独立可训练组件，完美覆盖全部 InfiniteTalk 音频模块
```

### 13.7 训练参数调优建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `frame_num` | **17** (保守) / 33 (激进) | 帧数越多效果越好但显存越大，必须是 4n+1 |
| `lora_rank` | **16** | 平衡效果和速度 |
| `lr` | **1e-4** | LoRA 一般用比全参更大的学习率 |
| `max_steps` | **500~2000** | 视数据量调整 |
| `quant` | **int8** | 5090 必须量化基础模型 |

> [!WARNING]
> - `frame_num` 必须是 **4n+1** (5, 9, 13, 17, 21, ..., 81)，因为 VAE 时间压缩比为 4
> - 训练帧数 (如17) 和推理帧数 (81) 不同是正常的，模型可以泛化
> - 显存不足时优先降 `frame_num`，其次降 `lora_rank`

### 13.8 相关文件

| 文件 | 说明 |
|------|------|
| [train_lora.py](file:///d:/InfiniteTalk/train_lora.py) | LoRA 微调训练脚本 |
| [prepare_data.py](file:///d:/InfiniteTalk/prepare_data.py) | 数据预处理脚本 |
| [wan_lora.py](file:///d:/InfiniteTalk/wan/wan_lora.py) | LoRA 推理加载器 (项目自带) |
