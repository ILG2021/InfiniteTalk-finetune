# InfiniteTalk 单人 LoRA 微调（81 帧训练窗口）

本仓库训练已有 InfiniteTalk 权重上的 LoRA。参考 [论文](https://arxiv.org/html/2508.14033v1) 的上下文和邻近参考帧设计；LoRA、量化、噪声采样和 dropout 是本项目配置，不是官方训练配方。

## 数据准备

### 权重下载

在训练环境安装 `huggingface_hub` 后运行：

```powershell
python download_weights.py --profile training
python download_weights.py --profile training --download
python download_weights.py --profile caption --download
python download_weights.py --profile all --check
```

第一行仅打印下载计划。`training` 包含官方单人 InfiniteTalk FP8 完整 DiT 及量化 JSON、Wan config、VAE、CLIP、T5、两个 tokenizer，以及 wav2vec2 的 config/preprocessor/safetensors；`caption` 下载 Qwen3-VL-2B-Instruct。所有源固定到已核对的 commit，支持重跑下载。wav2vec2 使用上游 README 引用的 safetensors 转换分支的固定版本，不重复下载 `.bin` 或 fairseq 权重。`--check` 只检查文件存在且非空，不是校验哈希或实际模型加载。下载到其他磁盘可传 `--weights_dir D:\models`，后续训练/预处理参数也需指向该目录。

默认直接加载官方 `quant_models/infinitetalk_single_fp8.safetensors`，保留 E4M3 数据和逐通道 FP32 scale，不重新量化。该文件已包含 InfiniteTalk 音频层，无需原始 Wan 七个分片或单独的音频层权重；相邻同名 JSON 必须保留。ComfyUI 格式不支持此加载入口。仅下载 FP8 文件和 JSON 可用 `--profile fp8 --download`；需要原始权重时用 `--profile training-original --download`。脚本不下载多人、INT8、推理加速 LoRA 等无关权重。YOLO 的本地 `weights/yolov8n.pt` 仍需单独准备。

### 切片与打标

切片和训练默认均为 **81 帧、25 fps（3.24 秒）**。每片独立打标，训练读取完整片段，避免标签描述片段之外的动作。先保证源素材没有镜头切换；本脚本不自动切镜头。

本地打标默认使用 [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)。本机检测到 RTX 4060 Laptop 8GB，因此默认每片采样 8 帧、限制视觉像素预算；这不代表已验证显存一定够用。显存更充足可改用同系列 8B Instruct。短促动作可能被抽帧遗漏，可增加 `--caption_frames 16` 并人工复核。YOLO 检测仍遍历全部 81 帧。

安装匹配显卡的 CUDA PyTorch/torchvision、`requirements-preprocess.txt`，将 FFmpeg/ffprobe 放入 PATH。准备本地 COCO YOLO 权重 `weights/yolov8n.pt`、Qwen3-VL 模型目录 `weights/Qwen3-VL-2B-Instruct` 和 `weights/chinese-wav2vec2-base`。脚本不会自动下载模型。Windows 单卡训练使用 `requirements-train-windows.txt`，避免安装单卡 SDPA 不需要的 xfuser/FlashAttention；还需上述 `training` 权重清单。依赖清单需在目标 CUDA 环境完成安装和启动验证。

```bash
python prepare_data.py --video_dir ./raw_videos --output_dir ./training_data --device cuda:0
```

处理顺序：统一 25 fps → 连续不重叠 81 帧切片 → 在窗口外左右最多 25 帧中随机选参考帧 → 逐帧检测人物并取全部框的并集 → 左右各扩展 200 像素、上下各扩展 50 像素 → 同框裁剪视频与参考图 → 本地视频打标 → 音频特征。padding 按源画面像素计算，越界部分截到画面边缘；多人时包含所有检出人物，不做人物跟踪。宽高调整到 16 的倍数，可用 `--target_h 512` 指定输出高度。音频只做采样率和时间轴对齐，不做响度归一化或平滑。

递归保留素材子目录，输出示例：

```text
training_data/
  videos/原始视频名_000001.mp4
  ref_images/原始视频名_000001ref.jpg
  captions/原始视频名_000001.txt
  audio/原始视频名_000001.wav
  audio_embs/原始视频名_000001.pt
  manifest.json
  metadata.json
```

末尾不足 81 帧会丢弃；无法取得片段外参考帧（例如原视频恰好 81 帧）或整片无人检出会跳过并记入 manifest。部分帧漏检时使用其余帧边界，manifest 记录检出帧数，供人工筛查。参考图沿用该片裁剪框，应复核人物在邻近参考帧中是否仍完整位于框内。源视频须有音轨。无损规范化临时文件可能较大，可指定 `--temp_dir`。

可以分阶段执行，逐步检查：

```bash
python prepare_data.py --stage clips --video_dir ./raw_videos --output_dir ./training_data
python prepare_data.py --stage captions --output_dir ./training_data
python prepare_data.py --stage audio --output_dir ./training_data
```

切片阶段要求空输出目录。打标与音频阶段可重跑，已有非空 txt 和有效形状的音频特征会复用。人工修改 txt 后重跑 captions 同步 metadata。更换模型时须另建数据目录或自行移除相应旧产物，否则仍会复用。标签和音频特征全部齐备后才发布 metadata.json。YOLO、打标模型和音频模型依次加载、释放，不同时驻留。

新数据的邻近参考图在预处理时抽取并保存，训练直接使用，且要求 `--frame_num` 与切片帧数一致。旧的长视频 metadata 仍支持训练时动态邻近采样；`--reference_mode fixed` 对旧数据使用已有 ref_image。首段训练分支仍使用目标片段首帧作为起始条件。

## 训练

```bash
python train_lora.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --fp8_checkpoint weights/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors \
    --data_dir ./training_data --quant fp8 \
    --lora_rank 64 --lora_alpha 64 --lr 1e-4 --audio_lr 1e-4 \
    --max_steps 5000 --frame_num 81 \
    --reference_mode adjacent --ref_neighbor_frames 25 \
    --cfg_drop_text_prob 0.1 --cfg_drop_audio_prob 0.1 --cfg_drop_both_prob 0.05 \
    --cfg_drop_clip_prob 0 --cfg_drop_ref_prob 0 \
    --use_8bit_optim --blocks_to_swap 38 --gradient_checkpointing --use_amp \
    --output_dir output/my_lora --save_every 250 --log_every 10 --debug_assert_shapes
```

这是起始配置，不保证任意分辨率下都能在 32GB 显卡上运行。先验证少量训练步与生成结果，再决定学习率、rank 和总步数。CPU 权重流式加载增加内存与传输开销；FP8 压缩基础权重，计算时反量化，不等于原生 FP8 矩阵计算加速。

- **81 帧窗口**：对应 21 个 latent。续接前 9 帧是 3 个干净 latent，仅监督后 18 个 latent（72 个新视频帧）。首段使用窗口第一帧作为参考图，固定第一个 latent，并排除其损失。推理对应 `--frame_num 81 --motion_frame 9`。
- **缓存**：随机参考图与首段参考图不缓存 CLIP/VAE 条件；固定图按名称缓存，VAE 缓存同时区分帧数。完整视频 latent 按视频名、起始帧和长度缓存。四类缓存共用 `--cpu_cache_gb` 指定的总预算（各占四分之一），达到上限淘汰最久未使用项；每次启动重新构建。
- **噪声采样**：全程固定使用标准 logit-normal：`sigma = sigmoid(z), z ~ N(0,1)`，均值 0、标准差 1，不额外施加 shift，也不分阶段。它决定每个训练样本的噪声强度；推理仍可独立使用 Euler 和 shift=11。旧参数 `--timestep_bias`、`--bias_switch_step`、`--override_shift` 已移除，启动命令中需要删除。旧 checkpoint 可以加载，但恢复后使用新的采样分布。
- **条件 dropout**：文本、音频和同时丢弃的概率之和不超过 1。CLIP 和 VAE 参考 dropout 默认关闭。显式启用 VAE dropout 时，mask 与 latent 都清零，表达参考条件缺失，不再把位置 mask 随机化；该实验需要单独评估。
- **训练参数**：基础权重和归一化层冻结，只训练指定视觉与音频线性层的 LoRA。默认训练音频 LoRA，`--no-train_audio` 可关闭。省略 audio_lr 时使用 lr。
- **量化**：`--fp8_checkpoint` 自动启用 FP8，保留文件中的量化权重和 scale，原文件未量化的层使用 BF16，显式关闭 autocast 的时间嵌入和输出头保留 FP32；不能同时传 `--infinitetalk_dir`。LoRA 参数使用 FP32，基础权重在计算时反量化为 BF16，并非原生 FP8 GEMM。checkpoint 记录 `fp8_weight_scaling=checkpoint_preserved`。原始权重路径仍支持 `--infinitetalk_dir ... --quant fp8` 本地逐通道量化，记录 `per_output_channel_v1`；不传 quant 则使用 BF16。INT8 训练未实现，会在加载前报错。

## Windows / RTX 5090 显存优化

实现参考 SimpleTuner 的 [CPU 权重流式加载](https://github.com/bghira/SimpleTuner/blob/main/simpletuner/helpers/ramtorch/modules/linear.py) 和 [检查点激活卸载](https://github.com/bghira/SimpleTuner/blob/main/simpletuner/helpers/training/offloaded_gradient_checkpointer.py)，使用本项目原生 PyTorch 实现。LoRA 参数和优化器留在 GPU，冻结 Linear 权重可常驻 CPU，在前向、反向需要时传入。FP8 权重先以压缩格式传输，再展开计算；反向重新展开，避免保存所有 BF16 权重。启动阶段直接加载预量化 FP8 权重并按最终位置放置参数，避免先把整模型放上 GPU。基础模型使用 meta 初始化与权重赋值加载，减少 CPU 加载时的重复副本。

5090 启动配置：81 帧、rank 64、FP8、最后 32 个 block 的冻结 Linear 权重卸载、检查点激活卸载、VAE 按需加载、8 GiB CPU 缓存、原生 SDPA、数据加载进程数 0。分辨率由数据决定，建议先准备 `--target_h 512` 的素材；启动脚本不会缩放已生成的数据。这是待实测的起始配置，不是显存或速度保证。

```powershell
.\train_windows_5090.ps1 -CkptDir 'D:\models\Wan2.1-I2V-14B-480P' -Fp8Weights 'D:\models\InfiniteTalk\quant_models\infinitetalk_single_fp8.safetensors' -DataDir 'D:\data\training_data' -MaxSteps 3
```

三步检查成功后去掉 `-MaxSteps 3`，默认训练 5000 步。使用 `-Python` 指定已有 CUDA 环境的 python.exe；`-Preview` 只打印参数。`-ResumeFrom` 接续检查点。

| 参数 | 作用及取舍 |
| --- | --- |
| `-CpuOffloadBlocks 32` / `--cpu_offload_blocks 32` | 卸载末尾 N 个 block 的冻结 Linear；允许 0–40。增加数量通常减少权重显存，但增加 PCIe 传输。旧参数 `--blocks_to_swap` 是同一选项的别名，训练已改为此流式实现。 |
| `--activation_offload` | 将整块梯度检查点的边界输入保存到 CPU；反向仍重新计算块内部。需要开启梯度检查点。 |
| `--vae_cpu_offload` | 默认开启，VAE 编码完成后移回 CPU；缓存命中可减少重复编码。 |
| `-CpuCacheGiB 8` / `--cpu_cache_gb 8` | 限制 CPU 张量缓存，设 0 关闭；不包含模型权重、激活和进程自身内存。 |
| `-PinMemory` / `--offload_pin_memory` | 可选锁页 CPU 权重和激活，可能改善传输；默认关闭，避免占用过多不可分页物理内存。先测普通 CPU 内存模式。 |
| `--attention_backend sdpa` | 默认使用 PyTorch 注意力及其反向；CUDA 路径只允许融合内核，避免悄悄退到会生成巨大注意力矩阵的 math 实现。内核不可用会明确报错。 |

日志在首步及每 `log_every` 步报告峰值 allocated/reserved 和 CPU 缓存使用量，TensorBoard 同时记录显存峰值。PyTorch 数字不涵盖桌面程序、驱动内存或 Windows 实际迁移开销，还需观察任务管理器与每步耗时。低于 32GB 的 dedicated VRAM 留有余量后，可以减少卸载 block 数以换取速度；OOM 时先增加卸载或降低数据分辨率。

你提到的约 96GB 共享显存来自系统内存，不能按同速的额外 96GB 显存配置模型。这里显式使用 CPU 内存进行卸载；页面文件也不能替代用于锁页的物理内存。没有修改 NVIDIA 控制面板、Windows 共享内存设置或页面文件。

GPU 环境须支持 Blackwell。PyTorch 从 [2.7 / CUDA 12.8](https://pytorch.org/blog/pytorch-2-7/) 开始支持，较新版本的 CUDA 构建和驱动需求请按其官方安装说明匹配。保留已能运行 5090 的环境即可；本次没有替换你的训练环境。单卡注意力不再强制导入 xformers/xfuser，训练不使用 SageAttention 推理内核；多卡 sequence parallel 仍需要相应依赖。

数值回归使用真实 PyTorch 对比冻结层前向、输入梯度、LoRA 梯度、检查点随机数重放及 padding mask。CUDA 卸载测试可在目标机器运行：`python -m unittest discover -s tests -p test_training_memory.py -v`。尚未取得 5090 上 14B / 81 帧的训练实测；CPU 测试不证明 CUDA 速度、峰值或完整模型可运行。

## 续训

在原训练命令上添加：

```bash
--resume_from output/my_lora/checkpoint-250 --override_lr --lr 5e-5
```

若希望音频学习率跟随 lr，应移除原命令中的 audio_lr。override_lr 重设两个组的学习率，并按剩余步数建立新调度；省略此开关则恢复原学习率和调度。max_steps 是包含已完成步数的总上限。续训必须保持 LoRA rank、alpha 和音频训练开关一致，不一致会在恢复前报错；禁止按参数顺序裁剪优化器状态。已有 33 帧权重可继续用 81 帧微调，但过去的条件错误与窗口分布不会自动消除，需要重新验证。

checkpoint 包含 adapter_model.safetensors、optimizer.pt、scheduler.pt、rng_state.pth、trainer_state.json。各个保存目录另有官方命名的 lora_for_inference.safetensors 和 WanVideoWrapper 命名的 lora_for_comfyui.safetensors，均已合入 alpha/rank 缩放。BF16 训练不保存 GradScaler，也不保存额外归一化差分文件。

## 推理与验收

### ComfyUI / WanVideoWrapper

每个训练 checkpoint 现在另存 `lora_for_comfyui.safetensors`。将它放入 ComfyUI 的 `models/loras`，在 `WanVideoLoraSelect` 选择这个文件。官方脚本继续使用 `lora_for_inference.safetensors`；`adapter_model.safetensors` 用于续训。两份推理文件都已将 alpha/rank 合入 up 权重，只加载其中适用于当前推理器的一份。

该区别来自 [WanVideoWrapper 加载器](https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/main/nodes_model_loading.py)：它将 `audio_proj` 改名为 `multitalk_audio_proj`。原来直接使用官方命名的导出文件，会漏掉四组音频投影 LoRA。视觉层和 block 内音频注意力键无需改名。

项目内的 `workflows/linda-inf-81-validation.json` 是原工作流的验收副本：81 帧、9 帧 context、FP8 权重/BF16 计算、Euler 30 步、CFG 5、固定 seed，旁路加速 LoRA，关闭额外响度归一化和平滑。先按此基线比较新 LoRA 强度 0 / 0.5 / 1，保持图像、音频、提示词和种子相同，再单独验证 5 步加速方案。这些是对比起始设置，并非效果保证。原文件中真正接入的是 T2V LightX2V，I2V LightX2V 节点未接入模型；不能仅凭两个文件都在画布中就判断启用了哪一个。

训练默认 `--clip_crop center`，与工作流 CLIP 中心裁剪、抗锯齿和像素取整一致；改用其他 CLIP 节点时需核对。FP8 官方 Quanto 和工作流 KJ scaled 版本使用不同量化实现；不能把二者视为逐元素相同，也没有在本机进行 14B CUDA 与 ComfyUI 的端到端生成验证。

训练前按**原始视频**留出验收素材，避免相邻切片和参考图泄漏到两边。抽检字幕是否只描述片内实际动作、参考图人物是否被裁掉、音画是否同步、有无切镜和多人混入。预处理不会自动纠正这些内容问题。用留出的参考图和音频比较首段、至少三个续接窗口和一分钟生成；检查口型、手部、身份和背景，选择生成效果最好的 checkpoint，不仅看训练 loss。

以下原始权重推理命令需要额外下载 `--profile training-original`。默认 FP8 训练下载清单只供上述训练入口使用；ComfyUI 推理沿用已有工作流。

```bash
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --lora_dir output/my_lora/checkpoint-final/lora_for_inference.safetensors \
    --lora_scale 1.0 --size infinitetalk-480 \
    --frame_num 81 --motion_frame 9 --mode streaming --input_json input.json
```

固定参考图、音频、随机种子和推理参数，对比训练前后的首段及多个连续窗口，检查口型、身份、背景和衔接。loss 下降不能替代生成验收。显存不足时优先降低预处理分辨率或 rank，尽量保留 81 帧窗口。

回归测试：`python -m unittest discover -s tests -v`。包括 NumPy 条件构造测试、真实 FFmpeg 对齐测试和 PyTorch 显存机制数值测试。缺少 PyTorch/CUDA 时对应测试会跳过，不能替代 14B 模型的真实 GPU 前后向和生成验收。
