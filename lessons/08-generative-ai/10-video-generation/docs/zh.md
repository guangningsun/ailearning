# 视频生成

> 图像是 2 维张量。视频是 3 维的。理论相同，计算量却难 10-100 倍。OpenAI 的 Sora（2024 年 2 月）证明了可行性。到 2026 年，Veo 2、Kling 1.5、Runway Gen-3、Pika 2.0 和 WAN 2.2 已经可以生成 1080p 的生产级文本到视频——而开源权重模型（CogVideoX、HunyuanVideo、Mochi-1、WAN 2.2）落后约 12 个月。

**类型：** 构建型
**语言：** Python
**前置条件：** 第 8 阶段 · 07（潜空间扩散）、第 7 阶段 · 09（ViT）、第 8 阶段 · 06（DDPM）
**时间：** 约 45 分钟

## 问题

一段 10 秒 1080p 24fps 的视频有 240 帧，每帧 1920×1080×3 像素。未经压缩时每个片段约 1.5 GB。像素空间的扩散不可行，你需要：

1. **时空压缩。** 用一个 VAE 将视频（而非单帧）编码为一序列时空 patches。
2. **时间一致性。** 各帧需要在数秒内共享内容、光照和物体身份。网络必须建模运动。
3. **算力预算。** 同等模型规模下，视频训练比图像训练贵 10-100 倍。
4. **条件控制。** 文本、首帧图像、音频或另一段视频。大多数生产模型支持全部四种。

解决这个问题的架构是将 **Diffusion Transformer（DiT）** 应用于时空 patches，并在海量（prompt、caption、video）数据集上训练。与第 6 课相同的扩散损失。

## 概念

![视频扩散：patchify、DiT、解码](../assets/video-generation.svg)

### Patchify

用 3D VAE（学习到的时空压缩）编码视频。潜空间形状为 `[T_latent, H_latent, W_latent, C_latent]`。切分为大小为 `[t_p, h_p, w_p]` 的 patches。对于 Sora 风格模型，`t_p = 1`（逐帧 patches）或 `t_p = 2`（每两帧）。一段 10 秒 1080p 视频压缩后约 20,000-100,000 个 patches。

### 时空 DiT

Transformer 处理 patches 的扁平序列。每个 patch 有 3D 位置编码（时间 + y + x）。注意力通常分解为：

- **空间注意力** 作用于每帧的 patches 内部。
- **时间注意力** 跨帧作用于相同空间位置。
- **完整 3D 注意力** 计算量是前者的 16-100 倍；仅在低分辨率或研究场景使用。

### 文本条件

与大型文本编码器（T5-XXL 用于 Sora，CogVideoX-5B 也用 T5-XXL）做交叉注意力。长 prompt 很重要——Sora 训练集用了 GPT 生成的高密度 re-caption，平均每个片段 200 tokens。

### 训练

在时空潜空间上用标准扩散损失（ε 或 v 预测）。数据：网络视频 + 约 1 亿条精选片段 + 合成文本 caption。算力：一次小型研究实验需要 10,000+ GPU 小时；Sora 规模则需 100,000+。

## 2026 年生产格局

| 模型 | 日期 | 最大时长 | 最大分辨率 | 开源权重？ | 备注 |
|-------|------|--------------|---------|---------------|---------|
| Sora（OpenAI） | 2024-02 | 60 秒 | 1080p | 否 | 首个在规模上展现世界模拟器特性的模型 |
| Sora Turbo | 2024-12 | 20 秒 | 1080p | 否 | 生产级 Sora，推理速度快 5 倍 |
| Veo 2（Google） | 2024-12 | 8 秒 | 4K | 否 | 2025 年最高质量 + 物理效果 |
| Veo 3 | 2025 Q3 | 15 秒 | 4K | 否 | 原生音频，更强的镜头控制 |
| Kling 1.5 / 2.1（快手） | 2024-2025 | 10 秒 | 1080p | 否 | 2025 Q1 最好的人体运动 |
| Runway Gen-3 Alpha | 2024-06 | 10 秒 | 768p | 否 | 基于专业视频工具构建 |
| Pika 2.0 | 2024-10 | 5 秒 | 1080p | 否 | 最强的角色一致性 |
| CogVideoX（THUDM） | 2024 | 10 秒 | 720p | 是（2B、5B） | 首个开源 5B 规模视频模型 |
| HunyuanVideo（腾讯） | 2024-12 | 5 秒 | 720p | 是（13B） | 2024 年底开源 SOTA |
| Mochi-1（Genmo） | 2024-10 | 5.4 秒 | 480p | 是（10B） | 许可最宽松 |
| WAN 2.2（阿里巴巴） | 2025-07 | 5 秒 | 720p | 是 | 2025 年中开源最强模型 |

开源权重正在以比图像领域更快的速度追赶：到 2026 年中，HunyuanVideo + WAN 2.2 LoRA 已为大多数开源工作流提供动力。

## 构建它

`code/main.py` 模拟核心时空 DiT 思想：将小型合成视频 patchify，添加每个 patch 的位置编码，用 Transformer 风格的注意力处理整个序列去噪。全程纯 Python，不依赖 numpy。我们证明，即使在 1 维情况下，当相邻帧的 patches 共享去噪器和位置编码时，时间一致性也会自然涌现。

### 第 1 步：patchify 一个合成 1-D"视频"

```python
def make_video(T_frames=8, rng=None):
    # 一个"视频"是一个沿着平滑轨迹的 1-D 值序列
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### 第 2 步：每帧的位置编码

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### 第 3 步：去噪器看到整个序列

去噪器不独立处理每帧，而是将所有帧的值及其位置编码拼接起来，联合预测所有帧的噪声。

### 第 4 步：时间一致性测试

训练后，对视频进行采样。测量帧间 delta。如果模型学到了时间结构，delta 会比独立采样每帧时更小。

## 陷阱

- **独立逐帧采样 = 闪烁。** 如果对每帧单独运行图像扩散，输出会闪烁，因为每帧的噪声是独立的。视频扩散通过注意力或共享噪声耦合帧来解决这个问题。
- **朴素 3D 注意力 = OOM。** 在 10 秒 1080p 潜空间上做完整 3D 注意力需要数千亿次运算。分解为空间 + 时间注意力。
- **数据 captioning 的重要性超过规模。** Sora 相对于先前工作的主要升级是在更详细的 caption 上训练（GPT-4 重新标记的片段）。OpenAI 的技术报告明确指出了这一点。
- **首帧条件。** 大多数生产模型也接受将图像作为第一帧。这就是"图生视频"模式；训练包含这个变体。
- **物理漂移。** 长片段（>10 秒）会积累细微的不一致性。滑动窗口生成 + 关键帧锚定有助于缓解。

## 使用它

| 使用场景 | 2026 年推荐 |
|----------|-----------|
| 最高质量文本到视频（托管服务） | Veo 3 或 Sora |
| 镜头控制的电影感 | Runway Gen-3 + 运动笔刷 |
| 跨片段的角色一致性 | Pika 2.0 或 Kling 2.1 |
| 开源权重，快速微调 | WAN 2.2 + LoRA |
| 图生视频 | WAN 2.2-I2V、Kling 2.1 I2V 或 Runway |
| 音频到视频口型同步 | Veo 3（原生音频）或专用口型同步模型 |
| 视频编辑 | Runway Act-Two、Kling Motion Brush、Flux-Kontext（静帧） |

2024 到 2026 年间，同等质量下每秒视频的成本下降了 20 倍。

## 交付它

保存 `outputs/skill-video-brief.md`。技能接受一个视频简报（时长、宽高比、风格、镜头计划、主体一致性、音频）并输出：模型 + 托管服务、prompt 框架（镜头语言、主体描述、运动描述符）、种子 + 可复现性协议，以及逐帧 QA 检查清单。

## 练习

1. **简单。** 在 `code/main.py` 中，比较（a）独立逐帧采样、（b）联合序列采样的帧间 delta。报告 delta 的均值和方差。
2. **中等。** 添加首帧条件：将第 0 帧固定为给定值，采样其余帧。测量固定值如何传播。
3. **困难。** 用 HuggingFace diffusers 在本地 GPU 上运行 CogVideoX-2B。对 6 秒片段在 720p 下计时 20 步推理。分析时空注意力以定位瓶颈。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 视频 VAE | "3D VAE" | 将 `(T, H, W, C)` 压缩为时空潜空间的编码器。 |
| Patches | "tokens" | 潜空间的固定大小 3D 块；DiT 的输入。 |
| 分解注意力 | "空间 + 时间" | 先在空间上做注意力，再在时间上做；跳过完整 3D 注意力。 |
| 图生视频（I2V） | "让这张照片动起来" | 模型接收图像 + 文本，输出以此为起始的视频。 |
| 关键帧条件 | "锚定帧" | 固定特定帧以控制视频的弧线。 |
| 运动笔刷 | "方向提示" | 用户在图像上绘制运动向量的 UI 输入。 |
| Re-captioning | "密集 caption" | 用 LLM 用详细 prompt 重新标记训练片段。 |
| 闪烁 | "时间伪影" | 帧间不一致；通过耦合去噪修复。 |

## 生产注记：视频潜空间是内存带宽问题

一段 10 秒 1080p 24fps 片段有 240 帧 × 1920 × 1080 × 3 ≈ 1.5 GB 原始像素。经过 4× 视频 VAE 压缩（`2 × 空间 × 2 × 时间`）后潜空间约 100 MB 每请求。用时空 DiT 处理 30 步、batch=1，你要在 HBM 中搬运约 3 GB/步——瓶颈是内存带宽，不是 FLOPs。

三个生产推理旋钮，均来自生产推理文献推理章节：

- **DiT 上的 TP。** 文生视频模型通常 ≥10B 参数。TP=4 跨 4 张 H100 是标准配置；405B 级模型用 PP=2 × TP=2。每步延迟随 TP 近似线性下降，直至撞上 all-reduce 墙。
- **帧 batching = 连续 batching。** 在生成时，视频本质上是注意力链接的帧批次。连续 batching（飞行中调度）适用：当帧 `t-1` 正在返回时，开始渲染帧 `t+1`——如果模型架构支持滑动窗口生成的话。
- **片段级预填充缓存。** 对于图生视频，首帧条件类似于 LLM 的 prompt 预填充：计算一次，在时间解码器 passes 中复用。这本质上是视频的 KV-cache。

## 延伸阅读

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — Sora 技术报告。
- [Yang et al. (2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) — CogVideoX。
- [Kong et al. (2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603) — HunyuanVideo。
- [Genmo (2024). Mochi-1 Technical Report](https://www.genmo.ai/blog/mochi) — Mochi-1。
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/) — 2025 年中开源 SOTA。
- [Ho, Salimans, Gritsenko et al. (2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458) — 开创性视频扩散论文。
- [Blattmann et al. (2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818) — Stable Video Diffusion 的祖先。