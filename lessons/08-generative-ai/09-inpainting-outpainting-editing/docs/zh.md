# 修复、扩展与图像编辑

> 文生图创造新事物。修复完善旧事物。在生产环境中，70% 的可计费图像工作是编辑——换背景、去水印、扩展画布、重生成手部。修复是扩散模型真正发挥作用的地方。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 8 · 07（潜空间扩散）、阶段 8 · 08（ControlNet 与 LoRA）
**时间：** 约 75 分钟

## 问题

客户发来一张完美的产品照片，背景中有个碍眼的标识。你想擦掉标识，其他像素保持完全一致。你不能从头跑文生图——结果会有不同的颜色、不同的光线、不同的产品角度。你只想重新生成*仅限*掩码区域的图像，而且希望重新生成的内容尊重周围上下文。

这就是修复。变体：

- **修复（Inpainting）。** 在掩码内重新生成，保留外部像素。
- **扩展（Outpainting）。** 在掩码外重新生成（或者说在画布外），保留内部。
- **图像编辑。** 重新生成整张图像，但保持对原始图像的语义或结构保真度（SDEdit、InstructPix2Pix）。

2026 年的每个扩散流水线都带修复模式。Flux.1-Fill、Stable Diffusion Inpaint、SDXL-Inpaint、DALL-E 3 Edit。它们基于相同的原理工作。

## 概念

![修复：掩码感知的去噪与上下文保留的重新注入](../assets/inpainting.svg)

### 朴素方法（以及为什么它是错的）

用掩码运行标准文生图。在每个采样步骤中，将噪声潜空间的未掩码区域的像素替换为前向扩散的干净图像。它能工作……但效果很差。边界伪影会渗透，因为模型没有关于掩码区域内是什么的信息。

### 正确的修复模型

训练一个修改过的 U-Net，接受 9 个输入通道而不是 4 个：

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

额外的通道是 VAE 编码的源图像的副本加一个单通道掩码。在训练时，你随机掩码图像区域，训练模型仅在掩码区域去噪，而未掩码区域作为干净条件信号给出。在推理时，模型能"看到"掩码区域周围是什么，并产生连贯的补全。

SD-Inpaint、SDXL-Inpaint、Flux-Fill 都使用这个 9 通道（或类似）输入。Diffusers `StableDiffusionInpaintPipeline`、`FluxFillPipeline`。

### SDEdit（Meng 等，2022）—— 免费编辑

将源图像加噪到某个中间步 `t`，然后用新提示词从 `t` 向下运行反向链到 0。无需重训练。起始 `t` 的选择权衡保真度与创意自由度：

- `t/T = 0.3` → 与源图像几乎相同，小的风格变化
- `t/T = 0.6` → 中等编辑，保留粗略结构
- `t/T = 0.9` → 从近噪声生成，最小源图像保留

### InstructPix2Pix（Brooks 等，2023）

在（输入图像、指令、输出图像）三元组上微调扩散模型。在推理时，以输入图像和文本指令（"让它变成日落"、"加一条龙"）为条件。两个 CFG 尺度：图像尺度和文本尺度。

### RePaint（Lugmayr 等，2022）

保持标准的无条件扩散模型。在每个反向步骤中，重新采样——偶尔跳回更噪声的状态并重新生成。避免边界伪影。用于没有训练好的修复模型的情况。

## 构建它

`code/main.py` 在 5 维数据上实现了一个玩具 1-D 修复方案。我们在 5-D 混合数据上训练 DDPM，其中每个样本是来自两个簇之一的 5 个浮点数。在推理时，我们"掩码"5 个维度中的 2 个，在每一步注入未掩码三个的噪声前向版本，并仅重新生成掩码维度。

### 第 1 步：5-D DDPM 数据

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### 第 2 步：在所有 5 个维度上训练去噪器

标准 DDPM。网络为 5-D 噪声输入输出 5-D 噪声预测。

### 第 3 步：在推理时，掩码感知的反向

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # 用新鲜噪声化的干净源版本替换未掩码维度
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...然后对 x_t 运行正常的反向步骤
```

这是朴素方法，在玩具 1-D 数据上有效。真实图像修复使用 9 通道输入，因为纹理一致性更重要。

### 第 4 步：扩展

扩展是掩码反转的修复：掩码新的（之前不存在的）画布，用原始图像填充其余部分。训练目标相同。

## 陷阱

- **接缝。** 朴素方法会留下可见边界，因为梯度信息无法跨越掩码流动。修复：膨胀掩码 8-16 像素，或使用正确的修复模型。
- **掩码泄漏。** 如果条件图像的未掩码区域质量低或噪声大，它会污染掩码内的生成。稍微去噪或模糊。
- **CFG 与掩码大小相互作用。** 小掩码上的高 CFG = 饱和色块。为小编辑降低 CFG。
- **SDEdit 保真度悬崖。** 从 `t/T = 0.5` 到 `t/T = 0.6` 可能丢失主体的身份。扫描并检查点。
- **提示词不匹配。** 提示词应该描述*整张*图像，而不仅仅是新内容。"A cat sitting on a chair" 而不是 "a cat"。

## 使用它

| 任务 | 流水线 |
|------|----------|
| 去除物体，小掩码 | SD-Inpaint 或 Flux-Fill，标准提示词 |
| 替换天空 | SD-Inpaint + "blue sky at sunset" |
| 扩展画布 | SDXL outpaint 模式（8px 羽化）或 Flux-Fill 与 outpaint 掩码 |
| 重生成手部/脸部 | SD-Inpaint + 重新描述主体的提示词 + ControlNet-Openpose |
| 改变一个区域的风格 | 在掩码区域上使用 `t/T=0.5` 的 SDEdit |
| "让它变成日落" | InstructPix2Pix 或 Flux-Kontext |
| 背景替换 | SAM 掩码 → SD-Inpaint |
| 超高保真度 | Flux-Fill 或 GPT-Image（托管）用于最难的情况 |

SAM（Meta 的 Segment Anything，2023）+ 扩散修复是 2026 年背景去除流水线。SAM 2（2024）支持视频。

## 交付它

保存 `outputs/skill-editing-pipeline.md`。Skill 接受原始图像 + 编辑描述 + 可选掩码（或 SAM 提示词）并输出：掩码生成方法、基础模型、CFG 尺度（图像 + 文本）、SDEdit-t 或修复模式，以及 QA 检查清单。

## 练习

1. **简单。** 在 `code/main.py` 中，将掩码维度的比例从 0.2 变化到 0.8。在什么比例下修复质量（掩码维度的残差）等于无条件生成？
2. **中等。** 实现 RePaint：在每第 10 个反向步骤，跳回 5 步（加噪）并重新去噪。测量它是否减少了掩码边缘的边界残差。
3. **困难。** 使用 Hugging Face diffusers 比较：SD 1.5 Inpaint + ControlNet-Openpose 与 Flux.1-Fill 在 20 个脸部重生成任务上的表现。分别评分姿态遵循度和身份保留度。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|-----------------------|
| 修复（Inpainting） | "填补空洞" | 在掩码内重新生成；保留外部像素。 |
| 扩展（Outpainting） | "扩展画布" | 在画布外重新生成；保留内部。 |
| 9 通道 U-Net | "正确的修复模型" | 以 `noisy \| encoded-source \| mask` 作为输入的 U-Net。 |
| SDEdit | "带噪声级别的图生图" | 噪声到时刻 `t`，用新提示词去噪。 |
| InstructPix2Pix | "纯文本编辑" | 在（图像、指令、输出）三元组上微调的扩散。 |
| RePaint | "无需重训练" | 在反向过程中定期重新噪声以减少接缝。 |
| SAM | "分割任何东西" | 通过点击或框生成掩码；与修复配合。 |
| Flux-Kontext | "带上下文的编辑" | Flux 变体，接受参考图像 + 指令进行编辑。 |

## 生产笔记：编辑流水线对延迟敏感

用户编辑图像期望着 5 秒以内的往返。1024² 的 30 步 SDXL-Inpaint 在 L4 上是 3-4 秒，加上 SAM 掩码生成（约 200 毫秒）和 VAE 编码/解码（约 500 毫秒合计）。在生产框架中，这是 TTFT 约束而非吞吐量约束——批处理 1，低并发，最小化每个阶段：

- **SAM-H 是慢的那个。** SAM-H 在 1024² 上约 200 毫秒；SAM-ViT-B 约 40 毫秒，质量略有损失。SAM 2（视频）增加时间开销；不要用于单图像编辑。
- **尽可能跳过编码。** `pipe.image_processor.preprocess(img)` 编码到潜空间。如果你有上一次生成的潜空间（迭代编辑 UI 中的典型情况），通过 `latents=...` 直接传递以跳过一次 VAE 编码。
- **掩码膨胀对吞吐量也很重要。** 小掩码意味着大部分 U-Net 前向传递被浪费（未掩码像素无论如何都被限制）。`diffusers` 的 `StableDiffusionInpaintPipeline` 运行完整的 U-Net；只有 9 通道正确的修复变体利用掩码计算。
- **Flux-Kontext 是 2025 年的答案。** 对（源图像、指令）的单次前向传递——没有单独的掩码，没有 SDEdit 噪声扫描。在 H100 上约 1.5 秒完成编辑。架构教训：合并阶段。

## 进一步阅读

- [Lugmayr 等 (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — 无需训练的修复。
- [Meng 等 (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit。
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — 文本指令编辑。
- [Kirillov 等 (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM，掩码来源。
- [Ravi 等 (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — 视频 SAM。
- [Hertz 等 (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — 注意力级编辑。
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — 2024 工具。