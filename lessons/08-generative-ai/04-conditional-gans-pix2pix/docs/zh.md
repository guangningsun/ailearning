# 条件 GAN 与 Pix2Pix

> 2014-2017 年间第一个重大突破是控制 GAN 生成的内容。附加一个标签，或一张图像，或一句话。Pix2Pix 实现了图像版本，在窄域图像到图像任务上至今仍能打败所有通用文生图模型。

**类型：** 动手构建
**语言：** Python
**前置条件：** 阶段 8 · 03（GAN）、阶段 4 · 06（U-Net）、阶段 3 · 07（CNN）
**时间：** 约 75 分钟

## 问题

无条件 GAN 采样任意人脸。用于演示尚可，用于生产则毫无用处。你想要的是：*把草图映射为照片*、*把地图映射为航拍图*、*把日间场景映射为夜间*、*给灰度图上色*。在所有这些任务中，你给定一张输入图像 `x`，必须输出语义对应的 `y`。每个 `x` 对应多个合理的 `y`。均方误差会把它们压成一团糊。对抗损失不会，因为"看起来真实"是尖锐的。

条件 GAN（Mirza & Osindero，2014）在 `G` 和 `D` 的输入中都加入了条件 `c`。Pix2Pix（Isola 等，2017）在此基础上做了专门化：条件是完整的输入图像，生成器是 U-Net，判别器是*基于 patch 的分类器*（PatchGAN），损失函数是对抗损失 + L1。这个配方在窄域图像到图像任务上从零训练的文生图模型中胜出，甚至在 2026 年依然如此，因为它在*配对数据*上训练——你拥有恰好所需的信号。

## 概念

![Pix2Pix：U-Net 生成器，PatchGAN 判别器](../assets/pix2pix.svg)

**条件 G。** `G(x, z) → y`。在 Pix2Pix 中，`z` 是 G 内部的 dropout（没有输入噪声——Isola 发现显式噪声会被忽略）。

**条件 D。** `D(x, y) → [0, 1]`。输入是*配对*（条件，输出）。这是关键区别：D 必须判断 `y` 是否与 `x` 一致，而不只是 `y` 看起来是否真实。

**U-Net 生成器。** 编码器-解码器，在瓶颈处有跳跃连接。对于输入和输出共享低级结构（边缘、轮廓）的任务至关重要。没有跳跃连接，高频细节会消失。

**PatchGAN 判别器。** D 不是输出单个真/假分数，而是输出一个 `N×N` 网格，每个单元格判断一个感受野（约 70×70 像素）。取平均。这是一个马尔可夫随机场假设：真实感是局部的。训练更快，参数更少，输出更锐利。

**损失函数。**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1 项稳定训练并推动 G 靠近已知目标。L1 比 L2 给出更锐利的边缘（用的是中位数，不是均值）。Pix2Pix 默认 `λ = 100`。

## CycleGAN——当没有配对数据时

Pix2Pix 需要配对的 `(x, y)` 数据。CycleGAN（Zhu 等，2017）以额外的损失为代价取消了这个要求：*循环一致性损失*。两个生成器 `G: X → Y` 和 `F: Y → X`。训练它们使得 `F(G(x)) ≈ x` 且 `G(F(y)) ≈ y`。这让你可以在没有配对示例的情况下把马变成斑马、把夏天变成冬天。

在 2026 年，非配对图像到图像主要通过扩散模型（ControlNet、IP-Adapter）来完成而不是 CycleGAN，但循环一致性思想几乎出现在每一篇非配对域适应论文中。

## 动手构建

`code/main.py` 在 1 维数据上实现了一个小型条件 GAN。条件 `c` 是一个类别标签（0 或 1）。任务：为给定类别从条件分布中产生样本。

### 第 1 步：将条件附加到 G 和 D 的输入

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

独热编码是最简单的方式。更大的模型使用可学习的 embedding、FiLM 调制或交叉注意力。

### 第 2 步：训练条件模型

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

生成器必须匹配给定条件下真实分布，而不是边缘分布。

### 第 3 步：验证每个类别的输出

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 陷阱

- **条件被忽略。** G 学会边缘化，D 从不惩罚，因为条件信号太弱。修复：更激进地对 D 加条件（早期层，不只是后期），使用投影判别器（Miyato & Koyama 2018）。
- **L1 权重太低。** G 漂移到任意看起来真实的输出，而不是忠于输入的输出。Pix2Pix 类任务从 λ≈100 开始。
- **L1 权重太高。** G 产生模糊输出，因为 L1 仍然是 L_p 范数。训练稳定后逐步降低。
- **D 中的真实标签泄露。** 将 `(x, y)` 连接作为 D 输入，而不只是 `y`。没有这个，D 无法检查一致性。
- **每个类别独立发生模式崩溃。** 每个类别都可能独立崩溃。运行类别条件多样性检查。

## 使用场景

2026 年图像到图像任务现状：

| 任务 | 最佳方案 |
|------|---------------|
| 草图 → 照片，同域，配对数据 | Pix2Pix / Pix2PixHD（仍然快速，仍然锐利） |
| 草图 → 照片，非配对 | 带 Scribble 条件模型的 ControlNet |
| 语义分割 → 照片 | SPADE / GauGAN2 或 SD + ControlNet-Seg |
| 风格迁移 | 带 IP-Adapter 或 LoRA 的扩散模型；GAN 方法已是legacy |
| 深度图 → 照片 | 基于 Stable Diffusion 的 ControlNet-Depth |
| 超分辨率 | Real-ESRGAN（GAN）、ESRGAN-Plus 或 SD-Upscale（扩散） |
| 上色 | ColTran、基于扩散的上色器、或 Pix2Pix-color |
| 日间 → 夜间、季节、天气 | CycleGAN 或基于 ControlNet 的方法 |

当（a）你有数千个配对样本、（b）任务窄且可重复、（c）你需要快速推理时，Pix2Pix 仍然是正确的工具。在通用开放域任务上，扩散模型胜出。

## 交付

保存 `outputs/skill-img2img-chooser.md`。技能接收任务描述、数据可用性（配对 vs 非配对、N 样本）和延迟/质量预算，然后输出：方案（Pix2Pix、CycleGAN、ControlNet 变体、SDXL + IP-Adapter）、训练数据需求、推理成本和评估协议（LPIPS、FID、特定任务指标）。

## 练习

1. **简单。** 修改 `code/main.py` 添加第三个类别。确认 G 仍然将每个类别的噪声映射到正确的模态。
2. **中等。** 在 1 维设置中将 L1 替换为感知风格损失（例如一个小型冻结的 D 作为特征提取器）。这会改变条件分布的锐利度吗？
3. **困难。** 在 1 维设置中设计 CycleGAN：两个分布、两个生成器、循环损失。证明它可以在没有配对数据的情况下学习在两者之间映射。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Conditional GAN | "带标签的 GAN" | G(z, c), D(x, c)。两个网络都看到条件。 |
| Pix2Pix | "图到图 GAN" | 带 U-Net G 和 PatchGAN D + L1 损失的配对条件 GAN。 |
| U-Net | "带跳跃连接的编码器-解码器" | 对称卷积网络；跳跃连接保留高频。 |
| PatchGAN | "局部真实感分类器" | D 输出逐 patch 分数而不是全局分数。 |
| CycleGAN | "非配对图像翻译" | 两个 G + 循环一致性损失；无需配对数据。 |
| SPADE | "GauGAN" | 用语义图规范化中间激活；分割到图像。 |
| FiLM | "特征级线性调制" | 来自条件的逐特征仿射变换；廉价的条件化方式。 |

## 生产注意事项：Pix2Pix 作为延迟受限的基线

当你有配对数据和窄任务（草图 → 渲染、语义图 → 照片、日 → 夜）时，Pix2Pix 的一次推理比扩散模型在延迟上快一个数量级。生产中的对比通常是：

| 路径 | 步数 | L4 单卡上 512² 的典型延迟 |
|------|-------|----------------------------------------|
| Pix2Pix（U-Net 前向） | 1 | 约 30 ms |
| SD-Inpaint 或 SD-Img2Img | 20 | 约 1.2 s |
| SDXL-Turbo Img2Img | 1-4 | 约 0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | 约 3-5 s |

Pix2Pix 在静态批处理上赢得吞吐量（每个请求的 FLOPs 相同）。扩散模型在质量和泛化上胜出。现代策略通常是：为窄任务部署一个类 Pix2Pix 的蒸馏模型作为主方案，用扩散模型处理尾部输入。

## 延伸阅读

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — cGAN 论文。
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix。
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN。
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD。
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN。
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — 投影判别器。