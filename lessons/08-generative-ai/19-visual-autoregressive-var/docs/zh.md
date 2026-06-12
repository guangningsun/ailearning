# 视觉自回归建模（VAR）：下一尺度预测

> 扩散模型在时间上迭代采样（去噪步骤）。VAR 在尺度上迭代采样——它预测一个 1×1 token，然后 2×2，然后 4×4，直到最终分辨率，每一尺度都以前一尺度为条件。2024 年的论文表明 VAR 匹配 GPT 风格的缩放定律用于图像生成，并在相同计算预算下超越 DiT。本节构建其核心机制。

**类型：** 动手构建
**语言：** Python（配合 PyTorch）
**前置条件：** 阶段 7 第 03 节（多头注意力）、阶段 8 第 06 节（DDPM）
**时间：** 约 90 分钟

## 问题

自回归生成主导了语言建模，因为它具有可预测的缩放性：更多计算、更多参数、更低困惑度、更好输出。图像生成在 2024 年之前有过两次主要 AR 尝试：PixelRNN/PixelCNN（逐像素）和 DALL-E 1 /Parti / MuseGAN（逐 VQ-VAE code 的 token）。

两者都受困于生成顺序问题。像素和 token 排列在 2D 网格中，但 AR 模型必须以 1D 光栅顺序访问它们。早期的角落像素无法知道图像最终会变成什么样。生成质量比 GPT 在文本上的扩展性差得多，从未达到匹配计算量下扩散模型的质量。

VAR 通过改变生成内容来解决生成顺序问题。VAR 不是在空间中逐个预测图像 token，而是以递增分辨率预测整个图像。第 1 步：预测一个 1×1 token（整体图像"摘要"）。第 2 步：预测一个 2×2 token 网格（较粗糙的特征）。第 3 步：预测一个 4×4 网格。第 K 步：预测最终的 (H/8)×(W/8) 网格。

每一尺度都 attend 到所有前序尺度（按"尺度顺序"因果），在同一尺度内并行。顺序问题消失了：整个图像在尺度 k 通过一次 transformer 前向传播生成。

## 概念

### VQ-VAE 多尺度 Tokenizer

VAR 需要一个**多尺度离散 tokenizer**。对于图像 x，它产生一系列递进高分辨率的 token 网格：

```
x -> encoder -> latent f
f -> 在 1x1 上 tokenize：形状为 (1, 1) 的 token 网格 z_1
f -> 在 2x2 上 tokenize：形状为 (2, 2) 的 token 网格 z_2
...
f -> 在 (H/p)x(W/p) 上 tokenize：形状为 (H/p, W/p) 的 token 网格 z_K
```

每个 z_k 使用相同的码本（典型大小 4096-16384）。每一尺度的 tokenize 并不独立——它的训练方式是让每一尺度的残差求和重建 f：

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

这是**残差 VQ** 变体。尺度 k 捕获了尺度 1..k-1 遗漏的内容。解码器接收所有尺度 embedding 的和并生成图像。

多尺度 VQ tokenizer 训练一次（如 VQGAN），然后冻结。所有生成工作都交给上面的自回归模型。

### 下一尺度预测

生成模型是一个 transformer，它看到所有前序尺度的 token 并预测下一尺度的 token。

输入序列结构：
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

位置 embedding 同时编码尺度索引和该尺度内的空间位置。注意力在尺度顺序上是因果的：尺度 k、位置 (i, j) 处的 token 可以 attend 到尺度 1..k 的所有 token，以及尺度 k 本身中按某种尺度内顺序更早出现的 token（VAR 使用固定位置注意力，尺度内无因果性——尺度内的所有位置并行预测）。

训练损失：在每一尺度 k，给定所有前序尺度 token，预测 token z_k。对离散 VQ codes 的交叉熵损失。与 GPT 结构相同，只是"序列"现在是尺度结构化的。

### 生成

推理时：
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens 并行
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens 并行
...
decode: f = scales 1..K 的 embed-and-upsample 之和
image = VAE_decoder(f)
```

对于 K = 10 个尺度，生成需要 10 次 transformer 前向传播。每次前向传播并行产生整个尺度——尺度内没有逐 token 自回归。对于 256×256 图像，这大约是 10 次传递 vs DiT 的 28-50 次。

### 为什么下一尺度优于下一 token

三个结构优势：
1. **从粗到细符合自然图像统计。** 人类视觉感知和图像数据集都呈现尺度依赖规律性：低频结构稳定且可预测；高频细节以低频内容为条件。下一尺度预测利用了这一点。
2. **尺度内并行生成。** 与 GPT 风格的 token AR 不同，VAR 在一次步骤中产生一个尺度的所有 token。有效生成长度是对数而非线性的。
3. **无生成顺序偏差。** 尺度 k 的 token 可以看到尺度 k-1 的全部；没有"左边"或"上边"偏差，迫使早期 token 在后期上下文可用之前就做出决定。

### 缩放定律

Tian 等人证明了 VAR 遵循 FID 在 ImageNet 上的幂律缩放曲线——就像 GPT 对困惑度的关系。参数或计算翻倍，可靠地使误差减半。这是第一个以与语言模型一样清晰的方式呈现这种缩放行为的图像生成模型。结果是 VAR 的缩放预测可以从计算中预测，而不是每个架构的经验猜测。

### 与扩散的关系

VAR 和扩散共享相同的数据压缩故事：都将生成问题分解为一系列更容易的子问题。

- 扩散：逐步添加噪声，学习撤销一步。
- VAR：逐步添加分辨率，学习预测下一尺度。

它们是穿过这个问题不同轴。两个都产生可处理的条件分布。从经验上看，VAR 推理更快（传递更少，尺度内全并行），在类别条件 ImageNet 上匹配或超越 DiT。文本条件 VAR（VARclip、HART）是一个活跃的研究方向。

## 动手构建

在 `code/main.py` 中你将：
1. 在合成"图像"数据（2D 高斯环）上构建一个微型**多尺度 VQ tokenizer**。
2. 训练一个**VAR 风格 transformer** 来下一尺度预测 token。
3. 通过调用 transformer 4 次（4 个尺度）并解码来采样。
4. 验证尺度顺序训练使尺度内生成并行。

这是一个玩具实现。重点是看到尺度结构化注意力掩码和尺度内并行生成实际工作。

## 交付

本节产出 `outputs/skill-var-tokenizer-designer.md`——一个用于设计多尺度 tokenizer 的 skill：尺度数量、尺度比例、码本大小、残差共享、解码器架构。

## 练习

1. **尺度数量消融。** 用 4、6、8、10 个尺度训练 VAR。测量重建质量与自回归传递次数的关系。更多尺度 = 更细的残差 = 更好的质量但更多传递。

2. **码本大小。** 用码本大小 512、4096、16384 训练 tokenizer。更大的码本带来更好的重建但更难的预测。找到拐点。

3. **尺度内并行检查。** 对于训练好的 VAR，显式测量注意力模式。在尺度 k 内，模型是否 attend 到跨尺度位置但不 intra-scale？验证掩码实现。

4. **VAR vs DiT 缩放。** 在相同的 ImageNet 类别条件任务上，用匹配参数预算（例如 33M、130M、458M）训练 VAR 和 DiT。绘制 FID vs 计算曲线。VAR 应该在每个规模上领先 DiT——在小规模复现论文结果。

5. **文本条件。** 扩展 VAR 以通过 adaLN 接受文本 embedding（CLIP pooled）作为额外条件输入。这就是 HART 配方。在文本对齐采样上 FID 改进了多少？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|-----------------------|
| VAR | "视觉自回归" | 通过在 VQ token 网格金字塔上进行下一尺度预测的图像生成 |
| 下一尺度预测 | "预测较粗的，然后较细的" | 模型以递增分辨率尺度预测 token，以所有前序尺度为条件 |
| 多尺度 VQ tokenizer | "残差 VQ" | VQ-VAE 产生 K 个递增分辨率的 token 网格，解码器求和各尺度 |
| 尺度 k | "金字塔级别 k" | K 个分辨率级别之一，从 k=1 的 1×1 到 k=K 的 (H/p)×(W/p) |
| 尺度内并行 | "每尺度一次前向传播" | 尺度 k 的所有 token 在一次 transformer 传递中预测，非自回归 |
| 跨尺度因果 | "尺度顺序注意力" | 尺度 k 的 token 可以 attend 到尺度 1..k 的全部，但不能 attend 到 k+1..K |
| 残差 VQ | "加性 tokenize" | 每尺度的 token 编码下层尺度留下的残差；解码器求和各尺度 embedding |
| VAR 缩放定律 | "图像 GPT 缩放" | FID 在计算中遵循可预测的幂定律，类似于语言模型的困惑度 |
| HART | "混合 VAR + 文本" | 结合 MaskGIT 风格迭代解码与 VAR 尺度结构的文本条件 VAR 变体 |
| 尺度位置 embedding | "（尺度，行，列）三元组" | 位置编码同时携带尺度索引和该尺度内的空间坐标 |

## 延伸阅读

- [Tian 等人，2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) — VAR 论文，权威参考
- [Peebles 和 Xie，2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) — DiT，扩散比较基线
- [Esser 等人，2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) — VQGAN，VAR 多尺度 tokenizer 延伸的 tokenizer 家族
- [van den Oord 等人，2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) — VQ-VAE，离散图像 tokenize 的基础
- [Tang 等人，2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) — 文本条件 VAR
