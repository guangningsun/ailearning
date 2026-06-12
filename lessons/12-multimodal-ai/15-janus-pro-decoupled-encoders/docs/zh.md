# Janus-Pro：统一多模态模型的解耦编码器

> 统一多模态模型有一个无法回避的张力。理解需要语义特征——SigLIP 或 DINOv2 输出的向量富含概念级信息。生成需要重建友好的编码——VQ Token 可以干净地组合回清晰的像素。这两个目标在单一编码器中是不兼容的。Janus（DeepSeek，2024年10月）和 Janus-Pro（DeepSeek，2025年1月）认为解决办法是停止尝试：解耦两个编码器。在任务间共享 transformer 主干，但理解路由到 SigLIP，生成路由到 VQ 分词器。在 7B 规模下，Janus-Pro 在 GenEval 上击败 DALL-E 3，同时在 MMMU 上匹配 LLaVA。本文将解读为什么两个编码器能在单一编码器失败的地方成功。

**类型：** 构建型
**语言：** Python（标准库，双编码器路由 + 共享主体信号）
**前置条件：** 阶段 12 · 13（Transfusion）、阶段 12 · 14（Show-o）
**时间：** 约 120 分钟

## 学习目标

- 解释为什么单一共享编码器会在理解或生成质量上做出妥协。
- 描述 Janus-Pro 的路由：SigLIP 特征用于输入侧的理解，VQ Token 用于输入和输出的生成。
- 追踪数据混合的缩放——这使得 Janus-Pro 在 Janus 失败的地方成功。
- 对比解耦（Janus-Pro）、耦合连续（Transfusion）和耦合离散（Show-o）架构。

## 问题

统一模型在理解和生成之间共享 transformer 主干。此前的尝试（Chameleon、Show-o、Transfusion）都对两个方向使用一个视觉分词器。这个分词器是一种妥协：

- 针对重建优化（生成）：VQ-VAE 捕获细粒度像素细节，但产生的 Token 语义连贯性弱。
- 针对语义优化（理解）：SigLIP 嵌入将"猫"图像聚集在"猫"Token 附近，但不允许好的重建。

Show-o 和 Transfusion 为此付出了可见的质量税，一个方向上。Janus-Pro 问：当任务有不同的需求时，为什么非要一个分词器？

## 概念

### 解耦视觉编码

Janus-Pro 的架构分离了两个编码器：

- 理解路径。输入图像 → SigLIP-SO400m → 2 层 MLP → transformer 主干。
- 生成路径。输入图像（如果以现有图像为条件）→ VQ 分词器 → Token ID → transformer 主干。
- 输出生成。Transformer 预测的图像 Token → VQ 解码器 → 像素。

Transformer 主干是共享的。主干的上游和下游都是任务专属的。

输入通过提示格式来区分：`<understand>` 标签路由到 SigLIP；`<generate>` 路由到 VQ。或者路由来自任务隐式确定。

### 为什么这有效

理解损失获得 SigLIP 特征，这些是 CLIP 风格的预训练调优过的语义相似性。模型的感知基准测试比 Show-o / Transfusion 有所提升，因为输入特征对任务来说更好。

生成损失获得 VQ Token，这些是分词器调优过用于重建的。图像质量比 Show-o 有所提升，因为 VQ 编码可以干净地组合回像素。

共享的 transformer 主干看到两种输入分布（SigLIP 和 VQ），并学会与两者一起工作。论点是：足够的数据 + 足够的参数，主干吸收切换。

### 数据缩放——Janus vs Janus-Pro

Janus（原始，arXiv 2410.13848）引入了解耦，但在小规模（1.3B 参数，有限数据）。Janus-Pro（arXiv 2501.17811）做了扩展：

- 7B 参数（对比 1.3B）。
- 第一阶段（对齐）90M 图像-文本对（对比 72M）。
- 第二阶段（统一）72M（对比 26M）。
- 第三阶段增加了 200k 图像生成指令样本。

结果是：Janus-Pro-7B 在 MMMU 上匹配 LLaVA（60.3 vs ~58），在 GenEval 上击败 DALL-E 3（0.80 vs 0.67）。一个开放模型，在统一光谱的两端都有竞争力。

### JanusFlow——矫正流变体

JanusFlow（arXiv 2411.07975）将 VQ 生成路径替换为矫正流生成路径（连续的）。分割变成了 SigLIP-理解 + 矫正流-生成。质量天花板进一步提升。架构保持解耦编码器共享主干。

### 共享主干的职责

Transformer 主干处理统一序列，但有两种输入分布。它的职责是：

- 对于理解：消费 SigLIP 特征 + 文本 Token → 自回归发出文本。
- 对于生成：消费文本 Token +（可选图像 VQ Token）→ 自回归发出图像 VQ Token。

主干在每个 block 没有模态专属权重。它是你期望在 Qwen 或 Llama 中找到的文本风格 transformer，加上两个输入适配器。

有趣的是，这意味着 Janus-Pro 的主干可以从预训练 LLM 初始化。Janus-Pro 确实从 DeepSeek-MoE-7B 初始化。那个选择很重要：LLM 贡献了纯从零开始的统一模型难以达到的推理能力。

### 与 InternVL-U 对比

InternVL-U（第 12.10 课）是 2026 年的后续。它结合了：

- 原生多模态预训练（InternVL3 主干）。
- 解耦编码器路由（SigLIP 输入，VQ + 扩散头输出）。
- 统一理解 + 生成 + 编辑。

InternVL-U 将 Janus-Pro 的架构选择吸收到一个更大的框架中。解耦编码器思想现在是大规模统一模型的默认选择。

### 局限性

解耦编码器增加了架构复杂性。需要训练两个分词器、维护两条输入路径、两套失败模式。对于不需要生成的产品，Janus-Pro 是过度工程——选择 LLaVA 家族的理解模型。

对于不需要理解的产品，Janus-Pro 是大材小用——选择 Stable Diffusion 3 / Flux 模型。

对于两者都需要的产品，Janus-Pro 现在是参考开放架构。

## 使用它

`code/main.py` 模拟 Janus-Pro 路由：

- 两个模拟编码器：SigLIP 风格（产生 256 维语义向量）和 VQ 风格（产生整数编码）。
- 一个提示符路由器，根据任务标签选择编码器。
- 一个共享主体（替代品），处理 Token 序列，不管它们是由哪个编码器产生的。
- 从第一阶段（对齐）到第三阶段（指令微调）加权采样调度的切换。

为三个示例打印路由路径：图像 QA、T2I、图像编辑。

## 交付它

本课产出 `outputs/skill-decoupled-encoder-picker.md`。给定一个产品需要在前沿附近的质量上实现统一生成 + 理解，它选择 Janus-Pro、JanusFlow 或 InternVL-U，并给出具体的数据规模建议。

## 练习

1. Janus-Pro-7B 在 GenEval 上击败 DALL-E 3。解释为什么 7B 开放模型可以在生成上匹配前沿专有模型，但在理解上不行。

2. 实现一个路由器函数：给定提示文本，分类为 `understand` 或 `generate`。如何处理"描述然后画出来"这样的歧义提示？

3. JanusFlow 用矫正流替换 VQ 路径。现在 transformer 主干输出什么，损失有什么变化？

4. 为 Janus-Pro 架构提议第四个任务，只需要再解耦一个编码器。例子：图像分割（DINO 风格）、深度（MiDaS 风格）。

5. 阅读 Janus-Pro 第 4.2 节关于数据缩放。哪个数据阶段对 T2I 质量收益贡献最大（对比 Janus）？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 解耦编码 | "两个视觉编码器" | 每个方向一个单独的分词器或编码器：语义用于理解，重建用于生成 |
| 共享主体 | "一个 transformer" | 单一 transformer 处理任一编码器的输出；没有模态专属权重 |
| SigLIP 用于理解 | "语义特征" | CLIP 家族的视觉塔，提供丰富的概念特征但重建能力差 |
| VQ 用于生成 | "重建编码" | 矢量量化的 Token，可以干净地解码回像素 |
| JanusFlow | "矫正流变体" | Janus-Pro，用连续流匹配生成头替代 VQ |
| 路由标签 | "任务标签" | 选择输入编码器的提示标记（`<understand>` / `<generate>`） |

## 延伸阅读

- [Wu 等 — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen 等 — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma 等 — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong 等 — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)