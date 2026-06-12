# Chameleon 与早期融合 Token-Only 多模态模型

> 到目前为止我们见过的每一个 VLM 都将图像和文本分开。视觉 token 来自视觉编码器，流入投影器，然后在 LLM 内部与文本相遇。视觉和文本词汇表从不重叠。Chameleon（Meta，2024 年 5 月）问：如果它们重叠呢？训练一个 VQ-VAE，将图像转换为来自共享词汇表的离散 token 序列。每个多模态文档现在是一个序列——文本 token 和图像 token 交错，一个单一自回归 loss。副作用：模型可以生成混合模态输出——在单次推理调用中交替输出文本和图像 token。本课解读早期融合论点并从头构建一个玩具版本。

**类型：** 构建型
**语言：** Python（标准库、VQ-VAE 分词器 + 交错解码器）
**前置条件：** 阶段 12 · 05、阶段 8（生成式 AI）
**时间：** 约 180 分钟

## 学习目标

- 解释共享词汇表 + 单一 loss 如何改变模型的能力。
- 描述 VQ-VAE 如何将图像分词为与 transformer 的 next-token 目标兼容的离散序列。
- 说出 Chameleon 的训练稳定性技巧：QK-Norm、dropout 位置、LayerNorm 顺序。
- 比较 Chameleon 与 BLIP-2 的 Q-Former 方法，并描述何时选择其中之一。

## 问题

基于适配器的 VLM（LLaVA、BLIP-2、Qwen-VL）将文本和图像视为两件不同的事。文本 token 通过 `embed(text_token)`；图像通过 `visual_encoder(image) → projector → ... pseudo_tokens`。模型有两条输入路径，在中间某处合并。

三个后果：

1. LLM 只能消费图像，不能发出它们。输出只能是文本。
2. 混合模态文档（段落和图像交替，如一篇文章）很别扭——你要么在模型外部解析多模态输入，要么链式生成。
3. 分布不匹配。视觉 token 和文本 token 活在隐藏空间的不同区域，造成微妙的对齐问题。

Chameleon 拒绝这个前提：图像只是来自共享词汇表的离散 token 序列。在交错的文档上训练，一个 loss，一个自回归解码器，你就免费解锁了混合模态生成。

## 概念

### VQ-VAE 作为图像分词器

分词器是一个向量量化变分自编码器。架构：

- 编码器：CNN + ViT，将图像映射到空间特征图，比如 32x32 个 dim 256 的特征。
- 码本：K 个向量的可学习词汇表（Chameleon 使用 8192），dim 也是 256。
- 量化：对每个空间特征，通过 L2 距离查找最近的码本条目。用整数索引替换连续特征。
- 解码器：CNN，将量化特征还原为像素。

训练：VAE 重构 loss + 承诺 loss + 码本 loss。码本索引构成图像的离散字母表。

对于 Chameleon：一张图像变为 32*32 = 1024 个 token，来自 8192 的词汇表。与文本 token（来自 LLM 的 BPE 词汇表，比如 32000）拼接。最终词汇表：40192。Transformer 看到一个序列，一个 loss。

### 共享词汇表

Chameleon 的词汇表将文本 token、图像 token 和模态分隔符组合在一起。每个 token 有一个单一 ID。输入 embedding 层将每个 ID 映射到 D 维隐藏向量。输出投影将隐藏向量映射回词汇表 logit。Softmax 选择下一个 token，无论什么模态。

分隔符很重要：`<image>` 和 `</image>` 标签包围图像 token 序列。在生成时，如果模型输出 `<image>`，下游软件知道接下来 1024 个 token 是 VQ 索引，发送到解码器进行像素渲染。

### 混合模态生成

推理是共享词汇表中的 next-token 预测。示例提示："画一只猫并描述它。"Chameleon 输出：

```
<image> 4821 1029 2891 ... (1024 个图像 token) </image>
这只猫是橙色的，坐在窗台上...
```

模型自主选择顺序——可能先生成图像再文本，或先文本再图像，或交错。相同的解码器，相同的 loss。

对比基于适配器的 VLM，其生成只能是文本。Chameleon 重开了模型输出模态的问题。

### 训练稳定性——QK-Norm、dropout、LayerNorm 顺序

早期融合训练在大规模下不稳定。Chameleon 的论文记录了三个技巧：

- QK-Norm。在 attention 内部对 query 和 key 投影应用 LayerNorm，在点积之前。防止 logit 幅度在大深度上爆炸。被多个 2024 年后的大型模型使用。
- Dropout 位置。在每个 residual-add 后 dropout，而不仅是 attention 和 MLP 之后。当图像 token 的梯度可能占主导时需要更多正则化。
- LayerNorm 顺序。在残差分支上使用 Pre-LN（标准），并在最后一个 block 的 skip 连接上额外加一个 LN。稳定化最终层的梯度流。

没有这些技巧，34B 参数的 Chameleon 训练在多个检查点处发散。有了它们，它收敛了。训练配方与架构本身一样是贡献。

### 分词器的重构上限

VQ-VAE 是有损的。在 8192 个码本条目和每张 512x512 图像 1024 token 时，重构 PSNR 上限约为 26-28 dB。这对于可识别的图像生成足够，但明显比连续空间扩散差（Stable Diffusion 3 达到 32+ dB）。

分词器是瓶颈。更好的分词器（MAGVIT-v2、IBQ、SBER-MoVQGAN）提升上限。Emu3（第 12.12 课）仅凭更好的分词器就达到了 SDXL 质量的生成。

### Chameleon vs BLIP-2 / LLaVA

Chameleon（早期融合，共享词汇表）：
- 一个 loss，一个解码器。
- 生成混合模态输出。
- 分词器是质量上限。
- 昂贵：推理路径上每个生成的图像需要 VQ-VAE 解码器。

BLIP-2 / LLaVA（晚期融合，单独塔）：
- 视觉输入，文本输出。
- 重用预训练 LLM。
- 理解无分词器瓶颈。
- 便宜：单次前向传播。

按任务选择。如果需要图像生成，Chameleon 系列。如果只需要理解，适配器 VLM 更简单，重用更多预训练计算。

### Fuyu 和 AnyGPT

Fuyu（Adept，2023）是相关方法：完全跳过单独的视觉编码器，将原始图像 patch 作为 token 通过 LLM 的输入投影输入，就像它们是 token 一样，不需要分词器。比 Chameleon 更简单，但失去了共享词汇表输出生成。

AnyGPT（Zhan 等，2024）将 Chameleon 扩展到四种模态：文本、图像、语音、音乐。每个都使用相同的 VQ-VAE 技巧，共享 transformer。任意到任意生成。在第 12.16 课有更多介绍。

## 使用它

`code/main.py` 构建一个玩具端到端早期融合模型：

- 一个小型 VQ-VAE 风格量化器，将 8x8 patch 映射到码本索引（K=16）。
- 一个共享词汇表（文本 id 0..31）+（图像 id 32..47）+（分隔符 48, 49）。
- 一个玩具自回归解码器（二元语法表），在合成 caption + 图像 token 序列上训练。
- 给定提示时发出交替文本 + 图像 token 的采样循环。

代码故意保持 transformer 很小（二元语法），这样你可以从头到尾追踪信号流。

## 交付它

本课产出 `outputs/skill-tokenizer-vs-adapter-picker.md`。给定产品规格（仅理解 vs 理解 + 生成、所需图像质量、成本预算），它在 Chameleon 系列（早期融合）和 LLaVA 系列（晚期融合）之间选择，并用定量经验法则论证理由。

## 练习

1. Chameleon 使用 K=8192 个码本条目和每张 512x512 图像 1024 token。估算相比 24 位 RGB 图像的压缩比。它有损吗？损耗多大？

2. 一张 4K 图像（3840x2160）在相同 VQ-VAE 密度下产生多少图像 token？Chameleon 风格的模型能在单次推理调用中生成 4K 图像吗？什么最先出问题——上下文、分词器质量，还是 KV 缓存？

3. 在纯 Python 中实现 QK-Norm。给定 64 维 query 和 key，展示 LayerNorm 前后的点积。为什么在大深度上幅度控制很重要？

4. 阅读 Chameleon 第 2.3 节关于训练稳定性。描述论文在 34B 无 QK-Norm 时观察到的确切失败模式。"norm 爆炸"的特征是什么？

5. 将玩具解码器扩展为给定纯文本提示时发出混合模态响应。测量模型在训练数据分布 60% 文本优先 / 40% 图像优先下选择图像优先 vs 文本优先的频率。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|------------------------|
| 早期融合 | "统一 token" | 图像从第 1 步起就转换为与 transformer 词汇表共享的离散 token |
| VQ-VAE | "图像分词器" | CNN + ViT + 码本，将图像映射到 transformer 可以预测的整数索引 |
| 共享词汇表 | "一个字典" | 覆盖文本 + 图像 + 模态分隔符的单一 token ID 空间 |
| QK-Norm | "注意力稳定器" | LayerNorm 应用在 query 和 key 上，在它们点积之前，防止 norm 爆炸 |
| 混合模态生成 | "文本 + 图像输出" | 在单次传递中自主生成交错文本和图像 token 的推理 |
| 码本大小 | "K 个条目" | VQ-VAE 可以量化的离散向量数量；以压缩换保真度 |
| 分词器上限 | "重构极限" | 解码 VQ token 可达到的最佳 PSNR；决定模型的图像质量 |

## 延伸阅读

- [Chameleon 团队 — Chameleon: 混合模态早期融合基础模型 (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan 等 — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu 等 — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan 等 — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B 博客 (adept.ai)](https://www.adept.ai/blog/fuyu-8b)