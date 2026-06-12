# Show-o 与离散扩散统一模型

> Transfusion 混合了连续和离散表示。Show-o（Xie 等，2024年8月）则走另一条路：文本 Token 用因果下一个 Token 预测，图像 Token 用掩码离散扩散（遵循 MaskGIT 的精神）。两者都放在一个 transformer 里，配以混合注意力掩码。结果是在一个主干网络、每种模态一个分词器、一个损失公式（下一个 Token 扩展到掩码预测）上统一了 VQA、文本到图像、修复和混合模态生成。本文将走一遍 Show-o 设计——为什么掩码离散扩散是并行的、少步的图像生成器——并与 Transfusion 和 Emu3 对比。

**类型：** 学习型
**语言：** Python（标准库，掩码离散扩散采样器）
**前置条件：** 阶段 12 · 13（Transfusion）
**时间：** 约 120 分钟

## 学习目标

- 解释掩码离散扩散：以统一方式掩码 Token 然后要求 transformer 恢复它们的调度。
- 对比并行图像解码（Show-o、MaskGIT）与自回归图像解码（Chameleon、Emu3）的速度和质量。
- 说出 Show-o 在一个检查点中处理的三个任务：T2I、VQA、图像修复。
- 选择一种掩码调度（余弦、线性、截断）并推理其对样本质量的影响。

## 问题

Transfusion 的双损失训练有效，但动态更复杂——连续扩散损失与离散 NTP 损失处于不同的数值尺度上。平衡损失权重是一次超参数搜索。架构有效但复杂。

Show-o 的回答：保持两种模态都离散（像 Chameleon），但通过掩码离散扩散并行生成图像，而非顺序生成。训练目标变成了单一的掩码 Token 预测，自然地泛化了下一个 Token 预测。

## 概念

### 掩码离散扩散（MaskGIT）

原始的 Chang 等（2022）MaskGIT 技巧很优雅。从一个全掩码图像开始（每个 Token 都是特殊的 `<MASK>` id）。在每一步，并行预测所有被掩码的 Token，然后保留 top-K 最自信的预测，其余重新掩码。经过约 8-16 次迭代，所有 Token 都填充完毕。每步掩码多少 Token 的调度是调优的——余弦调度效果很好。

训练很简单：从 [0, 1] 均匀采样一个掩码比例，应用到图像的 VQ Token 上，训练 transformer 恢复被掩码的部分。正是 BERT 对文本做的事，扩展到图像生成。

### Show-o：一个 transformer，混合掩码

Show-o 将 MaskGIT 放入因果语言模型 transformer。注意力掩码：

- 文本 Token：因果的（标准 LLM）。
- 图像 Token：在图像块内完全双向（这样被掩码的 Token 在预测时可以看见每个其他图像 Token）。
- 文本到图像：文本 attend 到前面的图像，图像 attend 到前面的文本。

训练交替进行：
1. 文本序列上的标准 NTP。
2. T2I 样本：文本 → 图像，有掩码图像 Token，掩码 Token 预测损失。
3. VQA 样本：图像 → 文本，有掩码文本 Token（其实就是 NTP）。

统一损失是对 `<MASK>` Token 的交叉熵，覆盖了文本 NTP（只有最后一个 Token 是"掩码的"）和图像掩码扩散（随机子集被掩码）。

### 并行采样

Show-o 生成一张图像约需 16 步，而非约 1000 步（逐 Token 自回归）或约 20 步（扩散）。每一步，并行预测所有被掩码的 Token；提交 top-K 自信的；重复。

对比：
- Chameleon / Emu3（逐 Token 自回归）：N_tokens 次前向传播，通常每张图像 1024-4096 次。
- Transfusion（连续扩散）：约 20 步，每步一次完整 transformer 传播。
- Show-o（掩码离散扩散）：约 16 步，每步一次完整 transformer 传播。

Show-o 在类似规模模型上比 Chameleon 更快，与 Transfusion 步数大致相当，但每步成本更低（离散词表 logits vs 连续 MSE 损失）。

### 一个检查点中的任务

Show-o 在推理时支持四种任务，由提示格式选择：

- 文本生成：标准自回归文本输出。
- VQA：图像输入，文本输出。
- T2I：文本输入，通过掩码离散扩散输出图像。
- 修复：有些 Token 被掩码的图像，填入。

修复能力来自掩码预测训练的免费附赠。在 VQ Token 网格中掩码一个区域，输入其余部分加文本提示符，预测被掩码的 Token。

### 掩码调度

每步掩码多少 Token 的调度影响质量。Show-o 推荐余弦：

```
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T
```

在第 0 步，所有 Token 被掩码（比例 1.0）。在第 T 步，没有掩码。余弦将质量集中在中段比例，此时预测信息量最大。线性调度也能工作但更快达到 plateau。

### Show-o2

Show-o2（2025 年后续，arXiv 2506.15564）扩展 Show-o：更大的 LLM 基座、更好的分词器、改进了掩码调度。架构模式相同。

### Show-o 处于什么位置

在 2026 年分类学中：

- 离散 Token + NTP：Chameleon、Emu3。简单但推理慢。
- 离散 Token + 掩码扩散：Show-o、MaskGIT、LlamaGen、Muse。并行采样，但通过分词器仍有损失。
- 连续 + 扩散：Transfusion、MMDiT、DiT。最高质量，训练更复杂。
- VLM 中的连续 + 流匹配：JanusFlow、InternVL-U。最新。

按任务选择：当你想在一个开放权重模型中获得 T2I + 修复 + VQA 且速度合理时选 Show-o；当质量至上且能承受双损失管道时选 Transfusion。

## 使用它

`code/main.py` 模拟 Show-o 采样：

- 16 个 VQ Token 的玩具网格。
- 一个模拟"transformer"，基于提示符和当前未掩码的 Token 预测 logits。
- 8 步余弦调度的并行掩码采样。
- 打印中间状态（掩码模式演变）和最终 Token。

运行它，观察掩码一步一步消融。

## 交付它

本课产出 `outputs/skill-unified-gen-model-picker.md`。给定一个产品需要同时具备理解（VQA、图像描述）和生成（T2I、修复）能力，并有开放权重约束，在 Show-o 家族、Transfusion/MMDiT 家族和 Emu3/Chameleon 家族之间选择，并给出具体权衡。

## 练习

1. 掩码离散扩散采样约 16 步。为什么不是 1 步？如果在第 0 步全不掩码，会发生什么？

2. 掩码扩散的修复是免费的。提出一个产品用例（真实的或假设的），其中 Show-o 的修复胜过专用模型。

3. 余弦调度 vs 线性调度：追踪 T=8 时每步未掩码 Token 的数量。哪个更均衡？

4. 512×512 的 Show-o 图像是 1024 个 Token。在词表 K=16384 下，模型发出 1024 * log2(16384) = 14,336 比特（约 1.75 KiB）的数据。Stable Diffusion 输出 512*512*24 比特 = 6,291,456 比特（约 768 KiB）的原始像素。压缩比是多少，它买了什么质量？

5. 阅读 LlamaGen（arXiv:2406.06525）。LlamaGen 的类别条件自回归图像模型与 Show-o 的掩码方法有什么不同？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 掩码离散扩散 | "MaskGIT 风格" | 训练预测被掩码的 Token；在推理时迭代式地为最自信的预测解除掩码 |
| 余弦调度 | "解除掩码调度" | 掩码比例在推理步数上的衰减；将置信度增长集中在中段 |
| 并行解码 | "一次预测所有 Token" | 每一步在一个前向传播中预测完整序列的被掩码 Token，然后提交 top-K |
| 混合注意力 | "因果 + 双向" | 对文本 Token 因果、对图像块内双向的掩码 |
| 修复 | "填入生成" | 以某些 Token 被掩码的图像为条件，预测缺失的 Token；这是训练目标的免费附赠 |
| 提交率 | "每步 top-K" | 每轮迭代宣布多少 Token"完成"；控制推理与质量的权衡 |

## 延伸阅读

- [Xie 等 — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang 等 — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun 等 — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang 等 — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)