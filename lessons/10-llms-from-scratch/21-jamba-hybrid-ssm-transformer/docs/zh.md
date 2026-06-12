# Jamba — 混合 SSM-Transformer

> 状态空间模型（SSM）和 Transformer 各有所求。Transformer 以二次方代价换取质量。SSM 以线性时间推理和常数级内存换取循环，但质量落后。AI21 的 Jamba（2024 年 3 月）和 Jamba 1.5（2024 年 8 月）将两者整合到同一模型中：每 7 层 Mamba 搭配 1 层 Transformer，每隔一层使用 MoE，256k 上下文窗口可塞进单张 80GB GPU。Mamba-3（ICLR 2026）通过复数值状态空间和 MIMO 投影强化了 SSM 侧。本课从端到端解读两种架构，并解释为何在纯 SSM 和纯 Transformer 长上下文尝试均未成功的背景下，混合方案已历经三年 scaling 存活下来。

**类型：** 学习型
**语言：** Python（标准库、层配比计算器）
**前置条件：** 阶段 10 · 14（开放模型架构），阶段 10 · 17（原生稀疏注意力）
**时间：** 约 60 分钟

## 学习目标

- 解释 Jamba 块的三个基本组件 — Transformer 层、Mamba 层、MoE — 以及 1:7:even 的交错排布方案。
- 从高层描述 SSM 的循环是什么样的，以及为何它能实现常数级内存推理。
- 计算 Jamba 模型在 256k 上下文下的 KV cache 占用，并与纯 Transformer 模型进行比较。
- 说出 Mamba-3 的三项创新（指数梯形离散化、复数值状态更新、MIMO）以及各自针对的问题。

## 问题

注意力对序列长度是二次方的。状态空间模型是线性的。这个差异会累积：在 256k token 时，Transformer 的注意力图每头有 65B 个条目；而 SSM 的循环状态大小固定，不随序列长度变化。

纯 SSM 模型（Mamba、Mamba-2）在小规模下匹配 Transformer 的困惑度，但在状态跟踪任务上落后，在某些类别的上下文检索上失败。直觉上：SSM 将历史压缩到固定状态中，当历史很长时，信息会泄漏。注意力精确记住一切，但付出二次方代价。

显而易见的修复：两者都用。在需要精确召回的地方放 Transformer 层。其他地方用 SSM 层。调好比例。Jamba 是第一个将这种混合方案以生产规模落地的模型（52B 总计、12B 激活、256k 上下文、单张 80GB GPU）。Jamba 1.5 将该系列扩展到 398B 总计 / 94B 激活。Mamba-3（ICLR 2026）是当前最好的纯 SSM 基线，混合模型可以围绕它重建。

本课阅读全部三篇论文，形成"如何选择合适比例"的心智模型。

## 概念

### 一页纸的 SSM

状态空间模型通过固定大小的状态 `h` 处理序列 `x_1, ..., x_N`：

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

每一步，状态通过线性动力学 `A` 演化，接收输入 `B x_t`，发出输出 `C h_t`。`A、B、C` 可以学习。注意关键性质：计算 `y_t` 只需要 `h_{t-1}` 和 `x_t`，不需要更早的 `x`。内存是常数级的。推理每个 token 是 O(1)。

建模质量的关键在于 `A` 的结构。S4（Gu 2021）使用了一个高度结构化的矩阵，在训练时可以作为长卷积高效求值。Mamba（Gu, Dao 2023）用数据依赖的 A、B、C（"选择性"部分）取代了固定的 A、B、C。Mamba-2（2024）进一步简化了结构。Mamba-3（2026）在特定位置重新引入复数。

关键性质：对于解码器 LLM，SSM 层是注意力层的直接替代品，用固定大小的每层状态取代了不断增长的 KV cache。

### Jamba 块

Jamba 块按照两个数字交错排列各层：

- `l`：注意力层与 Mamba 层的比例。Jamba 使用 `l = 8`，即每 7 层 Mamba 搭配 1 层 Transformer（7 Mamba + 1 Attention = 每组 8 层）。
- `e`：MoE 的频率。Jamba 使用 `e = 2`，即每隔一层应用 MoE。

块内的层序列：

```
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (| 表示应用 MoE 的位置)
```

每个 Jamba 块是 8 层。4 个块深（32 层总计），得到 28 层 Mamba 和 4 层 Attention。其中 16 层使用 MoE。

### 为什么是 1:7 的比例

AI21 做了消融实验：什么样的注意力与 Mamba 比例能在他们的长上下文评测上给出最好的困惑度每参数比 AND 上下文召回率？

- 注意力过多（1:1）：质量上升，但内存和速度下降。
- 注意力过少（1:15）：内存很好，但上下文检索失败。
- 最佳点：1:7 或 1:8。

直觉：Transformer 层处理精确召回和状态跟踪。Mamba 层处理便宜的批量处理。

### 位置编码

Mamba 层本身是位置感知的（通过循环）。原始基于 Mamba 的混合模型中的注意力层不使用 RoPE — SSM 层提供了位置信息。Jamba 1.5 为注意力层添加了 RoPE，用于更长的上下文泛化，这是基于经验性长上下文评估的事后改进。

### 内存预算

对于 Jamba-1 结构（32 层：28 Mamba + 4 Attention，hidden 4096，32 个注意力头）：

- KV cache（仅限注意力层）：`2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB`（256k BF16）。只有 4 层注意力层贡献。
- SSM 状态：`28 * hidden * state_size` per token 前缀，但这是每层固定大小，不随序列长度缩放。典型 Mamba state 为每个特征 16，hidden 4096：`28 * 4096 * 16 * 2 = 3.7 MB` 总计。

对比同 shape 的纯 Transformer（32 层，相同 hidden，全 MHA 32 头）：`2 * 32 * 32 * 128 * 256k * 2 = 128 GB`（256k BF16）。KV cache 减少 8 倍。即使对比大多数 2024 模型使用的 GQA(8) 基线（`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`），Jamba 的 1:7 混合方案 16 GB 仍然小 2 倍。

这就是 AI21 所说的"256k 上下文塞进单张 80GB GPU"。纯 Transformer 全 MHA 的 KV cache 放不下；即使 GQA 基线也留不出权重和激活的空间；而 Jamba 的可以。

### Mamba-3：2026 年的纯 SSM 基线

Mamba-3（ICLR 2026，arXiv:2603.15569）在纯 SSM 侧引入三项创新：

1. **指数梯形离散化。** 用更具表达力的循环替换 Mamba-2 中的 Euler 方法离散化。在核心循环内部对状态-输入施加类卷积操作，而不是作为对 `x_t` 的外层卷积。

2. **复数值状态更新。** 之前的 Mamba 将状态矩阵从复数（S4）降到实对角（Mamba）再到缩放恒等式（Mamba-2）。Mamba-3 重新引入复数值 — 等价于对状态的数据依赖旋转嵌入。这恢复了之前实值简化所损失的状态跟踪能力。

3. **多输入多输出（MIMO）投影。** 不是每个特征的标量投影，而是使用矩阵值投影。在不增加解码延迟的情况下提升建模能力和推理时硬件利用率。

在 1.5B 参数下，Mamba-3 在平均下游准确率上比 Gated DeltaNet 高 0.6 分；MIMO 变体再增加 1.2 分，总计 1.8 分。在相同 state 大小下，Mamba-3 用一半的 state 匹配 Mamba-2。

Mamba-3 尚未在规模化生产混合模型中落地 — 但它是下一代 Jamba 类模型 SSM 侧的明显候选。

### 何时选择混合架构

混合架构胜出时：

- 上下文足够长，纯 Transformer KV cache 变得难以承受（64k+）。
- 任务混合了短程结构（SSM 擅长）和长程召回（需要 Transformer）。
- 你想部署在单 GPU 内存预算下，而单是 Transformer KV cache 就放不下。

混合架构败下风时：

- 上下文很短（小于 16k）。SSM 开销被浪费；纯 Transformer 就够了。
- 任务需要任意-to-任意注意力（深度推理、多文档交叉引用）。混合模型中注意力层的稀疏性会受伤。
- 你在 scaling 到万亿参数的前沿模型。纯 Transformer + MLA + MoE（DeepSeek-V3 风格）目前正在能力竞赛中领先。

### 竞争格局

| 模型 | 系列 | 规模 | 独特主张 |
|-------|--------|------|-------------|
| Mamba-2 | 纯 SSM | 3B | 线性时间，常数内存 |
| Jamba | 混合 | 52B/12B | 256k 塞进 80GB |
| Jamba 1.5 Large | 混合 | 398B/94B | 企业级长上下文 |
| Mamba-3 | 纯 SSM | 1.5B（论文） | 状态跟踪能力恢复 |
| DeepSeek-V3 | 纯 Transformer + MoE | 671B/37B | 前沿能力 |

2026 年格局：纯 Transformer MoE 主导前沿，但混合架构拥有 256k 及以上上下文的空间。Mamba-3 的状态跟踪能力提升可能推动下一代混合比例降低（更多 SSM，更少注意力）。

## 使用它

`code/main.py` 是混合架构的内存计算器。给定 SSM-Transformer 比例和 hidden-size / layer-count 配置，它计算：

- 目标上下文下的 KV cache。
- SSM 状态内存。
- 一系列模型 shape 下上下文 N 的总内存。

计算器支持：

- 纯 Transformer 基线（KV cache 随 N 增长）。
- Jamba 风格 1:7 混合。
- 纯 SSM（完全没有 KV cache）。

数字直接来自 Jamba-1 和 Jamba-1.5 论文的已发布 shape，并对假设变体进行了外推。

实际部署的集成考量：

- 大多数生产推理服务器（vLLM、SGLang）支持 Jamba 和 Mamba。请检查具体版本。
- 在 256k 上下文下，Jamba 的内存优势体现在并发请求吞吐量上。在相同 VRAM 下你可以塞进更多 Jamba 序列而非 Transformer 序列。
- Mamba-3 作为独立模型尚未在生产中落地 — 研究预览版 1.5B。

## 交付它

本课产出 `outputs/skill-hybrid-picker.md`。给定工作负载规格（上下文长度轮廓、任务混合、内存预算），它推荐纯 Transformer、Jamba 风格混合和纯 SSM 之间的选择，并附有关于内存和质量权衡的明确推理。

## 练习

1. 运行 `code/main.py` 计算 32 层纯 Transformer（hidden 4096，32 头）在 256k 上下文下的 KV cache，以及同 shape 的 Jamba-1 混合模型。验证 AI21 论文所声称的约 8 倍内存减少。

2. 修改计算器以建模 1:3 混合（4 Mamba : 1 Attention）和 1:15 混合（14 Mamba : 1 Attention）。绘制 KV cache vs 比例图。在什么比例下 KV cache 等于 SSM 状态内存？

3. 阅读 Jamba 论文第 3 节（arXiv:2403.19887）。解释为什么 AI21 使用 Mamba-1 而不是 Mamba-2，尽管 Mamba-2 更快。提示：混合消融部分记录了这一点。

4. 计算 Jamba 1.5 Large（398B 总计，94B 激活）每隔一层 MoE 的参数开销。将激活比例与 DeepSeek-V3（37B/671B）比较，并解释为什么 Jamba 的架构将激活比例推得更高。

5. 阅读 Mamba-3 论文第 3 节（arXiv:2603.15569）。用三句话解释为什么复数值状态更新等价于数据依赖的旋转嵌入。将答案与阶段 7 · 课 04 的 RoPE 推导联系起来。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| 状态空间模型（SSM） | "带固定状态的循环" | 一种带有学习循环的层 `h_t = A h_{t-1} + B x_t`；每个 token 常数内存 |
| 选择性 SSM | "Mamba 的技巧" | 数据依赖的 A、B、C 参数，以线性时间给予模型类似门控的选择性 |
| 注意力-Mamba 比例 | "有多少注意力层" | 在 Jamba 中，`l = 8` 表示每 7 层 Mamba 搭配 1 层注意力层 |
| Jamba 块 | "8 层组" | 一个注意力层 + 七个 Mamba 层 + 交替位置的 MoE |
| SSM 状态 | "隐藏缓冲区" | 每层固定大小的状态，取代 Mamba 层的 KV cache |
| 256k 上下文 | "Jamba 的旗舰数字" | Jamba-1 塞进单张 80GB GPU 的序列长度；纯 Transformer 在该规模下做不到 |
| Mamba-3 | "2026 纯 SSM" | 当前最好的纯 SSM 架构，带复数状态和 MIMO；混合模型重建的基线 |
| MIMO | "多输入多输出" | Mamba-3 的创新，使用矩阵值投影而非每个特征的标量 |
| 指数梯形离散化 | "Mamba-3 的循环" | 更有表达力的循环，包含了 Mamba-2 的 Euler 方法离散化 |
| 混合架构 | "混合注意力和 SSM" | 任何交替排列 Transformer 层和 SSM 层的模型；Jamba 是生产原型 |

## 延伸阅读

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — 原始 Jamba 论文，比例消融，256k 上下文主张
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) — 规模化系列，398B/94B 和 12B/52B 公开版本
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) — Jamba 所基于的选择性 SSM 论文
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) — 简化后的结构化状态空间后继者
- [Lahoti et al. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569) — 复数值状态，MIMO，2026 纯 SSM 前沿
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) — S4 论文，SSM 在 LLM 中的谱系起点