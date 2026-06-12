# DualPipe 并行性

> DeepSeek-V3 在 2,048 张 H800 GPU 上训练，MoE 专家分散在各个节点之间。跨节点专家 all-to-all 通信每消耗 1 GPU-小时的算力就需要 1 GPU-小时的通信。GPU 一半时间在闲置。DualPipe（DeepSeek，2024 年 12 月）是一种双向流水线，将前向和反向计算与它们触发的 all-to-all 通信重叠。气泡消除，吞吐量上升，而保持两份模型参数副本（这就是"dual"一词的由来）在 Expert Parallelism 已经将专家分散到各 rank 的情况下很便宜。本节是 Learn 类型的 walkthrough，讲解 DualPipe 实际做什么，以及为什么 Sea AI Lab 的 DualPipeV 改进以稍微更大的气泡为代价去掉了 2 倍参数成本。

**类型：** 学习型
**语言：** Python（标准库、调度模拟器）
**前置条件：** 阶段 10 · 05（分布式训练、FSDP、DeepSpeed）、阶段 10 · 14（开放模型架构和 MoE）
**时间：** 约 60 分钟

## 学习目标

- 说出 DualPipe 前向-反向 chunk 的四个组成部分，以及为什么每个部分都有自己的重叠窗口。
- 解释大规模下的流水线气泡问题，以及"无气泡"在实践与营销中的实际含义。
- 手工追踪 8 个 PP rank 和 16 个微批次的 DualPipe 调度，确认前向和反向流填充彼此的空闲槽。
- 陈述 DualPipeV（Sea AI Lab，2025）的权衡：当 Expert Parallelism 不活跃时，以稍微更大的气泡为代价去掉 2 倍参数复制。

## 问题

在 2k H800 GPU 上训练 671B MoE 模型会遇到三个叠加瓶颈：

1. **内存压力。** 每张 GPU 持有模型的一个分片。序列 8k、61 层、128 个头下的激活内存非常大。
2. **流水线气泡。** 传统流水线并行（GPipe，1F1B）让 GPU 在等待其阶段的输入或梯度时闲置。在 8 个阶段下，即使有 1F1B 调度，大约 12% 的 GPU 时间可能是气泡。
3. **跨节点 all-to-all。** MoE 配合专家并行将专家分散到各个节点。每次前向传递都会触发一次 all-to-all 来将 token 分派给它们的专家，再触发一次来合并。在 2k GPU 上这很容易变成 1:1 的算力与通信比。

这些问题各有独立的解决方案：梯度检查点解决内存、Zero Bubble（Sea AI Lab，2023）解决流水线气泡、专家并行通信内核解决 all-to-all。DualPipe 所做的是让它们协同工作。调度将计算和通信重叠在单个前向-反向 chunk 内，同时从流水线两端注入微批次，并用 resulting schedule 将 all-to-all 隐藏在其中的计算窗口内。

报告的结果：流水线气泡近乎消除，在 DeepSeek-V3 的 14.8T token 训练运行中 GPU 利用率超过 95%。

## 概念

### 流水线并行回顾

将 N 层模型分割到 P 个设备上。设备 `i` 持有层 `i * N/P .. (i+1) * N/P - 1`。一个微批次向前流经设备 0 到 P-1，然后反向从 P-1 到 0。每个设备只能在前一个设备发送其输出时才能开始其前向阶段，只能在下游设备发送上游梯度时才能开始反向阶段。

GPipe（Huang 等人，2019）一次调度一个微批次，浪费了大部分 GPU 时间。1F1B（Narayanan 等人，2021）交错多个微批次的前向和反向传递。Zero Bubble（Qi 等人，2023）将反向传递分成两部分——反向-for-输入（B）和反向-for-权重（W）——并调度它们填充气泡。Zero Bubble 之后，流水线几乎紧耦合。

DualPipe 是下一步。它在此基础上添加了两个想法：

### 想法 1：chunk 分解

每个前向 chunk 被分成四个组成部分：

- **注意力。** Q/K/V 投影、注意力、输出投影。
- **All-to-all 分派。** 跨节点通信，将 token 发送给他们各自的专家。
- **MLP。** MoE 专家计算。
- **All-to-all 合并。** 跨节点通信，将专家输出取回。

反向 chunk 添加每个部分的梯度版本。DualPipe 调度它们，使 all-to-all 分派与下一个 chunk 的注意力计算并行发生，all-to-all 合并与下一个 chunk 的 MLP 计算并行发生。

### 想法 2：双向调度

大多数流水线调度从阶段 0 注入微批次并流向阶段 P-1。DualPipe 从两端同时注入微批次。阶段 0 看到源自那里的前向微批次；阶段 P-1 也看到源自那里的前向微批次。两股流在中间相遇。

为了做到这一点，设备 `i` 必须同时持有早期流水线层 `i` 和晚期流水线层 `P - 1 - i`。这就是 DualPipe 中"dual"的含义：每个设备保留它需要服务的模型层副本（每个方向一个）。在 DeepSeek-V3 的规模下，这是 2 倍的参数复制成本。这是可以承受的，因为 Expert Parallelism 已经将 MoE 专家分散得很稀疏，复制非专家层两次是小钱。

关键的是，一个方向的前向流和另一个方向的反向流正好在单方向调度中会出现气泡的地方重叠。气泡消失了。

### 手工追踪的调度

考虑 P = 4 个 rank，8 个微批次，分成 4 个前向 / 4 个反向。时间从左到右；行是设备 rank。

```
           时间 →
	rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
	rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
	rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
	rank 3:           F1  F2/F5R F3/F6R    ...
```

读取 "F4/F5R" 符号：rank 1 在同一时间槽运行微批次 4 的前向（在流水线中从左到右）和微批次 5 的前向（从右到左）。这就是"双向"在操作上的含义。

在 rank 2，跨流重叠得更早；在 rank 0 和 P-1 重叠得最晚。在调度的稳定中间阶段，每个 rank 运行一个方向的前向与另一个方向的反向重叠。计算是忙碌的。前向传递的 all-to-all 分派隐藏在后向计算内部。前向传递的 all-to-all 合并隐藏在前向计算内部。气泡被挤出去了。

### 气泡核算

标准 1F1B 流水线气泡（每个 rank 浪费的时间）：

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble 改进将其降低但不到零。DualPipe 在稳定阶段，如果微批次数量可被 2 倍流水线深度整除，气泡为零。在稳定阶段之外（预热和冷却），有一些气泡，但它不随微批次数量增长——这是论文强调的一个关键特性。

在营销术语中："无气泡"。在技术术语中：气泡不随微批次数量增长。Sea AI Lab 的后续分析（DualPipeV / Cut-in-half）显示，只有当 Expert Parallelism 不是瓶颈时才能完全零气泡；在 EP 驱动的 all-to-all 下，某些调度妥协总是存在的。

### DualPipeV — 改进

Sea AI Lab（2025 年）观察到，当 EP 通信重叠不是重点时，2 倍参数复制是浪费的。他们的 DualPipeV 调度将双向注入折叠成一个"V 形"调度，只运行一份参数副本。气泡比 DualPipe 稍大，但内存节省很可观。DeepSeek 在他们的开源 DualPipe 实现中采用了 DualPipeV 作为 EP 关闭模式。

权衡：

| 特性 | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| 每设备参数副本 | 2 | 1 | 1 | 1 |
| 气泡 vs 微批次 | 常数 | 轻微增长 | 增长 | 增长 |
| 计算-通信重叠 | 完整 | 部分 | 最小 | 部分 |
| 何时使用 | EP 重度 MoE | 密集或 EP-轻量 | 基线 | 任何流水线 |

### 对于 14.8T token 运行意味着什么

DeepSeek-V3 的预训练在约 2.8M GPU-小时内消耗了 14.8T token。使用朴素的 1F1B，他们会失去 12-15% 的时间来处理流水线气泡——340-420K GPU-小时，足够训练一个完整的 70B 模型。DualPipe 回收了大部分。没有内部日志，直接量化贡献是困难的，但论文中的声称是训练期间平均 GPU 利用率超过 95%。

对于较小的运行（不到 1k GPU），DualPipe 是过度设计的——流水线气泡相对于总成本较小，密集模型训练很少遇到 all-to-all 瓶颈。对于数千 GPU 规模的前沿 MoE 训练，它是有效的要求。

### 它在堆栈中的位置

- 补充 **FSDP**（阶段 10 · 05）。FSDP 跨 rank 分片模型参数；DualPipe 跨 rank 调度计算。它们结合使用。
- 与 **ZeRO-3** 梯度分片兼容。两份副本复制的账本需要与 ZeRO 的分片梯度配合。
- 需要针对特定集群拓扑调优的**自定义 all-to-all 内核**。DeepSeek 的开源内核是参考实现。

## 使用它

`code/main.py` 是一个流水线调度模拟器。它接受 `(P, n_micro_batches, schedule)` 并为 1F1B、Zero Bubble、DualPipe 和 DualPipeV 打印稳定阶段利用率。这是一个教学工具——数字与论文中的定性声称相匹配，它们不是关于生产实测加速的声称。

模拟器的价值：用不同的 P 和微批次计数运行它，观察气泡分数如何为 1F1B 增长，但不为 DualPipe 增长。

真实训练运行的集成注意事项：

- 选择能干净地整除微批次计数的流水线并行深度。
- 确保你的专家并行网格支持双向 all-to-all。DeepSeek 的内核是参考。
- 期望在调度本身燃烧一周调试时间。账本很繁琐。
- 按 rank 监控 GPU 利用率，而不仅仅是总量。DualPipe 的好处来自收紧落后者。

## 交付它

本课产出 `outputs/skill-dualpipe-planner.md`。给定训练集群规范（GPU 数量、拓扑、互联、模型形状），它推荐流水线并行策略、要使用的调度算法，以及目标规模下预期的气泡比例。

## 练习

1. 在 `(P=8, micro_batches=16, schedule=dualpipe)` 和 `(P=8, micro_batches=16, schedule=1f1b)` 上运行 `code/main.py`。计算 GPU 利用率差异，并将其表示为每训练百万 token 回收的 GPU-小时数。

2. 手工绘制 `(P=4, micro_batches=8, schedule=dualpipe)` 的调度表。用微批次 ID 和方向标记每个时间槽。找出第一个没有气泡的时间槽。

3. 阅读 DeepSeek-V3 技术报告（图 5，arXiv:2412.19437）。识别 DualPipe 前向 chunk 内 all-to-all 分派的重叠窗口。解释计算调度如何将其隐藏。

4. 计算 DualPipe 对于 P=8 流水线阶段的 70B 密集模型和 P=16 流水线阶段的 671B MoE 模型的 2 倍参数开销。说明为什么 MoE 情况的开销比例更小（大多数参数是专家，在大的 EP 组中分片）。

5. 将 DualPipe 与 Chimera（2021 年的竞争性双向调度器）进行比较。使用论文第 3.4 节作为参考，确定 DualPipe 添加的两个特定属性 Chimera 没有。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| 流水线气泡 | "每个 rank 的空闲时间" | 因为流水线阶段在等待其输入或梯度而浪费的 GPU 周期 |
| 1F1B | "默认流水线调度" | 一个前向 / 一个反向交错调度；DualPipe 超越的基线 |
| Zero Bubble | "Sea AI Lab 2023" | 将反向分成 B（输入梯度）和 W（权重梯度）；几乎完全收紧流水线 |
| DualPipe | "DeepSeek-V3 调度" | 双向流水线 + 计算-通信重叠；气泡不随微批次数量增长 |
| DualPipeV | "减半" | V 形改进，以稍微更大的气泡为代价去掉 2 倍参数复制 |
| Chunk | "流水线工作单元" | 一个微批次通过一个流水线阶段的一次前向或反向传递 |
| All-to-all 分派 | "发送 token 给专家" | 将 token 路由到各自 MoE 专家的跨节点通信 |
| All-to-all 合并 | "取回专家输出" | MLP 之后收集专家输出的跨节点通信 |
| 专家并行（EP） | "专家跨 GPU" | 将 MoE 专家分片到各 rank，使不同 GPU 持有不同专家 |
| 流水线并行（PP） | "层跨 GPU" | 将模型层分片到各 rank；DualPipe 调度的维度 |
| 气泡比例 | "浪费的 GPU 时间" | (bubble_time / total_time)；DualPipe 推向零的比例 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 技术报告（arXiv:2412.19437），第 3.3.2 节和图 5](https://arxiv.org/abs/2412.19437) — 主要 DualPipe 参考
- [DeepSeek — DualPipe GitHub 仓库](https://github.com/deepseek-ai/DualPipe) — 开源参考实现，包括 DualPipeV（Cut-in-half）模式
- [Qi 等人 — 零气泡流水线并行（arXiv:2401.10241，Sea AI Lab 2023）](https://arxiv.org/abs/2401.10241) — Zero Bubble 前辈
- [Sea AI Lab — DualPipe 没有 Dual 可能会更好](https://sail.sea.com/blog/articles/63) — 影响了 DeepSeek EP-off 模式的 DualPipeV 分析
- [Narayanan 等人 — PipeDream / 1F1B（arXiv:1806.03377，2018-2021）](https://arxiv.org/abs/1806.03377) — DualPipe 比较的 1F1B 调度
- [Huang 等人 — GPipe（arXiv:1811.06965，2018）](https://arxiv.org/abs/1811.06965) — 原始流水线并行论文和气泡问题