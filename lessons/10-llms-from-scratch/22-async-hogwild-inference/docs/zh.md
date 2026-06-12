# 异步与 Hogwild! 推理

> 投机解码（阶段 10 · 15）将一个序列内的 token 并行化。多智能体框架跨整个序列并行化，但需要显式协调（投票、子任务拆分）。Hogwild! 推理（Rodionov et al., arXiv:2504.06261）另辟蹊径：让 N 个相同 LLM 实例并行运行，对抗一个共享的键值缓存。每个 worker 立即看到其他所有 worker 生成的 token。现代推理模型 — QwQ、DeepSeek-R1 — 可以通过该共享缓存自我协调，无需任何微调。该方法尚属实验性，但它开启了一条全新的推理并行化轴线，与投机解码正交。本课用标准库 Python 实现了一个双 worker Hogwild! 模拟器，并解释为何共享缓存协作是从现有模型的推理能力中自然涌现的。

**类型：** 构建型
**语言：** Python（标准库）
**前置条件：** 阶段 10 · 12（推理优化），阶段 10 · 15（投机解码）
**时间：** 约 60 分钟

## 学习目标

- 描述三种常见的并行 LLM 拓扑（投票、子任务、Hogwild!）并指出各自针对的问题。
- 陈述 Hogwild! 的核心设置：多个 worker、一个共享 KV cache、通过自提示产生的-emergent 协调。
- 计算 Hogwild! 的墙上时间加速比，作为 worker 数量 `N`、任务级并行度 `p` 和协调开销 `c` 的函数。
- 在一个玩具问题上实现双 worker Hogwild! 模拟器，并观察涌现的任务划分。

## 问题

现代 LLM 通过产生长链推理来解决难题 — 逐步逻辑 5000 token 很常见，深度数学问题可达数万 token。在 70B 模型上以 35 token/秒解码，50k token 需要 24 分钟。交互式模型不是这样的。

投机解码（阶段 10 · 15）通过将一个序列内的 token 并行化获得 3-5 倍加速。超过这个范围，自回归解码的顺序依赖是硬性上限。每个新 token 依赖于所有先前的 token。

显而易见的问题：我们可以跨序列并行化吗？运行多个相同模型的副本在同一个问题上，让他们合作，让他们分工？

先前的工作：投票集成（运行 N 个模型，取多数答案）、思维树（分支推理路径并重新合并）和多智能体框架（给每个智能体分配子任务，使用协调器）。这些在特定任务领域都有帮助。但它们也都引入了显式协调机制 — 投票规则、分支剪枝逻辑、智能体间消息传递协议。

Hogwild! 推理采用了不同的方法。N 个 worker 共享一个 KV cache。每个 worker 立即看到其他所有 worker 生成的 token，就好像它们是自己的上下文一样。这些 worker — 无需任何训练或微调 — 自己想出如何分工。现代推理模型（QwQ、DeepSeek-R1、Claude 系列推理模式）可以读取共享缓存并说出这样的话："我看到 worker 2 已经处理了基 case，所以我来负责归纳步。"

加速比取决于工作负载，截至 2026 年 4 月仍属实验性。但这个想法值得了解，因为它开启了一条新的推理并行化轴线。

## 概念

### 设置

初始化 N 个 worker 进程，都运行同一个 LLM。不是每个 worker 独立的 KV cache，而是维护一个共享 cache。当 worker `i` 生成 token `t_j` 时，该 token 被写入共享 cache 的下一个位置。当 worker `k` 进行下一步时，它读取 cache 的当前状态（包括截至目前所有 N 个 worker 生成的内容）。

在步进时间上，worker 竞相写入 token。没有每个 worker 的位置索引 — cache 是一个单一的增长序列。顺序由写入到达时间决定。

### 为什么协调会涌现

Worker 们共享一个提示词。通常类似于："你是 N 个实例之一，共同处理这个问题。每个实例读取共享内存，可以看到其他实例写的内容。避免重复工作。"提示词加上共享 cache 就足够了。推理模型读取缓存，注意问题的哪些部分已经被尝试过，然后（通常是但并非总是）转向未探索的部分。

Hogwild! 论文（Rodionov et al., 2025）报告了以下观察：

- Worker 制定计划并通过缓存将计划传达给其他 worker。
- Worker 注意到其他 worker 推理中的错误并指出。
- Worker 在计划失败时进行调整并提出替代方案。
- 当提示检查冗余时，worker 检测到它并转向。

这些都不需要微调。涌现行为来自于模型已有的推理能力。

### 命名由来

论文名称是在 riff（模仿）Hogwild! SGD（Recht et al., 2011），一个异步更新优化器。类比：SGD 的异步 worker 都向共享参数向量写入；Hogwild! 推理的 worker 都向共享 KV cache 写入。两者都依赖经验收敛而非同步保证。

### RoPE 使这成为可能

旋转位置嵌入（RoPE，Su et al. 2021）通过 Q 和 K 向量的旋转来编码位置信息。因为位置是旋转而不是 baked-in 的偏移量，一个 token 的位置可以移动而无需重新计算 KV cache 条目。当 worker `i` 在位置 `p` 写入共享 cache 时，其他读取该位置的 worker 可以直接使用缓存条目 — 无需重新旋转。

在基于学习位置或绝对位置的模型中，Hogwild! 每次并发写入都需要 cache 失效。RoPE 让 cache 保持稳定。

### 墙上时间计算

设 `T_serial` 为一个 worker 单独解决问题的时间。设 `p` 为任务级可并行的比例。设 `c` 为每步协调开销（读取扩展后的缓存，决定写什么）。

单 worker 时间：`T_serial`。
N 个 worker 的 Hogwild! 时间，如果协调是免费的：`T_serial * ((1 - p) + p / N)`。经典的 Amdahl 定律。
有协调开销时：`T_serial * ((1 - p) + p / N) + c * steps_per_worker`。

对于一个 worker 来说要有产出，`c` 必须相对于每步解码时间很小。在产生 5k+ token 的推理模型上，worker 可以承受数百 token 的协调开销而仍然领先。在短对话任务上，协调占主导，Hogwild! 比串行更差。

### 具体例子

推理问题：10k token 的思维链。假设问题有 `p = 0.7` 的可并行内容（不同的证明策略、不同的 case 分析）和每 worker `c = 200` token 的协调开销。用 `N = 4` 个 worker：

- 串行时间：10000 解码步。
- Hogwild! 时间：10000 * (0.3 + 0.7 / 4) + 200 * 4 = 10000 * 0.475 + 800 = 5550 解码步。
- 加速比：10000 / 5550 = 1.8 倍。

这不算多。但在更长的推理问题（50k token）上，协调开销被摊销，加速比推到 2.5-3 倍。Hogwild! 是推理中等价于线程级并行性的东西 — 在一门让你自然编写多线程代码的语言中。

### 何时使用 Hogwild!

- 长推理问题（数千 token），其中任务可以跨独立子目标并行化。
- 被训练为逐步思考的推理模型。非推理模型自我协调能力不强。
- 单节点部署，VRAM 足够容纳共享 cache 加上 N 个 worker 进程。Cache 是共享的，但每个 worker 有自己的激活内存。

### 何时不用

- 短交互式对话。协调开销占主导。
- 不能并行的任务（单线性证明、单次编译）。N=1 是最大值。
- 非推理模型。没有协调涌现。
- 多节点部署。共享 cache 需要非常快的跨 worker 同步。节点内可以；跨节点是延迟灾难。

### 实验现状

截至 2026 年 4 月，Hogwild! 是带有开源 PyTorch 实现的研究方法。生产采纳尚未发生。三个阻碍：

1. 跨并发进程的共享 KV cache 管理是非平凡的工程。
2. 涌现协调是任务相关的；基准测试仍在建设中。
3. 相对于投机解码已经提供的加速比，这些加速比不大，两者可以结合但组合的工程又是另一层。

值得了解。值得做实验。尚不值得押注产品。

## 构建它

`code/main.py` 实现了一个玩具 Hogwild! 模拟器：

- 两个 worker 进程，每个是一个确定性"LLM"，产生多种 token 类别之一（work-token、observe-token、coordinate-token），概率已知。
- 一个共享 cache（就是一个 token 列表），两个 worker 都读写。
- 一个简单的协调逻辑：当 worker 看到另一个已经在某个类别产生了足够多的 work token 时，它选择不同的类别。

模拟器运行固定的步数预算并报告：

- 产生的总 work-token 数。
- 总墙上时间（worker 步数）。
- 相对于单个 worker 的有效加速比。
- 哪个 worker 写了哪个 token 的追踪。

### 第 1 步：共享 cache

一个两个 worker 都 append 的列表。在真实实现中用简单锁定（Python `threading.Lock`）；我们用计数器模拟。

### 第 2 步：worker 循环

每个 worker，在每一步：

- 读取当前共享 cache。
- 根据已有内容决定写什么类别的 token。
- 写一个 token。

### 第 3 步：协调启发式

如果类别 X 在 cache 中已有 K 个 token，而 worker's intended 类别是 X，worker 切换到类别 Y。这是一个玩具替代品，代表推理模型的行为："注意到这已经被覆盖了，改做点别的。"

### 第 4 步：测量加速比

用 N=1 worker 和 N=2 worker 运行模拟器，相同总步数预算。统计产生的 work-token 数。N=2 应该因为协调驱动的任务划分产生大约 1.5-1.8 倍的 work-token。

### 第 5 步：压力测试协调

降低协调启发式的敏感度。重新运行。观察在没有良好协调的情况下，N=2 冗余地产生相同的 token，加速比降到 1 以下。这与论文的观察一致：诀窍只在 worker 有自我协调的推理能力时才有效。

## 使用它

截至 2026 年 4 月，Hogwild! 在生产中的集成是研究级的。来自 Yandex/HSE/IST 的参考实现基于 PyTorch，目标是 DeepSeek-R1 和 QwQ 模型上的单节点多进程设置。

务实的采纳路径：

1. 对你的推理任务工作负载进行 profile。测量探索性 token（多策略、case 分析、搜索）与线性 token 的比例。
2. 如果探索占主导，运行双 worker Hogwild! 实验。测量墙上时间改进。
3. 如果改进小于 1.3 倍，你处于协调主导区间。回到单 worker。
4. 如果改进大于 1.5 倍，推进到 N=4 并重新测量。收益递减通常在 N=4-8 左右出现。

与投机解码结合：每个 Hogwild! worker 可以独立使用投机解码。两种加速比相乘（大致），将 3x 投机解码和 1.8x Hogwild! 带到有效 5.4x 相对于朴素单 worker 解码。

## 交付它

本课产出 `outputs/skill-parallel-inference-router.md`。给定推理工作负载 profile（token 预算、任务并行度 profile、模型系列、部署目标），它路由到投票、思维树、多智能体、Hogwild! 和投机解码策略。

## 练习

1. 用默认设置运行 `code/main.py`。确认 N=2 Hogwild! 配置在相同墙上时间内产生比 N=1 基线更多的 work-token。

2. 降低协调启发式的强度（设置 `coordination_weight=0.1`）。重新运行。展示加速比崩溃。解释原因：worker 在无法协调时重复做功。

3. 计算 50k token 推理任务在 `p=0.8, c=500` 和 N=4 worker 下的预期 Hogwild! 加速比。同样的计算对 1k token 对话任务 `p=0.3, c=200` 和 N=4。为什么一个有收益而另一个没有？

4. 阅读 Hogwild! 论文第 4 节（初步评估）。找出作者报告的两种失败模式。描述更好的协调提示词如何缓解每一种。

5. 在玩具中结合 Hogwild! 与投机解码：每个 worker 内部使用 2-token 投机解码。报告乘法加速比。当两个 worker都想扩展同一个共享 cache 前缀时，出现了什么记账问题？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| Hogwild! | "并行 worker，共享 cache" | N 个相同 LLM 实例并发运行，共用一个 KV cache；通过自提示产生涌现协调 |
| 共享 KV cache | "协调介质" | 所有 worker 都读写的单一增长 KV 缓冲区；实现跨 worker 即时 token 可视 |
| 涌现协调 | "无需训练" | 有推理能力的 LLM 可以读取共享 cache 并分工，无需任何微调或显式协议 |
| 协调开销（c） | "定向消耗的 token" | 每个 worker 读取扩展缓存和决定做什么的开销；必须相对于总解码时间保持较小 |
| 可并行比例（p） | "什么可以并行运行" | 任务级并行性：不属于内在顺序的总工作的比例 |
| RoPE 使 Hogwild! 成为可能 | "旋转位置是平移不变的" | 因为位置是旋转，写入共享 cache 不需要重新计算先前的 token |
| 投票集成 | "运行 N 个，取多数" | 最简单的并行推理拓扑；对分类有用，对长形式推理用处不大 |
| 思维树 | "分支并剪枝" | 探索多个分支并剪枝的推理策略；显式协调逻辑 |
| 多智能体框架 | "分配子任务" | 每个智能体获得一个角色；协调器编排；协议开销大 |

## 延伸阅读

- [Rodionov et al. — Hogwild! Inference: Parallel LLM Generation via Concurrent Attention (arXiv:2504.06261)](https://arxiv.org/abs/2504.06261) — Hogwild! 论文，在 QwQ 和 DeepSeek-R1 上的初步评估
- [Recht, Re, Wright, Niu — Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent (arXiv:1106.5730, NeurIPS 2011)](https://arxiv.org/abs/1106.5730) — 原始 Hogwild!，命名由来
- [Su et al. — RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864) — RoPE，使共享缓存推理成为可能的性质
- [Yao et al. — Tree of Thoughts: Deliberate Problem Solving with Large Language Models (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — 思维树推理策略，Hogwild! 与之正交
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — 投机解码，Hogwild! 可组合的序列内并行化
- [Hogwild! 参考 PyTorch 实现](https://github.com/eqimp/hogwild_llm) — 论文实验的唯一权威来源