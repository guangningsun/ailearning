# 差分注意力（V2）

> Softmax 注意力在每个不匹配的 token 上散布少量概率。超过 100k token 时噪声累积会淹没信号。差分 Transformer（Ye 等，ICLR 2025）通过计算两个 softmax 的差作为注意力来修复它，减去共享的噪声底。DIFF V2（Microsoft，2026 年 1 月）是生产栈的重写：解码延迟匹配 baseline Transformer，无需自定义核，与 FlashAttention 兼容。本课是 V1 到 V2 端到端的工作，带着一个可以运行在标准库 Python 中的差分操作的玩具实现。

**类型：** 构建型
**语言：** Python（标准库）
**前置条件：** 第 7 阶段 · 02（自注意力）、第 7 阶段 · 15（注意力变体）、第 10 阶段 · 14（架构 walkthrough）
**时间：** 约 60 分钟

## 学习目标

- 精确陈述为什么 softmax 注意力有噪声底，以及为什么它随上下文长度增长。
- 推导差分注意力公式，并解释为什么减法消除了共享噪声成分同时保留了信号。
- 走过 V1 到 V2 的差异：什么变快了，什么变简单了，什么变稳定了，以及为什么每个改动对生产预训练是必要的。
- 在纯 Python 中从头实现差分注意力，并在合成信号加噪声查询上经验验证噪声消除属性。

## 问题

标准 softmax 注意力有一个数学特性，在规模上会变成操作上的头痛。对于一个 query `q`，注意力权重是 `softmax(qK^T / sqrt(d))`。Softmax 永远不会产生精确的零——每个不匹配的 token 都获得一些正质量。那个残留质量是噪声，它随上下文长度缩放。在 128k token 时，即使每个不匹配的 token 只获得 0.001% 的概率，127,999 个 token 合计贡献约 12%。模型必须学会绕过随上下文增长的噪声底。

经验上这表现为注意力头干扰：长上下文 RAG 中的幻觉引用、100k token 检索任务中的"迷失在中间"失败，以及在 32k 之后 haystack 针基准上的微妙准确率下降。差分 Transformer 论文（arXiv:2410.05258，ICLR 2025）测量了差距：DIFF Transformer 比同规模 baseline 获得更低的困惑度、更高的长上下文准确率和更少的幻觉。

DIFF V1 有三个问题使其无法进入前沿预训练流水线。它的 value 缓存在每个解码步骤中需要加载两次，它需要打破 FlashAttention 兼容性的自定义 CUDA 核，而且它的每头 RMSNorm 在 70B 及以上规模的训练中会破坏稳定性。DIFF V2（Microsoft unilm 博客，2026 年 1 月 20 日）修复了全部三个。本课走过两个版本，构建差分算子，并在玩具查询上对噪声消除进行基准测试。

## 概念

### Softmax 的噪声底

对于一个 query `q` 和 keys `K = [k_1, ..., k_N]`，注意力权重是：

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

没有任何 `w_i` 是零。如果 `k_i` 与 `q` 完全无关，分数 `q . k_i` 不是 0——它围绕零波动，方差为 `||q||^2 / d`。经过 softmax 归一化后，每个无关 token 仍然贡献 `O(1/N)` 到加权和。无关 token 的总贡献是 `O((N-1)/N) = O(1)`——不是一个小量。

模型想要的是类似 hard top-k 的东西：在匹配的 token 上高权重，其他地方接近零权重。Softmax 太平滑了，无法直接做到这一点。

### 差分思想

将每个头的 Q 和 K 投影分成两个：Q = (Q_1, Q_2) 和 K = (K_1, K_2)。计算两个注意力图：

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

输出：

```
DiffAttn = (A_1 - lambda * A_2) V
```

减法消除了两个图共享的任何噪声分布。如果两个图在 127k 个无关 token 上都有大致均匀的权重（在随机初始化时它们会的），这些就会相互抵消。信号——在少数真正相关 token 上的峰值权重——只有在两个图中以相同大小出现时才会抵消，这在模型训练后是不会发生的。

`lambda` 是每个头的可学习标量，参数化为 `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`。它可以为负。`lambda_init` 默认为 0.8 这样的小正数。

### 为什么这匹配了有头的噪声消除

想象两个嘈杂的麦克风录制同一个声音。两者都拾取说话者加上相关的背景噪声。将一个减去另一个，共享的噪声就会消失。声音保留下来，因为两个信号在相位或幅度上有足够的差异来防止完全抵消。每个头的 `lambda` 学到的正是这种平衡。

### V1 与 V2：差异

V1 保持参数数量等于 baseline Transformer。为了在每个头上获得两个 query，它将头维度减半。这牺牲了头的表达能力，更痛苦的是——每头的 value 缓存减半。解码必须在每步加载 value 缓存两次（每 softmax 分支一次）。结果：尽管匹配参数数量，解码比 baseline 慢。

V2 加倍 query 头的数量并保持 KV 头不变（从 up-projection 借用参数）。头维度与 baseline 保持一致。减法之后，额外的维度被投影回来以匹配 baseline Transformer 的 O_W 投影。三件事同时发生：

1. 解码速度匹配 baseline（KV 缓存只加载一次）。
2. FlashAttention 原样运行（无自定义核）。
3. 解码时的算术强度增加（从 HBM 加载的每字节有更多计算）。

V2 还移除了 V1 用于稳定减法的每头 RMSNorm。在 70B 级预训练规模上，那个 RMSNorm 破坏了后期训练。V2 用一个更简单的初始化方案取代它，在没有额外模块的情况下保持训练稳定。

### 何时使用它

| 工作负载 | 收益 |
|----------|---------|
| 长上下文 RAG（64k+） | 更清晰的注意力图，更少的幻觉引用 |
| Haystack 中找针基准 | 在 32k 之后准确率实质性提升 |
| 多文档 QA | 更少的跨文档干扰 |
| 8k 代码补全 | 边缘收益，不值得架构变更 |
| 短聊天（< 4k） | 与 baseline 本质上无法区分 |

价值随上下文长度增长。在 4k token 时噪声底很小，标准注意力没问题。在 128k 时它在伤害你。

### 它如何与其他 2026 年旋钮叠加

| 特性 | 与 DIFF V2 兼容？ |
|---------|------------------------|
| GQA | 兼容（V2 增加 Q 头，不增加 KV 头） |
| MLA（DeepSeek） | 原则上兼容，没有发表过将两者结合的论文 |
| MoE | 兼容（注意力独立于 MLP 块） |
| RoPE | 兼容（不变） |
| YaRN / 长上下文缩放 | 兼容（正是 DIFF 最大帮助的地方） |
| FlashAttention | V2 中兼容（V1 中不兼容） |
| 投机解码 | 兼容（注意力变化对 spec-decode 循环不可见） |

## 构建它

`code/main.py` 在纯 Python 中实现差分注意力。一个具有已知信号加噪声结构的玩具 query 让您可以直接测量噪声消除比。

### 第 1 步：标准 softmax 注意力

标准库矩阵运算：列表的列表、手动 matmul、带数值稳定性最大值减法的 softmax。

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### 第 2 步：将 Q、K 分成两半

V1 风格：头维度减半。V2 风格：保持头维度并加倍头数。玩具实现为清晰起见使用 V1——数学相同，只是记账方式不同。

### 第 3 步：两个 softmax 分支 + 减法

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

注意：输出权重可以为负。这没问题——value 缓存仍然处理带符号的贡献。后续的 V 投影吸收符号。

### 第 4 步：噪声消除测量

构建一个长度为 1024 的合成序列。将信号 token 放在已知位置，其余填充噪声。计算（a）信号位置上的标准 softmax 注意力权重和（b）差分注意力权重。测量每个的信噪比。DIFF 注意力可靠地产生高 3 倍到 10 倍的信噪比，取决于两个分支被训练来差异化的程度。

### 第 5 步：V1 与 V2 参数核算

给定配置（hidden=4096，heads=32，d_head=128），打印：

- Baseline Transformer：Q、K、V 每个大小 `hidden * hidden`，MLP 为 4 * hidden。
- DIFF V1：Q、K 每个大小 `hidden * hidden`，V 大小 `hidden * hidden`（不变），头维度内部减半。添加每头 `lambda` 参数（O(heads * d_head)）。
- DIFF V2：Q 大小 `2 * hidden * hidden`，K 大小 `hidden * hidden`，V 大小 `hidden * hidden`。额外维度在 O_W 之前投影回来。添加相同的 `lambda` 参数。

玩具测量 V2 每注意力块的额外参数成本（约 `hidden * hidden` 额外）并打印出来。

## 使用它

DIFF V2 在 2026 年 4 月尚未在每个生产推理服务器中发货，但正在 vLLM 和 SGLang 中进行集成。与此同时，该模式出现在：

- Microsoft 内部长上下文生产模型。
- 在针对 256k 及以上上下文的几个开放模型训练运行的研究复现中。
- 将 DIFF 注意力与交替层上的滑动窗口注意力结合的混合架构。

在 2026 年何时使用它：

- 从头训练针对 64k 及以上有效上下文的新模型。从一开始添加差分注意力；后期再训练成本高昂。
- 微调一个长上下文模型，其中"迷失在中间"失败主导您的评估。Q 投影上的 LoRA 可以近似 DIFF 结构。

何时不使用：

- 您正在服务一个具有稳定长上下文性能的预训练稠密模型。在现有权重上重新训练的成本很少能收回。
- 您的上下文始终低于 16k。噪声底可以忽略不计。

## 交付它

本课产出 `outputs/skill-diff-attention-integrator.md`。给定一个模型架构、目标上下文长度、幻觉配置文件和训练预算，它生成将差分注意力添加到新预训练运行或 LoRA 微调的集成计划。

## 练习

1. 运行 `code/main.py`。验证差分注意力报告的信号噪声比高于在合成查询上的标准 softmax 注意力。改变噪声幅度并显示标准注意力变得不可用的交叉点。

2. 计算从 baseline 到 DIFF V1 和从 baseline 到 DIFF V2 的参数量增量，用于 7B 级模型（hidden=4096，heads=32，d_head=128，32 层）。显示哪些组件增加了参数，哪些保持不变。

3. 阅读 DIFF V1 论文第 3 节（arXiv:2410.05258）和 DIFF V2 Hugging Face 博客第 2 节。用两句话解释为什么 V1 每头 RMSNorm 是必要的，以及为什么 V2 可以移除它而不会导致训练发散。

4. 实现一个消融：计算 `lambda = 0`（纯第一个 softmax）和 `lambda = 1`（完全减法）的差分注意力。在合成查询上，测量信噪比如何跨 sweep 变化。识别使信噪比最大化的 `lambda`。

5. 将玩具扩展到 GQA + DIFF V2。选择 8 个 KV 头和 32 个 Q 头。证明 KV 缓存大小匹配具有相同（8, 32）配置的 baseline GQA 模型。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| 差分注意力 | "两个 softmax 互相减去" | 将 Q、K 分成两半，计算两个 softmax 图，用第二个（按 lambda 缩放）从第一个中减去，然后乘以 V |
| 噪声底 | "Softmax 的非零尾巴" | Softmax 给每个无关 token 的 O(1/N) 权重，在长上下文上求和为 O(1) |
| lambda | "减法缩放" | 每个头的可学习标量，参数化为 `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`；可以为负 |
| DIFF V1 | "ICLR 2025 版本" | 原始差分 Transformer；为保持参数量将头维度减半，需要自定义核，解码更慢 |
| DIFF V2 | "2026 年 1 月的修复" | 加倍 Q 头保持 KV 头；匹配 baseline 解码速度并与 FlashAttention 一起工作 |
| 每头 RMSNorm | "V1 稳定器" | V1 在差值之后应用的额外归一化；V2 移除它以防止后期训练不稳定 |
| 信噪比 | "有多少注意力被浪费" | 在真实信号位置上权重与在无关位置上平均权重的比率 |
| 迷失在中间 | "长上下文失败模式" | 经验现象，检索准确率在长上下文中间位置的文档上下降——DIFF 注意力减少了这种情况 |
| 算术强度 | "每加载字节的 FLOP" | V2 在解码时通过加倍每个 KV 加载的 query 增加的比率；对内存受限的解码很重要 |

## 进一步阅读

- [Ye 等 — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) —— 原始论文，包含噪声消除理论和长上下文消融
- [Microsoft unilm — Differential Transformer V2 (Hugging Face 博客，2026 年 1 月)](https://huggingface.co/blog/microsoft/diff-attn-v2) —— 生产栈重写，匹配 baseline 解码，与 FlashAttention 兼容
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) —— 为什么减法恢复预训练注意力结构的理论分析
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) —— 参数共享变体
- [Vaswani 等 — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) —— DIFF 减去的 baseline Transformer
- [Liu 等 — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) —— DIFF 注意力针对的长上下文基准