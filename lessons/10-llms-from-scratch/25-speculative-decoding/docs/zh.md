# 投机解码与 EAGLE

> 前沿 LLM 生成一个 token 需要一次跨越数十亿参数的完整前向传播。这次前向传播严重过度配置：大多数时候，一个小得多的模型可以正确猜出下一个 3-5 个 token，而大模型只需要*验证*这个猜测。猜对了，你就用买一个 token 的价格得到了 5 个 token。投机解码（Leviathan et al. 2023）将这一点精确化，而 EAGLE-3（2025）将接受率推到约 4.5 token/验证 — 在匹配输出分布下 4-5 倍加速。

**类型：** 构建型
**语言：** Python（带 numpy）
**前置条件：** 阶段 10 · 课 12（推理优化），阶段 10 · 课 04（预训练 Mini-GPT）
**时间：** 约 75 分钟

## 问题

70B 类模型在 H100 上的解码吞吐量通常为 40-80 token/秒。每个 token 需要一次完整前向传播，从 HBM 读取所有模型权重。你不能在不改变输出的情况下让模型更小。你不能把 batch size 提高到超过内存限制。你被困住了 — 除非你能让模型一次输出超过一个 token。

自回归生成看起来本质上是串行的：`x_{t+1} = sample(p(· | x_{1:t}))`。但这里有一个并发机会。如果有一个便宜的预测器说"接下来的 4 个 token 可能是 [a, b, c, d]"，你可以在大模型的**单次前向传播**中验证全部 5 个位置，并接受最长匹配前缀。

Leviathan、Kalai、Matias（2023，"Fast Inference from Transformers via Speculative Decoding"）通过一个巧妙的接受/拒绝规则使这精确化，同时保持目标模型的采样分布。相同的输出分布，2-4 倍更快。

## 概念

### 双模型设置

- **目标模型** `M_p`：大、慢、高质量，你实际想要从中采样的模型。分布：`p(x)`。
- **草稿模型** `M_q`：小、快、低质量模型。分布：`q(x)`。小 5-30 倍。

每步：

1. 草稿模型自回归地提出 `K` 个 token：`x_1, x_2, ..., x_K ~ q`。
2. 目标模型在所有 `K+1` 位置上运行一次前向传播并行计算，产生每个被提议 token 的 `p(x_k)`。
3. 通过以下修改后的拒绝采样规则从左到右接受/拒绝每个 token。接受最长匹配前缀。
4. 如果任何 token 被拒绝，从校正后的分布中采样替代并停止。否则从 `p(· | x_1...x_K)` 中采样一个奖励 token。

如果草稿与目标完美匹配，你得到 K+1 token 每目标前向。如果草稿在位置 1 就错了，你只得到 1 个 token。

### 精确性规则

投机解码**在分布上证明等价于从 p 采样**。拒绝规则：

```
对于每个起草的 token x_t：
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        接受 x_t
    else:
        从残差中采样替代：(p - q)+ / ||(p - q)+||_1
        停止
```

其中 `(p - q)+` 表示逐点差值的正部。当草稿和目标一致时（`p ≈ q`）接受率接近 1。当它们不一致时，残差分布被构造成使整体采样仍然是精确的 `p`。

**贪婪情况。** 对于温度=0 采样，只需检查 `argmax(p) == x_t`。如果是的，接受；如果不是，输出 `argmax(p)` 并停止。

### 预期加速比

如果草稿模型的 token 级接受率是 `α`，每目标前向传播的预期输出 token 数是：

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = 草稿长度，α ∈ [0, 1]
```

在 `α = 0.8, K = 4`：`1 - 0.8^5)/(1 - 0.8) = 3.36` token 每前向。单次目标前向成本大致为 `cost_q * K + cost_p`（K 步草稿加一步目标验证）。如果 `cost_p >> cost_q * K`，吞吐量加速比是 `3.36× / 1 = 3.36×`。

唯一的真实参数是 `α`，它完全取决于草稿-目标的对齐程度。一个好的草稿就是一切。

### 训练草稿：蒸馏

随机小模型是一个糟糕的草稿。标准方案是从目标蒸馏：

1. 选一个小架构（约 70B 目标用 1B，约 7B 目标用 500M）。
2. 在大型文本语料上运行目标模型；存储其下一个 token 分布。
3. 用相对于目标分布的 KL 散度训练草稿（不是相对于真实 token）。

结果是：`α` 在 coding 上通常为 0.6-0.8，在自然语言聊天上为 0.7-0.85。生产中加速比 2-3 倍。

### EAGLE：树形草稿 + 特征复用

Li、Wei、Zhang、Zhang（2024，"EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"）观察到标准投机解码中的两个低效：

1. 草稿做 K 步串行，每步都是完整堆栈。但草稿可以复用目标最近一次验证的特征（隐藏状态）— 目标已经计算了丰富的表示，而草稿从头重新推导。
2. 草稿输出一个线性链。如果草稿可以输出一个*树*形的候选（每个节点多个猜测），目标模型的单次前向传播可以通过树注意力掩码并行验证多条候选路径，并选取最长接受的分支。

EAGLE-1 的改变：
- 草稿输入 = 目标在位置 t 的最终隐藏状态，而非原始 token。
- 草稿架构 = 1 层 transformer 解码器（而非独立的小模型）。
- 输出 = 每深度 K = 4-8 个候选的树，深度 4-6。

EAGLE-2（2024）添加动态树拓扑：树在草稿不确定的地方变宽，在确定的地方保持狭窄。在不增加验证成本的情况下提高 `α_effective`。

EAGLE-3（Li et al. 2025，"EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"）移除了固定的顶层特征依赖，并在新的"测试时模拟"损失下训练草稿 — 草稿在匹配目标测试时分布而非教师强制训练分布的输出上训练。接受率从 0.75（EAGLE-2）升至 0.82（EAGLE-3），平均 token/验证从 3.0 升至 4.5。

### 树注意力验证

当草稿输出一棵树时，目标模型使用**树注意力掩码**在单次前向传播中验证它 — 一个编码树拓扑而非纯线的因果掩码。每个 token 只关注其在树中的祖先。验证传播仍然是一次前向、一次 matmul；拓扑掩码只多花几个 KV 条目。

```
        根节点
       /    \
      a      b
     / \    / \
    c  d   e   f
```

如果 `a, b` 是竞争的第一 token 候选，`c, d, e, f` 是第二 token 候选，所有六个位置在一次前向传播中被验证。输出是任意接受路径上的最长前缀。

### 何时有效，何时无效

**有效：**
- 可预测文本的聊天/补全（code、常见英文、结构化输出）。`α` 高。
- 解码期间有未使用 GPU 计算的配置（内存受限阶段）。树形草稿利用可用的 FLOPs。

**无效/无收益：**
- 高随机性输出（高温创造性写作）。`α` 降至接近 `1/|vocab|`。
- 高并发批量服务 — batch 已经填充了 FLOPs，树验证空间很小。
- 非常小的目标模型，此时草稿并没有小多少。

生产团队通常报告聊天 2-3 倍墙上时间加速，code 生成 3-5 倍，创意写作接近零。

## 构建它

`code/main.py`：

- 一个引用 `speculative_decode(target, draft, prompt, K, temperature)` 实现精确拒绝规则并验证它保持目标分布（经验 KL < 0.01 vs 朴素目标采样）。
- 一个 EAGLE 风格树起草器，用 top-p 分支构建深度 K 的树。
- 一个树注意力掩码构建器，为验证器产生正确的因果模式。
- 一个接受率测试工具，在一个小 LM 上运行两者（从一个 GPT-2-medium 目标蒸馏一个 GPT-2-small 草稿）。

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """一轮投机解码。返回接受的 token 列表。"""
    # 1. 起草 K 个 token
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. 目标在每个起草位置 + 1 个额外位置计算 p
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. 从左到右接受/拒绝
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. 全部 K 个被接受 → 从目标采样一个奖励 token
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 使用它

- **vLLM** 和 **SGLang** 提供一流投机解码。标志：`--speculative_model`、`--num_speculative_tokens`。通过 `--spec_decoding_algorithm eagle` 标志支持 EAGLE-2/3。
- **NVIDIA TensorRT-LLM** 原生支持 Medusa 和 EAGLE 树。
- **参考草稿模型**：`Qwen/Qwen3-0.6B-spec`（为 Qwen3-32B 起草）、`meta-llama/Llama-3.2-1B-Instruct-spec`（为 70B 起草）。
- **Medusa 头**（Cai et al. 2024，"Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"）：不是草稿模型，而是在目标本身上添加 K 个并行预测头。部署更简单，接受率比 EAGLE 略低。

## 交付它

本课产出 `outputs/skill-speculative-tuning.md` — 一个对目标模型工作负载进行 profile 并选择：草稿模型、K（草稿长度）、树宽度、温度，以及何时回退到朴素解码的技能。

## 练习

1. 实现精确拒绝规则并经验验证。运行 10K 样本通过 `speculative_decode` 和通过朴素目标采样；计算两个输出分布之间的 TV 距离。应该 < 0.01。

2. 计算加速比公式。给定固定的 `α` 和 `K`，绘制每目标前向的预期 token 数。找出 α ∈ {0.5, 0.7, 0.9} 的最优 K。

3. 训练一个小草稿。用 KL 损失在 100M token 上从一个 124M GPT-2 目标蒸馏一个 30M GPT-2 草稿。在留出文本上测量 `α`。预期：0.6-0.7。

4. 实现 EAGLE 风格树形起草。不是链，而是在每个深度输出 top-3 分支。构建树注意力掩码。验证目标接受最长正确分支。

5. 测量失败模式。在 temperature=1.5（高随机性）下运行投机解码。展示 α 崩溃，算法因草稿开销而比朴素解码慢。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|------------------------|
| 目标模型 | "大模型" | 你想要采样的慢、高质量模型（p 分布） |
| 草稿模型 | "投机者" | 小、快预测器（q 分布）；小 5-30 倍 |
| K / 草稿长度 | " lookahead" | 每验证轮次投机起草的 token 数 |
| α / 接受率 | "命中率" | 草稿提议被接受的逐 token 概率 |
| 精确拒绝规则 | "接受测试" | r < p/q 比较，保持目标分布 |
| 残差分布 | "校正后的 p-q" | (p - q)+ / ||(p - q)+||_1，在拒绝时采样的分布 |
| 树形起草 | "分支投机" | 草稿输出一棵候选树，用树结构注意力掩码一次验证 |
| 树注意力掩码 | "拓扑掩码" | 编码树拓扑的因果掩码，使每个节点只关注其祖先 |
| Medusa 头 | "并行头" | 目标本身的 K 个额外预测头；无需独立草稿模型 |
| EAGLE 特征复用 | "隐藏状态草稿" | 草稿输入是目标的最后隐藏状态，而非原始 token，缩小草稿 |
| 测试时模拟损失 | "EAGLE-3 训练" | 在匹配目标测试时分布而非教师强制的输出上训练草稿 |

## 延伸阅读

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — 精确拒绝规则和理论加速比分析
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — DeepMind 的并发投机采样论文
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — 并行头替代草稿模型
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — 特征复用和树形起草
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — 动态树拓扑
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — 训练时-测试时匹配
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — Jacobi/ lookahead 解码，无需草稿的替代方案