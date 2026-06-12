# 推测解码 —— 草稿、验证、重复

> 自回归解码是串行的。每个 token 都要等前一个。推测解码打破了这个链条：用一个小模型起草 N 个 token，再用大模型一次验证全部 N 个。草稿正确时，你用一次大模型前向传播换了 N 个生成 token。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 7 · 07（GPT 因果语言模型）、阶段 7 · 12（KV Cache 与 Flash Attention）
**时间：** 约 60 分钟

## 问题

一个 70B 的 LLM 在 H100 上采样一个 token 大约需要 30 ms。一个 3B 的草稿模型大约需要 3 ms。如果我们让 3B 模型提前起草 5 个 token，然后运行 70B 一次验证全部 5 个，总耗时是 `5×3 + 30 = 45 ms`，最多可接受 5 个 token——而直线式生成需要 `5×30 = 150 ms`。这就是推测解码的完整思路：用少量额外的 GPU 内存（草稿模型）换取 2–4 倍更低的解码延迟。

关键在于保持分布不变。Leviathan 等人（2023）以及陈等人同时提出的推测采样保证了输出序列与大模型独立采样时的分布**完全相同**。没有质量权衡，只是更快。

2026 年，四类草稿-验证器组合主导着推理领域：

1. **原始推测解码（Leviathan 2023）。** 独立的草稿模型（如 Llama 3 1B）+ 验证器（如 Llama 3 70B）。
2. **Medusa（Cai 2024）。** 验证器上的多个解码头并行预测位置 t+1..t+k。没有独立的草稿模型。
3. **EAGLE 系列（Li 2024, 2025）。** 轻量级草稿，复用验证器的隐藏状态；比原始版本更高的接受率；通常 3–4 倍加速。
4. **前瞻解码（Fu 2024）。** 雅可比迭代；完全不需要草稿模型。自我推测。小众但无依赖。

2026 年，每家生产推理框架都默认搭载推测解码。vLLM、TensorRT-LLM、SGLang 和 llama.cpp 都至少支持原始版 + EAGLE-2。

## 核心概念

### 核心算法

给定一个验证器 `M_q` 和一个更便宜的草稿 `M_p`：

1. 设 `x_1..x_k` 为已解码的前缀。
2. **起草**：用 `M_p` 自回归地提出 `d_{k+1}, d_{k+2}, ..., d_{k+N}`，附带草稿概率 `p_1..p_N`。
3. **并行验证**：对 `x_1..x_k, d_{k+1}, ..., d_{k+N}` 运行一次 `M_q`，得到位置 k+1..k+N+1 的验证器概率 `q_1..q_{N+1}`。
4. **从左到右接受/拒绝每个草稿 token**：对每个 `i`，以概率 `min(1, q_i(d_i) / p_i(d_i))` 接受。
5. 在位置 `j` 首次拒绝时：从"残差"分布 `(q_j - p_j)_+` 归一化后采样 `t_j`。j 之后的所有草稿都丢弃。
6. 全部接受 N 个时：从 `q_{N+1}` 采样一个额外 token `t_{N+1}`（免费 bonus token）。

残差分布技巧是保持输出完全按照 `M_q` 独立采样的数学核心。

### 什么决定加速比

设 `α` = 每个草稿 token 的期望接受率。设 `c` = 草稿与验证器的成本比。每步：

- 朴素生成每 token 做一次大模型调用。
- 推测解码每 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` 个 token 做一次大模型调用（α 较大时）。

`α = 0.75`，`N = 5` 时的经验法则：大模型调用减少 3 倍。草稿成本是 5 倍便宜。总时钟时间下降约 2.5 倍。

**α 取决于：**

- 草稿对验证器的近似程度。同家族 / 同训练数据会显著提升 α。
- 解码策略。贪婪草稿对贪婪验证器：α 高。温度采样：更难匹配；接受率下降。
- 任务类型。代码和结构化输出接受更多（可预测）；自由形式创意写作接受更少。

### Medusa —— 无草稿模型的草稿

Medusa 用验证器上的额外输出头取代了草稿模型。在位置 `t`：

```
shared trunk → hidden h_t
    ├── head_0: 预测 t+1 位的 token（标准 LM head）
    ├── head_1: 预测 t+2 位的 token
    ├── head_2: 预测 t+3 位的 token
    ├── head_3: 预测 t+4 位的 token
```

每个 head 输出自己的 logits。推理时从每个 head 采样得到候选序列，然后用一种树注意力机制一次验证所有候选续篇。

优点：无第二个模型。缺点：增加可训练参数；需要监督微调阶段（约 1B tokens）；接受率略低于配合好草稿的原始推测解码。

### EAGLE —— 通过复用隐藏状态实现更好的草稿

EAGLE-1/2/3（Li 等，2024–2025）让草稿模型成为一个极小的 transformer（通常 1 层），摄入验证器最后一层的隐藏状态。由于草稿能看到验证器的特征表示，它的预测与验证器的输出分布高度相关。接受率从 ~0.6（原始版）攀升至 0.85+。

EAGLE-3（2025）增加了对候选续篇的树搜索。vLLM 和 SGLang 为 Llama 3/4 和 Qwen 3 默认搭载 EAGLE-2/3 作为推测解码路径。

### KV 缓存之舞

验证在一轮前向传播中向验证器输入 N 个草稿 token。这将验证器的 KV 缓存扩展了 N 条。如果某些草稿被拒绝，必须将缓存回滚到被接受的前缀长度。

生产实现（vLLM 的 `--speculative-model`、TensorRT-LLM 的 LookaheadDecoder）用临时 KV 缓冲区处理。先写入，接受后提交。这在概念上不难，但很繁琐。

## 构建它

参见 `code/main.py`。我们实现核心推测采样算法（拒绝步骤 + 残差分布），包含：

- 一个"大模型"，是对手工编码分布的确定性 softmax（这样我们就能解析验证接受率数学）。
- 一个"草稿模型"，是大模型的扰动版本。
- 一个接受/拒绝循环，产生与直接采样相同的边缘分布。

### 第 1 步：拒绝步骤

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` 是均匀随机数。`q_prob` 是验证器对起草 token 的概率。`p_prob` 是草稿模型的概率。Leviathan 定理是：这个伯努利决策，加上在拒绝时从残差采样，精确保持了验证器的分布。

### 第 2 步：残差分布

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

逐元素从 `q` 减去 `p`，将负值截断到零，再归一化。在任何拒绝时从该分布采样。

### 第 3 步：一步推测

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

五个接受 → 一个 bonus → 一次验证器传递产生六个 token。

### 第 4 步：测量接受率

在不同的草稿质量水平下运行 10,000 步推测。绘制接受率 vs 草稿与验证器分布之间的 KL 散度。你应该看到一个干净的单调关系。

### 第 5 步：验证分布等价

经验验证：推测循环产生的 token 直方图应与直接从验证器采样产生的直方图一致。这是 Leviathan 定理的实践。卡方检验确认在采样误差范围内一致。

## 使用它

生产环境：

```bash
# vLLM + EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM + 原始草稿模型
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

TensorRT-LLM 在 2026 年中期拥有最快的 Medusa 路径。`faster-whisper` 用一个小草稿为 Whisper-large 封装了推测解码。

**选择草稿：**

| 策略 | 何时选择 | 加速比 |
|------|----------|--------|
| 原始草稿（1B/3B Llama 系列） | 快速原型，无需训练 | 1.8–2.3× |
| Medusa 头 | 你可以微调验证器 | 2–3× |
| EAGLE-2 / 3 | 生产环境，最大加速 | 3–4× |
| 前瞻 | 无草稿，无训练，无额外参数 | 1.3–1.6× |

**何时不要用推测解码：**

- 生成 1–5 个 token 的单序列。开销占主导。
- 高度创意 / 高温度采样（α 下降）。
- 内存受限的部署（草稿模型增加 VRAM）。

## 交付它

参见 `outputs/skill-spec-decode-picker.md`。这个 skill 为新的推理工作负载选择推测解码策略（原始 / Medusa / EAGLE / 前瞻）和调参（N、草稿温度）。

## 练习

1. **简单。** 运行 `code/main.py`。确认在 50,000 个 token 上推测 token 分布与验证器直接采样的分布在卡方 p > 0.05 范围内一致。
2. **中等。** 绘制加速比（大模型前向传播的 token 数）作为 N 的函数，针对 `α = 0.5, 0.7, 0.85`。找出每个 α 的最优 N。（提示：每次验证调用的期望 token 数 = `(1 - α^{N+1}) / (1 - α)`。）
3. **困难。** 实现一个小型 Medusa：取第 14 课的综合 GPT，添加 3 个额外 LM head 预测位置 t+2、t+3、t+4。用联合多头损失在 tinyshakespeare 上训练。比较接受率 vs 截断同一模型制成的原始草稿。
4. **困难。** 实现回滚：从 10 个 token 的前缀 KV 缓存开始，输入 5 个草稿 token，模拟在位置 3 拒绝。验证你的缓存在下一次迭代中正确匹配"前缀 + 前 2 个被接受的草稿"。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 草稿模型 | "便宜的那个" | 提出候选 token 的较小模型；通常比验证器便宜 10–50 倍。 |
| 验证器 | "大的那个" | 我们保持其分布的目标模型；每步推测运行一次。 |
| 接受率（α） | "草稿对多少次" | 验证器接受草稿的每 token 概率。典型值 0.7–0.9。 |
| 残差分布 | "拒绝时的备选" | `(q - p)_+` 归一化；在拒绝时从该分布采样保持验证器的分布。 |
| Bonus token | "免费的那个" | 当所有 N 个草稿被接受时，从验证器的下一步分布采样一个。 |
| Medusa | "无草稿的推测" | 验证器上的多个 LM head 并行预测位置 t+1..t+k。 |
| EAGLE | "隐藏状态草稿" | 基于验证器最后一层隐藏状态的极小 transformer 草稿。 |
| 前瞻解码 | "雅可比迭代" | 使用不动点迭代的自我推测；无草稿模型。 |
| 树注意力 | "一次验证多个候选" | 分支验证，同时考虑多个草稿续篇。 |
| KV 回滚 | "撤销被拒的草稿" | 临时 KV 缓冲区；接受时提交，拒绝时丢弃。 |

## 延伸阅读

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 核心算法与等价定理。
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — 同时期的介绍；干净的伯努利拒绝证明。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa 论文；树注意力验证。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1；隐藏状态条件草稿。
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2；动态树深度。
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3。
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — 前瞻，无草稿方法。
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 所有四种策略的生产参考。
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) — EAGLE-1/2/3 的参考代码。