# 投机解码与 EAGLE-3

> 第 7 阶段 · 第 16 课证明了数学原理：Leviathan 拒绝规则精确保持验证器的分布。本课是 2026 年生产级投机解码的训练栈视角。EAGLE-3 将 draft 模型从一个廉价的近似方案转变为一个针对验证器自身隐藏状态训练的目的构建的小型网络，并添加了一个在训练时对齐训练分布和推理分布的测试循环。结果：端到端加速 3× 到 6.5×，聊天场景下单 token 接受率超过 0.9，无需在分布上做权衡。2026 年每一个生产推理栈都默认搭载它。

**类型：** 构建型
**语言：** Python（标准库）
**前置条件：** 第 7 阶段 · 16（投机解码数学）、第 10 阶段 · 12（推理优化）
**时间：** 约 75 分钟

## 学习目标

- 一句话陈述 Leviathan 定理，并证明投机循环产生的样本与验证器直接采样的分布相同。
- 走过从 vanilla 投机解码（Leviathan 2023）到 EAGLE、EAGLE-2 和 EAGLE-3 的两年发展历程，指出每一步移除的确切限制。
- 根据接受率 `α` 和 draft-to-verifier 成本比 `c` 计算预期加速比，并为每种配置选择最优的 draft 长度 `N`。
- 从零实现完整的投机循环：draft、验证、从残差中拒绝采样、回滚 KV 缓存（在拒绝时）、在全接受时发出奖励 token。

## 问题

在 H100 上对 70B 模型做自回归解码，速度大约是每秒 35 个 token。GPU 远未饱和。内存带宽是瓶颈：每个 token 需要从 HBM 加载 70B 权重，做一步算术运算，产出一个浮点数。计算单元大部分时间处于空闲状态。

投机解码把这变成了一个你可以真正解决的吞吐问题。一个廉价的 draft 在 N 次小前向传播中提出 N 个 token。验证器在前缀加上所有 N 个 draft 之后运行一次。如果验证器在位置 i 的分布与 draft 一致（以我们将要精确化的统计意义上），则接受；如果不一致，则拒绝并从残差分布中采样一个修正。在一次大模型前向传播中，最多可以接受 N+1 个 token，而不是一个。

重要的定理是 Leviathan、Kalman、Matias（ICML 2023）：输出分布与从验证器直接采样的分布完全相同。不是近似，是完全相同。这就是投机解码在生产环境中可以接受的全部原因——这是一个纯粹的低延迟优化，没有质量权衡。

第 7 阶段 · 第 16 课给你的是数学。本课给你的是训练栈。一个好的 draft 比一个廉价的 draft 多带来 2× 的加速。EAGLE、EAGLE-2 和 EAGLE-3（Li 等，2024–2025）把"draft = 同一个模型的小版本"变成了一门精确的工程学科。2026 年的生产推理服务器默认使用 EAGLE-3。

## 概念

### 不变式：Leviathan 拒绝采样

设 `p(t)` 是给定某个前缀的下一个 token 的 draft 分布，`q(t)` 是验证器的分布。从 `p` 采样一个 draft token `d ~ p`。以概率 `min(1, q(d) / p(d))` 接受。拒绝时，从残差分布 `(q - p)_+ / ||(q - p)_+||_1` 采样。结果样本按 `q` 分布。这在 `p` 有多差的情况下都成立——越差就越频繁拒绝，但输出保持精确。

使用一次验证器在前缀 + d_1 + ... + d_N 上的前向传播，将 N 个这样的调用堆叠在一起。验证器同时返回 `q_1, q_2, ..., q_{N+1}`。从左向右遍历。在位置 j 的第一次拒绝时，从 `residual(q_j, p_j)` 采样并停止。在全接受时，从 `q_{N+1}` 采样一个奖励 token。

### 什么决定了加速比

设 `α` 为每个 draft token 的预期接受率。设 `c = cost(draft) / cost(verifier)` 为成本比。每个验证器前向传播接受的 token 预期数量为：

```
E[accepted] = (1 - α^(N+1)) / (1 - α)
```

每个被接受 token 的预期总墙钟时间是 `(N * c + 1) / E[accepted]`。对其关于 `N` 最小化得到最佳点。对于 `α = 0.8, c = 0.05`：最优 `N` 在 5–7 左右，加速比 3.2×。对于 `α = 0.95, c = 0.02`：最优 `N` 在 8–10 左右，加速比接近 5×。

最大的杠杆是 `α`。在固定 `N = 5` 的情况下，从 `α = 0.6`（vanilla draft）到 `α = 0.9`（EAGLE-3），将每个验证器前向传播的预期接受 token 数从 2.2 提升到 4.1。同样的验证器，吞吐几乎翻倍。

### 两年发展历程

**Vanilla 投机（Leviathan，2023）。** Draft 模型是同一家族中独立训练的更小的 LLM。容易接线，`α ≈ 0.6`，最好情况下加速比约 2×。

**EAGLE-1（Li 等，2024）。** Draft 是一个微小的 transformer——通常是一层或两层——以验证器的最后一层隐藏状态为输入，直接预测下一个 token。因为 draft 看到了验证器的特征表示，它的分布更接近验证器。`α` 上升到 0.7–0.8。

**EAGLE-2（Li 等，2024）。** 添加动态 draft 树：不是提议一个单独的 N 个 token 序列，而是提议一棵候选小树，用一次前向传播中的验证器对每个进行评分（树注意力），然后走概率最高的路径。Draft 长度在每个步骤中变为自适应的。沿接受路径的每个 token 的 `α` 超过 0.85。

**EAGLE-3（Li 等，2025，NeurIPS）。** 另有两项改动。首先，完全放弃特征预测损失——EAGLE-1/2 训练 draft 去匹配验证器的隐藏状态，这限制了数据能提供多少帮助。EAGLE-3 直接在 token 预测上进行训练。其次，训练时测试（TTT）：在 draft 训练期间，将 draft 自身之前的预测作为输入反馈回来，多步运行，与推理时的运行方式相同。这对齐了训练和测试分布，阻止了误差积累。实测加速比：在聊天上最高 6.5×，在 H100 上 SGLang 批大小 64 时吞吐提升 38%。

### KV 缓存回滚

验证在一个传播中将验证器的 KV 缓存扩展 N 条目。如果拒绝发生在位置 j，位置 j-1 之后的缓存内容现在是错误的。两种常见实现：写入临时缓冲区并在接受时提交（vLLM、TensorRT-LLM），或者保持物理 KV 缓存加上逻辑长度并在拒绝时截断。无论哪种方式，回滚成本是每层每头字节数，与前向传播成本相比可以忽略不计。

对于 EAGLE-2 树搜索，验证器使用尊重树拓扑的非因果掩码运行注意力。工程上比较繁琐，但计算是一个带有自定义掩码的标准 flash-attention 调用。

### 2026 年的 Draft 架构

| 策略 | Draft 类型 | `α` | 加速比 | 训练成本 |
|----------|-----------|-----|---------|---------------|
| Vanilla | 独立小型 LLM | 0.55-0.70 | 1.8-2.3× | 无（复用已有小型模型） |
| Medusa | 验证器上的额外 LM 头 | 0.65-0.75 | 2-3× | ~10 亿 SFT token |
| EAGLE-1 | 隐藏状态上的 1 层 transformer | 0.70-0.80 | 2.5-3× | ~600 亿 token |
| EAGLE-2 | EAGLE-1 + 动态 draft 树 | 0.80-0.88 | 3-4× | ~600 亿 token |
| EAGLE-3 | 多层特征融合 + TTT | 0.88-0.92 | 3.5-6.5× | ~600-2000 亿 token |
| Lookahead | 无 draft（Jacobi 迭代） | N/A | 1.3-1.6× | 无 |

2026 年生产环境：vLLM 和 SGLang 在可用时默认使用 EAGLE-3，否则使用 EAGLE-2。TensorRT-LLM 对于 Meta 和 NVIDIA 公共模型有最快的 Medusa 路径。llama.cpp 为 CPU 部署提供 vanilla draft。

## 构建它

参见 `code/main.py`。这是完整的 Leviathan 投机循环，包含所有组件：N 个 draft、验证器并行传播、逐位置拒绝、残差采样、奖励 token、KV 回滚，以及经验验证输出分布与从 `q` 直接采样的分布一致。

### 第 1 步：拒绝规则

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### 第 2 步：残差分布

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### 第 3 步：完整的投机步骤

`spec_step` 函数从 `p` 中 draft N 个 token，然后在一个并行的 `q` 求值中验证所有 token。对每个 draft token 应用拒绝规则，在第一次拒绝时从残差中采样修正。如果全部接受，从 `q_{N+1}` 发出一个奖励 token。

### 第 4 步：KV 回滚记账

模拟器跟踪每个 worker 的逻辑 `kv_length`。在接受 k 个 draft 时，`kv_length += k`。在位置 j 拒绝时，缓存已经写过了 j，但逻辑长度被设置为 `prefix_length + j + 1`——修正 token 之后的一位。后续读取截断到逻辑长度。

### 第 5 步：Leviathan 检查

运行 50,000 次投机步骤。计算接受 token 的经验分布。与 50,000 次从 `q` 直接采样的结果比较。卡方统计量应该远低于临界值。定理在实践中通过检验。

### 第 6 步：加速比与 α

通过以不同幅度将 `p` 从 `q` 扰开来扫描 draft 质量。测量 `α`，然后绘制每个验证器调用预期 token 数作为 `α` 和 `N` 的函数。代码打印一个表格，显示 EAGLE-3 级 draft 质量（`α ≈ 0.9`）如何在每个验证器调用中解锁 4–5 个 token。

## 使用它

使用 EAGLE-3 的生产级 `vllm serve`：

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
```

SGLang 在 H100 上批大小 64 使用 EAGLE-3：据 EAGLE-3 论文，比批大小 64 vanilla 解码的吞吐高约 1.38×。

何时使用投机解码：

- 任何交互式聊天工作负载，其中 p50 延迟比峰值吞吐更重要。
- 代码生成和结构化输出（JSON、SQL）。`α` 高于 0.9，因为目标分布高度可预测。
- 长文本生成（数千个 token）。摊销后的加速比持续支付回报。

何时不使用：

- 非常小的模型（< 3B）。Draft 不比验证器便宜多少。
- 微小批大小 1 的 CPU 部署。Draft 模型的内存开销可能不值得。
- 非常高位温的创意采样，`α` 会崩溃。

## 交付它

本课产出 `outputs/skill-eagle3-tuner.md`。给定一个推理工作负载（模型、批大小、目标延迟、任务配置文件），它推荐一种投机解码策略和调优参数（draft 家族、`N`、树深度、温度感知切换）。

## 练习

1. 运行 `code/main.py`。确认在 50,000 个样本上 Leviathan 分布检查的卡方统计量保持在 95% 临界值以下。

2. 将 `N` 从 1 扫到 10，同时保持 `α` 为 0.9 和 `c` 为 0.04。绘制每个验证器调用预期 token 数和每个 token 的实际墙钟时间。找到使墙钟时间最小的 `N`。解释曲线的形状。

3. 修改代码来模拟 EAGLE-2 树搜索：在每一步，draft 提议一个形状为 `[2, 2, 2]` 的树（八条候选路径）。验证器运行一次，得分最高的接受路径获胜。计算每叶的 `α` 和每个验证器调用的总 token 数。在等效计算下与线性链式投机解码比较。

4. 实现一个用于两条并发序列的批处理 KV 回滚模拟器。序列 A 所有 draft 都被接受；序列 B 在位置 2 拒绝。证明正确的 `kv_length` 是按序列更新的，并且没有浪费工作。

5. 阅读 EAGLE-3 论文第 4 节（训练时测试）。用两句话解释为什么没有 TTT 的朴素 draft 训练会受到曝光偏差的影响，以及为什么在训练期间将 draft 自身的预测反馈给它可以修复它。将答案与 seq2seq 文献中的 scheduled sampling 联系起来。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| Leviathan 规则 | "min(1, q over p)" | 伯努利接受/拒绝，概率为 `min(1, q(d)/p(d))`，在拒绝时从残差采样，精确保持验证器分布 |
| 残差分布 | "(q 减 p) 取正后归一化" | `(q - p)_+` 在零处截断并重新归一化——拒绝时采样的正确分布 |
| 接受率 α | "draft 多常是对的" | 在拒绝规则下每个 token 的预期伯努利成功概率；控制所有加速比计算 |
| EAGLE-1 | "隐藏状态 draft" | 微小 transformer draft，以验证器的最后一层隐藏状态为条件（Li 等，2024） |
| EAGLE-2 | "动态 draft 树" | EAGLE-1 加上在一次验证器传播中用树注意力评分的一棵候选延续树 |
| EAGLE-3 | "训练时测试" | 放弃特征预测损失，在直接 token 预测上训练，draft 在训练期间被喂入自身输出 |
| 训练时测试（TTT） | "曝光偏差修复" | 在训练期间自回归运行 draft，使训练和测试输入分布匹配——scheduled sampling 的直接类比 |
| KV 回滚 | "撤销被拒绝的 draft" | 在拒绝后将验证器的 KV 缓存重置为接受前缀长度的记账操作 |
| 奖励 token | "免费的那个" | 当所有 N 个 draft 接受时，从 `q_{N+1}` 采样一个额外的 token，不增加验证器成本 |
| 树注意力 | "一次验证多个候选" | 带有尊重 draft 树拓扑的非因果掩码的注意力；在一次前向传播中计算树中每个节点的 `q_i` |

## 进一步阅读

- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) —— 基础论文和等价定理
- [Chen 等 — Accelerating Large Language Model Decoding with Speculative Sampling (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) —— 并发独立引入，证明清晰
- [Li 等 — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) —— EAGLE-1，隐藏状态条件化的 draft
- [Li 等 — EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) —— 动态树搜索
- [Li 等 — EAGLE-3: Scaling up Inference Acceleration via Training-Time Test (arXiv:2503.01840, NeurIPS 2025)](https://arxiv.org/abs/2503.01840) —— 2026 年生产默认
- [Cai 等 — Medusa: Multiple Decoding Heads (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) —— 无 draft 的替代方案
- [vLLM Speculative Decoding 文档](https://docs.vllm.ai/en/latest/features/spec_decode.html) —— 生产参考，包含所有策略的接线方式