# 策略梯度——从零实现 REINFORCE

> 停止估计价值。直接参数化策略，计算预期回报的梯度，沿上坡方向迈步。Williams（1992）用一个定理写完了它。这就是为什么 PPO、GRPO 和每一个 LLM RL 循环存在的原因。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 3 · 03（反向传播）、阶段 9 · 03（蒙特卡洛）、阶段 9 · 04（TD 学习）
**时间：** 约 75 分钟

## 问题

Q-learning 和 DQN 参数化的是*价值*函数。你通过 `argmax Q` 选择动作。这对离散动作和离散状态没问题。但当动作连续时（`argmax` 怎么操作 10 维力矩？）或者你想要随机策略时（`argmax` 天然是确定性的），它就崩溃了。

策略梯度改为参数化*策略*。`π_θ(a | s)` 是一个输出动作分布的神经网络。从中采样以执行动作。计算预期回报关于 `θ` 的梯度。沿上坡方向迈步。没有 `argmax`。没有 Bellman 递归。只是在 `J(θ) = E_{π_θ}[G]` 上做梯度上升。

REINFORCE 定理（Williams 1992）告诉你这个梯度是可计算的：`∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`。运行一个 episode。计算回报。将每一步的 `∇ log π_θ(a | s)` 乘以回报。取平均。梯度上升。完成。

2026 年的每一个 LLM-RL 算法——PPO、DPO、GRPO——都是 REINFORCE 的改进版。在自己手指里理解它，是本阶段其余内容、以及阶段 10 · 07（RLHF 实现）和阶段 10 · 08（DPO）的前置要求。

## 概念

![策略梯度：softmax 策略、log-π 梯度、回报加权更新](../assets/policy-gradient.svg)

**策略梯度定理。** 对于任意由 `θ` 参数化的策略 `π_θ`：

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

其中 `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` 是从步骤 `t` 开始的折扣回报。期望是关于从 `π_θ` 中采样的完整轨迹 `τ`。

**证明很短。** 对 `J(θ) = Σ_τ P(τ; θ) G(τ)` 在期望下求导。使用 `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)`（对数导数技巧）。分解 `log P(τ; θ) = Σ log π_θ(a_t | s_t) + 不依赖 θ 的环境项`。环境项消失。两行代数给出定理。

**方差缩减技巧。** 原始 REINFORCE 有巨大的方差——回报是噪声，`∇ log π` 是噪声，它们的乘积噪声非常大。两种标准修复：

1. **基线减去。** 用 `G_t - b(s_t)` 替代 `G_t`，其中 `b(s_t)` 是不依赖 `a_t` 的任意基线。因为 `E[b(s_t) · ∇ log π(a_t | s_t)] = 0` 所以无偏。典型选择：`b(s_t) = V̂(s_t)`，由批评者学习 → 演员-评论家（第 07 课）。
2. **奖励到-go。** 用 `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)` 替代 `Σ_t G_t · ∇ log π_θ(a_t | s_t)`。对于给定动作，只有未来回报重要——过去奖励贡献的是零均值噪声。

综合得到：

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

这就是带基线的 REINFORCE——A2C（第 07 课）和 PPO（第 08 课）的直系祖先。

**Softmax 策略参数化。** 对于离散动作，标准选择：

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

其中 `f_θ` 是输出每个动作分数的任意神经网络。梯度有简洁的形式：

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

即所执行动作的分数减去其在策略下的期望值。

**连续动作的高斯策略。** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`。`∇ log N(a; μ, σ)` 有闭合形式。这就是阶段 9 · 07 的 SAC 所需的全部。

## 构建

### 第 1 步：softmax 策略网络

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

对表格环境使用线性策略（每个动作一个权重向量）。对 Atari 换成 CNN，保留 softmax 头。

### 第 2 步：采样和对数概率

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### 第 3 步：收集 log-prob 的 rollout

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### 第 4 步：REINFORCE 更新

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

梯度 `∇ log π(a|s) = e_a - π(·|s)`（`a` 的 one-hot 减去概率）是 softmax 策略梯度的核心。把它刻进肌肉记忆里。

### 第 5 步：基线

最近 episode 上 `G` 的运行均值足以提供方差缩减，让 4×4 GridWorld 跑起来；收敛约需 500 个 episode。将基线升级为学习的 `V̂(s)`，你就得到了演员-评论家。

## 陷阱

- **梯度爆炸。** 回报可能非常大。在乘以 `∇ log π` 之前始终将 `G` 归一化到批量上的 `~N(0, 1)`。
- **熵崩溃。** 策略过早收敛到近似确定性动作，停止探索，陷入困境。修复：在目标函数中添加熵奖励 `β · H(π(·|s))`。
- **高方差。** 原始 REINFORCE 需要数千个 episode。批评者基线（第 07 课）或 TRPO/PPO 的信任域（第 08 课）是标准修复。
- **样本效率低。** 同策略意味着每次更新后丢弃每条转移。重要性采样的离策略修正可以复用数据，但代价是方差（PPO 的比率是一个裁剪的 IS 权重）。
- **非平稳梯度。** 100 个 episode 前的相同梯度使用的是旧的 `π`。同策略方法因此每几个 rollout 更新一次。
- **信用分配。** 没有奖励到-go，过去奖励贡献噪声。始终使用奖励到-go。

## 使用

2026 年，REINFORCE 很少直接运行，但其梯度公式无处不在：

| 使用场景 | 衍生方法 |
|----------|---------------|
| 连续控制 | 带高斯策略的 PPO / SAC |
| LLM RLHF | 带 KL 惩罚的 PPO，在 token 级策略上运行 |
| LLM 推理（DeepSeek） | GRPO——带群组相对基线的 REINFORCE，无批评者 |
| 多智能体 | 集中批评者 REINFORCE（MADDPG、COMA） |
| 离散动作机器人 | A2C、A3C、PPO |
| 仅偏好设置 | DPO——重写为偏好似然损失的 REINFORCE，无需采样 |

当你在 2026 年的训练脚本中看到 `loss = -advantage * log_prob` 时，那就是带基线的 REINFORCE。整篇论文（DPO、GRPO、RLOO）都是在这行代码上叠加的方差缩减技巧。

## 交付

保存为 `outputs/skill-policy-gradient-trainer.md`：

```markdown
---
name: policy-gradient-trainer
description: 为给定任务生成 REINFORCE / 演员-评论家 / PPO 训练配置，并诊断方差问题。
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

给定一个环境（离散 / 连续动作、视界、奖励统计），输出：

1. 策略头。Softmax（离散）或高斯（连续）及参数数量。
2. 基线。无（原始）、运行均值、学习的 `V̂(s)`，或 A2C 批评者。
3. 方差控制。默认开启奖励到-go、回报归一化、梯度裁剪值。
4. 熵奖励。系数 β 及衰减调度。
5. 批量大小。每个更新的 episode 数；同策略数据新鲜度契约。

拒绝在视界 > 500 步时使用无基线 REINFORCE。拒绝用 softmax 头做连续动作控制。标记任何 β = 0 且观察到策略熵 < 0.1 的运行为熵崩溃。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上用线性 softmax 策略实现 REINFORCE。无基线训练 1,000 个 episode。绘制学习曲线；测量方差（回报的 std）。
2. **中等。** 添加运行均值基线。重新训练。将样本效率和方差与原始版本比较。基线减少了多少步才收敛？
3. **困难。** 添加熵奖励 `β · H(π)`。扫描 `β ∈ {0, 0.01, 0.1, 1.0}`。绘制最终回报和策略熵。在这个任务上，最佳点在哪里？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|-----------------------|
| 策略梯度 | "直接训练策略" | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`；从对数导数技巧推导。 |
| REINFORCE | "原始 PG 算法" | Williams (1992)；蒙特卡洛回报乘以 log-策略梯度。 |
| 对数导数技巧 | "得分函数估计器" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`；使期望的梯度可处理。 |
| 基线 | "方差缩减" | 任意从 `G` 中减去的 `b(s)`；因为 `E[b · ∇ log π] = 0` 所以无偏。 |
| 奖励到-go | "只有未来回报重要" | 用 `G_t^{from t}` 替代完整的 `G_0`；正确且低方差。 |
| 熵奖励 | "鼓励探索" | `+β · H(π(·\|s))` 项防止策略崩溃。 |
| 同策略 | "用刚看到的东西训练" | 梯度期望关于当前策略——不能直接复用旧数据。 |
| 优势 | "比平均好多少" | `A(s, a) = G(s, a) - V(s)`；带基线 REINFORCE 乘以的有符号量。 |

## 延伸阅读

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — 原始 REINFORCE 论文。
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — 带函数逼近的现代策略梯度定理。
- [Sutton & Barto (2018). 第 13 章 — 策略梯度方法](http://incompleteideas.net/book/RLbook2020.pdf) — 教材表述。
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — 清晰的教学讲解，带 PyTorch 代码。
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — 方差缩减和自然梯度视角，连接 REINFORCE 与信任域家族（TRPO、PPO）。