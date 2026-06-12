# Actor-Critic — A2C 与 A3C

> REINFORCE 有噪声。加入一个学习 `V̂(s)` 的 critic，从回报中减去它，得到一个期望相同但方差低得多的优势。这就是 actor-critic。A2C 同步运行它；A3C 跨线程运行它。两者是 2015 年之后所有现代深度强化学习方法的思维模型。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 9 · 04（TD 学习）、阶段 9 · 06（REINFORCE）
**时间：** 约 75 分钟

## 问题

原始 REINFORCE 能用，但方差很糟糕。蒙特卡洛回报 `G_t` 在不同回合之间可能相差 10 倍。用这种噪声乘以 `∇ log π` 再取平均，产生的梯度估计器需要数千个回合才能把策略移动同样距离，而 DQN 的更新次数要少得多。

方差来自于使用原始回报。如果你从中减去一个关于状态的基线 `b(s_t)` —— 任何关于状态的函数，包括一个学到的值 —— 期望不变，但方差下降了。最可行的基线是 `V̂(s_t)`。现在乘以 `∇ log π` 的量就是**优势**：

`A(s, a) = G - V̂(s)`

如果一个动作产生的回报高于平均，则它是好的；如果低于平均，则是坏的。带有学习 critic 的 REINFORCE 就是 **actor-critic**。Critic 给 actor 一个低方差的"老师"。这是 2015 年之后所有深度策略方法的基础（A2C、A3C、PPO、SAC、IMPALA）。

## 概念

![Actor-critic：策略网络 + 价值网络，TD 残差作为优势](../assets/actor-critic.svg)

**两个网络，一个共享的损失：**

- **Actor** `π_θ(a | s)`：策略。采样以执行动作。用策略梯度训练。
- **Critic** `V_φ(s)`：估计从该状态开始的期望回报。训练以最小化 `(V_φ(s) - target)²`。

**优势。** 两种标准形式：

- *MC 优势：* `A_t = G_t - V_φ(s_t)`。无偏，方差较高。
- *TD 优势：* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`。有偏（使用 `V_φ`），但方差低得多。也称为 **TD 残差** `δ_t`。

**n 步优势。** 在两者之间插值：

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` 是纯 TD。`n = ∞` 是 MC。大多数实现对 Atari 使用 `n = 5`，对 MuJoCo 上的 PPO 使用 `n = 2048`。

**广义优势估计（GAE）。** Schulman 等人（2016）提出了对所有 n 步优势的指数加权平均：

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

其中 `λ ∈ [0, 1]`。`λ = 0` 是 TD（低方差，高偏差）。`λ = 1` 是 MC（高方差，无偏）。`λ = 0.95` 是 2026 年的默认值——调参直到偏差/方差的旋钮到你想要的位置。

**A2C：同步优势 actor-critic。** 在 `N` 个并行环境中收集 `T` 步。为每步计算优势。在组合批次上更新 actor 和 critic。重复。A3C 的更简单、更可扩展的兄弟版本。

**A3C：异步优势 actor-critic。** Mnih 等人（2016）。生成 `N` 个工作线程，每个运行一个环境。每个工作线程在自己的 rollout 上本地计算梯度，然后异步地将它们应用到共享的参数服务器。不需要回放缓冲区——工作线程通过运行不同的轨迹去相关。A3C 证明了你可以大规模地在 CPU 上训练。在 2026 年，基于 GPU 的 A2C（批处理并行环境）占主导地位，因为 GPU 需要大批量。

**组合损失。**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

三个项：策略梯度损失、价值回归、熵奖励。`c_v ~ 0.5`、`c_e ~ 0.01` 是标准的起始点。

## 构建

### 第 1 步：Critic

线性 critic `V_φ(s) = w · features(s)`，用 MSE 更新：

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

在表格化环境中，critic 在几百个回合内收敛。在 Atari 上，用共享的 CNN 主干 + value head 替换线性 critic。

### 第 2 步：n 步优势

给定长度为 `T` 的 rollout 和一个自助法得到的最终 `V(s_T)`：

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` 是 critic 的目标。`advantages` 是乘以 `∇ log π` 的量。

### 第 3 步：组合更新

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

On-policy，每个更新用一个 rollout，actor 和 critic 使用不同的学习率。

### 第 4 步：并行化（A3C vs A2C）

- **A3C：** 启动 `N` 个线程。每个运行自己的环境和自己独立的前向传播。定期将梯度更新推送到共享的主节点。主节点不加锁——竞争是允许的，它们只是增加了噪声。
- **A2C：** 在单个进程中运行 `N` 个环境实例，将观测堆叠成 `[N, obs_dim]` 的批次，批次前向传播，批次反向传播。更高的 GPU 利用率，确定性，更容易理解。2026 年的默认选择。

我们的演示代码是单线程的以保持清晰；重写为批处理 A2C 只需要三行 numpy。

## 陷阱

- **在 actor 梯度之前 critic 存在偏差。** 如果 critic 是随机的，它的基线没有信息量，你就是在纯噪声上训练。在开启策略梯度之前，先让 critic 预热几百步，或者使用较小的 actor 学习率。
- **优势归一化。** 将优势归一化到均值为 0、方差为 1 per batch。以极低成本极大地稳定训练。
- **共享主干。** 在图像输入上，对 actor 和 critic 使用共享的特征提取器。各自的头。共享特征在两个损失上"搭便车"。
- **On-policy 约束。** A2C 将数据用于恰好一次更新。更多的话你的梯度就会有偏（重要性采样修正正是 PPO 所添加的）。
- **熵崩溃。** 如果没有 `c_e > 0`，策略在几百次更新内就变得接近确定性，不再探索。
- **奖励尺度。** 优势的大小取决于奖励尺度。归一化奖励（例如，用运行时标准差除）以保持跨任务梯度大小一致。

## 使用

A2C/A3C 在 2026 年很少是最终选择，但它们是后续所有方法的架构基础：

| 方法 | 与 A2C 的关系 |
|--------|----------------|
| PPO | A2C + 剪切的重要性比率，用于多轮更新 |
| IMPALA | A3C + V-trace 离策略修正 |
| SAC（阶段 9 · 07） | 带软价值 critic 的离策略 A2C（下一课） |
| GRPO（阶段 9 · 12） | 没有 critic 的 A2C——组相对优势 |
| DPO | A2C 坍缩为偏好排序损失，无需采样 |
| AlphaStar / OpenAI Five | 带联盟训练 + 模仿预训练的 A2C |

如果你在 2026 年的论文中看到"优势"，想想 actor-critic。

## 交付

保存为 `outputs/skill-actor-critic-trainer.md`：

```markdown
---
name: actor-critic-trainer
description: 为给定环境生成 A2C / A3C / GAE 配置，包含优势估计和损失权重。
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

给定一个环境和计算预算，输出：

1. 并行化。A2C（GPU 批处理）vs A3C（CPU 异步）以及工作进程数量。
2. Rollout 长度 T。每个环境每次更新的步数。
3. 优势估计器。n 步或 GAE(λ)；指定 λ。
4. 损失权重。`c_v`（价值）、`c_e`（熵）、梯度裁剪。
5. 学习率。Actor 和 critic（如果使用的话分开设置）。

拒绝在 horizon > 1000 的环境中使用单工作进程 A2C（太 on-policy，太慢）。拒绝在不进行优势归一化的情况下交付。任何 `c_e = 0` 且观测熵 < 0.1 的运行都标记为熵崩溃。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上用 MC 优势（`G_t - V(s_t)`）训练 actor-critic。与第 06 课中带运行均值基线的 REINFORCE 比较样本效率。
2. **中等。** 切换到 TD 残差优势（`r + γ V(s') - V(s)`）。测量优势批次的方差。下降了多少？
3. **困难。** 实现 GAE(λ)。遍历 `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`。绘制最终回报与样本效率的关系图。对于这个任务，偏差/方差的最佳平衡点在哪里？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| Actor | "策略网络" | `π_θ(a|s)`，通过策略梯度更新。 |
| Critic | "价值网络" | `V_φ(s)`，通过 MSE 回归到回报 / TD 目标来更新。 |
| 优势 | "比平均水平好多少" | `A(s, a) = Q(s, a) - V(s)` 或其估计量。乘以 `∇ log π` 的因子。 |
| TD 残差 | "δ" | `δ_t = r + γ V(s') - V(s)`；一步优势估计。 |
| GAE | "插值旋钮" | n 步优势的指数加权求和，用 `λ` 参数化。 |
| A2C | "同步 actor-critic" | 跨环境批处理；每个 rollout 一步梯度更新。 |
| A3C | "异步 actor-critic" | 工作线程将梯度推送到共享参数服务器。原始论文；2026 年较少见。 |
| Bootstrap | "在边界用 V" | 截断 rollout，加上 `γ^n V(s_{t+n})` 来闭合求和。 |

## 延伸阅读

- [Mnih 等人（2016）. Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C，原始异步 actor-critic 论文。
- [Schulman 等人（2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE。
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 基础；当 critic 是神经网络时与第 9 章函数逼近结合阅读。
- [Espeholt 等人（2018). IMPALA](https://arxiv.org/abs/1802.01561) — 可扩展的分布式 actor-critic，带 V-trace 离策略修正。
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — 值得阅读的生产级 A2C/PPO 实现。
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — 两次时间尺度 actor-critic 分解的基础收敛结果。