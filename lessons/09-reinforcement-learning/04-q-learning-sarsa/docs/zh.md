# 时序差分——Q-Learning 与 SARSA

> 蒙特卡洛要等到回合结束。TD 每步之后通过 bootstrap 下一个价值估计来更新。Q-learning 是离策略且乐观的；SARSA 是在线策略且保守的。两者都是一行代码。两者都是本阶段每个深度 RL 方法的基石。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 9 · 01（MDP）、阶段 9 · 02（动态规划）、阶段 9 · 03（蒙特卡洛）
**时间：** 约 75 分钟

## 问题

蒙特卡洛有效，但它有两个昂贵的代价。它需要能终止的回合，而且只在最终回报出来之后才更新。如果你的回合是 1,000 步，MC 要等 1,000 步才能更新任何东西。它方差高、偏差低，在实践中收敛慢。

动态规划有相反的特点——零方差的 bootstrap 备份——但需要已知模型。

时序差分（TD）学习在两者之间取长补短。从一个单独的转移 `(s, a, r, s')` 出发，形成一步目标 `r + γ V(s')`，然后把 `V(s)` 向它推动。不需要模型。不需要完整回合。RHS 上使用近似 `V` 带来偏差，但比 MC 的方差大幅降低，且从第一步就能在线更新。

这就是现代 RL——DQN、A2C、PPO、SAC——所围绕的支点。本阶段剩余的内容都是建立在你将在本课写出的一步 TD 更新之上的函数近似和技巧。

## 概念

![Q-learning vs SARSA：离策略 max vs 在线策略 Q(s', a')](../assets/td.svg)

**V 的 TD(0) 更新：**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

括号里的量是 TD 误差 `δ = r + γ V(s') - V(s)`。它是 MC 中 `G_t - V(s_t)` 的在线版本。收敛需要 `α` 满足 Robbins-Monro（`Σ α = ∞`，`Σ α² < ∞`），且所有状态被无限次访问。

**Q-learning。** 一种用于控制的离策略 TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max` 假设从 `s'` 往后会跟随*贪心*策略，不管智能体实际采取什么动作。这种解耦使得 Q-learning 在智能体通过 ε-贪心探索的同时学习 `Q*`。Mnih et al. (2015) 将其转化为深度 Q-learning 用于 Atari（第 5 课）。

**SARSA。** 一种用于控制的在线策略 TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

名称来自元组 `(s, a, r, s', a')`。SARSA 使用智能体*实际*采取的下一个动作 `a'`，而不是贪心的 `argmax`。收敛到当前运行的 ε-贪心 `π` 对应的 `Q^π`，当 `ε → 0` 时极限情况下变成 `Q*`。

**悬崖行走差异。** 在经典的悬崖行走任务中（跌落悬崖 = 奖励 -100），Q-learning 学到沿悬崖边缘的最优路径，但偶尔会在探索时受到惩罚。SARSA 学到一条距悬崖一步之遥的更安全路径，因为它的 Q 值把探索噪声纳入了考虑。随着训练，当 `ε → 0` 时两者都达到最优。在实践中这很重要：当探索在部署时真实发生时，SARSA 的行为更保守。

**期望 SARSA。** 用 `π` 下 `Q(s', a')` 的期望值替代 `Q(s', a')`：

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

比 SARSA 方差更低（没有 `a'` 的采样），在线策略目标相同。在现代教材中通常是默认选项。

**n 步 TD 和 TD(λ)。** 通过等待 n 步再 bootstrap 在 TD(0) 和 MC 之间插值。`n=1` 是 TD，`n=∞` 是 MC。TD(λ) 用几何权重 `(1-λ)λ^{n-1}` 对所有 n 做平均。大多数深度 RL 使用 n 在 3 到 20 之间。

## 动手实现

### 第 1 步：SARSA 在 ε-贪心策略上

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

八行代码。与 Q-learning 的*唯一*区别在目标行。

### 第 2 步：Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` 将目标与行为解耦。这一个符号便是在线策略与离策略之间的全部差异。

### 第 3 步：学习曲线

跟踪每 100 个回合的平均回报。Q-learning 在简单确定性 GridWorld 上收敛更快；SARSA 在悬崖行走上更保守。在 `code/main.py` 的 4×4 GridWorld 上，两者在大约 2,000 个回合后都接近最优，`α=0.1, ε=0.1`。

### 第 4 步：与 DP 真值对比

运行价值迭代（第 2 课）得到 `Q*`。检查 `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`。一个健康的表格 TD 智能体在 4×4 GridWorld 上运行 10,000 个回合后误差在 `~0.5` 以内。

## 陷阱

- **初始 Q 值很重要。** 乐观初始化（对负奖励任务 `Q = 0`）鼓励探索。悲观初始化可能让贪心策略永远被困住。
- **α 调度。** 常数 `α` 对非平稳问题没问题。衰减 `α_n = 1/n` 在理论上保证收敛但实践中太慢——将 `α` 固定在 `[0.05, 0.3]`，监控学习曲线。
- **ε 调度。** 从高开始（`ε=1.0`），衰减到 `ε=0.05`。"GLIE"（无限探索下极限贪心）是收敛条件。
- **Q-learning 中的 max 偏误。** 当 Q 有噪声时，`max` 算子有向上偏误。导致过估计——Hasselt 的 Double Q-learning（第 5 课的 DDQN 使用了它）用两个 Q 表修复了这个问题。
- **非终止回合。** TD 可以在没有终止的情况下学习，但你需要么设置步数上限，么在上限处正确处理 bootstrap。标准做法：将上限视为非终止，继续 bootstrap。
- **状态哈希。** 如果状态是元组/张量，使用可哈希的键（元组而非列表；四舍五入的浮点元组而非原始值）。

## 实际使用

2026 年 TD 领域格局：

| 任务 | 方法 | 原因 |
|------|------|------|
| 小型表格环境 | Q-learning | 直接学习最优策略。 |
| 在线策略安全关键场景 | SARSA / 期望 SARSA | 探索期间更保守。 |
| 高维状态 | DQN（阶段 9 · 05） | 用 replay 和 target net 的神经网络 Q 函数。 |
| 连续动作 | SAC / TD3（阶段 9 · 07） | 在 Q 网络上做 TD 更新；策略网络输出动作。 |
| LLM RL（基于奖励模型） | PPO / GRPO（阶段 9 · 08、12） | 通过 GAE 用 TD 风格的优势做 Actor-Critic。 |
| 离线 RL | CQL / IQL（阶段 9 · 08） | 带保守正则化的 Q-learning。 |

你在 2026 年论文中读到的 90% 的"RL"都是 Q-learning 或 SARSA 的某种变体。在把这些表格更新写进手指之前，不要去读更深入的东西。

## 交付

保存为 `outputs/skill-td-agent.md`：

```markdown
---
name: td-agent
description: 为表格或小特征 RL 任务在 Q-learning、SARSA、期望 SARSA 之间选择。
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

给定一个表格或小特征环境，输出：

1. 算法。Q-learning / SARSA / 期望 SARSA / n 步变体。一句话理由，关联在线策略 vs 离策略和方差。
2. 超参数。α、γ、ε、衰减调度。
3. 初始化。Q_0 值（乐观 vs 零）及其理由。
4. 收敛诊断。目标学习曲线、若可做 DP 则检查 `|Q - Q*|`。
5. 部署注意事项。推理时探索行为如何？是否需要 SARSA 的保守性？

拒绝将表格 TD 应用于超过 10⁶ 状态的空间。拒绝交付没有 max 偏误警告的 Q-learning 智能体。将用 ε=1.0 全程训练的智能体标记为问题（没有利用阶段）。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现 Q-learning 和 SARSA。绘制 2,000 个回合的学习曲线（每 100 回合平均回报）。谁收敛更快？
2. **中等。** 构建一个悬崖行走环境（4×12，最后一行是悬崖，奖励 -100，重置到起点）。比较 Q-learning 和 SARSA 的最终策略。截图每种策略走的路径。哪个更靠近悬崖？
3. **困难。** 实现 Double Q-learning。在带噪声奖励的 GridWorld（每步奖励加上高斯噪声 σ=5）上，展示 Q-learning 对 `V*(0,0)` 有意义的过估计，而 Double Q-learning 不会。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|-----------------------|
| TD 误差 | "更新信号" | `δ = r + γ V(s') - V(s)`，bootstrapped 残差。 |
| TD(0) | "一步 TD" | 每步转移后只用下一个状态的估计来更新。 |
| Q-learning | "离策略 RL 入门" | 带 `max` over 下一状态动作的 TD 更新；不管行为策略学习 `Q*`。 |
| SARSA | "在线策略 Q-learning" | 使用实际下一动作的 TD 更新；学习当前 ε-贪心 π 对应的 `Q^π`。 |
| 期望 SARSA | "低方差 SARSA" | 用 π 下 `a'` 的期望替代采样的 `a'`。 |
| GLIE | "正确的探索调度" | 极限无限探索下的贪心；Q-learning 收敛的条件。 |
| Bootstrapping | "在目标中使用当前估计" | TD 区别于 MC 的地方。偏差的来源，但大幅降低方差。 |
| 最大化偏误 | "Q-learning 过估计" | 对噪声估计取 `max` 是向上偏误的；Double Q-learning 修复了这个问题。 |

## 延伸阅读

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) —— 原始论文及收敛性证明。
- [Sutton & Barto (2018). 第 6 章 — 时序差分学习](http://incompleteideas.net/book/RLbook2020.pdf) —— TD(0)、SARSA、Q-learning、期望 SARSA。
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) —— 最大化偏误的修复。
- [Seijen, Hasselt, Whiteson, Wiering (2009). 期望 SARSA 的理论与实证分析](https://ieeexplore.ieee.org/document/4927542) —— 期望 SARSA 的动机。
- [Rummery & Niranjan (1994). 使用连接主义系统在线 Q 学习](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) —— 发明 SARSA 一词的论文（当时称为"改进的连接主义 Q 学习"）。
- [Sutton & Barto (2018). 第 7 章 — n 步 Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) —— 将 TD(0) 推广到 TD(n)，从 Q-learning 到 eligibility traces 再到后来 PPO 中 GAE 的路径。