# 蒙特卡洛方法——从完整回合中学习

> 动态规划需要模型。蒙特卡洛只需要回合。运行策略，观察回报，求平均。RL 中最朴素的思想——也是解锁下游一切的那把钥匙。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 9 · 01（MDP）、阶段 9 · 02（动态规划）
**时间：** 约 75 分钟

## 问题

动态规划很优雅，但它假设你可以查询每个状态和动作的 `P(s' | s, a)`。现实世界中几乎没有任何东西能满足这个条件。机器人无法解析关节扭矩后相机像素的分布。定价算法无法对每一种可能的客户反应做积分。一个 LLM 无法枚举一个 token 之后所有可能的续写。

你需要一种只需要能够从环境*采样*的方法。运行策略，得到一条轨迹 `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`。用它来估计价值。这就是蒙特卡洛。

从 DP 到 MC 的转变在哲学上很重要：我们从*已知模型 + 精确备份*走向*采样 rollout + 平均回报*。方差变大了，但适用性爆炸了。此后每一节 RL 算法——TD、Q-learning、REINFORCE、PPO、GRPO——本质上都是一个蒙特卡洛估计器，有时还叠了一层 bootstrapping。

## 概念

![蒙特卡洛：rollout，计算回报，平均；首次访问 vs 每次访问](../assets/monte-carlo.svg)

**一句话概括核心思想：** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)`，其中 `G^{(i)}(s)` 是策略 `π` 下每次访问 `s` 时观察到的回报。

**首次访问 vs 每次访问 MC。** 给定一个多次访问状态 `s` 的回合，首次访问 MC 只计算第一次访问的回报；每次访问 MC 统计所有访问。两者在极限下都是无偏的。首次访问更易于分析（iid 样本）。每次访问每个回合利用更多数据，通常在实践中收敛更快。

**增量均值。** 不存储所有回报，而是更新运行均值：

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

重新整理：`V_new = V_old + α · (target - V_old)`，其中 `α = 1/n`。把 `1/n` 换成常数步长 `α ∈ (0, 1)`，就得到了一个能跟踪 `π` 变化的非平稳 MC 估计器。这一步就是从 MC 到 TD 再到现代 RL 算法的全部跨越。

**探索现在成了问题。** DP 通过枚举遍历每个状态。MC 只看到策略访问过的状态。如果 `π` 是确定性的，整个状态空间区域永远不会被采样，它们的价值估计永远停留在零。有三种修复方法，按时间顺序：

1. **探索起始（Exploring starts）。** 从随机的 (s, a) 对开始每个回合。保证覆盖；但在实践中不现实（你无法把机器人"重置"到任意状态）。
2. **ε-贪心。** 以当前 Q 为依据做贪心选择，但以概率 `ε` 随机选择一个动作。所有状态-动作对最终都会被采样到。
3. **离策略 MC。** 在行为策略 `μ` 下收集数据，通过重要性采样学习目标策略 `π`。方差高，但它是通往 DQN 等 replay-buffer 方法的桥梁。

**蒙特卡洛控制。** 评估 → 改进 → 评估，就像策略迭代一样，但评估是基于采样的：

1. 运行 `π`，得到一个回合。
2. 从观察到的回报更新 `Q(s, a)`。
3. 使 `π` 对 Q 做 ε-贪心。
4. 重复。

在温和条件下（每对状态-动作被无限次访问，`α` 满足 Robbins-Monro）以概率 1 收敛到 `Q*` 和 `π*`。

## 动手实现

### 第 1 步：rollout → (s, a, r) 列表

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

不需要模型，只需要 `env.reset()` 和 `env.step(s, a)`。接口和 gym 环境一样，只是精简了。

### 第 2 步：计算回报（逆向扫描）

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

一次遍历，`O(T)`。逆向递推 `G_t = r_{t+1} + γ G_{t+1}` 避免了重新求和。

### 第 3 步：首次访问 MC 评估

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

三行代码完成核心工作：首次访问时标记状态、增加计数、更新运行均值。

### 第 4 步：ε-贪心 MC 控制（在线策略）

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### 第 5 步：与 DP 金标准对比

你的 MC 估计 `V^π` 应该在回合数 → ∞ 时与 DP 结果一致。实践中：在 4×4 GridWorld 上运行 50,000 个回合，误差在 DP 答案的 `~0.1` 以内。

## 陷阱

- **无限回合。** MC 要求回合必须*终止*。如果你的策略可能永远循环，设置 `max_steps` 上限，并把上限视为隐式失败。随机策略下的 GridWorld 经常超时——这是正常的，只需确保正确计数。
- **方差。** MC 使用完整回报。在长回合中，方差巨大——末尾一个不幸的奖励会把 `V(s_0)` 移动相同的量。TD 方法（第 4 课）通过 bootstrapping 来削减这个问题。
- **状态覆盖。** 在全新 Q 上用贪心 MC 遇到平局时，永远只会尝试一个动作。你*必须*探索（ε-贪心、探索起始、UCB）。
- **非平稳策略。** 如果 `π` 发生变化（如 MC 控制中），旧的回报来自不同的策略。常数-α MC 能处理这个问题；样本均值 MC 不能。
- **离策略重要性采样。** 权重 `π(a|s)/μ(a|s)` 在轨迹上相乘。方差随视距爆炸。用每次决策加权 IS 限制，或切换到 TD。

## 实际使用

2026 年蒙特卡洛方法的角色：

| 使用场景 | 为什么用 MC |
|----------|------------|
| 短视距游戏（21点、德州扑克） | 回合自然终止；回报干净。 |
| 日志化策略的离线评估 | 对存储轨迹的折扣回报求平均。 |
| 蒙特卡洛树搜索（AlphaZero） | 从树叶的 MC rollout 引导选择。 |
| LLM RL 评估 | 对给定策略的采样续写计算平均奖励。 |
| PPO 中的基线估计 | 优势目标 `A_t = G_t - V(s_t)` 使用 MC 的 `G_t`。 |
| RL 教学 | 最简单且真正有效的算法——去掉 bootstrapping 才能看清核心。 |

现代深度 RL 算法（PPO、SAC）在纯 MC（完整回报）和纯 TD（一步 bootstrap）之间通过 n 步回报或 GAE 进行插值。两端都是同一估计器的实例。

## 交付

保存为 `outputs/skill-mc-evaluator.md`：

```markdown
---
name: mc-evaluator
description: 通过蒙特卡洛 rollout 评估策略，并在可能的情况下输出与 DP 对比的收敛报告。
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

给定一个环境（回合制，有 reset+step API）和一个策略，输出：

1. 方法。首次访问 vs 每次访问 MC。给出理由。
2. 回合预算。目标数量、方差诊断、期望标准误。
3. 探索计划。ε 调度（如需要）或探索起始。
4. 金标准对比。若为表格形式则输出 DP 最优 V*；否则从 Q-learning / PPO 基线给出一个界。
5. 终止检查。最长步数上限、超时处理、非终止轨迹的处理方式。

拒绝在没有有限视距上限的情况下对非回合任务运行 MC。拒绝为表格任务报告少于每状态 100 个回合的 V^π 估计。将零方差动作的任何策略标记为探索风险。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上对均匀随机策略实现首次访问 MC 评估。运行 10,000 个回合。将 `V(0,0)` 随回合数变化的曲线与 DP 答案对比绘图。
2. **中等。** 实现 ε-贪心 MC 控制，`ε ∈ {0.01, 0.1, 0.3}`。比较 20,000 个回合后的平均回报。曲线长什么样？偏差-方差权衡在哪里？
3. **困难。** 实现带重要性采样的*离策略* MC：在均匀随机策略 `μ` 下收集数据，估计确定性最优策略 `π` 的 `V^π`。比较普通 IS、每次决策 IS 和加权 IS。哪个方差最低？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|-----------------------|
| 蒙特卡洛 | "随机采样" | 通过从分布中平均 iid 样本来估计期望。 |
| 回报 `G_t` | "未来奖励" | 从步 `t` 到回合结束的折扣奖励和：`Σ_{k≥0} γ^k r_{t+k+1}`。 |
| 首次访问 MC | "每个状态只计一次" | 只有回合中第一次访问贡献价值估计。 |
| 每次访问 MC | "利用所有访问" | 每次访问都贡献；略有偏但样本效率更高。 |
| ε-贪心 | "探索噪声" | 以概率 `1-ε` 选贪心动作；以概率 `ε` 选随机动作。 |
| 重要性采样 | "对采样自错误分布进行修正" | 用 `π(a|s)/μ(a|s)` 乘积加权回报，用 `μ` 的数据估计 `V^π`。 |
| 在线策略 | "从自己的数据中学习" | 目标策略 = 行为策略。朴素 MC、PPO、SARSA。 |
| 离策略 | "从别人的数据中学习" | 目标策略 ≠ 行为策略。重要性采样 MC、Q-learning、DQN。 |

## 延伸阅读

- [Sutton & Barto (2018). 第 5 章 — 蒙特卡洛方法](http://incompleteideas.net/book/RLbook2020.pdf) —— 标准处理。
- [Singh & Sutton (1996). 用替换 Eligibility Traces 进行强化学习](https://link.springer.com/article/10.1007/BF00114726) —— 首次访问 vs 每次访问分析。
- [Precup, Sutton, Singh (2000). 离策略策略评估的 Eligibility Traces](http://incompleteideas.net/papers/PSS-00.pdf) —— 离策略 MC 与方差控制。
- [Mahmood et al. (2014). 加权重要性采样用于离策略学习](https://arxiv.org/abs/1404.6362) —— 现代低方差 IS 估计器。
- [Tesauro (1995). TD-Gammon，一个自学的西洋双陆棋程序](https://dl.acm.org/doi/10.1145/203330.203343) —— 首次大规模实证证明 MC/TD 自对弈收敛到超越人类水平；本阶段后半部分每一课的概念先驱。