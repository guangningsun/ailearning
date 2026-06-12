# 动态规划——策略迭代与价值迭代

> 动态规划是"作弊版"的强化学习。你已经知道转移函数和奖励函数；只需要反复迭代贝尔曼方程，直到 `V` 或 `π` 不再变化。它是每一个采样方法试图逼近的基准。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 9 · 01（MDPs）
**时间：** 约 75 分钟

## 问题

你有一个已知模型的 MDP：可以对任何状态-动作对查询 `P(s' | s, a)` 和 `R(s, a, s')`。库存经理知道需求分布。棋类游戏有确定性转移。GridWorld 只有四行 Python。你有一个*模型*。

无模型 RL（Q-learning、PPO、REINFORCE）是被发明来处理你没有模型的情况——你只能从环境中采样。但当你确实有模型时，有更快、更好的方法：动态规划。贝尔曼在 1957 年设计了它们。这些方法至今仍是正确性的定义：当人们说"这个 MDP 的最优策略"时，他们指的是 DP 返回的策略。

在 2026 年你仍然需要它们，原因有三。第一，RL 研究中的每一个表格环境（GridWorld、FrozenLake、CliffWalking）都用 DP 来产生金牌标准策略。第二，精确的价值让你能够*调试*采样方法：如果 Q-learning 对 `V*(s_0)` 的估计与 DP 答案相差 30%，你的 Q-learning 就有 bug。第三，现代离线 RL 和规划方法（MCTS、AlphaZero 的搜索、第 9 · 10 课中的基于模型的 RL）都在一个学习到或给定的模型上迭代贝尔曼备份。

## 概念

![策略迭代和价值迭代，并排对比](../assets/dp.svg)

**两个算法，都是在贝尔曼方程上的不动点迭代。**

**策略迭代。** 交替执行两步，直到策略不再变化。

1. *评估：* 给定策略 `π`，反复应用 `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]` 直到收敛，得到 `V^π`。
2. *改进：* 给定 `V^π`，让 `π` 对 `V^π` 贪婪：`π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`。

收敛是有保证的，因为 (a) 每次改进要么保持 `π` 不变，要么严格增加某些状态的 `V^π`，(b) 确定性策略的空间是有限的。即使对于大型状态空间，通常也只需约 5–20 次外层迭代就能收敛。

**价值迭代。** 将评估和改进合并为一次扫描。应用贝尔曼*最优性*方程：

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

重复直到 `max_s |V_{new}(s) - V(s)| < ε`。最后通过取贪婪动作来提取策略。每次迭代严格更快——没有内层评估循环——但通常需要更多迭代才能收敛。

**广义策略迭代（GPI）。** 统一的框架。价值函数和策略被锁在一个双向改进循环中；任何驱使两者都趋向相互一致的方法（异步价值迭代、修正策略迭代、Q-learning、演员-评论家、PPO）都是 GPI 的一个实例。

**为什么 `γ < 1` 很关键。** 贝尔曼算子在无穷范数下是一个 `γ`-压缩映射：`||T V - T V'||_∞ ≤ γ ||V - V'||_∞`。压缩映射意味着唯一不动点和几何收敛。去掉 `γ < 1` 你就失去这个保证——你需要有限视野或吸收性终止状态。

## 动手实现

### 第 1 步：构建 GridWorld MDP 模型

使用第 01 课中相同的 4×4 GridWorld。我们添加一个随机变体：以概率 `0.1` 智能体滑向一个随机的垂直方向。

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` 返回一个 `(s', r, p)` 列表。这就是整个模型。

### 第 2 步：策略评估

给定策略 `π(s) = {action: prob}`，反复迭代贝尔曼方程，直到 `V` 不再变化：

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### 第 3 步：策略改进

用对 `V` 贪婪的策略替换 `π`。如果 `π` 没有变化，返回——我们已到达最优。

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### 第 4 步：拼接在一起

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # 任意起点
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

在 4×4 上典型收敛：4–6 次外层迭代。输出 `V*(0,0) ≈ -6` 和一个严格减少步数的策略。

### 第 5 步：价值迭代（单循环版本）

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

相同的不动点，更少的代码行数。

## 陷阱

- **忘记处理终止状态。** 如果你对吸收状态应用贝尔曼，它仍然会选出一个"最佳动作"而不改变任何东西。用 `if s == terminal: V[s] = 0` 来保护。
- **无穷范数 vs L2 收敛。** 使用 `max |V_new - V|`，而不是平均值。理论保证是基于无穷范数的。
- **原地更新 vs 同步更新。** 原地更新 `V[s]`（高斯-塞德尔）比分离的 `V_new` 字典（雅可比）收敛更快。生产代码使用原地更新。
- **策略平局。** 如果两个动作有相同的 Q 值，`argmax` 可能在每次迭代中以不同方式打破平局，导致"策略稳定"检查振荡。使用稳定的平局打破规则（按固定顺序的第一个动作）。
- **状态空间爆炸。** DP 每次扫描是 `O(|S| · |A|)`。适用于约 10⁷ 个状态。超过这个规模，你需要函数近似（第 9 · 05 课起）。

## 实际使用

在 2026 年，DP 是正确性基准和规划器的内层循环：

| 用例 | 方法 |
|----------|--------|
| 精确求解小型表格 MDP | 价值迭代（更简单）或策略迭代（外层步骤更少） |
| 验证 Q-learning / PPO 实现 | 在玩具环境中与 DP 最优 V* 比较 |
| 基于模型的 RL（第 9 · 10 课） | 在学习的转移模型上进行贝尔曼备份 |
| AlphaZero / MuZero 中的规划 | 蒙特卡洛树搜索 = 异步贝尔曼备份 |
| 离线 RL（CQL、IQL） | 保守 Q 迭代——对 OOD 动作带惩罚的 DP |

每当有人说"最优价值函数"时，他们的意思是"DP 不动点"。当你在论文中看到 `V*` 或 `Q*` 时，想象这个循环。

## 交付物

保存为 `outputs/skill-dp-solver.md`：

```markdown
---
name: dp-solver
description: 通过策略迭代或价值迭代精确求解小型表格 MDP。报告收敛行为。
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

给定一个具有已知模型的 MDP，输出：

1. 选择。策略迭代 vs 价值迭代。理由与 |S|、|A|、γ 相关。
2. 初始化。V_0，初始策略。收敛敏感性。
3. 停止。无穷范数容差 ε。预期扫描次数。
4. 验证。精确计算 V*(s_0)。提取贪婪策略。
5. 使用。这个基准将如何用于调试/评估采样方法。

拒绝在状态空间 > 10⁷ 上运行 DP。拒绝在没有无穷范数检查的情况下声称收敛。将任何无限视野任务上 `γ ≥ 1` 标记为保证违反。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上运行价值迭代，`γ ∈ {0.9, 0.99}`。多少次扫描直到 `max |ΔV| < 1e-6`？将 `V*` 打印为 4×4 网格。
2. **中等。** 在*随机* GridWorld（滑倒概率 `0.1`）上比较策略迭代 vs 价值迭代。统计：扫描次数、墙上时间、最终 `V*(0,0)`。哪个在迭代次数上收敛更快？哪个在墙上时间上更快？
3. **困难。** 构建修正策略迭代：在评估步中只运行 `k` 次扫描而不是直到收敛。将 `V*(0,0)` 误差相对于 `k` 绘图，`k ∈ {1, 2, 5, 10, 50}`。曲线告诉你关于评估/改进权衡的什么信息？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 策略迭代 | "DP 算法" | 交替评估（`V^π`）和改进（相对于 `V^π` 的贪婪 `π`），直到策略不再变化。 |
| 价值迭代 | "更快的 DP" | 一次扫描中应用的贝尔曼最优性备份；几何收敛到 `V*`。 |
| 贝尔曼算子 | "这个递归" | `(T V)(s) = max_a Σ P (r + γ V(s'))`；在无穷范数下的 `γ`-压缩映射。 |
| 压缩映射 | "DP 收敛的原因" | 任何满足 `||T x - T y|| ≤ γ ||x - y||` 的算子 `T` 都有唯一不动点。 |
| GPI | "一切都是 DP" | 广义策略迭代：任何驱使 `V` 和 `π` 趋向相互一致的方法。 |
| 同步更新 | "雅可比风格" | 在整个扫描中使用旧的 `V`；分析清晰但较慢。 |
| 原地更新 | "高斯-塞德尔风格" | 在 `V` 更新过程中使用它；在实践中收敛更快。 |

## 延伸阅读

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) —— 策略迭代和价值迭代的标准表述。
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html) —— 压缩映射论证的严格处理。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) —— 修正策略迭代及其收敛分析。
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) —— 原始策略迭代论文。
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) —— 从 DP 到后续每一课使用的近似-DP/深度 RL 的桥梁。