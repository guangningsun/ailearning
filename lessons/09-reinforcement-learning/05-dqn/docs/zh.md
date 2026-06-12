# 深度 Q 网络（DQN）

> 2013：Mnih 用原始像素训练了一个 Q 学习网络，在七款 Atari 游戏上击败了所有经典 RL 智能体。2015：扩展到 49 款游戏，发表在 Nature，引发了深度 RL 时代。DQN 是 Q-learning 加上三个让函数逼近稳定的技巧。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 3 · 03（反向传播）、阶段 9 · 04（Q-learning、SARSA）
**时间：** 约 75 分钟

## 问题

表格型 Q-learning 需要为每一个（状态、动作）对存储一个 Q 值。国际象棋棋盘约有 10⁴³ 个状态。一帧 Atari 画面是 210×160×3 = 100,800 个特征。表格型 RL 在数千个状态时就已经崩溃了，更别说数十亿个。

解决方案回头看很明显：用神经网络替代 Q 表，即 `Q(s, a; θ)`。但这个"事后诸葛亮"花了数十年才出现：Q-learning 加上朴素的函数逼近在"致命三件套"下会发散——函数逼近 + 自举 + 离策略学习。Mnih 等人（2013, 2015）找到了三个工程技巧来稳定学习：

1. **经验回放** 解关联转移。
2. **目标网络** 冻结自举目标。
3. **奖励裁剪** 归一化梯度量级。

Atari 上的 DQN 是第一个用单一架构、同一组超参数从原始像素解决数十个控制问题的方案。此后所有"深度 RL"——DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57——都建立在这三个技巧之上。

## 概念

![DQN 训练循环：环境、回放缓冲区、在线网络、目标网络、Bellman TD 损失](../assets/dqn.svg)

**目标函数。** DQN 在神经 Q 函数上最小化单步 TD 损失：

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = 在线网络，每步通过梯度下降更新。`θ^-` = 目标网络，定期从 `θ` 复制（约每 10,000 步）。`D` = 过去转移的回放缓冲区。

**三个技巧，按重要性排序：**

**经验回放。** 一个约 10⁶ 条转移的环形缓冲区。每步训练随机均匀采样一个小批量。这打破了时间相关性（连续帧几乎相同），让网络能多次从稀有的奖励转移中学习，并解关联了连续的梯度更新。没有它，在 Atari 上使用神经网络的同策略 TD 会发散。

**目标网络。** 在 Bellman 方程两边使用同一个网络 `Q(·; θ)` 会让目标在每次更新时移动——"追自己的尾巴"。解决方案：保留第二个网络 `Q(·; θ^-)` 并冻结权重。每隔 `C` 步，将 `θ → θ^-`。这在数千个梯度步内稳定了回归目标。软更新 `θ^- ← τ θ + (1-τ) θ^-`（用于 DDPG、SAC）是一种更平滑的变体。

**奖励裁剪。** Atari 奖励量级从 1 到 1000+ 不等。裁剪到 `{-1, 0, +1}` 可以防止任何单一游戏主导梯度。当奖励量级本身很重要时这是错误的；但对 Atari 来说只有符号重要，所以没问题。

**Double DQN。** Hasselt（2016）修复了最大化偏差：用在线网络*选择*动作，目标网络*评估*动作。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

直接替换，一致地更好。默认使用。

**其他改进（Rainbow, 2017）：** 优先回放（更频繁地采样高 TD 误差的转移）、双流架构（分离 `V(s)` 和优势头）、噪声网络（学习型探索）、n 步返回、分布式 Q（C51/QR-DQN）、多步自举。每个都增加几个百分点；收益大致可叠加。

## 构建

这里的代码是纯标准库、无 NumPy——我们在小型连续 GridWorld 上使用手动编写的单隐藏层 MLP，所以每步训练在微秒级完成。该算法与大规模 Atari DQN 完全相同。

### 第 1 步：回放缓冲区

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari 约需 50,000 容量；我们的玩具环境 5,000 就够了。

### 第 2 步：一个微型 Q 网络（手动 MLP）

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

前向传播：线性 → ReLU → 线性。这就是整个网络。

### 第 3 步：DQN 更新

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

形式上与第 04 课的 Q-learning 相同，但有两点不同：(a) 通过可微的 `Q(·; θ)` 反向传播，而不是查表，(b) 目标使用 `Q(·; θ^-)`。

### 第 4 步：外层循环

每个 episode，在 `Q(·; θ)` 上执行 ε-贪婪，将转移推入缓冲区，采样一个小批量，执行一步梯度，每隔 `C` 步同步 `θ^- ← θ`。模式如下：

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

在我们的 16 维 one-hot 状态的小型 GridWorld 上，智能体在约 500 个 episode 内学习到接近最优的策略。在 Atari 上，将规模扩展到 2 亿帧，并添加 CNN 特征提取器。

## 陷阱

- **致命三件套。** 函数逼近 + 离策略 + 自举可能导致发散。DQN 用目标网络 + 回放缓解；不要移除任何一个。
- **探索。** ε 必须衰减，通常从前 10% 训练期的 1.0 衰减到 0.01。没有足够的早期探索，Q 网络会收敛到局部盆地。
- **过估计。** 对噪声 Q 取 `max` 会向上偏移。在生产环境中始终使用 Double DQN。
- **奖励尺度。** 裁剪或归一化奖励；梯度量级与奖励量级成正比。
- **回放缓冲区冷启动。** 在缓冲区有数千条转移之前不要训练。早期在约 20 个样本上的梯度会过拟合。
- **目标同步频率。** 太频繁 ≈ 没有目标网络；太不频繁 ≈ 陈旧目标。Atari DQN 使用 10,000 步环境交互。经验法则：每训练视界的约 1/100 同步一次。
- **观察预处理。** Atari DQN 堆叠 4 帧来构成马尔可夫状态。任何有速度信息的环境需要帧堆叠或循环状态。

## 使用

2026 年，DQN 很少是最先进的，但仍然是离策略算法的参考：

| 任务 | 推荐方法 | 为什么不选 DQN？ |
|------|----------|--------------------|
| 离散动作 Atari 类 | Rainbow DQN 或 Muesli | 同一框架，更多技巧。 |
| 连续控制 | SAC / TD3（阶段 9 · 07） | DQN 没有策略网络。 |
| 同策略 / 高吞吐量 | PPO（阶段 9 · 08） | 无回放缓冲区；更容易扩展。 |
| 离线 RL | CQL / IQL / Decision Transformer | 保守 Q 目标，无自举爆炸。 |
| 大离散动作空间（推荐系统） | 带动作嵌入的 DQN，或 IMPALA | 可以；装饰更重要。 |
| LLM RL | PPO / GRPO | 序列级，不是步级；损失函数不同。 |

这些经验仍然有效。回放和目标网络出现在 SAC、TD3、DDPG、SAC-X、AlphaZero 的自玩缓冲区，以及每个离线 RL 方法中。奖励裁剪作为 PPO 的优势归一化延续下来。该架构是蓝图。

## 交付

保存为 `outputs/skill-dqn-trainer.md`：

```markdown
---
name: dqn-trainer
description: 为离散动作 RL 任务生成 DQN 训练配置（缓冲区、目标同步、ε 调度、奖励裁剪）。
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

给定一个离散动作环境（观察形状、动作数量、视界、奖励尺度），输出：

1. 网络。架构（MLP / CNN / Transformer）、特征维度、深度。
2. 回放缓冲区。容量、小批量大小、预热大小。
3. 目标网络。同步策略（每 C 步硬同步或软同步 τ）。
4. 探索。ε 起始 / 结束 / 调度长度。
5. 损失。Huber vs MSE、梯度裁剪值、奖励裁剪规则。
6. Double DQN。默认开启，除非有明确理由禁用。

拒绝交付没有目标网络、没有回放缓冲区、或 ε 保持在 1 的 DQN。拒绝连续动作任务（转至 SAC / TD3）。标记任何每步均值奖励范围 > 10× 的情况，需要裁剪或尺度归一化。
```

## 练习

1. **简单。** 运行 `code/main.py`。绘制每个 episode 的返回曲线。运行均值超过 -10 需要多少个 episode？
2. **中等。** 禁用目标网络（对 Bellman 目标的两边都使用在线网络）。测量训练不稳定性——返回是否振荡或发散？
3. **困难。** 添加 Double DQN：用在线网络选择 `argmax a'`，目标网络评估。在有噪声奖励的 GridWorld 上，比较 1,000 个 episode 后有/无 Double DQN 时 `Q(s_0, best_a)` 与真实 `V*(s_0)` 的偏差。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|-----------------------|
| DQN | "深度 Q 学习" | Q-learning 与神经 Q 函数、回放缓冲区和目标网络。 |
| 经验回放 | "打乱的转移" | 每梯度步均匀采样的环形缓冲区；解关联数据。 |
| 目标网络 | "冻结的自举" | 用于 Bellman 目标的 Q 的定期复制；稳定训练。 |
| 致命三件套 | "RL 为什么发散" | 函数逼近 + 自举 + 离策略 = 无收敛保证。 |
| Double DQN | "修复最大化偏差" | 在线网络选择动作，目标网络评估。 |
| Dueling DQN | "V 和 A 头" | 分解 Q = V + A - mean(A)；相同输出，更好的梯度流。 |
| Rainbow | "所有技巧" | DDQN + PER + dueling + n-step + noisy + 分布式合一。 |
| PER | "优先回放" | 按 TD 误差量级比例采样转移。 |

## 延伸阅读

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — 引发深度 RL 的 2013 年 NeurIPS  workshop 论文。
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Nature 论文，49 游戏 DQN。
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN。
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — dueling DQN。
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — 堆叠技巧论文。
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — 清晰的现代讲解。
- [Sutton & Barto (2018). 第 9 章 — 带逼近的同策略预测](http://incompleteideas.net/book/RLbook2020.pdf) — "致命三件套"（函数逼近 + 自举 + 离策略）的教材论述，DQN 的目标网络和回放缓冲区正是为驯服它而设计。
- [CleanRL DQN 实现](https://docs.cleanrl.dev/rl-algorithms/dqn/) — 消融研究中使用的参考单文件 DQN；适合与本课从零实现的版本对照阅读。