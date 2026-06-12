# 近端策略优化（PPO）

> A2C 每次更新后丢弃 rollout。PPO 将策略梯度包裹在剪切的重要性比率中，这样你就可以在相同数据上做 10+ 轮更新而不会导致策略爆炸。Schulman 等人（2017）。在 2026 年仍然是默认的策略梯度算法。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 9 · 06（REINFORCE）、阶段 9 · 07（Actor-Critic）
**时间：** 约 75 分钟

## 问题

A2C（第 07 课）是 on-policy：梯度 `E_{π_θ}[A · ∇ log π_θ]` 需要从**当前** `π_θ` 采样的数据。做一次更新后，`π_θ` 就变了；你用过的数据现在就 off-policy 了。重用它，你的梯度就有偏。

Rollout 是昂贵的。在 Atari 上，一个跨 8 个环境 × 128 步的 rollout = 1024 个转换和十几秒的环境时间。在一次梯度更新后就丢弃是浪费的。

信任域策略优化（TRPO，Schulman 2015）是第一个解决方案：约束每次更新，使新旧策略之间的 KL 散度保持在 `δ` 以下。理论上很干净，但每次更新需要共轭梯度求解。2026 年没人用 TRPO。

PPO（Schulman 等人 2017）用简单的剪切目标替换了硬信任域约束。额外一行代码。每个 rollout 十轮更新。不需要共轭梯度。足够好的理论保证。九年后，它仍然是所有领域的默认策略梯度算法，从 MuJoCo 到 RLHF。

## 概念

![PPO 剪切代理目标：比率在 1 ± ε 处剪切](../assets/ppo.svg)

**重要性比率。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

这是新策略与收集数据的策略的似然比率。`r_t = 1` 表示没有变化。`r_t = 2` 表示新策略采取 `a_t` 的可能性是旧策略的两倍。

**剪切代理。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

两项：

- 如果优势 `A_t > 0` 且比率试图增长超过 `1 + ε`，剪切将梯度展平——不要把一个好的动作推到旧概率的 `+ε` 以上。
- 如果优势 `A_t < 0` 且比率试图增长超过 `1 - ε`（这意味着相比被剪切的减少，我们会更多地采取一个坏动作），剪切将梯度封顶——不要把一个坏动作推到 `-ε` 以下。

`min` 处理另一个方向：如果比率已经向**有益的**方向移动，你仍然得到梯度（在你可能受损的一侧不进行剪切）。

典型的 `ε = 0.2`。将目标绘制成 `r_t` 的函数：一段分段线性的函数，在"好的一侧"有一个平的屋顶，在"坏的一侧"有一个平的地板。

**完整的 PPO 损失。**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

与 A2C 相同的 actor-critic 结构。三个系数，通常 `c_v = 0.5`、`c_e = 0.01`、`ε = 0.2`。

**训练循环。**

1. 在 `N` 个并行环境中，每个环境收集 `N × T` 个转换，每个环境 `T` 步。
2. 计算优势（GAE），将它们冻结为常数。
3. 将 `π_{θ_old}` 冻结为当前 `π_θ` 的快照。
4. 对于 `K` 轮，对于每个 `(s, a, A, V_target, log π_old(a|s))` 的小批次：
   - 计算 `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`。
   - 应用 `L^{CLIP}` + 价值损失 + 熵。
   - 梯度步。
5. 丢弃 rollout。返回步骤 1。

`K = 10` 和大小为 64 的小批次是一组标准的超参数。PPO 很稳健：精确的数字在 ±50% 范围内很少有关系。

**KL 惩罚变体。** 原始论文提出了一个使用自适应 KL 惩罚的替代方案：`L = L^{PG} - β · KL(π_θ || π_old)`，其中 `β` 根据观测到的 KL 调整。剪切版本成为主流；KL 变体在 RLHF 中保留（因为到参考策略的 KL 是一个单独的约束，你总是想要的）。

## 构建

### 第 1 步：在 rollout 时捕获 `log π_old(a | s)`

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

快照在 rollout 时拍摄一次。在更新轮次期间不会改变。

### 第 2 步：计算 GAE 优势（第 07 课）

与 A2C 相同。在批次中归一化。

### 第 3 步：剪切代理更新

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

"剪切 → 梯度为零"模式是 PPO 的核心。如果新策略已经在有益方向上漂移太远，更新停止。

### 第 4 步：价值和熵

在 critic 目标上添加标准 MSE，在 actor 上添加熵奖励，与 A2C 相同。

### 第 5 步：诊断

每次更新时观察三件事：

- **平均 KL** `E[log π_old - log π_θ]`。应该保持在 `[0, 0.02]` 内。如果超过 `0.1`，减少 `K_EPOCHS` 或 `LR`。
- **剪切比例** —— 比率超出 `[1-ε, 1+ε]` 的样本比例。应该是 `~0.1-0.3`。如果是 `~0`，剪切从未触发 → 提高 `LR` 或 `K_EPOCHS`。如果是 `~0.5+`，你正在过度拟合 rollout → 降低它们。
- **解释方差** `1 - Var(V_target - V_pred) / Var(V_target)`。Critic 质量指标。应该随着 critic 学习而接近 1。

## 陷阱

- **剪切系数调参不当。** `ε = 0.2` 是事实上的标准。降到 `0.1` 使更新过于谨慎；`0.3+` 引发不稳定。
- **轮次太多。** `K > 20` 经常不稳定，因为策略远离 `π_old`。限制轮次，特别是对于大网络。
- **没有奖励归一化。** 大奖励尺度吞噬剪切范围。在计算优势之前归一化奖励（运行时标准差）。
- **忘记优势归一化。** 每批次零均值/单位标准差归一化是标准的。跳过它会破坏大多数基准上的 PPO。
- **学习率没有衰减。** PPO 从线性 LR 衰减到零中受益。恒定 LR 通常更差。
- **重要性比率数学错误。** 始终使用 `exp(log_new - log_old)` 以保证数值稳定性，而不是 `new / old`。
- **梯度符号错误。** 最大化代理 = **最小化** `-L^{CLIP}`。符号翻转是最常见的 PPO bug。

## 使用

PPO 在 2026 年是跨多个领域出乎意料地通用的默认 RL 算法：

| 用例 | PPO 变体 |
|----------|-------------|
| MuJoCo / 机器人控制 | PPO + 高斯策略，GAE(0.95) |
| Atari / 离散游戏 | PPO + 类别策略，滚动 128 步 rollout |
| LLM 的 RLHF | PPO + 到参考模型的 KL 惩罚，奖励来自响应末端的 RM |
| 大规模游戏智能体 | IMPALA + PPO（AlphaStar、OpenAI Five） |
| 推理 LLM | GRPO（第 12 课）—— 没有 critic 的 PPO 变体 |
| 仅偏好数据 | DPO —— PPO+KL 的闭式坍缩，无需在线采样 |

PPO **损失形状** —— 剪切代理 + 价值 + 熵 —— 是 DPO、GRPO 和几乎每个 RLHF 管道的脚手架。

## 交付

保存为 `outputs/skill-ppo-trainer.md`：

```markdown
---
name: ppo-trainer
description: 为给定环境生成 PPO 训练配置和诊断计划。
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

给定一个环境和训练预算，输出：

1. Rollout 大小。`N` 个环境 × `T` 步。
2. 更新计划。`K` 轮，小批次大小，LR schedule。
3. 代理参数。`ε`（剪切）、`c_v`、`c_e`，优势归一化开启。
4. 优势。GAE(`λ`)，带有明确的 `γ` 和 `λ`。
5. 诊断计划。KL、剪切比例、解释方差阈值及警报。

拒绝 `K > 30` 或 `ε > 0.3`（不安全的信任域）。拒绝任何没有优势归一化或 KL/剪切监控的 PPO 运行。将持续高于 0.4 的剪切比例标记为漂移。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上用 `ε=0.2, K=4` 运行 PPO。在匹配的环境步数下与 A2C（每个 rollout 一轮）比较样本效率。
2. **中等。** 遍历 `K ∈ {1, 4, 10, 30}`。绘制回报 vs 环境步数，并跟踪每次更新的平均 KL。在什么 `K` 下这个任务的 KL 会爆炸？
3. **困难。** 用自适应 KL 惩罚替换剪切代理（如果 `KL > 2·target` 则 `β` 加倍，如果 `KL < target/2` 则减半）。比较最终回报、稳定性和无剪切性。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 重要性比率 | "r_t(θ)" | `π_θ(a|s) / π_old(a|s)`；偏离收集数据的策略的程度。 |
| 剪切代理 | "PPO 的主要技巧" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`；在有益方向上越过剪切后梯度变平。 |
| 信任域 | "TRPO / PPO 的意图" | 限制每次更新的 KL 以保证单调改进。 |
| KL 惩罚 | "软信任域" | 替代 PPO：`L - β · KL(π_θ || π_old)`。自适应 `β`。 |
| 剪切比例 | "剪切多久触发一次" | 诊断——应该是 0.1-0.3；超出意味着调参不当。 |
| 多轮训练 | "数据复用" | 每个 rollout 的 K 轮；用方差代价换取样本效率。 |
| 近 on-policy | "大部分是 on-policy" | PPO 名义上是 on-policy，但 K>1 轮使用略微离策略的数据是安全的。 |
| PPO-KL | "另一个 PPO" | KL 惩罚变体；用于 RLHF，那里到参考的 KL 本身就是一个约束。 |

## 延伸阅读

- [Schulman 等人（2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — 原始论文。
- [Schulman 等人（2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO，PPO 的前身。
- [Andrychowicz 等人（2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — 每个 PPO 超参数都被消融了。
- [Ouyang 等人（2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT；PPO 在 RLHF 中的配方。
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — 带有 PyTorch 的现代清晰阐述。
- [CleanRL PPO 实现](https://github.com/vwxyzjn/cleanrl) — 被许多论文使用的参考单文件 PPO。
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — 语言模型上 PPO 的生产配方；与第 09 课（RLHF）一起阅读。
- [Engstrom 等人（2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — "37 个代码级优化"论文；哪些 PPO 技巧是实质性的，哪些是 folklore。