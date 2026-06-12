# 游戏中的强化学习 —— AlphaZero、MuZero 与 LLM 推理时代

> 1992：TD-Gammon 用纯 TD 在十五子棋上击败了人类冠军。2016：AlphaGo 击败了李世石。2017：AlphaZero 从零开始横扫象棋、将棋和围棋。2024：DeepSeek-R1 证明同样的配方，用 GRPO 替代 PPO，在推理上同样有效。游戏是这一阶段每一次突破的基准。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 9 · 05（DQN）、阶段 9 · 08（PPO）、阶段 9 · 09（RLHF）、阶段 9 · 10（MARL）
**时间：** 约 120 分钟

## 问题

游戏拥有 RL 需要的一切。清晰的奖励（胜/负）。无限的回合（自我对弈重置）。完美的仿真（游戏本身就是仿真器）。离散或小型连续动作空间。强制对抗鲁棒性的多智能体结构。

而且游戏是每一次重大 RL 突破的测试方式。TD-Gammon（十五子棋，1992）。Atari-DQN（2013）。AlphaGo（2016）。AlphaZero（2017）。OpenAI Five（Dota 2，2019）。AlphaStar（星际争霸 II，2019）。MuZero（学习模型，2019）。AlphaTensor（矩阵乘法，2022）。AlphaDev（排序算法，2023）。DeepSeek-R1（数学推理，2025）—— 最新证明游戏-RL 技术在文本上同样有效。

本总结课通过一个统一的视角审视三个里程碑架构 —— AlphaZero、MuZero 和 GRPO：**自我对弈 + 搜索 + 策略改进**。每一个都是对前一个的推广；特别是 GRPO 是 AlphaZero 的配方应用于 LLM 推理，token 作为动作，数学验证作为胜利信号。

## 概念

![AlphaZero ↔ MuZero ↔ GRPO：相同的循环，不同的环境](../assets/rl-games.svg)

**统一的循环。**

```
while True:
    trajectory = self_play(current_policy, search)     # 与自我对弈
    policy_target = search.improved_policy(trajectory) # 搜索改进原始策略
    policy_net.update(policy_target, value_target)     # 在搜索输出上监督训练
```

**AlphaZero（2017）。** Silver 等人。给定一个游戏（象棋、将棋、围棋），规则已知：

- 策略-价值网络：一个塔 `f_θ(s) → (p, v)`。`p` 是合法动作上的先验。`v` 是预期游戏结果。
- 蒙特卡洛树搜索（MCTS）：每一步，展开一棵可能延续的树。用 `(p, v)` 作为先验 + 引导。用 UCB（PUCT）选择节点：`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- 自我对弈：智能体对智能体下棋。在步 `t`，MCTS 访问分布 `π_t` 成为策略训练目标。
- 损失：`L = (v - z)² - π · log p + c · ||θ||²`。`z` 是游戏结果（+1 / 0 / -1）。

零人类知识。零手工启发式。同一套配方，在各自几千万次自我对弈后分别掌握了象棋、将棋和围棋。

**MuZero（2019）。** Schrittwieser 等人。移除了规则已知的假设。

- 不使用固定环境，而是学习一个*潜在动力学模型* `(h, g, f)`：
  - `h(s)`：将观测编码为潜在状态。
  - `g(s_latent, a)`：预测下一个潜在状态 + 奖励。
  - `f(s_latent)`：预测策略先验 + 价值。
- MCTS 在*学习到的潜在空间*中运行。相同的搜索，相同的训练循环。
- 在围棋、象棋、将棋**和** Atari 上都有效 —— 同一算法，无需规则知识。

**随机 MuZero（2022）。** 添加随机动力学和机会节点；扩展到十五子棋类游戏。

**Muesli、Gumbel MuZero（2022-2024）。** 在样本效率和确定性搜索上的改进。

**GRPO（2024-2025）。** DeepSeek-R1 配方。相同的 AlphaZero 形循环，应用于语言模型推理：

- "游戏"：回答一个数学/编程/推理问题。"胜利" = 验证器（测试用例通过、数字答案匹配）返回 1。
- 策略：LLM。动作：token。状态：提示 + 到目前为止的回复。
- 没有评论员（PPO 风格的 V_φ）。相反，对每个提示，从策略采样 `G` 个完成。计算每个的奖励。用**组相对优势** `A_i = (r_i - mean_r) / std_r` 作为 REINFORCE 风格更新的信号。
- KL 惩罚到参考策略以防止漂移（如 RLHF）。
- 完整损失：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

没有奖励模型，没有评论员，没有 MCTS。组相对基线取代了这三者。在推理基准测试上以一小部分计算量匹配或超过 PPO-RLHF 的质量。

**R1 配方的完整版。** DeepSeek-R1（DeepSeek 2025）是一篇论文中的两个模型：

- **R1-Zero。** 从 DeepSeek-V3 基模型开始。不做 SFT。直接应用 GRPO，使用两个奖励成分：*准确性奖励*（基于规则的 —— 最终答案是否解析为正确的数字 / 代码是否通过单元测试）和*格式奖励*（完成是否将其思维链包裹在 `<think>…` 标签中）。经过数千步，平均回复长度从约 100 增长到约 10,000 个 token，数学基准分数攀升至接近 o1-preview 的水平。模型从零开始学会推理。缺点：其思维链通常难以阅读、混用语言、缺乏文体风格。
- **R1。** 用四阶段管道解决 R1-Zero 的可读性问题：
  1. **冷启动 SFT。** 收集几千个格式清晰的 long-CoT 演示。在它们上监督微调基模型。这给了一个可读的起点。
  2. **推理导向的 GRPO。** 应用 GRPO，使用准确性+格式奖励，外加一个*语言一致性*奖励以防止代码切换。
  3. **拒绝采样 + SFT 第二轮。** 从 RL 检查点采样约 600K 个推理轨迹，只保留那些最终答案正确且 CoT 可读的，并与约 200K 个非推理 SFT 示例（写作、QA、自我认知）结合。再次微调基模型。
  4. **全光谱 GRPO。** 又一轮 RL，覆盖推理（基于规则的奖励）和通用对齐（有用性/无害性基于偏好的奖励）。

结果以开放权重匹配 o1 在 AIME 和 MATH-500 上的表现，且足够小可以提炼。同篇论文还通过在 R1 的推理轨迹上 SFT'蒸馏出六个密集模型（Qwen-1.5B 到 Llama-70B）—— 学生没有 RL。从强 RL 教师蒸馏始终在学生规模上击败从零 RL。

**为什么推理用 GRPO 而不是 PPO。** DeepSeekMath 论文（2024 年 2 月）给出三个原因：(1) 没有价值网络要训练，内存减半；(2) 组基线自然处理推理任务产生的稀疏的回合末奖励；(3) 每提示归一化使优势在不同难度的问题上可比，而 PPO 的单个评论员做不到。

**无搜索 vs 基于搜索。** 游戏已经分叉：

- *完美信息游戏，长视野*（围棋、象棋）：仍然是基于搜索的。AlphaZero / MuZero 主导。
- *LLM 推理*：生产中还没有 MCTS；GRPO 在完整回滚上，最佳-of-N 用于推理计算。过程奖励模型（PRM）暗示步级搜索正在被加回。

## 构建

`code/main.py` 中的代码实现了**迷你版 GRPO** —— 一个带多组样本的多臂老虎机。算法与在 LLM 上相同；只是策略和环境更简单。它教的是*损失*和*组相对优势*，这是 2024 年的创新。

### 第 1 步：一个微型验证器环境

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

在真实 GRPO 中验证器运行单元测试或检查数学等式。

### 第 2 步：策略：每个提示的 K 个回答 token 上的 softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

等价于 LLM 在给定提示条件下的最终层输出。

### 第 3 步：组采样和组相对优势

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL 惩罚：将 theta 拉向参考
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

组相对优势是 2024 年 DeepSeek 的技巧。不需要评论员。"基线"是组均值，归一化用组标准差。

### 第 4 步：与 REINFORCE 基线对比（无价值）

相同设置，相同计算， plain REINFORCE。GRPO 收敛更快、更稳定。

### 第 5 步：观察熵和 KL

与 RLHF 相同的诊断：到参考的平均 KL、策略熵、随时间的奖励。一旦这些稳定，训练就完成了。

## 陷阱

- **通过验证器游戏进行奖励黑客。** GRPO 继承了 RLHF 的风险：如果验证器是错误的或可利用的，LLM 会找到利用。健壮的验证器（多个测试用例、形式证明）很重要。
- **组太小。** 组基线的方差类似 `1/√G`。低于 `G = 4`，优势信号有噪声；标准选择是 `G = 8` 到 `64`。
- **长度偏差。** 不同长度的 LLM 完成有不同的对数概率。按 token 计数归一化，或使用序列级对数概率，或截断到最大长度。
- **纯自我对弈循环。** AlphaZero 风格的训练可能在一般和游戏中陷入主导循环。通过多样化的对手池（联盟对弈，第 10 课）来缓解。
- **搜索-策略不匹配。** AlphaZero 训练策略模仿搜索输出。如果策略网络太小以至于无法表示搜索的分布，训练就会停滞。
- **计算下限。** MuZero / AlphaZero 需要大量计算。单个消融实验通常是数百 GPU 小时。迷你演示存在（如 Connect Four 上的 AlphaZero）用于学习。
- **验证器覆盖。** 对有 bug 的解决方案通过的单元测试会强化 bug。设计能捕获边缘情况的验证器。

## 使用

2026 年游戏-RL 领域格局，按领域：

| 领域 | 主导方法 |
|--------|-----------------|
| 两人零和棋盘游戏（围棋、象棋、将棋） | AlphaZero / MuZero / KataGo |
| 不完美信息扑克 | CFR + 深度学习（DeepStack、Libratus、Pluribus） |
| Atari / 像素游戏 | Muesli / MuZero / IMPALA-PPO |
| 大型多人策略（Dota、星际争霸） | PPO + 自我对弈 + 联盟（OpenAI Five、AlphaStar） |
| LLM 数学/代码推理 | GRPO（DeepSeek-R1、Qwen-RL、开源复现） |
| LLM 对齐 | DPO / RLHF-PPO（不是 GRPO；验证器是偏好而非可验证） |
| 机器人学 | PPO + DR（不是游戏-RL，但使用相同的策略梯度工具） |
| 组合问题 | AlphaZero 变体（AlphaTensor、AlphaDev） |

**配方** —— 自我对弈、搜索增强改进、策略提炼 —— 跨越文本、像素和物理控制。GRPO 是最年轻的实例；更多正在到来。

## 交付

保存为 `outputs/skill-game-rl-designer.md`：

```markdown
---
name: game-rl-designer
description: 为给定领域设计游戏-RL 或推理-RL 训练管道（AlphaZero / MuZero / GRPO）。
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

给定目标（完美信息游戏 / 不完美信息 / Atari / LLM 推理 / 组合），输出：

1. 环境契合度。规则已知？马尔可夫？随机？多智能体？决定 AlphaZero vs MuZero vs GRPO。
2. 搜索策略。MCTS（带学习先验的 PUCT）、Gumbel 采样、最佳-of-N，或无。
3. 自我对弈计划。对称自我对弈 / 联盟 / 离线数据 / 验证器生成。
4. 目标信号。游戏结果 / 验证器奖励 / 偏好 / 学习模型。包括鲁棒性计划。
5. 诊断指标。对基线的胜率、ELO 曲线、验证器通过率、到参考的 KL。

不完美信息游戏拒绝 AlphaZero（路线到 CFR）。没有可信验证器拒绝 GRPO。没有固定基线对手集拒绝任何游戏-RL 管道（否则自我对弈 ELO 未经校准）。
```

## 练习

1. **简单。** 在 `code/main.py` 中实现 GRPO 多臂老虎机。在 2 个提示 × 每个 4 个回答 token 上训练。用 `G=8` 在 < 1,000 次更新内收敛。
2. **中等。** 插入 PPO（截断）和 vanilla REINFORCE。在相同的多臂老虎机上比较样本效率和奖励方差与 GRPO。
3. **困难。** 扩展到长度-2 的"推理链"：智能体发出两个 token，验证器奖励这个配对。测量 GRPO 如何处理两步序列的信用分配。（提示：按*完整序列*计算组优势，传播到两个 token 位置。）

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| MCTS | "带学习网络的树搜索" | 蒙特卡洛树搜索；带学习 `(p, v)` 先验的 UCB1/PUCT 选择。 |
| AlphaZero | "自我对弈 + MCTS" | 策略-价值网络训练以匹配 MCTS 访问和游戏结果。 |
| MuZero | "学习模型的 AlphaZero" | 相同循环，但在潜在空间通过学习到的动力学。 |
| GRPO | "无评论员的 PPO" | 组相对策略优化；带组均值基线和 KL 的 REINFORCE。 |
| PUCT | "AlphaZero 的 UCB" | `Q + c · p · √N / (1 + N_a)` —— 用先验平衡价值估计。 |
| 自我对弈 | "智能体 vs 过去的自己" | 零和的标准；对称训练信号。 |
| 联盟对弈 | "基于人群的自我对弈" | 过去 + 现在 + 攻击者作为对手采样。 |
| 验证器奖励 | "可验证的 RL" | 奖励来自确定性检查器（测试通过、答案匹配）。 |
| 过程奖励 | "PRM" | 对每个推理步评分，而不只对最终答案。 |

## 扩展阅读

- [Silver 等人 (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver 等人 (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser 等人 (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals 等人 (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — 引入 GRPO 和组相对基线的论文。
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — 完整的四阶段 R1 配方加上 R1-Zero 消融。
- [Brown 等人 (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — 大规模 CFR + 深度学习。
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — 开创一切的论文。
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — 使用自定义奖励函数应用 GRPO的生产参考。
- [Qwen 团队 (2024). Qwen2.5-Math — GRPO 复现](https://github.com/QwenLM/Qwen2.5-Math) — 在多个规模上开源复现 R1 配方。
- [Sutton & Barto (2018). 第 17 章 — 强化学习前沿](http://incompleteideas.net/book/RLbook2020.pdf) — 自我对弈、搜索和"设计奖励"的教科书框架，R1 在 LLM 规模上实现。