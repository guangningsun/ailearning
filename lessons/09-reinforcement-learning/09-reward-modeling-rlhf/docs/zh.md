# 奖励建模与 RLHF

> 人类无法写出一个"好的助手回复"的奖励函数，但他们可以比较两个回复并选出更好的那个。用这些比较来拟合一个奖励模型，然后用 RL 针对它优化语言模型。Christiano 2017。InstructGPT 2022。这个配方将 GPT-3 变成了 ChatGPT。到了 2026 年，它在很大程度上正被 DPO 取代——但心智模型依然成立。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 5 · 05（情感分析）、阶段 9 · 08（PPO）
**时间：** 约 45 分钟

## 问题

你用下一个 token 预测目标训练了一个语言模型。它能写出语法正确的英文。但它也会撒谎、漫谈、该拒绝时却不拒绝。更多的预训练解决不了这个问题——网络文本才是问题所在，而不是解药。

你想要一个**标量奖励**，它能告诉你"对于指令 X，回复 A 比回复 B 更好"。手工编写这个奖励函数是不可能的。"有用性"不是 token 之上的闭合形式表达式。但人类可以比较两个输出并标记偏好。这个收集起来成本低，可以规模化。

RLHF（Christiano 等，2017；Ouyang 等，2022）将偏好转化为一个奖励模型，然后通过 PPO 针对该奖励优化 LM。分三步：SFT → RM → PPO。这就是 ChatGPT、Claude、Gemini 以及 2023–2025 年每一个对齐 LLM 所使用的配方。

到了 2026 年，PPO 步骤在很大程度上被 DPO（阶段 10 · 08）取代了，因为它更便宜，在对齐调优上几乎一样好。但**奖励模型**这一块依然支撑着每一个 Best-of-N 采样器、每一个基于可验证奖励的 RL 流程，以及每一个使用过程奖励模型的推理模型。理解了 RLHF，你就理解了整个对齐技术栈。

## 概念

![三阶段 RLHF：SFT、基于成对偏好的 RM 训练、带有 KL 惩罚的 PPO](../assets/rlhf.svg)

**第一阶段：监督微调（SFT）。** 从预训练基模型开始。在人类-written 的目标行为演示（遵循指令的回复、有用的回复等）上微调。结果：一个模型 `π_SFT`，它**偏向良好行为**，但仍有无界的动作空间。

**第二阶段：奖励模型训练。**

- 收集对 prompt `x` 的成对回复 `(y_+, y_-)`，由人类标记为"y_+ 优于 y_-"
- 训练一个奖励模型 `R_φ(x, y)` 来给 `y_+` 分配更高的分数
- 损失函数：**Bradley-Terry 成对逻辑斯谛**：

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ 是 sigmoid。奖励差异意味着偏好的对数几率。BT 自 1952 年起就是标准（Bradley-Terry），也是现代 RLHF 的主流选择。

- `R_φ` 通常从 SFT 模型初始化，上面加一个标量头。同样的 transformer 主干；单个线性层输出奖励。

**第三阶段：针对 RM 运行 PPO，并带 KL 惩罚。**

- 将可训练策略 `π_θ` 从 `π_SFT` 初始化。保留一个冻结的**参考** `π_ref = π_SFT`
- 在回复 `y` 末尾的奖励：

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL 惩罚防止 `π_θ` 任意偏离 `π_SFT`——它是**正则化器**，不是硬信任域。`β` 通常在 `0.01`-`0.05` 之间。
- 使用这个奖励运行 PPO（第 08 课）。优势在 token 级轨迹上计算，但 RM 只对完整回复打分。

**为什么需要 KL？** 没有它，PPO 会欣然找到奖励黑客策略——RM 只在分布内补全上训练过。分布外的回复可能比任何人写的回复得分都高。KL 让 `π_θ` 保持在 RM 训练时所在的流形上。这是 RLHF 中最重要的旋钮。

**2026 年的状态：**

- **DPO**（Rafailov 2023）：闭合形式代数将第二和第三阶段合并为一个在偏好数据上的单一监督损失。无 RM，无 PPO。在对齐基准上质量相当，计算量却只是一小部分。见阶段 10 · 08。
- **GRPO**（DeepSeek 2024–2025）：用组相对基线代替 critic 的 PPO，奖励来自**验证器**（代码运行 / 数学答案匹配）而不是人工训练的 RM。是推理模型的主导方案。见阶段 9 · 12。
- **过程奖励模型（PRM）：** 对部分解打分（每个推理步骤），在 RLHF 和 GRPO 变体中用于推理。
- **宪法人工智能 / RLAIF：** 使用一个对齐的 LLM 来生成偏好，而不是人类。可以扩展偏好预算。

## 构建它

本课使用极简的合成"prompt"和"回复"，用字符串表示。RM 是一个基于词袋表示的线性评分器。没有真正的 LLM——管道的**形状**才是重要的，不是规模。见 `code/main.py`。

### 第一步：合成偏好数据

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

在真正的 RLHF 中，这一步由人工标注员完成。形状——`(prompt, preferred_response, rejected_response)`——是一样的。

### 第二步：Bradley-Terry 奖励模型

线性评分：`R(x, y) = w · bag(y)`。训练它来最小化 BT 成对对数损失：

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

经过几百次更新后，`w` 给好的 token 分配正权重，给坏的 token 分配负权重。

### 第三步：基于 RM 的类 PPO 策略

我们的玩具策略从词汇表中生成一个 token。我们在 RM 下对该 token 打分，计算 `log π_θ(token | prompt)`，添加一个 KL-to-reference 惩罚，并应用裁剪的 PPO surrogate。

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### 第四步：监控 KL

每轮更新追踪平均 `KL(π_θ || π_ref)`。如果它越过 `~5-10`，策略就已经远离 `π_SFT` 了——要么 `β` 需要提高，要么奖励黑客正在开始。这是真实 RLHF 中最重要的诊断指标。

### 第五步：使用 TRL 的生产配方

一旦你理解了这个玩具管道，这就是一个真实库用户写的同样循环。Hugging Face 的 [TRL](https://huggingface.co/docs/trl) 是参考实现——`RewardTrainer` 用于第二阶段，`PPOTrainer`（内置 KL-to-reference）用于第三阶段。

```python
# 第二阶段：从成对偏好中训练奖励模型
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# 数据集行：{"prompt", "chosen", "rejected"} —— Bradley-Terry 格式
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# 第三阶段：针对 RM 运行 PPO，并带 KL 惩罚到 SFT 参考
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats 包括：mean_kl、clip_frac、value_loss —— 三个 PPO 诊断指标
```

库为你做了三件事。`adap_kl_ctrl=True` 实现了自适应 β 调度：如果观察到的 KL 超过 `target_kl`，β 加倍；如果低于一半，β 减半。参考模型按惯例冻结——你不能意外地与 `policy` 共享参数。值头与策略位于同一主干上（`AutoModelForCausalLMWithValueHead` 附加了一个标量 MLP 头），这就是为什么 TRL 分别报告 `policy/kl` 和 `value/loss`。

## 陷阱

- **过度优化 / 奖励黑客。** RM 是不完善的；`π_θ` 找到得分高但实际很差的对抗性补全。症状：奖励无限上升，而人工评估分数持平或下降。修复：早停、提高 `β`、扩大 RM 训练数据。
- **长度黑客。** 在有用回复上训练的 RM 通常隐含地奖励长度。策略学会给回复添padding。补救：长度归一化奖励，或使用长度感知 RM 的 RLAIF。
- **RM 太小。** RM 需要至少与策略一样大。太小的 RM 无法忠实地对策略的输出打分。
- **KL 调参。** β 太低 → 漂移和奖励黑客。β 太高 → 策略几乎不变。标准技巧是针对每步固定 KL 的**自适应** β。
- **偏好数据噪声。** 约 30% 的人工标签是有噪声或模糊的。通过在一致性过滤数据上训练 RM 来校准，或在 BT 上使用温度。
- **离策略问题。** PPO 数据在第一个 epoch 之后稍微离策略。按第 08 课监控裁剪比例。

## 使用它

2026 年的 RLHF 是分层的：

| 层级 | 目标 | 方法 |
|-------|--------|--------|
| 指令遵循、有用性、无害性 | 对齐 | DPO（阶段 10 · 08）优于 RLHF-PPO。 |
| 推理正确性（数学、代码） | 能力 | 带验证器奖励的 GRPO（阶段 9 · 12）。 |
| 长 horizon 多步任务 | 代理性 | 使用过程奖励模型 over 步骤的 PPO / GRPO。 |
| 安全性 / 拒绝行为 | 安全性 | 带独立安全 RM 的 RLHF-PPO，或宪法人工智能。 |
| 推理时的 Best-of-N | 快速对齐 | 在解码时使用 RM；无需策略训练。 |
| 奖励蒸馏 | 推理计算 | 在冻结 LM 上训练一个小的"奖励头"。 |

RLHF 是 2022–2024 年的**标杆**方法。到了 2026 年，生产对齐管道以 DPO 为主，PPO 仅用于 RM 密集型或安全关键步骤。

## 交付它

保存为 `outputs/skill-rlhf-architect.md`：

```markdown
---
name: rlhf-architect
description: 为语言模型设计一个 RLHF / DPO / GRPO 对齐管道，包括 RM、KL 和数据策略。
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

给定一个基 LM、目标行为（对齐 / 推理 / 拒绝 / 代理）和偏好或验证器预算，输出：

1. 阶段。SFT？RM？DPO？GRPO？附理由。
2. 偏好或验证器来源。人类、AI 反馈、基于规则、单元测试通过，或奖励蒸馏。
3. KL 策略。固定 β、自适应 β，或 DPO（隐式 KL）。
4. 诊断指标。平均 KL、奖励稳定性、过度优化防护（留出人工评估）。
5. 安全门。红队集、拒绝率、安全 RM 与有用性 RM 分开。

没有 KL 监控就不交付 RLHF-PPO。不使用比目标策略小的 RM。不使用纯长度奖励。标记任何没有留出盲法人工评估集来防护过度优化的管道。
```

## 练习

1. **简单。** 在 500 个合成偏好对上训练 `code/main.py` 中的 Bradley-Terry 奖励模型。在留出的 100 对上测量成对准确率。应该超过 90%。
2. **中等。** 用 `β ∈ {0.0, 0.1, 1.0}` 运行玩具 PPO-RLHF 循环。对于每个值，绘制 RM 分数 vs 相对于参考的 KL 随更新的变化。哪个会出现奖励黑客？
3. **困难。** 在相同的偏好数据上实现 DPO（闭合形式偏好似然损失），并与 RLHF-PPO 管道在使用的计算量和最终获得的 RM 分数上进行比较。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| RLHF | "对齐 RL" | 三阶段 SFT + RM + PPO 管道（Christiano 2017，Ouyang 2022）。 |
| 奖励模型（RM） | "评分网" | 通过 Bradley-Terry 拟合成对偏好的学习标量函数。 |
| Bradley-Terry | "成对逻辑斯谛损失" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`；标准 RM 目标。 |
| KL 惩罚 | "保持在参考附近" | 奖励中的 `β · KL(π_θ || π_ref)`；反奖励黑客正则化器。 |
| 奖励黑客 | "Goodhart 定律" | 策略利用 RM 缺陷；症状：奖励上升，人工评估持平。 |
| RLAIF | "AI 标记的偏好" | 标签来自另一个 LM 而不是人类的 RLHF。 |
| PRM | "过程奖励模型" | 对部分推理步骤打分；用于推理管道。 |
| 宪法人工智能 | "Anthropic 的方法" | 由明确规则引导的 AI 生成偏好。 |

## 延伸阅读

- [Christiano 等（2017）。从人类偏好中进行深度强化学习](https://arxiv.org/abs/1706.03741) —— 开启 RLHF 的论文。
- [Ouyang 等（2022）。训练语言模型通过人类反馈遵循指令](https://arxiv.org/abs/2203.02155) —— ChatGPT 背后的配方。
- [Stiennon 等（2020）。通过人类反馈学习摘要](https://arxiv.org/abs/2009.01325) —— 更早的 RLHF 用于摘要。
- [Rafailov 等（2023）。直接偏好优化](https://arxiv.org/abs/2305.18290) —— DPO；2026 年 RLHF 后的默认选择。
- [Bai 等（2022）。宪法人工智能：来自 AI 反馈的无害性](https://arxiv.org/abs/2212.08073) —— RLAIF 和自我批判循环。
- [Anthropic RLHF 论文（Bai 等 2022）。训练一个有帮助且无害的助手](https://arxiv.org/abs/2204.05862) —— HH 论文。
- [Hugging Face TRL 库](https://huggingface.co/docs/trl) —— 生产的 `RewardTrainer` 和 `PPOTrainer`。阅读训练器源码以了解自适应 KL 和值头的细节。
- [Hugging Face —— 通过人类反馈图解强化学习](https://huggingface.co/blog/rlhf)，作者：Lambert、Castricato、von Werra、Havrilla —— 带图的三阶段管道权威教程。
- [von Werra 等（2020）。TRL：Transformer 强化学习](https://github.com/huggingface/trl) —— 库；`examples/` 有针对 Llama、Mistral 和 Qwen 的端到端 RLHF 脚本。
- [Sutton & Barto（2018）。第 17.4 节——设计奖励信号](http://incompleteideas.net/book/RLbook2020.pdf) —— 奖励假设视角；思考奖励黑客的必备前提。