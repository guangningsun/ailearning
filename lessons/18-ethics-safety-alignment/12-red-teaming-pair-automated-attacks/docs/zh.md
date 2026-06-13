# 红队：PAIR 与自动化攻击

> Chao, Robey, Dobriban, Hassani, Pappas, Wong (NeurIPS 2023, arXiv:2310.08419)。PAIR——Prompt Automatic Iterative Refinement——是规范的自动化黑盒越狱。一个带有红队系统提示的攻击者 LLM 迭代地为目标 LLM 提出越狱建议，通过上下文反馈积累尝试和响应。PAIR 通常在 20 个查询内成功，比 GCG（Zou 等人的 token 级梯度搜索）高效几个数量级，且不需要白盒访问。PAIR 现在是 JailbreakBench（arXiv:2404.01318）和 HarmBench 的标准基线，与 GCG、AutoDAN、TAP 和 Persuasive Adversarial Prompt 并列。

**类型：** 构建
**语言：** Python (stdlib，对玩具目标进行模拟 PAIR 循环)
**前置条件：** 阶段 18 · 01 (instruction-following)，阶段 14 (agent engineering)
**时间：** 约 75 分钟

## 学习目标

- 描述 PAIR 算法：攻击者系统提示、迭代细化、上下文反馈。
- 解释为什么 PAIR 在目标是黑盒时严格比 GCG 更高效。
- 命名四个其他自动化攻击基线（GCG、AutoDAN、TAP、PAP）并陈述每个的一个显著特征。
- 描述 JailbreakBench 和 HarmBench 评估协议，以及在每个协议下"攻击成功率"的含义。

## 问题

红队曾经是手动活动。少数专家测试人员构建对抗性提示并跟踪哪些有效。这不能扩展：攻击成功率需要统计样本，而且目标是每个模型发布都会变化的红队。PAIR 将红队操作化为一个优化问题，以黑盒目标作为测试对象。

## 概念

### PAIR 算法

输入：
- 目标 LLM T（我们正在攻击的模型）。
- 判断 LLM J（评分响应是否是越狱）。
- 攻击者 LLM A（红队优化器）。
- 目标字符串 G："响应 [有害指令]"。
- 预算 K（通常 20 个查询）。

循环，对于 k = 1..K：
1. A 被提示目标 G 和迄今为止的（提示，响应）对的歷史。
2. A 发出一个新提示 p_k。
3. 将 p_k 提交给 T；接收响应 r_k。
4. J 根据目标对（p_k, r_k）评分。
5. 如果分数 >= 阈值，停止——找到越狱。
6. 否则，将（p_k, r_k）追加到 A 的历史；继续。

实证结果（NeurIPS 2023）：对 GPT-3.5-turbo 和 Llama-2-7B-chat 的攻击成功率超过 50%，成功查询的平均次数在 10-20 范围内。

### PAIR 高效的原因

GCG（Zou 等人 2023）通过梯度搜索对抗性 token 后缀；它需要白盒模型访问并产生难以理解的字符串。PAIR 是黑盒的，产生可在模型间转移的自然语言攻击。PAIR 的上下文反馈让攻击者从每次拒绝中学习；GCG 没有等效功能。

### 相关自动化攻击

- **GCG（Zou 等人 2023，arXiv:2307.15043）。** Token 级梯度搜索用于对抗性后缀。白盒、可转移、产生难以理解的字符串。
- **AutoDAN（Liu 等人 2023）。** 提示的进化搜索，由分层目标引导。
- **TAP（Mehrotra 等人 2024）。** 带剪枝的攻击树——分支多个 PAIR 风格的 rollout。
- **PAP（Zeng 等人 2024）。** 有说服力的对抗性提示——将人类说服技术编码为提示模板。

### JailbreakBench 和 HarmBench

两者（2024）都标准化了评估：

- JailbreakBench（arXiv:2404.01318）。100 个跨 10 个 OpenAI 策略类别的有害行为。作为主要指标的攻击成功率（ASR）。需要判断者（GPT-4-turbo、Llama Guard 或 StrongREJECT）。
- HarmBench（Mazeika 等人 2024）。跨 7 个类别的 510 个行为，具有语义和功能危害测试。将 18 种攻击与 33 个模型进行比较。

ASR 通常在固定查询预算下报告。比较攻击需要匹配预算；90% ASR @ 200 查询与 85% ASR @ 20 不可比较。

### 为什么对 2026 年部署重要

每个前沿实验室现在都在发布前对生产模型运行 PAIR 和 TAP。ASR 轨迹出现在模型卡片（第 26 课）和安全案例附录中（第 18 课）。攻击并非异类——它是标准基础设施。

### 这在第 18 课中的位置

第 12 课是自动化攻击基础。第 13 课（Many-Shot Jailbreaking）是互补的长度利用。第 14 课（ASCII Art / Visual）是编码攻击。第 15 课（Indirect Prompt Injection）是 2026 年生产攻击面。第 16 课涵盖防御工具对应物（Llama Guard、Garak、PyRIT）。

## 使用它

`code/main.py` 构建一个简单的 PAIR 循环。目标是拒绝"明显"有害提示的模拟分类器（关键词过滤）。攻击者是一个基于规则的精炼器，尝试释义、角色扮演框架和编码。判断者对响应评分。你看着攻击者在关键词过滤器下成功，并在语义过滤器下失败。

## 发布它

本课产出 `outputs/skill-attack-audit.md`。给定红队评估报告，它审计：运行了哪些攻击（PAIR、GCG、TAP、AutoDAN、PAP），每个的预算是多少，使用了哪个判断者，在哪个有害行为集上（JailbreakBench、HarmBench、内部）。

## 练习

1. 运行 `code/main.py`。测量三个内置攻击者策略的平均查询成功率。解释每个策略利用的目标防御假设。

2. 实现第四个攻击者策略（例如，翻译成另一种语言、base64 编码）。报告对关键词过滤目标和语义过滤目标的新平均查询成功率。

3. 阅读 Chao 等人 2023 年图 5（PAIR vs GCG 比较）。描述两个即使考虑 PAIR 效率优势仍偏好 GCG 的场景。

4. JailbreakBench 报告固定目标集上的 ASR。设计一个额外的指标来衡量攻击多样性（成功提示的多样性）。解释为什么多样性对防御评估重要。

5. TAP（Mehrotra 2024）用分支 + 剪枝扩展了 PAIR。在 `code/main.py` 上绘制 TAP 风格扩展的草图，并描述计算成本与成功率之间的权衡。

## 关键术语

| 术语 | 人们怎么说 | 实际意味着什么 |
|------|-----------------|------------------------|
| PAIR | "自动化越狱" | 提示自动迭代精炼；攻击者-LLM + 判断者-LLM 循环 |
| GCG | "梯度越狱" | 白盒 token 级梯度搜索，用于对抗性后缀 |
| 攻击成功率 (ASR) | "k 查询时的越狱百分比" | 主要指标；必须与查询预算和判断者身份一起报告 |
| 判断 LLM | "评分者" | 对响应是否满足有害目标进行分级的 LLM |
| JailbreakBench | "评估" | 带标记类别的标准化有害行为集 |
| HarmBench | "更广泛的基准" | 510 个行为，功能 + 语义危害测试 |
| TAP | "攻击树" | 带分支 + 剪枝的 PAIR；在更高计算下获得更好的 ASR |

## 进一步阅读

- [Chao 等人——用二十个查询越狱黑盒 LLM（arXiv:2310.08419）](https://arxiv.org/abs/2310.08419) — PAIR 论文，NeurIPS 2023
- [Zou 等人——对齐 LLM 的普遍可转移对抗攻击（arXiv:2307.15043）](https://arxiv.org/abs/2307.15043) — GCG 论文
- [Chao 等人——JailbreakBench（arXiv:2404.01318）](https://arxiv.org/abs/2404.01318) — 标准化评估
- [Mazeika 等人——HarmBench（ICML 2024）](https://arxiv.org/abs/2402.04249) — 更广泛的评估