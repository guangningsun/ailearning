# LLM 特性的 A/B 测试 — GrowthBook、Statsig 与 Vibes 问题

> 传统 A/B 测试并非为非确定性 LLM 构建。关键区别：评测回答"模型能完成这项工作吗？" A/B 测试回答"用户在乎吗？" 两者都需要；靠感觉检查上线已经过时了。2026 年可测试的内容：提示词工程（措辞）、模型选择（GPT-4 vs GPT-3.5 vs 开源；准确率 vs 成本 vs 延迟）、生成参数（temperature、top-p）。真实案例：一个聊天机器人奖励模型变体带来 +70% 对话长度和 +30% 留存；Nextdoor AI 主题行实验在奖励函数优化后带来 +1% CTR；Khan Academy Khanmigo 在延迟-数学准确率轴上迭代。平台格局：**Statsig**（2025 年 9 月被 OpenAI 以 $1.1B 收购）—— 序贯测试、CUPED、一体化。**GrowthBook** —— 开源、仓库原生、贝叶斯+频率学派+序贯引擎、CUPED、SRM 检查、Benjamini-Hochberg + Bonferroni 校正。根据仓库-SQL 偏好以及"被 OpenAI 收购"对你的组织是否有影响来选择。

**类型：** 学习型
**语言：** Python（标准库、自制序贯测试模拟器）
**前置条件：** 阶段 17 · 13（可观测性）、阶段 17 · 20（渐进式部署）
**时间：** 约 60 分钟

## 学习目标

- 区分评测（"模型能完成工作吗"）和 A/B 测试（"用户在乎吗"）。
- 列举三个可测试维度（提示词、模型、参数）并为每个选择指标。
- 解释 CUPED、序贯测试和 Benjamini-Hochberg 多重比较校正。
- 根据仓库-SQL 姿态和公司收购立场选择 Statsig 或 GrowthBook。

## 问题

你手工调优了一个系统提示词。感觉好多了。你上线了它。转化率在噪声中变化。你怪指标。或者你上线了一个新模型，转化率没动——模型退化了吗还是变化太小检测不出来？你不知道，因为你上线时没有做 A/B。

评测回答模型能否在带标签的集合上完成任务。它不回答用户是否更喜欢输出。只有受控的在线实验才能回答这个问题，而且只有在实验有足够功效、控制了非确定性、并校正了多重比较时才行。

## 概念

### 评测 vs A/B 测试

**评测** —— 离线、带标签集合、评判（评分标准或 LLM-as-judge 或人工）。回答："在这个固定分布上，输出是否正确/有用/安全？"

**A/B 测试** —— 在线、真实用户、随机分组。回答："新变体是否移动了重要的用户级指标？"

两者都需要。评测在暴露前捕获回退；A/B 在之后确认产品影响。

### 可测试的内容

1. **提示词工程** —— 措辞、系统提示词结构、示例。指标：任务成功率、用户留存、每请求成本。
2. **模型选择** —— GPT-4 vs GPT-3.5-Turbo vs Llama-OSS。指标：准确率（任务）+ 每请求成本 + P99 延迟。多目标。
3. **生成参数** —— temperature、top-p、max_tokens。指标：任务相关（输出多样性 vs 确定性）。

### CUPED — 方差缩减

使用实验前数据的受控实验。比较实验后期之前先回归掉前周期方差。典型方差缩减：30-70%。有效样本量免费增加。

实现：Statsig 和 GrowthBook 都有实现。

### 序贯测试

经典 A/B 假设固定样本量。序贯测试（"窥视并决策"）控制重复查看下的假阳性率。始终有效的序贯程序（mSPRT、Howard 置信序列）允许在明显赢家出现时提前停止。

### 多重比较校正

运行 20 个 A/B 测试，置信度 95%，会随机产生一个假阳性。Bonferroni 校正收紧每个测试的 α；Benjamini-Hochberg 控制假发现率。GrowthBook 两者都有实现。

### SRM — 样本比例不匹配

分配哈希将用户随机分组到变体。如果 50/50 分组交付了 47/53，说明有问题——SRM 检查标记它。两个平台都有实现。

### Statsig vs GrowthBook

**Statsig**：
- 2025 年 9 月被 OpenAI 以 $1.1B 收购。托管、SaaS。
- 序贯测试、CUPED、隔离人群。
- 一体化：特性开关 + 实验 + 可观测性。
- 最佳适用：团队已经想要打包产品、不关心 OpenAI 所有权。

**GrowthBook**：
- 开源（MIT）；仓库原生（直接从 Snowflake/BigQuery/Redshift 读取）。
- 多引擎：贝叶斯、频率学派、序贯。
- CUPED、SRM、Bonferroni、BH 校正。
- 自托管或托管云。
- 最佳适用：仓库-SQL 团队、数据团队控制指标层、需要开源。

### 非确定性使功效复杂化

相同提示词产生不同输出。传统功效计算假设 IID 观测。有了 LLM 非确定性，有效样本量低于标称值。作为安全边际，将所需样本量乘以约 1.3-1.5 倍。

### 真实案例结果

- 聊天机器人奖励模型变体：+70% 对话长度，+30% 留存。
- Nextdoor 主题行：奖励函数优化后 +1% CTR。
- Khan Academy Khanmigo：延迟-数学准确率之间的迭代权衡。

### 反模式：靠感觉上线

每个高级工程师都能举出一个因为"感觉更好"没有 A/B 就上线的特性。其中大多数在几个月后回退了团队没注意到的产品指标。A/B 就是强迫函数。

### 需要记住的数字

- Statsig 被 OpenAI 收购：$1.1B，2025 年 9 月。
- GrowthBook：MIT 开源；贝叶斯 + 频率学派 + 序贯。
- CUPED 方差缩减：30-70%。
- LLM 非确定性 → 样本量缓冲 +30-50%。

## 动手实现

`code/main.py` 模拟带固定和序贯边界的序贯 A/B 测试。展示序贯如何允许提前停止。

## 交付物

本课产出 `outputs/skill-ab-plan.md`。给定特性变更、工作负载、基线，选择平台、关卡、样本量。

## 练习

1. 运行 `code/main.py`。对于预期提升 5%、基线转化率 3%，80% 功效需要多大样本量？
2. 为医疗监管的本地客户选择 Statsig 或 GrowthBook。
3. 设计一个 A/B 测试 GPT-4 vs GPT-3.5 在每个已解决工单成本上的对比。主要指标、护栏指标、次要指标是什么？
4. 你的金丝雀通过了但 A/B 显示转化率 -1.2%。你上线吗？写出升级标准。
5. 对一个前周期方差占后周期 60% 的情况应用 CUPED。计算有效样本量提升。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| 评测 | "离线测试" | 对模型能力的带标签集合评估 |
| A/B 测试 | "实验" | 在用户上的实时随机比较 |
| CUPED | "方差缩减" | 前周期回归以减少方差 |
| 序贯测试 | "窥视 ok 测试" | 允许提前停止的始终有效程序 |
| 多重比较 | "族误差" | 运行多个测试会膨胀假阳性 |
| Bonferroni | "紧校正" | α 除以测试数量 |
| Benjamini-Hochberg | "BH FDR" | 假发现率控制，不那么保守 |
| SRM | "坏分组" | 样本比例不匹配；分配 bug |
| Statsig | "OpenAI 拥有" | 商业一体化，2025 年被收购 |
| GrowthBook | "那个开源的" | MIT 仓库原生平台 |
| mSPRT | "序贯概率比检验" | 经典序贯程序 |

## 延伸阅读

- [GrowthBook — 如何对 AI 做 A/B 测试](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — 超越提示词：数据驱动的 LLM 优化](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook 对比](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — 置信序列](https://arxiv.org/abs/1810.08240)