# 对齐研究生态系统 — MATS、Redwood、Apollo、METR

> 五个组织构成了 2026 年非实验室对齐研究层。MATS（ML 对齐与理论学者）：自 2021 年底以来 527+ 研究人员、180+ 篇论文、10K+ 引用、h-index 47；2024 年夏季 cohort 作为 501(c)(3) 机构运营，约 90 名学者和 40 名导师；2025 年前校友中 80% 从事安全/安保工作，其中 200+ 在 Anthropic、DeepMind、OpenAI、英国 AISI、RAND、Redwood、METR、Apollo。Redwood Research：由 Buck Shlegeris 创立的应用对齐实验室；引入了 AI Control（课程 10）；与英国 AISI 合作开展控制安全案例研究。Apollo Research：前沿实验室部署前谋略评估；著有《上下文中的谋略》（课程 8）和《迈向 AI 谋略安全案例》。METR（模型评估与威胁研究）：基于任务的能力评估、自主任务时间跨度研究；《前沿 AI 安全政策的共同要素》比较了各实验室框架。Eleos AI Research：模型福利部署前评估（课程 19）；完成了 Claude Opus 4 福利评估。

**类型：** 学习型
**语言：** 无
**前置条件：** 阶段 18 · 01-27（前期阶段 18 课程）
**时间：** 约 45 分钟

## 学习目标

- 识别非实验室对齐研究生态系统的五个组织及其核心产出。
- 描述 MATS 的规模（学者、论文、h-index）及其作为人才输送管道的作用。
- 描述 Redwood 的 AI Control 议程及其与英国 AISI 的合作。
- 描述 METR 基于任务的评估方法论。

## 问题

前沿实验室（课程 18）在内部进行安全评估并选择性地发布结果。实验室外部的生态系统是评估被验证的地方，是新颖失败模式首先被发现的地方，也是人才被培养的地方。理解这个生态系统有助于解读哪些研究发现被谁信任。

## 概念

### MATS（ML 对齐与理论学者）

始于 2021 年底。研究指导项目；学者与资深研究人员共事 10-12 周，研究一个特定的对齐问题。

规模（2026 年）：
- 自成立以来 527+ 研究人员。
- 180+ 篇已发表论文。
- 10K+ 引用。
- h-index 47。
- 2024 年夏季：90 名学者 + 40 名导师；注册为 501(c)(3) 机构。

职业去向：2025 年前校友中约 80% 从事安全/安保工作。200+ 在 Anthropic、DeepMind、OpenAI、英国 AISI、RAND、Redwood、METR、Apollo。

### Redwood Research

应用对齐实验室。由 Buck Shlegeris 创立。引入了 AI Control 议程（课程 10）。与英国 AISI 合作开展控制安全案例研究。为 DeepMind 和 Anthropic 提供评估设计咨询。

代表性论文：Greenblatt, Shlegeris 等，《AI Control》（arXiv:2312.06942，ICML 2024）；对齐伪装（Greenblatt, Denison, Wright 等，arXiv:2412.14093，与 Anthropic 合作）。

风格：具体的威胁模型、最坏情况对手、可压力测试的明确协议。

### Apollo Research

前沿实验室部署前谋略评估。著有《上下文中的谋略》（课程 8，arXiv:2412.04984）。2025 年 OpenAI 反谋略训练合作项目的合作方。产出《迈向 AI 谋略安全案例》（2024）。

风格：能涌现欺骗的智能体设置评估；三支柱分解（不对齐、目标导向、情境意识）。

### METR（模型评估与威胁研究）

基于任务的能力评估。自主任务完成时间跨度研究。《前沿 AI 安全政策的共同要素》（metr.org/common-elements，2025）比较了各实验室框架。

与 Apollo 合作撰写 AI 谋略安全案例草稿。

风格：长时间跨度任务评估、实证能力测量、框架综合。

### Eleos AI Research

模型福利部署前评估。完成了系统卡第 5.3 节记录的 Claude Opus 4 福利评估。为课程 19 中福利相关主张提供外部方法论核查。

### 人才流动

MATS 培养研究人员。毕业生流向 Anthropic、DeepMind、OpenAI（实验室安全团队）或 Redwood、Apollo、METR、Eleos（外部评估）。外部评估方与实验室及英国 AISI / CAISI 合作。出版物反馈给生态系统，供下一届 MATS 使用。

### 为什么这层很重要

单一来源的评估不可靠：实验室评估自己的模型存在结构性利益冲突。外部评估方可以提出和验证实验室可能少报的失败模式。2024 年《休眠代理》论文（课程 7）是 Anthropic + Redwood；对齐伪装是 Anthropic + Redwood；《上下文中的谋略》是 Apollo；反谋略是 Apollo + OpenAI。多组织结构即是质量控制。

### 在阶段 18 中的位置

课程 7-11 引用了 Redwood 和 Apollo 的工作；课程 18 引用了 METR 的框架比较；课程 19 引用了 Eleos。课程 28 是其他课程所依赖的生态系统的明确组织地图。

## 使用它

无代码。阅读 METR 的《前沿 AI 安全政策的共同要素》作为外部综合如何为实验室内部政策工作增加价值的示例。

## 交付它

本课产出 `outputs/skill-ecosystem-map.md`。给定一个对齐主张或评估，它识别相关组织、发表场所和方法论风格，并与已知对应组织进行交叉核查。

## 练习

1. 从课程 7-15 中选择一篇论文，识别涉及的机构。将作者与 MATS 校友和当前生态系统 affiliation 进行交叉核查。

2. 阅读 METR 的《前沿 AI 安全政策的共同要素》。识别他们强调的三个跨实验室趋同点和两个最大分歧点。

3. MATS 职业去向中约 80% 从事安全/安保工作。论证这种选择压力是适应性的（培养该领域）还是有偏的（过滤掉异端立场）。

4. Redwood 和 Apollo 都从事控制/谋略工作，但风格不同。选择一个失败模式，描述每个组织会如何调查它。

5. Eleos AI 是唯一的纯模型福利组织。设计一个假想的第二个组织，聚焦于另一个福利相关问题（认知自由、机器人具身等），并阐明其方法论。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|------------------------|
| MATS | "指导项目" | ML 对齐与理论学者；自 2021 年以来 527+ 研究人员 |
| Redwood Research | "控制实验室" | 应用对齐；AI Control 作者；英国 AISI 合作方 |
| Apollo Research | "谋略评估" | 前沿实验室部署前谋略评估 |
| METR | "任务跨度评估" | 基于任务的能力评估；框架综合 |
| Eleos AI | "福利实验室" | 模型福利部署前评估 |
| 人才输送管道 | "MATS -> 实验室" | MATS 毕业生流向 Anthropic、DM、OpenAI、Redwood、Apollo、METR |
| 外部评估 | "非实验室核查" | 非模型生产方进行的评估；增加可信度 |

## 进一步阅读

- [MATS（ML 对齐与理论学者）](https://www.matsprogram.org/) — 指导项目
- [Redwood Research](https://www.redwoodresearch.org/) — AI Control 论文
- [Apollo Research](https://www.apolloresearch.ai/) — 谋略评估
- [METR — 前沿 AI 安全政策的共同要素](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — 框架比较
- [Eleos AI Research](https://www.eleosai.org/research) — 模型福利方法论
