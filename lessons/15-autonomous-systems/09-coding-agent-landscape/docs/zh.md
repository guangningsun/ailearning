# 自主编码智能体领域（2026）

> SWE-bench Verified 在不到三年内从 4% 上升到 80.9%。同一 Claude Sonnet 4.5 在 SWE-agent v1 上得分 43.2%，在 Cline 自主模式上得分 59.8%——模型周围的脚手架现在与模型本身同样重要。OpenHands（原 OpenDevin）是最活跃的 MIT 许可平台，其 CodeAct 循环直接在沙箱中执行 Python 操作而非 JSON 工具调用。 headline 数字掩盖了一个方法论问题：500 个 SWE-bench Verified 任务中有 161 个只需要 1-2 行修改，而 SWE-bench Pro（10 行以上任务）在同一前沿模型上仅为 23-59%。

**类型：** 学习型
**语言：** Python（标准库、CodeAct 与 JSON 工具调用比较）
**前置条件：** 阶段 14 · 07（工具使用）、阶段 15 · 01（长时程智能体）
**时间：** 约 45 分钟

## 问题

"哪个编码智能体最好"是错误的问题。正确的问题是：在与我的工作相匹配的任务分布上，在我将在生产中运行的脚手架下，我能获得怎样的端到端可靠性？

在 2022 年到 2026 年之间，该领域认识到脚手架——检索层、规划器、沙箱、编辑验证循环、反馈格式——是承重的。Claude Sonnet 4.5 在 SWE-agent v1 上对 SWE-bench Verified 得分 43.2%；同一模型在 Cline 自主脚手架内得分 59.8%。16.6 个绝对百分点的差异，相同的权重。基础模型是一个组件；循环是产品。

配套问题是基准测试饱和隐藏了回归。SWE-bench Verified 接近饱和，而简单任务尾（500 个任务中有 161 个只需要 ≤2 行）将最高分拉高。现实世界的质量在 SWE-bench Pro（10 行以上修改）等分布上衡量更好，同一领先者仍停留在 23-59%。

## 概念

### SWE-bench，一段话

SWE-bench（Jimenez 等人）取真实的 GitHub issue 和带真值 patch，要求智能体生成使测试套件通过的 patch。SWE-bench Verified（OpenAI，2024）是一个人工策划的 500 任务子集，移除了歧义和损坏的任务。SWE-bench Pro 是更难的后继者——需要 10 行以上修改的任务，当前前沿智能体停留于 23-59%。

### 2022 → 2026 曲线实际展示了什么

- **2022**：原始 SWE-bench 上研究模型约 4%。
- **2024**：GPT-4 + Devin 风格脚手架约 14%；SWE-agent 约 12%。
- **2025**：Claude 3.5/3.7 Sonnet 在 Aider 和 SWE-agent 内推进到 40-55% 范围。
- **2026**：Claude Sonnet 4.5 和前沿竞争者在 SWE-bench Verified 上达到 70-80%+。Epoch AI 的排行榜实时追踪这一趋势。

斜率来自三个复合来源：更好的基础模型、更好的脚手架（CodeAct、反思、验证器循环）、更好的基准（Verified 移除噪声）。

### CodeAct 与 JSON 工具调用

OpenHands（All-Hands-AI，arXiv:2407.16741，原 OpenDevin）采取了一个特定架构赌注：不是让模型发出 JSON 工具调用由主机解码执行，而是让模型发出 Python 代码，由 Jupyter 风格的内核在沙箱中运行。智能体可以在一个操作内循环遍历文件、链接工具并捕获自己的异常。

权衡：

- **JSON 工具调用**：每个动作是一轮；易于审计；组合性有限；每个调用通过显式验证器，默认安全。
- **CodeAct**：一个操作可以是一个完整程序；组合性强；需要加固沙箱（OpenHands 使用 Docker 隔离）；失败模式包括沙箱运行时允许的任何行为。

两种架构都在生产中。CodeAct 在开放平台占主导（OpenHands、smolagents）。JSON 工具调用在托管服务中仍占主导（Anthropic Managed Agents、OpenAI Assistants），因为提供商控制执行器。

### 2026 年领域的脚手架

| 脚手架 | 许可证 | 执行模型 | 显著属性 |
|---|---|---|---|
| OpenHands (OpenDevin) | MIT | Docker 中的 CodeAct | 最活跃的开放平台；事件流可重放 |
| SWE-agent | MIT | 智能体-计算机接口 (ACI) | 首个端到端 SWE-bench 脚手架 |
| Aider | Apache-2 | 在本地仓库通过 diff 编辑 | 最小脚手架，强大的回归稳定性 |
| Cline | Apache-2 | 带工具策略的 VS Code 智能体 | Sonnet 4.5 上最高得分的开放脚手架 |
| Devin (Cognition) | 专有 | 托管 VM + 规划器 | 首个"AI 软件工程师"产品类别 |
| Claude Code | 专有 | 权限模式 + 例程 | 第 10 课详细涵盖智能体循环 |

### 为什么脚手架占主导地位

编码运行是一条长时程轨迹（第 1 课）。可靠性在各步骤间复合。脚手架在三个地方赢得分数：

1. **检索**：找到正确的文件来读取是静默瓶颈。SWE-agent 的 ACI、OpenHands 的文件索引和 Aider 的 repo-map 都针对此问题。
2. **验证器循环**：运行测试、读取堆栈跟踪和重新尝试在 SWE-bench 上产生 10+ 分差。
3. **故障containment**：出错时回滚的沙箱可防止复合损害。有无验证器循环的同一模型看起来像两个不同的产品。

### 基准测试饱和与真实分布

OpenHands 作者和 Epoch AI 都指出 SWE-bench Verified 有一个简单尾：500 个任务中有 161 个只需要 1-2 行修改。高分部分由这个尾驱动。SWE-bench Pro 限制为 10 行以上修改，对前沿系统返回 23-59% 的分数。你的生产分布几乎肯定更接近 Pro 而非 Verified。

选择智能体的启示：运行你自己 bug 待办中类似 Pro 的子集。重要的分数是在与你交付内容相符的任务上的分数。

## 使用

`code/main.py` 在固定的迷你任务分布上比较两个玩具智能体脚手架：

1. 一个**JSON 工具调用**脚手架，每轮执行一个动作。
2. 一个 **CodeAct** 脚手架，每动作可以发出一小段 Python。

两者都使用存根"模型"（确定性规则），因此比较将脚手架与模型质量隔离。输出显示 CodeAct 脚手架以更少的轮次解决更多任务，代价是每动作爆炸半径更大。

## 交付

`outputs/skill-scaffold-audit.md` 帮助你在采用前审计拟议的编码智能体脚手架：检索质量、验证器存在、沙箱隔离以及基准到分布的契合度。

## 练习

1. 运行 `code/main.py`。每个脚手架在同一任务集上用多少轮？每个的每动作爆炸半径是多少？

2. 阅读 OpenHands 论文（arXiv:2407.16741）。该论文认为 CodeAct 在复杂任务上优于 JSON 工具调用。找出论文承认的一个失败模式，并用一句话写出该模式何时在生产中占主导。

3. 从你的 bug 待办中选一个需要跨两个文件修改 10+ 行的任务。估计前沿模型在 (a) JSON 工具调用和 (b) CodeAct 下的端到端成功概率。说明差距的理由。

4. SWE-bench Verified 有 161 个单文件、1-2 行任务。构建一个排除它们的分数。排行榜如何重新排列？

5. 阅读"Introducing SWE-bench Verified"（OpenAI）。解释用于移除歧义任务的具体方法，并命名策划会遗漏的一个类别。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|---|---|---|
| SWE-bench | "编码基准测试" | 带真值 patch 和测试套件的真实 GitHub issue |
| SWE-bench Verified | "清理子集" | 500 个人工策划任务，存在简单尾 |
| SWE-bench Pro | "更难子集" | 10 行以上修改；前沿停留在 23-59% |
| CodeAct | "代码即动作" | 智能体发出 Python；Jupyter 风格内核在沙箱中执行 |
| JSON 工具调用 | "函数调用" | 每个动作是一个结构化 JSON 有效载荷，在执行前验证 |
| 脚手架 | "智能体框架" | 围绕基础模型的检索 + 规划器 + 执行器 + 验证器循环 |
| ACI（智能体-计算机接口） | "SWE-agent 的格式" | 为 LLM 人体工程学设计而非人类 shell 的命令集 |
| 验证器循环 | "测试并重试" | 运行测试、读取输出、修改 patch；最大的非模型可靠性增益 |

## 延伸阅读

- [Jimenez 等人——SWE-bench](https://www.swebench.com/) — 原始基准测试和方法论。
- [OpenAI——Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 策划子集是如何构建的。
- [Wang 等人——OpenHands：一个 AI 软件开发者的开放平台](https://arxiv.org/abs/2407.16741) — CodeAct 架构和事件流设计。
- [Epoch AI——SWE-bench 排行榜](https://epoch.ai/benchmarks) — 实时追踪分数。
- [Anthropic——衡量智能体自主性](https://www.anthropic.com/research/measuring-agent-autonomy) — 长时程编码智能体可靠性框架。