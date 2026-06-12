# 智能体框架权衡 — LangGraph vs CrewAI vs AutoGen vs Agno

> 每个框架卖的都是同一个演示（研究智能体撰写报告），隐藏的都是同一个 bug（状态 schema 与编排层打架）。选择抽象与问题形状匹配的框架；其余的都是你写两遍的胶水代码。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 11 · 09（函数调用）、阶段 11 · 16（LangGraph）
**时间：** 约 45 分钟

## 问题

你有一个需要超过一次 LLM 调用才能完成的任务。也许是一个研究工作流（计划、搜索、摘要、引用）。也许是一个代码审查流水线（解析差异、批评、打补丁、验证）。也许是一个多回合助手，可以预订航班、写电子邮件和提交费用报告。你选择了一个框架。

三天后，你发现框架的抽象在泄漏。CrewAI 给你角色，但当"研究者"需要将结构化计划交给"写作者"时会与你冲突。AutoGen 给你智能体之间的聊天，但没有一等状态，所以你的检查点是对话日志的 pickle。LangGraph 给你状态图，但强迫你在知道智能体会做什么之前命名每个转换。Agno 给你单智能体抽象，当你试图扇出到三个并发 worker 时就会尖叫。

解决方案不是"选择最好的框架"。而是将框架的核心抽象与问题的形状匹配。本课绘制那张地图。

## 概念

![智能体框架矩阵：核心抽象 vs 问题形状](../assets/framework-matrix.svg)

四个框架在 2026 年占主导地位。它们的核心理念抽象并不相同。

| 框架 | 核心抽象 | 最适合 | 最不适合 |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph` — 类型化状态、节点、条件边、检查点保存器。 | 具有显式状态和人工介入中断的工作流；需要时间回溯调试的生产级智能体。 | 松散的、角色驱动的头脑风暴，拓扑未知。 |
| **CrewAI** | `Crew` — 角色（目标、背景故事）、任务、流程（顺序或层级）。 | 具有短线性/层级计划的角色扮演或 persona 驱动工作流。 | 超出 crew 回转历史的任何有状态内容；复杂分支。 |
| **AutoGen** | `ConversableAgent` 对 — 两个或多个智能体轮流对话直到退出条件。 | 多智能体*对话*（师生、提议者-批评者、演员-评审者），思维从聊天中涌现。 | 已知 DAG 的确定性工作流；任何需要在重启之间持久化的状态。 |
| **Agno** | `Agent` — 单个 LLM + 工具 + 记忆，可组合成团队。 | 快速构建的单智能体和轻量级团队；强大的多模态和内置存储驱动。 | 具有自定义归约器的深度显式分支图。 |

### "抽象"实际意味着什么

框架的核心抽象是当你在白板上展示架构时画的东西。

- **LangGraph** → 你画一个图。节点是步骤，边是转换，每个点的状态对象是类型化的。心智模型是一个状态机。
- **CrewAI** → 你画一张组织图。每个角色有职位描述，经理路由任务。心智模型是一个专家小团队。
- **AutoGen** → 你画一个 Slack DM。两个智能体互相发消息；如果需要moderator，第三个加入。心智模型是聊天。
- **Agno** → 你画一个带工具挂载的单个框。把框并排放就是一个团队。心智模型是"自带电池的智能体"。

### 状态问题

状态是大多数框架选择在生产中崩溃的地方。

- **LangGraph。** 类型化状态（`TypedDict` 或 Pydantic 模型）、逐字段归约器、一等检查点保存器（SQLite/Postgres/Redis）。恢复、中断和时间回溯是免费的。*（见阶段 11 · 16。）*
- **CrewAI。** 状态通过 `context` 字段在任务之间作为字符串流动，或通过 `output_pydantic` 结构化传输。开箱即用没有持久的 per-crew 存储；如果 crew 必须存活重启，你得自己附加。
- **AutoGen。** 状态是聊天历史和任何用户定义的 `context`。对话记录持久化；任意工作流状态不会持久化，除非你写适配器。
- **Agno。** 内置存储驱动（SQLite、Postgres、Mongo、Redis、DynamoDB）通过 `storage=` 附加到 `Agent`——对话会话和用户记忆自动持久化。不是完整的图检查点保存器；是会话存储。

### 分支问题

每个非平凡的智能体都会分支。谁决定分支很重要。

- **LangGraph** — 你通过条件边决定。路由是一个带有命名分支的 Python 函数。分支是编译图中的一等公民；检查点保存器记录走了哪个分支。
- **CrewAI** — 在层级模式下由经理决定；在顺序模式下你在构建时决定。路由隐含在任务列表中；在经理提示之外没有一等的"if"。
- **AutoGen** — 智能体通过聊天决定。分支是从谁说下一句话中涌现的。`GroupChatManager` 选择下一个发言者；你可以手写一个 `speaker_selection_method`，但默认是 LLM 驱动的。
- **Agno** — 智能体由下一步调用哪个工具决定。团队有协调器/路由器/协作者模式；除此之外的分支由开发者负责。

### 可观测性问题

- **LangGraph** — 通过 LangSmith 或任何 OTel 导出器的 OpenTelemetry。每个节点转换是一个跟踪跨度；检查点兼作可重放的跟踪。LangSmith 是一方选项；Langfuse/Phoenix 也有适配器。
- **CrewAI** — 自 2025 年末以来一等 OpenTelemetry；与 Langfuse、Phoenix、Opik、AgentOps 集成。
- **AutoGen** — 通过 `autogen-core` 的 OpenTelemetry 集成；AgentOps 和 Opik 有连接器。跟踪粒度是 per-agent-message，不是 per-node。
- **Agno** — 内置 `monitoring=True` 标志加 OpenTelemetry 导出器；与 Langfuse 的会话跟踪紧密集成。

### 成本和延迟

所有四个框架都增加每次调用开销（框架逻辑、验证、序列化）。大致按增加开销排序：Agno ≈ LangGraph < CrewAI ≈ AutoGen。差异主要由框架做了多少额外 LLM 路由决定。CrewAI 的层级管理器花费 token 决定谁接下来；AutoGen 的 `GroupChatManager` 同样。LangGraph 只在你写 `llm.invoke` 的地方花费 token。Agno 的单智能体路径很薄。

当每次运行成本重要时，优先选择显式路由（LangGraph 边、AutoGen `speaker_selection_method`）而不是 LLM 选择的路由。

### 互操作性

- **LangGraph** ↔ **LangChain** 工具、检索器、LLM。一等 MCP 适配器（工具作为 MCP 服务器导入）。
- **CrewAI** ↔ 工具继承自 `BaseTool`；LangChain 工具、LlamaIndex 工具和 MCP 工具都可以适配。crew 到 crew 委托通过 `allow_delegation=True`。
- **AutoGen** → `FunctionTool` 包装任何 Python 可调用对象；有 MCP 适配器。与 AG2 生态系统紧密耦合，用于智能体到智能体模式。
- **Agno** → `@tool` 装饰器或 BaseTool 子类；MCP 适配器；工具可以在智能体和团队之间共享。

## 技能

> 你能用一句话解释为什么某个框架适合某个智能体问题。

预构建检查清单：

1. **画出形状。** 这是一个图（类型化状态、命名转换）？一场角色扮演（专家交接工作）？一场聊天（智能体对话直到完成）？一个带工具的单智能体？
2. **决定谁分支。** 开发者决定分支 → LangGraph。经理-智能体决定 → CrewAI 层级。聊天涌现 → AutoGen。工具调用决定 → Agno。
3. **检查状态预算。** 你需要从检查点恢复？时间回溯？运行中的人工中断？如果是，LangGraph 是默认值；Agno 会话覆盖会话作用域的状态。
4. **检查成本预算。** LLM 选择的路由每回合花费额外 token。如果智能体每天运行数千次，优先选择显式路由。
5. **计算框架开销。** 每个框架都是另一个依赖。如果任务只需要两次 LLM 调用和一个工具，写 30 行普通 Python；没有框架比没有框架更便宜。

在你能在白板上画出图、组织图、聊天或智能体框之前，不要伸手要框架。拒绝选择一个迫使你为实际需要的东西与它的状态模型打架的框架。

## 决策矩阵

| 问题形状 | 首选框架 | 为什么 |
|---------------|---------------------|-----|
| 具有类型化状态、人工审批、长期运行的 workflow DAG | LangGraph | 一等状态、检查点保存器、中断、时间回溯。 |
| 具有不同角色的研究/写作流水线 | CrewAI（顺序）或 LangGraph 子图 | CrewAI 中角色-per-task 表达廉价；当分支变复杂时用 LangGraph 扩展。 |
| 提议者-批评者或师生对话 | AutoGen | 两个智能体聊天是其原生形状。 |
| 带工具、会话、记忆的单智能体 | Agno | 最薄设置，内置存储和记忆。 |
| 具有归约器的数千个并行扇出 | LangGraph + `Send` | 唯一具有一等并行分派 API 的。 |
| 快速原型，不承诺框架 | 普通 Python + 提供商 SDK | 没有框架是最快的框架。 |

## 练习

1. **简单。** 取相同任务——"研究 Anthropic 总部，写 200 字简报，引用来源"——在 LangGraph（四个节点：计划、搜索、写作、引用）和 CrewAI（三个角色：研究者、写作者、编辑）中实现。报告每次运行的 token 成本和代码行数。
2. **中等。** 在 AutoGen（研究者和写作者聊天，编辑通过 `GroupChat` 加入）和 Agno（带 `search_tools` 和 `write_tools` 的单个智能体，外加会话存储）中构建相同任务。排名四个实现：(a) 每次运行成本，(b) 崩溃后恢复能力，(c) 在写作步骤前注入人工审批的能力。
3. **困难。** 构建一个决策树脚本 `pick_framework.py`，接受简短问题描述（JSON：`{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`），返回一个带一句话理由的建议。用你自己设计的六个案例验证。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 编排（Orchestration） | "智能体如何协调" | 决定下一个运行哪个节点/角色/智能体的层。 |
| 持久状态（Durable state） | "重启后恢复" | 存活进程死亡的状态，附加到检查点或会话存储。 |
| LLM 选择的路由 | "让模型决定" | 计划 LLM 每回合选择下一步；灵活但每次决策都要花费 token。 |
| 显式路由 | "开发者决定" | Python 函数或静态边选择下一步；廉价且可审计。 |
| Crew | "CrewAI 团队" | 角色 + 任务 + 流程（顺序或层级）绑定到单个可运行对象。 |
| GroupChat | "AutoGen 的多智能体聊天" | N 个智能体之间带发言者选择器的托管对话。 |
| 团队（Agno） | "多智能体 Agno" | 一组智能体上的路由/协调/协作模式。 |
| StateGraph | "LangGraph 的图" | 类型化状态、节点、条件边、检查点保存器抽象。 |

## 延伸阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — StateGraph、检查点保存器、中断、时间回溯。
- [CrewAI 文档](https://docs.crewai.com/) — Crews、Flows、Agents、Tasks、Processes。
- [AutoGen 文档](https://microsoft.github.io/autogen/) — ConversableAgent、GroupChat、teams、tools。
- [Agno 文档](https://docs.agno.com/) — Agent、Team、Workflow、storage、memory。
- [Anthropic — 构建有效的智能体（2024 年 12 月）](https://www.anthropic.com/research/building-effective-agents) — 模式库（提示词链、路由、并行化、orchestrator-workers、evaluator-optimizer）框架无关。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — 每个框架都打扮的循环。
- [Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)](https://arxiv.org/abs/2308.08155) — AutoGen 的设计论文。
- [Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023)](https://arxiv.org/abs/2304.03442) — CrewAI 风格 persona 栈构建的角色扮演基础。
- 阶段 11 · 16（LangGraph）—— 本课与之对比的框架。
- 阶段 11 · 19（Reflexion）—— 一种干净地映射到 LangGraph 但别扭地映射到 CrewAI 的模式。
- 阶段 11 · 22（生产可观测性）—— 如何检测你选择的任何框架。
