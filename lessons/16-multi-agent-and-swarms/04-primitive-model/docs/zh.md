# 多智能体原语模型

> 2026 年发布的每一个多智能体框架——AutoGen、LangGraph、CrewAI、OpenAI Agents SDK、Microsoft Agent Framework——都只是四维设计空间中的一个点。四个原语，仅此而已：智能体（Agent）、交接（Handoff）、共享状态（Shared State）、编排器（Orchestrator）。本节课从零开始构建这四者，在一个玩具系统上运行全部四种，然后把它们映射到相同的轴上——让你读完任何新框架的发布公告后只需读一段就能理解。

**类型：** 学习型
**语言：** Python（标准库）
**前置条件：** 阶段 14（智能体工程）、阶段 16·01（为什么需要多智能体）
**时间：** 约 60 分钟

## 问题

每六个月就会有一个新的多智能体框架发布。2023 年的 AutoGen。2024 年的 CrewAI。2024 年的 LangGraph 和 OpenAI Swarm。2025 年 4 月的 Google ADK。2026 年 2 月的 Microsoft Agent Framework RC。每篇新闻稿都声称自己是"正确的抽象"。

如果你一个一个地去学，会耗尽精力。API 长得不一样。文档对"智能体"的定义各执一词。一个框架把共享内存叫"黑板"，另一个叫"消息池"，第三个叫"StateGraph"。你开始怀疑这个领域只是在空转。

其实不是。在营销术语之下，四个原语是稳定的。学一次，就能用一段话读懂每一个新框架。

## 概念

### 四个原语

1. **智能体（Agent）**——系统提示词加工具列表。无状态；每次运行都从其系统提示词和当前消息历史开始。
2. **交接（Handoff）**——从一个智能体到另一个智能体的结构化控制转移。机制上，是一个返回新智能体的工具调用，或一条跟随条件判断的图边。
3. **共享状态（Shared state）**——任何可被多个智能体读取（有时是写入）的数据结构。消息池、黑板、键值存储、向量内存。
4. **编排器（Orchestrator）**——决定下一个谁发言的人。选项包括：显式图（确定性）、LLM 说话者选择器（软路由）、上一个说话者的交接调用（OpenAI Swarm），或队列上的调度器（swarm 架构）。

这就是整个设计空间。每个框架在每个轴上选默认值；其余的都是表面语法。

### 2026 年各框架如何映射到它上面

| 框架 | 智能体 | 交接 | 共享状态 | 编排器 |
|-----------|-------|---------|--------------|--------------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | 工具返回 Agent | 调用者的 problem | LLM 的下一次交接调用 |
| AutoGen v0.4 / AG2 | `ConversableAgent` | GroupChat 上的说话者选择器 | 消息池 | 选择器函数（LLM 或轮询） |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | 链接的任务输出 | 管理者 LLM 或静态顺序 |
| LangGraph | 节点函数 | 图边 + 条件 | `StateGraph` 归约器 | 图，确定性 |
| Microsoft Agent Framework | 智能体 + 编排模式 | 模式特定 | 线程 / 上下文 | 模式特定 |
| Google ADK | 智能体 + A2A 卡 | A2A 任务 | A2A 制品 | 主机决定 |

表面差异看起来很大。内在：相同的四个旋钮。

### 为什么这很重要

一旦看清了原语，框架比较就变成了一张简短的检查清单：

- 编排器是否信任 LLM 来路由（Swarm），还是把路由写死在代码里（LangGraph）？
- 共享状态是完整历史（GroupChat）还是投影的（StateGraph 归约器）？
- 智能体能否修改彼此的提示词（CrewAI 管理者）还是只能交接（Swarm）？

这三个问题回答了"哪个框架适合某个问题"的 80%。你不再去挑选"最好的多智能体框架"，而是开始为真正关心的那个轴做设计。

### 无状态的洞察

除了共享状态之外，每个原语都是无状态的。智能体是（提示词，工具）的函数。交接是一个函数调用。编排器是一个调度器。**系统中唯一有状态的东西是共享状态。** 那里才是一切有趣的 bug 所在：记忆污染（第 15 课）、消息排序、版本控制、写竞争。

隐藏共享状态的框架（Swarm）把问题推给了调用者。集中化共享状态的框架（LangGraph 检查点、AutoGen 池）使其可检查，但将协调成本转移到了共享状态的实现上。

### 单个原语的解剖

#### 智能体

```
Agent = (system_prompt, tools, model, optional_name)
```

无记忆。无状态。两个具有相同系统提示词和工具的智能体可以互换。所有看起来像"每个智能体各自的状态"的东西，实际上都在共享状态里或在交接协议里。

#### 交接

```
Handoff = (from_agent, to_agent, reason, payload)
```

三种实现占主导：

- **函数返回**——工具返回下一个智能体。这是 OpenAI Swarm 模式。智能体在工具模式中携带路由信息。
- **图边**——LangGraph。边是声明式的。LLM 产生一个值；条件选择下一个节点。
- **说话者选择**——AutoGen GroupChat。一个选择器函数（有时本身就是一个 LLM 调用）读取池并挑选下一个发言者。

#### 共享状态

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

最少需要一个消息列表。通常还有更多：结构化制品（CrewAI 任务输出）、类型化上下文（LangGraph 归约器）、外部记忆（MCP、向量数据库）。

两种拓扑：**完整池**（每个智能体看到每条消息）和**投影**（智能体看到一个角色作用域的视图）。完整池简单但扩展性差。投影池可扩展但需要预先设计模式。

#### 编排器

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

四种口味：

- **静态**——图在构建时固定（LangGraph 确定性、CrewAI 顺序）。
- **LLM 选择**——LLM 读取池并挑选下一个发言者（AutoGen、CrewAI 层级）。
- **交接驱动**——当前智能体通过调用交接工具来决定（Swarm）。
- **队列驱动**——工作线程从共享队列中拉取工作；没有明确的下一个发言者（swarm 架构、Matrix）。

### 框架之间什么会变化

一旦原语固定，剩下的设计决策是：

- **记忆策略**——临时 vs 持久化检查点（LangGraph 检查点器）。
- **安全边界**——谁可以批准一次交接（人在环中）。
- **成本核算**——每个智能体的 token 预算。
- **可观测性**——追踪交接，持久化状态以供回放。

所有这些都可以在原语之上实现。没有一个是新的原语。

## 构建它

`code/main.py` 用约 150 行标准库 Python 实现了四个原语。没有真正的 LLM——每个智能体都是一个脚本化的策略，这样重点始终放在协调结构上。

该文件导出：

- `Agent`——名称、系统提示词、工具、策略函数的数据类。
- `Handoff`——返回一个新智能体的函数。
- `SharedState`——线程安全的消息池。
- `Orchestrator`——三种变体：`StaticOrchestrator`、`HandoffOrchestrator`、`LLMSelectorOrchestrator`（模拟）。

演示用所有三种编排器类型运行相同的三个智能体管道（研究 → 写作 → 审查），并在最后打印消息池。你可以看到输出只在*谁选择下一个*上有所不同；智能体和共享状态在所有运行中是相同的。

运行它：

```
python3 code/main.py
```

预期输出：三次编排器运行，每个模式一次。每次都打印最终的消息池。如果研究者提前决定完成，则交接驱动的运行会到达更少的智能体——这是 LLM 路由权衡的微型版本。

## 使用它

`outputs/skill-primitive-mapper.md` 是一个技能，读取任何多智能体代码库或框架文档，返回四原语映射。在新框架发布时运行它，可以在深入阅读文档之前获得一段话的理解。

## 发布它

在采用新框架之前，为它写一个原语映射。如果你写不出来，文档是不完整的，或者该框架发明了一个第五原语（罕见——检查是否有你还没见过的共享状态风味）。

把映射固定在你的架构文档中。当新团队成员加入时，在 API 文档之前发送映射给他们。当框架版本更改时，diff 映射，而不是 changelog。

## 练习

1. 用不同的智能体策略运行三次 `code/main.py`。观察编排器选择如何改变运行的智能体。
2. 实现第四种编排器类型：一个队列驱动的，智能体从共享状态中轮询工作。什么死锁可能发生，你如何检测它？
3. 取 LangGraph 快速入门（https://docs.langchain.com/oss/python/langgraph/workflows-agents）并将其重写为四个原语。LangGraph 的哪些抽象 1:1 映射，哪些是便利包装？
4. 阅读 OpenAI Swarm cookbook（https://developers.openai.com/cookbook/examples/orchestrating_agents）。识别四个原语中 Swarm 最符合人体工程学的哪一个，以及它把哪一个推给了调用者。
5. 在本表中找一个完全隐藏共享状态的框架。解释当智能体需要在交接时协调而不重新读取历史时，什么会出问题。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| 智能体（Agent） | "带工具的 LLM" | 一个 `(system_prompt, tools, model)` 三元组。无状态。 |
| 交接（Handoff） | "控制转移" | 一个结构化调用，命名下一个智能体和可选的有效载荷。三种实现：函数返回、图边、说话者选择。 |
| 共享状态（Shared state） | "记忆" / "上下文" | 多智能体系统中唯一有状态的部分。消息池或黑板。 |
| 编排器（Orchestrator） | "协调器" | 决定谁下一个运行的人。静态图、LLM 选择器、交接驱动或队列驱动。 |
| 原语（Primitive） | "抽象" | 每个框架参数化的四个轴之一。不是框架功能。 |
| 消息池（Message pool） | "共享聊天历史" | 完整历史的共享状态。易于推理，扩展性差。 |
| 投影状态（Projected state） | "作用域视图" | 共享状态的角色特定视图。可扩展，需要模式设计。 |
| 说话者选择（Speaker selection） | "下一个谁说" | 编排器模式，一个函数（通常是 LLM）从一组中挑选下一个智能体。 |

## 进一步阅读

- [OpenAI cookbook：编排智能体——例程和交接](https://developers.openai.com/cookbook/examples/orchestrating_agents)——最清晰的交接驱动编排阐述
- [AutoGen 稳定文档](https://microsoft.github.io/autogen/stable/)——GroupChat + 说话者选择是 LLM 选择编排的参考
- [LangGraph 工作流和智能体](https://docs.langchain.com/oss/python/langgraph/workflows-agents)——图边编排和基于归约器的共享状态
- [CrewAI 介绍](https://docs.crewai.com/en/introduction)——角色-目标-背景故事智能体，顺序/层级流程
- [AG2（社区 AutoGen 延续）](https://github.com/ag2ai/ag2)——微软将 v0.4 转入维护后，AutoGen v0.2 线的活跃延续
