# CrewAI：基于角色的 Crew 与 Flows

> CrewAI 是 2026 年基于角色的多智能体框架。四个原语：Agent、Task、Crew、Process。两种顶层形态：Crews（自主、基于角色的协作）和 Flows（事件驱动、确定性）。文档直言不讳："对于任何生产就绪的应用，从 Flow 开始。"

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置条件：** 阶段 14 · 12（工作流模式）、阶段 14 · 14（Actor 模型）
**时间：** 约 75 分钟

## 学习目标

- 说出 CrewAI 的四个原语（Agent、Task、Crew、Process）以及每个原语拥有什么。
- 区分 Sequential、Hierarchical 和计划中的 Consensus 流程；根据工作负载选择其一。
- 区分 Crews（自主基于角色）与 Flows（事件驱动确定性），并解释文档的生产建议。
- 用 `@tool` 装饰器和 `BaseTool` 子类连接工具；理解结构化输出与自由文本的区别。
- 说出四种 CrewAI 内存类型以及各自的适用场景。
- 实现一个标准库的三智能体 crew（研究员、作家、编辑），产出简报。
- 识别三种 CrewAI 失败模式：提示词膨胀、管理员 LLM 税、脆弱的交接。

## 问题

采用多智能体框架的团队总会撞上同一堵墙。"自主协作"在演示中听起来很棒。然后客户提交了一个 bug，你需要确定性回放。或者财务问你一个 LLM 路由的 crew 每次运行要花多少钱。或者值班人员在凌晨 3 点想知道哪个智能体卡住了。

自由形式的 LLM 路由 crew 无法干净利落地回答这些问题。纯 DAG 可以回答所有问题，但会失去头脑风暴智能体需要的探索性形态。

CrewAI 的分裂诚实面对了这个权衡。Crews 用于协作、基于角色、探索性的工作。Flows 用于事件驱动、代码所有、可审计的生产。同一个框架，两种形态，按表面选择。

## 概念

### 四个原语

CrewAI 的表面很小。记住这个，其余的都是配置。

- **Agent。** `role + goal + backstory + tools + (optional) llm`。backstory 是承重的。它塑造语气、判断、以及智能体何时停止。tools 是智能体可以调用的函数（详见下文）。
- **Task。** `description + expected_output + agent + (optional) context + (optional) output_pydantic`。一个可重用的工作单元。`expected_output` 是契约。`context` 列出上游任务的输出，这些输出会被传入。`output_pydantic` 强制一种结构化形态。
- **Crew。** 容器。拥有 `agents` 列表、`tasks` 列表、`process`，以及可选的 `memory` + `verbose` + `manager_llm` 设置。
- **Process。** 执行策略。Sequential、Hierarchical、Consensus（计划中）。选择运行的形态。

智能体之间不直接互相看见。Task 引用智能体。 Crew 对任务进行排序。Process 决定谁来挑选下一个任务。这就是整个心智模型。

> **已针对** CrewAI 0.86（2026-05）验证。新版本可能重命名或合并流程类型；在依赖特定形态之前，请查看 [CrewAI Processes 文档](https://docs.crewai.com/concepts/processes)。

### Sequential vs Hierarchical vs Consensus

- **Sequential。** 任务按声明顺序运行。任务 N 的输出可作为 `context` 供任务 N+1 使用。最低成本。最可预测。当顺序固定时使用。
- **Hierarchical。** 一个管理智能体（单独的 LLM 调用）在专家之间路由。 CrewAI 从你的 `manager_llm` 配置或默认值中生成管理智能体。管理智能体每轮选择下一个任务，并且可以拒绝或重新路由。当你有四个或更多专家且顺序真正依赖于先前输出时使用。
- **Consensus。** 计划中，公开 API 中尚未实现。文档为未来的投票流程保留了这个名称。今天不要依赖它。

Hierarchical 在每个专家调用之上增加了每轮 LLM 调用（管理智能体）。在五步运行中，Token 成本可能增加两倍。只有在需要路由时才为此付费。

### Crews vs Flows

这是 2026 年文档开篇的框架。

- **Crew。** LLM 驱动的自主性。框架在运行时选择形态。适用于：研究、头脑风暴、初稿、路径本身就是答案的任何地方。难以回放。难以测试。原型便宜。
- **Flow。** 你拥有的事件驱动图。`@start` 标记入口。`@listen(topic)` 标记一个步骤，该步骤在另一个步骤发出该主题时触发。每个步骤都是普通 Python（可以在内部调用 Crew）。适用于：生产。可观察。可测试。确定性。

文档的 2026 年生产建议：从 Flow 开始。当自主性证明其成本值得时，在 Flow 步骤内部以 `Crew.kickoff()` 调用的形式加入 Crew。Flow 给你审计跟踪，Crew 给你探索性。组合使用，不要选择。

### 工具集成

给 Agent 配备工具的三种方式。选择最适合你的最简单的。

1. **`@tool` 装饰器。** 纯函数变成工具。签名是模式；文档字符串是 LLM 看到的描述。最适合一次性辅助函数。

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` 子类。** 带显式参数模式、异步支持、重试的类工具。当工具有状态（客户端、缓存）或需要结构化参数时使用。

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **内置工具包。** CrewAI 附带了一手适配器：`SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`。一个导入即可连接。

结构化输出使用 Pydantic。在 Task 上传递 `output_pydantic=MyModel`。 CrewAI 根据模型验证 LLM 响应，要么强制转换要么重试。将此与严格的 `expected_output` 字符串配对。自由文本输出适合初稿；结构化输出是下游 Flows 可以消费的内容。

### 内存钩子

 CrewAI 开箱即用四种内存类型。它们可以组合：一个 Crew 可以同时启用所有四种。

> **已针对** CrewAI 0.86（2026-05）验证。最新版本通过统一的 `Memory` 系统路由所有内容，包装了这四个存储区。下面的概念模型仍然成立，但公开类表面可能在更新版本中折叠为单个 `Memory` 入口点；查看 [CrewAI 内存文档](https://docs.crewai.com/concepts/memory) 了解当前 API。

- **短期。** 单次运行中的对话缓冲区。在结束时清除。
- **长期。** 跨运行持久化。存储在向量数据库中（默认为 Chroma，可交换）。通过与当前任务的相似度检索。
- **实体。** 每个实体的facts。"客户 X 在企业计划中。"按实体键控，而非按相似度。跨运行存活。
- **上下文。** 组装时检索。在 Agent 需要它的时刻拉取相关内存，而不是预加载。

在 Crew 上用 `memory=True` 或按类型配置启用。由你配置的嵌入提供程序支持（默认为 OpenAI，可交换为本地）。内存是 CrewAI 相对于更轻量级框架的优势之一；纯 LangGraph 需要你自己连接这些。### CrewAI 何时适用

- 三到六个具有命名角色和协作工作流的智能体。起草、审查、规划、头脑风暴。
- 路由，其中 LLM 对下一步的判断是价值的一部分（Hierarchical）。
- 团队更愿意阅读 `role + goal + backstory` 而不是阅读图定义的地方。

### CrewAI 何时不适用

- 具有严格顺序的确定性 DAG。使用 LangGraph（第 13 课）。图形态是正确的抽象； CrewAI 的角色框架是摩擦。
- 亚秒级延迟预算。 Hierarchical 增加往返。即使 Sequential 也会序列化包含背景故事和先前输出的提示。
- 单智能体循环。跳过框架；智能体循环（第 1 课）加上工具注册表更短。

第 17 课（智能体框架权衡）以矩阵形式说明了这一点。简短版本： CrewAI 位于"协作基于角色"的角落。

### 依赖形态

独立于 LangChain。Python 3.10 到 3.13。使用 `uv`。Star 数量：见 [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)（截至 2026-05 的快照）。AWS Bedrock 集成有文档记录；供应商基准测试报告在 QA 工作负载上比 LangGraph 有显著加速，但方法论（数据集、硬件、评估指标）未公布，因此将框架供应商数字视为方向性指导。

### 这个模式哪里出了问题

- **背景故事导致的提示词膨胀。** 每个智能体 2000 字的背景故事和五个智能体的 crew 在第一次工具调用之前就烧掉了上下文预算。将背景故事保持在 200 字以下。在智能体之间重用短语；不要重复五次 house style。
- **管理 LLM Token 税。** Hierarchical 流程在每个专家调用之前添加一个管理 LLM 调用。在五个任务的 crew 上，那是六个 LLM 调用而不是五个，并且管理调用携带完整任务列表加上先前输出。除非路由依赖于输出，否则切换到 Sequential。
- **脆弱的交接。** 任务 N 的 `expected_output` 是"一个大纲"。任务 N+1 将其作为 `context` 读取并尝试解析三个部分。LLM 产生了四个。下游智能体即兴发挥。用 `output_pydantic` 修复任务 N，这样任务 N+1 读取一个类型化对象，而不是自由文本。
- **Crew 即生产。** 自由形式的 Crew 运送到生产环境而没有 Flow 包装器。输出变异性高；回放不可能；值班人员无法将坏运行与好运行进行差异对比。用 Flow 包装。

## 构建它

`code/main.py` 实现了两种形态的标准库版本以及一个三智能体 crew。

形态：

- `Agent`、`Task` 数据类匹配 CrewAI 的表面。
- `SequentialCrew.kickoff(inputs)` 按声明顺序运行任务，通过 `context` 传递输出。
- `HierarchicalCrew.kickoff(topic)` 添加一个管理智能体，每轮选择下一个专家，停在"完成"。
- `Flow` 带有 `@start` 和 `@listen(topic)` 装饰器、一个小型事件循环和一个跟踪。
- `tool(name)` 装饰器镜像 CrewAI 的 `@tool` 形态。
- `Memory` 带有 `short_term`、`long_term`、`entity` 存储；模拟相似度使用 numpy。
- 模拟 LLM 响应是基于角色加上输入前缀的硬编码字符串。没有网络。确定性。

具体演示：研究员、作家、编辑 crew 产出关于"2026 智能体工程"的简报。研究人员拉取（模拟的）来源。作家起草。编辑收紧。同一个 crew 通过 Flow 运行以展示确定性形态。

运行它：

```bash
python3 code/main.py
```

跟踪涵盖：顺序 crew 通过 `context` 传递输出、带有管理器选择的研究员、作家、编辑然后"完成"的分层 crew、带有明确主题（`researched`、`drafted`、`edited`）的相同三步 Flow、通过 `@tool` 路由的工具调用，以及跨两次 kickoff 存活的长期内存。

Crew 跟踪是流畅的；管理器原则上可以重新排序。 Flow 跟踪是固定的。这个选择就是教训。

## 使用它

- **CrewAI Flow** 用于生产。即使 Flow 是一个调用 `Crew.kickoff()` 的单一步骤。 Flow 给出了审计边界。
- **CrewAI Crew (Sequential)** 用于顺序清晰的协作工作，特别是初稿和审查循环。
- **CrewAI Crew (Hierarchical)** 当路由依赖于输出且你有四个或更多专家时。
- **LangGraph**（第 13 课）用于显式状态机、持久恢复、严格排序。
- **AutoGen v0.4**（第 14 课）用于 Actor 模型并发和故障隔离。
- **OpenAI Agents SDK**（第 16 课）用于带有交接和护栏的 OpenAI 优先产品。
- **Claude Agent SDK**（第 17 课）用于带有子智能体和会话存储的 Claude 优先产品。

## 发货

`outputs/skill-crew-or-flow.md` 为任务选择 Crew vs Flow 并搭建最小实现。硬性拒绝没有背景故事的 Crew、没有显式主题的 Flow、少于三个专家的 Hierarchical。

## 陷阱

- **背景故事作为风味。** 它塑造输出。为每个智能体测试三种变体；变异性是真实的。选择一个，冻结它。
- **跳过 `expected_output`。** 没有每个任务的契约，下游任务会接收到 LLM 产生的任何内容。 Crew 运行；审计失败。
- **内存始终开启。** 每次运行都写入长期数据。向量数据库增长。检索变得嘈杂。将写入范围限定在事实持久化的任务上。
- **管理提示词漂移。** Hierarchical 的管理提示词是隐式的。如果路由变得奇怪，在 verbose 模式下转储并读取。
- **Crews 中的工具副作用。** Crew 可能比预期更频繁地调用工具。POST、DELETE、支付属于 Flow 步骤，绝不是 Crew 工具。

## 练习

1. 将 Sequential crew 转换为 Flow。计算变异性下降的接触点。注意可读性下降的地方。
2. 将实体内存添加到 crew：关于客户的事实在 kickoff 之间持久化。验证检索拉取正确的实体。
3. 实现一个 Hierarchical 流程，其中管理器在作家的输出至少有三段之前拒绝路由到编辑。跟踪重试。
4. 为（模拟的）网络搜索连接一个 `BaseTool` 子类。与 `@tool` 装饰器版本比较跟踪形态。
5. 将 `output_pydantic=Brief` 添加到编辑任务，其中 `Brief` 有 `title`、`summary`、`sections`。让作家任务输出一次格式错误的 JSON；验证 CrewAI 在跟踪中的重试行为。
6. 阅读 CrewAI 的文档介绍。将玩具移植到真正的 `crewai` API。标准库版本跳过了哪些保证？
7. 将 AgentOps 或 Langfuse（第 24 课）连接到真实运行。你在标准库版本中错过了哪些跟踪？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| Agent | "Persona" | 角色 + 目标 + 背景故事 + 工具 |
| Task | "工作单元" | 描述 + 期望输出 +  assignee + 可选结构化输出 |
| Crew | "智能体团队" | Agent + Task + Process 的容器 |
| Process | "执行策略" | Sequential / Hierarchical / Consensus（计划中） |
| Flow | "确定性工作流" | 事件驱动、代码所有、可测试 |
| Backstory | "Persona 提示词" | 为 Agent 塑造语气和判断 |
| `@tool` | "函数工具" | 将函数转换为 Agent 可调用工具的装饰器 |
| `BaseTool` | "类工具" | 带参数模式、重试、异步支持的类工具 |
| Entity memory | "每个实体的facts" | 作用域限定为客户 / 账户 / 问题 的内存 |
| Long-term memory | "跨运行内存" | 在 kickoff 之间存活的向量支持内存 |
| Contextual memory | "即时检索" | 在 Agent 需要它的时刻拉取的内存 |
| Manager LLM | "路由智能体" | Hierarchical 流程中挑选下一个任务的额外 LLM |
| `expected_output` | "任务契约" | 告诉 Agent（和审计）返回什么形态的字符串 |

## 进一步阅读

- [CrewAI 文档介绍](https://docs.crewai.com/en/introduction)：概念和推荐的生产路径
- [CrewAI Flows 指南](https://docs.crewai.com/en/concepts/flows)：事件驱动形态、`@start`、`@listen`
- [CrewAI 工具参考](https://docs.crewai.com/en/concepts/tools)：`@tool`、`BaseTool`、内置工具包
- [CrewAI 内存](https://docs.crewai.com/en/concepts/memory)：短期、长期、实体、上下文
- [Anthropic，构建有效的智能体](https://www.anthropic.com/research/building-effective-agents)：多智能体何时有帮助何时没有
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview)：状态机替代方案