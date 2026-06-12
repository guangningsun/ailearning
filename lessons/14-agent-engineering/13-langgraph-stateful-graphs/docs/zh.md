# LangGraph：有状态图与持久化执行

> LangGraph 是 2026 年低级有状态编排的参考实现。智能体是一个状态机；节点是函数；边是转换；状态是不可变的，且每步后都会 checkpoint。从任何失败点精确恢复到哪里中断了。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置条件：** 阶段 14 · 01（智能体循环）、阶段 14 · 12（工作流模式）
**时间：** 约 75 分钟

## 学习目标

- 描述 LangGraph 的核心模型：带有不可变状态、函数节点、条件边和每步 checkpoint 的状态机。
- 说出文档强调的四项能力：持久化执行、流式输出、人在回路、全面记忆。
- 解释 LangGraph 支持的三种编排拓扑：监督者、对等（swarm）、层级（嵌套子图）。
- 用标准库实现一个带有不可变状态、条件边和 checkpoint/恢复循环的状态图。

## 问题

智能体和工作流共享一个问题：当一个 40 步的运行在第 38 步失败时，你想从第 38 步恢复，而不是从头开始。二流的状态模型让运维人员在假设全新运行的库上费力编写重试逻辑。

LangGraph 的设计回答：状态是一等公民的 typed 对象，变更显式化，checkpoint 在每个节点后持久化。恢复只需一个 `load_state(session_id)` 调用。

## 概念

### 图

图由以下定义：

- **状态类型。** 每个节点读取和变更的 typed dict（或 Pydantic 模型）。
- **节点。** 纯函数 `(state) -> state_update`。返回后更新被合并到状态中。
- **边。** 节点之间的条件或直接转换。
- **入口和出口。** `START` 和 `END` 哨兵节点标记边界。

示例：一个带有 `classify`、`refund`、`bug`、`sales`、`done` 节点的智能体 —— 作为图的路由工作流。

### 持久化执行

每个节点返回后，运行时将状态序列化并写入 checkpoint 存储（SQLite、Postgres、Redis、自定义）。在第 N 步失败时，运行时可以 `resume(session_id)` 并从第 N+1 步精确恢复状态。

LangGraph 文档明确强调了这样做很重要的生产用户：Klarna、Uber、J.P. Morgan。重点不是图的形状；而是图的形状加上 checkpointing 使恢复成本低廉。

### 流式输出

每个节点都可以产生部分输出。图将每节点增量事件流式传递给调用者，以便 UI 在图运行时更新。

### 人在回路

在节点之间检查和修改状态。实现方式：在关键节点前暂停，将状态呈现给人工，接受修改，恢复运行。checkpoint 存储使这变得容易，因为状态已经是序列化的。

### 记忆

短期（单次运行内 —— 状态中的对话历史）和长期（跨运行 —— 通过 checkpoint 存储持久化加上独立的长期存储）。LangGraph 通过工具与外部记忆系统（Mem0、自定义）集成。

### 三种拓扑

1. **监督者。** 中央路由器 LLM 分发给专家子智能体。在 `langgraph-supervisor` 中使用 `create_supervisor()`（不过 LangChain 团队在 2026 年推荐通过工具调用直接实现，以获得更好的上下文控制）。
2. **Swarm / 对等。** 智能体通过共享工具表面直接交接。没有中央路由器。
3. **层级。** 监督者管理子监督者，作为嵌套子图实现。

### 这种模式的常见错误

- **Checkpoint 太小。** 只 checkpoint 对话轮次会让工具状态和记忆写入无法恢复。必须序列化完整状态。
- **非确定性节点。** 恢复假设节点输入产生相同的状态更新。随机种子、墙钟时间、外部 API 必须被捕获。
- **过度使用条件边。** 每个边都是条件的图是一个无法推理的状态机。优先使用带有偶尔分支的线性链。

## 构建

`code/main.py` 实现了一个标准库有状态图：

- `State` —— 带 `messages`、`step`、`route`、`output`、`human_approval` 的 typed dict。
- `Node` —— 接受状态并返回更新字典的可调用对象。
- `StateGraph` —— 节点 + 边 + 条件边 + 运行 + 恢复。
- `SQLiteCheckpointer`（内存中的假实现） —— 每次节点后序列化状态；`load(session_id)` 恢复。
- 一个演示图：classify -> branch(refund / bug / sales) -> human gate -> send。

运行：

```
python3 code/main.py
```

轨迹显示第一次运行在人工门控处失败、持久化，然后恢复产生最终输出。

## 使用

- **LangGraph** —— 参考实现，生产就绪。使用 `create_react_agent`、`create_supervisor`，或构建你自己的图。
- **AutoGen v0.4**（第 14 课）—— 用于高并发场景的参与者模型替代方案。
- **Claude Agent SDK**（第 17 课） —— 带有内置会话存储的托管工具集。
- **自定义** —— 当你需要精确控制状态形状或 checkpoint 存储后端时。

## 交付

`outputs/skill-state-graph.md` 在任何目标运行时生成 LangGraph 形态的状态图，并接入了 checkpointing 和恢复。

## 练习

1. 当分类置信度低于阈值时，添加从 `classify` 到 `end` 的条件边。在人工手动设置 `route` 后恢复运行。
2. 将类 SQLite 假实现换成真实的 SQLite checkpointer。测量每步序列化开销。
3. 实现并行边：两个节点并发运行，通过自定义 reducer 合并。不可变状态在这里带来了什么？
4. 阅读 `langgraph-supervisor` 参考。将示例移植到 `create_supervisor`。比较轨迹形状。
5. 添加流式输出：每个节点在运行时产生部分状态。在增量到达时打印它们。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| 状态图 | "作为状态机的智能体" | Typed 状态 + 节点 + 边 + reducer |
| Checkpointer | "持久化后端" | 每次节点后序列化状态；启用恢复 |
| Reducer | "状态合并器" | 将当前状态与节点更新组合的函数 |
| 条件边 | "分支" | 由状态的函数选择的边 |
| 子图 | "嵌套图" | 作为另一个图内的节点使用的图 |
| 持久化执行 | "从失败恢复" | 用精确状态在上一次成功的节点处重启 |
| 监督者 | "路由器 LLM" | 专家子智能体的中央分发器 |
| Swarm | "P2P 智能体" | 智能体通过共享工具交接；无中央路由器 |

## 延伸阅读

- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) —— 参考文档
- [langgraph-supervisor 参考](https://reference.langchain.com/python/langgraph/supervisor/) —— 监督者模式 API
- [AutoGen v0.4，Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) —— 参与者模型替代方案
- [Claude Agent SDK 概述](https://platform.claude.com/docs/en/agent-sdk/overview) —— 会话存储和子智能体