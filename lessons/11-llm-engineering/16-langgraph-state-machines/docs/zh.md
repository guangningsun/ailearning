# LangGraph — 智能体的状态机

> 手写的 ReAct 循环是一个 `while True`。用 LangGraph 写的 ReAct 循环是一个你可以保存检查点、中断、分支和时间回溯的图。智能体没有变。周围的工具链变了。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 11 · 09（函数调用）、阶段 11 · 14（模型上下文协议）
**时间：** 约 75 分钟

## 问题

你发布了一个函数调用智能体。它工作了三个回合，然后出了问题：模型尝试一个返回 500 的工具，用户在任务中途改变主意，或者智能体决定在没有人工签名的情况下退款。`while True:` 循环没有钩子。你无法暂停它，无法回滚它，也无法分支到"如果模型选择了另一个工具会怎样"。一旦你将其发布超过演示阶段，智能体就变成了一个黑箱——要么工作了，要么没有。

下一步一旦你看到就显而易见了。智能体已经是一个状态机——系统提示词加消息历史加待处理工具调用加下一个动作。将状态机显式化：节点代表"模型思考"、"工具运行"、"人工审批"，边代表它们之间的条件转换。一旦图显式化，工具链免费获得四样东西：检查点（在步骤之间保存状态）、中断（暂停等待人工）、流式处理（流式传输 token 和中间事件）和时间回溯（回滚到先前状态并尝试不同分支）。

LangGraph 就是提供这种抽象的库。它不是 LangChain 意义上的智能体框架（"这里有一个 AgentExecutor，祝你好运"）。它是一个图运行时，具有一等状态、一等持久化和一等中断。智能体循环是你画出来的东西，而不是手写的东西。

## 概念

![LangGraph StateGraph：节点、边和检查点保存器](../assets/langgraph-stategraph.svg)

`StateGraph` 有三个组成部分。

1. **状态。** 一个流经图的类型化字典（TypedDict 或 Pydantic 模型）。每个节点接收完整状态并返回部分更新，LangGraph 使用每个字段的*归约器*合并——对于应该累积的列表使用 `operator.add`，默认是覆盖。
2. **节点。** Python 函数 `state -> partial_state`。每个节点是一个离散步骤："调用模型"、"运行工具"、"摘要"。
3. **边。** 节点之间的转换。静态边去一个地方。条件边接受路由函数 `state -> next_node_name`，以便图可以在模型输出上分支。

编译图。Compile 绑定拓扑，附加检查点保存器（可选但对生产至关重要），返回一个可运行对象。你用初始状态和一个 `thread_id` 调用它。执行的每个步骤都保存一个以 `(thread_id, checkpoint_id)` 为键的检查点。

### 四个超能力

**检查点。** 每次节点转换都将新状态写入存储（测试用内存，生产用 Postgres/Redis/SQLite）。通过使用相同的 `thread_id` 再次调用图来恢复。图从暂停的地方继续。

**中断。** 用 `interrupt_before=["human_review"]` 标记节点，执行在该节点运行之前停止。状态保留。你的 API 向用户响应"等待审批"。后续对相同 `thread_id` 的请求附带 `Command(resume=...)` 恢复执行。

**流式处理。** `graph.stream(state, mode="updates")` 在发生时产生状态增量。`mode="messages"` 在模型节点内部流式传输 LLM token。`mode="values"` 产生完整快照。你选择向 UI 提供什么。

**时间回溯。** `graph.get_state_history(thread_id)` 返回完整检查点日志。将任意先前的 `checkpoint_id` 传递给 `graph.invoke`，你就会从那个点分叉。非常适合调试（"如果模型选择了工具 B 会怎样？"）和重放生产轨迹的回归测试。

### 归约器才是关键

每个状态字段都有一个归约器。大多数默认值都没问题——新值覆盖旧值。但消息列表需要 `operator.add`，以便新消息追加而不是替换。并行边通过归约器合并它们的更新。如果两个节点都更新了 `messages` 而你忘记了 `Annotated[list, add_messages]`，第二个节点会静默获胜，你丢失了半个回合。归约器是库中唯一微妙的东西；用它对了，剩下的就自然组合了。

### 四个节点的 ReAct 图

一个生产级 ReAct 智能体是四个节点和两条边：

1. `agent` — 用当前消息历史调用 LLM。返回助手消息（可能包含 tool_calls）。
2. `tools` — 执行最后一个助手消息中的任何 tool_calls，将工具结果追加为工具消息。
3. 从 `agent` 出发的条件边——如果最后一条消息有 tool_calls 则路由到 `tools`，否则到 `END`。
4. 从 `tools` 返回 `agent` 的静态边。

就这样。你获得了完整的 ReAct 循环（思考 → 行动 → 观察 → 思考 → …），包含检查点、中断和流式处理，大约 40 行代码。

### StateGraph vs Send（扇出）

`Send(node_name, state)` 让一个节点分派并行子图。例如：智能体决定同时查询三个检索器。每个 `Send` 生成目标节点的并行执行；它们的输出通过状态归约器合并。这就是 LangGraph 表达 orchestrator-workers 模式而不需要线程原语的方式。

### 子图

一个编译后的图可以作为另一个图中的节点。外图看到单个节点；内图有自己的状态和自己的检查点。这就是团队构建 supervisor-worker 智能体的方式：supervisor 图将用户意图路由到每个域的 worker 子图。

## 构建

### 步骤 1：状态和节点

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` 是使消息列表累积而不是覆盖的归约器。忘记它是最常见的 LangGraph bug。

### 步骤 2：用线程运行

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

每次更新都是一个字典 `{node_name: state_delta}`。你的前端可以将这些流式传输到 UI，让用户看到"智能体正在思考… 正在调用 search_web… 获得结果… 正在回答。"

### 步骤 3：添加人工介入中断

标记一个节点，使执行在其运行之前暂停。

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # 在每个工具调用之前暂停
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] 已设置。检查提议的工具调用。
# 如果批准：
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# 如果拒绝：写入拒绝消息并恢复
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

状态、检查点和线程在中断期间都保留。除执行期间外，没有任何内容在内存中。

### 步骤 4：用于调试的时间回溯

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# 从先前的检查点分叉
target = history[3].config  # 回退三步
for event in app.stream(None, target, stream_mode="values"):
    pass  # 从那一点向前重放
```

传入 `None` 作为输入从给定检查点重放；传入一个值在恢复之前将其作为更新追加到该检查点的状态。这就是你重现不良智能体运行而不必重跑整个对话的方式。

### 步骤 5：将检查点保存器换成生产级

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

提供了 SQLite、Redis 和 Postgres。`MemorySaver` 用于测试。任何需要跨重启持久化的东西都需要一个真实存储。

## 技能

> 你将智能体构建为图，而不是 `while True` 循环。

在你求助于 LangGraph 之前，做一个 60 秒的设计：

1. **命名节点。** 每个离散决策或副作用动作都是一个节点。"智能体思考"、"工具运行"、"审核者审批"、"响应流式传输"。如果你列不出来，任务还不是智能体形状。
2. **声明状态。** 最小的 TypedDict，每个列表字段都有归约器。不要把所有东西塞进 `messages`；将任务特定字段（一个工作中的 `plan`、一个 `budget` 计数器、一个 `retrieved_docs` 列表）提升到顶层。
3. **画边。** 静态的，除非下一步取决于模型输出。每条条件边需要一个带有命名分支的路由函数。
4. **提前选择检查点保存器。** 测试用 `MemorySaver`，其他用 Postgres/Redis/SQLite。不要不带检查点就发货——没有检查点意味着无法恢复、无法中断、无法时间回溯。
5. **在工具运行之前决定中断，而不是之后。** 审批放在副作用节点的入边上，这样你可以在伤害发生前取消；验证放在模型的出边上，这样你可以廉价地拒绝错误调用。
6. **默认流式处理。** UI 用 `mode="updates"`，模型节点内部的 token 级流式传输用 `mode="messages"`，评估期间的全快照用 `mode="values"`。

拒绝发货一个没有检查点保存器的 LangGraph 智能体。拒绝发货一个在副作用*之后*中断的。拒绝发货一个 `messages` 字段没有 `add_messages` 作为其归约器的。

## 练习

1. **简单。** 用计算器工具和网络搜索工具实现上面的四节点 ReAct 图。验证 `list(app.get_state_history(config))` 在两回合对话中返回至少四个检查点。
2. **中等。** 添加一个在 `agent` 之前运行的 `planner` 节点，并将结构化 `plan: list[str]` 写入状态。让 `agent` 标记计划步骤为已完成。如果 `plan` 在检查点恢复中丢失则测试失败（错误的归约器）。
3. **困难。** 构建一个 supervisor 图，使用 `Send` 在三个子图（`researcher`、`writer`、`reviewer`）之间路由。每个子图有自己的状态和检查点。在外图上添加 `interrupt_before=["writer"]`，以便人工可以审批研究简报。确认从先前检查点的时间回溯只重运行分叉的分支。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| StateGraph | "LangGraph 图" | 在编译前添加节点和边的构建器对象。 |
| 归约器（Reducer） | "字段如何合并" | 当节点返回该字段的更新时应用的函数 `(old, new) -> merged`；默认是覆盖，`add_messages` 追加。 |
| 线程（Thread） | "对话 ID" | 一个 `thread_id` 字符串，作用域为一次会话的所有检查点。 |
| 检查点（Checkpoint） | "暂停的状态" | 在节点转换后持久化的完整图状态快照，以 `(thread_id, checkpoint_id)` 为键。 |
| 中断（Interrupt） | "暂停等待人工" | `interrupt_before` / `interrupt_after` 在节点边界停止执行；用 `Command(resume=...)` 恢复。 |
| 时间回溯（Time-travel） | "从先前步骤分叉" | `graph.invoke(None, config_with_old_checkpoint_id)` 从该检查点向前重放。 |
| Send | "并行子图分派" | 一个节点可以返回的构造函数，用于生成目标节点的 N 个并行执行。 |
| 子图（Subgraph） | "作为节点的编译图" | 用作另一个图中的节点的编译后的 StateGraph；保留自己的状态作用域。 |

## 延伸阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — StateGraph、归约器、检查点保存器和中断的规范参考。
- [LangGraph 概念：状态、归约器、检查点保存器](https://langchain-ai.github.io/langgraph/concepts/low_level/) — 本课使用的心智模型，直接来自源头。
- [LangGraph 持久化和检查点](https://langchain-ai.github.io/langgraph/concepts/persistence/) — Postgres/SQLite/Redis 存储、检查点命名空间和线程 ID 的详细信息。
- [LangGraph 人工介入](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — `interrupt_before`、`interrupt_after`、`Command(resume=...)` 和编辑状态模式。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — 每个 LangGraph 智能体实现的模式；阅读它了解推理轨迹原理。
- [Anthropic — 构建有效的智能体（2024 年 12 月）](https://www.anthropic.com/research/building-effective-agents) — 哪种图形状（链、路由器、orchestrator-workers、evaluator-optimizer）应优先选择以及何时。
- 阶段 11 · 09（函数调用）—— 每个 LangGraph 智能体节点复用的工具调用原语。
- 阶段 11 · 14（模型上下文协议）—— 通过 MCP 适配器插入 LangGraph `ToolNode` 的外部工具发现。
- 阶段 11 · 17（智能体框架权衡）—— 何时选择 LangGraph 而不是 CrewAI、AutoGen 或 Agno。
