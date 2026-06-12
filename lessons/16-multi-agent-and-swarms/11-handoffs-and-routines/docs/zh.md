# 交接与Routine —— 无状态编排

> OpenAI 的 Swarm（2024 年 10 月）将多 Agent 编排精简为两个原语：**routine**（指令 + 工具作为系统提示）和 **handoff**（返回另一个 Agent 的工具）。没有状态机，没有分支 DSL —— LLM 通过调用正确的 handoff 工具来路由。OpenAI Agents SDK（2025 年 3 月）是其生产级继承者。Swarm 本身仍然是最清晰的概念参考 —— 它的全部源代码只有几百行。这个模式之所以病毒式传播，是因为 API surface 大致是"agent = prompt + tools; handoff = 返回 agent 的函数"。局限性：无状态，所以记忆是调用者的问题。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置条件：** 阶段 16 · 04（原始模型）
**时间：** 约 60 分钟

## 问题

每个多 Agent 框架都希望你学习它的 DSL：LangGraph 的节点和边、CrewAI 的团队和任务、AutoGen 的 GroupChat 和管理器。DSL 是真实的抽象，但它们让这东西感觉比实际需要的更重。

Swarm 向相反的方向推进：利用模型已有的工具调用能力。交接变成工具调用。编排器是当前持有对话的 Agent。状态机隐含在 Agent 的系统提示中。

## 概念

### 两个原语

**Routine。** 定义 Agent 角色和可用工具的系统提示。可以把它想象成一组作用域化指令："你是一个分诊 Agent；如果用户问关于退款的事，就交接给退款 Agent。"

**Handoff。** Agent 可以调用的工具，返回一个新的 Agent 对象。Swarm 运行时检测到 Agent 返回值并在下一轮切换活动 Agent。

这就是全部的抽象。

```
def transfer_to_refunds():
    return refund_agent  # Swarm 检测到 Agent 返回 → 切换活动 Agent

triage_agent = Agent(
    name="triage",
    instructions="Route the user to the right specialist.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

分诊 Agent 的系统提示使其根据用户消息选择正确的交接。LLM 的工具调用完成路由。

### 为什么它病毒式传播

- **API 很小。** 两个概念要学。
- **利用模型已有的能力。** 工具调用在提供商之间已经是生产级的。
- **没有状态机负担。** 你不需要描述图；Agent 的提示描述了它们会交接给谁。

### 无状态的权衡

Swarm 在运行之间是明确无状态的。框架在运行期间保持消息历史，但不持久化任何东西。记忆、连续性、长期运行的任务 —— 都是调用者的问题。

在生产环境中（OpenAI Agents SDK，2025 年 3 月），这是主要变化之一：SDK 在保持 handoff 原语的同时添加了内置的会话管理、护栏和追踪。

### 何时适合使用 Swarm/handoff

- **分诊模式。** 前线 Agent 将用户路由到专家。
- **基于技能的交接。** "如果任务需要代码，调用编码者；如果需要研究，调用研究者。"
- **短期有界对话。** 客户支持、FAQ 到工单、简单工作流。

### Swarm 在哪里挣扎

- **具有共享内存的长期会话。** 交接将对话状态重置为新 Agent 的提示加上历史。没有调用者管理的内存，Agent 之间就没有持久状态。
- **并行执行。** 交接是一次一个 —— 活动 Agent 切换。并行需要调用者编排多个 Swarm 运行。
- **审计和回放。** 无状态运行很难精确回放；LLM 的交接选择不是确定性的。

### OpenAI Agents SDK（2025 年 3 月）

生产级继承者添加了：

- **会话状态。** 跨运行的持久线程。
- **护栏。** 输入/输出验证钩子。
- **追踪。** 每次工具调用和交接都被记录。
- **交接过滤器。** 控制什么上下文在交接时传输。

handoff 原语得以保留；生产级人体工程学被添加到周围。

### Swarm vs GroupChat

两者都使用 LLM 驱动的路由，但它们在 **谁选择下一个** 上有所不同：

- GroupChat：选择器（函数或 LLM）从外部选择下一个发言者。
- Swarm：当前 Agent 通过调用 handoff 工具来选择其继任者。

Swarm 是"Agent 决定下一个是什么"；GroupChat 是"管理器决定下一个是什么"。Swarm 的决定存在于活动 Agent 的工具调用中；GroupChat 的决定存在于 `GroupChatManager` 中。

## 构建

`code/main.py` 从头实现 Swarm：一个 Agent 数据类、一个 handoff 机制（工具返回 Agent）和一个检测 Agent 切换的运行循环。

演示：一个分诊 Agent 路由到退款、销售或支持专家。每个专家都有自己的工具。运行循环打印每个交接。

运行：

```
python3 code/main.py
```

## 使用

`outputs/skill-handoff-designer.md` 为给定任务设计 handoff 拓扑：存在哪些 Agent、它们可以调用哪些交接、什么上下文被传输。

## 交付

检查清单：

- **交接日志记录。** 每次交接都写一个跟踪事件，包含源 Agent、目标 Agent、上下文快照。
- **上下文传输规则。** 决定什么在交接时移动：完整历史（昂贵）、最后 N 条消息，或摘要。
- **交接护栏。** 交接到具有不同工具权限的专家时必须进行身份验证 —— 否则提示注入可以强制进行不必要的交接。
- **循环检测。** 两个 Agent 来回交接是一种常见失败；用简单的最后 K 环检查来检测。
- **回退 Agent。** 如果交接目标不存在，则回退到安全的默认值。

## 练习

1. 运行 `code/main.py`，分诊到退款 Agent。确认第二圈的活动 Agent 是退款。
2. 添加一个循环检测规则：如果两个相同的 Agent 连续交接 3 次，强制退出。设计回退方案。
3. 阅读 OpenAI Agents SDK 文档中关于 handoff 过滤器的内容。实现一个"交接时摘要"版本：传出 Agent 在传入 Agent接管之前将上下文压缩为要点摘要。
4. 比较 Swarm handoff 与 GroupChatManager 选择器。哪个模式使提示注入更严重，为什么？
5. 阅读 Swarm cookbook（https://developers.openai.com/cookbook/examples/orchestrating_agents）。识别 Swarm 做出的一个明确设计决策，OpenAI Agents SDK 改变了或保留了它。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| Routine | "Agent 提示" | 系统提示 + 工具列表。定义角色和可用的交接。 |
| Handoff | "转移到另一个 Agent" | 活动 Agent 可以调用的工具，返回一个新的 Agent。运行时切换活动 Agent。 |
| 无状态 | "运行之间没有记忆" | Swarm 不持久化任何东西；记忆是调用者的责任。 |
| 活动 Agent | "现在谁在说话" | 当前持有对话的 Agent。交接会改变这个。 |
| 上下文传输 | "交接时什么移动" | 传入 Agent 看到什么历史的策略：完整、最后 N 条，或摘要。 |
| 交接循环 | "Agent 乒乓球" | 两个 Agent 持续来回交接的失败模式。 |
| OpenAI Agents SDK | "生产级 Swarm" | 2025 年 3 月的继承者；在 handoff 原语之上添加了会话、护栏、追踪。 |
| 交接过滤器 | "转移时的门控" | SDK 特性，在交接边界检查和修改上下文。 |

## 延伸阅读

- [OpenAI cookbook —— 编排 Agent：Routine 和 Handoff](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 参考阐述
- [OpenAI Swarm 仓库](https://github.com/openai/swarm) — 原始实现，作为概念参考保留
- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) — 生产级继承者，包含会话和追踪
- [Anthropic 关于 Claude 中 handoff 的笔记](https://docs.anthropic.com/en/docs/claude-code) — Claude Code subagent 如何通过 `Task` 使用类似 handoff 的模式