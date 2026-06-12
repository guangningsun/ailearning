# AutoGen v0.4：参与者模型与智能体框架

> AutoGen v0.4（Microsoft Research，2025 年 1 月）围绕参与者模型重新设计了智能体编排。异步消息交换、事件驱动智能体、故障隔离、自然并发。该框架目前处于维护模式，而 Microsoft Agent Framework（2025 年 10 月公开预览）成为后继者。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置条件：** 阶段 14 · 01（智能体循环）、阶段 14 · 12（工作流模式）
**时间：** 约 75 分钟

## 学习目标

- 描述参与者模型：智能体作为参与者，消息作为唯一的 IPC，每个参与者故障隔离。
- 说出 AutoGen v0.4 的三层 API —— Core、AgentChat、Extensions —— 以及各自的用途。
- 解释为什么解耦消息传递与处理带来了故障隔离和自然并发。
- 用 Python 标准库实现一个参与者运行时，并将双智能体代码审查流程移植到它上面。

## 问题

大多数智能体框架是同步的：一个智能体产生，一个智能体消费，在调用栈中进行。故障会崩溃栈。并发是勉强加上的。分布式需要重写。

AutoGen v0.4 的答案是：参与者模型。每个智能体是一个拥有私有收件箱的参与者。消息是唯一的交互方式。运行时将传递与处理解耦。故障隔离到一个参与者。并发是原生的。分布式只是不同的传输层。

## 概念

### 参与者

一个参与者拥有：

- 一个私有状态（外部从不直接触碰）。
- 一个收件箱（消息队列）。
- 一个处理器：`receive(message) -> effects`，其中 effects 可以是"回复"、"发送给其他参与者"、"生成新参与者"、"更新状态"、"停止自己"。

两个参与者不能共享内存。它们只能发送消息。

### AutoGen v0.4 的三层 API

1. **Core。** 低级参与者框架。`AgentRuntime`、`Agent`、`Message`、`Topic`。异步消息交换、事件驱动。
2. **AgentChat。** 面向任务的高级 API（v0.2 的 ConversableAgent 的替代品）。`AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **Extensions。** 集成 —— OpenAI、Anthropic、Azure、工具、记忆。

### 解耦为何重要

在 v0.2 模型中，调用 `agent_a.chat(agent_b)` 会同步阻塞 agent_a 直到 agent_b 返回。在 v0.4 中，`send(agent_b, msg)` 将消息放入 agent_b 的收件箱并立即返回。运行时稍后传递。三个后果：

- **故障隔离。** 参与者 B 崩溃不会崩溃参与者 A —— 运行时在 B 的处理器中捕获故障并决定如何处理（日志、重试、死信）。
- **自然并发。** 同时有许多消息在飞；参与者并发处理它们的收件箱。
- **可分布式。** 收件箱 + 传输是相同的抽象，无论参与者在进程内还是在另一台主机上。

### 拓扑

- **RoundRobinGroupChat。** 智能体按固定轮换顺序轮流。
- **SelectorGroupChat。** 选择器智能体根据对话上下文选择下一个发言者。
- **Magentic-One。** 用于网页浏览、代码执行、文件处理的参考多智能体团队。构建在 AgentChat 之上。

### 可观测性

OpenTelemetry 支持内置。每个消息发出一个 span；工具调用按照 2026 年 OTel GenAI 语义约定携带 `gen_ai.*` 属性（第 23 课）。

### 状态：维护模式

2026 年初：AutoGen v0.7.x 稳定，适合研究和原型设计。Microsoft 已将积极开发转移到 Microsoft Agent Framework（2025 年 10 月 1 日公开预览；1.0 GA 目标是 2026 年 Q1 末）。AutoGen 模式可以平滑地向前移植 —— 参与者模型是持久的理念。

## 构建

`code/main.py` 实现了一个标准库参与者运行时：

- `Message` —— 带 `sender`、`recipient`、`topic`、`body` 的 typed payload。
- `Actor` —— 带 `receive(message, runtime)` 的抽象。
- `Runtime` —— 带共享队列、传递、故障隔离的事件循环。
- 一个双参与者演示：`ReviewerAgent` 审查代码，`ChecklistAgent` 运行检查清单；它们交换消息直到达成共识。

运行：

```
python3 code/main.py
```

轨迹显示消息传递、一个参与者中的模拟故障不会崩溃另一个，以及在共享裁决上收敛。

## 使用

- **AutoGen v0.4/v0.7**（维护中）—— 适合研究、原型设计、多智能体模式。
- **Microsoft Agent Framework**（公开预览）—— 前向路径；在刷新后的 API 中使用相同的参与者模型理念。
- **LangGraph swarm 拓扑**（第 13 课） —— 通过共享工具交接的类似模式。
- **自定义参与者运行时** —— 当你需要特定传输（NATS、RabbitMQ、gRPC）时。

## 交付

`outputs/skill-actor-runtime.md` 为给定的多智能体任务生成一个最小参与者运行时加团队模板（RoundRobin 或 Selector）。

## 练习

1. 添加死信队列：当处理器抛出异常时，将失败的消息停放以供人工检查。在你的玩具例子中 DLQ 被触发的频率如何？
2. 实现 `SelectorGroupChat`：一个选择器参与者根据对话状态选择谁处理下一条消息。
3. 添加分布式传输：将进程内队列换成 JSON-over-HTTP 服务器，这样参与者可以在独立进程中运行。
4. 为每条消息连线一个 OTel span（或一个无操作替代品）。按照第 23 课发出 `gen_ai.agent.name`、`gen_ai.operation.name`。
5. 阅读 AutoGen v0.4 的架构文章。将你的玩具移植到真实的 `autogen_core` API。你跳过了什么在生产中很重要的东西？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| 参与者 | "智能体" | 私有状态 + 收件箱 + 处理器；无共享内存 |
| 消息 | "事件" | Typed payload；参与者之间唯一的交互方式 |
| 收件箱 | "邮箱" | 每个参与者的待处理消息队列 |
| 运行时 | "智能体宿主" | 路由消息和隔离故障的事件循环 |
| 主题 | "通道" | 参与者之间的命名发布-订阅路由 |
| 故障隔离 | "让它崩溃" | 一个参与者失败不会崩溃其他参与者 |
| RoundRobinGroupChat | "固定轮换团队" | 智能体按顺序轮流 |
| SelectorGroupChat | "上下文路由团队" | 选择器选择下一个发言者 |
| Magentic-One | "参考团队" | 用于网页 + 代码 + 文件的多智能体小队 |

## 延伸阅读

- [AutoGen v0.4，Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) —— 重新设计文章
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) —— 图形态的替代方案
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —— AutoGen 默认发出的 span