# 构建 MCP 客户端 — 服务发现、调用与会话管理

> 大部分 MCP 内容都在教服务端教程，对客户端则一带而过。客户端代码才是真正复杂的编排工作所在：进程派生、能力协商、多服务器工具列表合并、采样回调、重连以及命名空间冲突解决。本课构建一个多服务器客户端，将三个不同的 MCP 服务器统一到一个扁平的工具命名空间供模型使用。

**类型：** 构建
**语言：** Python（标准库，多服务器 MCP 客户端）
**前置条件：** 阶段 13 · 07（构建 MCP 服务器）
**时间：** 约 75 分钟

## 学习目标

- 将 MCP 服务器作为子进程派生，完成 `initialize`，并发送 `notifications/initialized`。
- 维护每个服务器的会话状态（能力声明、工具列表、最近收到的通知 ID）。
- 将多个服务器的工具列表合并到一个命名空间中并处理冲突。
- 将工具调用路由到所属服务器并重新组装响应。

## 问题

真正的 agent 宿主（Claude Desktop、Cursor、Goose、Gemini CLI）会同时加载多个 MCP 服务器。用户可能同时运行一个文件系统服务器、一个 Postgres 服务器和一个 GitHub 服务器。客户端的工作是：

1. 派生每个服务器。
2. 独立地与每个服务器握手。
3. 对每个服务器调用 `tools/list` 并扁平化结果。
4. 当模型发出 `notes_search` 时，在合并的命名空间中查找并路由到正确的服务器。
5. 处理来自任何服务器的通知（`tools/list_changed`）而不阻塞。
6. 在传输失败时重连。

手写所有这些逻辑，才是"玩具"与"可服务"之间的分水岭。官方 SDK 有封装，但心理模型必须是你自己的。

## 概念

### 子进程派生

`subprocess.Popen`，配合 `stdin=PIPE, stdout=PIPE, stderr=PIPE`。设置 `bufsize=1`，使用文本模式逐行读取。每个服务器对应一个进程；客户端为每个服务器持有一个 `Popen` 句柄。

### 每服务器会话状态

每个服务器的 `Session` 对象包含：

- `process` — Popen 句柄。
- `capabilities` — 服务器在 `initialize` 时声明的能力。
- `tools` — 最近一次 `tools/list` 的结果。
- `pending` — 请求 ID 到等待响应的 promise/future 的映射。

请求本质上是异步的；发往服务器 A 的 `tools/call` 不会阻塞服务器 B 正在进行的调用。使用线程加队列，或者 asyncio。

### 合并命名空间

当客户端看到聚合的工具列表时，名称可能会冲突。两个服务器可能都暴露了 `search`。客户端有三种选择：

1. **按服务器名称加前缀。** `notes/search`、`files/search`。清晰但不够优雅。
2. **静默优先。** 后来的服务器的 `search` 覆盖先来的。有风险；会隐藏冲突。
3. **冲突拒绝。** 拒绝加载第二个服务器；通知用户。对安全敏感的主机最安全。

Claude Desktop 使用前缀方式。Cursor 使用带明确错误的冲突拒绝。VS Code MCP 也采用前缀方式。

### 路由

合并后，调度表将 `tool_name -> session` 映射。模型按名称发出调用；客户端找到对应的 session，并向该服务器 stdin 写入 `tools/call` 消息，然后等待响应。

### 采样回调

如果服务器在 `initialize` 时声明了 `sampling` 能力，它可能发送 `sampling/createMessage` 请求客户端运行其 LLM。客户端必须：

1. 在该示例解析之前阻塞对该服务器的进一步请求，或者如果其实现支持并发则进行流水线化。
2. 调用其 LLM 提供者。
3. 将响应发送回服务器。

第 11 课涵盖了采样的端到端内容。本课将其存根化以保证完整性。

### 通知处理

`notifications/tools/list_changed` 意味着重新调用 `tools/list`。`notifications/resources/updated` 意味着重新读取正在使用的资源。通知不会产生响应——不要尝试确认它们。

一个常见的客户端 bug：在 `tools/call` 阻塞读取循环时，通知已经出现在流中。使用后台读取线程将每条消息推入队列；主线程出队并分发。

### 重连

传输可能失败：服务器崩溃、操作系统杀死进程、stdio 管道断裂。客户端检测到 stdout 上的 EOF 并将会话视为已终止。选项：

- 静默重启服务器并重新握手。适用于纯只读服务器。
- 向用户呈现失败。对有用户可见会话的有状态服务器适用。

阶段 13 · 09 涵盖了 Streamable HTTP 的重连语义；stdio 更简单。

### 保活与会话 ID

Streamable HTTP 使用 `Mcp-Session-Id` 头。Stdio 没有会话 ID——进程标识就是会话。保活 ping 是可选的；stdio 管道在非活动状态下不会断开。

## 使用它

`code/main.py` 将三个模拟的 MCP 服务器作为子进程派生，与每个服务器握手，合并它们的工具列表，并将工具调用路由到正确的服务器。"服务器"实际上是其他运行 toy 响应器的 Python 进程（没有真正的 LLM）。运行它来查看：

- 三次初始化，每次都有各自的能力集。
- 三次 `tools/list` 结果合并为一个 7 工具的命名空间。
- 基于工具名称的路由决策。
- 通过命名空间前缀化防止的冲突。

需要关注的内容：

- `Session` 数据类干净地保存每服务器状态。
- 后台读取线程在 stdout 上逐行出队而不阻塞主线程。
- 调度表是一个简单的 `dict[str, Session]`。
- 冲突处理是显式的：当两个服务器声明相同名称时，后来的服务器会被加前缀重命名。

## 交付它

本课产出 `outputs/skill-mcp-client-harness.md`。给定一个 MCP 服务器的声明性列表（名称、命令、参数），该 skill 生成一个 harness 来派生它们、合并工具列表，并提供一个带有冲突解决的路由函数。

## 练习

1. 运行 `code/main.py` 并观察服务器派生日志。用 SIGTERM 终止其中一个模拟服务器进程，观察客户端如何检测到 EOF 并将该会话标记为已终止。

2. 实现命名空间前缀化。当两个服务器都暴露 `search` 时，将第二个重命名为 `<server>/search`。更新调度表并验证工具调用正确路由。

3. 添加连接池风格的重试退避：连续失败时指数退避，上限 30 秒，三次失败后向用户发出通知。

4. 设计一个支持 100 个并发 MCP 服务器的客户端。什么数据结构可以替代简单的调度字典？（提示：用于前缀命名空间化的 trie，加上每服务器工具计数的指标。）

5. 将客户端移植到官方 MCP Python SDK。SDK 包装了 `stdio_client` 和 `ClientSession`。代码应该从约 200 行缩减到约 40 行，同时保留多服务器路由。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| MCP 客户端 | "Agent 宿主" | 派生服务器并编排工具调用的进程 |
| 会话 | "每服务器状态" | 能力声明、工具列表和待处理请求的记账 |
| 合并命名空间 | "一个工具列表" | 所有活动服务器的工具名称的扁平集合 |
| 命名空间冲突 | "两个服务器同名工具" | 客户端必须为重复项加前缀、拒绝或采用优先策略 |
| 路由 | "谁处理这个调用？" | 从工具名称到所属服务器的调度 |
| 后台读取器 | "非阻塞 stdout" | 将服务器 stdout 排入队列的线程或任务 |
| 采样回调 | "LLM 即服务" | 客户端对服务器 `sampling/createMessage` 的处理程序 |
| `notifications/*_changed` | "原语发生变化" | 信号：客户端必须重新发现或重新读取 |
| 重连策略 | "服务器终止时" | 传输失败时的重启语义 |
| Stdio 会话 | "进程 = 会话" | 无会话 ID；子进程生命周期即为会话 |

## 延伸阅读

- [Model Context Protocol — 客户端规范](https://modelcontextprotocol.io/specification/2025-11-25/client) — 规范的客户端行为
- [MCP — 快速入门客户端指南](https://modelcontextprotocol.io/quickstart/client) — 使用 Python SDK 的 hello-world 客户端教程
- [MCP Python SDK — 客户端模块](https://github.com/modelcontextprotocol/python-sdk) — 参考 `ClientSession` 和 `stdio_client`
- [MCP TypeScript SDK — 客户端](https://github.com/modelcontextprotocol/typescript-sdk) — TypeScript 并行实现
- [VS Code — 扩展中的 MCP](https://code.visualstudio.com/api/extension-guides/ai/mcp) — VS Code 如何在单个编辑器宿主中复用多个 MCP 服务器
