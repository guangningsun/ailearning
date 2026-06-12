# MCP 基础 — 原语、生命周期与 JSON-RPC 基础

> 在 MCP 之前，每一个集成都是定制方案。Model Context Protocol 由 Anthropic 于 2024 年 11 月首次发布，现由 Linux Foundation 的 Agentic AI Foundation 托管，将发现和调用标准化，这样任何客户端都可以与任何服务器通信。2025-11-25 规范定义了六个原语（服务器三个，客户端三个）、三阶段生命周期和 JSON-RPC 2.0 线格式。掌握这些，其余的 MCP 章节就是阅读练习。

**类型：** 学习型
**语言：** Python（标准库、JSON-RPC 解析器）
**前置条件：** 阶段 13 · 01 到 05（工具接口和函数调用）
**时间：** 约 45 分钟

## 学习目标

- 说出全部六个 MCP 原语（服务器端：tools、resources、prompts；客户端：roots、sampling、elicitation）并分别给出一个用例。
- 走查三阶段生命周期（初始化、运行、关闭），并说明每个阶段谁发送哪条消息。
- 解析并发出 JSON-RPC 2.0 请求、响应和通知信封。
- 解释 `initialize` 时能力协商是什么，以及没有它会出什么错。

## 问题

在 MCP 之前，每个使用工具的智能体都有自己的协议。Cursor 有一个 MCP 形状但不兼容的工具系统。Claude Desktop 附带了一个不同的系统。VS Code 的 Copilot 扩展是第三个。一支团队构建了一个"Postgres 查询"工具，却为三个不同的宿主 API 各写了一遍。复用它需要复制代码。

结果是大量定制集成像寒武纪大爆发一样涌现，生态系统速度触及天花板。

MCP 通过标准化线格式解决了这个问题。单个 MCP 服务器可在每个 MCP 客户端中工作：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf，截至 2026 年 4 月已有 300+ 个客户端。每月 1.1 亿次 SDK 下载。10,000+ 个公共服务器。Linux Foundation 于 2025 年 12 月在新成立的 Agentic AI Foundation 下接手托管。

本阶段使用的规范版本是 **2025-11-25**。它添加了异步任务（SEP-1686）、URL 模式 elicitation（SEP-1036）、带工具的采样（SEP-1577）、增量范围同意（SEP-835）和 OAuth 2.1 资源指示符语义。阶段 13 · 09 到 16 涵盖这些扩展。本课止步于基础部分。

## 概念

### 三个服务器原语

1. **工具。** 可调用动作。与阶段 13 · 01 相同的四步循环。
2. **资源。** 暴露的数据。只读的、通过 URI 可寻址的内容：`file:///path`、`db://query/...`、自定义 scheme。
3. **提示词。** 可复用模板。宿主 UI 中的斜杠命令；服务器提供模板，客户端填充参数。

### 三个客户端原语

4. **根目录。** 服务器允许访问的 URI 集合。客户端声明它们；服务器尊重它们。
5. **采样。** 服务器请求客户端的模型完成某个生成。启用无服务器端 API 密钥的服务器端智能体循环。
6. **Elicitation。** 服务器在飞行中途请求客户端用户的结构化输入。表单或 URL（SEP-1036）。

MCP 中的每个能力都恰好属于这六个原语之一。阶段 13 · 10 到 14 将深入介绍每个。

### 线格式：JSON-RPC 2.0

每条消息都是一个带有以下字段的 JSON 对象：

- 请求：`{jsonrpc: "2.0", id, method, params}`。
- 响应：`{jsonrpc: "2.0", id, result | error}`。
- 通知：`{jsonrpc: "2.0", method, params}` — 无 `id`，不期待响应。

基础规范有约 15 个方法，按原语分组。重要的有：

- `initialize` / `initialized`（握手）
- `tools/list`、`tools/call`
- `resources/list`、`resources/read`、`resources/subscribe`
- `prompts/list`、`prompts/get`
- `sampling/createMessage`（服务器到客户端）
- `notifications/tools/list_changed`、`notifications/resources/updated`、`notifications/progress`

### 三阶段生命周期

**阶段 1：初始化。**

客户端发送带有其 `capabilities` 和 `clientInfo` 的 `initialize`。服务器以其自己的 `capabilities`、`serverInfo` 和它所支持的规范版本回复。客户端在消化响应后发送 `notifications/initialized`。从此刻起，任一方都可以根据协商的能力发送请求。

**阶段 2：运行。**

双向。客户端调用 `tools/list` 发现，然后 `tools/call` 调用。服务器如果声明了该能力，可能发送 `sampling/createMessage`。服务器在其工具集发生变化时可能发送 `notifications/tools/list_changed`。客户端在用户更改根范围时可能发送 `notifications/roots/list_changed`。

**阶段 3：关闭。**

任一方关闭传输。MCP 中没有结构化关闭方法；传输（stdio 或 Streamable HTTP，阶段 13 · 09）携带连接结束信号。

### 能力协商

`initialize` 握手中的 `capabilities` 就是契约。服务器示例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

服务器声明它可以发出 `tools/list_changed` 通知并支持 `resources/subscribe`。客户端通过声明自己的来同意：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

如果客户端没有声明 `sampling`，服务器不得调用 `sampling/createMessage`。对称地：如果服务器没有声明 `resources.subscribe`，客户端不得尝试订阅。

这就是防止生态系统漂移的原因。不支持采样的客户端仍然是有效的 MCP 客户端；不调用 `sampling` 的服务器仍然是有效的 MCP 服务器。它们只是不一起使用那个特性。

### 结构化内容和错误形式

`tools/call` 返回一个 `content` 类型化块数组：`text`、`image`、`resource`。阶段 13 · 14 向该列表添加了 MCP Apps（`ui://` 交互式 UI）。

错误使用 JSON-RPC 错误码。规范定义 additions：`-32002` "Resource not found"、`-32603` "Internal error"，加上 MCP 特定的错误数据作为 `error.data`。

### 客户端能力 vs 工具调用细节

一个常见混淆：`capabilities.tools` 是客户端是否支持工具列表更改通知。客户端是否会调用特定工具是由其模型驱动的运行时选择，而不是能力标志。能力标志是规范级别的契约。模型的选择是正交的。

### 为什么是 JSON-RPC 而不是 REST？

JSON-RPC 2.0（2010）是一个轻量级双向协议。REST 是客户端发起的。MCP 需要服务器发起的消息（采样、通知），所以具有对称请求/响应形状的 JSON-RPC 是自然的选择。JSON-RPC 还可以干净地组合在 stdio 和 WebSocket/Streamable HTTP 上，而无需重新发明 HTTP 的请求形式。

## 实际使用

`code/main.py` 带有一个最小的 JSON-RPC 2.0 解析器和发射器，然后手动走查 `initialize` → `tools/list` → `tools/call` → `shutdown` 序列，打印每条消息。没有真实传输；只有消息形式。与延伸阅读中链接的规范比较以验证每个信封。

要看的地方：

- `initialize` 双向声明能力；响应有 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回一个 `tools` 数组；每个条目有 `name`、`description`、`inputSchema`。
- `tools/call` 使用 `params.name` 和 `params.arguments`。
- 响应 `content` 是一个 `{type, text}` 块数组。

## 交付物

本课产出 `outputs/skill-mcp-handshake-tracer.md`。给定 MCP 客户端-服务器交互的 pcap 风格转录，该技能为每条消息标注属于哪个原语、哪个生命周期阶段以及它依赖哪个能力。

## 练习

1. 运行 `code/main.py`。找出能力协商发生的行，并描述如果服务器没有声明 `tools.listChanged` 会发生什么变化。

2. 扩展解析器以处理 `notifications/progress`。消息形式：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在长时间运行的 `tools/call` 进行中发出它，并确认客户端处理程序会显示进度条。

3. 从头到尾阅读 MCP 2025-11-25 规范 — 整个文档大约 80 页。找出大多数服务器不需要的能力标志。提示：它与资源订阅有关。

4. 在纸上勾画一个假设的"定时任务"功能属于哪个原语。（提示：服务器希望客户端在计划时间调用它。今天六个原语都不适合。）MCP 的 2026 路线图为此有一个草案 SEP。

5. 解析 GitHub 上一个开放 MCP 服务器的一个会话日志。统计请求 vs 响应 vs 通知消息。计算生命周期 vs 运行阶段各占多少流量。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| MCP | "Model Context Protocol" | 模型到工具发现和调用的开放协议 |
| 服务器原语 | "服务器暴露什么" | tools（动作）、resources（数据）、prompts（模板） |
| 客户端原语 | "客户端让服务器使用什么" | roots（范围）、sampling（LLM 回调）、elicitation（用户输入） |
| JSON-RPC 2.0 | "线格式" | 对称的请求/响应/通知信封 |
| `initialize` 握手 | "能力协商" | 第一个消息对；服务器和客户端声明它们支持的功能 |
| `tools/list` | "发现" | 客户端向服务器请求其当前工具集 |
| `tools/call` | "调用" | 客户端请求服务器执行带有参数的工具 |
| `notifications/*_changed` | "变更事件" | 服务器告诉客户端其原语列表已更改 |
| 内容块 | "类型化结果" | 工具结果中 `{type: "text" \| "image" \| "resource" \| "ui_resource"}` |
| SEP | "规范演进提案" | 命名草案提案（例如 SEP-1686 异步任务） |

## 延伸阅读

- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 规范文档
- [Model Context Protocol — Architecture concepts](https://modelcontextprotocol.io/docs/concepts/architecture) — 六原语心智模型
- [Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) — 2024 年 11 月发布帖
- [MCP blog — First MCP anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 一周年回顾和 2025-11-25 规范变更
- [WorkOS — MCP 2025-11-25 spec update](https://workos.com/blog/mcp-2025-11-25-spec-update) — SEP-1686、1036、1577、835 和 1724 的摘要