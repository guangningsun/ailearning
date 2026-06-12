# MCP 资源和提示词 — 超越工具的上下文暴露

> 工具占据了 MCP 90% 的注意力。另外两个服务器原语解决不同的问题。资源暴露数据供读取；提示词暴露可复用模板作为斜杠命令。许多服务器应该使用资源而不是将读取包装在工具中，提示词而不是在客户端提示词中硬编码工作流。本课给出决策规则并详解 `resources/*` 和 `prompts/*` 消息。

**类型：** 构建
**语言：** Python（标准库，资源 + 提示词处理程序）
**前置条件：** 阶段 13 · 07（MCP 服务器）
**时间：** 约 45 分钟

## 学习目标

- 决定将某个能力暴露为工具、资源还是提示词。
- 实现 `resources/list`、`resources/read`、`resources/subscribe` 并处理 `notifications/resources/updated`。
- 实现 `prompts/list` 和 `prompts/get`，包含参数模板。
- 识别宿主何时将提示词作为斜杠命令展示 vs 自动注入上下文。

## 问题

一个天真的笔记应用 MCP 服务器将所有内容都暴露为工具：`notes_read`、`notes_list`、`notes_search`。这将每个数据访问都包装在模型驱动的工具调用中。后果：

- 模型必须决定是否为每个可能受益于上下文的查询调用 `notes_read`。
- 只读内容无法订阅或流式传输到宿主的面板。
- 客户端 UI（Claude Desktop 的资源附件面板、Cursor 的"包含文件"选择器）无法展示数据。

正确的划分：数据暴露为资源，变更或计算操作暴露为工具，可复用多步骤工作流暴露为提示词。每个原语都有其 UX 体现和访问模式。

## 概念

### 工具 vs 资源 vs 提示词 — 决策规则

| 能力 | 原语 |
|------------|-----------|
| 用户想要搜索、过滤或转换数据 | 工具 |
| 用户想要宿主将此数据作为上下文包含 | 资源 |
| 用户想要一个可以重新运行的可模板化工作流 | 提示词 |

指导原则：如果模型在每个相关查询中都会受益于调用它，它就是工具。如果用户会受益于将其附加到对话中，它就是资源。如果整个多步骤工作流是用户想要复用的单元，它就是提示词。

### 资源

`resources/list` 返回 `{resources: [{uri, name, mimeType, description?}]}`。`resources/read` 接受 `{uri}` 并返回 `{contents: [{uri, mimeType, text | blob}]}`。

URI 可以是任何可寻址的：

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14`（自定义方案）
- `memory://session-2026-04-22/recent`（服务器特定）

`contents[]` 同时支持文本和二进制。二进制使用 `blob` 作为 base64 编码字符串加上 `mimeType`。

### 资源订阅

在能力中声明 `{resources: {subscribe: true}}`。客户端调用 `resources/subscribe {uri}`。当资源更改时服务器发送 `notifications/resources/updated {uri}`。客户端重新读取。

用例：一个笔记服务器，其资源是磁盘上的文件；文件监视器触发更新通知；Claude Desktop 在外部编辑时将文件重新拉入上下文。

### 资源模板（2025-11-25 新增）

`resourceTemplates` 让你暴露一个参数化 URI 模式：`notes://{id}`，`id` 作为补全目标。客户端可以在资源选择器中自动补全 id。

### 提示词

`prompts/list` 返回 `{prompts: [{name, description, arguments?}]}`。`prompts/get` 接受 `{name, arguments}` 并返回 `{description, messages: [{role, content}]}`。

提示词是一个模板，填充后成为宿主馈送给其模型的消息列表。例如，`code_review` 提示词接受 `file_path` 参数并返回一个三条消息的序列：一条系统消息、一条包含文件正文的用户消息，以及一条带有推理模板的助手启动消息。

### 宿主与提示词

Claude Desktop、VS Code 和 Cursor 在聊天 UI 中将提示词作为斜杠命令展示。用户输入 `/code_review` 并从表单中选择参数。服务器的提示词是"用户快捷方式"和"发送给模型的完整提示词"之间的契约。

并非所有客户端都支持提示词——检查能力协商。声明了提示词能力但客户端不支持的服务器根本不会看到斜杠命令。

### "列表已更改"通知

资源和提示词在集合发生变化时都会发出 `notifications/list_changed`。一个刚刚导入了 20 个新笔记的笔记服务器发出 `notifications/resources/list_changed`；客户端重新调用 `resources/list` 以获取新增内容。

### 内容类型约定

文本：`mimeType: "text/plain"`、`text/markdown`、`application/json`。
二进制：`image/png`、`application/pdf`，加上 `blob` 字段。
MCP Apps（第 14 课）：`text/html;profile=mcp-app`，位于 `ui://` URI 中。

### 动态资源

资源 URI 不必对应静态文件。`notes://recent` 可以在每次读取时返回最新的五个笔记。`db://query/users/active` 可以执行参数化查询。服务器可以动态计算内容。

规则：如果客户端可以按 URI 缓存，URI 必须稳定。如果计算是一次性的，URI 应该包含时间戳或随机数以防止客户端缓存过期。

### 订阅 vs 轮询

支持订阅的客户端通过 `notifications/resources/updated` 获取服务器推送。不支持订阅的客户端或宿主通过重新读取进行轮询。两者都符合规范。服务器的能力声明告诉客户端它支持哪种方式。

订阅的成本：服务器上每个会话的状态（谁订阅了什么）。保持订阅集合有界；断开的客户端应该超时。

### 提示词 vs 系统提示词

MCP 中的提示词不是系统提示词。宿主的系统提示词（其自身的操作指令）和 MCP 提示词（用户调用的服务器提供模板）并存。一个行为良好的客户端永远不会让服务器提示词覆盖其自身的系统提示词；它会将它们分层。

## 使用它

`code/main.py` 扩展了第 07 课的笔记服务器，增加了：

- 每个笔记的资源（`notes://note-1` 等），支持 `resources/subscribe`。
- 一个 `review_note` 提示词，渲染为三条消息模板。
- 一个文件监视器模拟，在笔记被修改时发出 `notifications/resources/updated`。
- 一个 `notes://recent` 动态资源，始终返回最新的五个笔记。

运行演示以查看完整流程。

## 交付它

本课产出 `outputs/skill-primitive-splitter.md`。给定一个提议的 MCP 服务器，该 skill 将每个能力分类为工具 / 资源 / 提示词，并附带理由。

## 练习

1. 运行 `code/main.py`。观察初始资源列表，然后触发笔记编辑并验证 `notifications/resources/updated` 事件是否触发。

2. 添加 `resources/list_changed` 发射器：当创建新笔记时，发送通知以便客户端重新发现。

3. 为 GitHub MCP 服务器设计三个提示词：`summarize_pr`、`triage_issue`、`release_notes`。每个都带参数模式。提示词主体应该可以直接运行而无需进一步修改。

4. 取第 07 课服务器中的一个现有工具，判断它应该保持为工具还是拆分为资源加工具对。用一句话说明理由。

5. 阅读规范中 `server/resources` 和 `server/prompts` 部分。找出 `resources/read` 中很少被填充但规范支持的字段。提示：查看资源内容上的 `_meta`。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| 资源 | "暴露的数据" | 宿主可读的 URI 可寻址内容 |
| 资源 URI | "指向数据的指针" | 带方案前缀的标识符（`file://`、`notes://` 等） |
| `resources/subscribe` | "监视更改" | 客户端选择加入的服务器推送更新，针对特定 URI |
| `notifications/resources/updated` | "资源已更改" | 信号：通知客户端订阅的资源有新内容 |
| 资源模板 | "参数化 URI" | 带宿主选择器补全提示的 URI 模式 |
| 提示词 | "斜杠命令模板" | 带参数槽的命名多消息模板 |
| 提示词参数 | "模板输入" | 宿主在渲染前收集的 typed 参数 |
| `prompts/get` | "渲染模板" | 服务器返回填充后的消息列表 |
| 内容块 | "类型化块" | `{type: text \| image \| resource \| ui_resource}` |
| 斜杠命令 UX | "用户快捷方式" | 宿主将提示词展示为以 `/` 开头的命令 |

## 延伸阅读

- [MCP — 概念：资源](https://modelcontextprotocol.io/docs/concepts/resources) — 资源 URI、订阅和模板
- [MCP — 概念：提示词](https://modelcontextprotocol.io/docs/concepts/prompts) — 提示词模板和斜杠命令集成
- [MCP — 服务器资源规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) — 完整的 `resources/*` 消息参考
- [MCP — 服务器提示词规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) — 完整的 `prompts/*` 消息参考
- [MCP — 协议信息站：资源](https://modelcontextprotocol.info/docs/concepts/resources/) — 社区指南，扩展了官方文档
