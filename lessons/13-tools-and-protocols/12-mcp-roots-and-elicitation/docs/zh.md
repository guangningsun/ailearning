# 根目录与引出 — 作用域与飞行中用户输入

> 硬编码路径在用户打开不同项目时就会失效。预填充的工具参数在用户输入不足时失效。根目录将服务器作用域限定为一组用户控制的 URI；引出在工具调用过程中暂停，向用户请求结构化输入（表单或 URL）。两个客户端原语，两种常见 MCP 失败模式的修复方案。SEP-1036（URL 模式引出，2025-11-25）在 2026 年 H1 前为实验性——依赖前请检查 SDK 版本。

**类型：** 构建型
**语言：** Python（标准库、根目录 + 引出演示）
**前置条件：** 阶段 13 · 07（MCP 服务器）
**时间：** 约 45 分钟

## 学习目标

- 声明 `roots` 并响应 `notifications/roots/list_changed`。
- 将服务器文件操作限制在声明的根目录集合内的 URI。
- 使用 `elicitation/create` 在工具调用过程中向用户请求确认或结构化输入。
- 在表单模式和 URL 模式引出之间选择（后者是实验性的；已注明漂移风险）。

## 问题

便笺 MCP 服务器在生产环境中遇到的两种具体失败。

**路径假设失效。** 服务器针对 `~/notes` 编写。另一台机器上笔记在 `~/Documents/Notes` 的用户会收到一个静默失败（找不到文件）的工具调用，或者更糟，写到了错误的地方。

**缺少用户知道的参数。** 用户要求"删除旧的 TPS 报告便笺"。模型调用 `notes_delete(title: "TPS report")`，但有三篇匹配的便笺，来自 2023、2024 和 2025 年。工具无法猜测。返回"模糊"很烦人；对全部三篇执行则是灾难性的。

根目录修复第一个：客户端在 `initialize` 时声明服务器可以访问的 URI 集合。引出修复第二个：服务器暂停工具调用并发送 `elicitation/create` 请求用户选择哪一篇。

## 概念

### 根目录

客户端在 `initialize` 时声明根目录列表：

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

服务器然后可以调用 `roots/list`：

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

服务器必须将根目录视为边界：任何对根目录集合之外文件的读取或写入都会被拒绝。客户端不会强制执行此规定（服务器仍是用户信任的代码），但符合规范的服务器会遵守。

当用户添加或移除根目录时，客户端发送 `notifications/roots/list_changed`。服务器重新调用 `roots/list` 并更新其边界。

### 为什么根目录是客户端原语

根目录由客户端声明，因为它们代表用户的同意模型。用户告诉 Claude Desktop"给这个便笺服务器访问这两个目录的权限"。服务器无法扩大该作用域。

### 引出：表单模式默认值

`elicitation/create` 接收一个表单模式加上自然语言提示：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Delete 'TPS report'? Multiple notes match; pick one.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

客户端呈现一个表单，收集用户答案，返回：

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

三种可能的操作：`accept`（用户填写了）、`decline`（用户关闭了）、`cancel`（用户中止了整个工具调用）。

表单模式是扁平的——嵌套对象在 v1 中不受支持。SDK 通常会拒绝任何比单层更复杂的内容。

### 引出：URL 模式（SEP-1036，实验性）

2025-11-25 新增。不是模式，服务器发送一个 URL：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Sign in to GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

客户端在浏览器中打开 URL，等待完成，当用户返回时返回。当表单不够用时，这适用于 OAuth 流程、支付授权和文档签名。

漂移风险提示：SEP-1036 响应形态仍在调整中；一些 SDK 返回回调 URL，其他返回完成 token。在生产环境中使用 URL 模式前请阅读 SDK 的发布说明。

### 何时使用引出是正确的

- 破坏性操作前的用户确认（破坏性提示 + 引出）。
- 消歧（从 N 个匹配项中选择一个）。
- 首次运行设置（API 密钥、目录、偏好）。
- OAuth 风格流程（URL 模式）。

### 何时使用引出是错误的

- 填写工具的必需参数，而模型本可以用文本询问。用正常的重新提示，而不是引出对话框。
- 高频调用。引出会中断对话；不要在循环内部触发它。
- 服务器事后可以验证的任何内容。验证后返回错误，让模型用文本向用户询问。

### 人在环中桥接

引出加上采样共同实现了 MCP 的"人在环中"模型。服务器的 Agent 循环可以暂停等待用户输入（引出）或模型推理（采样）。阶段 13 · 11 涵盖了采样；本课涵盖引出。将它们组合起来可实现完整的环中控制。

## 使用它

`code/main.py` 扩展了便笺服务器，包含：

- 服务器在根目录列表更改通知后重新查询的 `roots/list` 响应。
- 一个当多个便笺匹配时使用 `elicitation/create` 进行消歧的 `notes_delete` 工具。
- 一个使用 URL 模式引出打开首次运行配置页面的 `notes_setup` 工具（模拟）。
- 一个拒绝在声明根目录之外 URI 上执行操作的边界检查。

演示运行三个场景：快乐路径（一个匹配）、消歧（三个匹配，触发引出）、根目录外写入（被拒绝）。

## 交付它

本课产出 `outputs/skill-elicitation-form-designer.md`。给定一个可能需要用户确认或消歧的工具，该技能设计引出表单模式和消息模板。

## 练习

1. 运行 `code/main.py`。触发消歧路径；确认模拟的用户答案被路由回工具。

2. 添加一个新工具 `notes_archive`，每次都需要引出确认（破坏性提示）。检查用户体验：这与模型用文本重新询问相比如何？

3. 为首次运行 OAuth 流程实现 URL 模式引出。注意漂移风险并添加 SDK 版本保护。

4. 扩展 `roots/list` 处理：当通知到达时，服务器应原子性地重新读取并重新扫描现在可能超出范围的打开文件句柄。

5. 阅读 GitHub 上的 SEP-1036 问题讨论线程。找出一个影响服务器应如何处理 URL 模式回调的开放问题。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| 根目录 (Root) | "同意边界" | 客户端允许服务器访问的 URI |
| `roots/list` | "服务器请求作用域" | 客户端返回当前根目录集合 |
| `notifications/roots/list_changed` | "用户更改了作用域" | 客户端发出根目录集合已突变的信号 |
| 引出 (Elicitation) | "在调用中途询问用户" | 服务器发起的结构化用户输入请求 |
| `elicitation/create` | "该方法" | 引出请求的 JSON-RPC 方法 |
| 表单模式 (Form mode) | "模式驱动的表单" | 扁平的 JSON Schema 在客户端 UI 中呈现为表单 |
| URL 模式 (URL mode) | "浏览器重定向" | SEP-1036 实验性；在 URL 中打开并等待 |
| `accept` / `decline` / `cancel` | "用户响应结果" | 服务器处理的三种分支 |
| 消歧 (Disambiguation) | "选择一个" | 当工具有 N 个候选时常见的引出用例 |
| 扁平表单 (Flat form) | "仅顶层属性" | 引出模式不能嵌套 |

## 进一步阅读

- [MCP — 客户端根目录规范](https://modelcontextprotocol.io/specification/draft/client/roots) — 规范的根目录参考
- [MCP — 客户端引出规范](https://modelcontextprotocol.io/specification/draft/client/elicitation) — 规范的引出参考
- [Cisco — MCP 引出、结构化内容和 OAuth 增强的新特性](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) — 2025-11-25 新增功能演练
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) — URL 模式引出提案（实验性，漂移风险）
- [The New Stack — MCP 中的引出如何将人在环中带入 AI 工具](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) — 用户体验演练