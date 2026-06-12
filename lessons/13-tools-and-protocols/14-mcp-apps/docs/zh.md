# MCP 应用 —— 通过 `ui://` 的交互式 UI 资源

> 纯文本工具输出限制了智能体能展示的内容。MCP 应用（SEP-1724，2026 年 1 月 26 日正式发布）允许工具返回沙箱化的交互式 HTML，在 Claude Desktop、ChatGPT、Cursor、Goose 和 VS Code 中内联渲染。仪表盘、表单、地图、3D 场景，全部通过一个扩展实现。本课程讲解 `ui://` 资源方案、`text/html;profile=mcp-app` MIME 类型、iframe 沙箱 postMessage 协议，以及允许服务器渲染 HTML 所带来的安全面。

**类型：** 构建
**语言：** Python（标准库、UI 资源发射器）、HTML（示例应用）
**前置条件：** 阶段 13 · 07（MCP 服务器）、阶段 13 · 10（资源）
**时间：** 约 75 分钟

## 学习目标

- 从工具调用中返回 `ui://` 资源，并设置正确的 MIME 和元数据。
- 用 `_meta.ui.resourceUri`、`_meta.ui.csp` 和 `_meta.ui.permissions` 声明工具关联的 UI。
- 实现用于 UI 与主机通信的 iframe 沙箱 postMessage JSON-RPC。
- 应用 CSP 和权限策略默认值来防御来自 UI 的攻击。

## 问题

2025 年代的 `visualize_timeline` 工具可以返回"以下是按时间顺序组织的 14 条笔记：……"。这只是一段文字。用户实际想要的是交互式时间线。在 MCP 应用出现之前，选项有：客户端特定的部件 API（Claude artifacts、OpenAI Custom GPT HTML），或者根本没有 UI。

MCP 应用（SEP-1724，2026 年 1 月 26 日发布）标准化了这份契约。工具结果包含一个 URI 为 `ui://...`、MIME 类型为 `text/html;profile=mcp-app` 的 `resource`。主机将其在带有有限 CSP 且无网络访问权限（除非明确授权）的沙箱 iframe 中渲染。iframe 内的 UI 通过一个轻量级 postMessage JSON-RPC 方言向主机发送消息。

每个兼容客户端（Claude Desktop、ChatGPT、Goose、VS Code）都以相同方式渲染相同的 `ui://` 资源。一套服务器、一份 HTML 包、通用 UI。

## 概念

### `ui://` 资源方案

工具返回：

```json
{
  "content": [
    {"type": "text", "text": "Here is your notes timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

主机随后对 `ui://notes/timeline` URI 调用 `resources/read`，得到：

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### iframe 沙箱

主机在沙箱 `<iframe>` 中渲染 HTML，具有：

- `sandbox="allow-scripts allow-same-origin"`（或按服务器声明的更严格配置）
- 通过响应头应用服务器声明的 CSP。
- 无来自主机源的 cookies、localStorage。
- 网络访问受 CSP 中 `connectSrc` 限制。

### postMessage 协议

iframe 通过 `window.postMessage` 与主机通信。一个轻量级 JSON-RPC 2.0 方言：

始终将 `targetOrigin` 固定到对端的精确来源，在接收端处理任何 payload 前验证 `event.origin` 是否在允许列表中。切勿为此通道的任何一端使用 `"*"`，因为消息体携带工具调用和资源读取。

```js
// iframe 到主机（固定到主机来源）
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// 主机到 iframe（固定到 iframe 来源）
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// 双方接收方
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // 安全地处理 event.data
});
```

UI 可调用的主机端方法：

- `host.callTool(name, arguments)` — 调用服务器工具。
- `host.readResource(uri)` — 读取 MCP 资源。
- `host.getPrompt(name, arguments)` — 获取提示词模板。
- `host.close()` — 关闭 UI。

每次调用仍经过 MCP 协议并继承服务器的权限。

### 权限

`_meta.ui.permissions` 列表请求额外能力：

- `camera` — 访问用户摄像头（用于扫描文档 UI）。
- `microphone` — 语音输入。
- `geolocation` — 位置。
- `network:*` — 比单独 `connectSrc` 更广泛的网络访问。

每个权限都会在 UI 渲染前向用户显示提示。

### 安全风险

iframe 中的 HTML 仍然是 HTML。新的攻击面：

- **通过 UI 的提示词注入。** 恶意服务器 UI 可以显示看起来像系统消息的文本，欺骗用户。主机渲染应明显区分服务器 UI 和主机 UI。
- **通过 `connectSrc` 的数据泄露。** 如果 CSP 允许 `connect-src: *`，UI 可以向任何地方发送数据。默认应严格。
- **点击劫持。** UI 覆盖主机界面。主机必须防止 z-index 操作并执行不透明度规则。
- **窃取焦点。** UI 获得键盘焦点并捕获下一条消息。主机必须拦截。

阶段 13 · 15 在 MCP 安全部分深入介绍这些内容；本课程仅做引入。

### `ui/initialize` 握手

iframe 加载后，通过 postMessage 发送 `ui/initialize`：

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

主机响应能力列表和会话令牌。UI 在后续每次主机调用中使用会话令牌。

### AppRenderer / AppFrame SDK 原语

ext-apps SDK 提供两个便捷原语：

- `AppRenderer`（服务器端）— 包装 React / Vue / Solid 组件，生成具有正确 MIME 和元数据的 `ui://` 资源。
- `AppFrame`（客户端）— 接收资源，挂载 iframe，并调解 postMessage。

你可以使用这些或手写 HTML 和 JSON-RPC。

### 生态系统状态

MCP 应用于 2026 年 1 月 26 日发布。截至 2026 年 4 月的客户端支持：

- **Claude Desktop。** 自 2026 年 1 月起完全支持。
- **ChatGPT。** 通过应用 SDK 完全支持（相同的底层 MCP 应用协议）。
- **Cursor。** Beta 版；通过设置启用。
- **VS Code。** 仅内部构建版本。
- **Goose。** 完全支持。
- **Zed、Windsurf。** 已在路线图中。

生产环境中的服务器：仪表盘、地图可视化、数据表、图表构建器、沙箱 IDE 预览。

## 使用它

`code/main.py` 扩展了笔记服务器，增加了返回 `ui://notes/timeline` 资源的 `visualize_timeline` 工具，以及处理该 URI 上 `resources/read` 的处理器，后者返回一个带有 SVG 时间线的完整小型 HTML 包。HTML 使用标准库模板——无需构建系统。postMessage 在 JS 注释中概述，因为标准库无法驱动浏览器。

关注点：

- 工具响应上的 `_meta.ui` 携带 resourceUri、CSP、permissions。
- HTML 无网络访问即可渲染；所有数据均内联。
- JS 通过 `window.parent.postMessage` 调用 `host.callTool`（已记录但在此标准库演示中为惰性）。

## 交付它

本课程产出 `outputs/skill-mcp-apps-spec.md`。给定一个能从交互式 UI 中受益的工具，该技能生成完整的 MCP 应用契约：`ui://` URI、CSP、权限、postMessage 入口点和安全检查清单。

## 练习

1. 运行 `code/main.py` 并检查发出的 HTML。直接在浏览器中打开 HTML；验证 SVG 是否渲染。然后概述 UI 调用 `host.callTool("notes_update", ...)` 将使用的 postMessage 契约。

2. 收紧 CSP：移除 `'unsafe-inline'` 并使用基于 nonce 的脚本策略。HTML 生成代码中需要修改什么？

3. 添加第二个 UI 资源 `ui://notes/editor`，其中包含用于就地编辑笔记的表单。当用户提交时，iframe 调用 `host.callTool("notes_update", ...)`。

4. 审计 UI 的攻击面。恶意服务器可以在哪里注入内容？iframe 沙箱防御什么、不防御什么？

5. 阅读 SEP-1724 规范，找出 MCP 应用 SDK 中此玩具实现未使用的一个能力。（提示：组件级状态同步。）

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| MCP 应用 | "交互式 UI 资源" | 2026-01-26 发布的 SEP-1724 扩展 |
| `ui://` | "应用 URI 方案" | 用于 UI 包的资源方案 |
| `text/html;profile=mcp-app` | "MIME 类型" | MCP 应用 HTML 的内容类型 |
| iframe 沙箱 | "渲染容器" | 使用 CSP 和权限的浏览器 UI 沙箱 |
| postMessage JSON-RPC | "UI 到主机线路" | 用于主机调用的轻量级 postMessage JSON-RPC 方言 |
| `_meta.ui` | "工具-UI 绑定" | 将工具结果链接到 UI 资源的元数据 |
| CSP | "内容安全策略" | 声明脚本、网络、样式的允许来源 |
| AppRenderer | "服务器 SDK 原语" | 将框架组件转换为 `ui://` 资源 |
| AppFrame | "客户端 SDK 原语" | 调解 postMessage 的 iframe 挂载助手 |
| `ui/initialize` | "握手" | UI 到主机的第一个 postMessage |

## 延伸阅读

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — 参考实现和 SDK
- [MCP Apps specification 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) — 正式规范文档
- [MCP — Apps extension overview](https://modelcontextprotocol.io/extensions/apps/overview) — 高级文档
- [MCP blog — MCP Apps launch](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) — 2026 年 1 月发布帖子
- [MCP Apps API reference](https://apps.extensions.modelcontextprotocol.io/api/) — JSDoc 风格 SDK 参考