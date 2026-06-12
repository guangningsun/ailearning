# MCP 传输层 — stdio vs Streamable HTTP vs SSE 迁移

> stdio 用于本地，其他地方都不行。Streamable HTTP（2025-03-26）是远程标准。旧的 HTTP+SSE 传输已被废弃，将在 2026 年中期移除。选择错误的传输意味着迁移代价；选择正确的则能获得可远程托管的 MCP 服务器，具备会话连续性和 DNS 重绑定防护。

**类型：** 学习
**语言：** Python（标准库、Streamable HTTP 端点骨架）
**前置条件：** 阶段 13 · 07、08（MCP 服务器和客户端）
**时间：** 约 45 分钟

## 学习目标

- 根据部署形态在 stdio 和 Streamable HTTP 之间选择（本地 vs 远程、单进程 vs 集群）。
- 实现 Streamable HTTP 单端点模式：POST 用于请求，GET 用于会话流。
- 强制执行 `Origin` 验证和会话 ID 语义以防御 DNS 重绑定。
- 在 2026 年中期移除截止日期之前将遗留的 HTTP+SSE 服务器迁移到 Streamable HTTP。

## 问题

第一个 MCP 远程传输（2024-11）是 HTTP+SSE：两个端点，一个用于客户端的 POST，另一个用于服务器到客户端流的 Server-Sent-Events 通道。它能工作。但也很笨拙：每个会话两个端点，在某些 CDN 前面缓存会破坏，而且强烈依赖某些 WAF 会积极终止的长寿 SSE 连接。

2025-03-26 规范用 Streamable HTTP 取代了它：一个端点，POST 用于客户端请求，GET 用于建立会话流，两者共享 `Mcp-Session-Id` 头。自此以后构建或迁移的所有服务器都使用 Streamable HTTP。旧的 SSE 模式正在被废弃——Atlassian Rovo 于 2026 年 6 月 30 日将其移除；Keboola 于 2026 年 4 月 1 日移除；大多数剩余的企业服务器在 2026 年底之前移除。

而 stdio 对本地服务器仍然重要。Claude Desktop、VS Code 和每个 IDE 形状的客户端都通过 stdio 派生服务器。正确的心理模型：stdio 用于"这台机器"，Streamable HTTP 用于"通过网络"。没有交叉。

## 概念

### stdio

- 子进程传输。客户端派生服务器，通过 stdin/stdout 通信。
- 每行一个 JSON 对象。换行符分隔。
- 无会话 ID；进程标识就是会话。
- 不需要认证（子进程继承父进程的信任边界）。
- 永远不要用于远程服务器——你需要 SSH 或 socat 来隧道，此时应该使用 Streamable HTTP。

### Streamable HTTP

单个端点 `/mcp`（或任何路径）。支持三种 HTTP 方法：

- **POST /mcp。** 客户端发送 JSON-RPC 消息。服务器回复单个 JSON 响应，或 SSE 流式响应（用于批量响应和与该请求相关的通知）。
- **GET /mcp。** 客户端打开长寿的 SSE 通道。服务器使用它进行服务器到客户端的请求（采样、通知、征求）。
- **DELETE /mcp。** 客户端显式终止会话。

会话由服务器在第一个响应上设置的 `Mcp-Session-Id` 头标识，客户端在每个后续请求上回显该头。会话 ID 必须加密随机（128 位以上）；为安全起见，客户端选择的 ID 会被拒绝。

### 单端点 vs 两个端点

旧规范中的双端点模式在 2026 年仍然可以调用——规范称其为"遗留兼容"。但所有新服务器都应该是单端点的。官方 SDK 发出单端点；只有在与未迁移的远程服务器通信时才使用遗留模式。

### `Origin` 验证和 DNS 重绑定

浏览器不是 MCP 客户端（目前），但攻击者可以制作一个网页，说服浏览器 POST 到 `localhost:1234/mcp`——即用户本地 MCP 服务器监听的地方。如果服务器不检查 `Origin`，浏览器的同源策略不会拯救它，因为 `Origin: http://evil.com` 是有效的跨域请求。

2025-11-25 规范要求服务器拒绝 `Origin` 不在允许列表中的请求。允许列表通常包含 MCP 客户端主机（`https://claude.ai`、`vscode-webview://*`）和本地 UI 的 localhost 变体。

### 会话 ID 生命周期

1. 客户端发送第一个不带 `Mcp-Session-Id` 的请求。
2. 服务器分配一个随机 ID，在响应头中设置 `Mcp-Session-Id`。
3. 客户端在所有后续请求和 `GET /mcp` 流上回显该头。
4. 会话可以被服务器撤销；客户端在后续请求中看到 404，必须重新初始化。
5. 客户端可以显式 DELETE 会话以进行干净关闭。

### 保活与重连

SSE 连接会断开。客户端通过使用相同的 `Mcp-Session-Id` 重新 GET 来重新建立。服务器必须队列化在中断期间错过的事件（在合理窗口内）并通过客户端回显的 `last-event-id` 头重放。

阶段 13 · 13 涵盖了任务，它使长时间运行的工作即使在完全会话重连后也能存活。

### 向后兼容探测

想要同时支持新旧服务器的客户端：

1. POST 到 `/mcp`。
2. 如果响应是 `200 OK` 且带有 JSON 或 SSE，这是 Streamable HTTP。
3. 如果响应是 `200 OK` 且带有 `Content-Type: text/event-stream` 并且有一个 `Location` 头指向辅助端点，这是遗留 HTTP+SSE；跟随 `Location`。

### Cloudflare、ngrok 和托管

2026 年，生产级远程 MCP 服务器运行在 Cloudflare Workers（及其 MCP Agents SDK）、Vercel Functions 或容器化的 Node/Python 上。关键：你的托管必须支持长寿命 HTTP 连接以用于 SSE GET。Vercel 免费套餐限制为 10 秒，不适用。Cloudflare Workers 支持无限流。

### 网关组合

当用网关（阶段 13 · 17）前置多个 MCP 服务器时，网关是一个单一的 Streamable HTTP 端点，重写会话 ID 并多路复用到上游。工具在网关层合并；客户端看到的是单个逻辑服务器。

### 传输失败模式

- **stdio SIGPIPE。** 子进程在写入中途终止会引发 SIGPIPE；服务器应该干净退出。客户端应该检测到 EOF 并将会话标记为已终止。
- **HTTP 502/504。** Cloudflare、nginx 和其他代理在上游失败时发出这些。Streamable HTTP 客户端应该在短退避后重试一次。
- **SSE 连接断开。** TCP RST、代理超时或客户端网络变化关闭流。客户端使用 `Mcp-Session-Id` 和可选的 `last-event-id` 重连以恢复。
- **会话撤销。** 服务器使会话 ID 无效；客户端在下一个请求中看到 404。客户端必须重新握手。
- **时钟偏移。** 客户端的资源 TTL 计算与服务器不同步。客户端应该将服务器时间戳视为权威。

### 何时绕过 Streamable HTTP

一些企业在其内部网络内部部署 MCP 服务器，通过 gRPC 或消息队列传输。这是非标准的——MCP 的规范没有正式定义这些。网关可以向 MCP 客户端暴露 Streamable HTTP 表面，同时在内部使用 gRPC。保持外部表面符合规范；网关拥有翻译逻辑。

## 使用它

`code/main.py` 使用 `http.server`（标准库）实现了一个最小的 Streamable HTTP 端点。它处理 `/mcp` 上的 POST、GET 和 DELETE，在第一个响应上设置 `Mcp-Session-Id`，验证 `Origin`，并拒绝来自非允许列表来源的请求。处理程序重用了第 07 课笔记服务器的调度逻辑。

需要关注的内容：

- POST 处理程序读取 JSON-RPC 体，分派，并写入 JSON 响应（单响应变体；SSE 变体结构类似）。
- `Origin` 检查拒绝默认的 `http://evil.example` 探测但接受 `http://localhost`。
- 会话 ID 是随机的 128 位十六进制字符串；服务器在内存中保持每会话状态。

## 交付它

本课产出 `outputs/skill-mcp-transport-migrator.md`。给定一个 HTTP+SSE（遗留）MCP 服务器，该 skill 生成一个迁移计划到 Streamable HTTP，包含会话 ID 连续性、Origin 检查和向后兼容探测支持。

## 练习

1. 运行 `code/main.py`。从 `curl` POST 一个 `initialize`，观察 `Mcp-Session-Id` 响应头。POST 第二个请求回显该头并验证会话连续性。

2. 添加一个打开 SSE 流的 GET 处理程序。每五秒发送一个 `notifications/progress` 事件。使用相同的会话 ID 重新 GET 并确认服务器接受它。

3. 实现 `last-event-id` 重放逻辑。在重连时，重放自该 ID 以来生成的任何事件。

4. 扩展 `Origin` 验证以支持通配符模式（`https://*.example.com`），确认它接受 `https://app.example.com` 但拒绝 `https://evil.example.com.attacker.net`。

5. 从官方注册表中获取一个遗留 HTTP+SSE 服务器（有几个），并勾勒迁移方案：端点处理、会话 ID 生成和头语义的哪些部分发生了变化。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| stdio 传输 | "本地子进程" | 通过 stdin/stdout 的 JSON-RPC，换行符分隔 |
| Streamable HTTP | "远程传输" | 单端点 POST + GET + 可选 SSE，2025-03-26 规范 |
| HTTP+SSE | "遗留" | 双端点模式，将在 2026 年中期移除 |
| `Mcp-Session-Id` | "会话头" | 服务器分配的随机 ID，在每个后续请求上回显 |
| `Origin` 允许列表 | "DNS 重绑定防御" | 拒绝 `Origin` 未被批准的请求 |
| 单端点 | "一个 URL" | `/mcp` 处理所有会话操作的 POST / GET / DELETE |
| `last-event-id` | "SSE 重放" | 用于恢复断开流而不丢失事件的头 |
| 向后兼容探测 | "新旧检测" | 客户端响应形状检查，自动选择传输方式 |
| 长寿命 HTTP | "SSE 流式传输" | 服务器在单个 TCP 连接上推送分钟或小时的事件 |
| 会话撤销 | "强制重新初始化" | 服务器使会话 ID 无效；客户端必须重新握手 |

## 延伸阅读

- [MCP — 基础传输规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — stdio 和 Streamable HTTP 的规范参考
- [MCP — 基础传输规范 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) — 引入 Streamable HTTP 的修订版
- [Cloudflare — MCP 传输](https://developers.cloudflare.com/agents/model-context-protocol/transport/) — Workers 托管的 Streamable HTTP 模式
- [AWS — MCP 传输机制](https://builder.aws.com/content/35A0IphCeLvYzly9Sw40G1dVNzc/mcp-transport-mechanisms-stdio-vs-streamable-http) — 跨部署形态的比较
- [Atlassian — HTTP+SSE 废弃通知](https://community.atlassian.com/forums/Atlassian-Remote-MCP-Server/HTTP-SSE-Deprecation-Notice/ba-p/3205484) — 具体的迁移截止日期示例
