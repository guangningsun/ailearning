# MCP 安全 II —— OAuth 2.1、资源指示符、增量作用域

> 远程 MCP 服务器需要授权，而不仅仅是身份验证。2025-11-25 规范与 OAuth 2.1 + PKCE + 资源指示符（RFC 8707）+ 受保护资源元数据（RFC 9728）对齐。SEP-835 添加了增量作用域同意，在 403 WWW-Authenticate 上进行升级授权。本课程将升级流程实现为状态机，让你看到每一跳。

**类型：** 构建
**语言：** Python（标准库、OAuth 状态机模拟器）
**前置条件：** 阶段 13 · 09（传输）、阶段 13 · 15（安全 I）
**时间：** 约 75 分钟

## 学习目标

- 区分资源服务器与授权服务器的职责。
- 走查受 PKCE 保护的 OAuth 2.1 授权码流程。
- 使用 `resource`（RFC 8707）和受保护资源元数据（RFC 9728）防止混乱代理攻击。
- 实现升级授权：服务器响应 403 并带有 WWW-Authenticate，要求更高作用域；客户端重新提示用户同意并重试。

## 问题

早期 MCP（2025 年之前）使用临时 API 密钥甚至无认证来发布远程服务器。2025-11-25 规范用完整的 OAuth 2.1 配置封闭了这一缺口。

三个真实世界的需求：

- **普通远程服务器。** 用户安装访问其 Notion / GitHub / Gmail 的远程 MCP 服务器。OAuth 2.1 + PKCE 是正确的形态。
- **作用域升级。** 被授予 `notes:read` 的笔记服务器后来可能需要 `notes:write` 来执行特定操作。SEP-835 的升级（step-up）不需要重做整个流程，而是请求额外的作用域。
- **防止混乱代理。** 客户端持有一个针对服务器 A 范围限定作用域的令牌。服务器 A 是恶意的，试图将令牌呈现给服务器 B。资源指示符（RFC 8707）将令牌绑定到其预期受众。

OAuth 2.1 并不是新事物。新的 MCP 的配置是：特定必需的流程（仅授权码 + PKCE；无隐式流，默认无客户端凭证），每个令牌请求强制资源指示符，以及发布受保护资源元数据以便客户端知道去哪里。

## 概念

### 角色

- **客户端。** MCP 客户端（Claude Desktop、Cursor 等）。
- **资源服务器。** MCP 服务器（笔记、GitHub、Postgres，或任何）。
- **授权服务器。** 颁发令牌的机构。可以与资源服务器是同一服务，也可以是独立的 IdP（Auth0、Keycloak、Cognito）。

在 MCP 的配置中，资源和授权服务器可以是同一主机，但应用不同 URL 来区分。

### 授权码 + PKCE

流程：

1. 客户端生成 `code_verifier`（随机）和 `code_challenge`（SHA256）。
2. 客户端将用户重定向到 `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`。
3. 用户同意。授权服务器重定向到 `redirect_uri?code=...`。
4. 客户端 POST 到 `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...`。
5. 授权服务器验证令牌的哈希与存储的 challenge 匹配，颁发访问令牌。
6. 客户端使用令牌：`Authorization: Bearer ...` 对资源服务器的每个请求。

PKCE 防止授权码拦截攻击。资源指示符防止令牌在其他地方有效。

### 受保护资源元数据（RFC 9728）

资源服务器发布 `.well-known/oauth-protected-resource` 文档：

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

客户端从资源服务器发现授权服务器。减少配置——客户端只需资源 URL。

### 资源指示符（RFC 8707）

令牌请求中的 `resource` 参数将令牌的预期受众固定。颁发的令牌包含 `aud: "https://notes.example.com"`。另一个接收此令牌的 MCP 服务器检查 `aud` 并拒绝。

### 作用域模型

作用域是空格分隔的字符串。常见 MCP 约定：

- `notes:read`、`notes:write`、`notes:delete`
- `admin:*` 用于管理能力（谨慎使用）
- `profile:read` 用于身份

作用域选择应遵循最小权限：请求你现在需要的，必要时再升级。

### 升级授权（SEP-835）

用户授予 `notes:read`。后来要求智能体删除一条笔记。服务器响应：

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

客户端看到 insufficient_scope 错误，用额外作用域的同意对话框提示用户，为其执行迷你 OAuth 流程，用新令牌重试请求。

### 令牌受众验证

每个请求：服务器检查 `token.aud == self.resource_url`。不匹配 = 401。这阻止跨服务器令牌重用。

### 短期令牌和轮换

访问令牌应该是短期的（默认 1 小时）。刷新令牌在每次刷新时轮换。客户端在后台处理静默刷新。

### 不允许令牌传递

采样服务器（阶段 13 · 11）不得将客户端的令牌传递到其他服务。采样请求是边界。

### 防止混乱代理

令牌绑定到 `aud`。客户端绑定到 `client_id`。每个请求针对两者进行验证。规范明确禁止旧的"传递令牌"模式，这在 pre-MCP 远程工具生态系统中很常见。

### 客户端 ID 发现

每个 MCP 客户端在固定 URL 发布其元数据。授权服务器可以获取客户端的元数据文档以发现重定向 URI 和联系信息。这消除了手动客户端注册。

### 网关与 OAuth

阶段 13 · 17 展示企业网关如何处理 OAuth：网关持有上游服务器的凭证，发给客户端的令牌由网关颁发，上游令牌永不离开网关。这翻转了信任模型——用户向网关认证一次；网关处理 N 个服务器的授权。

## 使用它

`code/main.py` 将完整的 OAuth 2.1 升级流程模拟为状态机。它实现：

- PKCE code-verifier / challenge 生成。
- 带资源指示符的授权码流程。
- 受保护资源元数据端点。
- 带受众检查的令牌验证。
- 不足作用域时的升级。

本课程中无 HTTP 服务器；状态机在内存中运行，因此你可以跟踪每一跳。阶段 13 · 17 的网关课程将其连接到实际传输。

## 交付它

本课程产出 `outputs/skill-oauth-scope-planner.md`。给定一个带有工具的远程 MCP 服务器，该技能设计作用域集、pin 规则和升级策略。

## 练习

1. 运行 `code/main.py`。跟踪两作用域升级流程。记录升级时哪些跳重复。

2. 添加刷新令牌轮换：每次刷新发放新刷新令牌并使旧令牌失效。模拟被盗刷新令牌在轮换后被使用并确认失败。

3. 使用标准库 http.server 将受保护资源元数据端点实现为真实 HTTP 响应。镜像第 09 课的 /mcp 端点。

4. 为 GitHub MCP 服务器设计作用域层次结构：读取仓库、写入 PR、批准 PR、合并 PR、管理。用升级连接每个级别。

5. 阅读 RFC 8707 和 RFC 9728。找出 9728 中 MCP 使用方式与 RFC 示例不同的一个字段。（提示：涉及 `scopes_supported`。）

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| OAuth 2.1 | "现代 OAuth" | 合并的 RFC，要求 PKCE 并禁止隐式流 |
| PKCE | "持有证明" | 通过代码验证器 + challenge 击败授权码拦截 |
| 资源指示符 | "令牌受众" | RFC 8707 `resource` 参数将令牌绑定到一个服务器 |
| 受保护资源元数据 | "发现文档" | RFC 9728 `.well-known/oauth-protected-resource` |
| 升级授权 | "增量同意" | SEP-835 按需添加作用域的流程 |
| `insufficient_scope` | "带有 WWW-Authenticate 的 403" | 服务器信号，重新同意更大的作用域 |
| 混乱代理 | "跨服务令牌重用" | 受信任持有者不当地转发令牌的攻 |
| 短期令牌 | "访问令牌 TTL" | 快速过期的 Bearer；刷新令牌续期 |
| 作用域层次结构 | "最小权限堆栈" | 每个级别之间有升级的渐进作用域集 |
| 客户端 ID 元数据 | "客户端发现文档" | 客户端发布其自身 OAuth 元数据的 URL |

## 延伸阅读

- [MCP — Authorization spec](https://modelcontextprotocol.io/specification/draft/basic/authorization) — 规范的 MCP OAuth 配置
- [den.dev — MCP November authorization spec](https://den.dev/blog/mcp-november-authorization-spec/) — 2025-11-25 变更的演练
- [RFC 8707 — Resource indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — 受众绑定的 RFC
- [RFC 9728 — OAuth 2.0 protected resource metadata](https://datatracker.ietf.org/doc/html/rfc9728) — 发现文档 RFC
- [Aembit — MCP OAuth 2.1, PKCE and the future of AI authorization](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — 实用的升级流程演练