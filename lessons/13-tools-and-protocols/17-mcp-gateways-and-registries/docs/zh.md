# MCP 网关与注册表 — 企业控制平面

> 企业不能让每个开发人员随意安装各种 MCP 服务器。网关将认证、RBAC、审计、限速、缓存和工具投毒检测集中化，然后以单一 MCP 端点形式暴露合并后的工具面。官方 MCP 注册表（Anthropic + GitHub + PulseMCP + Microsoft，命名空间已验证）是规范的上游。本课介绍网关的适用场景，给出一个最小实现，并概述 2026 年供应商格局。

**类型：** 学习型
**语言：** Python（标准库，最小网关）
**前置条件：** 阶段 13 · 15（工具投毒）、阶段 13 · 16（OAuth 2.1）
**时间：** 约 45 分钟

## 学习目标

- 解释 MCP 网关的定位（介于 MCP 客户端与多个后端 MCP 服务器之间）。
- 实现网关的五大职责：认证、RBAC、审计、限速、策略。
- 在网关层强制执行固定工具哈希清单。
- 区分官方 MCP 注册表与元注册表（Glama、MCPMarket、MCP.so、Smithery、LobeHub）。

## 问题

一家财富 500 强企业有 30 个已批准的 MCP 服务器、5000 名开发人员、合规与审计要求，以及希望集中化策略的安全团队。允许每个开发人员在他们的 IDE 中随意安装服务器是不可接受的。

网关模式：

1. 网关作为单一 Streamable HTTP 端点运行，开发人员连接到该端点。
2. 网关持有每个后端 MCP 服务器的凭证。
3. 每个开发人员的请求通过网关自身的 OAuth 进行认证和作用域限定。
4. 网关将调用路由到后端服务器，并应用策略。
5. 所有调用均记录以供审计。

Cloudflare MCP Portals、Kong AI Gateway、IBM ContextForge、MintMCP、TrueFoundry、Envoy AI Gateway — 均在 2025-2026 年发布了网关或网关功能。

与此同时，官方 MCP 注册表作为规范的上游推出：经过筛选、命名空间已验证、反向 DNS 命名的服务器，网关可以从中拉取。元注册表（Glama、MCPMarket、MCP.so、Smithery、LobeHub）聚合来自多个源的服务器。

## 概念

### 网关的五大职责

1. **认证。** 使用 OAuth 2.1 识别开发人员；映射到用户角色。
2. **RBAC。** 逐用户策略：哪些服务器、哪些工具、哪些作用域。
3. **审计。** 每次调用记录谁、何时、何事、结果。
4. **限速。** 逐用户/逐工具/逐服务器上限，防止滥用。
5. **策略。** 拒绝投毒描述，强制执行"双签规则"，脱敏 PII。

### 网关作为单一端点

对开发人员来说，网关看起来像一个 MCP 服务器。内部它路由到 N 个后端。会话 ID（阶段 13 · 09）在边界处被重写。

### 凭证保险库

开发人员永远不会看到后端令牌。网关持有它们（或代理到执行此操作的身份提供者）。拥有网关 `notes:read` 权限的开发人员可以通过网关自身的后端凭证传递访问 notes MCP 服务器 —— 但仅在绑定传递访问的策略下。

### 网关层的工具哈希固定

网关持有已批准工具描述的清单（SHA256 哈希）。在发现时，它获取每个后端的 `tools/list`，将哈希与清单比较，并移除描述发生变化的任何工具。这是阶段 13 · 15 中的"拉地毯防御"在中心化应用。

### 策略即代码

高级网关使用 OPA/Rego、Kyverno 或 Styra 表达策略。像"用户 `alice` 只能在 `acme` 组织中的仓库上调用 `github.open_pr`"这样的规则以声明方式编码。简单的网关使用手写 Python。两种形态都有效。

### 会话感知路由

当用户的会话包含多个服务器时，网关进行多路复用：开发人员的单个 MCP 会话持有 N 个后端会话，每个服务器一个。来自任何后端的消息通过网关路由到开发人员的会话。

### 命名空间合并

网关从所有后端合并工具命名空间，通常在冲突时添加前缀。`github.open_pr`、`notes.search`。这使路由明确无歧义。

### 注册表

- **官方 MCP 注册表（`registry.modelcontextprotocol.io`）。** 在 Anthropic、GitHub、PulseMCP、Microsoft 托管下推出。命名空间已验证（反向 DNS：`io.github.user/server`）。预先过滤基本质量。
- **Glama。** 以搜索为中心的元注册表，聚合多个来源。
- **MCPMarket。** 商业导向的目录，有供应商列表。
- **MCP.so。** 社区目录；开放提交。
- **Smithery。** 包管理器风格的安装流程。
- **LobeHub。** 在他们的 LobeChat 应用中集成的 UI 注册表。

企业网关默认从官方注册表拉取，允许管理员从元注册表添加内容，并拒绝任何未固定的内容。

### 反向 DNS 命名

官方注册表要求公共服务器使用反向 DNS 名称：`io.github.alice/notes`。命名空间防止抢注，使信任委托更清晰。

### 供应商调查，2026 年 4 月

| 供应商 | 优势 |
|--------|----------|
| Cloudflare MCP Portals | 边缘托管；OAuth 集成；免费层 |
| Kong AI Gateway | K8s 原生；细粒度策略；日志发送到 OpenTelemetry |
| IBM ContextForge | 企业 IAM；合规；审计导出 |
| TrueFoundry | DevOps 导向；指标优先 |
| MintMCP | 开发者平台导向 |
| Envoy AI Gateway | 开源；可定制过滤器 |

阶段 17（生产基础设施）深入探讨网关运营。

## 使用它

`code/main.py` 以约 150 行代码实现了一个最小网关：通过伪造 Bearer token 认证用户，持有逐用户 RBAC 策略，将请求路由到两个后端 MCP 服务器，将每次调用写入审计日志，强制执行限速，并拒绝任何描述哈希与固定清单不匹配的后端工具。

需要关注的重点：

- `RBAC` 字典以 `user_id` 为键，包含允许的 `server_tool` 条目。
- `AUDIT_LOG` 是一个只追加的事件列表。
- 限速使用每个用户的令牌桶。
- 固定清单是一个 `server::tool -> hash` 的字典。

## 交付它

本课产出 `outputs/skill-gateway-bootstrap.md`。给定一个企业 MCP 计划（用户、后端、合规），该技能生成网关配置规范。

## 练习

1. 运行 `code/main.py`。以允许的用户身份进行调用；然后以禁止的用户身份调用；然后超出限速的突发调用。验证所有三种流程。

2. 添加一个策略，在结果返回客户端之前脱敏 PII。使用简单的正则表达式匹配 SSN 格式字符串；注意缺口（电子邮件、电话号码）。

3. 扩展审计日志以发出 OpenTelemetry GenAI span。阶段 13 · 20 涵盖确切的属性。

4. 为一个 50 名开发人员的团队设计 RBAC 策略，包含五个后端（notes、github、postgres、jira、slack）。谁获得每个的只读权限？谁获得写权限？

5. 从头到尾阅读 Cloudflare 企业 MCP 文章。识别 Cloudflare 提供的、本标准库网关没有的一个功能。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| 网关（Gateway） | "MCP 代理" | 在客户端与后端之间集中化服务器 |
| 凭证保险库（Credential vaulting） | "后端令牌留在服务器端" | 开发人员永远不会看到上游令牌 |
| 会话感知路由（Session-aware routing） | "多后端会话" | 网关将 N 个后端会话多路复用到每个开发人员会话 |
| 工具哈希固定（Tool-hash pinning） | "已批准清单" | 每个已批准工具描述的 SHA256；在中心阻止拉地毯攻击 |
| RBAC | "逐用户策略" | 针对工具和服务器的角色访问控制 |
| 策略即代码（Policy-as-code） | "声明式规则" | 在网关强制执行的 OPA/Rego、Kyverno、Styra 策略 |
| 审计日志（Audit log） | "谁、何时、何事" | 合规所需的只追加事件日志 |
| 限速（Rate limit） | "逐用户令牌桶" | 防止滥用的每分钟上限 |
| 官方 MCP 注册表（Official MCP Registry） | "规范上游" | `registry.modelcontextprotocol.io`，命名空间已验证 |
| 反向 DNS 命名（Reverse-DNS naming） | "注册表命名空间" | `io.github.user/server` 约定 |

## 延伸阅读

- [官方 MCP 注册表](https://registry.modelcontextprotocol.io/) — 规范上游，命名空间已验证
- [Cloudflare — 企业 MCP](https://blog.cloudflare.com/enterprise-mcp/) — 带 OAuth 和策略的网关模式
- [agentic-community — MCP 网关注册表](https://github.com/agentic-community/mcp-gateway-registry) — 开源参考网关
- [TrueFoundry — 什么是 MCP 网关？](https://www.truefoundry.com/blog/what-is-mcp-gateway) — 功能比较文章
- [IBM — MCP 上下文熔炉](https://github.com/IBM/mcp-context-forge) — IBM 的企业网关