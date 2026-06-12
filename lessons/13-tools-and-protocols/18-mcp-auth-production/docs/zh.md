# MCP 生产环境认证 —— 注册、JWKS 刷新与受众绑定令牌

> 第 16 课在内存中搭建了 OAuth 2.1 状态机。到 2026 年，你交付给真实组织的每一个 MCP 服务器都运行在生产级认证之后：可扩展到无限客户端群体的客户端注册（先 Client ID Metadata Documents，动态客户端注册作为向后兼容的备选方案）、授权服务器元数据发现（RFC 8414 *或* OpenID Connect Discovery）、不打断凌晨 3 点令牌验证的 JWKS 缓存刷新，以及拒绝跨资源重放的受众绑定令牌。本课用三个角色建模完整表面——授权服务器、资源服务器（MCP 服务器）和客户端——以便你追踪从发现到验证工具调用的每一个跃点。
>
> **规范备注（2025-11-25）：** 2025 年 11 月的 MCP 授权规范将动态客户端注册从 `SHOULD` 降级为 `MAY`，并将 **Client ID Metadata Documents (CIMD)** 设为推荐的默认注册机制。本课按规范优先级顺序教授两者，代码保留 DCR 的完整演练，因为它在一个进程内是完全自包含的。

**类型：** 构建
**语言：** Python（标准库）
**前置条件：** 阶段 13 · 16（OAuth 2.1 状态机）、阶段 13 · 17（网关）
**时间：** 约 90 分钟

## 学习目标

- 通过 RFC 8414 元数据发现授权服务器并验证契约。
- 实现 RFC 7591 动态客户端注册，使 MCP 客户端无需管理员介入即可注册。
- 按计划缓存和刷新 JWKS 密钥，使签名验证在密钥轮换时不受影响。
- 使用 RFC 8707 资源指示器将令牌绑定到单个 MCP 资源，并拒绝糊涂代理重用。
- 清晰分离三个角色——授权服务器、资源服务器、客户端——使每个角色仅执行属于它的检查。
- 阅读 IdP 能力矩阵，并在 IdP 无法满足 MCP 认证配置时拒绝部署。

## 问题背景

第 16 课的模拟器在内存中运行 OAuth 2.1。生产环境有三个内存模拟器看不到的操作缺口。

第一个缺口是注册。真实组织运行着数百个 MCP 服务器和数千个 MCP 客户端。运维人员不会手动将每个 Cursor 用户注册为 OAuth 客户端。2025-11-25 规范为客户端提供了解决此问题的优先级顺序：若有预注册的 `client_id` 则使用它，否则使用 **Client ID Metadata Document**（客户端使用其控制的 HTTPS URL 作为标识，授权服务器*拉取*元数据），否则回退到 **RFC 7591 动态客户端注册**（客户端*推送* `POST /register` 并立即获得 `client_id`），否则提示用户。CIMD 是推荐的默认方案，因为它完全取消了每个服务器的注册工作，同时保留了 DNS 根信任模型；保留 DCR 是为了向后兼容。两者都从授权服务器的元数据中发现入口：`client_id_metadata_document_supported` 用于 CIMD，`registration_endpoint` 用于 DCR。

第二个缺口是密钥轮换。JWT 验证依赖于授权服务器的签名密钥，以 JSON Web Key Set（JWKS）形式发布。授权服务器按计划轮换这些密钥（通常每小时一次，在事件响应期间有时更快）。MCP 服务器若仅在启动时获取一次 JWKS，在轮换窗口之前验证正常——之后每个请求都会失败，直到重启。生产环境将 JWKS 接为缓存值，并在上一个密钥过期前用刷新作业覆盖缓存，再加上缓存未命中时的后备获取（用于处理由比缓存更新的密钥签名的令牌到达的情况）。

第三个缺口是受众绑定。第 16 课介绍了 RFC 8707 资源指示器。在生产环境中，该指示器成为每个请求上的硬断言检查。MCP 服务器将自己的规范资源 URL 与 `token.aud` 进行比较，不匹配时以 HTTP 401 拒绝。这是唯一能防御上游 MCP 服务器（或持有针对某一服务器令牌的恶意客户端）在同一信任网状结构中将该令牌重放给另一服务器的防线。

本课将每个缺口映射到表面的具体部分。元数据文档是一个 HTTP 端点。JWKS 缓存刷新是计划作业加键值缓存。JWT 验证是资源服务器在分派任何工具前运行的例程。保持三个角色分离，每个角色仅执行它拥有的检查：授权服务器签发和轮换密钥，资源服务器缓存和验证，客户端发现和注册。

## 核心概念

### RFC 8414 —— OAuth 授权服务器元数据

位于 `/.well-known/oauth-authorization-server` 的文档描述了客户端所需的一切：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

客户端给定 MCP 资源 URL 后链式发现：RFC 9728 的 `oauth-protected-resource`（资源服务器的文档）命名了 issuer，然后 `oauth-authorization-server`（本 RFC）命名了每个端点。客户端永不硬编码授权 URL。

在信任 IdP 用于 MCP 之前你要验证的契约：

- `code_challenge_methods_supported` 包含 `S256`（RFC 7636 规定的 PKCE）。规范明确：若此字段**不存在**，授权服务器不支持 PKCE，客户端**必须**拒绝继续。
- `grant_types_supported` 包含 `authorization_code`，且拒绝 `password` 和 `implicit`。
- 至少公布了一条注册路径：`client_id_metadata_document_supported: true`（CIMD，优先）**或** `registration_endpoint`（RFC 7591 DCR，备选）。任一均满足契约；你不再硬性要求 DCR。
- `response_types_supported` 对于 OAuth 2.1 正好是 `["code"]`。

若缺少 `S256`，MCP 服务器拒绝部署到此 IdP——PKCE 没有降级模式。若**既没有**公布注册路径且你没有预注册的 `client_id`，你也无法注册；部署清单有问题，而非代码。

### RFC 9728（回顾）——受保护资源元数据

第 16 课覆盖了 RFC 9728。生产环境的差异：此文档是客户端查找*此* MCP 服务器信任的授权服务器的唯一位置。单个 MCP 服务器可能接受来自多个 IdP 的令牌（一个给员工，一个给合作伙伴）。RFC 9728 声明该集合；RFC 8414 记录每个 IdP 支持的内容。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### Client ID Metadata Documents（推荐的默认方案）

CIMD 将注册从*推送*反转 为*拉取*。客户端不使用授权服务器来创建 `client_id`，而是使用它控制的 HTTPS URL **作为** 其 `client_id`。该 URL 解析为一个 JSON 元数据文档；授权服务器在 OAuth 流程中按需获取。信任根植于 DNS：若服务器运营商信任 `app.example.com`，它就信任来自 `https://app.example.com/client.json` 的客户端。无注册往返、无 `client_id` 命名空间耗尽、无需在多个服务器间同步的状态。

客户端托管的元数据文档：

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

文档中的 `client_id` 值**必须**等于它所服务的 URL（授权服务器验证此点；不匹配将被拒绝）。授权服务器通过其 RFC 8414 元数据中的 `client_id_metadata_document_supported: true` 来公布支持。

规范明确指出两个安全事实：

- **SSRF。** 授权服务器获取攻击者提供的 URL。它必须防御服务器端请求伪造（不获取内部/管理端点）。
- **localhost 冒充。** 仅靠 CIMD 无法阻止本地攻击者声明合法客户端的元数据 URL 并绑定任意 `localhost` 重定向。授权服务器**必须**在授权时清晰显示重定向 URI 主机名，**应该**警告仅 `localhost` 的重定向。

因为 CIMD 无需服务器端状态，所以无需像 DCR 那样搭建注册机构。客户端侧是只读的：从静态 HTTPS 端点提供元数据文档，让授权服务器来拉取。

### RFC 7591 —— 动态客户端注册（备选 / 向后兼容）

DCR 现为 `MAY`，为与 2025-11-25 之前的部署以及尚不支持 CIMD 的 IdP 向后兼容而保留。没有它（也没有 CIMD 或预注册），每个 MCP 客户端（Cursor、Claude Desktop、自定义 agent）都需要与 IdP 管理员进行一次带外交换。有了 DCR，客户端发送：

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

服务器响应 `client_id` 和用于后续更新的 `registration_access_token`：

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`token_endpoint_auth_method: none` 是运行在用户设备上的 MCP 客户端的正确默认值。它们只获得一个 `client_id`——没有可被窃取的 `client_secret`。PKCE 提供公开客户端所需的存在证明。

三个生产环境陷阱：

- 注册端点必须按源 IP 限速。否则，恶意行为者可以脚本化数百万个虚假注册，耗尽 `client_id` 命名空间。在注册机构处理请求前运行限速检查。
- 某些企业 IdP 要求 `software_statement`（为客户端担保的签名 JWT）。本课的模拟跳过了它；生产环境需要验证步骤，拒绝来自非 localhost 重定向 URI 的未签名注册。
- `registration_access_token` 必须以哈希形式存储，而非明文。若此令牌被盗，攻击者可以重写客户端的重定向 URI。

### RFC 8707（回顾）——资源指示器

第 16 课确立了形态。生产规则：每个令牌请求都包含 `resource=<canonical-mcp-url>`，MCP 服务器在每次调用时验证 `token.aud` 与其自身的资源 URL 匹配。规范 URI 是服务器的最特定标识符：它使用小写方案和主机名，无片段，按惯例无尾部斜杠。路径组件**不是**按规则剥离的——当需要用于标识单个 MCP 服务器时，规范保留它。`https://mcp.example.com`、`https://mcp.example.com/mcp`、`https://mcp.example.com:8443` 和 `https://mcp.example.com/server/mcp` 都是有效的规范 URI。为每个服务器选一个，并将 `aud` 精确绑定到该 URI。（本课的模拟使用裸主机受众如 `https://notes.example.com` 以简洁；在同一来源下共托管多个 MCP 服务器的部署通过路径来区分它们。）

### RFC 7636（回顾）——PKCE

PKCE 在 OAuth 2.1 中是强制的。本课的授权码流程始终携带 `code_challenge` 和 `code_verifier`。服务器拒绝没有验证器或验证器哈希不等于存储挑战的令牌请求。

### MCP 规范 2025-11-25 认证配置

MCP 规范（2025-11-25）对 MCP 服务器的授权层必须做什么有精确规定：

- 实现 RFC 9728 受保护资源元数据，并通过 `WWW-Authenticate: Bearer resource_metadata="..."` 头（401 时）**或** well-known URI `/.well-known/oauth-protected-resource`（SEP-985 使头成为可选的，带 well-known 回退）提供其位置。元数据的 `authorization_servers` 字段**必须**命名至少一个服务器。
- 仅通过 `Authorization: Bearer ...` 在**每个**请求上接受令牌——永不在查询字符串中，永不在会话开始时仅验证一次。
- 每次请求时验证 `aud`、`iss`、`exp` 和所需作用域。服务器**必须**验证令牌是专门为其签发的（受众）；拒绝缺失或不匹配的 `aud`，从不将其视为通配符。
- 401/403 时，返回 `WWW-Authenticate: Bearer`，携带 `error=...`、`resource_metadata="<PRM-URL>"` 参数（元数据文档的 URL，**而非**裸资源）和 `insufficient_scope`（403）时的 `scope="..."`。注意：参数是 `resource_metadata`，是指向发现的指针——挑战中没有 `resource` 参数。
- 授权服务器发现接受**任一** RFC 8414 OAuth 元数据**或** OpenID Connect Discovery 1.0；客户端必须按优先级顺序尝试两个 well-known 后缀。

- 客户端（而非服务器）可防御**混合攻击**：它在重定向前记录预期的 `issuer`，并在兑换授权码之前验证授权响应中的 `iss` 参数（RFC 9207）。仅靠 PKCE 无法阻止混合攻击，因为客户端会将其 `code_verifier` 交给所指向的令牌端点。

OAuth 2.1 草案是底层实现；RFC 8414/7591/8707/9728/9207 + RFC 7636 + CIMD 是表面层；MCP 规范是配置文件。

### IdP 功能矩阵

并非每个 IdP 都支持完整的 MCP 配置。以下矩阵记录了截至 2025-11-25 规范的功能声明。这是一个**部署门槛**，而非推荐建议。

CIMD 在 2025-11-25 规范中发布，底层 OAuth 草案直到 2025 年 10 月才被采纳，因此厂商支持仍在陆续到来——将下面的"CIMD"视为"当前状态，请在自己的租户中验证"，而非永久性结论。

| IdP 类别 | AS 元数据（8414/OIDC） | CIMD | RFC 7591 DCR | RFC 8707 资源 | RFC 7636 S256 PKCE | 备注 |
|---|---|---|---|---|---|---|
| 自托管（Keycloak） | yes | emerging | yes | yes（自 24.x 起） | yes | MCP 配置文件的参考 IdP；完整 DCR 路径端到端，CIMD 跟随新规范。 |
| 企业 SSO（Microsoft Entra ID） | yes | emerging | yes（高级套餐） | yes | yes | DCR 可用性因租户套餐而异；部署前在目标租户中验证。 |
| 企业 SSO（Okta） | yes | emerging | yes（Okta CIC / Auth0） | yes | yes | DCR 在 Auth0（现为 Okta CIC）上可用；经典 Okta 组织需要管理员预注册。 |
| 社交登录 IdP（通用） | varies | no | rarely | rarely | yes | 大多数社交 IdP 将客户端视为静态合作伙伴；无自助注册服务。仅作为身份源使用，在其上层架设您自己的 MCP 感知授权服务器。 |
| 自定义/自研 | depends | depends | depends | depends | depends | 如果自建，请实现完整配置文件并优先使用 CIMD。跳过 PKCE 或受众绑定会破坏 MCP 认证契约。 |

部署清单的拒绝规则：如果所选 IdP 未在 `code_challenge_methods_supported` 中列出 `S256`，则 MCP 服务器拒绝启动——PKCE 没有降级模式。注册是一个更宽松的门槛：您需要一条可行路径（预注册的 `client_id`、`client_id_metadata_document_supported: true` 或 `registration_endpoint`）。DCR 的缺失本身不再是拒绝触发因素，因为 CIMD 或预注册可以覆盖它。

### JWKS 刷新模式（在 AS 端轮换，在资源服务器端刷新）

将两个动词区分清楚，因为混用它们是一个真实的生产 bug：

- **轮换（Rotate）** 是**授权服务器**执行的操作：生成新的签名密钥，将其发布到 JWKS，稍后淘汰旧密钥。资源服务器不参与此过程，也无法执行——它不持有 IdP 的私钥。
- **刷新（Refresh）** 是**资源服务器**执行的操作：重新 `GET` 发布的 JWKS 到其缓存中。这是资源服务器唯一执行的 JWKS 操作。

生产环境的故障模式是缓存过期。通过定时刷新作业加键值缓存来解决。资源服务器运行一个作业（cron、timer，或您运行时环境提供的任何机制），按固定间隔获取 `<issuer>/.well-known/jwks.json` 并覆盖 `cache[issuer] = {keys, fetched_at}`。验证器从缓存中读取。当令牌的 `kid` 在缓存中缺失时，触发**一次**同步刷新作为后备，然后重新检查。这同时处理两种情况：定时刷新，以及在新密钥签发的令牌到达时早于下次定时刷新而出现的密钥重叠窗口。

后备方案**必须是重新获取，而不是轮换**。如果您将缓存未命中路径连接到轮换和生成，将有两处被破坏：（1）生成新密钥会产生一个仍然与令牌不匹配的 `kid`，因此查找仍然失败；（2）攻击者喷射带有随机 `kid` 值的令牌，迫使其无限制地创建密钥——一种自我造成的 DoS。重新获取是幂等的，因此伪造的 `kid` 最多只会造成一次浪费的获取。

缓存结构：

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

同时持有两个密钥是稳态。授权服务器通过先引入下一个密钥（`k_2026_04`）再淘汰上一个（`k_2026_03`）来进行轮换，因此使用旧密钥签发的令牌在过期前仍然有效。缓存持有它们的并集；验证器按 `kid` 选择。

### 验证流程

MCP 服务器在分派任何工具前运行验证。`code/main.py` 使用的形式：

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate` 解码 JWT，从 JWKS 缓存解析签名密钥（未命中时刷新一次），验证签名，然后检查 `iss` 是否在允许列表中、`aud` 是否与此服务器的规范资源匹配、`exp`，以及所需范围——在首次失败时返回 `WWW-Authenticate` challenge。将验证保持为资源服务器上的单一流程，意味着每个入口点（每个工具调用、每个传输）都经过相同的检查；没有路径可以在不先验证的情况下到达工具。

### 受众-重放攻击演练（访问令牌权限限制）

服务器 A（`notes.example.com`）和服务器 B（`tasks.example.com`）都向同一个授权服务器注册。服务器 A 被攻陷。攻击者获取用户的笔记令牌，并在服务器 B 上重放它。

服务器 B 的验证器：

1. 解码 JWT，按 `kid` 获取 JWKS，验证签名。
2. 检查 `iss` 是否在其受保护资源元数据的 `authorization_servers` 中。（通过——同一 IdP。）
3. 检查 `aud == "https://tasks.example.com"`。（失败——令牌的 `aud` 是 `https://notes.example.com`。）
4. 返回 401，附带 `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="https://tasks.example.com/.well-known/oauth-protected-resource"`。

受众声明是协议层防御此攻击的唯一手段。为了性能而跳过它是最常见的生产错误；验证器必须在每个请求上运行，而不仅仅在会话开始时。规范将此称为**访问令牌权限限制**：MCP 服务器**必须**拒绝任何未在受众中命名它的令牌。

> **命名说明。** 规范保留术语*混乱代理*用于一个相关但不同的问题：一个 MCP 服务器作为 OAuth **代理**连接到第三方 API，使用静态客户端 ID，转发令牌而未获得每个客户端的用户同意。受众绑定可修复上述重放攻击；混乱代理的修复是每个客户端同意**加上**从不将入站令牌传递给上游 API（MCP 服务器**必须**获取自己的独立上游令牌）。

### 混合攻击（客户端防御，服务器无法提供）

客户端在一生中会与多个授权服务器通信。恶意 AS 可以试图让客户端在攻击者的令牌端点上兑换诚实 AS 的授权码。受众绑定在此无济于事——攻击在任何令牌存在之前就发生了。防御位于客户端（RFC 9207）：

1. 重定向前，客户端从已验证的 AS 元数据中记录预期的 `issuer`。
2. 在授权响应上，客户端将返回的 `iss` 参数与记录的发行者进行比较（简单字符串比较，无规范化），然后才将代码发送到任何地方。
3. 不匹配（或当 AS 声明了 `authorization_response_iss_parameter_supported` 时 `iss` 缺失）→ 拒绝，甚至不显示 `error` 字段。

仅靠 PKCE 无法阻止混合攻击，因为客户端会将其 `code_verifier` 交给所指向的令牌端点。这就是规范在与 PKCE 验证器和 `state` 相同的位置按请求记录发行者的原因。

### 故障模式

- **过期的 JWKS。** 验证器在 AS 轮换密钥后拒绝有效令牌。修复方法是上述的 cron 刷新 + 缓存未命中重新获取模式。永远不要在没有刷新作业的情况下缓存 JWKS。
- **将轮换作为后备。** 将缓存未命中路径连接到轮换和生成而不是重新获取是一个真实 bug：它永远不会产生缺失的 `kid`，并且会将攻击者控制的 `kid` 值变成密钥创建 DoS。后备方案必须是幂等的 `refresh-jwks`。
- **缺少 `aud` 声明。** 某些 IdP 默认省略 `aud`，除非令牌请求中存在 `resource`。验证器必须拒绝缺少 `aud` 的令牌，而不是将缺失视为通配符。
- **通过缺少 `iss` 检查的混合攻击。** 如果客户端不在重定向前验证 RFC 9207 `iss` 授权响应参数与记录的发行者匹配，可能会被引导到在攻击者的令牌端点上兑换诚实 AS 的代码。这是客户端故障；资源服务器无法弥补。
- **范围升级竞争。** 同一用户的两个并发升级流程都可能成功，并产生具有不同范围的两个访问令牌。验证器必须使用请求上呈现的令牌，而不是查找"用户的当前范围"——这会产生 TOCTOU 窗口。
- **注册令牌被盗。** 泄露的 `registration_access_token` 让攻击者可以重写重定向 URI。在静态存储时对这些进行哈希；要求客户端每次更新时提供明文；在怀疑时进行轮换。
- **`iss` 未固定。** 接受任何 `iss` 的验证器让攻击者可以架设自己的授权服务器，为目标受众注册客户端，并颁发令牌。受保护资源元数据中的 `authorization_servers` 列表是允许列表；必须强制执行它。

## Use It

`code/main.py` 通过纯标准库 Python 和三个角色——`AuthorizationServer`、`ResourceServer` 和 `Client`——演练完整的生产流程。流程：

1. 授权服务器在 `/.well-known/oauth-authorization-server` 发布 RFC 8414 元数据。
2. MCP 客户端调用元数据端点并检查其注册选项（`client_id_metadata_document_supported` 用于 CIMD，`registration_endpoint` 用于 DCR）和 `S256` PKCE 支持。
3. 演练采用 DCR 后备路径：客户端向 `/register` 发起 POST（RFC 7591）并接收 `client_id`。（CIMD 客户端将改为呈现自己的 HTTPS `client_id` URL 并跳过此步骤。）
4. MCP 客户端使用 `resource` 指示符（RFC 8707）运行 PKCE 保护的授权码流程（RFC 7636）。
5. MCP 客户端使用 `Authorization: Bearer ...` 调用 MCP 服务器上的工具。
6. MCP 服务器运行 `validate`，从 JWKS 缓存解析签名密钥。
7. IdP 轮换密钥；定时的 `refresh_jwks` 重新拉取 JWKS 到缓存中。
8. 下一个调用针对刷新的密钥进行验证，无需重启，并且在重叠窗口期间旧令牌仍然可以验证。
9. 针对不同 MCP 资源的受众-重放尝试会收到 401，附带 `audience mismatch` 和 `resource_metadata` 指针。

这里的 JWT 使用带共享密钥的 HS256（因此本课程仅使用纯标准库即可运行）。生产环境使用 RS256 或 EdDSA 以及上述 JWKS 模式；验证逻辑在其他方面相同。因为 IdP 和资源服务器在一个进程中，`refresh_jwks` 直接读取授权服务器的密钥列表；在网络上，它是对 `jwks_uri` 的 HTTP `GET`。

## Ship It

本课程生成 `outputs/skill-mcp-auth.md`。给定 MCP 服务器配置和 IdP 功能集，该 skill 会发出认证表面以进行架设——受保护资源元数据、要使用的注册路径（CIMD、预注册或 DCR 后备）、JWKS 刷新计划、范围映射，以及当 IdP 不支持完整 RFC 配置文件时要应用的拒绝规则。

## Exercises

1. 运行 `code/main.py`。追踪流程。注意 IdP 在步骤 6 中轮换密钥，定时的 `refresh_jwks` 重新拉取发布的集合，以及旧令牌（重叠窗口）和新令牌都无需重启即可验证。

2. 向受保护资源元数据的 `authorization_servers` 列表添加一个新的 IdP。颁发一个由新 IdP 签名的令牌，并确认验证器接受它。颁发一个由未列出 IdP 签名的令牌，并确认验证器拒绝，附带 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`。

3. 向 `register_client` 添加速率限制检查，在注册机构接受请求之前运行。使用按源 IP 分组的令牌桶，存储在以 IP 为键的小型字典中。

4. 阅读 RFC 7591 并找出课程 `/register` 处理程序未验证的两个字段。添加验证。（提示：`software_statement` 和 `redirect_uris` URI 方案。）

5. 添加客户端 ID 元数据文档路径。服务一个 `client.json`，其 `client_id` 等于其自己的 URL，并让授权服务器获取并验证它（如果 `client_id` ≠ URL 则拒绝）。确认 CIMD 客户端无需调用 `register_client` 即可注册。

6. 证明 DoS 修复。向验证器发送带有随机 `kid` 的令牌，并确认 `refresh_jwks` 最多运行一次且授权服务器的密钥数量不会增长。然后故意将后备重新连接到轮换和生成，并观察密钥数量随着伪造令牌攀升——之后恢复重新获取。

7. 实现客户端 RFC 9207 `iss` 检查（来自混合攻击部分）：在授权请求前记录预期的发行者，然后拒绝 `iss` 不匹配的授权响应。

## Key Terms

| 术语 | 人们通常说 | 实际含义 |
|------|----------------|------------------------|
| ASM | "OAuth 元数据文档" | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| CIMD | "客户端元数据 URL" | 客户端 ID 元数据文档——用作 `client_id` 的 HTTPS URL；AS 拉取 JSON。自 2025-11-25 起推荐的默认方案 |
| DCR | "自助客户端注册" | RFC 7591 `POST /register` 流程；在 2025-11-25 中降级为 `MAY` 后备方案 |
| JWKS | "用于 JWT 验证的公钥" | JSON Web Key Set，从 `jwks_uri` 获取，按 `kid` 索引 |
| 轮换 vs 刷新 | "更新密钥" | *轮换* = AS 生成/淘汰签名密钥；*刷新* = 资源服务器重新获取发布的集合。资源服务器只能刷新 |
| 资源指示符 | "受众参数" | RFC 8707 `resource` 参数将令牌固定到一台服务器 |
| `aud` 声明 | "受众" | 验证器与规范资源 URL 比较的 JWT 声明 |
| 受众重放 | "令牌重放" | 为服务器 A 签发但在服务器 B 上呈现的令牌；通过受众验证进行防御（规范：访问令牌权限限制） |
| 混乱代理 | "代理令牌误用" | 具有静态客户端 ID 转发令牌而未经每个客户端同意的 MCP 代理；与受众重放不同 |
| 混合攻击 | "错误的令牌端点" | 客户端被引导在攻击者端点上兑换诚实 AS 的代码；通过 RFC 9207 `iss` 在客户端进行防御 |
| `iss` 允许列表 | "可信授权服务器" | 受保护资源元数据 `authorization_servers` 中命名的集合 |
| `resource_metadata` | "在哪里找到 PRM 文档" | 401/403 上命名 RFC 9728 元数据 URL 的 `WWW-Authenticate` 参数 |
| 公共客户端 | "原生或浏览器客户端" | 没有 `client_secret` 的 OAuth 客户端；PKCE 补偿 |
| `WWW-Authenticate` | "401/403 响应头" | 携带驱动客户端恢复的 `Bearer error=...` 指令 |

## Further Reading

- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) — 本课程实现的 MCP 认证配置文件
- [MCP blog — One Year of MCP: November 2025 Spec Release](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 2025-11-25 的变化（CIMD、XAA、DCR 降级）
- [Aaron Parecki — Client Registration in the November 2025 MCP Authorization Spec](https://aaronparecki.com/2025/11/25/1/mcp-authorization-spec-update) — CIMD 优于 DCR 的理由
- [OAuth Client ID Metadata Document (draft-ietf-oauth-client-id-metadata-document-00)](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-client-id-metadata-document-00) — CIMD
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — 发现契约
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) — DCR（后备路径）
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — 公共客户端持有证明
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — 受众固定
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — 资源服务器发现
- [RFC 9207 — OAuth 2.0 Authorization Server Issuer Identification](https://datatracker.ietf.org/doc/html/rfc9207) — 用于防御混合攻击的 `iss` 参数
- [OAuth 2.1 draft](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — 合并的 OAuth 底层实现