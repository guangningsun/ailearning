# 安全——密钥、API 密钥轮换、审计日志与防护栏

> 通过集中式密钥库（HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）消除密钥蔓延。绝不将凭证存储在配置文件、Git 中的 env 文件或电子表格中。优先使用 IAM 角色而非静态密钥；CI/CD 使用 OIDC。2026 年的解决方案是 AI 网关模式：应用 → 网关 → 模型提供商，网关在运行时从密钥库获取凭证。在密钥库中轮换，所有应用在几分钟内自动生效——无需重新部署，无需在 Slack 上问"谁有新密钥"。轮换策略不超过 90 天；每次提交时使用 TruffleHog / GitGuardian / Gitleaks 扫描。零信任：MFA、SSO、RBAC/ABAC、短效令牌、设备态势。PII 清理使用实体识别在转发前遮蔽 PHI/PII；一致的令牌化（Mesh 方案）将敏感值映射到稳定的占位符，使 LLM 保留代码/关系语义。网络出口：LLM 服务置于专用 VPC/VNet 子网，仅允许白名单中的 `api.openai.com`、`api.anthropic.com` 等域名；屏蔽所有其他出口。2026 年的事件驱动因素：Vercel 供应链攻击——通过被入侵的 CI/CD 凭证在数千个客户部署中窃取 env vars。

**类型：** 学习型
**语言：** Python（标准库 + 演示用 PII 清理器 + 审计日志写入器）
**前置条件：** 阶段 17 · 19（AI 网关），阶段 17 · 13（可观测性）
**时间：** 约 60 分钟

## 学习目标

- 列举四种密钥管理的反模式（VCS 中的配置文件、硬编码 env、电子表格、静态密钥）并说出其替代方案。
- 解释 AI 网关从密钥库拉取模式作为 2026 年生产标准。
- 实现带有一致令牌化（相同值 → 相同占位符）的 PII 清理器，以保留语义。
- 说出 2026 年 Vercel 供应链事件及其对 CI/CD 凭证卫生的教训。

## 问题

一位实习生将包含 API 密钥的 `.env` 提交了。很快又删除了。但密钥已经在 Git 历史中——GitGuardian 扫描发现了它，你的轮换流程是"在 Slack 上通知团队、更新 40 个配置文件、重新部署所有服务。"8 小时后，一半服务已上线，另一半还在等待部署窗口。

另一方面，用户提示中包含"我的 SSN 是 123-45-6789。"提示被发送到了 OpenAI。你有 BAA，但内部策略要求在转发前遮蔽 PII。你没有。

再另一方面，你的 EKS 集群中 LLM pod 可以访问任何互联网主机。某人通过 DNS 查询将数据泄露到攻击者控制的域名。没有任何东西屏蔽它。

LLM 服务的安全必须同时解决这三个向量。密钥库支持的凭证。PII 清理。网络出口过滤。审计日志。

## 概念

### 集中式密钥库 + IAM 角色拉取

**密钥库**：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。单一真相来源。

**IAM 角色**：应用/网关通过其 IAM 身份进行认证，而非静态密钥。密钥库在令牌生命周期内返回密钥。

**AI 网关模式**：网关在请求时从密钥库拉取 `OPENAI_API_KEY`。在密钥库中轮换；下一次请求获取新密钥。无需重新部署。

### 轮换策略 ≤ 90 天

所有 API 密钥、密钥库根令牌、CI/CD 凭证。尽可能自动化轮换。手动轮换需记录和跟踪。

### 密钥扫描

- **TruffleHog** — 对提交进行正则 + 熵检测。
- **GitGuardian** — 商业级，高准确率。
- **Gitleaks** — 开源，在 CI 中运行。

每次提交时运行。如检测到新密钥，则阻止 PR。

### 零信任态势

- 所有账户必须使用 MFA。
- 通过 SAML/OIDC 实现 SSO。
- RBAC（基于角色）或 ABAC（基于属性）实现细粒度访问。
- 短效令牌（以小时计，而非天）。
- 设备态势——仅限启用了磁盘加密的公司设备。

### PII / PHI 清理

在提示离开你的基础设施之前：

1. 实体识别（spaCy NER、Presidio、商业方案）。
2. 遮蔽匹配的实体：`"我的 SSN 是 123-45-6789"` → `"我的 SSN 是 [SSN_TOKEN_A3F]"`.
3. 一致的令牌化（Mesh 方案）：相同值映射到相同占位符，使 LLM 保留关系。
4. 可选：LLM 响应的反向映射。

静态正则过滤器捕获基本模式；NER 捕获更多。两者兼用。

### 输入 + 输出防护栏

输入：阻止已知的越狱攻击、禁止话题；按用户限速。

输出：正则扫描泄露的密钥（API 密钥模式、拒绝上下文中的邮箱模式）、分类器检测策略违规。

### 网络出口白名单

LLM 服务位于专用子网：
- 白名单：`api.openai.com`、`api.anthropic.com`、向量数据库端点、密钥库端点。
- 其他所有：一律丢弃。
- DNS 通过仅允许列表解析器（避免 DNS 隧道泄露）。

### 审计日志

每条 LLM 调用的不可变日志，包含：
- 时间戳。
- 用户 / 租户。
- 提示哈希（出于隐私考虑，不记录原始提示）。
- 模型 + 版本。
- 令牌计数。
- 成本。
- 响应哈希。
- 任何防护栏触发。

按监管要求保留（SOC 2 保留 1 年，HIPAA 保留 6 年）。

### 2026 年 Vercel 事件

供应链攻击：被入侵的 CI/CD 凭证在数千个客户部署中窃取 env vars。教训：CI/CD 凭证等同于生产凭证。存储在密钥库中。范围要窄。轮换要激进。

### 需记住的数字

- 轮换策略：≤ 90 天。
- 每次提交时扫描：TruffleHog / GitGuardian / Gitleaks。
- Vercel 2026：CI/CD 凭证被入侵 → 数以千计的客户 env vars 泄露。
- 审计日志保留：SOC 2 = 1 年，HIPAA = 6 年。

## 使用它

`code/main.py` 实现了一个带有一致令牌化的演示 PII 清理器和一个仅追加的审计日志。

## 交付它

本课产出 `outputs/skill-llm-security-plan.md`。给定监管范围和现状，规划密钥库迁移、清理器、出口、审计日志。

## 练习

1. 运行 `code/main.py`。发送两条引用相同 SSN 的提示。确认两者获得相同的占位符。
2. 为调用 OpenAI + Anthropic + Weaviate 的 vLLM-on-EKS 部署设计网络出口策略。
3. 你在 Git 历史中发现了密钥（2 年前）。正确响应是什么——轮换密钥、清理历史，还是两者都要？给出理由。
4. 你的审计日志每天增长 10 GB。设计保留层级（热数据 30 天，暖数据 12 个月，冷数据 6 年）。
5. 论证反向令牌化（将真实值替换回 LLM 响应）是否值得其复杂性，还是保持占位符可见更好。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| 密钥库 (Vault) | "密钥存储" | 集中式凭证管理服务 |
| IAM 角色 | "基于身份的身份验证" | 应用所扮演的角色；返回短效凭证 |
| CI/CD 用 OIDC | "云签发令牌" | CI 中无静态密钥——通过 OIDC 实现身份 |
| TruffleHog / GitGuardian / Gitleaks | "密钥扫描器" | 提交时密钥检测 |
| RBAC / ABAC | "访问控制" | 基于角色 vs 基于属性 |
| PII 清理 | "数据遮蔽" | 移除或令牌化敏感实体 |
| 一致令牌化 | "稳定占位符" | 相同值 → 每次相同的令牌 |
| Mesh 方案 | "Mesh 令牌化" | 保留语义的令牌化模式 |
| 出口白名单 | "出口允许列表" | 仅允许可达的域名 |
| 审计日志 | "不可变历史" | 用于合规的仅追加记录 |

## 延伸阅读

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — PII 检测与匿名化。
- [HashiCorp Vault docs](https://developer.hashicorp.com/vault/docs)