# AI 网关 — LiteLLM、Portkey、Kong AI Gateway、Bifrost

> 网关位于你的应用和模型提供商之间。核心功能包括提供商路由、回退、重试、限速、密钥引用、可观测性和护栏。2026 年市场格局：**LiteLLM** 是 MIT 开源项目，支持 100+ 提供商、OpenAI 兼容，但在 ~2000 RPS 时会出现问题（8 GB 内存，在公开基准测试中出现级联故障）；最适合 Python、<500 RPS、开发/原型阶段。**Portkey** 主打控制平面定位（护栏、PII 重删、越狱检测、审计追踪），2026 年 3 月以 Apache 2.0 开源，延迟开销 20-40 ms，生产版 $49/月。**Kong AI Gateway** 构建在 Kong Gateway 之上——Kong 自测（相同 12 CPU）：比 Portkey 快 228%，比 LiteLLM 快 859%；定价 $100/模型/月（Plus 方案最多 5 个）；如果你已在使用 Kong，则是企业级选择。**Bifrost**（Maxim AI）—— 自动重试配可配置退避，OpenAI 429 时回退到 Anthropic。**Cloudflare / Vercel AI Gateway** —— 托管、零运维、基本重试。数据驻留决定了自托管的选择；Portkey 和 Kong 处于中间位置，开源+可选托管。

**类型：** 学习型
**语言：** Python（标准库、自制网关路由模拟器）
**前置条件：** 阶段 17 · 01（托管 LLM 平台）、阶段 17 · 16（模型路由）
**时间：** 约 60 分钟

## 学习目标

- 列举六个核心网关功能（路由、回退、重试、限速、密钥、可观测性、护栏）。
- 将四款 2026 年网关（LiteLLM、Portkey、Kong AI、Bifrost）映射到规模上限和适用场景。
- 引用 Kong 基准测试（比 Portkey 快 228%，比 LiteLLM 快 859%），并解释这对 >500 RPS 的重要性。
- 根据数据驻留和运维预算，选择自托管还是托管。

## 问题

你的产品调用 OpenAI、Anthropic 和自托管的 Llama。每个提供商有不同的 SDK、错误模型、限速和认证方案。你需要故障转移（如果 OpenAI 429，尝试 Anthropic）、统一的凭证存储、统一的可观测性，以及按租户的限速。

在应用层重新发明这些会把每个服务耦合到每个提供商。网关层将其合并为一个进程、一个 API（通常兼容 OpenAI），再分发到各个提供商。

## 概念

### 六个核心功能

1. **提供商路由** —— 通过一个 API 对接 OpenAI、Anthropic、Gemini、自托管等。
2. **回退** —— 遇到 429、5xx 或质量问题时，在其他地方重试。
3. **重试** —— 指数退避，有限次尝试。
4. **限速** —— 按租户、按密钥、按模型。
5. **密钥引用** —— 运行时从金库拉取凭证（绝不放在应用里）。
6. **可观测性** —— OTel + GenAI 属性（阶段 17 · 13）+ 成本归因。
7. **护栏** —— PII 重删、越狱检测、允许主题过滤。

### LiteLLM — MIT 开源，Python

- 支持 100+ 提供商，OpenAI 兼容，路由配置，回退，基本可观测性。
- 在 Kong 的基准测试中约 2000 RPS 时出现问题；内存占用 8 GB，持续负载下出现级联故障。
- 最佳适用：Python 应用、<500 RPS、开发/预发布网关、实验性路由。
- 成本：OSS 免费；云有免费套餐。

### Portkey — 控制平面定位

- 2026 年 3 月起 Apache 2.0 开源。护栏、PII 重删、越狱检测、审计追踪。
- 每个请求延迟开销 20-40 ms。
- 生产版 retention + SLA 套餐 $49/月。
- 最佳适用：受监管行业，需要护栏+可观测性打包。

### Kong AI Gateway — 规模导向

- 构建在 Kong Gateway 之上（成熟的 API 网关产品，lua+OpenResty）。
- Kong 自测（12 CPU 等效）：比 Portkey 快 228%，比 LiteLLM 快 859%。
- 定价：$100/模型/月，Plus 方案最多 5 个。
- 最佳适用：已在使用 Kong；>1000 RPS；愿意付费授权。

### Bifrost（Maxim AI）

- 自动重试配可配置退避。
- OpenAI 429 时回退到 Anthropic 是一个典型方案。
- 新进入者；商业产品。

### Cloudflare AI Gateway / Vercel AI Gateway

- 托管，零运维。基本的重试和可观测性。
- 最佳适用：Cloudflare/Vercel 上的边缘 JavaScript 应用。
- 在护栏和限速方面与 Kong/Portkey 相比有限。

### 自托管 vs 托管

数据驻留是决定性因素。医疗和金融默认自托管（LiteLLM 或 Portkey OSS 或 Kong）。消费产品默认托管（Cloudflare AI Gateway）或中间层（Portkey 托管）。混合方案：受监管租户自托管，其他租户托管。

### 延迟预算

- LiteLLM：典型开销 5-15 ms。
- Portkey：开销 20-40 ms。
- Kong：开销 3-8 ms。
- Cloudflare/Vercel：边缘优势，开销 1-3 ms。

网关延迟直接累加到 TTFT。对于 TTFT P99 < 100 ms SLA，选择 Kong 或 Cloudflare。对于 P99 < 500 ms，任一皆可。

### 限速语义很重要

简单的令牌桶在中低规模下可行。多租户需要滑动窗口+突发配额+按租户分层。LiteLLM 提供令牌桶；Kong 提供滑动窗口；Portkey 提供分层。

### 网关 + 可观测性 + 路由 组合

阶段 17 · 13（可观测性）+ 16（模型路由）+ 19（网关）在生产中是同一层。选择一款覆盖全部三项的工具，或仔细串联：大多数 2026 年部署将 Helicone（可观测性）或 Portkey（护栏）与 Kong（规模）结合，分担角色。

### 需要记住的数字

- LiteLLM：约 2000 RPS 时出问题，内存 8 GB。
- Portkey：开销 20-40 ms；2026 年 3 月起 Apache 2.0。
- Kong：比 Portkey 快 228%，比 LiteLLM 快 859%。
- Kong 定价：$100/模型/月，Plus 方案最多 5 个。
- Cloudflare/Vercel：边缘开销 1-3 ms。

## 动手实现

`code/main.py` 模拟在 429/5xx 注入下跨 3 个提供商的网关路由回退。报告延迟、重试率和回退命中率。

## 交付物

本课产出 `outputs/skill-gateway-picker.md`。根据规模、运维姿态、合规性、延迟预算选择网关。

## 练习

1. 运行 `code/main.py`。配置从 OpenAI→Anthropic→自托管的回退。在提供商错误率 5% 时，预期命中率是多少？
2. 你的 SLA 是 TTFT P99 < 200 ms，基线 300 ms。哪些网关保持在预算内？
3. 医疗客户要求自托管 + PII 重删 + 审计。选择 Portkey OSS 还是 Kong。
4. 比较 LiteLLM 和 Kong：在什么 RPS 上限时团队应该迁移？
5. 为多租户 SaaS 设计限速策略：免费层、试用层、付费层。令牌桶还是滑动窗口？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| 网关 | "API 经纪人" | 位于应用和提供商之间的进程 |
| LiteLLM | "那个 MIT 的" | Python 开源，100+ 提供商，2K RPS 时出问题 |
| Portkey | "护栏网关" | 控制平面+可观测性，Apache 2.0 |
| Kong AI Gateway | "那个规模型的" | 构建在 Kong Gateway 之上，基准测试领先 |
| Bifrost | "Maxim 的网关" | 重试+Anthropic 回退方案 |
| Cloudflare AI Gateway | "边缘托管" | 边缘部署托管网关，零运维 |
| PII 重删 | "数据擦除" | 正则+NER 掩码，发送到模型前处理 |
| 越狱检测 | "提示词注入护栏" | 用户输入上的分类器 |
| 审计追踪 | "受监管日志" | 每次 LLM 调用的不可变记录 |
| 令牌桶 | "简单限速" | 基于补充的限速器 |
| 滑动窗口 | "精确限速" | 时间窗口限速器；公平性更好 |

## 延伸阅读

- [Kong AI Gateway 基准测试](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — 2026 AI 网关终极指南](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — 2026 年顶级 LLM 网关工具](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway 文档](https://docs.konghq.com/gateway/latest/ai-gateway/)