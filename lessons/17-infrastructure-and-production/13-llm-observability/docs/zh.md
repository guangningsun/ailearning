# LLM 可观测性技术栈选型

> 2026 年可观测性市场分为两大类。开发平台（LangSmith、Langfuse、Comet Opik）将监控与评估、提示词管理、会话回放打包在一起。网关/插桩工具（Helicone、SigNoz、OpenLLMetry、Phoenix）专注于遥测。Langfuse 是 MIT 许可的核心产品，在开源生态中实现了良好的平衡（云端免费 50K 事件/月）。Phoenix 基于 OpenTelemetry 原生设计，采用 Elastic License 2.0——非常适合漂移检测和 RAG 可视化，但不适合作为持久化生产后端。Arize AX 使用零拷贝 Iceberg/Parquet 集成，声称比整体式可观测性方案便宜 100 倍。LangSmith 在 LangChain/LangGraph 生态中领先，$39/用户/月，仅在企业版支持自托管。Helicone 是基于代理的方案，15-30 分钟即可完成配置，每月免费 100K 请求，但在 Agent 调用链追踪方面深度不足。常见的生产模式：网关（Helicone/Portkey）+ 评估平台（Phoenix/TruLens），通过 OpenTelemetry 粘合。

**类型：** 学习型
**语言：** Python（标准库 + 玩具级追踪采样模拟器）
**前置条件：** 阶段 17 · 08（推理指标）、阶段 14（Agent 工程）
**时间：** 约 60 分钟

## 学习目标

- 区分开发平台（打包式：评估 + 提示词 + 会话）与网关/遥测工具（仅追踪 + 指标）。
- 将六大主流工具（Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik）映射到其许可模式、定价和最佳适用场景。
- 解释 OpenTelemetry 粘合模式——如何将网关工具与独立评估平台组合使用。
- 说出 2026 年的成本差异点（Arize AX 的零拷贝方案 vs 整体式摄取）并陈述大约 100 倍的倍数关系。

## 问题

你上线了一个 LLM 功能。它能跑。你对提示词失败、工具循环、延迟回退、成本飙升或提示词缓存命中率毫无可见性。你 Google 一下 "LLM 可观测性"，出来八个工具，都声称用三个不同价位解决同一个问题。

它们解决的并非同一个问题。LangSmith 回答"为什么这个 LangGraph 运行失败了？" Phoenix 回答"我的 RAG 流水线是否在漂移？" Helicone 回答"哪个应用在烧 token？" Langfuse 回答"我能自托管整套方案吗？" 不同工具，不同受众。

选型涉及四个维度：技术栈（LangChain？原生 SDK？多供应商？）、许可容忍度（仅 MIT？Elastic 可以？商业版也行？）、预算（免费档？$100/月？$1000/月？）和自托管（必须？最好？绝不？）。

## 概念

### 两大类别

**开发平台** 将可观测性与评估、提示词管理、数据集版本控制、会话回放打包在一起。你做实验，看哪个提示词效果好，用新提示词对旧赢家做数据集回归。LangSmith、Langfuse、Comet Opik。

**网关/遥测工具** 对推理调用进行插桩——提示词、响应、token、延迟、模型、成本。Helicone、SigNoz、OpenLLMetry、Phoenix。极简风格。可通过 OpenTelemetry 与独立评估工具组合使用。

### Langfuse — 开源平衡之选

- 核心代码 Apache / MIT 双许可；通过 Docker 自托管。
- 云端免费档：50K 事件/月。付费版：$29/月（团队）。
- 评估、提示词管理、追踪、数据集。四大开发平台功能覆盖合理。
- 最佳场景：需要 LangSmith 级别的功能，但必须自托管或坚持开源许可。

### Phoenix（Arize）— 遥测优先，OpenTelemetry 原生

- Elastic License 2.0；自托管非常简单。
- RAG 和漂移可视化表现优异。Embedding 空间散点图作为一等公民呈现。
- 不适合作为持久化生产后端——主要是开发阶段可观测性工具。
- 最佳场景：RAG 流水线开发、漂移调试，搭配独立网关用于生产环境。

### Arize AX — 规模化方案

- 商业产品。零拷贝数据湖集成（Iceberg/Parquet）。
- 声称在规模化场景下比整体式可观测性（Datadog 级别）便宜约 100 倍。算术逻辑：你将追踪存储在自己 S3 上的 Parquet 中；Arize 直接读取。
- 最佳场景：每日 >1000 万条追踪、已有数据湖、希望获得 LLM 特异性仪表盘但不想承受 Datadog 的定价。

### LangSmith — 优先支持 LangChain/LangGraph

- 商业产品，$39/用户/月。仅在企业版支持自托管。
- 在 LangChain 和 LangGraph 技术栈上是同类最佳。如果你不在这两个技术栈上，吸引力会大打折扣。
- 最佳场景：团队已深度投入 LangChain，愿意付费。

### Helicone — 基于代理的最简可行方案

- 切换 `OPENAI_API_BASE` 指向 Helicone 代理，15-30 分钟完成配置。
- MIT 许可；每月免费 100K 请求，付费 $20/月起。
- 包含故障转移、缓存、限流——也充当网关角色。
- 在 Agent / 多步骤追踪的深度上有所欠缺。
- 最佳场景：快速启动、单一技术栈应用、需要网关 + 可观测性一体化。

### Opik（Comet）— 开源开发平台

- Apache 2.0，完全开源。
- 功能集与 Langfuse 类似，有 Comet 血统。
- 最佳场景：已在 Comet 上的 ML 团队，希望在同一界面下获得 LLM 可观测性。

### SigNoz — OpenTelemetry 优先的全栈 APM

- Apache 2.0。既处理通用 APM，也通过 OpenTelemetry 处理 LLM。
- 最佳场景：跨服务和 LLM 调用统一可观测性。

### 粘合剂：OpenTelemetry + GenAI 语义约定

OpenTelemetry 在 2025 年底发布了 GenAI 语义约定（`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`）。支持 OTel 的工具可以互操作。新兴的生产模式：

1. 每次 LLM 调用都通过 GenAI 约定发射 OTel。
2. 路由到网关（Helicone / Portkey）处理日常事务。
3. 双发到评估平台（Phoenix / Langfuse）检测回归。
4. 归档到数据湖（Iceberg）供长期分析，通过 Arize AX 或 DuckDB 查询。

### 陷阱：在错误的层面插桩

在 Agent 框架内部插桩（例如添加 LangSmith 追踪）会将你耦合到该框架。在 HTTP/OpenAI-SDK 层插桩（通过 OpenLLMetry 或网关）是可移植的。

### 采样——你无法保留所有数据

当日均请求 >100 万时，完整追踪保留的成本超过 LLM 调用本身。按规则采样：100% 错误、100% 高成本、5% 成功。聚合数据始终保留；原始数据仅保留长尾部分。

### 需要记住的数字

- Langfuse 免费云端：50K 事件/月。
- LangSmith：$39/用户/月。
- Helicone 免费：100K 请求/月。
- Arize AX 声称：规模化场景下比整体式便宜约 100 倍。
- OpenTelemetry GenAI 语义约定：2025 年发布，2026 年广泛采用。

## 使用方法

`code/main.py` 模拟 100 万追踪/日的场景，对比不同保留策略（100% 摄入、采样、采样 + 错误）。报告每种策略的存储成本和数据损失情况。

## 交付物

本课产出 `outputs/skill-observability-stack.md`。根据技术栈、规模、预算、许可姿态，筛选出合适的工具（组合）。

## 练习

1. 你的团队使用 LangChain，想要开源自托管可观测性。选择 Langfuse 或 Opik 并给出理由。
2. 每天 500 万条追踪，Datadog 报价 $150K/月，计算 Arize AX 的盈亏平衡点。
3. 设计一套 OpenTelemetry GenAI 属性集，作为你们组织的标准——每个 LLM 调用都必须包含。
4. 论证 Phoenix 单独是否足以支持生产环境。什么时候不足？
5. Helicone 带来 20ms 代理开销。在 P99 TTFT 300ms 的情况下，这可以接受吗？如果 SLA 是 100ms 呢？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| OpenLLMetry | "OTel for LLMs" | LLM 的开源 OpenTelemetry 插桩方案 |
| GenAI conventions | "OTel attributes" | LLM 调用的标准 OTel 属性名 |
| LangSmith | "LangChain observability" | 捆绑在 LangChain 生态中的商业平台 |
| Langfuse | "OSS LangSmith" | MIT 开源，功能集类似 |
| Phoenix | "Arize dev tool" | OpenTelemetry 原生的开发/评估平台 |
| Arize AX | "scale observability" | 商业零拷贝 Iceberg/Parquet 可观测性 |
| Helicone | "proxy observability" | 收集 LLM 遥测数据的 HTTP 代理 + 网关功能 |
| Opik | "Comet LLM" | Comet 出品的 Apache 2.0 开源开发平台 |
| Session replay | "trace rerun" | 重放完整的 Agent 会话（含工具调用） |
| Eval | "offline test" | 在带标签的数据集上运行候选模型/提示词 |

## 延伸阅读

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)