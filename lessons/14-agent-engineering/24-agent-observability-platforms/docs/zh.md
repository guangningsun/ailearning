# Agent 可观测性平台：Langfuse、Phoenix、Opik

> 2026 年三大开源 Agent 可观测性平台。Langfuse（MIT 协议）—— 月安装量 600 万+，支持链路追踪、提示词管理、评估和会话回放。Arize Phoenix（Elastic 2.0 协议）—— 深度 Agent 特化评估、RAG 相关性、OpenInference 自动插桩。Comet Opik（Apache 2.0 协议）—— 自动提示词优化、护栏、LLM-judge 幻觉检测。

**类型：** 学习型
**语言：** Python（标准库）
**前置条件：** 阶段 14 · 23（OTel GenAI）
**时间：** 约 45 分钟

## 学习目标

- 说出三大开源 Agent 可观测性平台的名称及其许可证类型。
- 区分各自的优势领域：Langfuse（提示词管理 + 会话）、Phoenix（RAG + 自动插桩）、Opik（优化 + 护栏）。
- 解释为什么 89% 的组织在 2026 年已具备 Agent 可观测能力。
- 用标准库实现一条链路追踪到仪表盘的流水线，并带有 LLM-judge 评估。

## 问题

OTel GenAI（第 23 课）给了你 schema。但你仍然需要一个平台来摄取 span、运行评估、存储提示词版本，并暴露回归问题。三个竞争者各自强调生命周期中的不同环节。

## 概念

### Langfuse（MIT 协议）

- 月安装量 600 万+，GitHub 星标 19k+。
- 功能：链路追踪、带版本控制和 playground 的提示词管理、评估（LLM-as-judge、用户反馈、自定义）、会话回放。
- 2025 年 6 月：此前为商业模块的功能（LLM-as-a-judge、标注队列、提示词实验、Playground）以 MIT 协议开源。
- 最适合：提示词管理闭环紧密的端到端可观测性。

### Arize Phoenix（Elastic License 2.0）

- 更深入的 Agent 特化评估：链路聚类、异常检测、RAG 检索相关性。
- 原生 OpenInference 自动插桩。
- 与托管版 Arize AX 配合用于生产环境。
- 无提示词版本控制 —— 定位为行为漂移/回归工具，与更广泛的平台协同使用。
- 最适合：RAG 相关性、行为漂移、异常检测。

### Comet Opik（Apache 2.0 协议）

- 通过 A/B 实验自动优化提示词。
- 护栏（PII 删除、主题约束）。
- LLM-judge 幻觉检测。
- 据 Comet 自测：Opik 日志 + 评估耗时 23.44 秒 vs Langfuse 327.15 秒（约 14 倍差距）—— 请将厂商基准数据视为方向性参考。
- 最适合：优化闭环、自动实验、护栏执行。

### 行业数据

据 Maxim（2026 年实地分析）：89% 的组织已具备 Agent 可观测能力；质量问题是最主要的生成障碍（32% 的受访者提及）。

### 如何选择

| 需求 | 选择 |
|------|------|
| 带提示词管理的一体化平台 | Langfuse |
| 深度 RAG 评估 + 漂移检测 | Phoenix |
| 自动优化 + 护栏 | Opik |
| 开放许可证，无 ELv2 约束 | Langfuse（MIT）或 Opik（Apache 2.0） |
| Datadog / New Relic 集成 | 任一均可 —— 它们都导出 OTel |

### 这个模式容易出错的地方

- **没有评估策略。** 不带评估的链路追踪只是昂贵的日志记录。
- **自研 LLM-judge 缺乏外部事实依据。** CRITIC 模式（第 05 课）同样适用 —— 判断器需要外部工具进行事实核实。
- **提示词版本未与链路关联。** 生成回归时，无法定位到引发问题的提示词。

## 构建

`code/main.py` 实现了一个标准库链路采集器 + LLM-judge 评估器：

- 摄取 GenAI 格式的 span。
- 按会话分组，标记失败运行（护栏触发、评估置信度低）。
- 一个脚本化的 LLM-judge，按评分规则对 Agent 响应打分。
- 类似仪表盘的摘要：失败率、首要失败原因、评估分数分布。

运行：

```
python3 code/main.py
```

输出：每个会话的评估分数和失败分类，与 Langfuse/Phoenix/Opik 展示的内容一致。

## 使用

- **Langfuse** 自托管或云服务；通过 OTel 或其 SDK 连接。
- **Arize Phoenix** 自托管；自动插桩 OpenInference。
- **Comet Opik** 自托管或云服务；自动优化闭环。
- **Datadog LLM Observability** 适用于已使用 Datadog 的混合 ops+ML 团队。

## 交付

`outputs/skill-obs-platform-wiring.md` 选择一个平台，并将链路追踪 + 评估 + 提示词版本接入现有 Agent。

## 练习

1. 将一周的 OTel 链路导出到 Langfuse 云（免费套餐）。哪些会话失败了？为什么？
2. 为你的领域编写一个 LLM-judge 评分规则（事实正确性、语气、范围遵循度）。在 50 条链路上测试。
3. 对比 Langfuse 提示词版本控制与 Phoenix 的链路聚类。哪个能更快定位问题？
4. 阅读 Opik 的护栏文档。将一个 PII 删除护栏接入你的一条 Agent 运行中。
5. 在你的语料库上对三者做基准测试。忽略厂商公布的数据；自己测量。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| 链路追踪（Tracing） | "Span 采集器" | 摄取 OTel / SDK span；按会话索引 |
| 提示词管理（Prompt management） | "提示词 CMS" | 与链路关联的带版本提示词 |
| LLM-as-judge | "自动评估" | 用独立的 LLM 按评分规则对 Agent 输出打分 |
| 会话回放（Session replay） | "链路回放" | 回溯过往运行进行调试 |
| RAG 相关性（RAG relevancy） | "检索质量" | 检索到的上下文是否匹配查询 |
| 链路聚类（Trace clustering） | "行为分组" | 对相似运行聚类以检测漂移 |
| 护栏执行（Guardrail enforcement） | "日志时策略检查" | 对记录内容进行 PII/毒性/范围检查 |

## 延伸阅读

- [Langfuse 文档](https://langfuse.com/) —— 链路追踪、评估、提示词管理
- [Arize Phoenix 文档](https://docs.arize.com/phoenix) —— 自动插桩、漂移检测
- [Comet Opik](https://www.comet.com/site/products/opik/) —— 优化 + 护栏
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —— 三者共同消费的 schema