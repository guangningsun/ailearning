# 批处理 API——50% 折扣成为行业标准

> 每个主要供应商都提供异步批处理 API，附带 50% 折扣和约 24 小时周转。OpenAI、Anthropic、Google 和大多数推理平台（Fireworks 批处理层、Together batch）都实现相同模式。将批处理与提示词缓存叠加，夜间流水线可以降到同步无缓存成本的约 10%。规则简单粗暴：如果不是交互式的，就该上批处理。内容生成流水线、文档分类、数据提取、报告生成、批量标注、目录打标——任何能容忍 24 小时延迟的都是钱还躺在桌上，直到它搬到批处理。2026 年的生产模式是将每个新的 LLM 工作负载三车道分诊：交互式（同步 + 缓存）、半交互式（异步队列 + 回退）、批处理（夜间，缓存输入栈叠）。假装交互式但实际能容忍几分钟延迟的工作负载浪费了大部分收益。

**类型：** 学习型
**语言：** Python（标准库 + 玩具级批处理 vs 同步成本模拟器）
**前置条件：** 阶段 17 · 14（提示词与语义缓存）
**时间：** 约 45 分钟

## 学习目标

- 说出三个供应商批处理 API（OpenAI、Anthropic、Google）和通用的 50% 折扣 + 24 小时周转保证。
- 计算在夜间分类工作负载上叠加批处理 + 缓存输入的成本，并与同步无缓存基准比较。
- 将工作负载分诊到交互式 / 半交互式 / 批处理，并给出车道选择的理由。
- 说出两个陷阱：部分交互式（用户期望快于 24 小时）和输出模式漂移（每供应商批处理文件格式不同）。

## 问题

你的团队上线了一个夜间报告生成流水线。50,000 份文档，总结每份，对摘要做聚类，起草高管简报。同步运行需要 4 小时，每晚 $2,000。你听说了批处理 API。

批处理给你 5 折。你还启用了系统提示词的提示词缓存（所有 50k 调用共享）。栈叠后，账单降到每晚 $180——约基准的 9%。同一流水线，三个配置改动。

批处理是 LLM 成本工具箱里没人拉的那根最便宜的杠杆。原因主要是组织性的：团队想到"实时"而 SLA 实际上是"明天早上"。本课讲的是不要把 90% 的账单落在桌上。

## 概念

### 三个批处理 API

**OpenAI Batch API**：JSONL 文件上传，包含请求列表。承诺 24 小时周转（实际通常约 2-8 小时）。输入和输出 token 均 5 折。`/v1/batches` 端点。可缓存输入同样享受缓存输入定价。

**Anthropic Message Batches**：JSONL 上传。24 小时周转。5 折。支持 `cache_control`——缓存写入是显式的，读操作在批处理内部自动发生。

**Google Vertex AI Batch Prediction**：BigQuery 或 GCS 输入。Gemini 类似 5 折。与 Vertex 流水线集成。

### 语义：异步，不是慢

批处理是"我承诺 24 小时内返回"——不是说"这需要 24 小时"。P50 通常是 2-6 小时。供应商会在 GPU 库存未充分利用的峰值外窗口调度你的批处理。

### 与缓存栈叠

5 万文档摘要，相同的 4K token 系统提示词：

- 同步无缓存：50000 × ($input × 4000 + $output × 200)，全价。
- 同步缓存：系统提示词在第一次写入后缓存；剩余 49999 次享受 10 倍便宜的输入。
- 批处理缓存：上述全部加上读和写都 5 折。

栈叠：批处理 + 缓存 = 同步无缓存账单的约 10%。任何在夜间运行且有共享系统提示词的工作负载都应该用这个。

### 工作负载分诊

**交互式** — 用户等待响应。TTFT 很重要。同步调用加提示词缓存。不能批处理。

**半交互式** — 用户提交任务，几分钟后回来检查。异步队列，未能批处理时回退到同步。中等容量 RAG 索引属于此类。

**批处理** — 用户期望结果"明天早上"或"下一小时"。内容流水线、大规模分类、离线分析。始终批处理，始终栈叠缓存。

常见错误：因为流水线是生产环境就把一切归类为交互式。生产环境不是延迟规格——SLA 才是。

### 部分交互式陷阱

有些功能看起来是交互式的但能容忍 5-10 分钟。例子：夜间客户健康报告带有"刷新"按钮。用户点击刷新；等 10 分钟没问题。团队把它做成同步的。50 个并发刷新成本是批处理 + 邮件发送的 10 倍。

要问的问题："24 小时对这个用户意味着什么？"如果答案是"他们不会注意到"，就批处理。

### 输出模式陷阱

每供应商的批处理文件格式不同：

- OpenAI：JSONL，每行一个请求。
- Anthropic：JSONL，每行一条消息；响应格式内嵌。
- Vertex：BigQuery 表或带 TFRecord 的 GCS 前缀。

编写"一个批处理客户端"跨供应商意味着每个供应商都需要适配器代码。宣传多供应商批处理的网关（Portkey、LiteLLM 部分层级）仍然是薄包装原始格式。

### 需要记住的数字

- 跨供应商批处理折扣：输入 + 输出均 5 折。
- 周转 SLA：24 小时保证，实际 P50 通常 2-6 小时。
- 栈叠批处理 + 缓存输入：同步无缓存成本的约 10%。
- 工作负载分诊规则：如果 24 小时延迟可接受，始终批处理。

## 使用方法

`code/main.py` 计算 5 万文档工作负载在同步、同步+缓存、批处理、批处理+缓存四种模式下的成本。报告节省金额和百分比。

## 交付物

本课产出 `outputs/skill-batch-triager.md`。给定工作负载特征，分诊到交互式/半交互式/批处理并估算节省。

## 练习

1. 运行 `code/main.py`。对于 10 万文档流水线、3K token 系统提示词和 500 token 输出，计算全栈（批处理 + 缓存）vs 同步基准的节省。
2. 在你熟悉的一个真实产品中选三个功能。将每个分诊到交互式/半交互式/批处理。
3. 用户抱怨报告花了 3 小时。这是批处理分诊错误还是合法的交互式？写出决策标准。
4. 你的批处理 API 返回 SLA 是 24h 但 P99 是 20 小时。你如何向用户沟通这个——边缘情况下游系统行为是什么？
5. 计算盈亏平衡：在什么共享前缀长度下，批处理 + 缓存比自己用预留 GPU 过夜更便宜？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| Batch API | "async discount" | 5 折 + 24 小时周转 |
| JSONL | "batch format" | 每行一个 JSON 请求；OpenAI/Anthropic 标准 |
| Message Batches | "Anthropic batch" | Anthropic 的批处理 API 产品名称 |
| Batch prediction | "Vertex batch" | Vertex AI 的批处理 API 产品名称 |
| Turnaround SLA | "24h promise" | 保证，而非典型；实际通常是 2-6h |
| Workload triage | "interactivity decision" | 交互式 / 半交互式 / 批处理路由决策 |
| Output schema | "response format" | 每供应商 JSONL 布局；不可移植 |
| Stacked discount | "batch + cache" | 两者同时适用时约为无缓存同步账单的 10% |

## 延伸阅读

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) — JSONL 格式和 `/v1/batches` 语义。
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) — 批处理格式和 `cache_control` 交互。
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction) — Gemini 批处理语义。
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)