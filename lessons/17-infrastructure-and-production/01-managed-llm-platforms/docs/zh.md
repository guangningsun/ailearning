# 受管理 LLM 平台 — Bedrock、Vertex AI、Azure OpenAI

> 三大超大规模云服务商，三种截然不同的策略。AWS Bedrock 是一个模型 marketplace — Claude、Llama、Titan、Stability、Cohere 汇聚于同一 API 之后。Azure OpenAI 是独家 OpenAI 合作，外加用于专用容量的 Provisioned Throughput Units (PTU)。Vertex AI 以 Gemini 为首，拥有最出色的长上下文和多模态能力。2026 年，Artificial Analysis 测量到 Azure OpenAI 的中位 TTFT 约为 50 ms，Bedrock 在 Llama 3.1 405B 等效模型上约为 75 ms — PTU 解释了这一差距，因为专用容量优于共享按需模式。决策规则不是"哪个最快"，而是"哪个模型目录和 FinOps 界面与我的产品更匹配"。本课教你带着写好的权衡来选择，而不是凭感觉。

**类型：** 学习型
**语言：** Python（标准库、toy 成本与延迟比较器）
**前置条件：** 阶段 11（LLM 工程）、阶段 13（工具与协议）
**时间：** 约 60 分钟

## 学习目标

- 说出三大平台策略（marketplace vs 独家 vs Gemini-first），并将每种策略匹配到一个产品用例。
- 解释 Azure OpenAI 中的 Provisioned Throughput Units (PTU) 为你购买了什么样的保障，以及为什么在 405B 规模上按需使用 Bedrock 通常慢约 25 ms。
- 画出每个平台的 FinOps 归属面（Bedrock Application Inference Profiles vs Vertex 项目-per-团队 vs Azure 作用域 + PTU 预留）。
- 写下"双提供商最低标准"策略，并解释为什么在 2026 年单供应商锁定是最昂贵的错误。

## 问题

你为产品选了 Claude 3.7 Sonnet。现在你需要提供服务。你可以直连 Anthropic API，或通过 AWS Bedrock 调用，或通过网关调用。直接 API 最简单；Bedrock 增加了 BAAs、VPC 端点、IAM 和 CloudWatch 归属。网关增加了故障转移、统一计费和跨提供商速率限制。

更深层的问题是目录。如果你需要同一个产品中同时用到 Claude、Llama 和 Gemini，你无法从一个地方买到所有这些，除非那个地方同时是 Bedrock + Vertex + Azure OpenAI。超大规模云服务商并非可互换的 — 他们各自在模型层押了不同的注。

本课将三种赌注、延迟差距、FinOps 差距和锁定风险一一映射出来。

## 概念

### 三种策略

**AWS Bedrock** — marketplace。Claude (Anthropic)、Llama (Meta)、Titan (AWS 自有)、Stability（图像）、Cohere（embedding）、Mistral，加上图像和 embedding 子目录。一个 API、一个 IAM 面、一个 CloudWatch 导出。Bedrock 赌的是客户更想要可选性，而不是单一模型。

**Azure OpenAI** — 独家合作。你可以拿到 GPT-4 / 4o / 5 / o 系列、DALL·E、Whisper，以及在 Azure 数据中心对 OpenAI 模型进行微调。"Azure OpenAI Service"目录中没有非 OpenAI 模型 — 那些模型在 Azure AI Foundry（独立产品）中。Azure 赌的是 OpenAI 保持前沿地位，客户希望对该特定关系拥有企业级控制权。

**Vertex AI** — Gemini 第一，其他第二。Gemini 1.5 / 2.0 / 2.5 Flash 和 Pro，加上 Model Garden（第三方）。Vertex 赌的是多模态长上下文 — 100 万 token 的 Gemini 上下文是差异化所在。

### 大规模下的延迟差距

Artificial Analysis 运行持续基准测试。在等效的 Llama 3.1 405B 部署（共享按需）上，Azure OpenAI 首 token 延迟中位数约为 50 ms；Bedrock 约为 75 ms。差距不是 AWS 的失败 — 而是容量模型差异。Azure 销售 PTU（Provisioned Throughput Units），为你的租户预留 GPU 容量。Bedrock 的等效方案（Provisioned Throughput）存在，但起价约为每小时 $21，大多数客户仍使用共享按需。

按需共享容量与所有其他客户的流量竞争。专用容量则不会。如果你的产品 SLA 要求 P99 TTFT < 100 ms，你要么在 Azure 上购买 PTU，要么购买 Bedrock Provisioned Throughput，要么接受默认方差。

### 预留吞吐量经济学

Azure PTU：一块预留的推理计算。相对于可预测工作负载可节省约 70%。成本按小时固定计算，无论流量如何 — 即使空闲你也为预留付费。盈亏平衡点通常在约 40-60% 的持续利用率。

Bedrock Provisioned Throughput：根据模型和区域，每小时 $21-$50。类似的数学 — 盈亏平衡点在峰值利用率的一半左右。需要月度承诺。

Vertex 预留容量按 Gemini SKU 销售；根据模型和区域定价，公开程度较低。

### FinOps 面 — 真正的差异化

**Bedrock Application Inference Profiles** 是 marketplace 中最干净的归属。用 `team`、`product`、`feature` 标记一个 profile；通过它路由所有模型调用；CloudWatch 无需后处理即可按 profile 分摊成本。2025 年新增，至今仍是超大规模云原生中最细粒度的。

**Vertex** 归属是项目-per-团队加无处不在的标签。你将每个团队建模为一个 GCP 项目，在每个资源上贴标签，使用 BigQuery Billing Export + DataStudio 进行汇总。工作量更大，但 BigQuery 给你对成本数据的任意 SQL 查询能力。

**Azure** 依赖订阅/资源组作用域加标签，PTU 预留作为一等成本对象。标签从资源组继承，而非请求，因此每请求归属需要 Application Insights 自定义指标或一个在 header 上盖章的网关。

模式：Bedrock 原生最干净，Vertex 通过 BigQuery 最灵活，Azure 最不透明（除非你做了检测工具）。

### 锁定是 2026 年的风险

当一个模型占主导地位时，单一超大规模云承诺没问题。在 2026 年，前沿每月都在移动 — 一个季度是 Claude 3.7，下一个是 Gemini 2.5，再下个是 GPT-5。锁定一个平台意味着你被前沿的三分之二拒之门外。

有成效的团队采用的模式：任何产品关键 LLM 调用至少使用两个提供商。Bedrock 加 Azure OpenAI 是常见组合 — 一个提供 Claude，一个提供 GPT，两者之间故障转移，置于同一网关之后。成本增加可以忽略不计，因为网关会路由到最优方；但在宕机期间（如 2025 年 1 月 Azure OpenAI 事件、AWS us-east-1 宕机）的可用性提升是决定性的。

### 数据驻留、BAAs 和受监管行业

Bedrock：大多数区域有 BAAs；VPC 端点；护栏。常见的金融科技默认值。
Azure OpenAI：HIPAA、SOC 2、ISO 27001；欧盟数据驻留；企业级受监管的默认值。
Vertex：HIPAA、GDPR、按区域的数据驻留；Google Cloud 的合规体系。

三者都满足基本清单。差异在于数据保留策略、日志处理方式，以及滥用监控是否读取你的流量（大多数默认选入；企业可选择退出）。

### 你应该记住的数字

- Azure OpenAI 在 Llama 3.1 405B 等效模型上的中位 TTFT：约 50 ms（使用 PTU）。
- Bedrock 按需中位 TTFT：约 75 ms。
- Bedrock Provisioned Throughput：每小时 $21-$50 每单元。
- Azure PTU 盈亏平衡点：约 40-60% 持续利用率。
- 高利用率下 PTU 相比按需节省：高达 70%。

## 使用它

`code/main.py` 在合成工作负载上比较三个平台 — 模拟按需 vs PTU 经济学、TTFT 方差和成本归属保真度。运行它看看 PTU 在哪里值得买，以及 marketplace 的模型广度何时能弥补 TTFT 差距。

## 交付它

本课产出 `outputs/skill-managed-platform-picker.md`。给定工作负载配置文件（所需模型、TTFT SLA、日间量、合规要求），它推荐一个主要平台、一个备用平台和一个 FinOps 检测计划。

## 练习

1. 运行 `code/main.py`。在什么持续利用率下 Azure PTU 在 70B 类模型上击败按需？计算盈亏平衡点并与广告宣传的 40-60% 区间比较。
2. 你的产品需要 Claude 3.7 Sonnet 和 GPT-4o。设计一个双提供商部署 — 哪个放到哪个超大规模云上，什么网关放在前面，故障转移策略是什么？
3. 一个受监管的医疗客户需要 BAAs、美国东部数据驻留和 P99 TTFT < 100ms。选择一个平台并用三个具体功能来证明。
4. 你发现你的 Bedrock 账单本月增长了 4 倍，但流量没有变化。如果没有 Application Inference Profiles，你会如何找到罪魁祸首？有 profiles 的话需要多长时间？
5. 阅读 Azure OpenAI 和 Bedrock 的定价页面。对于每月 1 亿 token 的 Claude 工作负载，哪个更便宜 — 直接 Anthropic API、Bedrock 按需，还是 Bedrock Provisioned Throughput？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Bedrock | "AWS LLM 服务" | 跨 Claude、Llama、Titan、Mistral、Cohere 的模型 marketplace |
| Azure OpenAI | "Azure 的 ChatGPT" | Azure 数据中心中的独家 OpenAI 模型，带企业级控制 |
| Vertex AI | "Google 的 LLM" | 以 Gemini 为首的平台，Model Garden 提供第三方模型 |
| PTU | "专用容量" | Provisioned Throughput Unit — 预留的推理 GPU，按小时定价 |
| Application Inference Profile | "Bedrock 标签" | 带标签的按产品成本/使用 profile，CloudWatch 原生 |
| Model Garden | "Vertex 目录" | Vertex AI 的第三方模型部分，与 Gemini 分开 |
| 双提供商最低标准 | "LLM 冗余" | 策略：每个关键 LLM 路径至少跨 2 个超大规模云运行 |
| BAA | "HIPAA 文件" | Business Associate Agreement；PHI 所必需；三者都提供 |
| 滥用监控 | "日志观察者" | 提供商端对 prompt/输出的安全扫描；企业可选择退出 |

## 延伸阅读

- [AWS Bedrock 定价](https://aws.amazon.com/bedrock/pricing/) — 权威价目表和 Provisioned Throughput 定价。
- [Azure OpenAI 服务定价](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) — PTU 经济性和价目表。
- [Vertex AI 生成式 AI 定价](https://cloud.google.com/vertex-ai/generative-ai/pricing) — Gemini 层和 Model Garden 附加费。
- [Artificial Analysis LLM 排行榜](https://artificialanalysis.ai/) — 跨提供商的持续延迟和吞吐量基准测试。
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO 指南 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — 企业决策框架。
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — 归属机制并排对比。