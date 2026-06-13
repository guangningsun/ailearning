# 提示词缓存与语义缓存经济学

> **定价快照日期为 2026 年 4 月。** 以下数字 claims 基于本课发布时供应商价目表获取；引用前请验证链接文档。

> 缓存在两个层级发生。L2（供应商层级）提示词/前缀缓存复用重复前缀的注意力 KV——Anthropic 的提示词缓存文档宣传最长提示词可节省 90% 成本和 85% 延迟；Claude 3.5 Sonnet 缓存读取价格为 $0.30/M vs $3.00/M 全新，包含 5 分钟 TTL 和 1 小时 TTL 选项（2 倍写入溢价）（docs.anthropic.com，2026-04）。OpenAI 提示词缓存对 ≥1024 token 的提示词自动应用，缓存输入价格约为全新的 90% 折扣（platform.openai.com，2026-04）；具体每模型缓存价格取决于实时价目表。L1（应用层级）语义缓存在 embedding 相似度命中时完全跳过 LLM。供应商"95% 准确率"指的是匹配正确性，而非命中率——生产环境报告的命中率从 10%（开放式聊天）到 70%（结构化 FAQ）不等；没有供应商发布官方基准，因此请将这些视为社区遥测数据而非保证。生产环境陷阱：并行化会杀死缓存（N 个并行请求在第一次缓存写入完成前发出，可使支出膨胀数倍），以及前缀内的动态内容会完全阻止缓存命中。ProjectDiscovery 报告通过将动态文本移出可缓存前缀，命中率从 7% 提升到 74%（2025-11）。

**类型：** 学习型
**语言：** Python（标准库 + 玩具级双层缓存模拟器）
**前置条件：** 阶段 17 · 04（vLLM 服务内部原理）、阶段 17 · 06（SGLang RadixAttention）
**时间：** 约 60 分钟

## 学习目标

- 区分 L2 提示词/前缀缓存（供应商层级 KV 复用）与 L1 语义缓存（相似提示词绕过 LLM）。
- 解释 Anthropic 的显式 `cache_control` 标记和两种 TTL 选项（5 分钟 vs 1 小时）及其价格倍数。
- 给定命中率、提示词/响应比例和 token 价格，计算预期月度节省。
- 说出使账单膨胀 5-10 倍的并行化反模式，以及使命中率崩溃的动态内容反模式。

## 问题

你给 RAG 服务加了提示词缓存。账单纹丝不动。你测量了命中率；是 7%。你的提示词看起来是静态的但实际上不是——系统提示词包含精确到分钟格式的当前日期、一个请求 ID，以及为多样性而随机重排的示例。每个请求都写入新的缓存条目，读取次数为零。

另外，你的 Agent 每用户问题发起十个并行工具调用。十个调用都在第一次缓存写入完成前到达供应商。十个写入，零次读取。你的账单是"加了缓存"应有的 5-10 倍。

缓存是一套协议，不是一个开关。两层缓存，两种不同的失败模式。

## 概念

### L2 — 供应商提示词/前缀缓存

供应商存储可缓存前缀的注意力 KV，并在下一个匹配该前缀的请求中复用。你付一次写入成本，读操作几乎免费。

**Anthropic（Claude 3.5 / 3.7 / 4 系列）**：请求中显式的 `cache_control` 标记。你标记哪些块可缓存。TTL：5 分钟（写入成本 1.25 倍基准）或 1 小时（写入成本 2 倍基准）。缓存读取：Claude 3.5 Sonnet 上 $0.30/M vs $3.00/M 全新——便宜 10 倍（docs.anthropic.com，2026-04）。费率因模型而异（Opus/Haiku 单独发布）；请始终核对实时定价页面。

**OpenAI**：对 ≥1024 token 的提示词自动缓存（platform.openai.com，2026-04）。无显式标记。在当前 gpt-4o/gpt-5 价目表上，缓存输入约为全新的 10% 价格。文档和发布说明均未公布官方命中率基准；社区报告集中在 30–60%（需要精心设计提示词）。监控 `usage.cached_tokens` 来测量你自己的命中率。

**Google（Gemini）**：通过显式 API 做上下文缓存；100 万 token 的上下文意味着缓存收益更高。

**自托管（vLLM、SGLang）**：阶段 17 · 06 讲解 RadixAttention——在你自己的计算资源上实现相同模式。

### L1 — 应用层级语义缓存

在完全调用 LLM 之前，对提示词做哈希和 embedding，然后查找相似的缓存请求（余弦相似度高于阈值，通常 0.95+）。命中时返回缓存响应。未命中时调用 LLM 并缓存结果。

开源方案：Redis Vector Similarity、GPTCache、Qdrant。商业方案：Portkey Cache、Helicone Cache。

供应商的准确率声称指的是返回的缓存响应在语义上合适的频率——而不是你命中的频率。生产环境命中率：

- 开放式聊天：10-15%。
- 结构化 FAQ / 支持：40-70%。
- 代码问题：20-30%（小变体会杀死命中）。
- 语音 Agent 重复提示词：50-80%（语音规范化修复了固定集合）。

### 并行化反模式

你的 Agent 并行发出 10 个工具调用。10 个调用都有相同的 4K token 系统提示词。Anthropic 缓存写入按请求计；第一次缓存写入在供应商看到提示词后约 300ms 完成。请求 2-10 在同一毫秒窗口内到达，每个都看到缓存未命中。你付了 10 次写入溢价，零次读取折扣。

修复：批量 + 顺序优先——先单独发请求 1，然后等 1 的缓存填充后再发 2-10。第一次工具调用增加 300ms；节省 5-10 倍账单。

### 动态内容反模式

你的系统提示词看起来像：

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

每个请求都是唯一的。每个请求都写入。零命中。

修复：将真正静态的内容移到可缓存前缀中；将动态内容追加到缓存边界之后：

```
[可缓存]
You are a helpful assistant. [规则、示例、指令]
[/可缓存]
[动态，不缓存]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery 这样做后从 7% 提升到 74% 缓存命中率，并公开了方法论。

### 栈叠批处理 + 缓存处理夜间工作负载

批处理 API（阶段 17 · 15）在 24 小时周转下提供 50% 折扣。在此基础上叠加缓存输入再获约 10 倍优惠。夜间分类、标注和报告生成工作负载通过栈叠可以降到同步无缓存成本的约 10%。

### 需要记住的数字

定价数据于 2026-04 从链接的供应商文档获取，几个月就会变动——依赖前请重新核对。

- Anthropic 缓存读取：Claude 3.5 Sonnet 上 $0.30/M，约为全新输入的 10 倍便宜（docs.anthropic.com）。
- Anthropic 缓存写入溢价：1.25 倍（5 分钟 TTL）或 2 倍（1 小时 TTL）。
- OpenAI 自动缓存：适用于 ≥1024 token 的提示词；在当前价目表上缓存输入定价约为全新的 10%（platform.openai.com）。
- 语义缓存命中率（社区报告）：开放式聊天约 10%；结构化 FAQ 高达约 70%。非供应商文档化的基准。
- ProjectDiscovery：将动态内容移出前缀后命中率从 7% 升至 74%（项目博客，2025-11）。
- 并行化反模式：当 N 个并行请求错过第一次缓存写入时，典型报告的账单膨胀为 5–10 倍。

## 使用方法

`code/main.py` 在混合工作负载上模拟 L1 + L2 缓存。报告命中率、账单，并展示并行化惩罚。

## 交付物

本课产出 `outputs/skill-cache-auditor.md`。给定提示词模板和流量，审计可缓存性并推荐重构方案。

## 练习

1. 运行 `code/main.py`。切换并行化标志。账单变化了多少？
2. 你的系统提示词包含日期。把它移出去。展示前/后命中率计算。
3. 计算 1 小时 TTL（2 倍写入）vs 5 分钟 TTL（1.25 倍写入）的盈亏平衡点，给定你的请求到达率。
4. 语义缓存在 0.95 阈值命中 20%。在 0.85 命中 50% 但你会看到不正确的缓存响应。选择正确的阈值并给出理由。
5. 你每用户问题批量 10 个并行子查询。重写为缓存友好且不增加端到端延迟。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| L2 prompt cache | "prefix cache" | 供应商存储重复前缀的 KV |
| `cache_control` | "Anthropic cache marker" | 显式属性标记可缓存块 |
| Cache write premium | "write tax" | 首次未命中到缓存的额外成本（1.25x 或 2x） |
| L1 semantic cache | "embedding cache" | 调用 LLM 前的应用层级哈希和 embedding |
| GPTCache | "LLM caching lib" | 流行的开源 L1 缓存库 |
| Cache hit rate | "hits / total" | 从缓存服务的请求比例 |
| Parallelization anti-pattern | "the N-write trap" | N 个并行请求 N 次错过缓存 |
| Dynamic content trap | "the time-in-prompt trap" | 前缀中的动态字节杀死命中率 |
| RadixAttention | "intra-replica cache" | SGLang 的前缀缓存实现 |

## 延伸阅读

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 官方 `cache_control` 语义和 TTL。
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) — 自动缓存行为和适用条件。
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)