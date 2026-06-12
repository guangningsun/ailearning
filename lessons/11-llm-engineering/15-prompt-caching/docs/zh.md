# 提示词缓存与上下文缓存

> 你的系统提示词是 4,000 个 token。你的 RAG 上下文是 20,000 个 token。你每次请求都发送这两个，而且每次都要为它们付费。提示词缓存让提供商在其端保持该前缀的热状态，并在复用时按正常费率的 10% 向你收费。使用得当，可将推理成本降低 50–90%，首 token 延迟降低 40–85%。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 11 · 01（提示词工程）、阶段 11 · 05（上下文工程）、阶段 11 · 11（缓存与成本）
**时间：** 约 60 分钟

## 问题

一个编码智能体在对话的每个回合都向 Claude 发送相同的 15,000 token 系统提示词。20 个回合，按 $3/M 输入 token 计，仅输入成本就是 $0.90——在用户实际消息之前。乘以 10,000 次每日对话，账单达到 $9,000/天，而这些文本从未改变。

你不能在不损害质量的情况下缩小提示词。你无法避免发送它——模型每个回合都需要它。唯一的办法是停止为提供商已经见过的前缀支付全价。

这个办法就是提示词缓存。Anthropic 于 2024 年 8 月推出此功能（2025 年推出了 1 小时延长 TTL 变体），OpenAI 同年晚些时候将其自动化，Google 随 Gemini 1.5 推出了显式上下文缓存，三家厂商现在都在其前沿模型上将此功能作为一等公民提供。

## 概念

![提示词缓存：写一次，读廉价](../assets/prompt-caching.svg)

**机制。** 当请求的前缀与最近请求匹配时，提供商从上一次运行中提供 KV-cache，而不是重新编码 token。第一次支付少量写溢价，之后每次读享受大幅折扣。

**2026 年的三种提供商风格。**

| 提供商 | API 风格 | 命中折扣 | 写溢价 | 默认 TTL | 最小可缓存 |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | 内容块上的显式 `cache_control` 标记 | 输入降低 90% | 加收 25% | 5 分钟（可延长至 1 小时） | 1,024 token（Sonnet/Opus），2,048（Haiku） |
| OpenAI | 自动前缀检测 | 输入降低 50% | 无 | 最多 1 小时（尽力而为） | 1,024 token |
| Google（Gemini） | 显式 `CachedContent` API | 按存储计费；读约为正常的 25% | 每个 token·小时存储费 | 用户设置（默认 1 小时） | 4,096 token（Flash），32,768（Pro） |

**不变式。** 三家都只缓存前缀。如果请求之间有任何 token 不同，从第一个不同 token 之后的所有内容都未命中。把**稳定的**部分放在顶部，**多变的**部分放在底部。

### 缓存友好的布局

```
[系统提示词]          <-- 缓存这个
[工具定义]           <-- 缓存这个
[少样本示例]          <-- 缓存这个
[检索到的文档]         <-- 如果复用则缓存，否则不缓存
[对话历史]            <-- 缓存到上一个回合
[当前用户消息]         <-- 从不缓存（每次都不同）
```

违反顺序——将用户消息置于系统提示词之上、在少样本之间交错动态检索——缓存永远不会命中。

### 盈亏平衡计算

Anthropic 的 25% 写溢价意味着缓存块必须至少被读取两次才能净节省。1 次写 + 1 次读平均每请求成本 0.675 倍（节省 32%）；1 次写 + 10 次读平均 0.205 倍（节省 80%）。经验法则：缓存任何你期望在 TTL 内至少复用 3 次的内容。

## 构建

### 步骤 1：Anthropic 带显式标记的提示词缓存

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` 标记告诉 Anthropic 将该块存储 5 分钟。在此窗口内复用命中；超过后过期并重新写入。

**响应 usage 字段：**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # 按 1.25 倍计费
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # 按 0.1 倍计费
```

在 CI 中检查这两个字段——如果 `cache_read_input_tokens` 在跨请求时始终为零，说明你的缓存键在漂移。

### 步骤 2：一小时延长 TTL

对于长时间运行的批处理作业，5 分钟默认值会在作业之间过期。设置 `ttl`：

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1 小时 TTL 的写溢价是 2 倍（比基线高 50%，而不是 25%），但在任何将前缀复用超过 5 次的批处理上都能快速回本。

### 步骤 3：OpenAI 自动缓存

OpenAI 不给你任何配置选项。任何超过 1,024 token 且与最近请求匹配的前缀都会自动获得 50% 折扣。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # 长且稳定
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # 折扣部分
```

相同的缓存友好布局规则也适用。两件事会杀死 OpenAI 的缓存但不会杀死 Anthropic 的：更改 `user` 字段（用作缓存键组件）和重新排序工具。

### 步骤 4：Gemini 显式上下文缓存

Gemini 将缓存作为一等对象你创建并命名：

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini 按每个 token·小时向存储收费，缓存存活多久就收多久费，读取约为正常输入费率的 25%。当你跨多个会话在数天内复用同一个巨大提示词时，这种模式是正确的选择。

### 步骤 5：在生产环境中测量命中率

参见 `code/main.py`，其中有一个模拟的三提供商会计员，追踪写/读/未命中计数并计算每 1K 请求的混合成本。在目标命中率上设置部署门槛——大多数生产级 Anthropic 设置在预热后应看到 >80% 的读取比例。

## 2026 年仍会发货的陷阱

- **动态时间戳在顶部。** 系统提示词顶部的 `"Current time: 2026-04-22 15:30:02"`。每次请求都未命中。将时间戳移到缓存断点以下。
- **工具重新排序。** 以稳定顺序序列化工具——部署之间的字典重排会打破每次命中。
- **自由文本近似重复。** "You are helpful." vs "You are a helpful assistant."——一个字节的差异 = 完全未命中。
- **过小的块。** Anthropic 强制执行 1,024 token 的下限（Haiku 为 2,048）。更小的块静默地不缓存。
- **盲目的成本仪表板。** 将"输入 token"拆分为缓存 vs 非缓存。否则流量下降看起来像缓存命中。

## 使用

2026 年的缓存技术栈：

| 场景 | 选择 |
|-----------|------|
| 具有稳定 10k+ 系统提示词、多回合的智能体 | Anthropic `cache_control`，5 分钟 TTL |
| 复用前缀 30+ 分钟的批处理作业 | Anthropic `ttl: "1h"` |
| GPT-5 上的无服务器端点，无自定义基础设施 | OpenAI 自动（只需使你的前缀稳定且长） |
| 多天复用大型代码/文档语料库 | Gemini 显式 `CachedContent` |
| 跨提供商回退 | 保持跨提供商的可缓存前缀布局相同，以便任何命中都能生效 |

与语义缓存（阶段 11 · 11）结合用于用户消息层：提示词缓存处理*token 完全相同*的复用，语义缓存处理*含义相同*的复用。

## 发货

保存 `outputs/skill-prompt-caching-planner.md`：

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## 练习

1. **简单。** 对 Claude 进行 10 回合对话，系统提示词 5,000 token。先不带 `cache_control` 运行，然后带。再报告每次的输入 token 账单。
2. **中等。** 编写一个测试工具，给定一个提示词模板和一个请求日志，计算每个提供商（Anthropic 5m、Anthropic 1h、OpenAI 自动、Gemini 显式）的预期命中率和美元节省。
3. **困难。** 构建一个布局优化器：给定一个提示词和标记为 `stable=True/False` 的字段列表，重写提示词以将单个缓存断点放在最大缓存友好的位置而不丢失信息。在真实的 Anthropic 端点上验证。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 提示词缓存 | "让长提示词变得廉价" | 复用提供商侧的 KV-cache 以匹配前缀；重复输入 token 降低 50-90%。 |
| `cache_control` | "Anthropic 标记" | 内容块属性，声明"到此处为止都是可缓存的"；`{"type": "ephemeral"}`。 |
| 缓存写入 | "支付溢价" | 首次填充缓存的请求；在 Anthropic 上按约 1.25 倍输入费率计费，OpenAI 免费。 |
| 缓存读取 | "折扣" | 匹配前缀的后续请求；按 10%（Anthropic）、50%（OpenAI）、约 25%（Gemini）计费。 |
| TTL | "存活时间" | 缓存保持热状态的时间秒数；Anthropic 默认 5m（可延长 1h），OpenAI 尽力而为最多 1h，Gemini 用户设置。 |
| 延长 TTL | "1 小时 Anthropic 缓存" | `{"type": "ephemeral", "ttl": "1h"}`；2 倍写溢价，但对于批处理复用值得。 |
| 前缀匹配 | "为什么我的缓存未命中" | 只有从开始到断点的每个 token 完全字节相同时才会命中。 |
| 上下文缓存（Gemini） | "显式那个" | Google 的命名、按存储计费的缓存对象；最适合多天复用大型语料库。 |

## 延伸阅读

- [Anthropic — 提示词缓存](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`、1h TTL、盈亏平衡表。
- [OpenAI — 提示词缓存](https://platform.openai.com/docs/guides/prompt-caching) — 自动前缀匹配。
- [Google — 上下文缓存](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` API 和存储定价。
- [Anthropic 工程 — 长上下文工作负载的提示词缓存](https://www.anthropic.com/news/prompt-caching) — 附带延迟数字的原始发布帖。
- 阶段 11 · 05（上下文工程）—— 如何分割提示词以便缓存落地。
- 阶段 11 · 11（缓存与成本）—— 在用户消息上配对语义缓存。
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) — 提示词缓存向用户暴露的 KV-cache 内存模型；解释为什么缓存前缀重读比重新计算便宜约 10 倍。
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) — prefill 是提示词缓存捷径的阶段；本文解释了为什么在缓存命中时 TTFT 急剧下降而 TPOT 不受影响。
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) — 提示词缓存与投机解码、Flash Attention 和 MQA/GQA 并列为弯曲推理成本曲线的杠杆；阅读本文了解其他三个。
