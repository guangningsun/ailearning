# 结构化输出与约束解码

> 让 LLM 返回 JSON。大部分时候能得到 JSON。在生产环境中，"大部分"就是问题所在。约束解码通过在采样前修改 logits 将"大部分"变成"百分之百"。

**类型：** 建造型
**语言：** Python
**前置条件：** 阶段 5 · 17（聊天机器人）、阶段 5 · 19（子词分词）
**时间：** 约 60 分钟

## 问题

分类器给 LLM 发提示："返回 {positive, negative, neutral} 之一。"模型返回"The sentiment is positive — this review is overwhelmingly favorable because the customer explicitly states that they ..."。你的解析器崩溃了。你的分类器 F1 是 0.0。

自由形式生成不是契约，只是一个建议。生产系统需要契约。

2026 年存在三个层次。

1. **提示工程。** 好好说。"只返回 JSON 对象。"在前沿模型上大概 80% 有效，小模型效果更差。
2. **原生结构化输出 API。** OpenAI `response_format`、Anthropic 工具使用、Gemini JSON 模式。在支持的模式上可靠，但会锁定供应商。
3. **约束解码。** 在每个生成步骤修改 logits，使模型*无法*发出无效 token。由构造保证 100% 有效。在任何本地模型上都有效。

本课建立对以上三层的直觉，并指出何时该选择哪种。

## 概念

![约束解码在每一步屏蔽无效 token](../assets/constrained-decoding.svg)

**约束解码如何工作。** 在每个生成步骤，LLM 在整个词表上产生一个 logit 向量（约 100k 个 token）。*Logit 处理器*位于模型和采样器之间。它根据目标语法（JSON Schema、正则表达式、上下文无关语法）在当前位置计算哪些 token 是有效的——并将所有无效 token 的 logits 设为负无穷。对剩余 logits 的 softmax 将概率质量集中在有效的延续上。

2026 年的实现：

- **Outlines。** 将 JSON Schema 或正则表达式编译成有限状态机。每个 token 都有 O(1) 的有效下一步 token 查找。基于 FSM，所以递归模式需要展平。
- **XGrammar / llguidance。** 上下文无关语法引擎。处理递归 JSON Schema。解码开销接近零。OpenAI 在 2025 年结构化输出实现中 credited llguidance。
- **vLLM 引导解码。** 内置 `guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`，通过 Outlines、XGrammar 或 lm-format-enforcer 后端。
- **Instructor。** 基于 Pydantic 的任意 LLM 包装器。验证失败时重试。跨供应商，但不修改 logits——依赖重试 + 结构化输出感知提示。

### 反直觉的结果

约束解码通常*比*无约束生成*更快*。两个原因。第一，它缩小了下一个 token 的搜索空间。第二，聪明的实现完全跳过强制 token（如脚手架 `{"name": "`——每个字节都已确定）的生成。

###会让你付出代价的坑

字段顺序很重要。把 `answer` 放在 `reasoning` 前面，模型在思考之前就提交了答案。JSON 是有效的。答案是错的。没有任何验证能catch到。

```json
// 不好
{"answer": "yes", "reasoning": "because ..."}

// 好
{"reasoning": "... therefore ...", "answer": "yes"}
```

Schema 字段顺序就是逻辑，不是格式。

## 动手实现

### 第 1 步：从零实现正则约束生成

参见 `code/main.py` 获取独立的 FSM 实现。核心思想 30 行代码：

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM 跟踪到目前为止我们已满足的语法部分。`valid_tokens(state, tokenizer)` 计算哪些词表 token 可以推进 FSM 而不离开接受路径。

### 第 2 步：用 Outlines 处理 JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

零验证错误。永远。FSM 使无效输出不可达。

### 第 3 步：用 Instructor 实现供应商无关的 Pydantic

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

不同的机制。Instructor 不触碰 logits。它将模式格式化为提示，解析输出，验证失败时重试（默认 3 次）。适用于任何供应商。重试增加延迟和成本。跨供应商可移植性是卖点。

### 第 4 步：原生供应商 API

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

服务端约束解码。对于支持的模式，可靠性与 Outlines 持平。不需要本地模型管理。但会锁定供应商。

## 坑

- **递归模式。** Outlines 将递归展平为固定深度。树结构输出（嵌套评论、AST）需要 XGrammar 或 llguidance（基于 CFG）。
- **超大枚举。** 10,000 个选项的枚举编译慢或超时。换用检索器：先预测 top-k 候选项，然后约束到这些。
- **语法太严格。** 强制 `date: "YYYY-MM-DD"` 正则，模型无法输出缺失日期的 `"unknown"`。模型通过编造日期来补偿。允许 `null` 或哨兵值。
- **过早提交。** 见上文字段顺序坑。总是把推理放在前面。
- **无模式的供应商 JSON 模式。** 纯 JSON 模式只保证有效 JSON，不保证*适用于你的用例*。总要提供完整模式。

## 实际使用

2026 年技术栈：

|场景 | 选择 |
|-----------|------|
| OpenAI/Anthropic/Google 模型，简单模式 | 原生供应商结构化输出 |
| 任何供应商，Pydantic 工作流，可容忍重试 | Instructor |
| 本地模型，需要 100% 有效性，平面模式 | Outlines（FSM） |
| 本地模型，递归模式 | XGrammar 或 llguidance |
| 自托管推理服务器 | vLLM 引导解码 |
| 可容忍重试的批处理 | Instructor + 最便宜的模型 |

## 交付物

保存为 `outputs/skill-structured-output-picker.md`：

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## 练习

1. **简单。** 不带约束解码地提示一个小开源模型（如 Llama-3.2-3B）输出 `Review(sentiment, confidence, evidence_span)`。测量 100 条评论中解析为有效 JSON 的比例。
2. **中等。** 同一语料库用 Outlines JSON 模式。比较合规率、延迟和语义准确率。
3. **困难。** 从零实现电话号码的正则约束解码器（`\d{3}-\d{3}-\d{4}`）。在 1000 个样本上验证零无效输出。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 约束解码 | 强制有效输出 | 在每个生成步骤将无效 token 的 logits 屏蔽。 |
| Logit 处理器 | 那个约束的东西 | 函数：`(logits, state) -> masked_logits`。 |
| FSM | 有限状态机 | 编译后的语法表示；O(1) 有效下一步 token 查找。 |
| CFG | 上下文无关语法 | 能处理递归的语法；比 FSM 慢但表达能力更强。 |
| Schema 字段顺序 | 这重要吗？ | 重要——第一个字段会提交；总是把推理放在答案前面。 |
| 引导解码 | vLLM 的叫法 | 相同概念，集成到推理服务器中。 |
| JSON 模式 | OpenAI 早期版本 | 保证 JSON 语法；不保证模式匹配。 |

## 延伸阅读

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — Outlines 论文。
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) —快速基于 CFG 的约束解码。
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — 推理服务器集成。
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — API 参考 + 陷阱。
- [Instructor library](https://python.useinstructor.com/) — 跨供应商的 Pydantic + 重试。
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — 6 个约束解码框架的基准测试。