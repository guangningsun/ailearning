# 关系抽取与知识图谱构建

> NER 找到了实体。实体链接将其锚定。关系抽取找到它们之间的边。知识图谱是节点、边及其来源的总和。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 5 · 06（命名实体识别）、阶段 5 · 25（实体链接）
**时间：** 约 60 分钟

## 问题

一位分析师读到："Tim Cook became CEO of Apple in 2011." 四个事实：

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

关系抽取（RE）将自由文本转换为结构化三元组 `(subject, relation, object)`。跨语料库聚合就得到知识图谱。聚合后再查询，你就有了 RAG、分析或合规审计的推理基座。

2026 年的问题：LLM 热情地抽取关系。太热情了。它们会产生源文本不支持的三元组。没有来源，你就无法区分真实三元组和看似合理的虚构。2026 年的答案是 AEVS 风格的锚定-验证管道。

## 概念

![文本 → 三元组 → 知识图谱](../assets/relation-extraction.svg)

**三元组形式。** `(subject_entity, relation_type, object_entity)`。关系来自封闭本体论（Wikidata 属性、FIBO、UMLS）或开放集（OpenIE 风格，任何关系皆可）。

**三种抽取方法。**

1. **规则 / 模式驱动。** Hearst 模式："X such as Y" → `(Y, isA, X)`。加上手工正则。脆弱但精确、可解释。
2. **监督分类器。** 给定句子中两个实体提及，从固定集合中预测关系。在 TACRED、ACE、KBP 上训练。2015–2022 年的标准方法。
3. **生成式 LLM。** 提示模型发出三元组。开箱即用。需要来源，否则会生成看似合理实则无效的垃圾。

**AEVS（锚定-抽取-验证-补充，2026）。** 当前的幻觉缓解框架：

- **锚定。** 用精确位置识别每个实体跨度 和关系短语跨度。
- **抽取。** 生成链接到锚定inter的inter。
- **验证。** 将每个三元组元素匹配回源文本；拒绝任何不支持的内容。
- **补充。** 覆盖率pass确保没有锚定的跨度被丢弃。

幻觉显著下降。需要更多计算但可审计。

**开放 vs 封闭的权衡。**

- **封闭本体论。** 固定属性列表（如 Wikidata 的 11000+ 属性）。可预测。可查询。难以凭空捏造。
- **开放 IE。** 任何口头短语都成为关系。高召回率。低精确率。查询麻烦。

生产 K 通常混合使用：开放 IE 用于发现，然后在合并到主图之前将关系规范化为封闭本体论。

## 构建它

### 第 1 步：基于模式的抽取

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

参见 `code/main.py` 获取完整的玩具抽取器。Hearst 模式仍然在领域特定管道中使用，因为它们可调试。

### 第 2 步：监督关系分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL 是一个 seq2seq 关系抽取器：文本输入，三元组输出，已经是 Wikidata 属性 id。在远监督数据上微调。标准的开源基线。

### 第 3 步：带锚定的 LLM 提示抽取

```python
prompt = f"""从文本中抽取 (subject, relation, object) 三元组。
对于每个三元组，包含源文本中的精确字符跨度。

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

只包含完全由文本支持的三元组。不要超出文本所述内容进行推断。
"""
```

验证每个返回的跨度与源文本的匹配。拒绝任何 `text[start:end] != triple_entity` 的内容。这是 AEVS"验证"步骤的最简形式。

### 第 4 步：规范化为封闭本体论

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (subject/object inverted)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

规范化通常是 60-80% 的工程工作。把它算进去。

### 第 5 步：构建小图并查询

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

这是每个 RAG-over-KG 系统的原子。用 RDF 三元组存储（Blazegraph、Virtuoso）、属性图（Neo4j）或向量增强图存储来扩展它。

## 陷阱

- **RE 前做共指。** "He founded Apple"——RE 需要知道"he"是谁。先跑共指（第 24 课）。
- **实体规范化。** "Apple Inc" 和 "Apple" 必须解析到同一节点。先做实体链接（第 25 课）。
- **三元组幻觉。** LLM 发出文本不支持的三元组。强制执行跨度验证。
- **关系规范化漂移。** 开放 IE 关系不一致（"was born in"、"came from"、"is a native of"）。折叠到规范 id，否则图无法查询。
- **时间错误。** "Tim Cook is CEO of Apple"——现在是真的，2005 年是假的。许多关系是有时间范围的。使用限定符（Wikidata 中的 P580 开始时间、P582 结束时间）。
- **领域不匹配。** REBEL 在 Wikipedia 上训练。法律、医疗和科学文本通常需要领域微调的 RE 模型。

## 使用它

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| 快速生产、通用领域 | REBEL 或 LlamaPred + Wikidata 规范化 |
| 领域特定（生物医学、法律） | SciREX 风格领域微调 + 定制本体论 |
| LLM 提示、审计输出 | AEVS 管道：锚定 → 抽取 → 验证 → 补充 |
| 大容量新闻 IE | 基于模式 + 监督混合 |
| 从零构建 KG | 开放 IE + 手动规范化 pass |
| 时序 KG | 带限定符抽取（开始/结束时间、时间点） |

集成模式：NER → 共指 → 实体链接 → 关系抽取 → 本体映射 → 图加载。每个阶段都是潜在的质量关卡。

## 交付它

保存为 `outputs/skill-re-designer.md`：

```markdown
---
name: re-designer
description: 设计一个带来源和规范化的关系抽取管道。
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

给定一个语料库（领域、语言、体量）和下游用途（KG-RAG、分析、合规），输出：

1. 抽取器。基于模式 / 监督 / LLM / AEVS 混合。理由与精确率 vs 召回率目标相关。
2. 本体论。封闭属性列表（Wikidata / 领域）或带规范化 pass 的开放 IE。
3. 来源。每个三元组带源字符跨度 + 文档 id。审计的必选项。
4. 合并策略。规范实体 id + 关系 id + 时序限定符；去重策略。
5. 评估。200 个手工标注三元组上的精确率 / 召回率 + LLM 抽取样本上的幻觉率。

在没有跨度验证（源来源）的情况下拒绝任何基于 LLM 的 RE 管道。在没有规范化的 flowing 到生产图的情况下拒绝开放 IE 输出。标记那些在时间范围关系（雇主、配偶、职位）上没有时序限定符的管道。
```

## 练习

1. **简单。** 在 5 篇新闻文章句子上运行 `code/main.py` 中的模式抽取器。手工检查精确率。
2. **中等。** 在同一组句子上使用 REBEL（或小 LLM）。比较三元组。哪个抽取器精确率更高？召回率更高？
3. **困难。** 构建 AEVS 管道：LLM 抽取 + 验证源文本跨度。在 50 个 Wikipedia 风格句子上测量验证步骤前后幻觉率。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 三元组 | 主语-关系-宾语 | `(s, r, o)` 元组，是 KG 的原子单位。 |
| 开放 IE | 抽取任何关系 | 开放词汇关系短语；高召回率，低精确率。 |
| 封闭本体论 | 固定模式 | 有界的relation类型集合（Wikidata、UMLS、FIBO）。 |
| 规范化 | 标准化一切 | 将表面名称 / 关系映射到规范 id。 |
| AEVS | 接地抽取 | 锚定-抽取-验证-补充管道（2026）。 |
| 来源 | 真实来源链接 | 每个三元组带文档 id + 字符跨度指向其源。 |
| 远监督 | 廉价标签 | 将文本与现有 KG 对齐以创建训练数据。 |

## 延伸阅读

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — 远监督论文。
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq RE 主力。
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — 联合 IE。
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — 2026 幻觉缓解设计。
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — 规范图查询。