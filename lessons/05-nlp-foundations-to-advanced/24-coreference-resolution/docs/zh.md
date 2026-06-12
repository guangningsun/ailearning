# 共指消解

> "她给他打了电话。他没有接。医生当时在吃午饭。"三个指称，两个人，却没有人被点名。共指消解弄清楚谁是谁。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 5 · 06（命名实体识别）、阶段 5 · 07（词性标注与句法分析）
**时间：** 约 60 分钟

## 问题

从一篇 300 词的文章中提取所有对 Apple Inc. 的提及。文章说"Apple"时很容易。当它说"the company"、"they"、"Cupertino 的科技巨头"或"Jobs 的公司"时，就难了。如果不把这些指称解析到同一个实体，你的 NER 流程会错过 60-80% 的提及。

共指消解将每一个指向同一现实世界实体的表达式链接成一个聚类。它是表层 NLP（NER、句法分析）和下游语义（信息抽取、问答、摘要、知识图谱）之间的粘合剂。

为什么它在 2026 年很重要：

- 摘要："CEO 宣布了……"与"Tim Cook 宣布了……"—— 摘要应该点出 CEO 的名字。
- 问答："她给谁打了电话？"需要解析"她"。
- 信息抽取：知识图谱中有"PER1 创立了 Apple"和"Jobs 创立了 Apple"两个独立条目是错误的。
- 跨文档信息抽取：合并关于同一事件的多篇文章中的指称是跨文档共指消解。

## 概念

![共指聚类：指称 → 实体](../assets/coref.svg)

**任务。** 输入：一份文档。输出：指称（跨度）的聚类，每个聚类指向一个实体。

**指称类型。**

- **命名实体。** "Tim Cook"
- **名词性指称。** "the CEO"、"the company"
- **代词性指称。** "he"、"she"、"they"、"it"
- **同位语。** "Tim Cook, Apple's CEO,"

**架构。**

1. **基于规则（Hobbs，1978）。** 基于句法树的代词消解，使用语法规则。不错的基线。在代词上出人意料地难以超越。
2. **指称对分类器。** 对每对指称（m_i, m_j），预测它们是否共指。通过传递闭包聚类。2016 年前的标准方法。
3. **指称排序。** 对每个指称，对候选先行词（包括"无先行词"）排序。选择排名最高的。
4. **基于跨度的端到端（Lee et al.，2017）。** Transformer 编码器。枚举所有长度上限以内的候选跨度。预测指称分数。预测每个跨度的先行词概率。贪心聚类。现代的默认选择。
5. **生成式（2024+）。** 给 LLM 提示："列出这段文本中每个代词及其先行词。"在简单案例上效果不错，在长文档和罕见指称上表现挣扎。

**评估指标。** 五个标准指标（MUC、B³、CEAF、BLANC、LEA），因为没有单一指标能捕捉聚类质量。报告前三个的平均值作为 CoNLL F1。2026 年在 CoNLL-2012 上的最新水平：约 83 F1。

**已知的困难案例。**

- 定指描述指向几百字前引入的实体。
- 桥接回指（"the wheels" → 之前提到的汽车）。
- 中文和日语等语言中的零形回指。
- 逆序指代（代词在指称之前）："当**她**走进来时，Mary 笑了。"

## 动手实现

### 第 1 步：预训练神经共指（AllenNLP / spaCy-experimental）

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

在较长的文档上，你会得到类似这样的结果：
- 聚类 1：[Apple, The company, they]
- 聚类 2：[new products]

### 第 2 步：基于规则的代词解析器（教学用）

参见 `code/main.py` 中的纯标准库实现：

1. 提取指称：命名实体（首字母大写的跨度）、代词（查字典）、定指描述（"the X"）。
2. 对每个代词，查看前 K 个指称并按以下方式评分：
   - 性别/数的一致性（启发式）
   - 最近性（更近的赢）
   - 句法角色（优先主语）
3. 链接得分最高的先行词。

与神经模型相比没有竞争力。但它展示了搜索空间和端到端模型必须做出的决策。

### 第 3 步：使用 LLM 做共指消解

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

两种需要关注的失败模式。第一，LLM 过度合并（"him"和"her"指向两个不同的人）。第二，LLM 在长文档中静默丢弃指称。始终用跨度偏移检查来验证。

### 第 4 步：评估

标准的 conll-2012 脚本计算 MUC、B³、CEAF-φ4 并报告平均值。对于内部评估，从标注测试集上的跨度级精确率和召回率开始，然后加上指称链接 F1。

## 陷阱

- **单例爆炸。** 一些系统将每个指称报告为自己的聚类。B³ 比较宽松。MUC 会惩罚这种做法。始终检查所有三个指标。
- **长上下文中的代词。** 在超过 2,000 token 的文档上性能下降约 15 F1。谨慎分块。
- **性别假设。** 硬编码的性别规则在非二元指称者、组织、动物上会出问题。使用学习到的模型或中性评分。
- **LLM 在长文档上的漂移。** 单次 API 调用无法可靠地跨 50+ 段聚类指称。使用滑动窗口 + 合并。

## 实际使用

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| 英语，单文档 | `en_coreference_web_trf`（spaCy-experimental）或 AllenNLP 神经共指 |
| 多语言 | 在 OntoNotes 或 Multilingual CoNLL 上训练的 SpanBERT / XLM-R |
| 跨文档事件共指 | 专用端到端模型（2025-26 最新水平） |
| 快速 LLM 基线 | GPT-4o / Claude 配合结构化输出共指提示 |
| 生产对话系统 | 基于规则的备选 + 神经主模型 + 关键槽位人工审核 |

2026 年 shipping 的集成模式：先跑 NER，再跑共指，再将共指聚类合并到 NER 实体中。下游任务看到的是每个聚类一个实体，而不是每个指称一个实体。

## 交付物

保存为 `outputs/skill-coref-picker.md`：

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## 练习

1. **简单。** 在 5 个手工制作的段落上运行 `code/main.py` 中的基于规则的解析器。相对于真值测量指称链接准确率。
2. **中等。** 在一篇新闻文章上使用预训练神经共指模型。将聚类与你自己的手动标注进行比较。它在哪里失败了？
3. **困难。** 构建一个共指增强的 NER 流程：先 NER，再通过共指聚类合并。在 100 篇文章上测量实体覆盖率相对于纯 NER 的提升。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 指称（Mention） | 一个引用 | 指向一个实体的文本跨度（名字、代词、名词短语）。 |
| 先行词（Antecedent） | "it" 指的是什么 | 后续指称与其共指的前面那个指称。 |
| 聚类（Cluster） | 实体的指称集合 | 全部指向同一个现实世界实体的指称集合。 |
| 回指（Anaphora） | 向后引用 | 后续指称指向前面的（"he" → "John"）。 |
| 逆序指代（Cataphora） | 向前引用 | 前面的指称指向后面的（"When he arrived, John..."）。 |
| 桥接（ Bridging） | 隐式引用 | "我买了一辆车。车轮坏了。"（那辆车的车轮。） |
| CoNLL F1 | 排行榜上的数字 | MUC、B³、CEAF-φ4 F1 分数的平均值。 |

## 进一步阅读

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 权威教科书章节。
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — 基于跨度的端到端方法。
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — 改进了共指的预训练。
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — 基准测试。
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — 基于规则的经典方法。