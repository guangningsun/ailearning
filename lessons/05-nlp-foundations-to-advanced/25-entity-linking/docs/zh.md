# 实体链接与消歧

> NER 找到了"Paris"。实体链接要决定：是法国巴黎？Paris Hilton？德克萨斯州巴黎？还是特洛伊王子Paris？没有链接，你的知识图谱始终是模糊的。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 5 · 06（命名实体识别）、阶段 5 · 24（共指消解）
**时间：** 约 60 分钟

## 问题

一句话："Jordan beat the press." NER 把"Jordan"标注为 PERSON。好。但**哪个** Jordan？

- Michael Jordan（篮球运动员）？
- Michael B. Jordan（演员）？
- Michael I. Jordan（伯克利 ML 教授——是的，这种混淆在 ML 论文中是真实存在的）？
- Jordan（约旦国）？
- Jordan（希伯来语名字）？

实体链接（EL）将每个提及解析到知识库中的唯一条目：Wikidata、Wikipedia、DBpedia，或你的领域知识库。两个子任务：

1. **候选生成。** 给定"Jordan"，哪些 KB 条目是合理的？
2. **消歧。** 给定上下文，哪个候选是正确的？

两个步骤都是可学习的。都有基准测试。组合管道已经稳定了十年——变化的是消歧器的质量。

## 概念

![实体链接管道：提及 → 候选 → 消歧实体](../assets/entity-linking.svg)

**候选生成。** 给定提及的表面形式（"Jordan"），在别名索引中查找候选。Wikipedia 别名字典覆盖了大多数命名实体："JFK" → John F. Kennedy、Jacqueline Kennedy、JFK 机场、JFK（电影）。典型索引每个提及返回 10-30 个候选。

**消歧：三种方法。**

1. **先验 + 上下文（Milne & Witten, 2008）。** `P(entity | mention) × context-similarity(entity, text)`。效果好，速度快，无需训练。
2. **基于 Embedding（ESS / REL / BLINK）。** 对提及 + 上下文编码。对每个候选的描述编码。选取余弦相似度最高者。2020-2024 年的默认方法。
3. **生成式（GENRE, 2021; 基于 LLM, 2023+）。** 逐 token 解码实体的规范名称。约束解码（见第 20 课）确保只能输出有效的实体名称，因此输出保证是有效的 KB id。

**端到端 vs 管道。** 现代模型（ELQ、BLINK、ExtEnD、GENRE）在一次前向传播中完成 NER + 候选生成 + 消歧。管道系统在生产环境中仍然占主导地位，因为你可以替换组件。

### 两个衡量指标

- **提及召回率（候选生成）。** 金标准提及中正确 KB 条目出现在候选列表中的比例。整个管道的下限。
- **消歧准确率 / F1。** 给定正确候选，top-1 正确的频率。

始终报告两个。99% 消歧率但候选召回率只有 80% 的系统，整体是 80%。

## 构建它

### 第 1 步：从 Wikipedia 重定向构建别名索引

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia 别名数据：约 1800 万对（别名，实体）。从 Wikidata  dumps 下载。存储为倒排索引。

### 第 2 步：基于上下文的消歧

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard 重叠只是一个玩具。用 embedding 上的余弦相似度替换（见 `code/main.py` 第 2 步的 transformer 版本）。

### 第 3 步：基于 Embedding（BLINK 风格）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

在索引时间，一次性 embedding 每个 KB 实体。在查询时间，一次性 embedding 提及 + 上下文，与候选池做点积，选取最大值。

### 第 4 步：生成式实体链接（概念）

GENRE 逐字符解码实体的 Wikipedia 标题。约束解码（第 20 课）确保只能输出有效标题。与 KB 支持的 trie 紧密集成。其现代后继是 REL-GEN 和带结构化输出的 LLM 提示 EL。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

配合白名单（Outlines `choice`），这是 2026 年最简单的可交付 EL 管道。

### 第 5 步：在 AIDA-CoNLL 上评估

AIDA-CoNLL 是标准的 EL 基准：1393 篇路透社文章，34k 个提及，Wikipedia 实体。报告 KB 内准确率（`P@1`）和 KB 外 NIL 检测率。

## 陷阱

- **NIL 处理。** 有些提及不在 KB 中（新出现的实体、不知名的人）。系统必须预测 NIL 而不是猜错实体。单独测量。
- **提及边界错误。** 上游 NER 漏掉部分跨度（"Bank of America" 只标注了"Bank"）。EL 召回率下降。
- **热门偏置。** 训练过的系统过度预测常见实体。ML 论文中提到"Michael I. Jordan"经常链接到篮球 Jordan。
- **跨语言 EL。** 将中文文本中的提及映射到英文 Wikipedia 实体。需要多语言编码器或翻译步骤。
- **KB 时效性。** 新公司、新事件、新人物不在去年的 Wikipedia dumps 中。生产管道需要刷新循环。

## 使用它

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| 通用英文 + Wikipedia | BLINK 或 REL |
| 跨语言，KB = Wikipedia | mGENRE |
| LLM 友好、提及量少 | 用候选列表 + 约束 JSON 提示 Claude/GPT-4 |
| 领域特定 KB（医疗、法律） | 定制 BERT + KB 感知检索 + 领域 AIDA 风格微调 |
| 极低延迟 | 仅精确匹配先验（Milne-Witten 基线） |
| 研究 SOTA | GENRE / ExtEnD / 生成式 LLM-EL |

2026 年可交付的生产模式：NER → 共指 → 对每个提及做 EL → 将簇折叠为每个簇一个规范实体。输出：文档中每个实体对应一个 KB id，而非每个提及对应一个。

## 交付它

保存为 `outputs/skill-entity-linker.md`：

```markdown
---
name: entity-linker
description: 设计一个实体链接管道——KB、候选生成器、消歧器、评估。
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

给定一个用例（领域 KB、语言、体量、延迟预算），输出：

1. 知识库。Wikidata / Wikipedia / 定制 KB。版本日期。刷新频率。
2. 候选生成器。别名索引、embedding 或混合。目标提及召回率 @ K。
3. 消歧器。先验 + 上下文、基于 embedding、生成式，或 LLM 提示。
4. NIL 策略。top 分数阈值、分类器或显式 NIL 候选。
5. 评估。留出集上的提及召回率 @ 30、top-1 准确率、NIL 检测 F1。

在没有提及召回率基线的情况下拒绝任何 EL 管道（不知道候选生成是否 surfaced 了正确实体，就无法评估消歧器）。在没有约束输出到有效 KB id 的情况下拒绝任何 LLM 提示的 EL 管道。标记那些在未做领域微调的情况下热门偏置影响少数实体（如名字冲突）的系统。

## 练习

1. **简单。** 在 10 个歧义提及（Paris、Jordan、Apple）上实现 `code/main.py` 中的先验 + 上下文消歧器。手工标注正确实体。测量准确率。
2. **中等。** 用句子 transformer 编码 50 个歧义提及。Embedding 每个候选的描述。将基于 embedding 的消歧与 Jaccard 上下文重叠进行比较。
3. **困难。** 构建一个 1k 实体领域 KB（如公司员工 + 产品）。实现端到端 NER + EL。在 100 个留出句子上测量精确率和召回率。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 实体链接（EL） | 链接到 Wikipedia | 将提及映射到唯一 KB 条目。 |
| 候选生成 | 它可能是谁？ | 返回提及的合理 KB 条目候选短列表。 |
| 消歧 | 选正确的那个 | 用上下文对候选评分，选取胜者。 |
| 别名索引 | 查找表 | 从表面形式 → 候选实体的映射。 |
| NIL | 不在 KB 中 | 明确预测没有 KB 条目匹配。 |
| KB | 知识库 | Wikidata、Wikipedia、DBpedia，或你的领域 KB。 |
| AIDA-CoNLL | 基准 | 带有金标准实体链接的 1393 篇路透社文章。 |

## 延伸阅读

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — 先验 + 上下文方法的奠基之作。
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — 基于 embedding 的主力方法。
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — 带约束解码的生成式 EL。
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — 基准论文。
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — 开放生产技术栈。