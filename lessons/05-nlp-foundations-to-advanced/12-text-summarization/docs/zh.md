# 文本摘要

> 抽取式系统告诉你文档说了什么。生成式系统告诉你作者的意思是什么。不同的任务，不同的陷阱。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 5 · 02（BoW + TF-IDF）、阶段 5 · 11（机器翻译）
**时间：** 约 75 分钟

## 问题

一篇 2000 字的新闻文章出现在你的信息流中。你需要 120 个词来概括它。你可以从文章中选取三个最重要的句子（抽取式），或者用自己的话重写内容（生成式）。两者都称为摘要。它们是完全不同的问题。

抽取式摘要是排序问题。给每个句子打分，返回前 k 个。输出总是语法正确的，因为它是逐字提取的。风险是遗漏分布在文章各处的内容。

生成式摘要是生成问题。Transformer 在输入条件下产生新文本。输出流畅且压缩性强，但可能产生源文本中没有的事实幻觉。风险是自信的捏造。

本课构建两种方法，以及各自拥有的失败模式。

## 概念

![抽取式 TextRank vs 生成式 transformer](../assets/summarization.svg)

**抽取式。** 将文章视为一个图，其中节点是句子，边是相似度。在图上运行 PageRank（或类似算法）来按句子与所有其他句子的连接程度打分。得分最高的句子构成摘要。典型实现是 **TextRank**（Mihalcea 和 Tarau，2004）。

**生成式。** 在文档-摘要对上微调 Transformer 编码器-解码器（BART、T5、Pegasus）。在推理时，模型读取文档，通过交叉注意力逐 token 生成摘要。Pegasus 特别使用间隙句子的预训练目标，使其无需太多微调就能出色地完成摘要。

使用 **ROUGE**（面向摘要的召回率辅助研究）评估。ROUGE-1 和 ROUGE-2 评分 unigram 和 bigram 重叠。ROUGE-L 评分最长公共子序列。越高越好，但 40 ROUGE-L 是 "良好"，50 是 "卓越"。每篇论文都报告这三个。使用 `rouge-score` 包。

## 构建

### 第 1 步：TextRank（抽取式）

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

有两个值得注意的点。相似度函数使用对数归一化的词重叠，这是原始 TextRank 变体。TF-IDF 向量的余弦也可以。阻尼因子 0.85 和迭代次数是 PageRank 的默认值。

### 第 2 步：使用 BART 的生成式

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(长新闻文章文本)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN 在 CNN/DailyMail 语料库上进行了微调。它开箱即用，能生成新闻风格的摘要。对于其他领域（科学论文、对话、法律），使用相应的 Pegasus 检查点，或在目标数据上进行微调。

### 第 3 步：ROUGE 评估

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

始终使用词干提取。没有它，"running" 和 "run" 被计为不同的词，ROUGE 会低估。

### 超越 ROUGE（2026 年摘要评估）

ROUGE 二十年来一直是主导的摘要指标，但在 2026 年仅靠它是不够的。一项大规模 NLG 论文荟萃分析表明：

- **BERTScore**（上下文嵌入相似度）在 2023 年获得关注，现在大多数摘要论文都与 ROUGE 一起报告。
- **BARTScore** 将评估视为生成：用预训练 BART 在给定源文本条件下分配给摘要的概率来评分。
- **MoverScore**（基于上下文嵌入的 Earth Mover 距离）在 2025 年摘要基准测试中位居榜首，因为它比 ROUGE 更好地捕捉语义重叠。
- **FactCC** 和 **基于 QA 的忠实度** 在 2021-2023 年很常见，现在经常被 **G-Eval**（GPT-4 提示链，用思维链推理评分连贯性、一致性、流畅性、相关性）取代。
- **G-Eval** 和类似的 LLM 即评判方法，当评分标准设计良好时，与人类判断的一致率约为 80%。

生产建议：为遗留比较报告 ROUGE-L，为语义重叠报告 BERTScore，为连贯性和事实性报告 G-Eval。用 50-100 个人工标注的摘要进行校准。

### 第 4 步：事实性问题

生成式摘要容易产生幻觉。抽取式摘要的幻觉风险要低得多，因为输出是从源文本逐字提取的，尽管如果源句子被去语境化、过时或按错误顺序引用，它们仍可能误导。这是生产系统仍对合规相关内容首选抽取式方法的最大原因。

需要命名的幻觉类型：

- **实体替换。** 源文本说 "John Smith"。摘要说 "John Brown"。
- **数字漂移。** 源文本说 "25,000"。摘要说 "2500 万"。
- **极性翻转。** 源文本说 "rejected the offer"。摘要说 "accepted the offer"。
- **事实捏造。** 源文本未提及 CEO。摘要说 CEO 批准了。

有效的评估方法：

- **FactCC。** 在源句子和摘要句子之间的 entailment 上训练的二元分类器。预测事实/非事实。
- **基于 QA 的事实性。** 向 QA 模型提问，其答案在源文本中。如果摘要支持不同答案，则标记。
- **实体级 F1。** 比较源文本和摘要中的命名实体。仅出现在摘要中的实体值得怀疑。

对于任何面向用户且事实性重要的内容（新闻、医学、法律、金融），抽取式是更安全的默认选择。生成式需要在循环中进行事实性检查。

## 使用

2026 年技术栈：

| 使用场景 | 推荐 |
|---------|-------------|
| 新闻、3-5 句摘要、英语 | `facebook/bart-large-cnn` |
| 科学论文 | `google/pegasus-pubmed` 或调优的 T5 |
| 多文档、长篇 | 具有 32k+ 上下文的任何 LLM，带提示 |
| 对话摘要 | `philschmid/bart-large-cnn-samsum` |
| 抽取式、结构上幻觉风险低 | TextRank 或 `sumy` 的 LSA / LexRank |

在 2026 年，当计算不是约束时，具有长上下文的 LLM 通常优于专业模型。权衡是成本和可重复性；专业模型给出更一致的输出。

## 发布

保存为 `outputs/skill-summary-picker.md`：

```markdown
---
name: summary-picker
description: 选择抽取式或生成式、命名库、事实性检查。
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

给定任务（文档类型、合规要求、长度、计算预算），输出：

1. 方法。抽取式或生成式。用一句话解释原因。
2. 起始模型/库。命名它。`sumy.TextRankSummarizer`、`facebook/bart-large-cnn`、`google/pegasus-pubmed` 或 LLM 提示。
3. 评估计划。ROUGE-1、ROUGE-2、ROUGE-L（使用带词干提取的 rouge-score）。如果生成式，加上事实性检查。
4. 一个需要探究的失败模式。在生成式新闻摘要中，实体替换最常见；标记源实体未出现在摘要中的样本。

对于医学、法律、金融或受监管内容，未经事实性门控，拒绝生成式摘要。标记超过模型上下文窗口的输入为需要分块 map-reduce 摘要（不仅仅是截断）。
```

## 练习

1. **简单。** 在 5 篇新闻文章上运行 TextRank。将前 3 个句子与参考摘要进行比较。测量 ROUGE-L。你应该在 CNN/DailyMail 风格的文章上看到 30-45 的 ROUGE-L。
2. **中等。** 实现实体级事实性：从源文本和摘要中提取命名实体（spaCy），计算源实体在摘要中的召回率和摘要实体相对于源文本的精确率。高精确率和低召回率意味着安全但简洁；低精确率意味着存在幻觉实体。
3. **困难。** 在 50 篇 CNN/DailyMail 文章上比较 BART-large-CNN 与 LLM（Claude 或 GPT-4）。报告 ROUGE-L、事实性（按实体 F1）和每个摘要的成本。记录各自的优势。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Extractive（抽取式） | 选取句子 | 从源文本逐字返回句子。永远不会幻觉。 |
| Abstractive（生成式） | 重写 | 在源文本条件下生成新文本。可能产生幻觉。 |
| ROUGE | 摘要指标 | 系统输出与参考之间的 n-gram / LCS 重叠。 |
| TextRank | 基于图的抽取式 | 在句子相似度图上的 PageRank。 |
| Factuality（事实性） | 是否正确 | 摘要声明是否被源文本支持。 |
| Hallucination（幻觉） | 编造的内容 | 摘要中源文本不支持的内容。 |

## 延伸阅读

- [Mihalcea 和 Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — 抽取式经典论文。
- [Lewis 等 (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — BART 论文。
- [Zhang 等 (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus 和间隙句子目标。
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — ROUGE 论文。
- [Maynez 等 (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — 事实性格局论文。
