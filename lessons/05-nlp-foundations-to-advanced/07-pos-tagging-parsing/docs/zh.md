# 词性标注与句法分析

> 语法曾经有一段时间不受待见。后来每个 LLM 流程都需要验证结构化抽取，语法又回来了。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 5 · 01（文本处理）、阶段 2 · 14（朴素贝叶斯）
**时间：** 约 45 分钟

## 问题

第一课承诺过，词形还原需要词性标注。不知道 `running` 是动词，词形还原器就无法将它还原为 `run`。不知道 `better` 是形容词，就无法还原为 `good`。

这个承诺背后藏着一个完整的子领域。词性标注为词赋予语法类别。句法分析还原句子的树结构：哪个词修饰哪个，哪个动词支配哪些论元。古典 NLP 花二十年打磨这两件事。随后深度学习把它们压缩成预训练 transformer 之上的 token 分类任务，研究社区转向了下一个战场。

但应用社区没有。每条结构化抽取流程在底层仍然使用 POS 和依存树。LLM 生成的 JSON 要按语法约束来验证。问答系统用依存分析分解查询。机器翻译质量评估器检查分析树的对齐情况。

值得了解。本课介绍词性标签集、基线模型，以及何时该停止从零实现、转向调用 spaCy。

## 概念

**词性标注（POS tagging）** 为每个 token 标注语法类别。**Penn Treebank（PTB）** 标签集是英文默认集。36 个标签，区分对普通读者来说过于讲究：`NN` 单数名词、`NNS` 复数名词、`NNP` 单数专有名词、`VBD` 动词过去式、`VBZ` 动词第三人称单数现在时，以此类推。**通用依存（Universal Dependencies, UD）** 标签集更粗粒度（17 个标签），且与语言无关；它成为跨语言工作的默认集。

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**句法分析（Syntactic parsing）** 生成一棵树。两种主要风格：

- **Constituency parsing（成分句法分析）。** 名词短语、动词短语、介词短语层层嵌套。输出是一棵以非终结类别（NP、VP、PP）为节点、词汇为叶子的树。
- **Dependency parsing（依存句法分析）。** 每个词都有一个它所依赖的中心词，并用语法关系标注。输出是一棵树，每条边都是一个（中心词，从属词，关系）三元组。

依存分析在 2010 年代胜出，因为它能干净地跨语言泛化，尤其适用于语序自由的语言。

```
running 是 ROOT
cats 是 running 的 nsubj
were 是 running 的 aux
at 是 running 的 prep
3pm 是 at 的 pobj
```

## 从零构建

### 第 1 步：最常见标签基线

最简单但能用的 POS 标注器。对于每个词，预测它在训练时出现次数最多的标签。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

在 Brown 语料库上，该基线达到约 85% 的准确率。不算好，但这是任何正经模型都不应跌破的底线。

### 第 2 步：二元 HMM 标注器

对序列的联合概率建模：

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

两张表：转移概率（前一个标签给定当前标签）、发射概率（标签给定当前词）。用拉普拉斯平滑从计数中估计两者。用 Viterbi 解码（动态规划穿越标签格子）。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

二元 HMM 在 Brown 上达到约 93% 的准确率。从 85% 到 93% 的跳跃主要来自转移概率——模型学到 `DET NOUN` 很常见而 `NOUN DET` 很罕见。

### 第 3 步：为什么现代标注器胜过它

转移概率和发射概率都是局部的。它们无法捕捉 `saw` 在 "I bought a saw" 中是名词、而在 "I saw the movie" 中是动词这一事实。具有任意特征的 CRF（后缀、词形、上下文词、词本身）达到约 97%。BiLSTM-CRF 或 transformer 达到约 98%+。

该任务的天花板由标注者分歧决定。Penn Treebank 上人类标注者约 97% 的时间是一致的。超过 98% 的模型可能是在过拟合测试集。

### 第 4 步：依存分析概述

从零实现完整依存分析超出范围；Jurafsky 和 Martin 的教科书中有权威论述。需要了解的两个经典家族：

- **基于转移的解析器**（arc-eager、arc-standard）运作方式类似移位-归约解析器：读取 token，将它们移入栈，并应用创建弧的归约动作。贪心解码速度快。经典实现是 MaltParser。现代神经网络版本：Chen 和 Manning 的基于转移的解析器。
- **基于图的解析器**（Eisner 算法、Dozat-Manning 双仿射）为每个可能的中心词-从属词边打分，然后选取最大生成树。速度较慢但更准确。

对于大多数应用工作，调用 spaCy：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

从下往上读 `dep` 列，句子的语法结构就显现出来了。

## 实际使用

每个生产级 NLP 库都将 POS 和依存解析器作为标准流程的一部分发货。

- **spaCy**（`en_core_web_sm` / `md` / `lg` / `trf`）。快速、准确，与分词 + NER + 词形还原集成。`token.tag_`（Penn）、`token.pos_`（UD）、`token.dep_`（依存关系）。
- **Stanford NLP（stanza）**。Stanford CoreNLP 的后继者。在 60+ 种语言上达到前沿水平。
- **trankit**。基于 transformer，UD 准确率高。
- **NLTK**。`pos_tag`。可用、慢、老旧。适合教学。

### 2026 年这仍然重要的地方

- **词形还原。** 第一课需要 POS 才能正确还原词形。始终如此。
- **从 LLM 输出中做结构化抽取。** 验证生成的句子是否符合语法约束（例如主谓一致、必需的修饰语）。
- **基于方面的情感分析。** 依存分析告诉你哪个形容词修饰哪个名词。
- **查询理解。** "movies directed by Wes Anderson starring Bill Murray" 通过句法分析分解为结构化约束。
- **跨语言迁移。** UD 标签和依存关系与语言无关，支持对新语言的零样本结构化分析。
- **低算力流程。** 如果无法上线 transformer，POS + 依存分析 + 地名词典能让你走得很远。

## 交付

保存为 `outputs/skill-grammar-pipeline.md`：

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## 练习

1. **简单。** 在小规模带标注语料库（例如 NLTK 的 Brown 子集）上使用最常见标签基线，测量在留出句子上的准确率。验证约 85% 的结果。
2. **中等。** 训练上述二元 HMM 并报告每个标签的精确率/召回率。HMM 最容易混淆哪些标签？
3. **困难。** 用 spaCy 的依存分析从 1000 句样本中抽取主谓宾三元组。在 50 个手动标注的三元组上评估。记录抽取失败的地方（通常是被动句、并列结构和省略主语）。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|-----------------------|
| POS 标签 | 词的类别 | 语法类别。PTB 有 36 个；UD 有 17 个。 |
| Penn Treebank | 标准标签集 | 英文专用。细粒度的动词时态和名词数。 |
| 通用依存 | 多语言标签集 | 比 PTB 粗粒度；语言中立；跨语言工作默认集。 |
| 依存分析 | 句子树 | 每个词有一个中心词，每条边有一个语法关系。 |
| Viterbi | 动态规划 | 给定发射概率和转移概率时找到最高概率的标签序列。 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — POS 和句法分析的权威教科书论述。
- [Universal Dependencies project](https://universaldependencies.org/) — 每个多语言解析器使用的跨语言标签集和树库集合。
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — `Token` 上每个属性的实用参考。
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — 将神经解析器引入主流的论文。