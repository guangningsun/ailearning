# 文本处理 — 分词、词干提取、词形还原

> 语言是连续的。模型是离散的。预处理是桥梁。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 2 · 14（朴素贝叶斯）
**时间：** 约 45 分钟

## 问题

模型读不了 "The cats were running." 它读的是整数。

每一个 NLP 系统开篇都要回答三个相同的问题。词从哪里开始。词的词根是什么。如何把 "run"、"running"、"ran" 在有帮助时视为同一个词，在无帮助时视为不同的词。

分词做错了，模型就学的是垃圾。如果你的分词器把 `don't` 当成一个 token，但把 `do n't` 当成两个，训练分布就分裂了。如果你的词干提取器把 `organization` 和 `organ` 压到同一个词干，主题建模就死了。如果你的词形还原器需要词性上下文但你没传，动词就会被当成名词。

本节从零构建三个预处理步骤，然后展示 NLTK 和 spaCy 如何做同样的工作，让你看清权衡。

## 概念

三个操作。每个都有其职责和失败模式。

**分词（Tokenization）** 把字符串切分成 token。"Token" 是故意模糊的，因为正确的颗粒度取决于任务。经典 NLP 用词级。Transformer 用子词。无空白字符的语言用字符级。

**词干提取（Stemming）** 用规则砍掉后缀。快、激进、笨。`running -> run`。`organization -> organ`。第二个就是失败模式。

**词形还原（Lemmatization）** 用语法知识把词还原为词典形式。更慢、准确、需要查表或形态分析器。`ran -> run`（需要知道 "ran" 是 "run" 的过去式）。`better -> good`（需要知道比较级形式）。

经验法则。当速度重要且可以容忍噪声时用词干提取（搜索索引、粗略分类）。当含义重要时用词形还原（问答、语义搜索、用户会读到的任何东西）。

## 动手实现

### 第 1 步：正则词分词器

最简单的实用分词器在非字母数字字符处切分，同时把标点保留为独立 token。不完美、不是最终版，但一行就能跑起来。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

三个模式按优先级排序。带可选内部撇号的词（`don't`、`it's`）。纯数字。任何单个非空白非字母数字字符作为独立 token（标点）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

需要注意的失败模式。`3pm` 被切成 `['3', 'pm']`，因为我们在字母串和数字串之间切换。对大多数任务够用了。URL、邮件、话题标签都会坏掉。生产环境的话，在通用模式之前加上特定模式。

### 第 2 步：Porter 词干提取器（仅第 1a 步）

完整的 Porter 算法有五组规则。第 1a 步单独就覆盖了最常见的英语后缀，并教给你模式。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

从上到下读规则。`ies -> i` 规则就是为什么 `ponies -> poni` 而不是 `pony`。真正的 Porter 有第 1b 步会修复它。规则互相竞争。早出现的规则优先。顺序比任何单条规则都重要。

### 第 3 步：基于查表的词形还原器

真正的词形还原需要形态学。一个可教学的版本用一个小词形表加一个后备方案。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一种情况是关键教学时刻。`watched` 不在表里，而我们的后备方案只处理 `ing`。真正的词形还原覆盖 `ed`、不规则动词、比较级形容词、有音变的复数（`children -> child`）。这就是为什么生产系统用 WordNet、spaCy 的形态分析器，或完整的形态学分析器。

### 第 4 步：把它们串联起来

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺失的一环是 POS 标注器。阶段 5 · 07（词性标注）会构建一个。现在，先把所有东西默认成 `NOUN` 并承认这个局限。

## 实际使用

NLTK 和 spaCy 发布了生产版本。各几行代码。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` 处理缩略词、Unicode、以及你的正则漏掉的边界情况。`PorterStemmer` 运行全部五组规则。`WordNetLemmatizer` 需要把 POS 标签从 NLTK 的 Penn Treebank 体系翻译成 WordNet 的缩写集。上面的翻译接线是大多数教程跳过的部分。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy 把整个 pipeline 藏在 `nlp(text)` 后面。分词、POS 标注和词形还原都一起跑了。比 NLTK 在规模上更快。开箱即用更准确。代价是你不能轻易替换单个组件。

### 何时选哪个

| 场景 | 选哪个 |
|-----------|------|
| 教学、研究、需要替换组件 | NLTK |
| 生产、多语言、速度重要 | spaCy |
| Transformer pipeline（你反正会用模型的 tokenizer 分词） | 用 `tokenizers` / `transformers`，跳过经典预处理 |

### 两个没人警告你的失败模式

大多数教程教完算法就停了。有两件事会咬到真正的预处理 pipeline，而且几乎从没人覆盖。

**可复现性漂移。** NLTK 和 spaCy 在版本之间会改变分词和词形还原行为。在 spaCy 2.x 里产生 `['do', "n't"]` 的可能在 3.x 里产生 `["don't"]`。你的模型训练时用的是一种分布。推理时跑的是另一种。准确率悄悄下降，没人知道为什么。在 `requirements.txt` 里固定库版本。写一个预处理回归测试，冻结 20 个样本句子的预期分词结果。每次升级都跑它。

**训练 / 推理不匹配。** 用激进预处理训练（转小写、去停用词、词干提取），在原始用户输入上部署，看着性能崩溃。这是生产级 NLP 最常见的单一失败。如果你在训练时做了预处理，推理时必须跑完全相同的函数。把预处理作为函数放在模型包里，而不是作为 notebook 单元格让服务团队重写。

## 交付物

一个可重用的提示词，帮助工程师在不需要读三本教科书的情况下选择预处理策略。

保存为 `outputs/prompt-preprocessing-advisor.md`：

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## 练习

1. **简单。** 扩展 `tokenize`，把 URL 保留为单个 token。测试：`tokenize("Visit https://example.com today.")` 应该产生一个 URL token。
2. **中等。** 实现 Porter 第 1b 步。如果一个词包含元音且以 `ed` 或 `ing` 结尾，就去掉它。处理双辅音规则（`hopping -> hop`，而不是 `hopp`）。
3. **困难。** 构建一个用 WordNet 作为查表但当 WordNet 没有条目时回退到 Porter 词干提取器的词形还原器。在一个标注语料库上对比纯 WordNet 和纯 Porter 测量准确率。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| Token | 一个词 | 模型消费的任意单位。可以是词、子词、字符或字节。 |
| Stem | 词根 | 基于规则的后缀剥离结果。不一定是真实单词。 |
| Lemma | 词典形式 | 你会查的那个形式。需要语法上下文才能正确计算。 |
| POS tag | 词性 | NOUN、VERB、ADJ 等类别。准确词形还原需要它。 |
| Morphology | 词形规则 | 词基于时态、数、格等如何变化形式。词形还原依赖它。 |

## 延伸阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 原始论文，五页，仍是最清晰的解释。
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — 真正的 pipeline 是如何接线的。
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — 你还没想到的分词边界情况。