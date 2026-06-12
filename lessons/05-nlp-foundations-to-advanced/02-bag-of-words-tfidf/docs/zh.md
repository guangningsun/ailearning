# 词袋模型、TF-IDF 与文本表示

> 先计数，再思考。TF-IDF 在定义良好的任务上到 2026 年仍然打败 embedding。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 5 · 01（文本处理）、阶段 2 · 02（从零实现线性回归）
**时间：** 约 75 分钟

## 问题

模型需要数字。你有字符串。

每一个 NLP pipeline 都要回答同一个问题。我们如何把一个可变长度的 token 流变成一个分类器能消费的定长向量。这个领域第一个得到的答案是最笨但能用的那个。计数词。做成向量。

这个向量承载了比任何 embedding 模型都多的生产级 NLP。垃圾邮件过滤、主题分类、日志异常检测、搜索排序（BM25 之前）、第一波情感分析、学术 NLP 基准的第一个十年。2026 年的从业者在窄分类任务上仍然先想到它。它快、可解释，而且在词存在性重要的任务上往往和一个 4 亿参数的 embedding 模型无法区分。

本节从零构建词袋模型，然后 TF-IDF。然后展示 scikit-learn 三行做同样的事。然后说出那个让你转向 embedding 的失败模式。

## 概念

**词袋模型（Bag of Words, BoW）** 扔掉顺序。对每个文档，计数每个词表词出现了多少次。向量长度是词表大小。位置 `i` 是词 `i` 的计数。

**TF-IDF** 重新加权 BoW。在每个文档都出现的词没有信息量，所以往下调。在整个语料中稀有但在单个文档中频繁的词是信号，所以往上调。

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

其中 `TF` 是词在文档中的频率，`df` 是文档频率（有多少文档包含该词），`N` 是总文档数。`log` 保持常用词的权重有界。

关键性质：两者都产生稀疏向量，轴是可解释的。你可以看一个训练好的分类器的权重，读出哪些词把文档推向哪个类别。在 768 维的 BERT embedding 上你做不到这件事。

## 动手实现

### 第 1 步：构建词表

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入：分好词的文档列表（任何词级分词器都行；本节的 `code/main.py` 使用简化的小写变体）。输出：`{word: index}` 字典。稳定的插入顺序意味着词索引 0 是第一个文档中第一次看到的词。惯例各有不同；scikit-learn 按字母序排序。

### 第 2 步：词袋模型

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

行是文档。列是词表索引。条目 `[i][j]` 是"词 `j` 在文档 `i` 中出现了多少次"。文档 1 有 `cat` 两次因为确实出现了。文档 0 有 `ran` 零次因为确实没出现。

### 第 3 步：词频率和文档频率

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

两个值得命名的平滑技巧。`(n+1)/(d+1)` 避免了 `log(x/0)`。末尾的 `+1` 确保出现在每个文档中的词仍然有 IDF 1（不是 0），匹配 scikit-learn 的默认值。其他实现用原始的 `log(N/df)`。两者都行；平滑版本更友好。

### 第 4 步：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

三个文档，五个词（`the`、`cat`、`sat`、`dog`、`ran`）。`the` 出现在所有三个文档中，所以它的 IDF 低。`dog` 只出现在一个文档中，所以它的 IDF 高。向量是稀疏的（大多数条目很小），有区分力的词跳出来了。

### 第 5 步：L2 归一化行

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

没有归一化的话，一个更长的文档得到一个更大的向量，在相似度得分上占主导。L2 归一化把每个文档放到单位超球面上。行之间的余弦相似度现在就是简单的点积。

## 实际使用

scikit-learn 发布了生产版本。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` 一行调用完成分词、词表和 BoW。`TfidfVectorizer` 加上 IDF 加权和 L2 归一化。两者都返回稀疏矩阵。对于 10 万个文档，稠密版本放不进内存；保持稀疏直到分类器要求稠密。

改变一切的旋钮：

| 参数 | 效果 |
|-----|--------|
| `ngram_range=(1, 2)` | 包含二元组。通常能提升分类。 |
| `min_df=2` | 丢弃出现在少于 2 个文档中的词。在噪声数据上剪枝词表。 |
| `max_df=0.95` | 丢弃出现在超过 95% 文档中的词。在没有硬编码停用词表的情况下近似停用词移除。 |
| `stop_words="english"` | scikit-learn 内置的停用词表。取决于任务——情感分析**不应该**丢弃否定词。 |
| `sublinear_tf=True` | 用 `1 + log(tf)` 而不是原始 `tf`。当一个词在一个文档中重复多次时有帮助。 |

### TF-IDF 仍然赢的场景（2026 年）

- 垃圾邮件检测、主题标注、日志异常标记。词存在性重要；语义细微差别不重要。
- 低数据 regime（几百个标注样本）。TF-IDF 加逻辑回归没有预训练成本。
- 任何需要延迟的地方。TF-IDF 加线性模型微秒级响应。通过 transformer embedding 一个文档需要 10-100ms。
- 必须解释预测的系统。检查分类器的系数。正向最大的词就是原因。

### TF-IDF 失败的场景

语义盲的失败。考虑这两个文档：

- "The movie was not good at all."
- "The movie was excellent."

一个是差评。一个是好评。它们的 TF-IDF 重叠恰好是 `{the, movie, was}`。一个词袋分类器只能靠死记硬背来学习 `not` 靠近 `good` 会翻转标签。它在足够数据上能学到，但永远不如理解句法的模型那样优雅。

另一个失败：推理时的未登录词。用 IMDb 评论训练的 BoW 模型不知道如何处理 `Zoomer-approved`，如果这个 token 在训练时从没出现过。子词 embedding（课程 04）处理这个。TF-IDF 不能。

### 混合：TF-IDF 加权 embedding

2026 年中等数据分类的务实默认：用 TF-IDF 权重作为词 embedding 的注意力。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

你从 embedding 获得语义容量，从 TF-IDF 获得稀有词强调。分类器在汇聚向量上训练。这在约 5 万标注样本以下的情感、主题和意图分类上单独超过两者。

## 交付物

保存为 `outputs/prompt-vectorization-picker.md`：

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## 练习

1. **简单。** 在 L2 归一化的 TF-IDF 输出上实现 `cosine_similarity(doc_vec_a, doc_vec_b)`。验证相同文档得 1.0，不相交词表的文档得 0.0。
2. **中等。** 给 `bag_of_words` 添加 `n-gram` 支持。参数 `n` 产生 `n`-gram 的计数。测试 `n=2` 在 `["the", "cat", "sat"]` 上产生 `["the cat", "cat sat"]` 的二元组计数。
3. **困难。** 用 GloVe 100d 向量构建上面的 TF-IDF 加权 embedding 混合（下载一次，缓存）。在 20 Newsgroups 数据集上对比纯 TF-IDF 和纯平均池化 embedding 的分类准确率。报告各自在哪里赢了。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| BoW | 词频向量 | 一个文档中词表的词计数。扔掉顺序。 |
| TF | 词频率 | 一个词在文档中的计数，可选地按文档长度归一化。 |
| DF | 文档频率 | 至少出现一次的文档数量。 |
| IDF | 逆文档频率 | `log(N / df)` 平滑版。下调出现在每处的词。 |
| 稀疏向量 | 大部分为零 | 词表通常是 1 万到 10 万词；大多数在任何给定文档中都不出现。 |
| 余弦相似度 | 向量夹角 | L2 归一化向量的点积。1 表示相同，0 表示正交。 |

## 延伸阅读

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 标准的 API 参考，外加每个旋钮的说明。
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — 让 TF-IDF 成为十年默认的论文。
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — 2026 年视角：老方法何时赢、为何赢。