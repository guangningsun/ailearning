# 词嵌入 —— 从零实现 Word2Vec

> 一个词由它所处的上下文定义。用一个浅层网络训练这个思想，几何性质便从中涌现。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 5 · 02（BoW + TF-IDF）、阶段 3 · 03（从零实现反向传播）
**时间：** 约 75 分钟

## 问题

TF-IDF 知道 `dog` 和 `puppy` 是不同的词。它不知道两者几乎表达同一个意思。一个在 `dog` 上训练的分类器无法泛化到关于 `puppy` 的评论。你可以通过列出同义词来掩盖这一点，但这在罕见词、领域术语以及所有你未预见到的语言上都会失效。

你需要一种表示方式，让 `dog` 和 `puppy` 在空间中距离接近。让 `king - man + woman` 落在 `queen` 附近。让一个在 `dog` 上训练的模型免费将部分信号迁移到 `puppy`。

Word2Vec 给了我们这个空间。两层神经网络，万亿 token 的训练，2013 年发表。架构简单到近乎尴尬。结果重塑了 NLP 十年。

## 概念

**分布假说**（Firth, 1957）："观其伴，知其义。"如果两个词出现在相似的上下文中，它们很可能意思相近。

Word2Vec 有两种变体，都利用了这个思想。

- **Skip-gram。** 给定中心词，预测周围词。窗口大小为 2 时，`cat -> (the, sat, on)`。
- **CBOW（连续词袋）。** 给定周围词，预测中心词。`(the, sat, on) -> cat`。

Skip-gram 训练更慢，但对罕见词处理更好。它成为了默认选择。

网络有一个不含非线性的隐藏层。输入是对词汇表的全 1-of-V 编码向量。输出是对词汇表的 softmax。在训练完成后，丢弃输出层。隐藏层的权重就是 embedding。

```
one-hot(中心词) ── W ──▶ hidden (d维) ── W' ──▶ softmax(词汇表)
                          ^
                          这就是 embedding
```

关键技巧：对 10 万词做 softmax 代价极高。Word2Vec 用**负采样**将其转化为二分类任务。预测"这个上下文词是否出现在这个中心词附近，是或否"。每个训练样本采样几个负样本（非共现词），而不是对整个词汇表计算 softmax。

## 实现

### 第 1 步：从语料库生成训练对

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "the", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

窗口中的每个（中心词，上下文）对都是一个正训练样本。

### 第 2 步：embedding 表

两个矩阵。`W` 是中心词的 embedding 表（这是你保留的）。`W'` 是上下文词的表（通常丢弃，有时与 `W` 平均）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

小随机初始化。词汇量 10k、维度 100 是现实的；教学中，50 词汇 × 16 维度就足够看到几何性质了。

### 第 3 步：负采样目标

对每个正样本对 `(center, context)`，从词汇表中采样 `k` 个随机词作为负样本。训练模型使正样本的点积 `W[center] · W'[context]` 变高，负样本的点积变低。

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

神奇公式：对正样本对做 logistic 损失（希望 sigmoid 接近 1）+ 对负样本对做 logistic 损失（希望 sigmoid 接近 0）。梯度同时流向两个表。完整推导见原始论文；如果想记住，用手和纸走一遍。

### 第 4 步：在玩具语料上训练

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

在足够大的语料上训练足够多的轮次后，拥有相似上下文的词会有相似的中心 embedding。在玩具语料上，你能隐约看到这个效果。在数十亿 token 上，效果显著。

### 第 5 步：类比技巧

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

在预训练的 300 维 Google News 向量上：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。这并不是因为模型知道什么是皇室。而是因为向量 `(king - man)` 捕获了某种像"皇室"一样的东西，加上 `woman` 就落在了皇室-女性的区域。

## 使用

从零写 Word2Vec 是教学。生产级 NLP 用 `gensim`。

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

真实工作中，你几乎从不自己训练 Word2Vec。你下载预训练向量。

- **GloVe** — 斯坦福的共现矩阵分解方法。50d、100d、200d、300d 检查点。通用覆盖好。第 04 课专门讲 GloVe。
- **fastText** — Facebook 的 Word2Vec 扩展，嵌入字符 n-gram。通过组合子词处理未登录词。第 04 课。
- **Google News 预训练 Word2Vec** — 300 维，300 万词词汇表，2013 年发布。至今每日被下载。

### 2026 年 Word2Vec 仍在胜出的场景

- 轻量级领域特定检索。在笔记本上用一小时在医学摘要上训练，获得通用模型无法捕获的专业向量。
- 类比式特征工程。`gender_vector = mean(man - woman pairs)`。从中减去其他词得到中性性别轴。公平性研究仍在使用。
- 可解释性。100 维足够小，可以用 PCA 或 t-SNE 绘图并真正看到聚类形成。
- 任何需要在无 GPU 的设备上运行推理的场景。Word2Vec 查询是单行提取。

### Word2Vec 失败的地方

多义词墙。`bank` 只有一个向量。`river bank` 和 `financial bank` 共用它。`table`（电子表格 vs. 家具）共用它。下游分类器无法从向量中区分这些含义。

上下文 embedding（ELMo、BERT、之后的每个 transformer）通过基于周围上下文为每个词的出现产生不同向量来解决这个问题。这就是从 Word2Vec 到 BERT 的跳跃：从静态到上下文。第 7 阶段涵盖 transformer 部分。

未登录词问题是另一个失败点。如果 `Zoomer-approved` 不在训练数据中，Word2Vec 从未见过它。没有回退方案。fastText 用子词组合修复了这个问题（第 04 课）。

## 交付

保存为 `outputs/skill-embedding-probe.md`：

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## 练习

1. **简单。** 在一个小语料（20 句关于猫和狗的句子）上运行训练循环。200 轮后，验证 `nearest(vocab, W, W[vocab["cat"]])` 在前 3 名中返回 `dog`。如果没有，增加轮次或词汇量。
2. **中等。** 添加高频词下采样。频率超过 `10^-5` 的词按与其频率成比例的概率从训练对中丢弃。测量对罕见词相似性的影响。
3. **困难。** 在 20 Newsgroups 语料上训练模型。计算两个偏置轴：`he - she` 和 `doctor - nurse`。将职业词投影到这两个轴上。报告哪些职业偏置差距最大。这是公平性研究人员使用的探测方法。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|-----------------------|
| 词嵌入 (Word embedding) | 词作为向量 | 从上下文中学到的密集低维（通常 100-300）表示。 |
| Skip-gram | Word2Vec 技巧 | 从中心词预测上下文词。比 CBOW 慢，但对罕见词更好。 |
| 负采样 (Negative sampling) | 训练捷径 | 用对 k 个随机词的二分类替代对整个词汇表的 softmax。 |
| 静态嵌入 (Static embedding) | 每个词一个向量 | 不管上下文如何，相同的向量。在多义词上失效。 |
| 上下文嵌入 (Contextual embedding) | 上下文敏感的向量 | 基于周围词为每个出现产生不同向量。这是 transformer 产生的。 |
| OOV | 未登录词 | 训练中未见过的词。Word2Vec 无法为这些词生成向量。 |

## 延伸阅读

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — 负采样论文。简短且可读性强。
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — 如果原始论文的数学让你觉得吃力，这是最清晰的梯度推导。
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) — 实际可用的生产训练设置。