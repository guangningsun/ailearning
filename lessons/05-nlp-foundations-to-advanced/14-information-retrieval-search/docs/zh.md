# 信息检索与搜索

> BM25 精确但脆弱。稠密检索撒网广但漏掉关键词。混合检索是 2026 年的默认方案。其他都是调优。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 5 · 02（BoW + TF-IDF），阶段 5 · 04（GloVe、FastText、子词）
**时间：** 约 75 分钟

## 问题

用户输入"如果有人为了钱撒谎会怎样"，期望找到实际覆盖该行为的法条："《印度刑法》第 420 条。"关键词搜索完全漏掉它（无共享词汇）。语义搜索也会漏掉——如果 embedding 没有在法律文本上训练过。真正的搜索必须同时处理两者。

IR 是每一个 RAG 系统、每一个搜索栏、每一个文档站点模糊查找的底层管道。2026 年在生产中有效的架构不是单一方法。它是一系列互补方法的链条，每个方法捕捉前一个方法的失败。

本课构建每个组件，并指出每个组件捕捉哪些失败。

## 概念

![混合检索：BM25 + 稠密 + RRF + 交叉编码器重排](../assets/retrieval.svg)

四个层次。按需选取。

1. **稀疏检索（BM25）。** 快速，精确匹配精确，但语义理解差。在倒排索引上运行。数百万文档上每次查询 sub-10ms。能精准找到法条引用、产品代码、错误信息、命名实体。
2. **稠密检索。** 将查询和文档编码为向量。最近邻搜索。捕捉释义和语义相似性。漏掉差一个字符的精确关键词匹配。使用 FAISS 或向量数据库每次查询 50-200ms。
3. **融合。** 合并稀疏和稠密的排序列表。倒数排序融合（RRF）是简单的默认方案，因为它忽略原始分数（不同尺度下无法比较）而只使用排名位置。当某一信号在领域中占主导地位时，加权融合是一个选项。
4. **交叉编码器重排。** 取融合后的 top-30。用交叉编码器（查询 + 文档一起，对每对打分）。保留 top-5。交叉编码器每对的推理速度比双编码器慢，但准确度高得多。通过只在 top-30 上运行来摊销成本。

三路检索（BM25 + 稠密 + 如 SPLADE 的学习稀疏检索）在 2026 年基准测试中优于两路，但需要学习稀疏索引的基础设施。对大多数团队来说，两路加交叉编码器重排是最佳选择。

## 动手构建

### 第 1 步：从零实现 BM25

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

两个值得了解的参数。`k1=1.5` 控制词频饱和度；越高对词重复的权重越大。`b=0.75` 控制长度归一化；0 忽略文档长度，1 完全归一化。这些默认值来自原始论文中 Robertson's 的建议，很少需要调优。

### 第 2 步：使用双编码器的稠密检索

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

L2 归一化 embedding 使点积等于余弦相似度。`all-MiniLM-L6-v2` 是 384 维、快速，对大多数英语检索足够强大。对于多语言工作，使用 `paraphrase-multilingual-MiniLM-L12-v2`。对于最高准确率，使用 `bge-large-en-v1.5` 或 `e5-large-v2`。

### 第 3 步：倒数排序融合

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60` 常数来自原始 RRF 论文。更高的 `k` 平滑排名差异的贡献；更低的 `k` 使排名靠前的结果占主导地位。60 是发布的默认值，很少需要调优。

### 第 4 步：混合搜索 + 重排

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

三个阶段组合。BM25 找到词汇匹配。稠密找到语义匹配。RRF 合并两个排名而无需分数校准。交叉编码器使用查询-文档对一起对 top-30 重新打分，这捕捉到了双编码器遗漏的细粒度相关性。保留 top-5。

### 第 5 步：评估

| 指标 | 含义 |
|--------|---------|
| Recall@k | 在正确文档存在的查询中，它出现在 top-k 中的频率是多少？ |
| MRR（平均倒数排名） | 第一个相关文档排名的 1/排名 的平均值。 |
| nDCG@k | 考虑相关性等级，不仅仅是二元的是/否相关。 |

对于 RAG 来说，检索器的 **Recall@k** 是最重要的数字。如果正确段落不在检索到的集合中，你的读者无法回答。

调试技巧：对于失败的查询，比较稀疏和稠密排序。如果一个找到了正确文档而另一个没有，你就遇到了词汇不匹配（修复：添加缺失的一半）或语义歧义（修复：更好的 embedding 或重排器）。

## 使用它

2026 年的技术栈：

| 规模 | 技术栈 |
|-------|-------|
| 1k-100k 文档 | 内存 BM25 + `all-MiniLM-L6-v2` embedding + RRF。无需独立数据库。 |
| 100k-10M 文档 | FAISS 或 pgvector 用于稠密 + Elasticsearch / OpenSearch 用于 BM25。并行运行。 |
| 10M+ 文档 | Qdrant / Weaviate / Vespa / Milvus 支持混合检索。在 top-30 上做交叉编码器重排。 |
| 最高质量前沿 | 三路（BM25 + 稠密 + SPLADE）+ ColBERT 后期交互重排 |

无论你选择什么，都要为评估做预算。在端到端 RAG 准确率基准测试之前，先对检索召回率做基准测试。读者无法弥补检索器遗漏的内容。

### 2026 年生产 RAG 的来之不易的经验

- **80% 的 RAG 失败追溯到摄取和分块，而不是模型。** 团队花数周时间交换 LLM 和调整提示词，而检索悄悄地在每三个查询中返回错误的上下文。先修复分块。
- **分块策略比块大小更重要。** 固定大小的分割会打断表格、代码和嵌套标题。句子感知是默认选项；对于技术文档和产品手册，基于语义或 LLM 的分块会得到回报。
- **父文档模式。** 检索小的"子"块以保证精确度。当同一父节段的多个子块出现时，换入父块以保留上下文。这通常能提升答案质量而无需重新训练。
- **k_rerank=3 通常是最优的。** 超过该值的每个额外块都会增加 token 成本和生成延迟，而不会提升答案质量。如果 k=8 对你仍然比 k=3 更好，说明重排器表现不佳。
- **HyDE / 查询扩展。** 从查询生成一个假设答案，对该答案做 embedding，然后检索。弥合短问题与长文档之间的措辞差距。免费提升精确度，无需训练。
- **上下文预算低于 8K token。** 在该限制下持续命中意味着重排器阈值太松。
- **对所有内容做版本控制。** 提示词、分块规则、embedding 模型、重排器。任何漂移都会悄悄破坏答案质量。CI 在忠实度、上下文精确度和未回答问题率上做门控，在用户看到之前阻止回归。
- **三路检索（BM25 + 稠密 + 如 SPLADE 的学习稀疏检索）在 2026 年基准测试中优于两路**，特别是对于混合专有名词和语义的查询。当基础设施支持 SPLADE 索引时用它。

根据 2026 年行业测量，正确的检索设计可将幻觉减少 70-90%。大多数 RAG 性能提升来自更好的检索，而不是模型微调。

## 交付它

保存为 `outputs/skill-retrieval-picker.md`：

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## 练习

1. **简单。** 在 500 份文档的语料库上实现上面的 `hybrid_search`。测试 20 个查询。比较 BM25-only、dense-only 和 hybrid 在 5 处的召回率。
2. **中等。** 添加 MRR 计算。对于每个有已知正确文档的测试查询，找出正确文档在 BM25、dense 和 hybrid 排名中的排名。报告每个的 MRR。
3. **困难。** 使用 MultipleNegativesRankingLoss（Sentence Transformers）对你的领域微调一个稠密编码器。从 500 个查询-文档对构建训练集。比较微调前后的召回率。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| BM25 | 关键词搜索 | Okapi BM25。通过词频、IDF 和长度对文档打分。 |
| 稠密检索 | 向量搜索 | 将查询 + 文档编码为向量，找最近邻。 |
| 双编码器 | Embedding 模型 | 独立编码查询和文档。查询时速度快。 |
| 交叉编码器 | 重排模型 | 将查询 + 文档一起编码。慢但准确。 |
| RRF | 排名融合 | 通过求和 `1/(k + 排名)` 来组合两个排名。 |
| Recall@k | 检索指标 | 相关文档出现在 top-k 中的查询比例。 |

## 延伸阅读

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — 权威的 BM25 论述。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，标准的双编码器。
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — 学习稀疏检索器，弥合与稠密的差距。
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF 论文。
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — 后期交互检索。
