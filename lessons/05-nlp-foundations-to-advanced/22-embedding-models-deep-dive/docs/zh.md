# Embedding 模型 — 2026 深度剖析

> Word2Vec 给每个词一个向量。现代 embedding 模型给每个段落一个向量，跨语言，同时支持稀疏、稠密和多向量视图，大小可适配你的索引。选错了，RAG 就会检索到错误的内容。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 5 · 03（Word2Vec）、阶段 5 · 14（信息检索）
**时间：** 约 60 分钟

## 问题

你的 RAG 系统检索错误率高达 40%。罪魁祸首很少是向量数据库或提示词。是 embedding 模型。

2026 年选择 embedding 意味着在五个轴上权衡：

1. **稠密 vs 稀疏 vs 多向量。** 每个段落一个向量，还是每个 token 一个向量，还是稀疏加权词袋。
2. **语言覆盖。** 单语英语模型在纯英语任务上仍然领先。多语言模型在语料混合时胜出。
3. **上下文长度。** 512 tokens vs 8,192 vs 32,768 —— 实际有效容量往往只有标称最大值的 60-70%。
4. **维度预算。** 3,072 个浮点数全精度 = 每个向量 12 KB。1 亿个向量，存储成本 $1,300/月。Matryoshka 截断可以把这个成本削减 4 倍。
5. **开源 vs 托管。** 开源权重意味着你控制整个栈和数据。托管意味着你用控制权换取始终最新。

这节课讲清楚这些权衡，让你凭证据选择，而不是凭上季度什么最流行。

## 概念

![稠密、稀疏和多向量 embedding](../assets/embedding-modes.svg)

**稠密 embedding。** 每个段落一个向量（通常 384–3,072 维）。余弦相似度按语义接近度对段落排序。OpenAI `text-embedding-3-large`、BGE-M3 稠密模式、Voyage-3。默认选择。

**稀疏 embedding。** SPLADE 风格。Transformer 为每个词表 token 预测一个权重，然后零掉大部分。结果是一个大小为 |vocab| 的稀疏向量。捕获词汇匹配（如 BM25），但带有学习到的 term 权重。在关键词密集的查询上表现强劲。

**多向量（后期交互）。** ColBERTv2、Jina-ColBERT。每个 token 一个向量。用 MaxSim 计分：对每个查询 token，找到最相似的文档 token，汇总得分。存储和计分更贵，但在长查询和领域特定语料上胜出。

**BGE-M3：三合一。** 单个模型同时输出稠密、稀疏和多向量表示。每个可以独立查询；分数通过加权和融合。2026 年当你想要从一个检查点获得灵活性时的默认选择。

**Matryoshka 表示学习。** 训练使得向量的前 N 维本身就是一个有用的独立 embedding。将 1,536 维向量截断到 256 维，准确率仅损失约 1%，但存储节省 6 倍。被 OpenAI text-3、Cohere v4、Voyage-4、Jina v5、Gemini Embedding 2、Nomic v1.5+ 支持。

### MTEB 排行榜讲了一个不完整的故事

大规模文本 Embedding 基准 —— 首发时 56 个任务覆盖 8 类任务类型（2022），在 MTEB v2 扩展到 100+ 任务。2026 年初，Gemini Embedding 2 在检索上领先（67.71 MTEB-R）。Cohere embed-v4 在通用任务上领先（65.2 MTEB）。BGE-M3 在开源多语言上领先（63.0）。排行榜是必要条件但不是充分条件 —— 一定要在你的领域上做基准测试。

### 三层模式

| 使用场景 | 模式 |
|----------|---------|
| 快速初筛 | 稠密双编码器（BGE-M3、text-3-small） |
| 召回提升 | 稀疏（SPLADE、BGE-M3 sparse）+ RRF 融合 |
| 前 50 名精确度 | 多向量（ColBERTv2）或交叉编码器重排器 |

大多数生产堆栈三者都用。

## 构建

### 第 1 步：基线 —— 用 Sentence-BERT 做稠密 embedding

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` 使得点积等于余弦相似度。一定要设这个。

### 第 2 步：Matryoshka 截断

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

截断后重新归一化。Nomic v1.5、OpenAI text-3 和 Voyage-4 的训练使得前几层截断几乎是无损的。非 Matryoshka 模型（原始 Sentence-BERT）在截断时性能急剧下降。

### 第 3 步：BGE-M3 多功能

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

一次推理调用，三个索引。分数融合：

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

在你的领域上调优权重。

### 第 4 步：在自定义任务上做 MTEB 评估

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

在有代表性的子集上运行候选模型。不要只看排行榜名次 —— 你的领域很重要。

### 第 5 步：从零手写余弦相似度

见 `code/main.py`。平均哈希技巧 embedding（仅用标准库）。与 transformer embedding 没有竞争力，但展示了基本流程：分词 → 向量 → 归一化 → 点积。

## 陷阱

- **查询和文档用同一模型。** 有些模型（Voyage、Jina-ColBERT）使用非对称编码 —— 查询和文档走不同路径。一定要查看模型卡。
- **缺少前缀。** `bge-*` 模型需要在查询前加 `"Represent this sentence for searching relevant passages: "`。忘了这个会有 3-5 分的召回率差距。
- **Matryoshka 过度截断。** 1,536 → 256 通常安全。1,536 → 64 不安全。在你的评估集上验证。
- **上下文截断。** 大多数模型在超过最大长度时静默截断输入。长文档需要分块（见第 23 课）。
- **忽略延迟尾部。** MTEB 分数隐藏了 p99 延迟。一个 600M 的模型可能比 335M 的模型高 2 分，但每次查询成本贵 3 倍。

## 使用

2026 技术栈：

| 场景 | 选择 |
|-----------|------|
| 仅英语、快速、API | `text-embedding-3-large` 或 `voyage-3-large` |
| 开源权重、英语 | `BAAI/bge-large-en-v1.5` |
| 开源权重、多语言 | `BAAI/bge-m3` 或 `Qwen3-Embedding-8B` |
| 长上下文（32k+） | Voyage-3-large、Cohere embed-v4、Qwen3-Embedding-8B |
| 仅 CPU 部署 | Nomic Embed v2（137M 参数，MoE） |
| 存储受限 | Matryoshka 截断 + int8 量化 |
| 关键词密集查询 | 加 SPLADE sparse，与稠密做 RRF 融合 |

2026 模式：从 BGE-M3 或 text-3-large 开始，用 MTEB 在你的领域上评估，如果领域特定模型领先超过 3 分就换掉它。

## 交付

保存为 `outputs/skill-embedding-picker.md`：

```markdown
---
name: embedding-picker
description: 为给定语料和部署场景选择 embedding 模型、维度和检索模式。
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

给定语料（规模、语言、领域、平均长度）、部署目标（云 / 边缘 / 本地）、延迟预算和存储预算，输出：

1. 模型。命名检查点或 API。一句话理由。
2. 维度。全量 / Matryoshka 截断 / int8 量化。与存储预算相关的理由。
3. 模式。稠密 / 稀疏 / 多向量 / 混合。理由。
4. 如果模型卡要求的话，查询前缀 / 模板。
5. 评估计划。与领域相关的 MTEB 任务 + 用 nDCG@10 做保留领域评估。

拒绝将 Matryoshka 截断到 <64 维且无领域验证的建议。拒绝将 ColBERTv2 用于小于 10k 段落的语料（开销不划算）。标记将长文档语料（>8k tokens）路由到 512-token 窗口的模型。

## 练习

1. **简单。** 用 `bge-small-en-v1.5` 在全维（384）和 Matryoshka 128 维上对 100 个句子编码。在 10 个查询上测量 MRR 下降。
2. **中等。** 在你的领域 500 个段落上比较 BGE-M3 稠密、稀疏和 colbert。哪个在 recall@10 上胜出？RRF 融合是否胜过最佳单模式？
3. **困难。** 在你的两个顶级领域任务上对三个候选模型跑 MTEB。报告 MTEB 分数、100 查询批次的 p99 延迟和每百万查询成本。选帕累托最优的那个。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 稠密 embedding | 向量 | 每个文本一个固定大小的向量。用余弦相似度排序。 |
| 稀疏 embedding | 学习到的 BM25 | 每个词表 token 一个权重；大部分为零；端到端训练。 |
| 多向量 | ColBERT 风格 | 每个 token 一个向量；MaxSim 计分；索引更大，召回更好。 |
| Matryoshka | 俄罗斯套娃技巧 | 前 N 维本身就是一个有效的更小 embedding。 |
| MTEB | 基准 | 大规模文本 Embedding 基准 —— 首发 56 个任务，v2 超过 100 个。 |
| BEIR | 检索基准 | 18 个零样本检索任务；常被引用以证明跨域鲁棒性。 |
| 非对称编码 | 查询 ≠ 文档路径 | 模型对查询和文档使用不同的投影。 |

## 延伸阅读

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — 双编码器论文。
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — 排行榜论文。
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — 统一三模式模型。
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — 维度阶梯训练目标。
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — 生产中的后期交互。
- [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) — 实时排名。
