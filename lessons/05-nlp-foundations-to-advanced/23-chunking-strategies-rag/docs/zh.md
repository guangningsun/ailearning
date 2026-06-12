# RAG 分块策略

> 分块配置对检索质量的影响不亚于 embedding 模型的选择（Vectara NAACL 2025）。分块做错了，再多的重排序也救不了你。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 5 · 14（信息检索）、阶段 5 · 22（Embedding 模型）
**时间：** 约 60 分钟

## 问题

你把一份 50 页的合同放进 RAG 系统。用户问："终止条款是什么？"检索器返回了封面页。为什么？因为模型在 512 token 的块上训练，而终止条款在第 20 页，被分页符切开了，没有任何本地关键词将它与查询联系起来。

解决办法不是"买一个更好的 embedding 模型"。解决办法是分块。多大？重叠多少？在哪切？要不要周围上下文？

2026 年 2 月的基准测试显示了令人惊讶的结果：

- Vectara 2026 研究：递归 512 token 分块准确率 69%，优于语义分块的 54%。
- SPLADE + Mistral-8B 在 Natural Questions 上：重叠没有提供任何可测量的收益。
- 上下文悬崖：响应质量在约 2,500 token 上下文时急剧下降。

"显而易见"的答案（语义分块、20% 重叠、1000 token）往往是错的。本课帮助你建立六种策略的直觉，并告诉你何时该用哪一种。

## 概念

![一种段落上的六种分块策略可视化](../assets/chunking.svg)

**固定分块。** 每 N 个字符或 token 切一次。最简单的基线。会切在句子中间。压缩率高，但连贯性差。

**递归分块。** LangChain 的 `RecursiveCharacterTextSplitter`。先尝试用 `\n\n` 切，再尝试 `\n`，再尝试 `.`，再尝试空格。优雅地回退。2026 年的默认选择。

**语义分块。** 对每个句子做 embedding。计算相邻句子之间的余弦相似度。在相似度降到阈值以下的地方切开。保持主题连贯。速度较慢；有时会产生微小的 40 token 碎片，损害检索效果。

**句子分块。** 在句子边界处切开。每块一个句子或 N 个句子的窗口。在 ~5k token 以内匹配语义分块，成本只是后者的一小部分。

**父文档分块。** 为检索存储小的子块，同时存储更大的父块作为上下文。按子块检索，返回父块。优雅降级：差的子块仍然返回合理的父块。

**后分块（Late chunking，2024）。** 先在 token 级别对整个文档做 embedding，然后将 token embedding 池化为块 embedding。保留跨块上下文。适用于长上下文 embedder（BGE-M3、Jina v3）。计算量更高。

**上下文检索（Anthropic，2024）。** 在每个块前加上 LLM 生成的该块在文档中位置的摘要（"此块是终止条款第 3.2 节……"）。在 Anthropic 自家基准测试中检索提升 35-50%。索引成本高昂。

### 打败所有默认配置的规则

让块大小与查询类型匹配：

| 查询类型 | 块大小 |
|------------|-----------|
| 事实型（"CEO 的名字是什么？"） | 256-512 token |
| 分析型 / 多跳 | 512-1024 token |
| 整节理解 | 1024-2048 token |

NVIDIA 2026 基准测试。块应该足够大，能包含答案加本地上下文，又足够小，使检索器的 top-K 返回聚焦于答案而非上下文噪声。

## 动手实现

### 第 1 步：固定分块和递归分块

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### 第 2 步：语义分块

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

根据你的领域调整 `threshold`。太高 → 碎片。太低 → 一个巨大的块。

### 第 3 步：父文档分块

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

关键洞察：对父块去重。多个子块可能映射到同一个父块；全部返回会浪费上下文。

### 第 4 步：上下文检索（Anthropic 模式）

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

索引上下文增强后的块。在查询时，检索受益于额外的周围信号。

### 第 5 步：评估

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

始终做基准测试。你的语料库的最佳策略可能与任何博客文章都不一致。

## 陷阱

- **只在事实型查询上评估分块。** 多跳查询揭示非常不同的赢家。使用按查询类型分层的评估集。
- **语义分块没有最小尺寸限制。** 会产生损害检索的 40 token 碎片。始终强制执行 `min_tokens`。
- **重叠作为盲目跟风。** 2026 年研究发现重叠通常没有收益且使索引成本翻倍。测量，而不是假设。
- **没有最小/最大强制执行。** 5 token 或 5000 token 的块都会破坏检索。加个 clamp。
- **跨文档分块。** 绝不让一个块跨越两个文档。始终按文档分块，然后再合并。

## 实际使用

2026 年技术栈：

| 场景 | 策略 |
|-----------|----------|
| 首次构建，语料库未知 | 递归，512 token，无重叠 |
| 事实型问答 | 递归，256-512 token |
| 分析型 / 多跳 | 递归，512-1024 token + 父文档 |
| 重度交叉引用（合同、论文） | 后分块或上下文检索 |
| 对话式 / 对话语料 | 按轮次分块 + 说话人元数据 |
| 短文本（推文、评论） | 一个文档 = 一个块 |

从递归 512 开始。在 50 条查询的评估集上测量 recall@5。从那里开始调优。

## 交付物

保存为 `outputs/skill-chunker.md`：

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## 练习

1. **简单。** 用 fixed(512, 0)、recursive(512, 0) 和 recursive(512, 100) 对一份 20 页文档分块。比较块数量和边界质量。
2. **中等。** 在 5 份文档上构建 30 条查询的评估集。测量 recursive、semantic 和 parent-document 的 recall@5。谁赢了？与博客文章一致吗？
3. **困难。** 实现上下文检索。测量相对于基线 recursive 的 MRR 提升。报告索引成本（LLM 调用）与准确率收益。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 块（Chunk） | 文档的一片 | 被 embedding、索引和检索的子文档单元。 |
| 重叠（Overlap） | 安全边界 | 相邻块之间共享的 N 个 token；在 2026 年基准测试中往往无用。 |
| 语义分块（Semantic chunking） | 智能分块 | 在相邻句子 embedding 相似度下降处切开。 |
| 父文档（Parent-document） | 两级检索 | 检索小子块，返回大父块。 |
| 后分块（Late chunking） | embedding 后再分块 | 在 token 级别对完整文档做 embedding，池化为块向量。 |
| 上下文检索（Contextual retrieval） | Anthropic 的技巧 | LLM 生成的摘要，在索引前添加到每个块前面。 |
| 上下文悬崖（Context cliff） | 2500 token 墙 | RAG 中在约 2.5k 上下文 token 时观察到的质量下降（2026 年 1 月）。 |

## 进一步阅读

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 生产环境中的默认选择。
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) — 分块的重要性不亚于 embedding 选择。
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — 后分块论文。
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — LLM 生成的上下文前缀使检索提升 35-50%。
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — 按查询类型的块大小建议。