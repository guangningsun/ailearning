# 问答系统

> 三个架构塑造了现代问答。抽取式找到答案片段。检索增强式将答案锚定在文档上。生成式产出答案。每一种现代 AI 助手都是这三者的组合。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 5 · 11（机器翻译），阶段 5 · 10（注意力机制）
**时间：** 约 75 分钟

## 问题

用户输入"第一代 iPhone 什么时候发布？"期望得到"2007 年 6 月 29 日。"而不是"苹果公司的历史悠久而多样。"也不是孤零零的"2007"没有任何上下文。一个直接、有据可查、正确的答案。

过去十年间，三种架构主导了问答领域。

- **抽取式问答。** 给定一个问题和一个已知包含答案的段落，找出答案片段在段落中的起始和结束索引。SQuAD 是标准基准。
- **开放域问答。** 段落未给定。需要先检索相关段落，再抽取或生成答案。这是今天每一个 RAG 流水线的基石。
- **生成式 / 闭卷问答。** 一个大型语言模型依靠其参数化记忆来回答。无检索步骤。推理速度最快，但对事实的可靠性最低。

2026 年的趋势是混合架构：检索最好的几个段落，然后提示生成模型在这些段落的基础上回答。这就是 RAG，第 14 课深入讲解检索部分。本课构建问答部分。

## 概念

![问答架构：抽取式、检索增强式、生成式](../assets/qa.svg)

**抽取式。** 用 transformer（BERT 系列）将问题与段落一起编码。训练两个头部，预测答案的起始和结束 token 索引。损失函数是有效位置上的交叉熵。输出是段落中的一个片段。从结构上不会产生幻觉（由构造决定），也不会处理段落无法回答的问题（由构造决定）。

**检索增强式（RAG）。** 两个阶段。第一阶段，检索器从语料库中找到 top-`k` 个段落。第二阶段，读者（抽取式或生成式）利用这些段落产出答案。检索器-读者分离使得两者可以独立训练和评估。现代 RAG 通常在两者之间加入一个重排器。

**生成式。** 一个仅解码器的 LLM（GPT、Claude、Llama）依靠学习到的权重来回答。无检索步骤。在常见知识上表现出色，在罕见或最新事实上一塌糊涂。幻觉率与预训练数据中事实出现的频率成反比。

## 动手构建

### 第 1 步：使用预训练模型进行抽取式问答

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` 在 SQuAD 2.0 上训练，其中包含了不可回答的问题。默认情况下，`question-answering` 流水线即使在模型的空答案分数获胜时也会返回得分最高的片段——它**不会**自动返回空答案。要获得明确的"无答案"行为，在流水线调用时传入 `handle_impossible_answer=True`：只有当空答案分数超过所有片段分数时，流水线才会返回空答案。无论哪种情况，都要检查 `score` 字段。

### 第 2 步：检索增强流水线（草图）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

两阶段流水线。稠密检索器（Sentence-BERT）通过语义相似性找到相关段落。抽取式读者（RoBERTa-SQuAD）从组合的 top 段落中抽取答案片段。适用于小规模语料库。对于百万级文档的语料库，使用 FAISS 或向量数据库。

### 第 3 步：RAG 生成式

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

提示词模式很重要。明确告诉模型在上下文中寻找答案，并在上下文不足以回答时返回"我不知道"，与简单提示相比可将幻觉率降低 40-60%。更复杂的模式可以添加引用、可信度分数和结构化抽取。

### 第 4 步：反映真实世界的评估

SQuAD 使用**精确匹配（EM）**和**token 级 F1**。EM 在规范化后（转小写、去标点、去冠词）是严格匹配——要么预测完全一致，要么得 0 分。F1 在预测和参考之间的 token 重叠上计算，给出部分分数。两者都对释义支持不足："June 29, 2007" vs "June 29th, 2007" 通常 EM 得 0（序数词破坏了规范化），但从重叠的 token 中仍能获得可观的 F1。

对于生产级问答：

- **答案准确率**（LLM 评判或人工评判，因为指标无法捕捉语义等价性）。
- **引用准确率。** 引用的段落是否实际支持答案？用生成引用与检索段落之间的字符串匹配可以轻松自动检查。
- **拒绝校准。** 当答案不在检索到的段落中时，系统是否正确地说"我不知道"？测量虚假置信率。
- **检索召回率。** 在评估读者之前，测量检索器是否将正确段落放入 top-`k`。读者无法弥补缺失的段落。

### RAGAS：2026 年生产评估框架

`RAGAS` 专为 RAG 系统构建，是 2026 年的出货默认选择。它从四个维度评分，无需黄金参考：

- **忠实度。** 答案中的每个声明是否来自检索到的上下文？通过基于 NLI 的蕴含来测量。你主要的幻觉指标。
- **答案相关性。** 答案是否回答了问题？通过从答案生成假设性问题并与真实问题比较来测量。
- **上下文精确度。** 检索到的块中，实际相关的比例是多少？低精确度 = 提示中的噪声。
- **上下文召回率。** 检索到的集合是否包含所有需要的信息？低召回率 = 读者无法成功。

无参考评分让你可以在实时生产流量上评估，无需精心的黄金答案。在开放式问题之上叠加 LLM-as-judge，因为精确匹配指标在这些情况下毫无用处。

`pip install ragas`。插入你的检索器 + 读者。每个查询得到四个标量。警惕回归。

## 使用它

2026 年的技术栈。

| 使用场景 | 推荐方案 |
|---------|-------------|
| 给定段落，找出答案片段 | `deepset/roberta-base-squad2` |
| 在固定语料库上，闭卷不可接受 | RAG：稠密检索器 + LLM 读者 |
| 文档库的实时搜索 | RAG + 混合（BM25 + 稠密）检索器 + 重排器（第 14 课） |
| 对话式问答（后续问题） | 带会话历史的 LLM + 每轮 RAG |
| 高事实性、受监管领域 | 在权威语料库上做抽取式；绝不单独使用生成式 |

抽取式问答在 2026 年已经过时，因为带 LLM 的 RAG 能处理更多情况。它仍在需要逐字引用的情况下使用：法律研究、监管合规、审计工具。

## 交付它

保存为 `outputs/skill-qa-architect.md`：

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## 练习

1. **简单。** 在 10 个维基百科段落上运行上面的 SQuAD 抽取式流水线。手工制作 10 个问题。测量答案正确的频率。如果段落和问题都很干净，你应该能看到 7-9 个正确。
2. **中等。** 添加一个拒绝分类器。当最高检索分数低于某个阈值（比如 0.3 余弦相似度）时，返回"我不知道"而不是调用读者。在留出集上调整阈值。
3. **困难。** 在你选择的 10,000 份文档语料库上构建 RAG 流水线。实现混合检索（BM25 + 稠密）+ RRF 融合（见第 14 课）。测量有无混合步骤时的答案准确率。记录哪些问题类型受益最多。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 抽取式问答 | 找到答案片段 | 在给定段落中预测答案的起始和结束索引。 |
| 开放域问答 | 在语料库上做问答 | 无给定段落；必须先检索再回答。 |
| RAG | 先检索再生成 | 检索增强生成。检索器 + 读者流水线。 |
| SQuAD | 标准基准 | 斯坦福问答数据集。EM + F1 指标。 |
| 幻觉 | 编造的答案 | 读者输出不被检索到的上下文所支持。 |
| 拒绝校准 | 知道何时闭嘴 | 当无法回答时，系统正确地说"我不知道"。 |

## 延伸阅读

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — 基准论文。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，QA 的标准稠密检索器。
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — 命名 RAG 的论文。
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — 全面的 RAG 综述。
