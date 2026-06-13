# Capstone 02 — 代码库 RAG（跨仓库语义搜索）

> 到 2026 年，每家正经的工程公司都运行着一个理解语义而非仅匹配字符串的内部代码搜索。Sourcegraph Amp、Cursor 的 codebase answers、Augment 的企业图、Aider 的 repomap、Pinterest 内部的 MCP——都是同样的形态。摄入多个仓库，用 tree-sitter 解析，在函数和类级别分块，混合搜索，重排序，带引用回答。本 capstone 让你构建一个能处理横跨 10 个仓库共 200 万行代码并在每次 git push 时增量重建索引的系统。

**类型：** Capstone
**语言：** Python（摄入）、TypeScript（API + UI）
**前置条件：** 阶段 5（NLP 基础）、阶段 7（Transformer）、阶段 11（LLM 工程）、阶段 13（工具）、阶段 17（基础设施）
**涉及阶段：** P5 · P7 · P11 · P13 · P17
**时间：** 30 小时

## 问题

到 2026 年，每家前沿编码智能体都自带代码库检索层，因为仅靠上下文窗口解决不了跨仓库问题。Claude 的 100 万令牌上下文有帮助；它无法消除对排序检索的需求。在生代码、monorepo 重复和长尾稀有符号上，简单的余弦搜索会污染结果。生产级答案是混合（dense + BM25）搜索，配合 AST 感知的分块和重排序器，由符号引用图支撑。

你通过索引真实的集群来学习——不是一个教程仓库——并测量 MRR@10、引用忠实度和增量新鲜度。失败模式是基础设施层面的：10 万文件的 monorepo、一次 touch 了半数文件的 push、需要跨四个仓库才能正确回答的查询。

## 概念

AST 感知的摄入管道用 tree-sitter 解析每个文件，提取函数和类节点，在节点边界而非固定令牌窗口处分块。每个块获得三种表示：一个 dense embedding（Voyage-code-3 或 nomic-embed-code）、稀疏 BM25 词项，以及一段简短的自然语言摘要。摘要增加了第三种可检索模态——用户问"X 是如何被授权的"，摘要提到"authz"，即使代码只有 `check_permission`。

检索是混合的。查询同时触发 dense 和 BM25 搜索，合并 top-k，将并集交给交叉编码器重排序器（Cohere rerank-3 或 bge-reranker-v2-gemma-2b）。重排序后的列表送入长上下文综合器（Claude Sonnet 4.7 带 prompt caching，或 Llama 3.3 70B 自托管），附带指令要求按文件和行范围引用每个声明。没有引用的答案会被后置过滤器拒绝。

增量新鲜度是基础设施问题。Git push 触发 diff：哪些文件变了，哪些符号变了。只重新 embedding 受影响的块。受影响的跨文件符号边（imports、方法调用）重新计算。索引保持一致，而无需在每次提交时重新处理 200 万行。

## 架构

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## 技术栈

- 解析： tree-sitter，17 种语言语法（Python、TS、Rust、Go、Java、C++ 等）
- Dense embedding： Voyage-code-3（托管）或 nomic-embed-code-v1.5（自托管），bge-code-v1 备选
- 稀疏索引： Tantivy（Rust）+ BM25F，符号名权重 4，正文权重 1
- 向量数据库： Qdrant 1.12 混合搜索，或 5000 万向量以下团队的 pgvector + pgvectorscale
- 块摘要模型： Claude Haiku 4.5 或 Gemini 2.5 Flash，prompt cached
- 重排序器： Cohere rerank-3 或 bge-reranker-v2-gemma-2b 自托管
- 编排： LlamaIndex Workflows（摄入）、LangGraph（查询智能体）
- 综合器： Claude Sonnet 4.7（100 万上下文）带 prompt caching
- 符号图： Neo4j（托管）或 kuzu（嵌入式），存储 import 和 call 边
- 可观测性： Langfuse，每个检索 + 综合步骤的 span

## 构建步骤

1. **摄入 walker。** 在每次 push hook 上迭代 git 历史。收集变更的文件。对每个文件，用 tree-sitter 解析，提取函数和类节点及其完整源跨度。发出块记录 `{repo, path, start_line, end_line, symbol, body}`。

2. **块摘要器。** 将块批量送入 Haiku 4.5 调用，系统前言带 prompt caching。Prompt："用一句话概括这个函数，命名其公开契约和副作用。"将摘要与块一起存储。

3. **Embedding 池。** 两个并行队列：dense（Voyage-code-3 批量 128）和摘要（同一模型，但作用于摘要字符串）。向量写入 Qdrant，有效载荷 `{repo, path, start_line, end_line, symbol, kind}`。

4. **BM25 索引。** 字段加权 Tantivy 索引：符号名权重 4，符号正文权重 1，摘要权重 2。使"查找名为 X 的函数"查询和"查找做 X 的函数"查询并存。

5. **符号图。** 对每个块，记录边：imports（本文件使用来自 repo Z 的符号 Y）、calls（本函数调用类 C 上的方法 M）、继承。存储在 kuzu 中。在查询时用于跨仓库边界扩展检索。

6. **查询智能体。** LangGraph，三个节点。`retrieve` 并行触发 dense + BM25，按 (repo, path, symbol) 去重。`rerank` 在 top-50 上运行交叉编码器，保留 top-10。`synth` 调用 Claude Sonnet 4.7，上下文是重排序后的块，缓存系统 prompt，要求 file:line 引用。

7. **引用强制。** 解析模型输出；任何没有 `(repo/path:start-end)` 锚点的声明都会被标记为重新询问或丢弃。只向用户返回带引用的答案。

8. **增量重建索引。** 每次 webhook，计算符号级 diff。只重新 embedding 文本改变的块。重新计算 import 改变的块的符号边。测量：50 文件的 push 在 200 万行代码的集群上 60 秒内重新索引完成。

9. **评估。** 用金标准 file:line 答案标注 100 个跨仓库问题。测量 MRR@10、nDCG@10、引用忠实度（带可验证锚点的声明比例）和 p50/p99 延迟。

## 使用方法

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## 交付

可交付技能 `outputs/skill-codebase-rag.md`。给定一个仓库语料库，它架起摄入管道、混合索引和查询智能体，并为任何跨仓库问题返回带引用的答案。评分标准：

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 检索质量 | 在 100 题 holdout 上 MRR@10 和 nDCG@10 |
| 20 | 引用忠实度 | 答案声明中带可验证 file:line 锚点的比例 |
| 20 | 延迟和规模 | 在索引语料库规模上 10k QPS 时 p95 查询延迟 |
| 20 | 增量索引正确性 | 从 git push 到可搜索，50 文件提交的时间 |
| 15 | 用户体验和答案格式 | 引用可点击性、片段预览、后续交互便利性 |
| **100** | | |

## 练习

1. 将 Voyage-code-3 换成 nomic-embed-code 自托管。测量 MRR@10 的变化。报告启用重排序后差距是否缩小。

2. 向语料库注入 20% 的生成代码（LLM 生成的样板）并重新评估。观察检索中毒。为有效载荷添加一个"generated"标志并降低这些命中的权重。

3. 在你的语料库规模上评测 Qdrant 混合搜索 vs pgvector + pgvectorscale。报告批量大小 1 时的 p99。

4. 添加基于采样的漂移检查：每周重新运行 100 题评估。MRR@10 下降 > 5% 时告警。

5. 扩展到跨语言符号解析：一个 Python 函数通过 gRPC 调用 Go 服务。用符号图链接它们。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| AST-aware chunking | "函数级切分" | 在 tree-sitter 节点边界而非固定令牌窗口处切割代码 |
| Hybrid search | "Dense + sparse" | 并行运行 BM25 和向量搜索，合并 top-k，重排序 |
| Cross-encoder rerank | "第二阶段排序" | 将每个（查询，候选） pair 一起评分的模型，比余弦更准确 |
| Prompt caching | "缓存的系统提示" | 2026 年 Claude / OpenAI 特性，对重复前缀令牌最多折扣 90% |
| Symbol graph | "代码图" | 跨文件和仓库的 imports、calls、继承边 |
| Citation faithfulness | "有据可查的答案率" | 用户可通过点击锚点并阅读引用跨度来验证的声明比例 |
| Incremental re-index | "push 到可搜索的时间" | 从 git push 到变更符号可查询的墙上时钟时间 |

## 延伸阅读

- [Sourcegraph Amp](https://ampcode.com) — 生产级跨仓库代码智能
- [Sourcegraph Cody RAG 架构](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — 本 capstone 的参考深度文章
- [Aider repo-map](https://aider.chat/docs/repomap.html) — tree-sitter 排序的仓库视图
- [Augment Code 企业图](https://www.augmentcode.com) — 商业符号图 RAG
- [Qdrant 混合搜索文档](https://qdrant.tech/documentation/concepts/hybrid-queries/) — 参考实现
- [Voyage AI 代码 embedding](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3 详情
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — 交叉编码器参考
- [Pinterest MCP 内部搜索](https://medium.com/pinterest-engineering) — 内部平台参考
