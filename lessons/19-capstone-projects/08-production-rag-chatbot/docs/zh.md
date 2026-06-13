# 顶点项目 08 — 受监管垂直领域的生产级 RAG 聊天机器人

> Harvey、Glean、Mendable 和 LlamaCloud 在 2026 年都运行着相同的生产形态。用 docling 或 Unstructured 摄入，配合 ColPali 处理视觉内容。混合搜索。用 bge-reranker-v2-gemma 重新排序。用 Claude Sonnet 4.7 合成，配合 60-80% 命中率的提示缓存。用 Llama Guard 4 和 NeMo Guardrails 做防护。用 Langfuse 和 Phoenix 监控。用 RAGAS 在 200 道题的黄金集上评分。在一个受监管的领域（法律、临床、保险）构建一个，顶点项目就是通过黄金集、红队演练和漂移仪表板。

**类型：** 顶点项目
**语言：** Python（流水线 + API）、TypeScript（聊天 UI）
**前置条件：** 阶段 5（NLP）、阶段 7（Transformer）、阶段 11（LLM 工程）、阶段 12（多模态）、阶段 17（基础设施）、阶段 18（安全）
**涉及的阶段：** P5 · P7 · P11 · P12 · P17 · P18
**时间：** 30 小时

## 问题

受监管领域的 RAG（法律合同、临床试验方案、保险政策）是 2026 年发货量最大的生产形态，因为 ROI 显而易见， stakes 具体。Harvey（Allen & Overy）用它做法律。Mendable 发货开发者文档风格。Glean 覆盖企业搜索。模式是：高保真摄入、混合检索配合重排序、用引用强制和提示缓存合成、多层安全防护、持续监控漂移。

困难的部分不是模型。它们是：辖区感知的合规（HIPAA、GDPR、SOC2）、引用级可审计性、成本控制（当命中率高时提示缓存可获得 60-90% 的折扣）、通过 RAGAS  faithfulness 的幻觉检测，以及当源文档更新但索引未跟上时的漂移检测。这个顶点项目要求你在一个 200 道题的黄金集和一个红队套件上交付所有这些。

## 概念

流水线有两面。**摄入**：docling 或 Unstructured 解析结构化文档；ColPali 处理视觉丰富的文档；块被赋予摘要、标签和基于角色的访问标签。向量存入 pgvector + pgvectorscale（5000 万向量以下）或 Qdrant Cloud；稀疏 BM25 并行运行。**对话**：LangGraph 处理记忆和多轮；每个查询运行混合检索、用 bge-reranker-v2-gemma-2b 重排序、用 Claude Sonnet 4.7（提示缓存）合成、通过 Llama Guard 4 和 NeMo Guardrails、发出带引用锚定的响应。

评估栈有四层。**黄金集**（200 个带引用的标注问答）用于正确性。**红队**（越狱、PII 提取尝试、领域外问题）用于安全。**RAGAS** 用于每轮自动化的 faithfulness / answer relevance / context precision。**漂移仪表板**（Arize Phoenix）每周监控检索质量和幻觉评分。

提示缓存是成本杠杆。Claude 4.5+ 和 GPT-5+ 支持缓存系统提示 + 检索到的上下文。在 60-80% 命中率的条件下，每查询成本下降 3-5 倍。流水线必须为稳定前缀（系统提示 + 重排序上下文优先）设计，以实现高缓存命中率。

## 架构

```
文档 (合同、方案、政策)
      |
      v
docling / Unstructured 解析 + ColPali 处理视觉内容
      |
      v
块 + 摘要 + 角色标签 + 辖区标签
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
查询 + 角色 + 辖区
      |
      v
LangGraph 对话智能体
   +--- 检索 (混合)
   +--- 按角色 + 辖区过滤
   +--- 重排序 (bge-reranker-v2-gemma-2b 或 Voyage rerank-2)
   +--- 合成 (Claude Sonnet 4.7, 提示缓存)
   +--- 防护 (Llama Guard 4 + NeMo Guardrails + Presidio 输出 PII 清除)
   +--- 引用 + 返回
      |
      v
评估:
  RAGAS faithfulness / answer_relevance / context_precision (在线)
  Langfuse 标注队列 (采样)
  Arize Phoenix 漂移 (每周)
  红队套件 (发布前)
```

## 技术栈

- 摄入：Unstructured.io 或 docling 用于结构化文档；ColPali 用于视觉丰富的 PDF
- 向量数据库：5000 万向量以下用 pgvector + pgvectorscale；其他用 Qdrant Cloud
- 稀疏：带字段权重的 Tantivy BM25
- 编排：LlamaIndex Workflows（摄入）+ LangGraph（对话）
- 重排序器：bge-reranker-v2-gemma-2b 自托管或 Voyage rerank-2 托管
- LLM：Claude Sonnet 4.7 配合提示缓存；备用 Llama 3.3 70B 自托管
- 评估：RAGAS 0.2 在线，DeepEval 用于幻觉和越狱套件
- 可观测性：自托管 Langfuse 配合标注队列；Arize Phoenix 用于漂移
- 护栏：Llama Guard 4 输入/输出分类器，NeMo Guardrails v0.12 策略，Presidio PII 清除
- 合规：块上的基于角色访问标签；GDPR/HIPAA 辖区标签

## 构建它

1. **摄入。** 用 Unstructured 或 docling 解析你的语料库（认真构建的话 1000-10000 个文档）。对于扫描/视觉密集的页面，通过 ColPali 路由。生成带摘要、角色标签、辖区标签的块。

2. **索引。** 密集嵌入（Voyage-3 或 Nomic-embed-v2）到 pgvector + pgvectorscale。BM25 旁索引通过 Tantivy。角色和辖区过滤器作为 payload。

3. **混合检索。** 首先按角色+辖区过滤；然后并行密集 + BM25；用倒数排名融合合并；取 top-20 到重排序器；取 top-5 到合成器。

4. **带提示缓存的合成。** 系统提示 + 静态策略放在缓存头部；重排序的上下文作为缓存扩展；用户问题作为未缓存后缀。在稳态下目标 60-80% 缓存命中率。

5. **护栏。** Llama Guard 4 在输入侧；NeMo Guardrails 轨道阻止领域外问题或策略禁止的话题；Presidio 清除输出中意外的 PII；引用强制后过滤器。

6. **黄金集。** 200 对问答，由领域专家标注（答案、引文）。在精确引用匹配、答案正确性、faithfulness（RAGAS）上评分。

7. **红队。** 50 个对抗性提示：越狱（PAIR、TAP）、PII 窃取尝试、领域外、跨辖区泄露。用通过/失败和严重程度评分。

8. **漂移仪表板。** Arize Phoenix 每周跟踪检索质量（nDCG、引用 faithfulness）。下降 5% 时警报。

9. **成本报告。** Langfuse：提示缓存命中率、每查询 token 数、各阶段 $/query 明细。

## 使用它

```
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  混合 top-20 过滤到 GDPR + analyst-role
[rerank]    保留 top-5
[synth]     claude-sonnet-4.7, 缓存命中 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## 交付它

`outputs/skill-production-rag.md` 描述了交付物。一个带有合规标签的受监管领域聊天机器人部署，通过评估标准，通过实时漂移监控观察。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | RAGAS faithfulness + 答案相关性 | 在黄金集（200 问答）上的在线评分 |
| 20 | 引文正确性 | 可验证源锚点的答案比例 |
| 20 | 护栏覆盖率 | Llama Guard 4 通过率 + 越狱套件结果 |
| 20 | 成本/延迟工程 | 提示缓存命中率、p95 延迟、$/query |
| 15 | 漂移监控仪表板 | Phoenix 实时仪表板，带每周检索质量趋势 |
| **100** | | |

## 练习

1. 在不同辖区下构建第二个语料库切片（例如 HIPAA 与 GDPR 并行）。演示角色+辖区过滤在 20 道跨辖区探测题上防止交叉泄露。

2. 测量一周生产流量下的提示缓存命中率。识别哪些查询破坏了缓存前缀。重构。

3. 用 10k token 的摘要缓冲区添加多轮记忆。测量随对话增长 faithfulness 是否下降。

4. 将 Claude Sonnet 4.7 换成 Llama 3.3 70B 自托管。测量 $/query 和 faithfulness 差异。

5. 添加"不确定"模式：如果 top 重排序分数低于阈值，智能体说"我没有自信的引文"而不是回答。测量虚假置信度降低。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 提示缓存 | "缓存系统 + 上下文" | Claude/OpenAI 特性：命中的缓存前缀 token 折扣 60-90% |
| RAGAS | "RAG 评估器" | faithfulness、答案相关性、上下文精确度的自动评分 |
| 黄金集 | "标注评估" | 200+ 专家标注的问答带引文；ground truth |
| 辖区标签 | "合规标签" | 附加到块上的 GDPR/HIPAA/SOC2 范围；由检索过滤器强制执行 |
| 引用 faithfulness | "基于答案率" | 由可检索源跨度支持的声明比例 |
| 漂移 | "检索质量衰减" | 每周 nDCG 或引用评分变化；警报阈值 5% |
| 红队 | "对抗性评估" | 发布前越狱、PII 提取、领域外探测 |

## 延伸阅读

- [Harvey AI](https://www.harvey.ai) — 参考法律生产栈
- [Glean 企业搜索](https://www.glean.com) — 企业规模 RAG 参考
- [Mendable 文档](https://mendable.ai) — 开发者文档 RAG 参考
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/) — 托管摄入
- [Anthropic 提示缓存](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 成本杠杆参考
- [RAGAS 0.2 文档](https://docs.ragas.io/) — 规范 RAG 评估框架
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 参考漂移可观测性
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 安全分类器
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 策略轨道框架