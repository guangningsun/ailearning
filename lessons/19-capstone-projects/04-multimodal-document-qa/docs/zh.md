# 毕业项目 04 — 多模态文档问答（视觉优先的 PDF、表格、图表）

> 2026 年的文档问答前沿已从"OCR 后转文本"转向"视觉优先的晚期交互"。ColPali、ColQwen2.5 和 ColQwen3-omni 将每一页 PDF 视为图像，用多向量晚期交互进行 embedding，让查询直接关注_patch_（图像块）。在财务 10-K 表格、科学论文和手写笔记上，这种模式远超"OCR 后转文本"。请从零构建这条流水线，在 10k 页文档上运行，并发布与 OCR-后-文本方案的并排对比结果。

**类型：** 毕业项目
**语言：** Python（流水线）、TypeScript（查看器 UI）
**前置条件：** 阶段 4（计算机视觉）、阶段 5（NLP）、阶段 7（Transformer）、阶段 11（LLM 工程）、阶段 12（多模态）、阶段 17（基础设施）
**涉及的阶段：** P4 · P5 · P7 · P11 · P12 · P17
**时间：** 约 30 小时

## 问题

企业的 PDF 文档是 OCR 流水线的噩梦：带有旋转表格的扫描版 10-K、满是公式的科学论文、只能作为图像理解的图表、手写批注。把这些当作"文本优先"处理意味着损失一半的信号。2026 年的解决方案是对原始页面图像进行晚期交互多向量检索。ColPali（Illuin Tech）首创此法；ColQwen2.5-v0.2 和 ColQwen3-omni 将精度推向新高度。在 ViDoRe v3 上，视觉优先检索的得分远高于 OCR-后-文本——在图表、表格和手写内容上差距更大。

代价是存储和延迟。一个 ColQwen embedding 每页约 2048 个_patch_向量，而非单一的 1024 维向量。原始存储量急剧膨胀。DocPruner（2026）实现了 50% 的剪枝，而精度损失可忽略不计。你将索引 10k 页文档，测量 ViDoRe v3 的 nDCG@5，在 2 秒内给出答案，并直接与 OCR-后-文本基线对比。

## 概念

晚期交互意味着每个查询 token 对每个_patch_ token 评分，每个查询 token 取最大得分后求和。你获得细粒度匹配，而无需一个单一的汇聚向量。多向量索引（Vespa、Qdrant multi-vector 或 AstraDB）存储每个_patch_的 embedding，并在检索时运行 MaxSim。

答案生成器是一个视觉-语言模型，输入查询加上 top-k 检索页面作为图像，输出带证据区域（边界框或页码引用）的答案。Qwen3-VL-30B、Gemini 2.5 Pro 和 InternVL3 是 2026 年的前沿选择。对于公式和科学符号，OCR 后备方案（Nougat、dots.ocr）作为可选文本通道接入。

评估是一个二维矩阵。一个维度：内容类型（纯文本段落、密集表格、柱状/折线图、手写笔记、公式）。另一个维度：检索方法（视觉优先晚期交互 vs OCR-后-文本 vs 混合）。每个单元格记录 nDCG@5 和答案准确率。该报告就是交付物。

## 架构

```
PDF -> 页面渲染器（PyMuPDF，180 DPI）
           |
           v
    ColQwen2.5-v0.2 embedding（每页多向量，约 2048 个 patch）
           |
           +------> DocPruner 压缩 50%
           |
           v
    多向量索引（Vespa 或 Qdrant multi-vector）
           |
查询 ----+----> 检索 top-k 页面（MaxSim）
           |
           v
    VLM 答案生成器：Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    输入：查询 + top-k 页面图像 + 可选 OCR 文本
           |
           v
    附带引用页码和证据区域的答案
           |
           v
    Streamlit / Next.js 查看器：在源页面上高亮显示框
```

## 技术栈

- 页面渲染：PyMuPDF（fitz），180 DPI，纵向标准化
- 晚期交互模型：ColQwen2.5-v0.2 或 ColQwen3-omni（vidore team，Hugging Face）
- 索引：Vespa 多向量字段，或 Qdrant multi-vector，或 AstraDB with MaxSim
- 剪枝：DocPruner 2026 策略（保留高方差 patch，压缩 50%，精度损失 < 0.5%）
- OCR 后备（公式/密集表格）：dots.ocr 或 Nougat
- VLM 答案生成器：自托管 Qwen3-VL-30B 或托管 Gemini 2.5 Pro；InternVL3 作为后备
- 评估：ViDoRe v3 基准测试，M3DocVQA 用于多页推理
- 查看器 UI：Next.js 15，带画布叠加层显示证据区域

## 构建步骤

1. **摄取。** 遍历 10k 页 PDF 语料库，涵盖 10-K 表格、科学论文和扫描文档。将每页渲染为 1536x2048 的 PNG。持久化 `{doc_id, page_num, image_path}`。

2. **Embedding。** 在每页图像上运行 ColQwen2.5-v0.2。输出形状约为 2048 个 patch embedding，维度 128。用 DocPruner 保留信息量最高的那一半。写入 Vespa 多向量字段或 Qdrant multi-vector。

3. **查询。** 对每个到来的查询，用查询塔进行 embedding（token 级 embedding）。对索引运行 MaxSim：对每个查询 token，取页面 patch embedding 上的最大点积，求和。返回 top-k 页面。

4. **合成。** 用查询和 top-5 页面图像调用 Qwen3-VL-30B。提示词："仅使用提供的页面回答。每个论点注明出处（doc_id, page），并说明区域名称（figure、table、paragraph）。"

5. **证据区域。** 后处理答案，提取被引用的区域。如果 VLM输出了边界框（Qwen3-VL 会），在查看器中将其渲染为叠加层。

6. **OCR 后备。** 对于通过启发式规则（基于图像方差）判定为公式密集的页面，运行 Nougat 或 dots.ocr，并将 OCR 文本作为图像之外的额外通道传入。

7. **评估。** 运行 ViDoRe v3（检索 nDCG@5）和 M3DocVQA（多页 QA 准确率）。同样在相同语料库上运行 OCR-后-文本流水线，使用相同的合成器。生成内容类型 × 方法矩阵。

8. **UI。** 先做 Streamlit 原型；Next.js 15 生产级查看器，带逐页证据区域叠加。

## 使用示例

```
$ doc-qa ask "2024 年 EMEA 分部的营业利润率变化是多少？"
[检索]    top-5 页面，320ms（ColQwen2.5，MaxSim，Vespa）
[合成]    qwen3-vl-30b，1.4s，引用了（form-10k-2024，第88页）+（...，第92页）
答案：
  EMEA 营业利润率从 18.2% 降至 16.8%，下降 140 个基点。
  引用：10-K-2024.pdf 第88页（表4，分部营业利润率）
         10-K-2024.pdf 第92页（MD&A，运营表现）
[查看器]   在第88页表4上打开，高亮边界框叠加
```

## 交付

`outputs/skill-doc-qa.md` 描述交付物：一个视觉优先的多模态文档 QA 系统，针对特定语料库调优，并在 ViDoRe v3 上相对于 OCR-后-文本基线进行评估。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA 准确率 | 基准测试数字 vs OCR-文本基线及已发布排行榜 |
| 20 | 证据区域 grounding | 被引用区域中实际包含答案跨度的比例 |
| 20 | 存储和延迟工程 | DocPruner 压缩比，索引 p95，答案 p95 |
| 20 | 多页推理 | 在人工标注的 100 题多页集合上的准确率 |
| 15 | 源码审查 UX | 查看器清晰度、叠加保真度、并排对比工具 |
| **100** | | |

## 练习

1. 在相同语料库上对比 ColQwen2.5-v0.2 和 ColQwen3-omni。分析两者各自正确而对方遗漏的页面。为索引添加"内容类别"标签以支持按类型路由。

2. 激进剪枝 embedding（75%、90%）。找出压缩悬崖：ViDoRe nDCG@5 跌破 OCR 基线的那一点。

3. 构建混合方案：并行运行 OCR-后-文本和 ColQwen，用 RRF 融合，用 cross-encoder 重排。混合方案是否胜过单独使用任一方案？在哪里帮助最大？

4. 将 Qwen3-VL-30B 替换为更小的 VLM（Qwen2.5-VL-7B）。测量准确率-成本曲线。

5. 添加手写笔记支持。用 ColQwen 对手写语料库进行 embedding，测量检索效果。与手写 OCR 流水线对比。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|------------------------|
| 晚期交互 (Late interaction) | "ColPali 式检索" | 查询 token 独立对页面 patch 评分；MaxSim 汇总 |
| 多向量 (Multi-vector) | "逐 patch embedding" | 每个文档有多个向量，而非一个汇聚向量 |
| MaxSim | "晚期交互评分" | 对每个查询 token，取文档向量上的最大相似度，求和 |
| DocPruner | "Patch 压缩" | 2026 年剪枝方法，保留 50% 的 patch，精度损失可忽略 |
| ViDoRe v3 | "文档检索基准测试" | 2026 年视觉文档检索测量标准 |
| 证据区域 (Evidence region) | "被引用的边界框" | 源页面上定位答案跨度的边界框 |
| OCR 后备 (OCR fallback) | "公式通道" | 在视觉通道之外，用于公式或表格密集页面的文本流水线 |

## 延伸阅读

- [ColPali（Illuin Tech）仓库](https://github.com/illuin-tech/colpali) — 晚期交互文档检索参考实现
- [ColPali 论文（arXiv:2407.01449）](https://arxiv.org/abs/2407.01449) — 基础方法论文
- [Hugging Face 上的 ColQwen 系列](https://huggingface.co/vidore) — 生产级检查点
- [M3DocRAG（Adobe）](https://arxiv.org/abs/2411.04952) — 多页多模态 RAG 基线
- [Vespa 多向量教程](https://docs.vespa.ai/en/colpali.html) — 参考服务栈
- [Qdrant 多向量支持](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — 备选索引
- [AstraDB 多向量](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — 备选托管索引
- [Nougat OCR](https://github.com/facebookresearch/nougat) — 支持公式的 OCR 后备方案