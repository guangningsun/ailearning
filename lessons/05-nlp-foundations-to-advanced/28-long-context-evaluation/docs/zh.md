# 长上下文评估 — NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro 宣传 10M token 的上下文。在 1M token 时，8-针 MRCR 降至 26.3%。宣传 ≠ 可用。长上下文评估告诉你实际部署模型的真实容量。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 5 · 13（问答）、阶段 5 · 23（分块策略）
**时间：** 约 60 分钟

## 问题

你有一份 200 页的合同。模型声称有 1M-token 的上下文。你把合同粘贴进去并提问："终止条款是什么？"模型回答了 —— 但回答的是封面页的内容，因为终止条款位于 120k token 深处，超过了模型实际关注的位置。

这就是 2026 年的上下文容量差距。规格表上写的是 1M 或 10M。现实是可用率为 60-70%，而且"可用"取决于任务。

- **检索（大海捞针）：** 在前沿模型上接近完美，可达广告的最大值。
- **多跳 / 聚合：** 在大多数模型上超过 ~128k 后急剧下降。
- **分散事实的推理：** 第一个失败的任务。

长上下文评估测量这些轴。本节命名基准测试、每个基准实际测量的内容，以及如何为你的领域构建自定义针测试。

## 概念

![NIAH 基线、RULER 多任务、LongBench 整体](../assets/long-context-eval.svg)

**大海捞针（NIAH，2023）。** 在长上下文中的受控深度放置一个事实（"魔法词是 pineapple"）。让模型检索它。扫掠深度 × 长度。最早的长上下文基准。前沿模型现在已在该基准上饱和；这是一个必要但不充分的基础。

**RULER（Nvidia，2024）。** 4 大类 13 种任务类型：检索（单键 / 多键 / 多值）、多跳追踪（变量跟踪）、聚合（常见词频率）、问答。可配置上下文长度（4k 到 128k+）。揭示在 NIAH 上饱和但在多跳上失败的模型。在 2024 年发布中，17 个声称有 32k+ 上下文的模型中只有一半在 32k 时保持了质量。

**LongBench v2（2024）。** 503 个选择题，8k-2M 词上下文，六类任务：单文档问答、多文档问答、长上下文学习、长对话、代码库、长结构化数据。真实世界长上下文行为的生产基准。

**MRCR（多轮共指消解）。** 大规模多轮共指。8-针、24-针、100-针变体。暴露模型在注意力衰退前能处理多少个事实。

**NoLiMa。** "非词汇针"。针和查询之间没有字面上的重叠；检索需要一步语义推理。比 NIAH 更难。

**HELMET。** 连接多个文档，从任一文档提问。测试选择性注意力。

**BABILong。** 在无关的海量文本中嵌入 bAbI 推理链。测试大海捞针中的推理，而不仅仅是检索。

### 实际应报告的内容

- **宣传的上下文窗口。** 规格表上的数字。
- **有效检索长度。** NIAH 通过率在某个阈值（例如 90%）。
- **有效推理长度。** 多跳或聚合在该阈值通过。
- **退化曲线。** 准确率 vs 上下文长度，按任务类型绘制。

你的规格表需要两个数字：检索有效性和推理有效性。通常推理有效值是宣传窗口的 25-50%。

## 构建

### 第 1 步：为你的领域构建自定义 NIAH

参见 `code/main.py`。骨架：

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # 重复填充文本直到足够长以填满草堆主体。
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

扫掠 `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}。绘制热力图。这就是你的目标模型在 NIAH 上的表现卡片。

### 第 2 步：多针变体

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

像"三个魔法词是什么？"这样的问题需要检索所有三个。单针成功不能预测多针成功。

### 第 3 步：多跳变量追踪（RULER 风格）

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

答案需要链接三个赋值。前沿模型在 128k 通常在这里降至 50-70% 准确率。

### 第 4 步：在你的技术栈上运行 LongBench v2

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

按类别报告准确率。聚合分数会掩盖大的任务级差异。

## 陷阱

- **仅用 NIAH 评估。** 在 1M token 上通过 NIAH 什么也说明不了关于多跳的情况。始终运行 RULER 或自定义多跳测试。
- **均匀深度采样。** 许多实现只测试 depth=0.5。测试 depth=0, 0.25, 0.5, 0.75, 1.0 —— "迷失在中间"效应是真实存在的。
- **与填充词的词汇重叠。** 如果针与填充词共享关键词，检索就变得trivial。使用 NoLiMa 风格的无重叠针。
- **忽略延迟。** 1M-token 提示词需要 30-120 秒进行预填充。测量首个 token 的时间以及准确率。
- **供应商自报数据。** OpenAI、Google、Anthropic 都发布自己的分数。始终在你的用例上独立重新运行。

## 使用

2026 技术栈：

| 场景 | 基准 |
|-----------|-----------|
| 快速合理性检查 | 自定义 NIAH，3 深度 × 3 长度 |
| 生产模型选择 | RULER（13 项任务）在目标长度 |
| 真实世界 QA 质量 | LongBench v2 单文档问答子集 |
| 多跳推理 | BABILong 或自定义变量追踪 |
| 对话 / 对话系统 | MRCR 8-针在目标长度 |
| 模型升级回归 | 固定内部 NIAH + RULER 框架，每次新模型运行 |

生产经验法则：永远不要信任你没有在目标长度上同时运行 NIAH + 1 个推理任务的上下文窗口。

## 交付

保存为 `outputs/skill-long-context-eval.md`：

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## 练习

1. **简单。** 构建 NIAH，3 深度（0.25, 0.5, 0.75）× 3 长度（1k, 4k, 16k）。在任何模型上运行。将通过率绘制为 3×3 热力图。
2. **中等。** 添加 3-针变体。在每个长度上测量所有 3 个的检索。与同一长度上的单针通过率比较。
3. **困难。** 构建变量追踪任务（X1 → X2 → X3，3 跳），嵌入 64k 填充文本中。在 3 个前沿模型上测量准确率。报告每个模型的有效推理长度。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| NIAH | 大海捞针 | 在填充文本中放置一个事实，让模型检索它。 |
| RULER | 强化版 NIAH | 跨越检索 / 多跳 / 聚合 / 问答的 13 种任务类型。 |
| 有效上下文 | 真实容量 | 准确率仍然保持在阈值以上的长度。 |
| 迷失在中间 | 深度偏差 | 模型对长输入中间的内容注意力不足。 |
| 多针 | 一次多个事实 | 多个放置；测试注意力的 juggling，而非仅仅是检索。 |
| MRCR | 多轮共指 | 8、24 或 100 针共指；暴露注意力饱和。 |
| NoLiMa | 非词汇针 | 针和查询之间没有字面共享的 token；需要推理。 |

## 延伸阅读

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 原始 NIAH 仓库。
- [Hsieh 等 (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — 多任务基准。
- [Bai 等 (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — 真实世界长上下文评估。
- [Modarressi 等 (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — 更难的针。
- [Kuratov 等 (2024). BABILong](https://arxiv.org/abs/2406.10149) — 大海捞针中的推理。
- [Liu 等 (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — 深度偏差论文。