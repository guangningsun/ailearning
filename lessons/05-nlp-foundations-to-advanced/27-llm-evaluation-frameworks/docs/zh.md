# LLM 评估 — RAGAS、DeepEval、G-Eval

> 精确匹配和 F1 忽略了语义等价性。人工审查无法规模化。LLM 即评判者是生产环境的答案 —— 只要校准足够可靠，这个数字才可信。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 5 · 13（问答）、阶段 5 · 14（信息检索）
**时间：** 约 75 分钟

## 问题

你的 RAG 系统回答："2007 年 6 月 29 日。"
标准答案是："June 29, 2007."
精确匹配得分 0。F1 得分约 75%。人类会给出 100%。

现在乘以 10,000 个测试用例。再乘以每一次对检索器、分块、提示词或模型的改动。你需要一个能理解语义的评估器，能以低成本规模化运行，不会对回归问题撒谎，并且能暴露正确的失败模式。

2026 年有三个框架来解决这个问题。

- **RAGAS。** 检索增强生成评估（Retrieval-Augmented Generation ASsessment）。四个 RAG 指标（忠实度、答案相关性、上下文精确率、上下文召回率），使用 NLI + LLM 评判器后端。研究支撑，轻量实现。
- **DeepEval。** LLMs 的 pytest。G-Eval、任务完成度、幻觉、偏见指标。CI/CD 原生。
- **G-Eval。** 一种方法（也是 DeepEval 的一个指标）：带思维链的 LLM 即评判者，自定义标准，0-1 评分。

三者都依赖 LLM 即评判者。本节建立对该方法及其信任层的直觉。

## 概念

![四个评估维度、LLM 即评判者架构](../assets/llm-evaluation.svg)

**LLM 即评判者。** 用一个 LLM 替代静态指标，根据评分规则对输出进行评分。给定 `(query, context, answer)`，向评判 LLM 提问："对忠实度打分 0-1。"返回分数。

为什么有效：LLMs 以极低成本近似人类判断。GPT-4o-mini 每个评分案例约 0.003 美元，使得 1000 样本的回归评估运行成本低于 5 美元。

为什么静默失败：

1. **评判者偏差。** 评判者偏好更长的答案、来自同一模型家族的答案、与提示词风格匹配的答案。
2. **JSON 解析失败。** 错误的 JSON → NaN 分数 → 被静默排除在聚合之外。RAGAS 用户深知其苦。用 try/except + 显式失败模式做守卫。
3. **跨模型版本漂移。** 升级评判器会改变所有指标。冻结评判模型 + 版本。

**RAG 四项指标。**

| 指标 | 问题 | 后端 |
|--------|----------|---------|
| 忠实度 | 答案中的每个陈述是否来自检索到的上下文？ | 基于 NLI 的蕴含判断 |
| 答案相关性 | 答案是否回答了问题？ | 从答案生成假设性问题；与真实问题比较 |
| 上下文精确率 | 检索到的块中，有多少比例是相关的？ | LLM 评判器 |
| 上下文召回率 | 检索是否返回了所需的一切？ | 基于标准答案的 LLM 评判器 |

**G-Eval。** 定义一个自定义标准："答案是否引用了正确的来源？"框架自动扩展为思维链评估步骤，然后打分 0-1。适用于 RAGAS 未覆盖的领域特定质量维度。

**校准。** 在对原始评判分数建立信任之前，必须与人工标签进行相关性验证。运行 100 个手工标注的样本。绘制评判者 vs 人工的散点图。计算 Spearman rho。如果 rho < 0.7，你的评判者规则需要改进。

## 构建

### 第 1 步：基于 NLI 的忠实度（RAGAS 风格）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` 是任意可调用对象：提示词 str -> 生成内容 str。
# 示例：llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

将答案分解为原子陈述。用 NLI 检查每个陈述相对于检索到的上下文。忠实度 = 支持的比例。

### 第 2 步：答案相关性

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: 任意实现 .encode(texts, normalize_embeddings=True) -> ndarray 的模型
# 例如：encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

如果答案暗示的问题与所问的问题不同，相关性就会下降。

### 第 3 步：G-Eval 自定义指标

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

评估步骤就是评分规则。显式步骤比隐式"打分 0-1"的提示词更稳定。

### 第 4 步：CI 门控

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

作为 pytest 文件交付。每次 PR 都运行。回归则阻止合并。

### 第 5 步：从零构建玩具评估器

参见 `code/main.py`。纯标准库实现的忠实度近似（答案声明与上下文的重叠）和相关性近似（答案词与问题词的重叠）。非生产级。只展示形态。

## 陷阱

- **无校准。** 与人工标签相关性 0.3 的评判者就是噪声。交付前必须进行校准运行。
- **自我评估。** 用同一个 LLM 生成和评判会使分数虚高 10-20%。评判者使用不同的模型家族。
- **配对评判中的位置偏差。** 评判者偏好第一个选项。始终随机顺序并双向运行。
- **原始聚合掩盖失败。** 平均分 0.85 通常掩盖了 5% 的灾难性失败。始终检查底部分位数。
- **标准数据集腐化。** 未版本化的评估集随时间漂移会破坏纵向比较。每次改动都给数据集打标签。
- **LLM 成本。** 规模化时，评判调用主导成本。使用满足校准阈值的最便宜模型。GPT-4o-mini、Claude Haiku、Mistral-small。

## 使用

2026 技术栈：

| 使用场景 | 框架 |
|---------|-----------|
| RAG 质量监控 | RAGAS（4 项指标） |
| CI/CD 回归门控 | DeepEval + pytest |
| 自定义领域标准 | DeepEval 中的 G-Eval |
| 在线实时流量监控 | 带参考自由模式的 RAGAS |
| 人工抽查 | LangSmith 或 Phoenix 带标注 UI |
| 红队 / 安全评估 | Promptfoo + DeepEval |

典型技术栈：RAGAS 用于监控，DeepEval 用于 CI，G-Eval 用于新维度。三个都跑；它们会有效地产生分歧。

## 交付

保存为 `outputs/skill-eval-architect.md`：

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## 练习

1. **简单。** 在 10 个有已知幻觉的 RAG 示例上使用 RAGAS。验证忠实度指标能捕获每一个。
2. **中等。** 手工标注 50 个 QA 答案的正确性 0-1。用 G-Eval 评分。测量评判者与人工之间的 Spearman rho。
3. **困难。** 用 DeepEval 构建 pytest CI 门控。故意让检索器回归。验证门控失败。通过对最低 10% 的阈值检查添加底部分位数告警。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| LLM 即评判者 | 用 LLM 评分 | 给定评分规则，提示评判模型对输出打分 0-1。 |
| RAGAS | RAG 指标库 | 开源评估框架，包含 4 个参考自由的 RAG 指标。 |
| 忠实度 | 答案有依据吗？ | 由检索上下文蕴含的答案陈述的比例。 |
| 上下文精确率 | 检索的块相关吗？ | Top-K 块中实际重要的比例。 |
| 上下文召回率 | 检索找到了所有内容吗？ | 由检索块支持的标准答案陈述的比例。 |
| G-Eval | 自定义 LLM 评判者 | 评分规则 + 思维链评估步骤 + 0-1 分数。 |
| 校准 | 信任但验证 | 评判分数与人工分数之间的 Spearman 相关性。 |

## 延伸阅读

- [Es 等 (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — RAGAS 论文。
- [Liu 等 (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — G-Eval 论文。
- [DeepEval 文档](https://deepeval.com/docs/metrics-introduction) — 开源生产级技术栈。
- [Zheng 等 (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — 偏差、校准、局限。
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — 统一框架，整合了 RAGAS、DeepEval、Phoenix。