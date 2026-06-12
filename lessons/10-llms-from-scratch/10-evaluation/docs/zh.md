# 评估：基准测试、评测、LM Harness

> 古德哈特定律：当一个指标变成目标时，它就不再是一个好指标。每个前沿实验室都在玩基准测试。MMLU 分数在涨，但模型仍然无法可靠地数出"strawberry"中字母 R 的数量。唯一重要的评测是你的评测——用你的任务、你的数据。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 10，第 01-05 课（从零构建 LLM）
**时间：** 约 90 分钟

## 学习目标

- 构建一个自定义评测工具，对语言模型运行多选和开放性基准测试
- 解释标准基准测试（MMLU、HumanEval）为何饱和并无法区分前沿模型
- 实现带正确指标的任务特定评测：精确匹配、F1、BLEU 和 LLM-as-judge 评分
- 设计针对你特定用例的自定义评测套件，而非仅依赖公开排行榜

## 问题

MMLU 于 2020 年发布，包含 15,908 道涵盖 57 个学科的问题。三年内，前沿模型就已将其饱和。GPT-4 得分 86.4%。Claude 3 Opus 得分 86.8%。Llama 3 405B 得分 88.6%。排行榜压缩在 3 分的范围内，差异是统计噪声，而非真实能力差距。

与此同时，这些模型在处理 10 岁孩子毫不费力就能完成的任务时却失败了。Claude 3.5 Sonnet 在 MMLU 上得分 88.7%，却无法数出"strawberry"中的字母——这项任务不需要任何世界知识，也不需要任何推理，只需要字符级迭代。HumanEval 用 164 道题测试代码生成。模型得分 90% 以上，却仍在产生任何初级开发人员都能发现边缘用例崩溃的代码。

基准测试性能与现实世界可靠性之间的差距是 LLM 评测的核心问题。基准测试告诉你模型在基准测试上的表现。它几乎无法告诉你该模型在你的特定任务、你的特定数据、你的特定失败模式下的表现。如果你正在构建一个客服机器人，MMLU 无关紧要。如果你正在构建一个代码助手，HumanEval 只覆盖函数级生成——它无法说明跨文件的调试、重构或代码解释。

你需要自定义评测。不是因为基准测试无用——它们对粗略的模型选择很有用——而是因为最终评测必须与你的部署条件完全匹配。

## 概念

### 评测领域

评测分为三类，各有不同的成本和信号质量。

**基准测试**是标准化测试套件。MMLU、HumanEval、SWE-bench、MATH、ARC、HellaSwag。你用模型运行基准测试并获得分数。优势：每个人都使用相同的测试，因此可以比较模型。劣势：模型和训练数据越来越多地污染了这些基准测试。实验室在包含基准测试问题的数据上训练。分数在涨。能力可能没有。

**自定义评测**是为你的特定用例构建的测试套件。你定义输入、期望输出和评分函数。法律文档摘要器在法律文档上评测。SQL 生成器在你的数据库模式上评测。这些创建成本昂贵，但它们是唯一能预测生产性能的评测。

**人工评测**使用付费标注员根据有用性、正确性、流畅性和安全性等标准评判模型输出。对于自动评分失败的开放性任务，这是黄金标准。Chatbot Arena 收集了 100 多个模型超过 200 万张人工偏好票。缺点：成本（每次判断 0.10-2.00 美元）和速度（数小时到数天）。

```mermaid
graph TD
    subgraph Eval["评测领域"]
        direction LR
        B["基准测试\n(MMLU, HumanEval)\n低成本，标准化\n可作弊，已过时"]
        C["自定义评测\n你的任务，你的数据\n信号最强\n构建成本高"]
        H["人工评测\n(Chatbot Arena)\n黄金标准\n速度慢，成本高"]
    end

    B -->|"粗略模型选择"| C
    C -->|"模糊情况"| H

    style B fill:#1a1a2e,stroke:#ffa500,color:#fff
    style C fill:#1a1a2e,stroke:#51cf66,color:#fff
    style H fill:#1a1a2e,stroke:#e94560,color:#fff
```

### 为何基准测试失效

三种机制导致基准测试分数停止反映真实能力。

**数据污染。** 训练语料从互联网抓取。基准测试问题存在于互联网上。模型在训练期间看到答案。这不是传统意义上的作弊——实验室不会故意包含基准测试数据。但网络规模抓取使其几乎不可能排除。

**应试教育。** 实验室优化训练混合以提高基准测试性能。如果训练混合中有 5% 是 MMLU 风格的四选一，模型就会学习格式和答案分布。MMLU 是四选一。模型了解到答案在 A/B/C/D 上大致均匀分布，这有助于模型在不知道答案时也能猜对。

**饱和。** 当每个前沿模型在基准测试上都得分 85-90% 时，基准测试就失去了区分能力。剩余的 10-15% 问题可能是模糊的、标记错误的或需要冷门领域知识的。在 MMLU 上从 87% 提高到 89% 可能意味着模型记住了两个更冷门的问题，而不是它变得更聪明了。

### 困惑度：快速健康检查

困惑度衡量模型对一系列 token 的惊讶程度。从形式上讲，它是指数化的平均负对数似然：

```
PPL = exp(-1/N * sum(log P(token_i | context)))
```

困惑度为 10 意味着模型在每个 token 位置平均与在 10 个选项中均匀选择一样不确定。越低越好。GPT-2 在 WikiText-103 上的困惑度约为 30。GPT-3 约为 20。Llama 3 8B 约为 7。

困惑度对于在同一测试集上比较模型很有用，但它有盲点。模型可以通过擅长预测常见模式来获得低困惑度，而在罕见但重要的模式上表现糟糕。它也不能说明指令遵循、推理或事实准确性。用它作为健全性检查，而不是最终裁决。

### LLM-as-Judge

用强模型评估弱模型的输出。想法很简单：让 GPT-4o 或 Claude Sonnet 在正确性、有用性和安全性方面以 1-5 分评估响应。使用 GPT-4o-mini 每次判断成本约 0.01 美元，并且与人工判断惊人地相关——在大多数任务上约 80% 一致性。

评分提示词比模型更重要。模糊的提示词（"评估此响应"）会产生嘈杂的分数。具有评分标准的结构化提示词（"如果答案事实正确且引用了来源则得 5 分，如果正确但无来源则得 4 分，如果部分正确..."）会产生一致、可重复的分数。

失败模式：评判模型表现出位置偏差（在成对比较中偏好第一个响应）、冗长偏差（偏好更长的响应）和自我偏好（GPT-4 给 GPT-4 输出评分高于等效的 Claude 输出）。缓解措施：随机顺序、归一化长度、使用与被评估模型不同的评判模型。

### 成对比较的 ELO 评分

Chatbot Arena 的方法。向同一条提示的两个不同模型响应展示给人类（或 LLM 评判器）。人类选择更好的一个。从数千次这些比较中，计算每个模型的 ELO 评分——与国际象棋使用的系统相同。

ELO 优势：相对排名比绝对评分更可靠，优雅地处理平局，并且比独立评分每个输出收敛得更快。截至 2026 年初，Chatbot Arena 排行榜显示 GPT-4o、Claude 3.5 Sonnet 和 Gemini 1.5 Pro 在顶部彼此相差 20 ELO 分以内。

```mermaid
graph LR
    subgraph ELO["ELO 评分流程"]
        direction TB
        P["提示词"] --> MA["模型 A 输出"]
        P --> MB["模型 B 输出"]
        MA --> J["评判器\n(人类或 LLM)"]
        MB --> J
        J --> W["A 胜 / B 胜 / 平局"]
        W --> E["ELO 更新\nK=32"]
    end

    style P fill:#1a1a2e,stroke:#0f3460,color:#fff
    style J fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#51cf66,color:#fff
```

### 评测框架

**lm-evaluation-harness**（EleutherAI）：标准的开源评测框架。支持 200 多个基准测试。用一条命令对任何 Hugging Face 模型运行 MMLU、HellaSwag、ARC 等。由 Open LLM Leaderboard 使用。

**RAGAS**：专门用于 RAG 流程的评测框架。衡量忠实度（答案是否与检索到的上下文匹配？）、相关性（检索到的上下文与问题相关吗？）和答案正确性。

**promptfoo**：配置驱动的提示工程评测。用 YAML 定义测试用例，对多个模型运行，获得通过/失败报告。对回归测试提示词很有用——确保提示词更改不会破坏现有测试用例。

### 构建自定义评测

唯一对生产重要的评测。流程如下：

1. **定义任务。** 模型具体应该做什么？要精确。"回答问题"太模糊。"给定客户投诉邮件，提取产品名称、问题类别和情感"是一个你可以评测的任务。

2. **创建测试用例。** 原型评测最少 50 个，生产环境 200 个以上。每个测试用例是一个（输入，期望输出）对。包括边缘用例：空输入、对抗性输入、模糊输入、其他语言的输入。

3. **定义评分。** 结构化输出的精确匹配。文本相似度的 BLEU/ROUGE。开放性质量的 LLM-as-judge。提取任务的 F1。用权重组合多个指标。

4. **自动化。** 每个评测用一条命令运行。没有手动步骤。以支持随时间比较的格式存储结果。

5. **跟踪趋势。** 单独的评测分数毫无意义。你需要趋势线。上次提示词更改后分数提高了吗？切换模型后倒退了吗？提示词版本和评测版本保持一致。

| 评测类型 | 每次判断成本 | 与人工一致性 | 最适用于 |
|-----------|------------------|----------------------|----------|
| 精确匹配 | ~$0 | 100%（当适用时） | 结构化输出、分类 |
| BLEU/ROUGE | ~$0 | ~60% | 翻译、摘要 |
| LLM-as-judge | ~$0.01 | ~80% | 开放性生成 |
| 人工评测 | $0.10-$2.00 | 不适用（本身就是真值） | 模糊、高风险任务 |

## 动手构建

### 第 1 步：最小评测框架

定义核心抽象。评测用例有输入、期望输出和可选的元数据字典。评分器接受预测和参考并返回 0 到 1 之间的分数。

```python
import json
from collections import Counter

class EvalCase:
    def __init__(self, input_text, expected, metadata=None):
        self.input_text = input_text
        self.expected = expected
        self.metadata = metadata or {}

class EvalSuite:
    def __init__(self, name, cases, scorers):
        self.name = name
        self.cases = cases
        self.scorers = scorers

    def run(self, model_fn):
        results = []
        for case in self.cases:
            prediction = model_fn(case.input_text)
            scores = {}
            for scorer_name, scorer_fn in self.scorers.items():
                scores[scorer_name] = scorer_fn(prediction, case.expected)
            results.append({
                "input": case.input_text,
                "expected": case.expected,
                "prediction": prediction,
                "scores": scores,
            })
        return results
```

### 第 2 步：评分函数

构建精确匹配、token F1 和模拟 LLM-as-judge 评分器。

```python
def exact_match(prediction, expected):
    return 1.0 if prediction.strip().lower() == expected.strip().lower() else 0.0

def token_f1(prediction, expected):
    pred_tokens = set(prediction.lower().split())
    exp_tokens = set(expected.lower().split())
    if not pred_tokens or not exp_tokens:
        return 0.0
    common = pred_tokens & exp_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(exp_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def llm_judge_simulated(prediction, expected):
    pred_words = set(prediction.lower().split())
    exp_words = set(expected.lower().split())
    if not exp_words:
        return 0.0
    overlap = len(pred_words & exp_words) / len(exp_words)
    length_penalty = min(1.0, len(prediction) / max(len(expected), 1))
    return round(overlap * 0.7 + length_penalty * 0.3, 3)
```

### 第 3 步：ELO 评分系统

实现带 ELO 更新的成对比较。这正是 Chatbot Arena 用来排名模型的系统。

```python
class ELOTracker:
    def __init__(self, k=32, initial_rating=1500):
        self.ratings = {}
        self.k = k
        self.initial_rating = initial_rating
        self.history = []

    def _ensure_player(self, name):
        if name not in self.ratings:
            self.ratings[name] = self.initial_rating

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def record_match(self, player_a, player_b, outcome):
        self._ensure_player(player_a)
        self._ensure_player(player_b)

        ea = self.expected_score(self.ratings[player_a], self.ratings[player_b])
        eb = 1 - ea

        if outcome == "a":
            sa, sb = 1.0, 0.0
        elif outcome == "b":
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        self.ratings[player_a] += self.k * (sa - ea)
        self.ratings[player_b] += self.k * (sb - eb)

        self.history.append({
            "a": player_a, "b": player_b,
            "outcome": outcome,
            "rating_a": round(self.ratings[player_a], 1),
            "rating_b": round(self.ratings[player_b], 1),
        })

    def leaderboard(self):
        return sorted(self.ratings.items(), key=lambda x: -x[1])
```

### 第 4 步：困惑度计算

使用 token 概率计算困惑度。在实践中，你会从模型的 logits 中获取这些。这里我们用概率分布模拟。

```python
import numpy as np

def perplexity(log_probs):
    if not log_probs:
        return float("inf")
    avg_neg_log_prob = -np.mean(log_probs)
    return float(np.exp(avg_neg_log_prob))

def token_log_probs_simulated(text, model_quality=0.8):
    np.random.seed(hash(text) % 2**31)
    tokens = text.split()
    log_probs = []
    for i, token in enumerate(tokens):
        base_prob = model_quality
        if len(token) > 8:
            base_prob *= 0.6
        if i == 0:
            base_prob *= 0.7
        prob = np.clip(base_prob + np.random.normal(0, 0.1), 0.01, 0.99)
        log_probs.append(float(np.log(prob)))
    return log_probs
```

### 第 5 步：聚合结果

计算评测运行的汇总统计：均值、中位数、阈值通过率以及按指标细分。

```python
def summarize_results(results, threshold=0.8):
    all_scores = {}
    for r in results:
        for metric, score in r["scores"].items():
            all_scores.setdefault(metric, []).append(score)

    summary = {}
    for metric, scores in all_scores.items():
        arr = np.array(scores)
        summary[metric] = {
            "mean": round(float(np.mean(arr)), 3),
            "median": round(float(np.median(arr)), 3),
            "std": round(float(np.std(arr)), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
            "pass_rate": round(float(np.mean(arr >= threshold)), 3),
            "n": len(scores),
        }
    return summary

def print_summary(summary, suite_name="Eval"):
    print(f"\n{'=' * 60}")
    print(f"  {suite_name} 汇总")
    print(f"{'=' * 60}")
    for metric, stats in summary.items():
        print(f"\n  {metric}:")
        print(f"    均值:      {stats['mean']:.3f}")
        print(f"    中位数:    {stats['median']:.3f}")
        print(f"    标准差:    {stats['std']:.3f}")
        print(f"    范围:      [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"    通过率:    {stats['pass_rate']:.1%} (阈值 >= 0.8)")
        print(f"    样本数:    {stats['n']}")
```

### 第 6 步：运行完整流程

将所有内容连接起来。定义任务、创建测试用例、模拟两个模型、运行评测、从成对比较计算 ELO 并打印排行榜。

```python
def demo_model_good(prompt):
    responses = {
        "What is the capital of France?": "Paris",
        "What is 2 + 2?": "4",
        "Who wrote Hamlet?": "William Shakespeare",
        "What language is PyTorch written in?": "Python and C++",
        "What is the boiling point of water?": "100 degrees Celsius",
    }
    return responses.get(prompt, "I don't know")

def demo_model_bad(prompt):
    responses = {
        "What is the capital of France?": "Paris is the capital city of France",
        "What is 2 + 2?": "The answer is four",
        "Who wrote Hamlet?": "Shakespeare",
        "What language is PyTorch written in?": "Python",
        "What is the boiling point of water?": "212 Fahrenheit",
    }
    return responses.get(prompt, "Unknown")

cases = [
    EvalCase("What is the capital of France?", "Paris"),
    EvalCase("What is 2 + 2?", "4"),
    EvalCase("Who wrote Hamlet?", "William Shakespeare"),
    EvalCase("What language is PyTorch written in?", "Python and C++"),
    EvalCase("What is the boiling point of water?", "100 degrees Celsius"),
]

suite = EvalSuite(
    name="General Knowledge",
    cases=cases,
    scorers={
        "exact_match": exact_match,
        "token_f1": token_f1,
        "llm_judge": llm_judge_simulated,
    },
)

results_good = suite.run(demo_model_good)
results_bad = suite.run(demo_model_bad)

print_summary(summarize_results(results_good), "Model A (concise)")
print_summary(summarize_results(results_bad), "Model B (verbose)")
```

"好"模型给出精确答案。"坏"模型给出冗长的改写。精确匹配严厉惩罚冗长模型。Token F1 和 LLM-as-judge 更宽容。这说明了指标选择的重要性：同一个模型根据评分方式看起来很好或很糟糕。

### 第 7 步：ELO 锦标赛

在多轮中运行模型之间的成对比较。

```python
elo = ELOTracker(k=32)

for case in cases:
    pred_a = demo_model_good(case.input_text)
    pred_b = demo_model_bad(case.input_text)

    score_a = token_f1(pred_a, case.expected)
    score_b = token_f1(pred_b, case.expected)

    if score_a > score_b:
        outcome = "a"
    elif score_b > score_a:
        outcome = "b"
    else:
        outcome = "tie"

    elo.record_match("model_a_concise", "model_b_verbose", outcome)

print("\nELO 排行榜:")
for name, rating in elo.leaderboard():
    print(f"  {name}: {rating:.0f}")
```

### 第 8 步：困惑度比较

比较不同质量级别"模型"的困惑度。

```python
test_text = "The quick brown fox jumps over the lazy dog in the garden"

for quality, label in [(0.9, "Strong model"), (0.7, "Medium model"), (0.4, "Weak model")]:
    log_probs = token_log_probs_simulated(test_text, model_quality=quality)
    ppl = perplexity(log_probs)
    print(f"  {label} (quality={quality}): perplexity = {ppl:.2f}")
```

## 实际使用

### lm-evaluation-harness（EleutherAI）

在任何模型上运行基准测试的标准工具。

```python
# pip install lm-eval
# 命令行:
# lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B --tasks mmlu --batch_size 8

# Python API:
# import lm_eval
# results = lm_eval.simple_evaluate(
#     model="hf",
#     model_args="pretrained=meta-llama/Llama-3.1-8B",
#     tasks=["mmlu", "hellaswag", "arc_easy"],
#     batch_size=8,
# )
# print(results["results"])
```

### promptfoo

配置驱动的提示工程评测。在 YAML 中定义测试并针对多个提供商运行。

```yaml
# promptfoo.yaml
providers:
  - openai:gpt-4o-mini
  - anthropic:claude-3-haiku

prompts:
  - "Answer in one word: {{question}}"

tests:
  - vars:
      question: "What is the capital of France?"
    assert:
      - type: contains
        value: "Paris"
  - vars:
      question: "What is 2 + 2?"
    assert:
      - type: equals
        value: "4"
```

### RAGAS 用于 RAG 评测

```python
# pip install ragas
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision
#
# result = evaluate(
#     dataset,
#     metrics=[faithfulness, answer_relevancy, context_precision],
# )
# print(result)
```

RAGAS 衡量通用评测遗漏的内容：模型的答案是否基于检索到的上下文，而不仅仅是答案在抽象意义上是否"正确"。

## 交付物

本课产出 `outputs/prompt-eval-designer.md`——一个用于为任何任务设计自定义评测套件的可复用提示词。给它一个任务描述，它会生成测试用例、评分函数和通过/失败阈值建议。

它还产出 `outputs/skill-llm-evaluation.md`——一个基于任务类型、预算和延迟要求选择正确评测策略的决策框架。

## 练习

1. 添加一个"一致性"评分器，对同一输入运行模型 5 次并测量输出匹配的频率。在确定性输入上不一致的答案揭示了脆弱的提示词或高温度设置。

2. 扩展 ELO 追踪器以支持多个评判函数（精确匹配、F1、LLM-as-judge）并对它们进行加权。当你对精确匹配加权较重与对 F1 加权较重时，比较排行榜如何变化。

3. 为特定任务构建评测套件：将电子邮件分类为 5 个类别。创建 100 个包含边缘用例的多样化测试用例（可属于多个类别的电子邮件、空电子邮件、其他语言的电子邮件）。测量不同"模型"（基于规则的、关键词匹配的、模拟 LLM）的表现。

4. 实现污染检测：给定一组评测问题和训练语料库，检查评测问题（或近似改写）中有多少百分比出现在训练数据中。这是研究人员审计基准测试有效性的方式。

5. 构建一个"模型 diff"工具。给定两个模型版本的评测结果，突出显示哪些具体测试用例改进了，哪些倒退了，哪些保持不变。这是代码 diff 的评测等价物——对于理解更改是否有帮助或有害至关重要。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|----------------------|
| MMLU | "那个基准测试" | 大规模多任务语言理解——15,908 道涵盖 57 个学科的四选一问题，截至 2025 年已在 88% 以上饱和 |
| HumanEval | "代码评测" | 来自 OpenAI 的 164 道 Python 函数补全问题，仅测试孤立的函数生成 |
| SWE-bench | "真实编码评测" | 来自 12 个 Python 仓库的 2,294 个 GitHub issue，衡量端到端 bug 修复，包括测试生成 |
| 困惑度 | "模型有多困惑" | exp(-avg(log P(token_i given context)))——越低意味着模型对实际 token 赋予更高的概率 |
| ELO 评分 | "模型的国际象棋排名" | 从成对胜负记录计算出的相对技能评分，Chatbot Arena 用它对 100 多个模型进行排名 |
| LLM-as-judge | "用 AI 评分 AI" | 强模型根据评分标准对弱模型的输出进行评分，与人工判断的一致性约 80%，每次判断成本约 0.01 美元 |
| 数据污染 | "模型看过测试" | 训练数据包含基准测试问题，在不提高真实能力的情况下提高分数 |
| 评测套件 | "一堆测试" | 衡量特定能力的版本化（输入、期望输出、评分器）三元组集合 |
| 通过率 | "正确的百分比" | 分数高于阈值的评测用例比例——比平均分数更有可操作性，因为它衡量的是可靠性 |
| Chatbot Arena | "模型排名网站" | LMSYS 平台，拥有 200 万张以上人工偏好票，通过 ELO 评分产生最可信的 LLM 排行榜 |

## 延伸阅读

- [Hendrycks et al., 2021 -- "Measuring Massive Multitask Language Understanding"](https://arxiv.org/abs/2009.03300) -- MMLU 论文，尽管饱和但仍是被引用最多的 LLM 基准测试
- [Chen et al., 2021 -- "Evaluating Large Language Models Trained on Code"](https://arxiv.org/abs/2107.03374) -- OpenAI 的 HumanEval 论文，建立了代码生成评测方法论
- [Zheng et al., 2023 -- "Judging LLM-as-a-Judge"](https://arxiv.org/abs/2306.05685) -- 系统分析使用 LLM 评估 LLM，包括位置偏差和冗长偏差的发现
- [LMSYS Chatbot Arena](https://chat.lmsys.org/) -- 众包模型比较平台，拥有 200 万张以上投票，通过 ELO 评分产生最可信的真实世界 LLM 排名