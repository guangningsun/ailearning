# Transformer 之前的文本生成 — N-gram 语言模型

> 如果一个词令人惊讶，模型就不好。困惑度让惊讶变成一个数字。平滑让它保持有限。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 5 · 01（文本处理）、阶段 2 · 14（朴素贝叶斯）
**时间：** 约 45 分钟

## 问题

在 transformers之前，在 RNNs 之前，在词 embedding 之前，语言模型通过计数前 n-1 个词之后它出现的频率来预测下一个词。计数"the cat" → "sat" 47 次，"the cat" → "jumped" 12 次，"the cat" → "refrigerator" 0 次。归一化得到概率分布。

这就是 N-gram 语言模型。它驱动了从 1980 年到 2015 年的每一个语音识别器、每一个拼写检查器和每一个基于短语的机器翻译系统。当你需要廉价的设备端语言建模时，它仍然在运行。

有趣的问题是如何处理未见的 N-gram。基于原始计数的模型给任何未见过的东西分配零概率，这是灾难性的，因为句子很长，几乎每个长句子都至少包含一个未见序列。五十年的平滑研究解决了这个问题。Kneser-Ney 平滑就是结果，现代深度学习继承了其经验传统。

## 概念

![N-gram 模型：计数、平滑、生成](../assets/ngram.svg)

**N-gram 概率：** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。固定 `n`（通常3 是三元组，4 是四元组）。从计数计算：

```text
P(w | context) = count(context, w) / count(context)
```

**零计数问题。** 训练中未见过的任何 N-gram 概率为零。2007 年对 Brown 语料库的研究发现，即使 4-gram 模型也有 30% 的保留 4-gram 在训练中未见。你不能不做平滑就评估任何真实文本。

**平滑方法，按复杂度排序：**

1. **拉普拉斯（加一）。** 每个计数加 1。简单，对罕见事件效果很差。
2. **Good-Turing。** 基于频率-频率的分布，将高频事件的概率质量重新分配给未见事件。
3. **插值。** 用可调权重组合 N-gram、(n-1)-gram 等估计。
4. **回退。** 如果 N-gram 计数为零，回退到 (n-1)-gram。Katz 回退对此进行了归一化。
5. **绝对折扣。** 从所有计数中减去一个固定折扣 `D`，重新分配给未见事件。
6. **Kneser-Ney。** 绝对折扣加上低阶模型的巧妙选择：使用*接续概率*（词出现在多少个上下文中）而不是原始频率。

Kneser-Ney 的见解很深刻。"San Francisco" 是一个常见二元组。单元词 "Francisco" 大多出现在 "San" 之后。朴素的绝对折扣给 "Francisco" 很高的单元概率（因为计数很高）。Kneser-Ney 注意到 "Francisco" 只出现在一个上下文中，并相应地降低其接续概率。结果：以"Francisco"结尾的新二元组获得适当低的概率。

**评估：困惑度。** 在保留测试集上每个词的平均负对数似然的指数。越低越好。困惑度为 100 意味着模型就像在 100 个词中均匀选择一样困惑。

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

## 构建

### 第 1 步：三元组计数

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

输入是分好词的句子列表。输出是 N-gram 计数和上下文计数。`<s>` 和 `</s>` 是句子边界。

### 第 2 步：拉普拉斯平滑

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

每个计数加 1。会平滑，但过度分配质量给未见事件，也伤害了罕见已知事件。

### 第 3 步：Kneser-Ney（二元组，插值）

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

三个运动部件。`continuation_prob` 捕获"这个词出现在多少个不同的上下文中？"（Kneser-Ney 的创新）。`lambda_prev` 是折扣释放的质量，用于加权回退。最终概率是折现主项加上加权接续项。

### 第 4 步：用采样生成文本

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

按概率比例采样。每次运行根据种子给出不同输出。对于类似束搜索的输出，每步取 argmax（贪心）并添加一个小的随机性旋钮（温度）。

### 第 5 步：困惑度

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

越低越好。对于 Brown 语料库，调优良好的 4-gram KN 模型困惑度约为 140。 transformer LM 在相同测试集上达到 15-30。差距约为 10 倍。这就是该领域继续前进的原因。

## 使用

- **经典 NLP 教学。** 你能获得的对平滑、MLE 和困惑度最清晰的讲解。
- **KenLM。** 生产级 N-gram 库。在语音和 MT 系统中用作重排序器，当低延迟很重要时。
- **设备端自动补全。** 键盘中的三元组模型。仍然在使用。
- **基线。** 在宣布你的神经 LM 良好之前始终计算 N-gram LM 困惑度。如果你的 transformer 没有大幅超越 KN，就有问题。

## 交付

保存为 `outputs/prompt-lm-baseline.md`：

```markdown
---
name: lm-baseline
description: 在训练神经 LM 之前构建可复现的 N-gram 语言模型基线。
phase: 5
lesson: 16
---

给定语料库和目标用途（下一个词预测、重排序、困惑度基线），输出：

1. N-gram 阶数。通用英语用三元组，语料库大用四元组，语音重排序用五元组。
2. 平滑。修正的 Kneser-Ney 是默认的；仅用于教学用拉普拉斯。
3. 库。生产用 `kenlm`，教学用 `nltk.lm`，仅在学习目的时自己实现。
4. 评估。保留集上的困惑度，训练和测试之间使用一致的 tokenization。

拒绝报告在不同 tokenization 的系统之间计算的困惑度——只有在使用相同的 tokenization 时困惑度数字才可比较。标记测试集中的 OOV 率；KN 处理 OOV 很差，除非你在训练期间保留一个特殊的<UNK> 标记。
```

## 练习

1. **简单。** 在 1,000 句莎士比亚语料库上训练三元组 LM。生成 20 句话。它们在局部上是可信的，但在全局上不连贯。这是经典演示。
2. **中等。** 在保留的莎士比亚分割上为你的 KN 模型实现困惑度。与拉普拉斯比较。你应该看到 KN 将困惑度降低 30-50%。
3. **困难。** 构建一个三元组拼写纠正器：给定一个拼写错误的词及其上下文，在 LM 下按上下文概率生成更正并排序。在 Birkbeck 拼写字典（公开）上评估。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| N-gram | 词序列 | `n` 个连续标记的序列。 |
| 平滑 (Smoothing) | 避免零 | 重新分配概率质量，使未见事件获得非零概率。 |
| 困惑度 (Perplexity) | LM 质量指标 | 保留数据上 `exp(-平均对数概率)`。越低越好。 |
| 回退 (Backoff) | 回退到更短上下文 | 如果三元组计数为零，使用二元组。Katz 回退将此形式化。 |
| Kneser-Ney | N-gram 最佳平滑 | 绝对折扣 + 低阶模型的接续概率。 |
| 接续概率 (Continuation probability) | KN特有 | 按`w`出现的上下文数量加权的 `P(w)`，而不是原始计数。 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — N-gram LM 和平滑的权威处理。
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — 确立 Kneser-Ney 为最佳 N-gram 平滑器的论文。
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — 原始 KN 论文。
- [KenLM](https://kheafield.com/code/kenlm/) — 快速生产级 N-gram LM，在 2026 年仍用于延迟敏感应用。