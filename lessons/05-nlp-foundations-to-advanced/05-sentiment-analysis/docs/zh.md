# 情感分析

> 经典的 NLP 任务。你需要了解的关于经典文本分类的几乎所有知识都在这里。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 5 · 02（BoW + TF-IDF）、阶段 2 · 14（朴素贝叶斯）
**时间：** 约 75 分钟

## 问题

"The food was not great." 正向还是负向？

情感分析听起来很简单。评论者说他喜欢或不喜欢某样东西。给句子打上标签。它成为经典 NLP 任务的原因在于：每一个看似简单的情况背后都藏着一个困难的情况。否定翻转语义。反讽反转语义。"Not bad at all" 虽然包含两个负向词却是正向的。表情符号传递的信号比周围文本更多。领域词汇很重要（音乐评论中的 `tight` 与时尚评论中的 `tight` 含义不同）。

情感分析是经典 NLP 的实验场。如果你理解了为什么每个朴素基线都有特定的失败模式，你就理解了为什么每个更丰富的模型会被发明出来。本课从零构建一个朴素贝叶斯基线，添加逻辑回归，并指出那些使生产级情感分析成为合规级别问题的陷阱。

## 概念

经典情感分析是一个两步流程。

1. **表示。** 把文本转化为特征向量。BoW、TF-IDF 或 n-gram。
2. **分类。** 在标注样本上拟合线性模型（朴素贝叶斯、逻辑回归、SVM）。

朴素贝叶斯是能work的最简单的模型。假设每个特征在给定标签的条件下互相独立。从计数中估计 `P(word | positive)` 和 `P(word | negative)`。推理时，将概率相乘。"朴素"的独立性假设看似可笑，但结果却出奇地强大。原因在于：对于稀疏文本特征和中等级别的数据，分类器更关注每个词倾向于哪一侧，而不是其强度。

逻辑回归修正了独立性假设。它为每个特征学习一个权重，包括负权重。`not good` 作为 bigram 特征会获得一个负权重。朴素贝叶斯无法对从未标注过的 bigram 做到这一点。

## 构建

### 第 1 步：一个真实的小数据集

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

刻意做小。真实工作需要数万条样本（IMDb、SST-2、Yelp polarity）。数学原理相同。

### 第 2 步：从零实现多项式朴素贝叶斯

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

加性平滑（alpha=1.0）即拉普拉斯平滑。没有它，未在某个类别中出现的词的概率为零，对数会爆炸。实践中常用 `alpha=0.01`。`alpha=1.0` 是教学默认值。

### 第 3 步：从零实现逻辑回归

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2 正则化在这里很重要。文本特征是稀疏的；没有 L2，模型会记住训练样本。从 `0.01` 开始调优。

### 第 4 步：处理否定（失败模式）

考虑 "not good" 和 "not bad"。BoW 分类器看到 `{not, good}` 和 `{not, bad}`，从训练中出现的更多的那一个学习。Bigram 分类器看到 `not_good` 和 `not_bad`，将它们作为不同的特征学习。通常这就足够了。

当你没有 bigram 时，一个更粗暴但有效的方法是：**否定作用域**。在否定词之后的 token 前面加上 `NOT_` 前缀，直到下一个标点符号。

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

现在 `good` 和 `NOT_good` 是不同的特征。分类器可以给它们赋予相反的权重。三行预处理，在情感基准上能带来可测量的准确率提升。

### 第 5 步：重要的评估指标

如果类别不平衡，单独看准确率会误导人。真实的情感语料通常 70-80% 是正向或 70-80% 是负向的；一个始终预测多数类的分类器就能达到 80% 准确率，但毫无价值。请报告以下所有指标：

- **每类精确率和召回率。** 每个类别一对。宏平均得到一个数字，尊重类别平衡。
- **宏-F1（类别不平衡数据的主要指标）。** 每类 F1 分数的平均值，等权重。当类别不平衡时用这个而不是准确率。
- **加权-F1（备选）。** 与宏相同，但按类别频率加权。当不平衡本身具有业务含义时，与宏-F1 一起报告。
- **混淆矩阵。** 原始计数。在信任任何标量指标之前总要检查它；它揭示了模型会在哪两个类别之间混淆。
- **每类错误样本。** 每个类别取 5 个错误预测。读它们。读实际错误无可替代。

对于严重不平衡的数据（> 95-5 比率），报告 **AUROC** 和 **AUPRC** 而不是准确率。AUPRC 对少数类更敏感，而这通常才是你关心的（垃圾邮件、欺诈、稀有情感）。

**需要避免的常见错误。** 在不平衡数据上报告 micro-F1 而不是 macro-F1 会给出一个看起来很高的数字，因为它被多数类主导。Macro-F1 强迫你看到少数类的表现。

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## 使用

scikit-learn 六行代码，正确实现。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

注意三点。`stop_words=None` 保留否定词。`ngram_range=(1, 2)` 添加 bigram，使 `not_good` 成为一个特征。`sublinear_tf=True` 抑制重复词。这三个参数是75% 准确率基线和 85% 准确率基线在 SST-2 上的区别。

### 何时使用 Transformer

- 讽刺检测。经典模型在这里完全失败。
- 长评论中情感在文档中途发生变化。
- 基于方面的情感。"相机很好但电池很糟糕。"你需要将情感归因于各个方面。只有 Transformer 或结构化输出模型。
- 非英语、低资源语言。多语言 BERT 为你免费提供零样本基线。

如果你需要以上任何一项，跳到阶段 7（Transformer 深度研究）。否则，在 TF-IDF 加 bigram 加否定处理上使用朴素贝叶斯或逻辑回归是你2026 年的生产基线。

### 可复现性陷阱（再次提醒）

重新训练情感模型是家常便饭。重新评估却不是。论文中报告的准确率数字使用了特定的划分、特定的预处理、特定的 tokenizer。如果你用不同的 pipeline 比较你的新模型和基线，你会得到误导性的差异。始终在你的 pipeline 上重新生成基线，而不是论文中的数字。

## 交付

保存为 `outputs/prompt-sentiment-baseline.md`：

```markdown
---
name: sentiment-baseline
description: 为新数据集设计情感分析基线。
phase: 5
lesson: 05
---

给定数据集描述（领域、语言、大小、标签粒度、延迟预算），你输出：

1. 特征提取方案。指定 tokenizer、n-gram 范围、停用词策略（通常保留）、否定处理（作用域前缀或 bigram）。
2. 分类器。基线用朴素贝叶斯，生产用逻辑回归，只有当领域需要讽刺/方面/跨语言时才用 Transformer。
3. 评估计划。报告精确率、召回率、F1、混淆矩阵和每类错误样本（不只是标量）。
4.部署后要监控的一个失败模式。领域漂移和讽刺是前两名。

拒绝为情感任务删除停用词。拒绝在类别不平衡时（如90% 正向）将准确率作为唯一指标报告。标记子词丰富的语言需要 FastText 或 Transformer embedding 而不是词级 TF-IDF。
```

## 练习

1. **简单。** 在 scikit-learn pipeline 中添加 `apply_negation` 作为预处理步骤，在小型情感数据集上测量 F1 变化。
2. **中等。** 实现类别加权逻辑回归（传递 `class_weight="balanced"` 给 scikit-learn，或自己推导梯度）。在合成的 90-10 类别不平衡上测量效果。
3. **困难。** 在情感模型残差上训练第二个分类器来构建讽刺检测器。记录你的实验设置。当准确率低于随机水平时警告读者（两类讽刺的随机水平约为 50%，大多数第一次尝试都落在这里）。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 极性 | 正向或负向 | 二元标签；有时扩展到中性或细粒度（五星）。 |
| 基于方面的情感 | 每方面极性 | 将情感归因于文本中提到的特定实体或属性。 |
| 否定作用域 | 反转附近 token | 在 "not" 之后的 token 前面加上 `NOT_` 前缀，直到标点符号。 |
| 拉普拉斯平滑 | 给计数加 1 | 防止朴素贝叶斯中出现零概率特征。 |
| L2 正则化 | 收缩权重 | 在损失中添加 `lambda * sum(w^2)`。对稀疏文本特征至关重要。 |

## 延伸阅读

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — 开创性综述。很长，但前四节覆盖了所有经典内容。
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — 这篇论文展示了 bigram 加朴素贝叶斯在短文本上难以击败。
- [scikit-learn 文本特征提取文档](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — `CountVectorizer`、`TfidfVectorizer` 和你将调优的每个旋钮的参考。