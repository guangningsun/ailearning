# 信息论

> 信息论度量惊讶。损失函数就建立在它之上。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 1，第 06 课（概率论）
**时间：** 约 60 分钟

## 学习目标

- 从零计算熵、交叉熵和 KL 散度，并解释三者之间的关系
- 推导为什么最小化交叉熵损失等价于最大化对数似然
- 计算特征与目标之间的互信息，据此对特征重要性排序
- 解释困惑度（perplexity）的含义：语言模型每一步实际在多大的词汇表里做选择

## 问题

你在每个分类模型里都调用 `CrossEntropyLoss()`。每篇语言模型论文里你都看到"困惑度"。你在 VAE、知识蒸馏和 RLHF 中读到 KL 散度。这些不是孤立的概念，它们是同一个想法戴上了不同的帽子。

信息论给了你一套语言，用来推理不确定性、压缩和预测。克劳德·香农 1948 年为了解决通信问题发明了它。结果发现，训练神经网络本身就是一个通信问题：模型试图通过一条由学习到的权重构成的噪声信道，把正确标签传过去。

本课从零推导每一个公式，让你看清它们从何而来、为什么有效。

## 概念

### 信息量（惊讶度）

当不太可能的事情发生时，它携带的信息更多。硬币掷出正面？不惊讶。中了彩票？非常惊讶。

概率为 p 的事件，其信息量为：

```
I(x) = -log(p(x))
```

以 2 为底得到比特（bits），以 e 为底得到奈特（nats）。同样的思想，不同的单位。

```
事件              概率          惊讶度（比特）
公平硬币正面       0.5           1.0
掷出 6 点         0.167         2.58
千分之一事件       0.001         9.97
必然事件           1.0           0.0
```

必然事件携带零信息 —— 你本来就知道它会发生。

### 熵（平均惊讶度）

熵是一个分布所有可能结果的信息量的期望。

```
H(P) = -sum( p(x) * log(p(x)) )  对所有 x
```

公平硬币的熵在二值变量中达到最大：1 比特。一枚有偏硬币（99% 正面）的熵很低：0.08 比特。你早就知道结果会是什么，所以每次抛掷几乎不提供任何信息。

```
公平硬币：  H = -(0.5 * log2(0.5) + 0.5 * log2(0.5)) = 1.0 比特
有偏硬币：  H = -(0.99 * log2(0.99) + 0.01 * log2(0.01)) = 0.08 比特
```

熵度量了分布中不可消除的不确定性。你无法压缩到熵值以下。

### 交叉熵（你每天都在用的损失函数）

交叉熵度量的是：用分布 Q 来编码实际上来自分布 P 的事件，平均惊讶度是多少。

```
H(P, Q) = -sum( p(x) * log(q(x)) )  对所有 x
```

P 是真实分布（标签），Q 是模型的预测。如果 Q 完美匹配 P，交叉熵就等于熵。任何不匹配都会让它变大。

在分类任务中，P 是独热向量（真实类别概率为 1，其他为 0）。交叉熵简化为：

```
H(P, Q) = -log(q(真实类别))
```

这就是分类问题中交叉熵损失的全部公式。最大化正确类别的预测概率。

### KL 散度（分布之间的距离）

KL 散度度量的是用 Q 代替 P 时，你多付出了多少惊讶度。

```
D_KL(P || Q) = sum( p(x) * log(p(x) / q(x)) )  对所有 x
             = H(P, Q) - H(P)
```

交叉熵等于熵加 KL 散度。训练时真实分布的熵是常数，因此最小化交叉熵就是最小化 KL 散度。你在把模型的分布推向真实分布。

KL 散度不对称：D_KL(P || Q) ≠ D_KL(Q || P)。它不是一个真正的距离度量。

### 互信息

互信息度量的是：知道一个变量能让你对另一个变量了解多少。

```
I(X; Y) = H(X) - H(X|Y)
        = H(X) + H(Y) - H(X, Y)
```

如果 X 和 Y 独立，互信息为零 —— 知道其中一个对另一个毫不透露。如果它们完全相关，互信息等于任一变量的熵。

在特征选择中，特征与目标之间的高互信息意味着该特征有用，低互信息意味着它是噪声。

### 条件熵

H(Y|X) 度量的是在观察到 X 之后，关于 Y 还剩下多少不确定性。

```
H(Y|X) = H(X,Y) - H(X)
```

两种极端情况：
- 如果 X 完全决定 Y，那么 H(Y|X) = 0。知道 X 就消除了关于 Y 的所有不确定性。例如：X = 摄氏温度，Y = 华氏温度。
- 如果 X 对 Y 毫不透露，那么 H(Y|X) = H(Y)。知道 X 完全没有减少你的不确定。例如：X = 抛硬币，Y = 明天的天气。

条件熵始终非负且不会超过 H(Y)：

```
0 <= H(Y|X) <= H(Y)
```

在机器学习中，条件熵出现在决策树里。每次分裂时，算法选择使 H(Y|X) 最小化的特征 X —— 即最能消除关于标签 Y 的不确定性的那个特征。

### 联合熵

H(X,Y) 是 X 和 Y 的联合分布的熵。

```
H(X,Y) = -sum sum p(x,y) * log(p(x,y))   对所有 x, y
```

核心性质：

```
H(X,Y) <= H(X) + H(Y)
```

等号在 X 和 Y 独立时成立。如果它们共享信息，联合熵小于各部分熵之和。"缺失"的那部分熵正好是互信息。

```mermaid
graph TD
    subgraph "信息论维恩图"
        direction LR
        HX["H(X)"]
        HY["H(Y)"]
        MI["I(X;Y)<br/>互信息"]
        HXgY["H(X|Y)<br/>= H(X) - I(X;Y)"]
        HYgX["H(Y|X)<br/>= H(Y) - I(X;Y)"]
        HXY["H(X,Y) = H(X) + H(Y) - I(X;Y)"]
    end

    HXgY --- MI
    MI --- HYgX
    HX -.- HXgY
    HX -.- MI
    HY -.- MI
    HY -.- HYgX
    HXY -.- HXgY
    HXY -.- MI
    HXY -.- HYgX
```

关系汇总：
- H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)
- I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
- H(X,Y) = H(X) + H(Y) - I(X;Y)

### 互信息（深入）

互信息 I(X;Y) 量化了知道一个变量能在多大程度上消除关于另一个变量的不确定性。

```
I(X;Y) = H(X) - H(X|Y)
       = H(Y) - H(Y|X)
       = H(X) + H(Y) - H(X,Y)
       = sum sum p(x,y) * log(p(x,y) / (p(x) * p(y)))
```

性质：
- I(X;Y) >= 0 恒成立。观察一个变量不会让你损失信息。
- I(X;Y) = 0 当且仅当 X 和 Y 独立。
- I(X;Y) = I(Y;X)。它是对称的，不像 KL 散度。
- I(X;X) = H(X)。一个变量与自己共享全部信息。

**互信息用于特征选择。** 在 ML 中，你需要对目标变量有信息量的特征。互信息给出了一套有理论依据的特征排序方法：

1. 对每个特征 X_i，计算 I(X_i; Y)，其中 Y 是目标变量。
2. 按互信息得分对特征排序。
3. 保留前 k 个特征。

无论特征与目标之间是什么关系 —— 线性的、非线性的、单调的或完全不单调的 —— 这个做法都有效。相关系数只能捕获线性关系，互信息能捕获一切。

| 方法 | 能检测 | 计算代价 | 支持分类变量？ |
|--------|---------|-------------------|---------------------|
| Pearson 相关系数 | 线性关系 | O(n) | 否 |
| Spearman 相关系数 | 单调关系 | O(n log n) | 否 |
| 互信息 | 任意统计依赖 | O(n log n)（分箱） | 是 |

### 标签平滑与交叉熵

标准的分类任务使用硬目标：[0, 0, 1, 0]。真实类别概率为 1，其他为 0。标签平滑将其替换为软目标：

```
soft_target = (1 - epsilon) * hard_target + epsilon / num_classes
```

以 epsilon = 0.1、4 个类别为例：
- 硬目标：[0, 0, 1, 0]
- 软目标：[0.025, 0.025, 0.925, 0.025]

从信息论的角度看，标签平滑增加了目标分布的熵。硬独热目标的熵为 0 —— 没有任何不确定性。软目标的熵为正。

为什么这有用：
- 防止模型把 logits 推到极端值（要完美匹配独热目标，在交叉熵下 logits 需要到无穷大）
- 起到正则化作用：模型不能 100% 确定
- 改善校准：预测概率更好地反映真实的不确定性
- 缩小训练和推理行为之间的差距

带标签平滑的交叉熵损失变为：

```
L = (1 - epsilon) * CE(hard_target, prediction) + epsilon * H_uniform(prediction)
```

第二项惩罚偏离均匀分布的预测 —— 这是对置信度的直接正则化。

### 为什么交叉熵是分类任务的标准损失

三个视角，同一个结论。

**信息论视角。** 交叉熵衡量用模型的分布代替真实分布时浪费了多少比特。最小化它让你的模型成为现实最高效的编码器。

**最大似然视角。** 对 N 个训练样本，真实类别为 y_i：

```
似然         = product( q(y_i) )
对数似然     = sum( log(q(y_i)) )
负对数似然   = -sum( log(q(y_i)) )
```

最后一行就是交叉熵损失。最小化交叉熵 = 在你的模型下最大化训练数据的似然。

**梯度视角。** 交叉熵对 logits 的梯度就是 (预测 - 真实)。干净、稳定、计算快。这是它和 softmax 配合得天衣无缝的原因。

### 比特 vs 奈特

唯一的区别是对数的底。

```
log 以 2 为底  → 比特    （信息论传统）
log 以 e 为底  → 奈特    （机器学习惯例）
log 以 10 为底 → 哈特利  （很少使用）
```

1 奈特 = 1/ln(2) 比特 = 1.4427 比特。PyTorch 和 TensorFlow 默认使用自然对数（奈特）。

### 困惑度

困惑度是交叉熵的指数。它告诉你模型实际在多少个等可能选项中犹豫不决。

```
困惑度 = 2^H(P,Q)   （若使用比特）
困惑度 = e^H(P,Q)   （若使用奈特）
```

一个困惑度为 50 的语言模型，平均来说和它必须在 50 个可能的 next token 中均匀选择一样困惑。越低越好。

GPT-2 在常见基准上达到约 30 的困惑度。现代模型在充分覆盖的领域里已经降到个位数。

## 动手实现

### 第 1 步：信息量与熵

```python
import math

def information_content(p, base=2):
    if p <= 0 or p > 1:
        return float('inf') if p <= 0 else 0.0
    return -math.log(p) / math.log(base)

def entropy(probs, base=2):
    return sum(
        p * information_content(p, base)
        for p in probs if p > 0
    )

fair_coin = [0.5, 0.5]
biased_coin = [0.99, 0.01]
fair_die = [1/6] * 6

print(f"Fair coin entropy:   {entropy(fair_coin):.4f} bits")
print(f"Biased coin entropy: {entropy(biased_coin):.4f} bits")
print(f"Fair die entropy:    {entropy(fair_die):.4f} bits")
```

### 第 2 步：交叉熵与 KL 散度

```python
def cross_entropy(p, q, base=2):
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')
            total += pi * (-math.log(qi) / math.log(base))
    return total

def kl_divergence(p, q, base=2):
    return cross_entropy(p, q, base) - entropy(p, base)

true_dist = [0.7, 0.2, 0.1]
good_model = [0.6, 0.25, 0.15]
bad_model = [0.1, 0.1, 0.8]

print(f"Entropy of true dist:     {entropy(true_dist):.4f} bits")
print(f"CE (good model):          {cross_entropy(true_dist, good_model):.4f} bits")
print(f"CE (bad model):           {cross_entropy(true_dist, bad_model):.4f} bits")
print(f"KL divergence (good):     {kl_divergence(true_dist, good_model):.4f} bits")
print(f"KL divergence (bad):      {kl_divergence(true_dist, bad_model):.4f} bits")
```

### 第 3 步：交叉熵作为分类损失

```python
def softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def cross_entropy_loss(true_class, logits):
    probs = softmax(logits)
    return -math.log(probs[true_class])

logits = [2.0, 1.0, 0.1]
true_class = 0

probs = softmax(logits)
loss = cross_entropy_loss(true_class, logits)

print(f"Logits:      {logits}")
print(f"Softmax:     {[f'{p:.4f}' for p in probs]}")
print(f"True class:  {true_class}")
print(f"Loss:        {loss:.4f} nats")
print(f"Perplexity:  {math.exp(loss):.2f}")
```

### 第 4 步：交叉熵等于负对数似然

```python
import random

random.seed(42)

n_samples = 1000
n_classes = 3
true_labels = [random.randint(0, n_classes - 1) for _ in range(n_samples)]
model_logits = [[random.gauss(0, 1) for _ in range(n_classes)] for _ in range(n_samples)]

ce_loss = sum(
    cross_entropy_loss(label, logits)
    for label, logits in zip(true_labels, model_logits)
) / n_samples

nll = -sum(
    math.log(softmax(logits)[label])
    for label, logits in zip(true_labels, model_logits)
) / n_samples

print(f"Cross-entropy loss:      {ce_loss:.6f}")
print(f"Negative log-likelihood: {nll:.6f}")
print(f"Difference:              {abs(ce_loss - nll):.2e}")
```

### 第 5 步：互信息

```python
def mutual_information(joint_probs, base=2):
    rows = len(joint_probs)
    cols = len(joint_probs[0])

    margin_x = [sum(joint_probs[i][j] for j in range(cols)) for i in range(rows)]
    margin_y = [sum(joint_probs[i][j] for i in range(rows)) for j in range(cols)]

    mi = 0.0
    for i in range(rows):
        for j in range(cols):
            pxy = joint_probs[i][j]
            if pxy > 0:
                mi += pxy * math.log(pxy / (margin_x[i] * margin_y[j])) / math.log(base)
    return mi

independent = [[0.25, 0.25], [0.25, 0.25]]
dependent = [[0.45, 0.05], [0.05, 0.45]]

print(f"MI (independent): {mutual_information(independent):.4f} bits")
print(f"MI (dependent):   {mutual_information(dependent):.4f} bits")
```

## 实际使用

同样的概念用 NumPy 来实现 —— 这才是实际工作中的写法：

```python
import numpy as np

def np_entropy(p):
    p = np.asarray(p, dtype=float)
    mask = p > 0
    result = np.zeros_like(p)
    result[mask] = p[mask] * np.log(p[mask])
    return -result.sum()

def np_cross_entropy(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    mask = p > 0
    return -(p[mask] * np.log(q[mask])).sum()

def np_kl_divergence(p, q):
    return np_cross_entropy(p, q) - np_entropy(p)

true = np.array([0.7, 0.2, 0.1])
pred = np.array([0.6, 0.25, 0.15])
print(f"Entropy:    {np_entropy(true):.4f} nats")
print(f"Cross-ent:  {np_cross_entropy(true, pred):.4f} nats")
print(f"KL div:     {np_kl_divergence(true, pred):.4f} nats")
```

你从零实现了 `torch.nn.CrossEntropyLoss()` 内部做的事。现在你知道训练中损失为什么下降了：模型预测的分布正在向真实分布靠近，度量单位是浪费了多少奈特的信息。

## 交付物

本课产出：
- `outputs/prompt-information-theory-tutor.md` —— 一个给 AI 助手的提示词，让它用信息论的直觉来解释损失函数

## 联系

本课的所有概念都与现代 AI 的具体部分相连接：

| 概念 | 出现在哪里 |
|---------|------------------|
| 熵 | 决策树分裂准则、强化学习中的策略熵正则化 |
| 交叉熵 | 每一个分类模型的损失函数、语言模型训练 |
| KL 散度 | VAE 的变分下界、知识蒸馏、RLHF 中的策略约束 |
| 互信息 | 特征选择、表示学习（InfoNCE 损失）、对比学习 |
| 困惑度 | 语言模型评估、机器翻译质量 |
| 标签平滑 | 图像分类、Transformer 训练、模型校准 |
| 条件熵 | 决策树、随机森林的分裂策略 |
| 负对数似然 | 最大似然估计、自回归模型训练 |

在 VAE 中，损失函数 = 重构误差 + KL 散度（后验对先验）。在知识蒸馏中，你用教师模型和学生模型预测之间的 KL 散度来训练学生。在 RLHF 中，你用一个 KL 惩罚项防止策略偏离参考模型太远。所有这些技术都建立在本课的几个公式之上。

## 练习

1. 计算均匀分布下英语字母表（26 个字母）的熵。再用实际字母频率估算一次。哪个更高？为什么？

2. 一个模型对某个样本输出 logits [5.0, 2.0, 0.5]，真实类别为 1。手算交叉熵损失，然后用你的 `cross_entropy_loss` 函数验证。什么样的 logits 会得到零损失？

3. 证明 KL 散度不对称。选两个分布 P 和 Q，计算 D_KL(P || Q) 和 D_KL(Q || P)。解释它们为什么不同。

4. 写一个函数，对一串 token 预测计算困惑度。给定一组 (true_token_index, predicted_logits) 对，返回序列的困惑度。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|----------------------|
| 信息量 | "惊讶度" | 编码一个事件所需的比特（或奈特）数：-log(p) |
| 熵 | "随机程度" | 一个分布所有结果的期望惊讶度。度量不可消除的不确定性。 |
| 交叉熵 | "那个损失函数" | 用模型分布 Q 来编码来自真实分布 P 的事件时的平均惊讶度。 |
| KL 散度 | "分布之间的距离" | 用 Q 代替 P 时多浪费的比特数。等于交叉熵减熵。不对称。 |
| 互信息 | "X 和 Y 的相关程度" | 知道 X 后关于 Y 的不确定性减少了多少。零意味着独立。 |
| Softmax | "把 logits 变成概率" | 先取指数再归一化。将任意实值向量映射为合法的概率分布。 |
| 困惑度 | "模型有多困惑" | 交叉熵的指数。模型每一步实际在多大的有效词汇表里做选择。 |
| 比特 | "香农的单位" | 以 2 为底的信息度量。1 比特消解一次公平硬币抛掷的不确定性。 |
| 奈特 | "机器学习的单位" | 以自然对数为底的信息度量。PyTorch 和 TensorFlow 默认使用。 |
| 负对数似然 | "NLL 损失" | 对独热标签而言，与交叉熵损失完全相同。最小化它就等于最大化正确预测的概率。 |

## 进一步阅读

- [Shannon 1948: A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) —— 原始论文，至今可读
- [Visual Information Theory (Chris Olah)](https://colah.github.io/posts/2015-09-Visual-Information/) —— 熵和 KL 散度最好的可视化解释
- [PyTorch CrossEntropyLoss 文档](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) —— 框架如何实现你刚刚从零写出来的东西
