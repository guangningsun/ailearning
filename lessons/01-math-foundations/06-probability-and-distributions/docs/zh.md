# 概率与分布

> 概率是 AI 表达不确定性的语言。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 1，第 01-04 课
**预计时间：** 约 75 分钟

## 学习目标

- 从零实现伯努利、类别、泊松、均匀和正态分布的 PMF 与 PDF
- 计算期望值与方差，并用中心极限定理解释为什么高斯分布无处不在
- 构建 softmax 与 log-softmax 函数，掌握数值稳定性技巧（减去最大 logit）
- 从 logits 计算交叉熵损失，并把它与负对数似然联系起来

## 问题

一个分类器输出 `[0.03, 0.91, 0.06]`。一个语言模型从 50,000 个候选中选出下一个词。一个扩散模型通过学习到的分布来生成图像。这背后全是概率。

模型做出的每一个预测都是一个概率分布。每一个损失函数都在衡量预测分布与真实分布之间的距离。每一次训练步都在调整参数，让一个分布更像另一个分布。不懂概率，你就读不懂任何一篇 ML 论文，调试不了任何一个模型，也无法理解为什么训练 loss 突然变成 NaN。

## 概念

### 事件、样本空间与概率

样本空间 S 是所有可能结果的集合。事件是样本空间的子集。概率把事件映射到 0 到 1 之间的数字。

```
掷硬币：
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

掷一枚骰子：
  S = {1, 2, 3, 4, 5, 6}
  P(偶数) = P({2, 4, 6}) = 3/6 = 0.5
```

三条公理定义了整个概率论：
1. 对任意事件 A，P(A) >= 0
2. P(S) = 1（总会有某个结果发生）
3. 当 A 和 B 不能同时发生时，P(A 或 B) = P(A) + P(B)

剩下的一切（贝叶斯定理、期望、分布）都从这三条规则推导出来。

### 条件概率与独立性

P(A|B) 是在已知 B 发生的条件下，A 发生的概率。

```
P(A|B) = P(A 且 B) / P(B)

例子：一副扑克牌
  P(老K | 人头牌) = P(老K 且 人头牌) / P(人头牌)
                  = (4/52) / (12/52)
                  = 4/12 = 1/3
```

两个事件独立，意味着知道一个事件不会给你关于另一个事件的任何信息：

```
独立：  P(A|B) = P(A)
等价于：P(A 且 B) = P(A) * P(B)
```

硬币的每次投掷是独立的。不放回地抽牌则不是。

### 概率质量函数 vs 概率密度函数

离散随机变量有概率质量函数（PMF）。每个结果都有一个确定概率，你可以直接读出。

```
PMF：P(X = k)

公平骰子：
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  所有概率之和 = 1
```

连续随机变量有概率密度函数（PDF）。单个点上的密度并不是概率。概率来自对密度在区间上的积分。

```
PDF：f(x)

P(a <= X <= b) = 从 a 到 b 对 f(x) 积分

f(x) 可以大于 1（这是密度，不是概率）
从 -∞ 到 +∞ 对 f(x) 积分 = 1
```

这个区别在 ML 里很重要。分类输出是 PMF（离散选择），VAE 的潜在空间用的是 PDF（连续）。

### 常见分布

**伯努利分布：** 一次试验，两个结果。用来建模二分类。

```
P(X = 1) = p
P(X = 0) = 1 - p
均值 = p,  方差 = p(1-p)
```

**类别分布：** 一次试验，k 个结果。用来建模多分类（softmax 的输出）。

```
P(X = i) = p_i,  其中所有 p_i 之和 = 1
例子：P(猫) = 0.7,  P(狗) = 0.2,  P(鸟) = 0.1
```

**均匀分布：** 所有结果等可能。用于随机初始化。

```
离散：P(X = k) = 1/n，其中 k ∈ {1, ..., n}
连续：f(x) = 1/(b-a)，其中 x ∈ [a, b]
```

**正态分布（高斯分布）：** 钟形曲线。由均值 (μ) 和方差 (σ²) 参数化。

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

标准正态：μ = 0, σ = 1
  68% 的数据落在 1σ 以内
  95% 落在 2σ 以内
  99.7% 落在 3σ 以内
```

**泊松分布：** 固定区间内稀有事件的计数。用于建模事件发生率。

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
均值 = lambda,  方差 = lambda
```

### 期望值与方差

期望值是结果的加权平均。

```
离散：  E[X] = Σ x_i * P(X = x_i)
连续：  E[X] = 对 x * f(x) dx 的积分
```

方差衡量结果围绕均值的分散程度。

```
Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²
标准差 = sqrt(Var(X))
```

在 ML 中，期望值以损失函数的形式出现（在数据分布上的平均损失）。方差告诉你模型的稳定性。梯度方差大意味着训练过程嘈杂。

### 联合分布与边缘分布

联合分布 P(X, Y) 同时描述两个随机变量。

联合 PMF 示例（X = 天气，Y = 带伞）：

| | Y=0 (没带伞) | Y=1 (带了伞) | 边缘 P(X) |
|---|---|---|---|
| X=0 (晴天) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1 (下雨) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **边缘 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

边缘分布通过对另一个变量求和得到：

```
P(X = x) = Σ_y P(X = x, Y = y)
```

上表中每一行和每一列的合计值就是边缘概率。

### 为什么正态分布随处可见

中心极限定理：大量独立随机变量的和（或均值）会趋近于正态分布，无论原始分布是什么形状。

```
掷 1 枚骰子：均匀分布（平坦）
掷 2 枚骰子取均值：三角分布（出现峰值）
掷 30 枚骰子取均值：近乎完美的钟形曲线

这对任何初始分布都成立。
```

这就是为什么：
- 测量误差近似正态（许多小的独立误差源的叠加）
- 神经网络的权重初始化使用正态分布
- SGD 中的梯度噪声近似正态（大量样本梯度的和）
- 正态分布是在给定均值和方差下熵最大的分布

### 对数概率

原始概率会引发数值问题。将许多小概率相乘，很快就会下溢到零。

```
P(句子) = P(词1) * P(词2) * ... * P(词_n)
        = 0.01 * 0.003 * 0.02 * ...
        -> 0.0（约 30 项后下溢）
```

对数概率解决了这个问题。乘法变成了加法。

```
log P(句子) = log P(词1) + log P(词2) + ... + log P(词_n)
            = -4.6 + -5.8 + -3.9 + ...
            -> 有限数字（不会下溢）
```

规则：
- log(a * b) = log(a) + log(b)
- 对数概率总是 ≤ 0（因为 0 < P ≤ 1）
- 越负 = 越不可能
- 交叉熵损失就是正确类别的负对数概率

### Softmax 作为概率分布

神经网络输出原始分数（logits）。Softmax 将它们转换成合法的概率分布。

```
softmax(z_i) = exp(z_i) / Σ_j exp(z_j)

性质：
  - 所有输出都在 (0, 1) 之间
  - 所有输出之和为 1
  - 保持输入的相对顺序
  - exp() 放大了 logits 之间的差异
```

Softmax 技巧：在取指数前减去最大的 logit，防止溢出。

```
z = [100, 101, 102]
exp(102) = 溢出

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  （安全）

结果相同，不会溢出。
```

Log-softmax 将 softmax 和对数结合，进一步提升数值稳定性。PyTorch 在交叉熵损失内部就是这样做的。

### 采样

采样就是从分布中抽取随机值。在 ML 中：
- Dropout 随机采样要归零的神经元
- 数据增强采样随机的变换
- 语言模型从预测分布中采样下一个 token
- 扩散模型采样噪声，然后逐步去噪

从任意分布中采样需要用到逆变换采样、拒绝采样或重参数化技巧（用于 VAE）等技术。

## 动手实现

### 第 1 步：概率基础

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### 第 2 步：从零实现 PMF 和 PDF

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### 第 3 步：期望值与方差

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### 第 4 步：从分布中采样

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### 第 5 步：Softmax 与对数概率

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### 第 6 步：中心极限定理演示

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 第 7 步：可视化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

完整实现及所有可视化见 `code/probability.py`。

## 实际使用

用 NumPy 和 SciPy，以上所有内容都是一行搞定：

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

你刚刚从零实现了这一切。现在你知道库调用背后到底在做什么。

## 交付物

本课产出：
- `code/probability.py` —— 完整实现了 PMF/PDF、采样器、softmax 与交叉熵损失，从零构建
- `outputs/prompt-probability-tutor.md` —— 一个提示词，让 AI 助手从 ML 实践的角度讲解概率分布

## 联系

概率分布是整个现代 AI 的数学骨架。本节学到的每一个概念都直接对应到具体的 ML 实践中：

| 概念 | 出现在哪里 |
|---------|------------------|
| 伯努利分布 | 二分类的 BCE 损失、Dropout 神经元开关 |
| 类别分布 | 多分类的 softmax 输出、语言模型的下一个 token 预测 |
| 正态分布 | 权重初始化（Kaiming/Xavier）、SGD 梯度噪声建模、VAE 先验 |
| 泊松分布 | 事件计数建模、推荐系统中的用户行为频率 |
| 均匀分布 | 随机初始化、数据增强中的随机参数采样 |
| 期望值 | 损失函数（数据分布上的平均损失）、策略梯度中的回报估计 |
| 方差 | 模型诊断（高方差 = 过拟合倾向）、BatchNorm 的归一化对象 |
| 中心极限定理 | 解释为什么梯度噪声是正态的，为什么大批量训练更稳定 |
| 对数概率 | 语言建模中的对数困惑度、强化学习中的对数策略梯度 |
| Softmax | 注意力权重归一化、分类头、策略网络的动作分布 |
| 交叉熵 | 分类训练的核心损失函数，等价于最小化 KL 散度 |
| 联合分布与边缘分布 | 贝叶斯推断、变分推断中的 ELBO 推导 |
| 采样 | 扩散模型的去噪过程、语言模型的生成解码、VAE 的重参数化技巧 |

贝叶斯推断值得展开讲一下。在给定观测数据 D 的情况下，参数 θ 的后验分布是：

```
P(θ|D) = P(D|θ) * P(θ) / P(D)
```

这个公式是所有贝叶斯深度学习的根基。变分自编码器（VAE）通过优化 ELBO（证据下界）来学习潜在分布 —— 本质上是用一个可学习的分布 q(z|x) 去逼近真实的后验 p(z|x)。扩散模型在正向过程中逐步加噪（把数据分布变成标准高斯），在反向过程中学习去噪（从高斯分布中采样并恢复数据）。强化学习的策略梯度方法直接对动作分布进行优化 —— 策略网络输出一个类别分布，采样出一个动作，然后根据奖励信号来调整这个分布。

你用几行代码写出的 softmax，正是这些模型每天都在调用的基础运算。

## 练习

1. 为指数分布实现逆变换采样。通过采样 10,000 个值并对比直方图与真实 PDF 来验证。

2. 为两颗灌铅骰子构建一个联合分布表。计算边缘分布，并检查这两颗骰子是否独立。

3. 计算一个 5 类分类器的交叉熵损失，其 logits 为 `[2.0, 0.5, -1.0, 3.0, 0.1]`，正确类别是索引 3。然后用 PyTorch 的 `nn.CrossEntropyLoss` 验证你的结果。

4. 写一个函数，接收对数概率列表，返回最可能的序列、总对数概率和等效的原始概率。用一个由 50 个词组成的句子来测试，每个词的概率都是 0.01。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|----------------------|
| 样本空间 | "所有可能性" | 实验中所有可能结果的集合 S |
| PMF | "概率函数" | 给出每个离散结果的精确概率的函数，所有概率之和为 1 |
| PDF | "概率曲线" | 连续变量的密度函数。对它在一段区间上积分才能得到概率 |
| 条件概率 | "已知某条件下的概率" | P(A\|B) = P(A 且 B) / P(B)。贝叶斯思维与贝叶斯定理的基石 |
| 独立性 | "它们互不影响" | P(A 且 B) = P(A) * P(B)。知道一个事件不会提供关于另一个事件的任何信息 |
| 期望值 | "平均值" | 所有结果的概率加权和。损失函数就是一个期望值 |
| 方差 | "有多分散" | 与均值的期望平方偏差。高方差 = 嘈杂、不稳定的估计 |
| 正态分布 | "钟形曲线" | f(x) = (1/sqrt(2*pi*sigma²)) * exp(-(x-μ)²/(2σ²))。由中心极限定理导致无处不在 |
| 中心极限定理 | "均值会变成正态的" | 大量独立样本的均值会趋近于正态分布，无论原始分布如何 |
| 联合分布 | "两个变量放在一起" | P(X, Y) 描述了 X 和 Y 每一种组合的概率 |
| 边缘分布 | "把另一个变量求和掉" | P(X) = Σ_y P(X, Y)。从联合分布中恢复单个变量的分布 |
| 对数概率 | "概率取对数" | log P(x)。把乘法变成加法，避免长序列中的数值下溢 |
| Softmax | "把分数变成概率" | softmax(z_i) = exp(z_i) / Σ_j exp(z_j)。将实值 logits 映射为合法概率分布 |
| 交叉熵 | "那个损失函数" | -Σ(p_true * log(p_predicted))。衡量两个分布之间的差异。越低越好 |
| Logits | "模型的原始输出" | softmax 之前的未归一化分数。得名自 logistic 函数 |
| 采样 | "抽取随机值" | 按概率分布生成数值。模型就是这样生成输出的 |

## 进一步阅读

- [3Blue1Brown：中心极限定理到底是什么？](https://www.youtube.com/watch?v=zeJD6dqJ5lo) —— 均值为什么会变成正态的视觉证明
- [斯坦福 CS229 概率复习](https://cs229.stanford.edu/section/cs229-prob.pdf) —— 涵盖以上所有内容的简明参考（及更多）
- [Log-Sum-Exp 技巧](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) —— 数值稳定性为何重要，以及如何实现
