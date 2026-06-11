# 朴素贝叶斯

> "朴素"的假设是错的，但它居然还能用。这就是它的美妙之处。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 2，第 01-07 课（分类、贝叶斯定理）
**时间：** 约 75 分钟

## 学习目标

- 从零实现带拉普拉斯平滑的多项式朴素贝叶斯，用于文本分类
- 解释为什么朴素的独立假设在数学上是错误的，但在实践中能产生正确的分类排名
- 比较多项式、伯努利和高斯三种朴素贝叶斯变体，并根据特征类型选择合适的变体
- 在高维稀疏数据上评估朴素贝叶斯与逻辑回归的表现，并解释其中发挥作用的偏置-方差权衡

## 问题

你需要对文本进行分类。邮件分为垃圾邮件或非垃圾邮件。客户评论分为正面或负面。工单分为不同类别。你的特征成千上万（每个词一个特征），但训练数据有限。

大多数分类器在这里都会卡住。逻辑回归需要足够的样本来可靠地估计成千上万个权重。决策树一次只看一个词进行分割，会严重过拟合。KNN 在 10,000 维的空间里毫无意义，因为每个点与其他所有点的距离都相等。

朴素贝叶斯能处理这个问题。它做了一个在数学上错误的假设（给定类别，每个特征都独立于其他所有特征），但它在文本分类上仍然胜过了"更智能"的模型，尤其是在训练集较小的情况下。它只需一次遍历数据就能完成训练。它可以扩展到数百万个特征。它能产生概率估计（虽然由于独立假设，概率通常校准得不好）。

理解为什么一个错误的假设能带来好的预测，教给你关于机器学习的一个基本原则：最好的模型不是最正确的模型，而是对你的数据有最好的偏置-方差权衡的模型。

## 概念

### 贝叶斯定理（快速回顾）

贝叶斯定理翻转条件概率：

```
P(class | features) = P(features | class) * P(class) / P(features)
```

我们想要 `P(class | features)` —— 给定文档中的词，该文档属于某个类别的概率。我们可以从以下内容计算：
- `P(features | class)` —— 在该类别的文档中看到这些词的似然
- `P(class)` —— 该类别的先验概率（垃圾邮件一般有多常见？）
- `P(features)` —— 证据，对所有类别都一样，所以在比较时可以忽略它

拥有最高 `P(class | features)` 的类别获胜。

### 朴素独立假设

精确计算 `P(features | class)` 需要估计所有特征在一起的联合概率。如果词汇表有 10,000 个词，你需要估计2^10,000 种可能组合的分布。不可能。

朴素假设：给定类别，每个特征条件独立。

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

不再需要一个不可能的联合分布，而是估计 n 个简单的逐特征分布。每个分布只需要一个计数。

这个假设显然是错误的。在任何文档中，"machine"和"learning"这两个词都不是独立的。但分类器不需要正确的概率估计。它需要正确的排名 —— 哪个类别有最高概率。独立假设引入了系统性误差，但这些误差对所有类别的影响大致相同，所以排名仍然正确。

### 为什么它仍然有效

三个原因：

1. **排名优于校准。** 分类只需要排名最高的类别正确。即使当真实概率是 0.7 时，P(spam) = 0.99999，分类器仍然正确地选择了垃圾邮件。我们不需要正确的概率。我们需要正确的胜者。

2. **高偏置，低方差。** 独立假设是一个强先验。它严重约束了模型，从而防止过拟合。在有限的训练数据下，一个稍微错误但稳定的模型，胜过理论上正确但极度不稳定的模型。这就是偏置-方差权衡在起作用。

3. **特征冗余相互抵消。** 相关特征提供冗余的证据。分类器对这些证据重复计算，但它对正确的类别也重复计算了。如果"machine"和"learning"总是同时出现，两者都为"技术"类别提供证据。NB 计算了两次，但它是替正确的类别计算了两次。

第四个实际原因：朴素贝叶斯极快。训练只需一次遍历数据，统计频率。预测是矩阵乘法。你可以在几秒钟内训练一百万个文档。这种速度意味着你可以更快地迭代、尝试更多的特征集、比使用较慢的模型运行更多的实验。

### 数学推导步骤

让我们通过一个具体的例子来跟踪。假设我们有两个类别：垃圾邮件和非垃圾邮件。我们的词汇表有三个词："free"、"money"、"meeting"。

训练数据：
- 垃圾邮件中"free"出现 80 次，"money"出现 60 次，"meeting"出现 10 次（共 150 个词）
- 非垃圾邮件中"free"出现 5 次，"money"出现 10 次，"meeting"出现 100 次（共 115 个词）
- 40% 的邮件是垃圾邮件，60% 是非垃圾邮件

带拉普拉斯平滑（alpha=1）：

```
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

新邮件包含："free"（2 次）、"money"（1 次）、"meeting"（0 次）。

```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

垃圾邮件以较大优势获胜。"free"出现两次是垃圾邮件的强证据。注意"meeting"不出现对两个 log 总和都贡献零（0 * log(P)）—— 在多项式 NB 中，不存在的词没有影响。伯努利 NB 才是明确对词的不存在建模的。

### 三种变体

朴素贝叶斯有三种变体。每种对 `P(feature | class)` 的建模方式不同。

#### 多项式朴素贝叶斯

将每个特征建模为计数。最适合特征是词频或 TF-IDF 值的文本数据。

```
P(word_i | class) = (count of word_i in class + alpha) / (total words in class + alpha * vocab_size)
```

`alpha` 是拉普拉斯平滑（见下文）。这个变体是文本分类的主力。

#### 高斯朴素贝叶斯

将每个特征建模为正态分布。最适合连续特征。

```
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

每个类别在每个特征上都有自己的均值和方差。当特征在每个类别内确实服从钟形曲线时，这个方法效果很好。

#### 伯努利朴素贝叶斯

将每个特征建模为二元（存在或不存在）。最适合短文本或二元特征向量。

```
P(word_i | class) = (docs in class containing word_i + alpha) / (total docs in class + 2 * alpha)
```

与多项式不同，伯努利明确惩罚词的不存在。如果"free"通常出现在垃圾邮件中，但在这封邮件中不存在，伯努利将其视为反对垃圾邮件的证据。

### 何时使用哪种变体

| 变体 | 特征类型 | 最适合 | 示例 |
|---------|-------------|----------|---------|
| 多项式 | 计数或频率 | 文本分类、词袋 | 邮件垃圾分类、主题分类 |
| 高斯 | 连续值 | 具有近似正态特征的表格数据 | 鸢尾花分类、传感器数据 |
| 伯努利 | 二元（0/1） | 短文本、二元特征向量 | SMS 垃圾分类、存在/不存在特征 |

### 拉普拉斯平滑

当测试数据中出现某个词，但该词在特定类别的训练数据中从未出现时，会发生什么？

没有平滑：`P(word | class) = 0/N = 0`。一个零乘以整个产品，使得 `P(class | features) = 0`，不管所有其他证据如何。一个看不见的词摧毁了整个预测，不管有多少其他证据支持它。

拉普拉斯平滑给每个特征计数添加一个小的计数 `alpha`（通常为 1）：

```
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

当 alpha=1 时，每个词至少获得一个极小的概率。"discombobulate"出现在测试邮件中不再杀死垃圾邮件概率。这个平滑有一个贝叶斯解释：它等价于对词分布放置一个均匀的狄利克雷先验。

更高的 alpha 意味着更强的平滑（更均匀的分布）。更低的 alpha 意味着模型更信任数据。Alpha 是一个你需要调优的超参数。

alpha 的效果：

| Alpha |效果 | 何时使用 |
|-------|--------|-------------|
| 0.001 | 几乎没有平滑，信任数据 | 非常大的训练集，不期望有看不见的特征 |
| 0.1 | 轻度平滑 | 大训练集 |
| 1.0 | 标准拉普拉斯平滑 | 默认起点 |
| 10.0 | 重平滑，平坦化分布 | 非常小的训练集，期望有很多看不见的特征 |

### 对数空间计算

乘以数百个概率（每个都小于 1）会导致浮点下溢。即使真实值是一个非常小的正数，乘积在浮点中也会变成零。

解决方案：在对数空间工作。不乘概率，而是加它们的对数：

```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

这将预测转化为点积：

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

矩阵乘法。这就是朴素贝叶斯预测如此快速的原因 —— 它与单层线性模型的操作相同。

### 朴素贝叶斯与逻辑回归

两者都是用于文本的线性分类器。区别在于它们建模的内容。

| 方面 | 朴素贝叶斯 | 逻辑回归 |
|--------|------------|-------------------|
| 类型 | 生成式（建模 P(X|Y)） | 判别式（建模 P(Y|X)） |
| 训练 | 统计频率 | 优化损失函数 |
| 小数据 | 更好（强先验有帮助） | 更差（没有足够的样本来估计权重） |
| 大数据 | 更差（错误假设有害） | 更好（灵活的决策边界） |
| 特征 | 假设独立 | 处理相关性 |
| 速度 | 一次遍历，非常快 | 迭代优化 |
| 校准 | 概率估计差 | 概率估计更好 |

经验法则：从朴素贝叶斯开始。如果你有足够的数据且 NB 达到瓶颈，切换到逻辑回归。

### 分类流程

```mermaid
flowchart LR
    A[原始文本] --> B[分词]
    B --> C[构建词汇表]
    C --> D[统计词频]
    D --> E[应用平滑]
    E --> F[计算对数概率]
    F --> G[预测：argmax P(class | words)]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

在实践中，我们在对数空间工作以避免浮点下溢。不是乘以许多小概率，而是加它们的对数：

```
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

## 动手实现

`code/naive_bayes.py` 中的代码从头实现了多项式朴素贝叶斯和高斯朴素贝叶斯。

### 多项式朴素贝叶斯

从头实现的步骤：

1. **fit(X, y)**：对每个类别，统计每个特征的频率。添加拉普拉斯平滑。计算对数概率。存储类别先验（类别频率的对数）。

2. **predict_log_proba(X)**：对每个样本，计算 log P(class) + 所有类别的 log P(feature_i | class) 之和。这是一个矩阵乘法：X @ log_probs.T + log_priors。

3. **predict(X)**：返回对数概率最高的类别。

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

关键洞察：拟合后，预测只是矩阵乘法加偏置。这就是朴素贝叶斯如此快速的原因。

### 高斯朴素贝叶斯

对于连续特征，我们估计每个类别每个特征的均值和方差：

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

预测使用每个特征的高斯 PDF，在特征间相乘（在對數空間相加）。

### 示例：文本分类

代码生成模拟两类（技术文章 vs 体育文章）的合成词袋数据。每个类别有不同的词频分布。多项式朴素贝叶斯使用词频进行分类。

合成数据的工作方式：我们创建 200 个"词"（特征列）。词 0-39 在技术文章中高频，在体育中低频。词 80-119 在体育中高频，在技术中低频。词 40-79 在两者中都中等频率。这创造了一个真实的场景：一些词是强类别指示器，另一些是噪声。

### 示例：连续特征

代码生成类似鸢尾花的数据（3 类，4特征，高斯簇）。高斯朴素贝叶斯使用每类的均值和方差进行分类。每个类别有不同的中心（均值向量）和不同的散布（方差），模仿真实世界数据中测量值在不同类别间系统性地不同。

代码还演示了：
- **平滑比较：** 用不同的 alpha 值训练多项式朴素贝叶斯，展示平滑强度对准确率的影响。
- **训练集大小实验：** 随着训练数据从 20 增长到 1600 样本，NB 准确率如何提升。NB 即使在非常少的样本下也能达到不错的准确率 —— 这是它的主要优势。
- **混淆矩阵：** 每类精确率、召回率和 F1 分数，展示 NB 在哪里犯错。

### 预测速度

朴素贝叶斯预测是一个矩阵乘法。对于 n 个样本、d 个特征和 k 个类别：
- 多项式朴素贝叶斯：一次矩阵乘法 (n x d) @ (d x k) = O(n * d * k)
- 高斯朴素贝叶斯：n * k 个高斯 PDF 计算，每个覆盖 d 个特征 = O(n * d * k)

两者在每个维度上都是线性的。将这与 KNN（需要计算到所有训练点的距离）或带 RBF 核的 SVM（需要对所有支持向量进行核评估）相比。NB 在预测时快几个数量级。

## 实际使用

使用 sklearn，两个变体都是一行代码：

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB准确率: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB 准确率: {mnb.score(X_test_counts, y_test):.3f}")
```

使用 sklearn 进行文本分类：

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

`naive_bayes.py` 中的代码将从头实现与 sklearn 在相同数据上进行比较，以验证正确性。

### TF-IDF 与朴素贝叶斯

原始词频给每次出现的所有词相同的权重。但像"the"和"is"这样的常见词在每个类别中都频繁出现 —— 它们不携带信息。TF-IDF（词频-逆文档频率）降低常见词的权重，提高稀有、有区分度的词的权重。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

TF-IDF 值是非负的，所以它们可以与多项式朴素贝叶斯一起使用。TF-IDF + 多项式朴素贝叶斯的组合是文本分类最强的基线之一。它经常在少于 10,000 个训练样本的数据集上击败更复杂的模型。

### 用于短文本的伯努利朴素贝叶斯

对于短文本（推文、短信、聊天消息），伯努利朴素贝叶斯可能优于多项式朴素贝叶斯。短文本词数很少，所以多项式朴素贝叶斯依赖的频率信息有噪声。伯努利朴素贝叶斯只关心存在或不存在，这在短文本上更可靠。

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

CountVectorizer 中的 `binary=True` 标志将所有计数转换为 0/1。没有它，伯努利朴素贝叶斯仍然可以工作，但看到的是它并非设计用来处理的计数。

### 校准 NB 概率

NB概率校准很差。当 NB 说 P(spam) = 0.95 时，真实概率可能是 0.7。如果你需要可靠的概率估计（例如，设置阈值或与其他模型组合），使用 sklearn 的 CalibratedClassifierCV：

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

这使用交叉验证在 NB 的原始分数上拟合逻辑回归。结果概率更接近真实类别频率。

### 常见陷阱

1. **负特征值。** 多项式朴素贝叶斯需要非负特征。如果你有负值（如某些设置下的 TF-IDF 或标准化特征），改用高斯朴素贝叶斯，或者将特征移为正数。

2. **零方差特征。** 高斯朴素贝叶斯除以方差。如果某个特征在某个类别上具有零方差（所有值相同），概率计算会出问题。代码添加了一个小的平滑项（1e-9）到所有方差以防止这个问题。

3. **类别不平衡。** 如果 99% 的邮件是非垃圾邮件，先验 P(not-spam) = 0.99 非常强，以至于压倒了似然证据。你可以使用 class_prior 参数手动设置类别先验。

4. **特征缩放。** 多项式朴素贝叶斯不需要缩放（它处理计数）。高斯朴素贝叶斯也不需要缩放（它估计逐特征统计）。这相对于逻辑回归和支持向量机是一个优势，它们对特征尺度敏感。

## 交付物

本课产出：
- `outputs/skill-naive-bayes-chooser.md` —— 一个用于选择正确 NB 变体的决策技能
- `code/naive_bayes.py` —— 从零实现的多项式朴素贝叶斯和高斯朴素贝叶斯，带 sklearn 比较

### 朴素贝叶斯何时失效

当独立假设导致错误的排名（不仅仅是错误的概率）时，NB 会失效。这发生在：

1. **强特征交互。** 如果类别取决于两个特征的组合而非单独任一个（类似 XOR 的模式），NB 会完全错过它。单独每个特征都不提供证据，NB 无法非线性地组合它们。

2. **高度相关但证据相反的特征。** 如果特征 A 说"垃圾邮件"，特征 B 说"非垃圾邮件"，但 A 和 B 完全相关（它们在现实中总是同意），NB 会在本来没有冲突的地方看到冲突的证据。

3. **非常大的训练集。** 有足够的数据后，像逻辑回归这样的判别模型学习真实的决策边界，胜过 NB。在小数据上有帮助的独立假设现在拖累了模型。

在实践中，这些失效模式在文本分类中很少见。文本特征众多、单独弱，且独立假设的误差往往会相互抵消。对于具有少量强相关特征的表格数据，先考虑逻辑回归或基于树的模型。

## 练习

1. **平滑实验。** 用 alpha值为 0.01、0.1、1.0、10.0 和 100.0 的多项式朴素贝叶斯训练文本数据。绘制准确率 vs alpha曲线。性能峰值在哪里？为什么非常高的 alpha 会损害性能？

2. **特征独立性测试。** 取一个真实文本数据集。选择两个明显相关的词（"machine"和"learning"）。计算 P(word1 | class) * P(word2 | class) 并与 P(word1 AND word2 | class) 比较。独立假设有多错误？它会影响分类准确率吗？

3. **伯努利实现。** 用伯努利朴素贝叶斯类扩展代码。将词袋转换为二元（存在/不存在），并在文本数据上与多项式朴素贝叶斯比较准确率。伯努利何时胜出？

4. **NB vs 逻辑回归。** 在文本数据上训练两者。从 100 个训练样本开始，增加到 10,000。绘制两个的准确率 vs 训练集大小曲线。逻辑回归在什么时候超过朴素贝叶斯？

5. **垃圾邮件过滤器。** 构建一个完整的垃圾邮件分类器：对原始邮件文本进行分词，构建词汇表，创建词袋特征，训练多项式朴素贝叶斯，用精确率和召回率评估（不仅仅是准确率 —— 为什么？）。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|----------------------|
| 朴素贝叶斯 | "简单的概率分类器" | 应用贝叶斯定理并假设给定类别时特征条件独立的分类器 |
| 条件独立 | "特征互不影响" | P(A, B | C) = P(A | C) * P(B | C) —— 一旦知道 C，知道 B 不会关于 A 告诉你任何新的东西 |
| 拉普拉斯平滑 | "加一平滑" | 给每个特征添加一个小的计数，以防止零概率主导预测 |
| 先验 | "看到数据前你相信的" | P(class) —— 观察任何特征前每个类别的概率 |
| 似然 | "数据拟合得有多好" | P(features | class) —— 如果已知类别，观察到这些特征的概率 |
| 后验 | "看到数据后你相信的" | P(class | features) —— 观察到特征后类别的更新概率 |
| 生成式模型 | "对数据如何生成建模" | 学习 P(X | Y) 和 P(Y)，然后使用贝叶斯定理得到 P(Y | X) 的模型 |
| 判别式模型 | "对决策边界建模" | 直接学习 P(Y | X) 而不建模 X 是如何生成的模型 |
| 对数概率 | "避免下溢" | 使用 log P 而不是 P，以防止许多小数的乘积在浮点中变成零 |

## 延伸阅读

- [scikit-learn 朴素贝叶斯文档](https://scikit-learn.org/stable/modules/naive_bayes.html) —— 所有三种变体及数学细节
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) —— 多项式与伯努利文本分类的经典比较
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) —— 文本 NB 的改进
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) —— 证明 NB 用更少的数据比 LR 收敛更快