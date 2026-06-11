# 处理不平衡数据

> 当99% 的数据是"正常"时，准确率是一个谎言。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 2，第 01-09 课（尤其是评估指标）
**时间：** 约 90 分钟

## 学习目标

- 从零实现 SMOTE，并解释合成过采样与随机复制有何不同
- 使用 F1、AUPRC 和 Matthews 相关系数而非准确率来评估不平衡分类器
- 比较类别加权、阈值调整和重采样策略，并为给定的失衡比率选择正确的方法
- 构建结合 SMOTE、类别权重和阈值优化的完整不平衡数据流程

## 问题

你构建了一个欺诈检测模型。它达到了 99.9% 的准确率。你庆祝。然后你意识到它对每一笔交易都预测"非欺诈"。

这不是 bug。当只有 0.1% 的交易是欺诈时，这是理性的做法。模型学会了一直猜测多数类以最小化总体误差。它在技术上是正确的，但完全无用。

这种情况在所有真实分类重要的领域都会发生。疾病诊断：1% 阳性率。网络入侵：0.01% 攻击。制造缺陷：0.5% 缺陷。垃圾邮件过滤：20% 垃圾邮件。流失预测：5% 流失者。少数类越重要，它往往就越罕见。

准确率失败，因为它对所有正确预测一视同仁。正确标记一笔合法交易和正确捕获欺诈都算作一个准确率点。但捕获欺诈是模型存在的全部原因。我们需要指标、技术和训练策略，迫使模型关注稀有但重要的类。

|  | 预测为正 | 预测为负 |
|--|---|---|
| 实际为正 | 0 (TP) | 10 (FN) |
| 实际为负 | 0 (FP) | 990 (TN) |

准确率 = (0 + 990) / 1000 = 99.0%

模型捕获了零笔欺诈。零疾病。零缺陷。但准确率说是 99%。这就是为什么准确率对不平衡问题是危险的。

### 更好的指标

**精确率** = TP / (TP + FP)。在所有被标记为正的里面，有多少实际上是正的？高精确率意味着少的误报。

**召回率** = TP / (TP + FN)。在所有实际为正的中，我们捕获了多少？高召回率意味着少的遗漏正类。

**F1 分数** = 2 * precision * recall / (precision + recall)。调和均值。它比算术均值更能惩罚精确率和召回率之间的极端不平衡。

**F-beta 分数** = (1 + beta^2) * precision * recall / (beta^2 * precision + recall)。当 beta > 1 时，召回率更重要。当 beta < 1 时，精确率更重要。F2 在欺诈检测中很常见（遗漏欺诈比误报更糟糕）。

**AUPRC**（精确率-召回率曲线下面积）。类似于 AUC-ROC，但对不平衡数据更有信息量。随机分类器的 AUPRC 等于正类比率（不像 ROC 那样是 0.5）。这使得改进更容易看到。

**Matthews 相关系数** = (TP * TN - FP * FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))。范围从 -1 到 +1。只有当模型在两个类上都表现良好时才给出高分数。即使类的大小非常不同也能保持平衡。

对于上述"始终预测负类"的模型：精确率 = 0/0（未定义，通常设为 0），召回率 = 0/10 = 0，F1 = 0，MCC = 0。这些指标正确地将模型识别为无用的。

### 不平衡数据流程

```mermaid
flowchart TD
    A[不平衡数据集] --> B{失衡比率?}
    B -->|轻度：80/20| C[类别权重]
    B -->|中度：95/5| D[SMOTE + 阈值调整]
    B -->|严重：99/1| E[SMOTE + 类别权重 + 阈值]
    C --> F[训练模型]
    D --> F
    E --> F
    F --> G[用 F1 / AUPRC / MCC 评估]
    G --> H{足够好?}
    H -->|否| I[尝试不同策略]
    H -->|是| J[部署并监控]
    I --> B
```

### SMOTE：合成少数类过采样技术

随机过采样复制现有的少数类样本。这有效，但有过度拟合的风险，因为模型反复看到相同的点。

SMOTE 创建新的合成少数类样本，它们是可信的但不是复制品。算法：

1. 对于每个少数类样本 x，在其他少数类样本中找到其 k 个最近邻居
2. 随机选择一个邻居
3. 在 x 和该邻居之间的线段上创建一个新样本

公式：`new_sample = x + random(0, 1) * (neighbor - x)`

这在真实少数类点之间进行插值，在特征空间的相同区域创建样本，而不仅仅是复制现有数据。

```mermaid
flowchart LR
    subgraph Original["原始少数类点"]
        P1["x1 (1.0, 2.0)"]
        P2["x2 (1.5, 2.5)"]
        P3["x3 (2.0, 1.5)"]
    end
    subgraph SMOTE["SMOTE 生成"]
        direction TB
        S1["选择 x1，邻居 x2"]
        S2["随机 t = 0.4"]
        S3["new = x1 + 0.4*(x2-x1)"]
        S4["new = (1.2, 2.2)"]
        S1 --> S2 --> S3 --> S4
    end
    Original --> SMOTE
    subgraph Result["增强后的集合"]
        R1["x1 (1.0, 2.0)"]
        R2["x2 (1.5, 2.5)"]
        R3["x3 (2.0, 1.5)"]
        R4["合成 (1.2, 2.2)"]
    end
    SMOTE --> Result
```

### 采样策略比较

**随机过采样**：复制少数类样本以匹配多数类数量。
- 优点：简单，无信息丢失
- 缺点：完全相同的复制品导致过拟合，增加训练时间

**随机欠采样**：移除多数类样本以匹配少数类数量。
- 优点：训练快，简单
- 缺点：丢弃可能有用的多数类数据，方差更高

**SMOTE**：通过插值创建合成少数类样本。
- 优点：生成新数据点，与随机过采样相比减少过拟合
- 缺点：可能在决策边界附近创建噪声样本，不考虑多数类分布

| 策略 | 改变的数据 | 风险 | 何时使用 |
|----------|-------------|------|-------------|
| 过采样 | 复制少数类 | 过拟合 | 小数据集，中度不平衡 |
| 欠采样 | 移除多数类 | 信息丢失 | 大数据集，想要快速训练 |
| SMOTE | 添加合成少数类 | 边界噪声 | 中度不平衡，有足够的少数类样本用于 k-NN |

### 类别权重

不改变数据，而是改变模型处理错误的方式。对误分类少数类赋予更高的权重。

对于 950 个负类和 50 个正类的二分类问题：
- 负类权重 = n_samples / (2 * n_negative) = 1000 / (2 * 950) = 0.526
- 正类权重 = n_samples / (2 * n_positive) = 1000 / (2 * 50) = 10.0

正类获得 19 倍的权重。误分类一个正类样本的代价相当于误分类 19 个负类样本。模型被迫关注少数类。

在逻辑回归中，这修改了损失函数：

```
weighted_loss = -sum(w_i * [y_i * log(p_i) + (1-y_i) * log(1-p_i)])
```

其中 w_i 取决于样本 i 的类别。

类别权重在期望上数学上等价于过采样，但不创建新的数据点。这使它们更快，并避免了复制样本的过拟合风险。

### 阈值调整

大多数分类器输出概率。默认阈值是 0.5：如果 P(正类) >= 0.5，预测为正类。但 0.5 是任意的。当类不平衡时，最优阈值通常低得多。

过程：
1. 训练一个模型
2. 在验证集上获取预测概率
3. 从 0.0 到 1.0 扫描阈值
4. 在每个阈值上计算 F1（或你选择的指标）
5. 选择使你的指标最大化的阈值

```mermaid
flowchart LR
    A[模型] --> B[预测概率]
    B --> C[扫描阈值 0.0 到 1.0]
    C --> D[在每个阈值计算 F1]
    D --> E[选择最佳阈值]
    E --> F[用于生产]
```

一个模型可能为一笔欺诈交易输出 P(fraud) = 0.15。在阈值 0.5 时，这被分类为非欺诈。在阈值 0.10 时，它被正确捕获。概率校准不如排名重要——只要欺诈获得比非欺诈更高的概率，就存在一个将它们分开的阈值。

### 成本敏感学习

类别权重的泛化。不是统一成本，而是分配特定的误分类成本：

|  | 预测为正 | 预测为负 |
|--|---|---|
| 实际为正 | 0（正确） | C_FN = 100 |
| 实际为负 | C_FP = 1 | 0（正确） |

遗漏一笔欺诈交易（FN）的代价是误报（FP）的 100 倍。模型优化总成本，而非总错误数。

当你可以估计现实世界的成本时，这是最原则性的方法。漏诊癌症的代价与导致额外活检的误报非常不同。明确这些成本会迫使正确的权衡。

### 决策流程图

```mermaid
flowchart TD
    A[开始：不平衡数据集] --> B{有多不平衡?}
    B -->|"< 70/30"| C["轻度：首先尝试类别权重"]
    B -->|"70/30 到 95/5"| D["中度：SMOTE + 类别权重"]
    B -->|"> 95/5"| E["严重：结合多种策略"]
    C --> F{数据够多?}
    D --> F
    E --> F
    F -->|"< 1000 样本"| G["过采样或 SMOTE，避免欠采样"]
    F -->|"1000-10000"| H["SMOTE + 阈值调整"]
    F -->|"> 10000"| I["欠采样可以，或类别权重"]
    G --> J[训练 + 用 F1/AUPRC 评估]
    H --> J
    I --> J
    J --> K{召回率够高?}
    K -->|否| L[降低阈值]
    K -->|是| M{精确率可接受?}
    M -->|否| N[提高阈值或添加特征]
    M -->|是| O[发布]
```

## 构建

### 步骤1：生成不平衡数据集

```python
import numpy as np


def make_imbalanced_data(n_majority=950, n_minority=50, seed=42):
    rng = np.random.RandomState(seed)

    X_maj = rng.randn(n_majority, 2) * 1.0 + np.array([0.0, 0.0])
    X_min = rng.randn(n_minority, 2) * 0.8 + np.array([2.5, 2.5])

    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(n_majority), np.ones(n_minority)])

    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]
```

### 步骤2：从零实现 SMOTE

```python
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def find_k_neighbors(X, idx, k):
    distances = []
    for i in range(len(X)):
        if i == idx:
            continue
        d = euclidean_distance(X[idx], X[i])
        distances.append((i, d))
    distances.sort(key=lambda x: x[1])
    return [d[0] for d in distances[:k]]


def smote(X_minority, k=5, n_synthetic=100, seed=42):
    rng = np.random.RandomState(seed)
    n_samples = len(X_minority)
    k = min(k, n_samples - 1)
    synthetic = []

    for _ in range(n_synthetic):
        idx = rng.randint(0, n_samples)
        neighbors = find_k_neighbors(X_minority, idx, k)
        neighbor_idx = neighbors[rng.randint(0, len(neighbors))]
        t = rng.random()
        new_point = X_minority[idx] + t * (X_minority[neighbor_idx] - X_minority[idx])
        synthetic.append(new_point)

    return np.array(synthetic)
```

### 步骤3：随机过采样和欠采样

```python
def random_oversample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_resampled = list(X)
    y_resampled = list(y)

    for cls, count in zip(classes, counts):
        if count < max_count:
            cls_indices = np.where(y == cls)[0]
            n_needed = max_count - count
            chosen = rng.choice(cls_indices, size=n_needed, replace=True)
            X_resampled.extend(X[chosen])
            y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]


def random_undersample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    X_resampled = []
    y_resampled = []

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        chosen = rng.choice(cls_indices, size=min_count, replace=False)
        X_resampled.extend(X[chosen])
        y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]
```

### 步骤4：带类别权重的逻辑回归

```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logistic_regression_weighted(X, y, weights, lr=0.01, epochs=200):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = sigmoid(z)
        error = pred - y
        weighted_error = error * weights

        gradient_w = (X.T @ weighted_error) / n_samples
        gradient_b = np.mean(weighted_error)

        w -= lr * gradient_w
        b -= lr * gradient_b

    return w, b


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weight_map = {}
    for cls, count in zip(classes, counts):
        weight_map[cls] = n_samples / (n_classes * count)
    return np.array([weight_map[yi] for yi in y])
```

### 步骤5：阈值调整

```python
def find_optimal_threshold(y_true, y_probs, metric="f1"):
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_probs >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        if metric == "f1":
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        elif metric == "recall":
            score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif metric == "precision":
            score = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
```

### 步骤6：评估函数

```python
def confusion_matrix_values(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp, tn, fp, fn


def compute_metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix_values(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }
```

### 步骤7：比较所有方法

```python
X, y = make_imbalanced_data(950, 50, seed=42)
split = int(0.8 * len(y))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 基线：无处理
w_base, b_base = logistic_regression_weighted(
    X_train, y_train, np.ones(len(y_train)), lr=0.1, epochs=300
)
probs_base = sigmoid(X_test @ w_base + b_base)
preds_base = (probs_base >= 0.5).astype(int)

# 过采样
X_over, y_over = random_oversample(X_train, y_train)
w_over, b_over = logistic_regression_weighted(
    X_over, y_over, np.ones(len(y_over)), lr=0.1, epochs=300
)
preds_over = (sigmoid(X_test @ w_over + b_over) >= 0.5).astype(int)

# SMOTE
minority_mask = y_train == 1
X_minority = X_train[minority_mask]
synthetic = smote(X_minority, k=5, n_synthetic=len(y_train) - 2 * int(minority_mask.sum()))
X_smote = np.vstack([X_train, synthetic])
y_smote = np.concatenate([y_train, np.ones(len(synthetic))])
w_sm, b_sm = logistic_regression_weighted(
    X_smote, y_smote, np.ones(len(y_smote)), lr=0.1, epochs=300
)
preds_smote = (sigmoid(X_test @ w_sm + b_sm) >= 0.5).astype(int)

# 类别权重
sample_weights = compute_class_weights(y_train)
w_cw, b_cw = logistic_regression_weighted(
    X_train, y_train, sample_weights, lr=0.1, epochs=300
)
probs_cw = sigmoid(X_test @ w_cw + b_cw)
preds_cw = (probs_cw >= 0.5).astype(int)

# 阈值调整（在留出验证集上调整，而非测试集）
probs_val = sigmoid(X_val @ w_cw + b_cw)
best_thresh, best_f1 = find_optimal_threshold(y_val, probs_val, metric="f1")
preds_thresh = (probs_cw >= best_thresh).astype(int)
```

代码文件在单个脚本中运行所有这些内容并打印结果。

## 使用

使用 scikit-learn 和 imbalanced-learn，这些技术是一行代码：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

model_weighted = LogisticRegression(class_weight="balanced")
model_weighted.fit(X_train, y_train)
print(classification_report(y_test, model_weighted.predict(X_test)))

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
model_smote = LogisticRegression()
model_smote.fit(X_resampled, y_resampled)
print(classification_report(y_test, model_smote.predict(X_test)))

pipeline = Pipeline([
    ("smote", SMOTE()),
    ("model", LogisticRegression(class_weight="balanced")),
])
pipeline.fit(X_train, y_train)
print(classification_report(y_test, pipeline.predict(X_test)))
```

从零实现的版本精确展示了每种技术的作用。SMOTE 只是对少数类的 k-NN 插值。类别权重乘以损失。阈值调整是 cutoff 上的 for 循环。没有魔法。

## 交付

本课程产生：
- `outputs/skill-imbalanced-data.md` -- 处理不平衡分类问题的决策清单

## 练习

1. **Borderline-SMOTE**：修改 SMOTE 实现，只为接近决策边界的少数类点生成合成样本（那些 k 最近邻居包含多数类样本的点）。在类重叠的数据集上与标准 SMOTE 比较结果。

2. **成本矩阵优化**：实现成本敏感学习，其中成本矩阵作为一个参数。创建一个函数，接受成本矩阵并返回最小化预期成本的最优预测。用不同的成本比率（1:10、1:100、1:1000）测试，并绘制精确率-召回率权衡如何变化。

3. **阈值校准**：实现 Platt 缩放（在模型的原始输出上拟合逻辑回归以产生校准概率）。比较校准前后的精确率-召回率曲线。展示校准不改变排名（AUC 保持不变），但使概率更有意义。

4. **平衡 bagging 集成**：训练多个模型，每个都在平衡的自助样本上（所有少数类 + 多数类的随机子集）。平均它们的预测。将这种方法与带有 SMOTE 的单个模型进行比较。测量运行之间的性能和方差。

5. **失衡比率实验**：取一个平衡数据集并逐步增加失衡比率（50/50、70/30、90/10、95/5、99/1）。对于每个比率，用 SMOTE 和不用 SMOTE 进行训练。为两种方法绘制 F1 与失衡比率的关系图。在什么比率下 SMOTE 开始产生有意义的差异？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
|类别不平衡 | "一个类有更多样本" | 数据集中类的分布显著偏斜，导致模型偏向多数类 |
| SMOTE | "合成过采样" | 通过在现有少数类样本与其 k 个最近少数类邻居之间进行插值来创建新的少数类样本 |
| 类别权重 | "让稀有类的错误更昂贵" | 用类别特定权重乘以损失函数，使模型更严厉地惩罚少数类误分类 |
| 阈值调整 | "移动决策边界" | 将分类的概率 cutoff 从默认的 0.5改变为优化所需指标的值 |
| 精确率-召回率权衡 | "你不能两者兼得" | 降低阈值捕获更多正类（更高召回率）但也标记更多假正类（更低精确率），反之亦然 |
| AUPRC | "PR 曲线下面积" | 将精确率-召回率曲线汇总为单个数字；当类严重不平衡时比 AUPRC更有信息量 |
| Matthews 相关系数 | "平衡指标" | 预测标签和实际标签之间的相关性，只有当模型在两个类上都表现良好时才产生高分数 |
| 成本敏感学习 | "不同的错误代价不同" | 将现实世界的误分类成本纳入训练目标，使模型优化总成本而非错误计数 |
| 随机过采样 | "复制少数类" | 重复少数类样本以平衡类计数；简单但有过度拟合到重复点的风险 |

## 扩展阅读

- [SMOTE: Synthetic Minority Over-sampling Technique (Chawla et al., 2002)](https://arxiv.org/abs/1106.1813) -- 原始 SMOTE 论文，仍然是不平衡学习领域被引用最多的工作
- [Learning from Imbalanced Data (He & Garcia, 2009)](https://ieeexplore.ieee.org/document/5128907) -- 涵盖采样、成本敏感和算法方法的综合调查
- [imbalanced-learn 文档](https://imbalanced-learn.org/stable/) --带有 SMOTE 变体、欠采样策略和流程集成的 Python 库
- [The Precision-Recall Plot Is More Informative than the ROC Plot (Saito & Rehmsmeier, 2015)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) -- 何时以及为什么对不平衡问题更倾向于 PR 曲线而不是 ROC 曲线
