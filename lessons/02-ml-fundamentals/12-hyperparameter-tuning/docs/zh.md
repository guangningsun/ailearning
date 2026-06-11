# 超参数调优

> 超参数是你在训练开始前就要拧动的旋钮。拧得好不好，决定了一个模型是平庸还是优秀。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 2，第 11 课（集成方法）
**时间：** 约 90 分钟

## 学习目标

- 从零实现网格搜索、随机搜索和贝叶斯优化，并比较它们的采样效率
- 解释为什么当大多数超参数具有较低有效维度时，随机搜索优于网格搜索
- 构建一个使用代理模型和采集函数引导搜索的贝叶斯优化循环
- 设计一个通过适当交叉验证避免过拟合验证集的超参数调优策略

## 问题

你的梯度提升模型有学习率、树的数量、最大深度、每叶最小样本数、子采样率和列采样率。共六个超参数。如果每个有 5 个合理值，网格就有 5^6 = 15,625 种组合。每个训练需要 10 秒。全部试一遍需要 43 小时。

网格搜索是最显而易见的做法，也是规模化时最差的做法。随机搜索用更少的计算做得更好。贝叶斯优化通过从过去的评估中学习做得更好。知道使用哪种策略，以及哪些超参数真正重要，可以节省数天的浪费 GPU 时间。

## 概念

### 参数 vs 超参数

参数在训练过程中学习（权重、偏置、分割阈值）。超参数在训练开始前设置，控制学习如何发生。

| 超参数 | 控制什么 | 典型范围 |
|---------------|-----------------|---------------|
| 学习率 | 每次更新的步长 | 0.001 到 1.0 |
| 树的数量/轮数 | 训练多久 | 10 到 10,000 |
| 最大深度 | 模型复杂度 | 1 到 30 |
| 正则化 (lambda) | 防止过拟合 | 0.0001 到 100 |
| 批量大小 | 梯度估计噪声 | 16 到 512 |
| Dropout 率 | 丢弃的神经元比例 | 0.0 到 0.5 |

### 网格搜索

网格搜索评估指定值的每一种组合。它是穷尽的且易于理解，但随超参数数量指数增长。

```
两个超参数的网格：

  learning_rate: [0.01, 0.1, 1.0]
  max_depth:     [3, 5, 7]

  评估次数：3 x 3 = 9 种组合

  (0.01, 3)  (0.01, 5)  (0.01, 7)
  (0.1,  3)  (0.1,  5)  (0.1,  7)
  (1.0,  3)  (1.0,  5)  (1.0,  7)
```

网格搜索有一个根本缺陷：如果一个超参数重要而另一个不重要，大部分评估就被浪费了。9 次评估只能得到重要参数的唯一值 3 个。

### 随机搜索

随机搜索从分布中采样超参数，而不是从网格中采样。用同样 9 次评估的预算，你得到每个超参数的 9 个唯一值。

```mermaid
flowchart LR
    subgraph 网格搜索
        G1[3 个唯一学习率]
        G2[3 个唯一最大深度]
        G3[9 次总评估]
    end

    subgraph 随机搜索
        R1[9 个唯一学习率]
        R2[9 个唯一最大深度]
        R3[9 次总评估]
    end
```

随机优于网格的原因（Bergstra & Bengio, 2012）：

- 大多数超参数具有较低的有效维度。对于给定问题，通常 6 个超参数中只有 1-2 个重要。
- 网格搜索在不重要的维度上浪费评估。
- 对于同样的预算，随机搜索在重要维度上覆盖得更密集。
- 60 次随机试验中，你有 95% 的概率找到一个在最优值 5% 范围内的点（如果搜索空间中存在这样一个点）。

### 贝叶斯优化

随机搜索忽略结果。它不知道高学习率会导致发散，或者深度 3 始终优于深度 10。贝叶斯优化利用过去的评估来决定下一步搜索哪里。

```mermaid
flowchart TD
    A[定义搜索空间] --> B[评估初始随机点]
    B --> C[将代理模型拟合到结果]
    C --> D[使用采集函数选择下一个点]
    D --> E[在该点评估模型]
    E --> F{预算耗尽？}
    F -->|否| C
    F -->|是| G[返回找到的最佳超参数]
```

两个关键组件：

**代理模型：** 一个评估成本低（通常是高斯过程）的模型，近似昂贵的目标函数。它在任何搜索空间点给出预测和不确定性估计。

**采集函数：** 通过平衡利用（在已知好点附近搜索）和探索（在不确定性高的地方搜索）来决定下一步在哪里评估。常见选择：

- **期望提升 (EI)：** 在这一点上，我们期望比当前最佳有多少改进？
- **上置信界 (UCB)：** 预测加上不确定性的倍数。较高的 UCB 意味着有前景或未被探索。
- **改进概率 (PI)：** 这个点击败当前最佳的概率是多少？

贝叶斯优化通常比随机搜索少用 2-5 倍的评估次数就能找到更好的超参数。拟合代理模型的开销与训练实际模型相比可以忽略不计。

### 早停

不是每个训练运行都需要完成。如果一个配置在 10 轮后明显很差，就停止它并继续。这是超参数搜索背景下的早停。

策略：
- **基于耐心：** 如果验证损失连续 N 轮没有改善则停止
- **中位数剪枝：** 如果试验的中间结果差于同一步骤已完成试验的中位数则停止
- **Hyperband：** 给许多配置分配小预算，然后逐步增加最佳配置的预算

Hyperband 特别有效。它从 81 个配置开始，每个训练 1 轮，保留前三分之一，给它们 3 轮，再保留前三分之一，以此类推。这比用完整预算评估所有配置快 10-50 倍。

### 学习率调度器

学习率几乎始终是最重要的超参数。调度器不是在训练中保持固定，而是在训练期间调整它。

| 调度器 | 公式 | 何时使用 |
|-----------|---------|-------------|
| 步衰减 | 每 N 轮乘以 0.1 | 经典 CNN 训练 |
| 余弦退火 | lr * 0.5 * (1 + cos(pi * t / T)) | 现代默认 |
| 预热 + 衰减 | 线性增加然后余弦衰减 | Transformer |
| 单周期 | 在一个周期内增加然后减少 | 快速收敛 |
|  plateau 时降低 | 指标停滞时按因子减少 | 安全默认值 |

### 超参数重要性

并非所有超参数都同等重要。关于随机森林（Probst 等，2019）和梯度提升的研究显示了一致的模式：

**高重要性：**
- 学习率（始终首先调优）
- 估计器数量/轮数（使用早停而不是调优）
- 正则化强度

**中等重要性：**
- 最大深度/层数
- 每叶最小样本数/权重衰减
- 子采样率

**低重要性：**
- 最大特征数（对于随机森林）
- 特定激活函数选择
- 批量大小（在合理范围内）

首先调优重要的，其余保持默认值。

### 实用策略

```mermaid
flowchart TD
    A[从默认值开始] --> B[粗粒度随机搜索：20-50 次试验]
    B --> C[识别重要超参数]
    C --> D[细粒度随机或贝叶斯搜索：在缩小空间中的 50-100 次试验]
    D --> E[使用最佳超参数训练最终模型]
    E --> F[在完整训练数据上重新训练]
```

具体工作流程：

1. **从库默认值开始。** 它们是由有经验的从业者选择的，通常已经达到了 80% 的效果。
2. **粗粒度随机搜索。** 宽范围，20-50 次试验。使用早停快速终止差的运行。
3. **分析结果。** 哪些超参数与性能相关？缩小搜索空间。
4. **细搜索。** 在缩小空间中的贝叶斯优化或集中随机搜索。50-100 次试验。
5. **用找到的最佳超参数在所有训练数据上重新训练。**

### 交叉验证集成

在单个验证分割上调优超参数是有风险的。最佳超参数可能过拟合于特定的验证折。嵌套交叉验证通过使用两个循环来解决这个问题：

- **外循环**（评估）：将数据分割为训练+验证和测试。报告无偏性能。
- **内循环**（调优）：将训练+验证分割为训练和验证。找到最佳超参数。

```mermaid
flowchart TD
    D[完整数据集] --> O1[外折 1：测试]
    D --> O2[外折 2：测试]
    D --> O3[外折 3：测试]
    D --> O4[外折 4：测试]
    D --> O5[外折 5：测试]

    O1 --> I1[在其余数据上进行内 5 折交叉验证]
    I1 --> T1[折 1 的最佳超参数]
    T1 --> E1[在外测试折 1 上评估]

    O2 --> I2[在其余数据上进行内 5 折交叉验证]
    I2 --> T2[折 2 的最佳超参数]
    T2 --> E2[在外测试折 2 上评估]
```

每个外折独立找到自己的最佳超参数。外分数是泛化性能的无偏估计。

使用 sklearn：

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

inner_cv = GridSearchCV(
    GradientBoostingRegressor(),
    param_grid={
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 5],
        "n_estimators": [50, 100, 200],
    },
    cv=5,
    scoring="neg_mean_squared_error",
)

outer_scores = cross_val_score(
    inner_cv, X, y, cv=5, scoring="neg_mean_squared_error"
)

print(f"嵌套交叉验证 MSE：{-outer_scores.mean():.4f} +/- {outer_scores.std():.4f}")
```

这是昂贵的（5 个外折 × 5 个内折 × 27 个网格点 = 675 次模型拟合），但它给你一个可信的性能估计。在论文中报告最终结果或决策 stakes 高时使用它。

### 实用技巧

**首先从学习率开始。** 对于基于梯度的方法，它始终是最重要的超参数。差的学习率使其他一切都无关紧要。首先将其他超参数固定在默认值并 sweep 学习率。

**对学习率和正则化使用对数均匀分布。** 0.001 和 0.01 之间的差异与 0.1 和 1.0 之间的差异同样重要。线性搜索在大端浪费预算。

**使用早停而不是调优 n_estimators。** 对于提升和神经网络，将 n_estimators 或轮数设高，让早停决定何时停止。这从搜索中移除一个超参数。

**预算分配。** 将调优预算的 60% 花在前两个最重要的超参数上。其余 40% 花在其他所有上。前两个占性能变化的大部分。

**规模很重要。** 永远不要在对数尺度上搜索批量大小（16、32、64 都可以）。始终在对数尺度上搜索学习率。将搜索分布与超参数如何影响模型相匹配。

| 模型类型 | 顶级超参数 | 推荐搜索 | 预算 |
|-----------|--------------------|--------------------|--------|
| 随机森林 | n_estimators, max_depth, min_samples_leaf | 随机搜索，50 次试验 | 低（训练快） |
| 梯度提升 | learning_rate, n_estimators, max_depth | 贝叶斯，100 次试验 + 早停 | 中 |
| 神经网络 | learning_rate, weight_decay, batch_size | 贝叶斯或随机，100+ 次试验 | 高（训练慢） |
| SVM | C, gamma (RBF 核) | 对数尺度网格，25-50 次试验 | 低（2 个参数） |
| Lasso/Ridge | alpha | 对数尺度 1D 搜索，20 次试验 | 非常低 |
| XGBoost | learning_rate, max_depth, subsample, colsample | 贝叶斯，100-200 次试验 + 早停 | 中 |

**有疑问时：** 随机搜索，超参数数量的 2 倍作为试验次数（例如，6 个超参数 = 最少 12 次试验）。你会惊讶于 50 次试验的随机搜索经常击败精心设计的网格搜索。

## 构建

### 第 1 步：从零实现网格搜索

`code/tuning.py` 中的代码实现了网格搜索、随机搜索和一个简单的贝叶斯优化器。

```python
def grid_search(model_fn, param_grid, X_train, y_train, X_val, y_val):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    best_score = -float("inf")
    best_params = None
    n_evals = 0

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        model = model_fn(**params)
        model.fit(X_train, y_train)
        score = evaluate(model, X_val, y_val)
        n_evals += 1

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score, n_evals
```

### 第 2 步：从零实现随机搜索

```python
def random_search(model_fn, param_distributions, X_train, y_train,
                  X_val, y_val, n_iter=50, seed=42):
    rng = np.random.RandomState(seed)
    best_score = -float("inf")
    best_params = None

    for _ in range(n_iter):
        params = {k: sample(v, rng) for k, v in param_distributions.items()}
        model = model_fn(**params)
        model.fit(X_train, y_train)
        score = evaluate(model, X_val, y_val)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score, n_iter
```

### 第 3 步：贝叶斯优化（简化版）

核心思想：将高斯过程拟合到观测到的（超参数，分数）对，然后使用采集函数决定下一步在哪里探索。

```python
class SimpleBayesianOptimizer:
    def __init__(self, search_space, n_initial=5):
        self.search_space = search_space
        self.n_initial = n_initial
        self.X_observed = []
        self.y_observed = []

    def _kernel(self, x1, x2, length_scale=1.0):
        dists = np.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=2)
        return np.exp(-0.5 * dists / length_scale ** 2)

    def _fit_gp(self, X_new):
        X_obs = np.array(self.X_observed)
        y_obs = np.array(self.y_observed)
        y_mean = y_obs.mean()
        y_centered = y_obs - y_mean

        K = self._kernel(X_obs, X_obs) + 1e-4 * np.eye(len(X_obs))
        K_star = self._kernel(X_new, X_obs)

        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_centered))
        mu = K_star @ alpha + y_mean

        v = np.linalg.solve(L, K_star.T)
        var = 1.0 - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-6)

        return mu, var

    def _expected_improvement(self, mu, var, best_y):
        sigma = np.sqrt(var)
        z = (mu - best_y) / (sigma + 1e-10)
        ei = sigma * (z * norm_cdf(z) + norm_pdf(z))
        return ei

    def suggest(self):
        if len(self.X_observed) < self.n_initial:
            return sample_random(self.search_space)

        candidates = [sample_random(self.search_space) for _ in range(500)]
        X_cand = np.array([to_vector(c) for c in candidates])
        mu, var = self._fit_gp(X_cand)
        ei = self._expected_improvement(mu, var, max(self.y_observed))
        return candidates[np.argmax(ei)]

    def observe(self, params, score):
        self.X_observed.append(to_vector(params))
        self.y_observed.append(score)
```

GP 代理在每个候选点给出两个东西：预测分数（mu）和不确定性（var）。期望提升平衡这些：它偏向模型预测高分或不确定性高的点。早期，大多数点有高不确定性，所以优化器探索。之后，它聚焦于最有前景的区域。

### 第 4 步：比较所有方法

在相同的合成目标上运行所有三种方法并比较。这个比较使用一个简化的包装器，调用每个优化器与一个直接目标函数（没有模型训练），所以 API 与上面的基于模型的实现不同：

```python
def synthetic_objective(params):
    lr = params["learning_rate"]
    depth = params["max_depth"]
    return -(np.log10(lr) + 2) ** 2 - (depth - 4) ** 2 + 10

param_grid = {
    "learning_rate": [0.001, 0.01, 0.1, 1.0],
    "max_depth": [2, 3, 4, 5, 6, 7, 8],
}

grid_best = None
grid_score = -float("inf")
grid_history = []
for combo in itertools.product(*param_grid.values()):
    params = dict(zip(param_grid.keys(), combo))
    score = synthetic_objective(params)
    grid_history.append((params, score))
    if score > grid_score:
        grid_score = score
        grid_best = params

param_dist = {
    "learning_rate": ("log_float", 0.001, 1.0),
    "max_depth": ("int", 2, 8),
}

rand_best = None
rand_score = -float("inf")
rand_history = []
rng = np.random.RandomState(42)
for _ in range(28):
    params = {k: sample(v, rng) for k, v in param_dist.items()}
    score = synthetic_objective(params)
    rand_history.append((params, score))
    if score > rand_score:
        rand_score = score
        rand_best = params

optimizer = SimpleBayesianOptimizer(param_dist, n_initial=5)
bayes_history = []
for _ in range(28):
    params = optimizer.suggest()
    score = synthetic_objective(params)
    optimizer.observe(params, score)
    bayes_history.append((params, score))
bayes_score = max(s for _, s in bayes_history)

print(f"{'方法':<20} {'最佳分数':>12} {'评估次数':>12}")
print("-" * 50)
print(f"{'网格搜索':<20} {grid_score:>12.4f} {len(grid_history):>12}")
print(f"{'随机搜索':<20} {rand_score:>12.4f} {len(rand_history):>12}")
print(f"{'贝叶斯优化':<20} {bayes_score:>12.4f} {len(bayes_history):>12}")
```

用相同的预算，贝叶斯优化通常最快找到最佳分数，因为它不会在明显糟糕的区域浪费评估。随机搜索比网格搜索覆盖更多 ground。网格搜索只在你有很少的超参数且能承受穷尽搜索时获胜。

## 使用

### Optuna 实战

Optuna 是严肃超参数调优的推荐库。它开箱即用地支持剪枝、分布式搜索和可视化。

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    n_est = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 2, 10)

    model = GradientBoostingRegressor(
        learning_rate=lr,
        n_estimators=n_est,
        max_depth=max_depth,
    )
    model.fit(X_train, y_train)
    return mean_squared_error(y_val, model.predict(X_val))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print(f"最佳参数：{study.best_params}")
print(f"最佳 MSE：{study.best_value:.4f}")
```

关键 Optuna 特性：
- `suggest_float(..., log=True)` 用于最好在对数尺度上搜索的参数（学习率、正则化）
- `suggest_int` 用于整数参数
- `suggest_categorical` 用于离散选择
- 内置 MedianPruner 用于早停差的试验
- `study.trials_dataframe()` 用于分析

### 带剪枝的 Optuna

剪枝提前停止没有前景的试验，节省大量计算。以下是模式：

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("lr", 1e-4, 0.5, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }

    model = GradientBoostingRegressor(**params)
    scores = cross_val_score(model, X_train, y_train, cv=3,
                             scoring="neg_mean_squared_error")
    mean_score = -scores.mean()

    trial.report(mean_score, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return mean_score

pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
study = optuna.create_study(direction="minimize", pruner=pruner)
study.optimize(objective, n_trials=200)
```

`MedianPruner` 如果试验的中间值差于同一步骤所有已完成试验的中位数则停止该试验。剪枝需要调用 `trial.report()` 来报告中间指标，`trial.should_prune()` 来检查试验是否应该停止。`n_startup_trials=10` 确保在剪枝启动前至少 10 个试验完全完成。这通常节省 40-60% 的总计算量。

### sklearn 内置调优器

对于快速实验，sklearn 提供 `GridSearchCV`、`RandomizedSearchCV` 和 `HalvingRandomSearchCV`：

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint

param_dist = {
    "learning_rate": loguniform(1e-4, 0.5),
    "max_depth": randint(2, 10),
    "n_estimators": randint(50, 500),
}

search = RandomizedSearchCV(
    GradientBoostingRegressor(),
    param_dist,
    n_iter=100,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42,
    n_jobs=-1,
)
search.fit(X_train, y_train)
print(f"最佳参数：{search.best_params_}")
print(f"最佳交叉验证 MSE：{-search.best_score_:.4f}")
```

从 scipy 对学习率和正则化使用 `loguniform`。对整数超参数使用 `randint`。`n_jobs=-1` 标志跨所有 CPU 核心并行化。

### 超参数调优中的常见错误

**通过预处理的数据泄漏。** 如果在交叉验证前对完整数据集拟合缩放器，信息从验证 fold 泄漏到训练中。始终将预处理放在 `Pipeline` 内部，这样它只对训练 fold 拟合。

**过拟合验证集。** 运行数千次试验实际上是在验证集上训练。使用嵌套交叉验证获得最终性能估计，或者保留一个在调优期间从不触碰的单独测试集。

**搜索范围太窄。** 如果你的最佳值在搜索空间的边界上，你还没有足够广泛地搜索。最优值可能在你的范围之外。始终检查最佳参数是否在边缘。

**忽略交互效应。** 学习率和估计器数量在提升中强烈交互。低学习率需要更多估计器。独立调优它们比一起调优得到更差的结果。

**对迭代模型不使用早停。** 对于梯度提升和神经网络，将 n_estimators 或轮数设高并使用早停。这严格优于将迭代次数作为超参数调优。

## 练习

1. 用相同的总预算（例如 50 次评估）运行网格搜索和随机搜索。比较找到的最佳分数。用不同种子运行实验 10 次。随机搜索赢了多少次？

2. 从零实现 Hyperband。从 81 个配置开始，每个训练 1 轮。在每轮保留前 1/3 并将它们的预算翻三倍。将总计算量（所有配置的所有轮次之和）与用完整预算运行 81 个配置进行比较。

3. 将学习率调度器（余弦退火）添加到第 11 课的梯度提升实现中。与固定学习率相比，它有帮助吗？

4. 使用 Optuna 在真实数据集上调优 RandomForestClassifier（例如 sklearn 的乳腺癌数据集）。使用 `optuna.visualization.plot_param_importances(study)` 查看哪些超参数最重要。它与本课的重要性排名匹配吗？

5. 实现一个简单的采集函数（期望提升）并展示探索与利用。绘制代理模型的均值和不确定性，并显示 EI 选择下一步在哪里评估。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|----------------------|
| 超参数 | "你选择的一个设置" | 训练前设置的值，控制学习过程，不是从数据中学习的 |
| 网格搜索 | "尝试每种组合" | 在指定参数网格上的穷尽搜索。指数成本。 |
| 随机搜索 | "只是随机采样" | 从分布中采样超参数。比网格搜索更好地覆盖重要维度。 |
| 贝叶斯优化 | "智能搜索" | 使用目标函数的代理模型来决定下一步在哪里评估，平衡探索和利用 |
| 代理模型 | "一个便宜的近似" | 一个模型（通常是高斯过程），从观测到的评估近似昂贵的目标函数 |
| 采集函数 | "下一步在哪里看" | 通过平衡期望提升与不确定性对候选点打分。EI 和 UCB 是常见选择。 |
| 早停 | "停止浪费时间" | 当验证性能停止改善时终止训练 |
| Hyperband | "配置的淘汰赛" | 自适应资源分配：用小预算启动许多配置，保留最好的并增加它们的预算 |
| 学习率调度器 | "训练期间改变 lr" | 一个在训练过程中调整学习率的函数，以获得更好的收敛 |

## 延伸阅读

- [Bergstra & Bengio: Random Search for Hyper-Parameter Optimization (2012)](https://jmlr.org/papers/v13/bergstra12a.html) -- 证明随机优于网格的论文
- [Snoek et al., Practical Bayesian Optimization of Machine Learning Algorithms (2012)](https://arxiv.org/abs/1206.2944) -- 贝叶斯优化用于 ML
- [Li et al., Hyperband: A Novel Bandit-Based Approach (2018)](https://jmlr.org/papers/v18/16-558.html) -- Hyperband 论文
- [Optuna: A Next-generation Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902) -- Optuna 论文
- [Probst et al., Tunability: Importance of Hyperparameters (2019)](https://jmlr.org/papers/v20/18-444.html) -- 哪些超参数重要