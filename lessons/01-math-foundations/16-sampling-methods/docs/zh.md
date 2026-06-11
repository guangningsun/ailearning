# 采样方法（Sampling Methods）

> 采样是 AI 探索可能性空间的方式。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 1，第 06-07 课（概率论、贝叶斯定理）
**时间：** 约 120 分钟

## 学习目标

- 仅使用均匀随机数，从零实现逆 CDF、拒绝采样和重要性采样（Importance Sampling）
- 为语言模型 token 生成构建温度（Temperature）、top-k 和 top-p（核）采样
- 解释重参数化技巧（Reparameterization Trick）以及它为何能让 VAE 中的采样支持反向传播
- 运行 Metropolis-Hastings MCMC，从非归一化的目标分布中采样

## 问题

语言模型处理完你的提示词，生成一个包含 50,000 个 logits 的向量。词汇表中的每个 token 对应一个。现在它该选哪一个？

如果永远选最高概率的 token，每次回答都一样。确定性的，很无聊。如果完全均匀随机地选，输出就是胡言乱语。答案就在这两个极端之间，而控制这个"之间"的机制，就是采样。

采样的应用远不止文本生成。强化学习通过采样轨迹来估计策略梯度。VAE 通过从学到的分布中采样并让梯度穿过随机性来学习隐表示。扩散模型（Diffusion Models）通过采样噪声并逐步去噪来生成图像。蒙特卡洛方法（Monte Carlo）估计没有闭式解的积分。MCMC 算法探索无法穷举的高维后验分布。

每个生成式 AI 系统本质上都是一个采样系统。采样策略决定了输出的质量、多样性和可控性。本课从零构建每一种主要的采样方法，从均匀随机数开始，一直延伸到驱动现代大语言模型和生成模型的技术。

## 概念

### 为什么采样很重要

采样在 AI 和机器学习中扮演着四个基础角色：

**生成。** 语言模型、扩散模型和 GAN 都通过采样来产生输出。采样算法直接控制创造力、连贯性和多样性。温度、top-k 和核采样（Nucleus Sampling）就是工程师每天在调的旋钮。

**训练。** 随机梯度下降采样 mini-batch。Dropout 采样要停用的神经元。数据增强采样随机变换。重要性采样通过重新加权样本来降低强化学习（PPO、TRPO）中的梯度方差。

**估计。** ML 中很多量没有闭式解。数据分布上的期望损失、能量模型的配分函数、贝叶斯推断中的证据。蒙特卡洛估计通过对样本取平均来近似所有这些量。

**探索。** MCMC 算法在贝叶斯推断中探索后验分布。进化策略采样参数扰动。Thompson 采样在老虎机问题中平衡探索与利用。

核心挑战：你只能直接从不多的几种简单分布（均匀、正态）中采样。对于其他所有分布，你需要一种方法，把简单样本转换成目标分布的样本。

### 均匀随机采样

所有采样方法都从这里开始。均匀随机数生成器在 [0, 1) 中产生值，其中每个等长子区间具有相同的概率。

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    其中 0 <= a <= b <= 1

性质：
  E[U] = 0.5
  Var(U) = 1/12
```

要从 n 个元素的离散集合中均匀采样，生成 U 并返回 floor(n * U)。要从连续区间 [a, b] 中采样，计算 a + (b - a) * U。

关键洞见：一个均匀随机数恰好包含足够产生任意分布一个样本的随机性。诀窍在于找到正确的变换。

### 逆 CDF 方法（逆变换采样，Inverse Transform Sampling）

累积分布函数（CDF）将值映射为概率：

```
F(x) = P(X <= x)

性质：
  F 是非递减的
  F(-inf) = 0
  F(+inf) = 1
  F 将实轴映射到 [0, 1]
```

逆 CDF 将概率映射回值。如果 U ~ Uniform(0, 1)，那么 X = F_inverse(U) 服从目标分布。

```
算法：
  1. 生成 u ~ Uniform(0, 1)
  2. 返回 F_inverse(u)

为什么有效：
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**指数分布示例：**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

解 F(x) = u 求 x：
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

由于 (1 - U) 和 U 同分布：
  x = -ln(u) / lambda
```

当你能写出 F_inverse 的闭式时，这个方法完美工作。对于正态分布，逆 CDF 没有闭式，因此我们使用其他方法（Box-Muller，或数值近似）。

**离散版本：** 对于离散分布，将 CDF 构建为累积和，生成 U，找到第一个累积和超过 U 的索引。这就是第 06 课中 `sample_categorical` 的工作原理。

### 拒绝采样（Rejection Sampling）

当无法求逆 CDF、但可以评估目标 PDF（差一个常数因子）时，拒绝采样可以工作。

```
目标分布：p(x)  （可以评估，可能非归一化）
提议分布：q(x)  （可以从中采样）
界：M，使得对所有 x，p(x) <= M * q(x)

算法：
  1. 采样 x ~ q(x)
  2. 采样 u ~ Uniform(0, 1)
  3. 如果 u < p(x) / (M * q(x))，接受 x
  4. 否则，拒绝并回到步骤 1

接受率 = 1/M
```

界 M 越紧，接受率越高。在低维（1-3 维）中，拒绝采样效果不错。在高维中，接受率呈指数下降，因为大部分提议体积被拒绝。这就是拒绝采样的维度灾难。

**示例：从截断正态分布采样。** 使用截断区间上的均匀提议。包络 M 是该区间内正态 PDF 的最大值。

**示例：从半圆采样。** 在外接矩形内均匀提议。如果点落在半圆内就接受。这就是蒙特卡洛计算圆周率的方式：接受率等于面积比 pi/4。

### 重要性采样（Importance Sampling）

有时你不需要从目标分布 p(x) 中采样。你需要估计 p(x) 下的期望，而你手头有来自另一个分布 q(x) 的样本。

```
目标：估计 E_p[f(x)] = f(x) * p(x) 的积分

改写：
  E_p[f(x)] = f(x) * (p(x)/q(x)) * q(x) 的积分
            = E_q[f(x) * w(x)]

其中 w(x) = p(x) / q(x)  是重要性权重。

估计量：
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    其中 x_i ~ q(x)
```

这在强化学习中至关重要。在 PPO（Proximal Policy Optimization，近端策略优化）中，你使用旧策略 pi_old 收集轨迹，但想优化新策略 pi_new。重要性权重是 pi_new(a|s) / pi_old(a|s)。PPO 对这些权重进行裁剪，防止新策略偏离旧策略太远。

重要性采样估计量的方差取决于 q 与 p 的相似程度。如果 q 与 p 差异很大，少数样本会获得巨大的权重并主导估计。自归一化重要性采样除以权重之和来缓解这个问题：

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### 蒙特卡洛估计（Monte Carlo Estimation）

蒙特卡洛估计通过对随机样本取平均来近似积分。大数定律保证收敛。

```
目标：估计 I = g(x) 在区域 D 上的积分

方法：
  1. 从 D 中均匀采样 x_1, ..., x_N
  2. I ~ (D 的体积 / N) * sum(g(x_i))

误差：O(1 / sqrt(N))   与维度无关
```

误差率与维度无关。这就是为什么蒙特卡洛方法在基于网格的积分不可行的高维场景中占据主导。

**估计圆周率 pi：**

```
从 [-1, 1] x [-1, 1] 中均匀采样 (x, y)
统计落在单位圆内的点数：x^2 + y^2 <= 1
pi ~ 4 * (圆内点数) / (总点数)
```

**估计期望：**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    其中 x_i ~ p(x)

样本均值收敛到真实期望。
估计量的方差 = Var(f(X)) / N
```

### 马尔可夫链蒙特卡洛（MCMC）：Metropolis-Hastings

MCMC 构造一条马尔可夫链（Markov Chain），使其平稳分布为目标分布 p(x)。经过足够多的步数后，从链中得到的样本（近似地）服从 p(x)。

```
目标：p(x)  （已知，差一个归一化常数）
提议：q(x'|x)  （给定当前状态，如何提议下一个状态）

Metropolis-Hastings 算法：
  1. 从某个 x_0 开始
  2. 对 t = 1, 2, ..., T：
     a. 提议 x' ~ q(x'|x_t)
     b. 计算接受率：
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. 以概率 min(1, alpha) 接受：
        - 如果 u < alpha (u ~ Uniform(0,1))：x_{t+1} = x'
        - 否则：x_{t+1} = x_t
  3. 丢弃前 B 个样本（预热期，burn-in）
  4. 返回剩余样本
```

对于对称提议（q(x'|x) = q(x|x')），接受率化简为 p(x')/p(x)。这是原始的 Metropolis 算法。

**为什么有效。** 接受规则确保了细致平衡（Detailed Balance）：处于 x 并移动到 x' 的概率等于处于 x' 并移动到 x 的概率。细致平衡意味着 p(x) 是链的平稳分布。

**实践考虑：**
- 预热期（Burn-in）：丢弃链达到平衡之前的早期样本
- 稀释（Thinning）：每 k 个样本保留一个以减少自相关
- 提议尺度：太小则链移动缓慢（接受率高，探索慢）；太大则大多数提议被拒绝（接受率低，停滞不前）
- 高维下高斯提议的最优接受率约为 0.234

### Gibbs 采样（Gibbs Sampling）

Gibbs 采样是 MCMC 针对多变量分布的特例。它不一次性在所有维度上提议移动，而是一次一个变量地从其条件分布中采样。

```
目标：p(x_1, x_2, ..., x_d)

算法：
  每次迭代 t：
    采样 x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    采样 x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    采样 x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs 采样要求你能从每个条件分布 p(x_i | x_{-i}) 中采样。这对许多模型来说很简单：
- 贝叶斯网络：条件分布由图的拓扑结构决定
- 高斯混合模型：条件分布是高斯分布
- Ising 模型：每个自旋的条件分布仅取决于其邻居

接受率始终为 1（每个提议都被接受），因为从精确的条件分布中采样自动满足细致平衡。

**局限性。** 当变量高度相关时，Gibbs 采样混合缓慢，因为一次更新一个变量无法在分布中做出大的对角移动。

### 温度采样（Temperature Sampling，用于大语言模型）

语言模型为词汇表中的每个 token 输出 logits z_1, ..., z_V。Softmax 将这些转换为概率。温度在 softmax 之前对 logits 进行缩放：

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0：标准 softmax（原始分布）
T -> 0： argmax（确定性，永远选最高 logit）
T -> inf：均匀分布（所有 token 等概率）
T < 1.0：锐化分布（更自信，多样性更低）
T > 1.0：拉平分布（更不自信，多样性更高）
```

**为什么有效。** 将 logits 除以 T < 1 放大了 logits 之间的差异。如果 z_1 = 2，z_2 = 1，除以 T = 0.5 得到 z_1/T = 4 和 z_2/T = 2，差距变大。经过 softmax 后，最高 logit 的 token 获得了大得多的份额。

**在实践中：**
- T = 0.0：贪婪解码（Greedy Decoding），最适合事实性问答
- T = 0.3-0.7：稍带创意，适合代码生成
- T = 0.7-1.0：平衡，适合一般对话
- T = 1.0-1.5：创意写作、头脑风暴
- T > 1.5：越来越随机，很少有用

温度不会改变哪些 token 是可能的。它改变分配给每个 token 的概率质量。

### Top-k 采样

Top-k 采样将候选集限制为概率最高的 k 个 token，然后重新归一化并从该受限集中采样。

```
算法：
  1. 计算所有 V 个 token 的 softmax 概率
  2. 按概率降序排序
  3. 只保留前 k 个 token
  4. 重新归一化：p_i' = p_i / sum(top-k 中所有 p_j)
  5. 从重新归一化的分布中采样

k = 1：  贪婪解码
k = V：  无过滤（标准采样）
k = 40： 典型设置，移除长尾中的低概率 token
```

Top-k 防止模型选择词汇分布长尾中存在的极端低概率 token（拼写错误、毫无意义的词）。问题在于：k 是固定的，与上下文无关。当模型很自信（某个 token 概率为 95%）时，k = 40 仍然允许 39 个备选。当模型不确定（概率分布在 1000 个 token 上）时，k = 40 会切掉合理的选项。

### Top-p（核）采样（Nucleus Sampling）

Top-p 采样动态调整候选集大小。它不是保留固定数量的 token，而是保留累积概率超过 p 的最小 token 集合。

```
算法：
  1. 计算所有 V 个 token 的 softmax 概率
  2. 按概率降序排序
  3. 找到最小的 k，使得 top-k 概率之和 >= p
  4. 只保留这 k 个 token
  5. 重新归一化并采样

p = 0.9：保留覆盖 90% 概率质量的 token
p = 1.0：无过滤
p = 0.1：非常严格，近乎贪婪
```

当模型自信时，核采样只保留很少的 token（可能 2-3 个）。当模型不确定时，它保留很多（可能 200 个）。这种自适应行为是核采样通常比 top-k 生成更好文本的原因。

**常见组合：**
- 温度 0.7 + top-p 0.9：好的通用设置
- 温度 0.0（贪婪）：最适合确定性任务
- 温度 1.0 + top-k 50：Fan et al. (2018) 原论文的设置

Top-k 和 top-p 可以组合使用。先应用 top-k，再在剩余集合上应用 top-p。

### 重参数化技巧（Reparameterization Trick，用于 VAE）

变分自编码器（VAE，Variational Autoencoder）通过将输入编码为隐空间中的分布、从该分布采样、再将样本解码回来进行学习。问题在于：你不能对采样操作进行反向传播。

```
标准采样（不可微）：
  z ~ N(mu, sigma^2)

  随机性阻断了梯度流。
  d/d_mu [从 N(mu, sigma^2) 中采样] = ???
```

重参数化技巧将随机性与参数分开：

```
重参数化采样：
  epsilon ~ N(0, 1)          （固定的随机噪声，不含参数）
  z = mu + sigma * epsilon   （参数的确定性函数）

  现在 z 是 mu 和 sigma 的确定性、可微函数。
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  梯度可以穿过 mu 和 sigma 流动。
```

这之所以有效，是因为 N(mu, sigma^2) 与 mu + sigma * N(0, 1) 具有相同的分布。关键洞见：将随机性移到一个不依赖参数的源头（epsilon），然后将采样表示为参数的可微变换。

**在 VAE 训练循环中：**
1. 编码器为每个输入输出 mu 和 log(sigma^2)
2. 采样 epsilon ~ N(0, 1)
3. 计算 z = mu + sigma * epsilon
4. 解码 z 以重建输入
5. 通过步骤 4、3、2、1 反向传播（可行，因为步骤 3 是可微的）

没有重参数化技巧，VAE 无法用标准反向传播训练。这一洞见让 VAE 变得真正可用。

### Gumbel-Softmax（可微分类采样）

重参数化技巧适用于连续分布（高斯）。对于离散分类分布，我们需要不同的方法。Gumbel-Softmax 提供了一种对分类采样的可微近似。

**Gumbel-Max 技巧（不可微）：**

```
要从具有对数概率 log(p_1), ..., log(p_k) 的分类分布中采样：
  1. 为每个类别采样 g_i ~ Gumbel(0, 1)
     （g = -log(-log(u))，其中 u ~ Uniform(0, 1)）
  2. 返回 argmax(log(p_i) + g_i)

这产生精确的分类样本。
```

**Gumbel-Softmax（可微近似）：**

```
将硬 argmax 替换为软 softmax：
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau（温度）控制近似程度：
  tau -> 0：  趋近于 one-hot 向量（硬分类）
  tau -> inf：趋近于均匀分布 (1/k, 1/k, ..., 1/k)
  tau = 1.0： 软近似
```

Gumbel-Softmax 产生离散样本的连续松弛。输出是一个概率向量（软 one-hot）而不是硬 one-hot。梯度通过 softmax 流动。在训练的前向传播中，你可以使用"直通"（Straight-Through）估计器：前向传播用硬 argmax，但反向传播用软 Gumbel-Softmax 的梯度。

**应用：**
- VAE 中的离散隐变量
- 神经架构搜索（选择离散操作）
- 硬注意力机制
- 离散动作的强化学习

### 分层采样（Stratified Sampling）

标准蒙特卡洛采样可能偶然在样本空间中留下空隙。分层采样通过将空间划分为层（strata）并从每一层中采样，强制均匀覆盖。

```
标准蒙特卡洛：
  从 [0, 1] 中均匀采样 N 个点
  某些区域可能有聚集，另一些区域可能有空隙

分层采样：
  将 [0, 1] 划分为 N 个等长的层：[0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  在每层内均匀采样一个点
  x_i = (i + u_i) / N   其中 u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

与标准蒙特卡洛相比，分层采样总是具有更低或相等的方差：

```
Var(分层) <= Var(标准蒙特卡洛)

当 f(x) 变化平滑时，改进最大。
对于分段常值函数，分层采样是精确的。
```

**应用：**
- 数值积分（拟蒙特卡洛，Quasi-Monte Carlo）
- 训练数据划分（确保每折中类别平衡）
- 带分层的重要性采样（结合两种技术）
- NeRF（神经辐射场，Neural Radiance Fields）在相机射线上使用分层采样

### 与扩散模型的联系（Connection to Diffusion Models）

扩散模型通过采样过程生成图像。前向过程在 T 步内向图像添加高斯噪声，直到变成纯噪声。反向过程学习去噪，逐步恢复原始图像。

```
前向过程（已知）：
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  其中 epsilon ~ N(0, I)

  T 步之后：x_T ~ N(0, I)  （纯噪声）

反向过程（学习得到）：
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  其中 z ~ N(0, I)

  每个去噪步骤都是一次采样步骤。
```

与本课各方法的联系：
- 每个去噪步骤使用重参数化技巧（采样噪声，应用确定性变换）
- 噪声调度 {alpha_t} 控制一种温度退火形式
- 训练使用蒙特卡洛估计来近似 ELBO（证据下界，Evidence Lower Bound）
- 扩散模型中的祖先采样（Ancestral Sampling）是一条马尔可夫链（每步仅依赖当前状态）

整个图像生成过程是迭代采样：从噪声开始，在每一步根据学到的去噪模型，采样一个略少噪声的版本。

## 动手实现

### 第 1 步：均匀采样与逆 CDF 采样

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

生成 10,000 个指数分布样本，验证均值是否为 1/lambda。

### 第 2 步：拒绝采样

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

用拒绝采样从截断正态分布中抽取样本。通过直方图验证形状。

### 第 3 步：重要性采样

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

使用均匀提议分布，估计正态分布下的 E[X^2]。与已知答案 (mu^2 + sigma^2) 比较。

### 第 4 步：用蒙特卡洛估计 pi

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### 第 5 步：Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

从双峰分布（两个高斯混合）中采样。可视化链的轨迹。

### 第 6 步：Gibbs 采样

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### 第 7 步：温度采样

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

展示温度如何改变一组 token logits 的输出分布。

### 第 8 步：Top-k 和 top-p 采样

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### 第 9 步：重参数化技巧

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

演示梯度如何穿过重参数化样本流动，但无法穿过直接采样的样本。

### 第 10 步：Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

展示降低温度如何使输出趋近于 one-hot 向量。

完整实现及所有可视化见 `code/sampling.py`。

## 实际使用

使用 NumPy 和 SciPy 的生产级版本：

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

对于大规模 MCMC，使用专用库：
- PyMC：完整贝叶斯建模，带 NUTS（自适应 HMC）
- emcee：集成 MCMC 采样器
- NumPyro/JAX：GPU 加速 MCMC

你从零构建了这些。现在你知道了库调用背后到底在做什么。

## 交付物

本课产出：
- `code/sampling.py` —— 包含逆 CDF、拒绝采样、重要性采样、MCMC、温度/top-k/top-p 采样及重参数化技巧的完整从零实现
- `outputs/prompt-sampling-tutor.md` —— 一个给 AI 助手的提示词，用来通过直觉讲解各种采样方法

## 联系

| 概念 | 出现在哪里 |
|---------|------------------|
| 逆 CDF | 标准库中的随机数生成器；从任意具有已知 CDF 的分布中生成样本 |
| 拒绝采样 | 低维自定义分布采样；为截断分布生成样本 |
| 重要性采样 | PPO/TRPO 中使用重要性权重对 off-policy 轨迹进行重新加权；off-policy 评估 |
| 蒙特卡洛估计 | VAE 中的 ELBO 估计；扩散模型中的训练损失；强化学习中的策略梯度 |
| Metropolis-Hastings | 贝叶斯推断中的后验探索；物理模拟中的采样 |
| Gibbs 采样 | 主题模型（LDA）；贝叶斯网络的推断；Ising 模型 |
| 温度采样 | GPT/Claude/Llama 中的每个文本生成调用；控制创造力的旋钮 |
| Top-k / Top-p 采样 | 防止大语言模型生成低概率 token（垃圾输出）；工业标准的解码策略 |
| 重参数化技巧 | VAE 训练使反向传播穿过采样操作；变分推断中必不可少的技巧 |
| Gumbel-Softmax | 离散隐变量 VAE；神经架构搜索；可微离散决策 |
| 分层采样 | NeRF 射线采样；拟蒙特卡洛积分；平衡数据集划分 |

扩散模型值得专门说一下。Stable Diffusion、DALL-E 和 Midjourney 从纯噪声开始，通过迭代采样步骤逐步生成图像。每一步都是条件采样操作：给定带噪图像和时间步 t，去噪模型预测噪声，然后采样一个略微干净的版本。整个从噪声到图像的旅程就是一条由迭代采样驱动的马尔可夫链。你在本课构建的 Metropolis-Hastings 和重参数化技巧，正是这些模型底层采样机制的核心要素。

## 练习

1. 为 Cauchy 分布实现逆 CDF 采样。CDF 为 F(x) = 0.5 + arctan(x)/pi。生成 10,000 个样本，画出直方图与真实 PDF 对比。注意重尾现象（远离中心的极端值）。

2. 使用拒绝采样从 Beta(2, 5) 分布生成样本，使用 Uniform(0, 1) 提议。将接受的样本与真实 Beta PDF 对比绘图。理论接受率是多少？

3. 使用蒙特卡洛方法分别用 1,000、10,000 和 100,000 个样本估计 sin(x) 从 0 到 pi 的积分。比较各水平的误差。验证误差按 O(1/sqrt(N)) 缩放。

4. 实现 Metropolis-Hastings 从二维分布 p(x, y) ∝ exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2) 中采样。绘制样本和链的轨迹。尝试不同的提议标准差。

5. 构建完整的文本生成演示：给定一个有 10 个词的词汇表及 logits，使用 (a) 贪婪、(b) 温度=0.7、(c) top-k=3、(d) top-p=0.9 生成长度为 20 的 token 序列。运行 5 次，比较输出的多样性。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|----------------------|
| 采样（Sampling） | "随机抽取数值" | 按照概率分布生成值。所有生成式 AI 背后的机制 |
| 均匀分布（Uniform distribution） | "所有情况等可能" | [a, b] 中每个值的概率密度均为 1/(b-a)。所有采样方法的起点 |
| 逆 CDF（Inverse CDF） | "概率变换" | F_inverse(U) 将均匀样本转换为具有已知 CDF 的任意分布样本。精确且高效 |
| 拒绝采样（Rejection sampling） | "提议然后接受/拒绝" | 从简单提议中生成，以与目标/提议比值成比例的概率接受。精确但浪费样本 |
| 重要性采样（Importance sampling） | "重新加权样本" | 使用来自 q(x) 的样本估计 p(x) 下的期望，通过将每个样本乘以 p(x)/q(x) 加权。RL 中 PPO 的核心 |
| 蒙特卡洛（Monte Carlo） | "对随机样本取平均" | 将积分近似为样本均值。无论维度多高，误差均为 O(1/sqrt(N)) |
| MCMC | "收敛的随机游走" | 构造一条马尔可夫链，其平稳分布是目标分布。Metropolis-Hastings 是基础算法 |
| Metropolis-Hastings | "上坡接受，下坡有时接受" | 提议移动，基于密度比决定接受与否。细致平衡确保收敛到目标分布 |
| Gibbs 采样（Gibbs sampling） | "一次一个变量" | 从每个变量的条件分布中采样，保持其他变量不动。100% 接受率 |
| 温度（Temperature） | "置信度旋钮" | 在 softmax 之前将 logits 除以 T。T<1 锐化（更自信），T>1 拉平（更多样） |
| Top-k 采样 | "保留最好的 k 个" | 将除概率最高的 k 个 token 外的所有 token 置零，重新归一化，采样。固定候选集大小 |
| 核采样 / Top-p（Nucleus sampling） | "保留那些概率高的" | 保留累积概率超过 p 的最小 token 集合。自适应候选集大小 |
| 重参数化技巧（Reparameterization trick） | "把随机性移到外面" | 将 z = mu + sigma * epsilon 写出，其中 epsilon ~ N(0,1)。使采样可微。VAE 训练的核心 |
| Gumbel-Softmax | "软分类采样" | 使用 Gumbel 噪声 + 带温度的 softmax 对分类采样进行可微近似 |
| 分层采样（Stratified sampling） | "强制覆盖" | 将采样空间划分为层，从每层中采样。方差始终低于朴素蒙特卡洛 |
| 预热期（Burn-in） | "热身阶段" | 在链达到平稳分布之前丢弃的初始 MCMC 样本 |
| 细致平衡（Detailed balance） | "可逆性条件" | p(x) * T(x->y) = p(y) * T(y->x)。p 成为马尔可夫链平稳分布的充分条件 |
| 扩散采样（Diffusion sampling） | "迭代去噪" | 从噪声开始，通过应用学到的去噪步骤来生成数据。每一步都是条件采样操作 |

## 进一步阅读

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - MCMC 基础的详细教程
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - Gumbel-Softmax 原始论文
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - 核（top-p）采样论文
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - VAE 论文，引入了重参数化技巧
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM 将采样与图像生成联系起来
