# 随机过程

> 有结构的随机性。支撑随机游走、马尔可夫链和扩散模型的数学基础。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 1，第 06-07 课（概率、贝叶斯）
**时间：** 约 75 分钟

## 学习目标

- 模拟 1D 和 2D 随机游走（random walk），并验证位移随 sqrt(n) 增长的缩放规律
- 构建马尔可夫链（Markov chain）模拟器，通过特征分解计算其平稳分布（stationary distribution）
- 实现 Metropolis-Hastings MCMC 和朗之万动力学（Langevin dynamics），从目标分布中采样
- 将前向扩散过程与布朗运动（Brownian motion）联系起来，并解释反向过程如何生成数据

## 问题

许多 AI 系统都涉及随时间演化的随机性。不是静态的随机性，而是有结构的、顺序的随机性，每一步都依赖于之前的状态。

语言模型一次生成一个 token。每个 token 依赖于前面的上下文。模型输出一个概率分布，从中采样，然后继续前进。这就是一个随机过程（stochastic process）。

扩散模型（diffusion model）逐步向图像添加噪声，直到它变成纯静态，然后反转这个过程，一步步去噪，直到一张新图像出现。前向过程是一个马尔可夫链，反向过程是一个学到的、反向运行的马尔可夫链。

强化学习智能体在环境中采取动作，每个动作以一定概率导向一个新状态。智能体在随机的世界中遵循随机的策略。整个过程是一个马尔可夫决策过程（Markov decision process，MDP）。

MCMC 采样 —— 贝叶斯推断的基石 —— 构造一个马尔可夫链，使其平稳分布正好是你想从中采样的后验分布。

所有这些都建立在四个基本思想之上：
1. 随机游走 —— 最简单的随机过程
2. 马尔可夫链 —— 有转移矩阵的结构化随机性
3. 朗之万动力学 —— 带噪声的梯度下降
4. Metropolis-Hastings —— 从任意分布中采样

## 概念

### 随机游走

从位置 0 出发。每一步抛一枚公平硬币。正面：向右走 (+1)。反面：向左走 (-1)。

n 步之后，你的位置是 n 个随机 ±1 值的和。期望位置是 0（游走是无偏的），但到起点的期望距离却随着 sqrt(n) 增长。

这反直觉。游走是公平的 —— 没有任何方向上的漂移。但时间一长，它会离起点越来越远。n 步后的标准差是 sqrt(n)。

```
第 0 步：   位置 = 0
第 1 步：   位置 = +1 或 -1
第 2 步：   位置 = +2、0 或 -2
...
第 100 步： 到起点的期望距离 ≈ 10   (sqrt(100))
第 10000 步：到起点的期望距离 ≈ 100 (sqrt(10000))
```

**在 2D 中，** 游走以等概率向上、下、左、右移动。同样的 sqrt(n) 缩放规律适用于到起点的距离。路径描绘出类似分形的图案。

**为什么是 sqrt(n)？** 每一步以等概率取 +1 或 -1。n 步后，位置 S_n = X_1 + X_2 + ... + X_n，其中每个 X_i 是 ±1。每一步的方差是 1，各步相互独立，因此 Var(S_n) = n。标准差 = sqrt(n)。根据中心极限定理，S_n / sqrt(n) 收敛于标准正态分布。

这个 sqrt(n) 缩放规律在 ML 中随处可见。SGD 的噪声缩放为 1/sqrt(batch_size)。嵌入维度的缩放为 sqrt(d)。平方根是独立随机累加的特征印记。

**与布朗运动的联系。** 取一个步长为 1/sqrt(n)、单位时间内有 n 步的随机游走。当 n 趋近无穷，该游走收敛到布朗运动 B(t) —— 一个连续时间过程，其中 B(t) 服从均值为 0、方差为 t 的正态分布。

布朗运动是扩散过程的数学基础。它描述了流体中粒子的随机颤动、股票价格的波动，以及 —— 至关重要的 —— 扩散模型中的噪声过程。

**赌徒破产问题（Gambler's ruin）。** 一个随机游走者从位置 k 出发，在 0 和 N 处有吸收壁（absorbing barrier）。在到达 0 之前到达 N 的概率是多少？对于公平游走：P(到达 N) = k/N。这个结果出奇地简洁和优美。它与鞅（martingale）理论相关 —— 公平随机游走是一个鞅（未来值的期望等于当前值）。

### 马尔可夫链

马尔可夫链是一个按照固定概率在状态之间转移的系统。其核心性质：下一个状态只取决于当前状态，与历史无关。

```
P(X_{t+1} = j | X_t = i, X_{t-1} = ...) = P(X_{t+1} = j | X_t = i)
```

这就是马尔可夫性质（Markov property）。它意味着你可以用一个转移矩阵（transition matrix）P 来描述整个动力学：

```
P[i][j] = 从状态 i 转移到状态 j 的概率
```

P 的每一行之和为 1（你必须去往某处）。

**示例 —— 天气：**

```
状态：晴天 (0), 雨天 (1), 多云 (2)

P = [[0.7, 0.1, 0.2],    （如果是晴天：70% 继续晴天，10% 转雨，20% 转多云）
     [0.3, 0.4, 0.3],    （如果是雨天：30% 转晴，40% 继续雨，30% 转多云）
     [0.4, 0.2, 0.4]]    （如果是多云：40% 转晴，20% 转雨，40% 继续多云）
```

从任意状态出发，经过足够多次转移之后，状态的分布会收敛到平稳分布（stationary distribution）π，满足 π * P = π。这是 P 的特征值为 1 的左特征向量。

对于天气链，平稳分布可能是 [0.53, 0.18, 0.29] —— 长期来看，无论从哪个状态开始，53% 的时间是晴天。

```mermaid
graph LR
    S["晴天"] -->|0.7| S
    S -->|0.1| R["雨天"]
    S -->|0.2| C["多云"]
    R -->|0.3| S
    R -->|0.4| R
    R -->|0.3| C
    C -->|0.4| S
    C -->|0.2| R
    C -->|0.4| C
```

**计算平稳分布。** 有两种方法：

1. **幂迭代法（Power method）**：将任意初始分布反复乘以 P。迭代足够多步后就会收敛。
2. **特征值法（Eigenvalue method）**：找到 P 的特征值为 1 的左特征向量。也就是 P^T 的特征值为 1 的特征向量。

两种方法都要求链满足收敛条件。

**收敛条件。** 马尔可夫链收敛到唯一的平稳分布，当它满足：
- **不可约（Irreducible）**：从任意状态都可以到达任意其他状态
- **非周期（Aperiodic）**：链不会以固定周期循环

你在 ML 中遇到的大多数链都同时满足这两个条件。

**吸收态（Absorbing state）。** 如果一个状态一旦进入就永远无法离开（P[i][i] = 1），它就是吸收态。吸收马尔可夫链建模了带有终止状态的过程 —— 会结束的游戏、会流失的客户、会遇到结束符的 token 序列。

**混合时间（Mixing time）。** 链需要多少步才能"接近"平稳分布？正式地定义，从初始分布到平稳分布的总变差距离降至某个阈值以下所需的步数。快速混合 = 所需步数很少。P 的谱间隙（1 减去第二大的特征值）控制着混合时间。间隙越大，混合越快。

### 与语言模型的联系

语言模型中的 token 生成近似为一个马尔可夫过程。给定当前上下文，模型输出下一个 token 的分布。温度（temperature）参数控制分布的尖锐程度：

```
P(token_i) = exp(logit_i / temperature) / sum(exp(logit_j / temperature))
```

- temperature = 1.0：标准分布
- temperature < 1.0：更尖锐（更确定性）
- temperature > 1.0：更平坦（更随机）
- temperature -> 0：argmax（贪心）

Top-k 采样截断到 k 个最高概率的 token。Top-p（核采样，nucleus sampling）截断到累积概率超过 p 的最小 token 集合。两者都在修改马尔可夫转移概率。

### 布朗运动

随机游走的连续时间极限。位置 B(t) 具有三个性质：
1. B(0) = 0
2. B(t) - B(s) 服从均值为 0、方差为 t - s 的正态分布（当 t > s 时）
3. 不重叠区间上的增量相互独立

布朗运动是连续的，但在任何点都不可微 —— 它在每个尺度上都颤动。其路径在平面上的分形维数为 2。

在离散模拟中，你这样近似布朗运动：

```
B(t + dt) = B(t) + sqrt(dt) * z,    其中 z ~ N(0, 1)
```

sqrt(dt) 的缩放是关键的，它来自中心极限定理在随机游走上的应用。

### 朗之万动力学

梯度下降找的是函数的最小值。朗之万动力学（Langevin dynamics）找的则是正比于 exp(-U(x)/T) 的概率分布，其中 U 是能量函数，T 是温度。

```
x_{t+1} = x_t - dt * gradient(U(x_t)) + sqrt(2 * T * dt) * z_t
```

两个力作用于粒子：
1. **梯度力**（-dt * gradient(U)）：推向低能量区域（类似梯度下降）
2. **随机力**（sqrt(2*T*dt) * z）：推向随机方向（探索）

温度 T = 0 时，这就是纯梯度下降。高温时，它几乎就是随机游走。在恰当的温度下，粒子探索能量景观，并在低能量区域花更多时间。

**与扩散模型的联系。** 扩散模型的前向过程是：

```
x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * noise
```

这是一个马尔可夫链，逐渐将数据与噪声混合。经过足够多步后，x_T 就是纯高斯噪声。

反向过程 —— 从噪声回到数据 —— 也是一个马尔可夫链，但其转移概率是由神经网络学习得到的。网络学习预测每一步中添加的噪声，然后减去它。

```mermaid
graph LR
    subgraph "前向过程（加噪）"
        X0["x_0 (数据)"] -->|"+ 噪声"| X1["x_1"]
        X1 -->|"+ 噪声"| X2["x_2"]
        X2 -->|"..."| XT["x_T (纯噪声)"]
    end
    subgraph "反向过程（去噪）"
        XT2["x_T (噪声)"] -->|"神经网络"| XR2["x_{T-1}"]
        XR2 -->|"神经网络"| XR1["x_{T-2}"]
        XR1 -->|"..."| XR0["x_0 (生成的数据)"]
    end
```

### MCMC：马尔可夫链蒙特卡洛

有时你需要从一个分布 p(x) 中采样，你能计算它的值（最多差一个常数因子），但无法直接采样。贝叶斯后验就是经典例子 —— 你知道似然乘以先验，但归一化常数无法计算。

**Metropolis-Hastings 算法**构造一个马尔可夫链，使其平稳分布为 p(x)：

1. 从某个位置 x 出发
2. 从提议分布（proposal distribution）Q(x'|x) 中提议一个新位置 x'
3. 计算接受比：a = p(x') * Q(x|x') / (p(x) * Q(x'|x))
4. 以概率 min(1, a) 接受 x'；否则留在 x
5. 重复

如果 Q 是对称的（例如 Q(x'|x) = Q(x|x') = N(x, sigma^2)），比率简化为 a = p(x') / p(x)。你只需要概率的比值 —— 归一化常数消掉了。

在温和条件下，链保证收敛到 p(x)。但如果提议步长太小（随机游走）或太大（高拒绝率），收敛可能会很慢。调整提议分布是 MCMC 的艺术所在。

**它为什么有效。** 接受比保证了细致平衡（detailed balance）：在 x 处并转移到 x' 的概率，等于在 x' 处并转移到 x 的概率。细致平衡意味着 p(x) 是链的平稳分布。因此经过足够多步之后，样本来自 p(x)。

**实践注意事项：**
- **预热期（Burn-in）**：丢弃前 N 个样本。链需要时间从起点到达平稳分布。
- **稀释（Thinning）**：每 k 个样本保留一个，以减少自相关性。
- **多链（Multiple chains）**：从不同的起点运行多条链。如果它们收敛到相同的分布，就是收敛的证据。
- **接受率（Acceptance rate）**：对于 d 维高斯提议分布，最优接受率约为 23%（Roberts & Rosenthal, 2001）。太高说明链几乎不动，太低说明几乎全部被拒绝。

### AI 中的随机过程

| 过程 | AI 应用 |
|---------|---------------|
| 随机游走 | 强化学习中的探索、Node2Vec 嵌入 |
| 马尔可夫链 | 文本生成、MCMC 采样 |
| 布朗运动 | 扩散模型（前向过程） |
| 朗之万动力学 | 基于分数的生成模型、SGLD |
| 马尔可夫决策过程 | 强化学习 |
| Metropolis-Hastings | 贝叶斯推断、后验采样 |

## 动手实现

### 第 1 步：随机游走模拟器

```python
import numpy as np

def random_walk_1d(n_steps, seed=None):
    rng = np.random.RandomState(seed)
    steps = rng.choice([-1, 1], size=n_steps)
    positions = np.concatenate([[0], np.cumsum(steps)])
    return positions


def random_walk_2d(n_steps, seed=None):
    rng = np.random.RandomState(seed)
    directions = rng.choice(4, size=n_steps)
    dx = np.zeros(n_steps)
    dy = np.zeros(n_steps)
    dx[directions == 0] = 1   # 向右
    dx[directions == 1] = -1  # 向左
    dy[directions == 2] = 1   # 向上
    dy[directions == 3] = -1  # 向下
    x = np.concatenate([[0], np.cumsum(dx)])
    y = np.concatenate([[0], np.cumsum(dy)])
    return x, y
```

1D 游走存储累加和。每一步是 +1 或 -1。n 步后，位置就是这些步的和。方差随 n 线性增长，因此标准差随 sqrt(n) 增长。

### 第 2 步：马尔可夫链

```python
class MarkovChain:
    def __init__(self, transition_matrix, state_names=None):
        self.P = np.array(transition_matrix, dtype=float)
        self.n_states = len(self.P)
        self.state_names = state_names or [str(i) for i in range(self.n_states)]

    def step(self, current_state, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        probs = self.P[current_state]
        return rng.choice(self.n_states, p=probs)

    def simulate(self, start_state, n_steps, seed=None):
        rng = np.random.RandomState(seed)
        states = [start_state]
        current = start_state
        for _ in range(n_steps):
            current = self.step(current, rng)
            states.append(current)
        return states

    def stationary_distribution(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.P.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        return np.abs(stationary)
```

平稳分布是 P 的特征值为 1 的左特征向量。我们通过对 P^T 求特征向量来找到它（转置将左特征向量变成右特征向量）。

### 第 3 步：朗之万动力学

```python
def langevin_dynamics(grad_U, x0, dt, temperature, n_steps, seed=None):
    rng = np.random.RandomState(seed)
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    for _ in range(n_steps):
        noise = rng.randn(*x.shape)
        x = x - dt * grad_U(x) + np.sqrt(2 * temperature * dt) * noise
        trajectory.append(x.copy())
    return np.array(trajectory)
```

梯度将 x 推向低能量区域，噪声防止它卡住。在平衡态下，样本的分布正比于 exp(-U(x)/temperature)。

### 第 4 步：Metropolis-Hastings

```python
def metropolis_hastings(target_log_prob, proposal_std, x0, n_samples, seed=None):
    rng = np.random.RandomState(seed)
    x = np.array(x0, dtype=float)
    samples = [x.copy()]
    accepted = 0
    for _ in range(n_samples - 1):
        x_proposed = x + rng.randn(*x.shape) * proposal_std
        log_ratio = target_log_prob(x_proposed) - target_log_prob(x)
        if np.log(rng.rand()) < log_ratio:
            x = x_proposed
            accepted += 1
        samples.append(x.copy())
    acceptance_rate = accepted / (n_samples - 1)
    return np.array(samples), acceptance_rate
```

算法提议一个新点，检查它是否有更高的概率（或以正比于比值的概率接受），然后重复。为了良好的混合效果，接受率应该在 23%-50% 左右。

## 实际使用

实践中，你使用成熟的库来实现这些算法，但理解其机制对调试和调参至关重要。

```python
import numpy as np

rng = np.random.RandomState(42)
walk = np.cumsum(rng.choice([-1, 1], size=10000))
print(f"Final position: {walk[-1]}")
print(f"Expected distance: {np.sqrt(10000):.1f}")
print(f"Actual distance: {abs(walk[-1])}")
```

### numpy 处理转移矩阵

```python
import numpy as np

P = np.array([[0.7, 0.1, 0.2],
              [0.3, 0.4, 0.3],
              [0.4, 0.2, 0.4]])

distribution = np.array([1.0, 0.0, 0.0])
for _ in range(100):
    distribution = distribution @ P

print(f"Stationary distribution: {np.round(distribution, 4)}")
```

将初始分布反复乘以 P。经过足够多次迭代后，无论从哪个起点出发，它都会收敛到平稳分布。这正是求主左特征向量的幂迭代法。

### 与真实框架的联系

- **PyTorch 扩散模型：** Hugging Face `diffusers` 中的 `DDPMScheduler` 实现了前向和反向马尔可夫链
- **NumPyro / PyMC：** 使用 MCMC（NUTS 采样器，对 Metropolis-Hastings 的改进）进行贝叶斯推断
- **Gymnasium（RL）：** 环境的 step 函数定义了一个马尔可夫决策过程

### 验证马尔可夫链收敛

```python
import numpy as np

P = np.array([[0.9, 0.1], [0.3, 0.7]])

eigenvalues = np.linalg.eigvals(P)
spectral_gap = 1 - sorted(np.abs(eigenvalues))[-2]
print(f"Eigenvalues: {eigenvalues}")
print(f"Spectral gap: {spectral_gap:.4f}")
print(f"Approximate mixing time: {1/spectral_gap:.1f} steps")
```

谱间隙告诉你链遗忘初始状态的速度。间隙为 0.2 意味着大约 5 步就能混合，间隙为 0.01 意味着大约需要 100 步。在运行长模拟之前务必检查这一点 —— 混合过慢的链会浪费计算资源。

## 交付物

本课产出：
- `outputs/prompt-stochastic-process-advisor.md` —— 一个提示词，帮助判断给定问题适用哪种随机过程框架

## 联系

| 概念 | 出现在哪里 |
|---------|------------------|
| 随机游走 | Node2Vec 图嵌入、强化学习中的探索 |
| 马尔可夫链 | LLM 中的 token 生成、MCMC 采样 |
| 布朗运动 | DDPM 前向扩散过程、基于 SDE 的模型 |
| 朗之万动力学 | 基于分数的生成模型、随机梯度朗之万动力学（SGLD） |
| 平稳分布 | MCMC 收敛目标、PageRank |
| Metropolis-Hastings | 贝叶斯后验采样、模拟退火 |
| 温度 | LLM 采样、强化学习中的 Boltzmann 探索、模拟退火 |
| 混合时间 | MCMC 收敛速度、谱间隙分析 |
| 吸收态 | 序列结束 token、强化学习中的终止状态 |
| 细致平衡 | MCMC 采样器的正确性保证 |

扩散模型值得特别关注。DDPM（Ho et al., 2020）定义了一个前向马尔可夫链：

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t) * x_{t-1}, beta_t * I)
```

其中 beta_t 是噪声调度（noise schedule）。经过 T 步后，x_T 近似为 N(0, I)。反向过程由一个神经网络参数化，用于预测噪声：

```
p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)
```

生成的每一步都是在学到的马尔可夫链中迈出的一步。理解了马尔可夫链，就理解了扩散模型如何以及为何能生成数据。

SGLD（随机梯度朗之万动力学，Stochastic Gradient Langevin Dynamics）将小批量梯度下降与朗之万噪声结合起来。不计算完整梯度，而是使用随机估计并加入校准噪声。随着学习率衰减，SGLD 从优化过渡到采样 —— 你免费获得了近似的贝叶斯后验样本。这是从神经网络中获取不确定性估计的最简单方法之一。

贯穿所有这些联系的核心洞察是：随机过程不仅仅是理论工具，它们是现代 AI 系统内部的计算机制。当你调节 LLM 的温度时，你在调整一个马尔可夫链。当你训练一个扩散模型时，你在学习反转一个类似布朗运动的过程。当你运行贝叶斯推断时，你在构造一个收敛到后验的链。

## 练习

1. **模拟 1000 次、每次 10000 步的随机游走。** 绘制最终位置的分布。验证它近似服从均值为 0、标准差为 sqrt(10000) = 100 的高斯分布。

2. **用马尔可夫链构建文本生成器。** 在一个小语料上训练：对每个词，统计它到下一个词的转移次数，构建转移矩阵。通过从链中采样来生成新的句子。

3. **使用 Metropolis-Hastings 实现模拟退火（simulated annealing）。** 从高温开始（几乎接受一切），逐渐降温（只接受改进）。用它来找一个具有许多局部极小值的函数的最小值。

4. **比较不同温度下的朗之万动力学。** 从双阱势 U(x) = (x^2 - 1)^2 中采样。低温下，样本聚集在一个阱中；高温下，它们散布在两个阱之间。找到链能跨阱混合的临界温度。

5. **实现前向扩散过程。** 从一个 1D 信号（例如正弦波）开始，在 100 步中以线性噪声调度逐步添加噪声。展示信号如何退化为纯噪声。然后实现一个简单的去噪器来反转该过程（即使是仅减去估计噪声的朴素方法也行）。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|----------------------|
| 随机游走（Random walk） | "抛硬币式的移动" | 每一步位置按随机增量变化的过程 |
| 马尔可夫性质（Markov property） | "无记忆" | 未来只取决于当前状态，与历史无关 |
| 转移矩阵（Transition matrix） | "概率表" | P[i][j] = 从状态 i 转移到状态 j 的概率 |
| 平稳分布（Stationary distribution） | "长期平均值" | 满足 π*P = π 的分布 —— 链的平衡态 |
| 布朗运动（Brownian motion） | "随机颤动" | 随机游走的连续时间极限，B(t) ~ N(0, t) |
| 朗之万动力学（Langevin dynamics） | "带噪声的梯度下降" | 结合确定性梯度和随机扰动的更新规则 |
| 马尔可夫链蒙特卡洛（MCMC） | "走向目标分布" | 构造一个马尔可夫链，使其平稳分布就是你想要的分布 |
| Metropolis-Hastings | "提议然后接受/拒绝" | 使用接受比确保收敛的 MCMC 算法 |
| 温度（Temperature） | "随机性旋钮" | 控制探索与利用之间权衡的参数 |
| 扩散过程（Diffusion process） | "噪声进，噪声出" | 前向：逐步加噪；反向：逐步去噪。用来生成数据 |

## 进一步阅读

- **Ho, Jain, Abbeel (2020)** —— "Denoising Diffusion Probabilistic Models." 开启扩散模型革命的 DDPM 论文。对前向和反向马尔可夫链做了清晰的推导。
- **Song & Ermon (2019)** —— "Generative Modeling by Estimating Gradients of the Data Distribution." 基于分数的方法，使用朗之万动力学进行采样。
- **Roberts & Rosenthal (2004)** —— "General state space Markov chains and MCMC algorithms." 关于 MCMC 何时以及为何有效的理论基础。
- **Norris (1997)** —— "Markov Chains." 标准教科书。涵盖收敛性、平稳分布和首中时间（hitting time）。
- **Welling & Teh (2011)** —— "Bayesian Learning via Stochastic Gradient Langevin Dynamics." 将 SGD 与朗之万动力学结合，实现可扩展的贝叶斯推断。
