# 权重初始化与训练稳定性

> 初始化错误，训练永远不会开始。初始化正确，50 层网络和 3 层一样流畅。

**类型：** 学习型
**语言：** Python
**前置条件：** 第 03.04 课（激活函数）、第 03.07 课（正则化）
**时间：** 约 90 分钟

## 学习目标

- 从零实现零初始化、随机初始化、Xavier/Glorot 和 Kaiming/He 初始化策略，并通过 50 层网络测量激活值幅度的变化
- 推导 Xavier 初始化为何使用 Var(w) = 2/(fan_in + fan_out)，Kaiming 为何使用 Var(w) = 2/fan_in
- 演示零初始化的对称问题，并解释为何仅靠随机缩放是不够的
- 根据激活函数匹配正确的初始化策略：sigmoid/tanh 用 Xavier，ReLU/GELU 用 Kaiming

## 问题

把所有权重初始化为零。什么都学不到。每个神经元计算相同的函数、接收相同的梯度、以相同方式更新。训练 10,000 个 epoch 后，你的 512 神经元隐藏层仍然是 512 个相同神经元的副本。你花了 512 个参数的钱，只得到了 1 个参数的效果。

把权重初始化得太大。激活值在网络中爆炸。到第 10 层，数值达到 1e15。到第 20 层，溢出到无穷大。梯度沿同一条路径反向传播。

从标准正态分布中随机初始化。3 层网络可以工作。50 层时，信号会收缩到零或炸到无穷大，取决于随机缩放是略微偏小还是略微偏大。"能用"和"坏了"之间的边界极薄。

权重初始化是深度学习中最容易被低估的决策。架构能发论文。优化器能上博客。初始化只能获得一个脚注。但一旦搞错，其他一切都不重要——你的网络在训练开始前就已经死了。

## 概念

### 对称问题

一层中的每个神经元都有相同的结构：输入乘以权重，加偏置，应用激活。如果所有权重都从相同的值开始（零是最极端的情况），每个神经元计算相同的输出。在反向传播期间，每个神经元接收相同的梯度。在更新步骤中，每个神经元以相同的量变化。

你被困住了。网络有数百个参数，但它们全部步调一致。这叫做对称，而随机初始化是暴力打破它的方法。每个神经元从权重空间中的不同点开始，所以每个学习不同的特征。

但"随机"是不够的。随机性的*尺度*决定了网络能否训练。

### 方差在层间的传播

考虑一层有 fan_in 个输入：

```
z = w1*x1 + w2*x2 + ... + w_n*x_n
```

如果每个权重 wi 来自方差为 Var(w) 的分布，每个输入 xi 的方差为 Var(x)，输出方差为：

```
Var(z) = fan_in * Var(w) * Var(x)
```

如果 Var(w) = 1 且 fan_in = 512，输出方差是输入方差的 512 倍。10 层后：512^10 = 1.2e27。你的信号爆炸了。

如果 Var(w) = 0.001，输出方差每层缩小 0.001 * 512 = 0.512。10 层后：0.512^10 = 0.00013。你的信号消失了。

目标：选择 Var(w) 使得 Var(z) = Var(x)。信号幅度在各层之间保持恒定。

### Xavier/Glorot 初始化

Glorot 和 Bengio（2010）为 sigmoid 和 tanh 激活函数推导出了解。为了在正向和反向传播中都保持方差恒定：

```
Var(w) = 2 / (fan_in + fan_out)
```

实践中，权重从以下分布中抽取：

```
w ~ Uniform(-limit, limit)  其中 limit = sqrt(6 / (fan_in + fan_out))
```

或者：

```
w ~ Normal(0, sqrt(2 / (fan_in + fan_out)))
```

这之所以有效，是因为 sigmoid 和 tanh 在零点附近大致线性，而正确初始化的激活值就落在这个区域。方差在数十层中保持稳定。

### Kaiming/He 初始化

ReLU 把一半的输出杀死（所有负值变成零）。有效的 fan_in 减半，因为平均一半的输入被清零。Xavier 初始化没有考虑到这一点——它低估了所需的方差。

He 等人（2015）调整了公式：

```
Var(w) = 2 / fan_in
```

权重从以下分布中抽取：

```
w ~ Normal(0, sqrt(2 / fan_in))
```

因子 2 补偿了 ReLU 清零一半激活值的影响。没有它，信号每层缩小约 0.5 倍。50 层后：0.5^50 = 8.8e-16。Kaiming 初始化防止了这个问题。

### Transformer 初始化

GPT-2 引入了一种不同的模式。残差连接将每个子层的输出加到其输入上：

```
x = x + sublayer(x)
```

每次加法都会增加方差。有 N 个残差层时，方差与 N 成比例增长。GPT-2 将残差层的权重缩放为 1/sqrt(2N)，其中 N 是层数。这保持了累积信号幅度的稳定。

Llama 3（405B 参数，126 层）使用类似的方案。没有这种缩放，残差流在 126 层注意力和前馈块中会无限增长。

```mermaid
flowchart TD
    subgraph "零初始化"
        Z1["层 1<br/>所有权重 = 0"] --> Z2["层 2<br/>所有神经元相同"]
        Z2 --> Z3["层 3<br/>仍然相同"]
        Z3 --> ZR["结果：1 个有效神经元<br/>无论宽度多大"]
    end

    subgraph "Xavier 初始化"
        X1["层 1<br/>Var = 2/(fan_in+fan_out)"] --> X2["层 2<br/>信号稳定"]
        X2 --> X3["层 50<br/>信号稳定"]
        X3 --> XR["结果：用 sigmoid/tanh<br/>可以训练"]
    end

    subgraph "Kaiming 初始化"
        K1["层 1<br/>Var = 2/fan_in"] --> K2["层 2<br/>信号稳定"]
        K2 --> K3["层 50<br/>信号稳定"]
        K3 --> KR["结果：用 ReLU/GELU<br/>可以训练"]
    end
```

### 50 层激活值幅度

```mermaid
graph LR
    subgraph "平均激活值幅度"
        direction LR
        L1["层 1"] --> L10["层 10"] --> L25["层 25"] --> L50["层 50"]
    end

    subgraph "结果"
        R1["随机 N(0,1)：第 5 层爆炸"]
        R2["随机 N(0,0.01)：第 10 层消失"]
        R3["Xavier + Sigmoid：第 50 层约 1.0"]
        R4["Kaiming + ReLU：第 50 层约 1.0"]
    end
```

### 选择正确的初始化

```mermaid
flowchart TD
    Start["用什么激活函数？"] --> Act{"激活函数类型？"}

    Act -->|"Sigmoid / Tanh"| Xavier["Xavier/Glorot<br/>Var = 2/(fan_in + fan_out)"]
    Act -->|"ReLU / Leaky ReLU"| Kaiming["Kaiming/He<br/>Var = 2/fan_in"]
    Act -->|"GELU / Swish"| Kaiming2["Kaiming/He<br/>(与 ReLU 相同)"]
    Act -->|"Transformer 残差"| GPT["按 1/sqrt(2N) 缩放<br/>N = 层数"]

    Xavier --> Check["验证：激活值幅度<br/>在各层保持在 0.5 到 2.0 之间"]
    Kaiming --> Check
    Kaiming2 --> Check
    GPT --> Check
```

## 动手实现

### 第 1 步：初始化策略

四种初始化权重矩阵的方法。每个返回一个列表的列表（2D 矩阵），有 fan_in 列和 fan_out 行。

```python
import math
import random


def zero_init(fan_in, fan_out):
    return [[0.0 for _ in range(fan_in)] for _ in range(fan_out)]


def random_init(fan_in, fan_out, scale=1.0):
    return [[random.gauss(0, scale) for _ in range(fan_in)] for _ in range(fan_out)]


def xavier_init(fan_in, fan_out):
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]


def kaiming_init(fan_in, fan_out):
    std = math.sqrt(2.0 / fan_in)
    return [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]
```

### 第 2 步：激活函数

我们需要 sigmoid、tanh 和 ReLU 来用各自对应的激活函数测试每种初始化策略。

```python
def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def tanh_act(x):
    return math.tanh(x)


def relu(x):
    return max(0.0, x)
```

### 第 3 步：穿越 50 层的前向传播

将随机数据通过一个深度网络，测量每层的平均激活值幅度。

```python
def forward_deep(init_fn, activation_fn, n_layers=50, width=64, n_samples=100):
    random.seed(42)
    layer_magnitudes = []

    inputs = [[random.gauss(0, 1) for _ in range(width)] for _ in range(n_samples)]

    for layer_idx in range(n_layers):
        weights = init_fn(width, width)
        biases = [0.0] * width

        new_inputs = []
        for sample in inputs:
            output = []
            for neuron_idx in range(width):
                z = sum(weights[neuron_idx][j] * sample[j] for j in range(width)) + biases[neuron_idx]
                output.append(activation_fn(z))
            new_inputs.append(output)
        inputs = new_inputs

        magnitudes = []
        for sample in inputs:
            magnitudes.append(sum(abs(v) for v in sample) / width)
        mean_mag = sum(magnitudes) / len(magnitudes)
        layer_magnitudes.append(mean_mag)

    return layer_magnitudes
```

### 第 4 步：实验

运行所有组合：零初始化、随机 N(0,1)、随机 N(0,0.01)、Xavier 加 sigmoid、Xavier 加 tanh、Kaiming 加 ReLU。打印关键层的幅度。

```python
def run_experiment():
    configs = [
        ("零初始化 + Sigmoid", lambda fi, fo: zero_init(fi, fo), sigmoid),
        ("随机 N(0,1) + ReLU", lambda fi, fo: random_init(fi, fo, 1.0), relu),
        ("随机 N(0,0.01) + ReLU", lambda fi, fo: random_init(fi, fo, 0.01), relu),
        ("Xavier + Sigmoid", xavier_init, sigmoid),
        ("Xavier + Tanh", xavier_init, tanh_act),
        ("Kaiming + ReLU", kaiming_init, relu),
    ]

    print(f"{'策略':<30} {'L1':>10} {'L5':>10} {'L10':>10} {'L25':>10} {'L50':>10}")
    print("-" * 80)

    for name, init_fn, act_fn in configs:
        mags = forward_deep(init_fn, act_fn)
        row = f"{name:<30}"
        for idx in [0, 4, 9, 24, 49]:
            val = mags[idx]
            if val > 1e6:
                row += f" {'爆炸':>10}"
            elif val < 1e-6:
                row += f" {'消失':>10}"
            else:
                row += f" {val:>10.4f}"
        print(row)
```

### 第 5 步：对称性演示

展示零初始化产生相同的神经元。

```python
def symmetry_demo():
    random.seed(42)
    weights = zero_init(2, 4)
    biases = [0.0] * 4

    inputs = [0.5, -0.3]
    outputs = []
    for neuron_idx in range(4):
        z = sum(weights[neuron_idx][j] * inputs[j] for j in range(2)) + biases[neuron_idx]
        outputs.append(sigmoid(z))

    print("\n对称性演示（4 个神经元，零初始化）：")
    for i, out in enumerate(outputs):
        print(f"  神经元 {i}：输出 = {out:.6f}")
    all_same = all(abs(outputs[i] - outputs[0]) < 1e-10 for i in range(len(outputs)))
    print(f"  所有相同：{all_same}")
    print(f"  有效参数：1（不是 {len(weights) * len(weights[0])})")
```

### 第 6 步：逐层幅度报告

打印 50 层激活值幅度的可视化条形图。

```python
def magnitude_report(name, magnitudes):
    print(f"\n{name}:")
    for i, mag in enumerate(magnitudes):
        if i % 5 == 0 or i == len(magnitudes) - 1:
            if mag > 1e6:
                bar = "X" * 50 + " 爆炸"
            elif mag < 1e-6:
                bar = "." + " 消失"
            else:
                bar_len = min(50, max(1, int(mag * 10)))
                bar = "#" * bar_len
            print(f"  层 {i+1:3d}：{bar} ({mag:.6f})")
```

## 实际使用

PyTorch 将这些作为内置函数提供：

```python
import torch
import torch.nn as nn

layer = nn.Linear(512, 256)

nn.init.xavier_uniform_(layer.weight)
nn.init.xavier_normal_(layer.weight)

nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

nn.init.zeros_(layer.bias)
```

当你调用 `nn.Linear(512, 256)` 时，PyTorch 默认为 Kaiming均匀初始化。这就是为什么大多数简单网络"开箱即用"——PyTorch 已经做出了正确的选择。但当你构建自定义架构或深度超过 20 层时，你需要理解正在发生的事情，并可能需要覆盖默认值。

对于 transformer，HuggingFace 模型通常在其 `_init_weights` 方法中处理初始化。GPT-2 的实现将残差投影缩放为 1/sqrt(N)。如果你从头构建一个 transformer，你需要自己添加这个。

## 交付物

本课产出：
- `outputs/prompt-init-strategy.md` —— 一个提示词，用于诊断权重初始化问题并推荐正确的策略

## 练习

1. 添加 LeCun 初始化（Var = 1/fan_in，专为 SELU 激活设计）。用 LeCun init + tanh 运行 50 层实验，与 Xavier + tanh 比较。

2. 实现 GPT-2 残差缩放：在添加到残差流之前将每层的输出乘以 1/sqrt(2*N)。用有缩放和无缩放运行 50 层，测量残差幅度增长有多快。

3.创建一个"初始化健康检查"函数，接收网络的层维度和激活函数类型，然后推荐正确的初始化，并在当前初始化会导致问题时发出警告。

4. 用 fan_in = 16 vs fan_in = 1024 运行实验。Xavier 和 Kaiming 适应 fan_in，但随机初始化不行。展示"能用"和"坏了"之间的差距如何随着层变大而扩大。

5. 实现正交初始化（生成随机矩阵，计算其 SVD，使用正交矩阵 U）。在 50 层 ReLU 网络上与 Kaiming 比较。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|----------------------|
| 权重初始化 | "随机设置起始权重" | 选择初始权重值的策略，决定了网络能否训练 |
| 对称打破 | "让神经元不同" | 使用随机初始化确保神经元学习不同的特征，而不是计算相同的函数 |
| Fan-in | "神经元的输入数量" | 输入连接的数量，决定了输入方差如何在加权和中被积累 |
| Fan-out | "神经元的输出数量" | 输出连接的数量，与反向传播期间保持梯度方差相关 |
| Xavier/Glorot 初始化 | "sigmoid 初始化" | Var(w) = 2/(fan_in + fan_out)，旨在保持 sigmoid 和 tanh 激活的方差 |
| Kaiming/He 初始化 | "ReLU 初始化" | Var(w) = 2/fan_in，考虑了 ReLU 清零一半激活值 |
| 方差传播 | "信号如何在层间增长或缩小" | 基于权重尺度对激活方差如何逐层变化的数学分析 |
| 残差缩放 | "GPT-2 的初始化技巧" | 按1/sqrt(2N) 缩放残差连接权重，以防止 N 个 transformer 层的方差增长 |
| 死网络 | "什么都训练不了" | 初始化不良导致所有梯度为零或所有激活值饱和的网络 |
| 激活值爆炸 | "数值趋于无穷大" | 当权重方差过高时，导致激活值幅度在层间指数增长 |

## 延伸阅读

- Glorot & Bengio，"理解训练深度前馈神经网络的困难"（2010）——原始 Xavier 初始化论文，包含方差分析
- He 等，"深入理解整流器"（2015）——为 ReLU 网络引入 Kaiming 初始化
- Radford 等，"语言模型是无监督多任务学习者"（2019）——GPT-2 论文，包含残差缩放初始化
- Mishkin & Matas，"你只需要一个好的初始化"（2016）——逐层单元方差初始化，一种经验性的分析公式替代方法
