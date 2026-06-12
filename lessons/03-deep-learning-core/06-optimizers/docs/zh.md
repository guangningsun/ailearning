# 优化器

> 梯度下降告诉你往哪个方向移动。它不会告诉你该移动多远、移动多快。SGD 是一个指南针。Adam 是带交通数据的 GPS。

**类型：** 建构型
**语言：** Python
**前置条件：** 第 03.05 课（损失函数）
**时间：** 约 75 分钟

## 学习目标

- 从零实现 SGD、带动量的 SGD、Adam 和 AdamW 优化器
- 解释 Adam 的偏差校正如何补偿早期训练步骤中零初始化矩估计的问题
- 证明 AdamW 在相同任务上比 Adam + L2 正则化具有更好的泛化能力
- 根据架构类型（Transformer、CNN、GAN、微调）为优化器选择合适的默认超参数

## 问题

你已经计算了梯度。你知道权重 #4,721 应该减少 0.003 来降低损失。但 0.003 是什么单位？要乘以什么缩放因子？而且第 1 步和第 1,000 步的步长应该一样吗？

原始梯度下降在每一步对每个参数都应用相同的学习率：w = w - lr * gradient。但这会产生三个实际问题，让神经网络训练变得痛苦。

第一，振荡。损失地貌很少是光滑的碗状。它更像是一条狭长的山谷。梯度指向山谷的横向（陡峭方向），而不是沿山谷方向（平缓方向）。梯度下降在窄维上来回弹跳，而在有用的方向上只取得微小的进展。你见过这种情况：损失快速下降然后停滞，这不是因为模型收敛了，而是因为它在振荡。

第二，一个学习率适用于所有参数是错误的。有些权重需要大的更新（它们处于早期、欠拟合阶段）。有些权重需要微小的更新（它们已接近最优值）。适用于前者的学习率会毁掉后者，反之亦然。

第三，鞍点。在高维空间中，损失地貌有大片平坦区域，梯度接近于零。原始 SGD 以梯度的速度（基本上是零）缓慢穿过这些区域。模型看起来卡住了。其实没有卡住——它在一个平坦区域，另一侧有可用的下降路径。但 SGD 没有机制推动它穿过。

Adam 解决了这三个问题。它为每个参数维护两个运行平均值——梯度均值（动量，处理振荡）和梯度平方均值（自适应率，处理不同尺度）。结合前几步的偏差校正，它提供了一个单一的优化器，在默认超参数下适用于 80% 的问题。本课从零构建它，让你精确理解它在另外 20% 的情况下何时以及为何失效。

## 概念

### 随机梯度下降 (SGD)

最简单的优化器。在小批量上计算梯度，然后向反方向迈出一步。

```
w = w - lr * gradient
```

"随机"意味着你使用数据的随机子集（小批量）来估计梯度，而不是整个数据集。这种噪声实际上是有用的——它帮助逃离尖锐的局部最小值。但噪声也会导致振荡。

学习率是唯一的旋钮。太高的损失发散。太低训练永远也完不了。最优值取决于架构、数据、批量大小和当前训练阶段。对于现代网络的原始 SGD，典型值在 0.01 到 0.1 之间。但即使在一次训练运行中，最佳学习率也会变化。

### 动量

球滚下山的类比虽然老套但很准确。你不只是按梯度步进，而是维护一个累积过去梯度的速度。

```
m_t = beta * m_{t-1} + gradient
w = w - lr * m_t
```

Beta（通常为 0.9）控制保留多少历史。当 beta = 0.9 时，动量大约是最近 10 个梯度的平均值（1 / (1 - 0.9) = 10）。

这为什么能解决振荡：指向相同方向的梯度会累积。方向翻转的梯度会相互抵消。在那条狭长的山谷中，"横向"分量每步翻转符号并被衰减。"纵向"分量保持一致并被放大。结果是在有用方向上的平滑加速。

真实数据：在病态损失地貌上，单独的 SGD 可能需要 10,000 步。带动量的 SGD（beta=0.9）在同一问题上通常需要 3,000-5,000 步。加速效果不是边际的。

### RMSProp

第一个真正起作用逐参数自适应学习率方法。由 Hinton 在 Coursera 课程中提出（从未正式发表）。

```
s_t = beta * s_{t-1} + (1 - beta) * gradient^2
w = w - lr * gradient / (sqrt(s_t) + epsilon)
```

s_t 跟踪梯度平方的运行平均值。具有持续大梯度的参数除以一个大数（更小的有效学习率）。具有小梯度的参数除以一个小数（更大的有效学习率）。

这解决了"一个学习率适用于所有参数"的问题。一个一直在获得大更新的权重可能已接近目标——让它慢下来。一个一直在获得微小更新的权重可能训练不足——让它快起来。

Epsilon（通常为 1e-8）防止参数未更新时除以零。

### Adam：动量 + RMSProp

Adam 结合了两种思想。它为每个参数维护两个指数移动平均值：

```
m_t = beta1 * m_{t-1} + (1 - beta1) * gradient        (一阶矩：均值)
v_t = beta2 * v_{t-1} + (1 - beta2) * gradient^2       (二阶矩：方差)
```

**偏差校正**是大多数解释都跳过的关键细节。在第 1 步，m_1 = (1 - beta1) * gradient。当 beta1 = 0.9 时，那是 0.1 * gradient——小了十倍。移动平均还没有热身。偏差校正弥补了这一点：

```
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
```

在 beta1 = 0.9 的第 1 步：m_hat = m_1 / (1 - 0.9) = m_1 / 0.1 = 实际梯度。在第 100 步：(1 - 0.9^100) 约为 1.0，所以校正消失。偏差校正在前约 10 步很重要，50 步以后就无关紧要了。

更新公式：

```
w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
```

Adam 默认值：lr = 0.001，beta1 = 0.9，beta2 = 0.999，epsilon = 1e-8。这些默认值适用于 80% 的问题。当不适用时，先改 lr。然后改 beta2。几乎从不改 beta1 或 epsilon。

### AdamW：正确的权重衰减

L2 正则化将 lambda * w^2 加入损失。在原始 SGD 中，这等价于权重衰减（每步从权重中减去 lambda * w）。在 Adam 中，这种等价性打破了。

Loshchilov & Hutter 的洞察：当你把 L2 加入损失然后 Adam 处理梯度时，自适应学习率也会缩放正则化项。具有大梯度方差的参数得到更少的正则化。具有小方差的参数得到更多。这不是你想要的——你希望无论梯度统计如何都有均匀的正则化。

AdamW 通过在 Adam 更新之后直接将权重衰减应用于权重来修复这个问题：

```
w = w - lr * m_hat / (sqrt(v_hat) + epsilon) - lr * lambda * w
```

权重衰减项（lr * lambda * w）不被 Adam 的自适应因子缩放。每个参数得到相同的比例收缩。

这看起来是个小细节。其实不是。AdamW 几乎在每个任务上都比 Adam + L2 正则化收敛到更好的解决方案。它是 PyTorch 中训练 Transformer、扩散模型和大多数现代架构的默认优化器。BERT、GPT、LLaMA、Stable Diffusion——都是用 AdamW 训练的。

### 学习率：最重要的超参数

```mermaid
graph TD
    LR["学习率"] --> TooHigh["太高 (lr > 0.01)"]
    LR --> JustRight["刚刚好"]
    LR --> TooLow["太低 (lr < 0.00001)"]

    TooHigh --> Diverge["损失爆炸<br/>NaN 权重<br/>训练崩溃"]
    JustRight --> Converge["损失稳步下降<br/>达到好的最小值<br/>泛化良好"]
    TooLow --> Stall["损失下降缓慢<br/>困在次优最小值<br/>浪费算力"]

    JustRight --> Schedule["通常需要调度"]
    Schedule --> Warmup["预热：从 0 线性上升到最大值<br/>训练的前 1-10%"]
    Schedule --> Decay["衰减：随时间减少<br/>余弦或线性"]
```

如果你只调一个超参数，就调学习率。学习率 10 倍的变化比你做出的任何架构决策都更重要。常见默认值：

- SGD：lr = 0.01 到 0.1
- Adam/AdamW：lr = 1e-4 到 3e-4
- 微调预训练模型：lr = 1e-5 到 5e-5
- 学习率预热：前 1-10% 步线性上升

### 优化器对比

```mermaid
flowchart LR
    subgraph "优化路径"
        SGD_P["SGD<br/>在山谷中振荡<br/>慢但能找到平坦最小值"]
        Mom_P["SGD + 动量<br/>更平滑的路径<br/>比 SGD 快 3 倍"]
        Adam_P["Adam<br/>逐参数自适应<br/>快速收敛"]
        AdamW_P["AdamW<br/>Adam + 正确的衰减<br/>最佳泛化"]
    end
    SGD_P --> Mom_P --> Adam_P --> AdamW_P
```

### 各优化器的适用场景

```mermaid
flowchart TD
    Task["你在训练什么？"] --> Type{"模型类型？"}

    Type -->|"Transformer / LLM"| AdamW["AdamW<br/>lr=1e-4, wd=0.01-0.1"]
    Type -->|"CNN / ResNet"| SGD_M["SGD + 动量<br/>lr=0.1, momentum=0.9"]
    Type -->|"GAN"| Adam2["Adam<br/>lr=2e-4, beta1=0.5"]
    Type -->|"微调"| AdamW2["AdamW<br/>lr=2e-5, wd=0.01"]
    Type -->|"还不确定"| Default["从 AdamW 开始<br/>lr=3e-4, wd=0.01"]
```

## 动手实现

### 第 1 步：原始 SGD

```python
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
```

### 第 2 步：带动量的 SGD

```python
class SGDMomentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocities = None

    def step(self, params, grads):
        if self.velocities is None:
            self.velocities = [0.0] * len(params)
        for i in range(len(params)):
            self.velocities[i] = self.beta * self.velocities[i] + grads[i]
            params[i] -= self.lr * self.velocities[i]
```

### 第 3 步：Adam

```python
import math

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)
```

### 第 4 步：AdamW

```python
class AdamW:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)
            params[i] -= self.lr * self.weight_decay * params[i]
```

### 第 5 步：训练对比

用四种优化器在第 05 课的圆形数据集上训练同一个双层网络。比较收敛情况。

```python
import random

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))

def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


class OptimizerTestNetwork:
    def __init__(self, optimizer, hidden_size=8):
        random.seed(0)
        self.hidden_size = hidden_size
        self.optimizer = optimizer

        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def get_params(self):
        params = []
        for row in self.w1:
            params.extend(row)
        params.extend(self.b1)
        params.extend(self.w2)
        params.append(self.b2)
        return params

    def set_params(self, params):
        idx = 0
        for i in range(self.hidden_size):
            for j in range(2):
                self.w1[i][j] = params[idx]
                idx += 1
        for i in range(self.hidden_size):
            self.b1[i] = params[idx]
            idx += 1
        for i in range(self.hidden_size):
            self.w2[i] = params[idx]
            idx += 1
        self.b2 = params[idx]

    def forward(self, x):
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(max(0.0, z))

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def compute_grads(self, target):
        eps = 1e-15
        p = max(eps, min(1 - eps, self.out))
        d_loss = -(target / p) + (1 - target) / (1 - p)
        d_sigmoid = self.out * (1 - self.out)
        d_out = d_loss * d_sigmoid

        grads = [0.0] * (self.hidden_size * 2 + self.hidden_size + self.hidden_size + 1)
        idx = 0
        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            d_h = d_out * self.w2[i] * d_relu
            grads[idx] = d_h * self.x[0]
            grads[idx + 1] = d_h * self.x[1]
            idx += 2

        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            grads[idx] = d_out * self.w2[i] * d_relu
            idx += 1

        for i in range(self.hidden_size):
            grads[idx] = d_out * self.h[i]
            idx += 1

        grads[idx] = d_out
        return grads

    def train(self, data, epochs=300):
        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for x, y in data:
                pred = self.forward(x)
                grads = self.compute_grads(y)
                params = self.get_params()
                self.optimizer.step(params, grads)
                self.set_params(params)

                eps = 1e-15
                p = max(eps, min(1 - eps, pred))
                total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            avg_loss = total_loss / len(data)
            accuracy = correct / len(data) * 100
            losses.append((avg_loss, accuracy))
            if epoch % 75 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")
        return losses
```

## 实际使用

PyTorch 优化器处理参数组、梯度裁剪和学习率调度：

```python
import torch
import torch.optim as optim

model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10),
)

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(torch.randn(32, 784))
    loss = torch.nn.functional.cross_entropy(output, torch.randint(0, 10, (32,)))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
```

模式永远是：zero_grad、forward、loss、backward、（clip）、step、（schedule）。记住这个顺序。弄错顺序（例如在 optimizer.step() 之前调用 scheduler.step()）是微妙 bug 的常见来源。

对于 CNN，许多从业者仍然偏好 SGD + 动量（lr=0.1，momentum=0.9，weight_decay=1e-4）配合阶梯或余弦调度。SGD 找到更平坦的最小值，这通常泛化更好。对于 Transformer 和 LLM，带预热 + 余弦衰减的 AdamW 是通用默认设置。没有充分测量的理由不要反对共识。

## 交付物

本课产出：
- `outputs/prompt-optimizer-selector.md`——一个决策提示词，用于为任何架构选择正确的优化器和学习率

## 练习

1. 实现 Nesterov 动量：在"前瞻"位置（w - lr * beta * v）计算梯度，而不是在当前位置。比较在圆形数据集上与标准动量的收敛性。

2. 实现学习率预热调度：前 10% 训练步从 0 线性上升到 max_lr，然后余弦衰减到 0。用 Adam + 预热与不带预热的 Adam 训练。测量在圆形数据集上达到 90% 准确率需要多少个 epoch。

3. 在 Adam 训练期间跟踪每个参数的有效学习率。有效学习率是 lr * m_hat / (sqrt(v_hat) + eps)。绘制 10、50 和 200 步后有效学习率的分布。所有参数以相同的速度更新吗？

4. 实现梯度裁剪（按全局范数裁剪）。将最大梯度范数设置为 1.0。用高学习率（Adam 下 lr=0.01）有裁剪和无裁剪地训练。在 10 个随机种子中统计有多少次发散（损失变为 NaN）。

5. 在具有大权重的网络上比较 Adam 与 AdamW。将所有权重初始化为 [-5, 5] 范围内的随机值（比正常大得多）。用 weight_decay=0.1 训练 200 个 epoch。绘制两种优化器训练过程中权重的 L2 范数。AdamW 应该显示更快的权重收缩。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|----------------------|
| 学习率 | "步长" | 梯度更新上的标量乘数；训练中影响最大的超参数 |
| SGD | "基本梯度下降" | 随机梯度下降：在一个小批量上计算梯度后通过减去 lr * gradient 更新权重 |
| 动量 | "滚球类比" | 过去梯度的指数移动平均；衰减振荡并加速一致方向 |
| RMSProp | "自适应学习率" | 用近期梯度均方根除每个参数的梯度；使学习率均等化 |
| Adam | "默认优化器" | 结合动量（一阶矩）和 RMSProp（二阶矩），并对初始步骤进行偏差校正 |
| AdamW | "正确的 Adam" | 带解耦权重衰减的 Adam；将正则化直接应用于权重而不是通过梯度 |
| 偏差校正 | "运行平均值的预热" | 除以 (1 - beta^t) 以补偿 Adam 矩估计的零初始化 |
| 权重衰减 | "收缩权重" | 每步减去权重值的一部分；惩罚大权重的正则化器 |
| 学习率调度 | "随时间改变 lr" | 训练期间调整学习率的函数；预热 + 余弦衰减是现代默认设置 |
| 梯度裁剪 | "限制梯度范数" | 当梯度向量范数超过阈值时向下缩放；防止梯度爆炸更新 |

## 延伸阅读

- Kingma & Ba，"Adam：随机优化方法"（2014）——原始 Adam 论文，包含收敛性分析和偏差校正推导
- Loshchilov & Hutter，"解耦权重衰减正则化"（2017）——证明了 L2 正则化和权重衰减在 Adam 中并不等价，并提出了 AdamW
- Smith，"训练神经网络的学习率周期策略"（2017）——引入了 LR 范围测试和周期调度，消除了调优固定学习率的需要
- Ruder，"梯度下降优化算法综述"（2016）——对所有优化器变体最全面的综述，包含清晰的比较和直觉解释
