# 扩散模型——DDPM 从零实现

> Ho、Jain、Abbeel（2020）给这个领域提供了一个它无法拒绝的配方。用一千个小步骤逐渐用噪声破坏数据。训练一个神经网络来预测噪声。在推理时逆转这个过程。今天，每一个主流的图像、视频、3D 和音乐模型都运行在这个循环上，可能还会有流匹配或一致性技巧叠加在上面。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 3 · 02（反向传播）、阶段 8 · 02（VAE）
**时间：** 约 75 分钟

## 问题

你想要一个 `p_data(x)` 的采样器。GAN 玩的是一个经常发散的 minimax 游戏。VAE 从高斯解码器产生模糊的样本。你真正想要的是一个训练目标，它满足：(a) 一个单一的稳定损失（无鞍点，无 minimax），(b) `log p(x)` 的下界（这样你就有了似然），以及 (c) 能匹配 SOTA 质量的样本。

Sohl-Dickstein 等人（2015）给出了一个理论答案：定义一个马尔可夫链 `q(x_t | x_{t-1})` 来逐渐添加高斯噪声，然后训练一个反向链 `p_θ(x_{t-1} | x_t)` 来去噪。Ho、Jain、Abbeel（2020）表明损失可以简化为一行——预测噪声——并整理了数学。2020 年这还是个 curiosity。2021 年它产生了 SOTA 的样本。2022 年它变成了 Stable Diffusion。2026 年它是基底层。

## 概念

![DDPM：前向噪声、反向去噪](../assets/ddpm.svg)

**前向过程 `q`。** 在 `T` 小步中添加高斯噪声。闭合形式——这就是数学可处理的原因——是累积步骤也是高斯分布：

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

其中 `α̅_t = ∏_{s=1..t} (1 - β_s)` 是一个 `β_t` 的调度表。选择 `β_t` 从 1e-4 到 0.02 在 T=1000 步上线性增长，`x_T` 近似于 `N(0, I)`。

**反向过程 `p_θ`。** 学习一个神经网络 `ε_θ(x_t, t)` 来预测被添加的噪声。给定 `x_t`，通过以下方式去噪：

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

其中 `σ_t` 要么是 `sqrt(β_t)` 要么是一个学习到的方差。表达式很丑但这只是代数——从后验 `q(x_{t-1} | x_t, x_0)` 求出 `x_{t-1}`，并用噪声预测估计替换 `x_0`。

**训练损失。**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

从数据中采样 `x_0`，选择一个随机 `t`，采样 `ε ~ N(0, I)`，通过闭合形式一次性计算带噪的 `x_t`，然后对噪声做回归。一个损失，无 minimax，无 KL，无重参数化技巧。

**采样。** 从 `x_T ~ N(0, I)` 开始。从 `t = T` 到 `1` 迭代反向步骤。完成。

## 为什么它有效

三个直觉：

1. **去噪容易；生成难。** 在 `t=T` 时，数据是纯噪声——网络要解决的是一个 trivial 的问题。在 `t=0` 时，网络只需要清理几个像素。在中间的 `t` 时，问题很难但网络从每个噪声水平通过相同权重接收到很多梯度。

2. **Score matching 的伪装。** Vincent（2011）证明了预测噪声等价于估计 `∇_x log q(x_t | x_0)`，即 *score*。反向 SDE 使用这个 score 来沿着密度梯度向上走——一个导向的随机游走，朝着高概率区域前进。

3. **ELBO 简化为简单 MSE。** 完整的变分下界在每个时间步都有一个 KL 项。使用 DDPM 的参数化，这些 KL 项简化为噪声预测上的 MSE，并带有特定的系数；Ho 丢弃了这些系数（称其为"简单"损失），质量却*提升*了。

## 构建它

`code/main.py` 实现了一个 1-D DDPM。数据是一个双模混合分布。"网络"是一个小型 MLP，接受 `(x_t, t)` 并输出预测的噪声。训练是单行损失。采样迭代反向链。

### 第 1 步：前向调度表（闭合形式）

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### 第 2 步：一次性采样 `x_t`

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### 第 3 步：一次训练步骤

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### 第 4 步：反向采样

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

对于一个 1-D 问题，40 个时间步和 24 个单元的 MLP，在约 200 个 epoch 后学习双模混合分布。

## 时间条件化

网络需要知道它正在对哪个时间步去噪。两个标准选项：

- **正弦嵌入。** 类似于 Transformer 的位置编码。`embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`。通过一个 MLP，广播到网络中。
- **FiLM / 组归一化条件化。** 将嵌入投影到每个块的逐通道 scale/bias（FiLM）。

我们的玩具代码使用正弦嵌入 → concat。生产级 U-Net 使用 FiLM。

## 陷阱

- **调度表很重要。** 线性 `β` 是 DDPM 默认值，但余弦调度表（Nichol & Dhariwal，2021）在相同计算量下给出更好的 FID。如果质量停滞就切换调度表。
- **时间步嵌入很脆弱。** 将原始 `t` 作为浮点数传递对玩具 1-D 有效，但对图像无效；始终使用适当的嵌入。
- **V-prediction 与 ε-prediction。** 对于狭窄区域（非常小或非常大的 t），`ε` 的信噪比很差。V-prediction（`v = α·ε - σ·x`）更稳定；SDXL、SD3 和 Flux 使用它。
- **无分类器引导。** 在推理时，计算条件和非条件的 `ε`，然后 `ε_cfg = (1 + w) · ε_cond - w · ε_uncond`，其中 `w ≈ 3-7`。在第 08 课中介绍。
- **1000 步太多了。** 生产使用 DDIM（20-50 步）、DPM-Solver（10-20 步）或蒸馏（1-4 步）。见第 12 课。

## 使用它

| 角色 | 2026 年典型技术栈 |
|------|-----------------------|
| 图像像素空间扩散（小、玩具） | DDPM + U-Net |
| 图像潜在扩散 | VAE 编码器 + U-Net 或 DiT（第 07 课） |
| 视频潜在扩散 | 时空 DiT（Sora、Veo、WAN） |
| 音频潜在扩散 | Encodec + 扩散 transformer |
| 科学（分子、蛋白质、物理） | 等变扩散（EDM、RFdiffusion、AlphaFold3） |

扩散是通用的生成骨干。流匹配（第 13 课）是在相同质量下通常在推理速度上获胜的 2024-2026 竞争者。

## 交付它

保存 `outputs/skill-diffusion-trainer.md`。技能接受数据集 + 计算预算并输出：调度表（线性/余弦/ sigmoid）、预测目标（ε/v/x）、步数、引导量、采样器家族以及评估协议。

## 练习

1. **简单。** 在 `code/main.py` 中将 T 从 40 改为 10。样本质量（输出视觉直方图）如何下降？在哪个 T 下双模结构崩溃？
2. **中等。** 从 ε-prediction 切换到 v-prediction。重新推导反向步骤。比较最终样本质量。
3. **困难。** 添加无分类器引导。用类别标签 `c ∈ {0, 1}` 做条件化，训练时 10% 的时间丢弃它，在采样时使用 `ε = (1+w)·ε_cond - w·ε_uncond`。在 `w = 0, 1, 3, 7` 时测量条件模命中率。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|-----------------------|
| 前向过程 | "添加噪声" | 固定的马尔可夫链 `q(x_t \| x_{t-1})`，破坏数据。 |
| 反向过程 | "去噪" | 学习到的链 `p_θ(x_{t-1} \| x_t)`，重建数据。 |
| β 调度表 | "噪声阶梯" | 每步方差；线性、余弦或 sigmoid。 |
| α̅ | "Alpha bar" | 累积乘积 `∏(1 - β)`；给出从 `x_0` 到 `x_t` 的闭合形式。 |
| 简单损失 | "噪声上的 MSE" | `\|\|ε - ε_θ(x_t, t)\|\|²`；所有变分推导都简化为这个。 |
| ε-prediction | "预测噪声" | 输出是添加的噪声；标准 DDPM。 |
| V-prediction | "预测速度" | 输出是 `α·ε - σ·x`；在 t 上有更好的条件化。 |
| DDPM | "那篇论文" | Ho 等人 2020；线性 β，1000 步，U-Net。 |
| DDIM | "确定性采样器" | 非马尔可夫采样器，20-50 步，相同训练目标。 |
| 无分类器引导 | "CFG" | 混合条件和非条件噪声预测以增强条件化。 |

## 生产笔记：扩散推理是一个步数问题

DDPM 论文运行 T=1000 个反向步。没有人会在生产中那样发货。每一个真实的推理栈都选择三种策略之一——每一种都干净地映射到生产文献中对"延迟来自哪里"的描述：

1. **更快的采样器，相同的模型。** DDIM（20-50 步）、DPM-Solver++（10-20）、UniPC（8-16）。反向循环的无缝替换；训练好的 `ε_θ` 权重纹丝不动。将延迟减少 20-50×。
2. **蒸馏。** 训练一个学生用更少的步数匹配老师：渐进式蒸馏（2 → 1）、一致性模型（任意 → 1-4）、LCM、SDXL-Turbo、SD3-Turbo。再减少 5-10× 延迟，需要重新训练。
3. **缓存和编译。** `torch.compile(unet, mode="reduce-overhead")`、TensorRT-LLM 的扩散后端、`xformers`/SDPA 注意力、bf16 权重。将每步延迟减少约 2×。与（1）和（2）叠加。

对于一个生产扩散服务器，预算对话与生产文献对 LLM 的描述相同：延迟是 `num_steps × step_cost + VAE_decode`，吞吐量是 `batch_size × (num_steps × step_cost)^-1`。TTFT 很小（一步）；等效的 TPOT 是完整响应时间，因为从用户角度来看图像生成是"一次完成"的。

## 延伸阅读

- [Sohl-Dickstein 等人（2015）。利用非平衡热力学进行深度无监督学习](https://arxiv.org/abs/1503.03585) —— 扩散论文，超越时代。
- [Ho、Jain、Abbeel（2020）。去噪扩散概率模型](https://arxiv.org/abs/2006.11239) —— DDPM。
- [Song、Meng、Ermon（2021）。去噪扩散隐式模型](https://arxiv.org/abs/2010.02502) —— DDIM，更少步数。
- [Nichol & Dhariwal（2021）。改进的 DDPM](https://arxiv.org/abs/2102.09672) —— 余弦调度表，学习方差。
- [Dhariwal & Nichol（2021）。扩散模型在图像合成上击败 GANs](https://arxiv.org/abs/2105.05233) —— 分类器引导。
- [Ho & Salimans（2022）。无分类器扩散引导](https://arxiv.org/abs/2207.12598) —— CFG。
- [Karras 等人（2022）。阐明基于扩散的生成模型的设计空间（EDM）](https://arxiv.org/abs/2206.00364) —— 统一 notation，最清晰的配方。