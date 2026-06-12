# Flow Matching 与 Rectified Flows

> 扩散模型需要 20-50 个采样步骤，因为它们走的是从噪声到数据的曲线路径。Flow matching（Lipman 等，2023）和 rectified flow（Liu 等，2022）训练直线路径。更直的路径意味着更少的步骤意味着更快的推理。Stable Diffusion 3、Flux.1 和 AudioCraft 2 在 2024 年都切换到了 flow matching。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 8 · 06（DDPM）、阶段 1 · 微积分
**时间：** 约 45 分钟

## 问题

DDPM 的反向过程是从 `N(0, I)` 回到数据分布的 1000 步随机行走。DDIM 将其压缩到 20-50 步确定性步骤。你想要更少的步骤——最好是一步。阻碍是解反过程的 ODE 是刚性的；路径是弯曲的。

如果你能训练模型使得从噪声到数据的路径是一条*直线*，那么从 `t=1` 到 `t=0` 的一步 Euler 就能工作。Flow matching 直接构建这一点：定义从 `x_1 ~ N(0, I)` 到 `x_0 ~ data` 的直线插值，训练向量场 `v_θ(x, t)` 来匹配其时间导数，然后在推理时积分。

Rectified flow（Liu 2022）更进一步：通过一种重流程序逐步拉直路径，产生逐渐接近线性的 ODE。经过两次重流迭代，2 步采样器就能达到 50 步 DDPM 的质量。

## 概念

![Flow matching：噪声和数据之间的直线路径插值](../assets/flow-matching.svg)

### 直线路径流

定义：

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

其中 `x_0 ~ data` 且 `x_1 ~ N(0, I)`。沿这条直线的时间导数是常数：

```
dx_t / dt = x_1 - x_0
```

定义一个神经向量场 `v_θ(x_t, t)` 并训练它来匹配这个导数：

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

这是**条件流匹配**损失（Lipman 2023）。训练是无模拟的：你永远不会展开 ODE。只需采样 `(x_0, x_1, t)` 并回归。

### 采样

在推理时，*逆向*时间积分学习到的向量场：

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

从 `x_1 ~ N(0, I)` 开始，Euler 步进到 `t=0`。

### Rectified flow（Liu 2022）

直线路径流有效，但学习到的路径*实际上不是直的*——它们是弯曲的，因为许多 `x_0` 可以映射到同一个 `x_1`。Rectified flow 的重流步骤：

1. 用随机配对训练流模型 v_1。
2. 通过从 `x_1` 积分到其着陆点 `x_0` 来采样 N 对 `(x_1, x_0)`。
3. 在这些配对样本上训练 v_2。因为配对现在是"ODE 匹配的"，它们之间的直线路线插值确实更平坦。
4. 重复。

实际上 2 次重流迭代就能让你接近线性，实现 2-4 步推理。SDXL-Turbo、SD3-Turbo、LCM 都是从 flow matching 蒸馏出来的模型。

### 为什么这在 2024 年的图像领域胜出

三个原因：

1. **无模拟训练**——训练期间无需 ODE 展开，实现简单。
2. **更好的损失几何**——直线路径具有一致的信噪比，而 DDPM ε-损失在调度边缘有糟糕的 SNR。
3. **更快的推理**——在 SDXL-Turbo 质量下只需 4-8 步；通过一致性蒸馏只需 1 步。

## Flow matching 与 DDPM 的精确联系

带有高斯条件路径的 Flow matching 是扩散*带有特定噪声调度*。选择 `x_t = α(t) x_0 + σ(t) x_1` 调度，flow matching 通过 `v = α'·x_0 - σ'·x_1` 恢复 Stratonovich 重构的扩散。对于高斯路径，两者是代数等价的。

Flow matching 增加的是：目标的*清晰度*（一个简单的速度）、更干净的损失，以及尝试非高斯插值的许可。

## 动手实现

`code/main.py` 在双模态高斯混合上实现一维 flow matching。向量场 `v_θ(x, t)` 是一个小型 MLP，用直线路径目标训练。在推理时，积分 1、2、4 和 20 个 Euler 步并比较样本质量。

### 第 1 步：训练损失

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update
```

### 第 2 步：多步推理

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### 第 3 步：比较步数

期望 4 步采样器已经能匹配 20 步的质量——这对于延迟来说是一件大事。

## 陷阱

- **时间参数化。** Flow matching 使用 `t ∈ [0, 1]`，其中 `t=0` 在数据处，`t=1` 在噪声处。DDPM 使用 `t ∈ [0, T]`，其中 `t=0` 在数据处，`t=T` 在噪声处。方向相同，尺度不同。论文经常搞错这个。
- **调度选择。** Rectified flow 的直线是"那个"flow matching 调度，但你可以使用余弦或 logit-normal t 采样（SD3 这样做）来获得更好的尺度覆盖。
- **重流成本。** 为重流生成配对数据集需要对每个样本进行完整推理传递。只在你真的需要 1-2 步推理时才做重流。
- **无分类器引导仍然适用。** 只需在线性组合中将 ε 换成 v：`v_cfg = (1+w) v_cond - w v_uncond`。

## 实际使用

| 用例 | 2026 年技术栈 |
|----------|-----------|
| 文本转图像，最佳质量 | Flow matching：SD3、Flux.1-dev |
| 文本转图像，1-4 步 | 蒸馏 flow matching：Flux.1-schnell、SD3-Turbo、SDXL-Turbo |
| 实时推理 | 从 flow-matched 基础进行一致性蒸馏（LCM、PCM） |
| 音频生成 | Flow matching：Stable Audio 2.5、AudioCraft 2 |
| 视频生成 | Flow matching 与扩散混合（Sora、Veo、Stable Video） |
| 科学 / 物理（粒子轨迹、分子） | Flow matching + 等变向量场 |

当一篇论文在 2025-2026 年说"比扩散更快"时，几乎总是 flow matching + 蒸馏。

## 交付物

保存 `outputs/skill-fm-tuner.md`。技能接受一个扩散风格模型规范并将其转换为 flow matching 训练配置：调度选择、时间采样分布（均匀 / logit-normal）、优化器、重流计划、目标步数、评估协议。

## 练习

1. **简单。** 运行 `code/main.py` 并比较 1 步 vs 20 步 MSE 与真实数据分布的对比。
2. **中等。** 将 t 采样从均匀分布切换到 logit-normal（集中在 mid-t）。模型质量有改善吗？
3. **困难。** 实现一次重流迭代：通过积分第一个模型生成配对 (x_0, x_1)，在配对上训练第二个模型，并比较 1 步样本质量。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| Flow matching | "直线路径扩散" | 训练 `v_θ(x, t)` 沿插值路径匹配 `x_1 - x_0`。 |
| Rectified flow | "重流" | 逐步拉直学习流的迭代过程。 |
| 速度场 | "v_θ" | 模型的输出——移动 `x_t` 的方向。 |
| 直线路径插值 | "路径" | `x_t = (1-t)·x_0 + t·x_1`；简单的目标导数。 |
| Euler 采样器 | "一阶 ODE 求解器" | 最简单的积分器；当路径是直线时效果很好。 |
| Logit-normal t | "SD3 采样" | 将 t 采样集中在中间值附近，那里的梯度最强。 |
| 一致性蒸馏 | "1 步采样器" | 训练一个学生将任何 `x_t` 直接映射到 `x_0`。 |
| 带速度的 CFG | "v-CFG" | `v_cfg = (1+w) v_cond - w v_uncond`；同样的技巧，新的变量。 |

## 生产注意事项：Flux.1-schnell 是 flow matching 的最快形态

Flow matching 的生产突破是 Flux.1-schnell——一个被蒸馏到 1-4 推理步骤的 flow-matched DiT，同时保持 Flux-dev 级质量。Niels 的"在 8GB 机器上运行 Flux"笔记是参考部署方案：T5 + CLIP 编码、量化 MMDiT 去噪（schnell 4 步 vs dev 50 步）、VAE 解码。成本核算：

| 变体 | 步数 | L4 上 1024² 延迟 | 总 FLOPs（相对） |
|---------|-------|------------------------|------------------------|
| Flux.1-dev（原始） | 50 | 约 15 秒 | 1.0× |
| Flux.1-schnell | 4 | 约 1.2 秒 | 0.08×（快 12 倍） |
| SDXL-base | 30 | 约 4 秒 | 0.25× |
| SDXL-Lightning 2 步 | 2 | 约 0.3 秒 | 0.03× |

生产规则：**flow-matched 基础 + 蒸馏 = 2026 年快速文本转图像的默认方案。** 每个主要厂商都提供这个组合：SD3-Turbo（SD3 + flow + 蒸馏）、Flux-schnell（Flux-dev + rectified-flow 拉直）、CogView-4-Flash。纯扩散基础仅用于遗留检查点。

## 延伸阅读

- [Liu、Gong、Li（2022）。Flow Straight and Fast：学习生成和迁移具有 Rectified Flow 的数据](https://arxiv.org/abs/2209.03003) — rectified flow。
- [Lipman 等（2023）。用于生成式建模的 Flow Matching](https://arxiv.org/abs/2210.02747) — flow matching。
- [Esser 等（2024）。用于高分辨率图像合成的缩放 Rectified Flow Transformers](https://arxiv.org/abs/2403.03206) — SD3，在规模上应用 rectified flow。
- [Albergo、Vanden-Eijnden（2023）。随机插值](https://arxiv.org/abs/2303.08797) — 覆盖 FM + 扩散的一般框架。
- [Song 等（2023）。一致性模型](https://arxiv.org/abs/2303.01469) — 扩散 / flow 的一次性蒸馏。
- [Sauer 等（2023）。对抗性扩散蒸馏（SDXL-Turbo）](https://arxiv.org/abs/2311.17042) — turbo 变体。
- [Black Forest Labs（2024）。Flux.1 模型](https://blackforestlabs.ai/announcing-black-forest-labs/) — 生产中的 flow matching。