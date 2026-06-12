# 自编码器与变分自编码器（VAE）

> 普通的自编码器压缩后重建。它会记忆。它不能生成。添加一个技巧 — 强制代码看起来像高斯分布 — 你就得到了一个采样器。那一个技巧，即 `z = μ + σ·ε` 的重参数化，就是为什么 2026 年你使用的每一个潜在扩散和流匹配图像模型在输入端都有一个 VAE。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 3·02（反向传播）、阶段 3·07（CNN）、阶段 8·01（分类法）
**时间：** 约 75 分钟

## 问题

把一个 784 像素的 MNIST 数字压缩为一个 16 数编码，然后重建。普通的自编码器会在重建 MSE 上表现优异，但代码空间是一团乱码。在代码空间中随机选一个点，解码它，你得到的是噪声。它没有采样器。它只是一个伪装成压缩模型的生成模型。

你真正想要的是：（a）代码空间是一个你可以从中采样的干净平滑分布 — 比如各向同性高斯分布 `N(0, I)`，（b）解码任何样本都能产生一个可信的数字，（c）编码器和解码器仍然压缩良好。三个目标，一种架构，一个损失。

Kingma 的 2013 VAE 通过训练编码器输出一个*分布* `q(z|x) = N(μ(x), σ(x)²)` 来解决这一问题，通过 KL 惩罚把这个分布拉向先验 `N(0, I)`，然后在解码前从 `q(z|x)` 中采样 `z`。在推理时，丢掉编码器，从 `z ~ N(0, I)` 采样，解码。KL 惩罚是迫使代码空间结构化的原因。

在 2026 年，VAE 很少单独发布 — 它们在原始图像质量上已被扩散超越 — 但它们是每个潜在扩散模型（SD 1/2/XL/3、Flux、AudioCraft）的编码器选择。学习 VAE，你就学习了每个你使用的图像管道中看不见的第一层。

## 概念

![自编码器 vs VAE：重参数化技巧](../assets/vae.svg)

**自编码器。** `z = encoder(x)`，`x̂ = decoder(z)`，损失 = `||x - x̂||²`。代码空间无结构。

**VAE 编码器。** 输出两个向量：`μ(x)` 和 `log σ²(x)`。它们定义 `q(z|x) = N(μ, diag(σ²))`。

**重参数化技巧。** 从 `q(z|x)` 采样不可微分。把采样重写为 `z = μ + σ·ε`，其中 `ε ~ N(0, I)`。现在 `z` 是 `(μ, σ)` 的确定性函数加上一个非参数噪声 — 梯度通过 `μ` 和 `σ` 流动。

**损失。** 证据下界（ELBO），两项：

```
loss = 重建 + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

重建把 `x̂` 推向 `x`。KL 把 `q(z|x)` 推向先验。它们相互权衡。小 β（<1）= 更锐利的样本，代码空间不够高斯。大 β（>1）= 更干净的代码空间，更模糊的样本。β-VAE（Higgins 2017）使这个旋钮声名鹊起，并开启了 disentanglement 研究。

**采样。** 推理时：从 `z ~ N(0, I)` 抽取，前向通过解码器。一次前向传播 — 没有像扩散那样的迭代采样。

## 构建它

`code/main.py` 实现了一个没有 numpy 或 torch 的微型 VAE。输入是从 8-D 中双成分高斯混合中抽取的 8 维合成数据。编码器和解码器是单隐藏层 MLP。我们实现 tanh 激活、前向传播、损失和手写反向传播。不是生产级 — 是教学级。

### 第 1 步：编码器前向

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

用 `log σ²` 而不是 `σ`，这样网络输出是无约束的（σ 的 softplus 是一个陷阱 — 梯度在 σ ≈ 0 时死亡）。

### 第 2 步：重参数化并解码

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### 第 3 步：ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

精确闭式 KL，因为两个分布都是高斯分布。不要数值积分。人们在 2026 年仍在发布带有蒙特卡洛 KL 估计的代码 — 它慢 3 倍而没有任何原因。

### 第 4 步：生成

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

这就是生成模型。五行代码。

## 陷阱

- **后验崩溃。** KL 项如此激进地驱动 `q(z|x) → N(0, I)` 以至于 `z` 不携带关于 `x` 的任何信息。修复：β 退火（从 β=0 开始，渐变到 1）、自由位或在非活跃维度上跳过 KL。
- **样本模糊。** 高斯解码器似然意味着 MSE 重建，它是 L2（均值）的贝叶斯最优 — 一组可信数字的均值是一张模糊的数字。修复：离散解码器（VQ-VAE、NVAE），或者只把 VAE 用作编码器并在潜在变量上堆叠扩散（这就是 Stable Diffusion 做的）。
- **β 太大、太早。** 见后验崩溃。从 β≈0.01 开始渐变。
- **潜在维数太小。** 16-D 适用于 MNIST，256-D 适用于 ImageNet 256²，2048-D 适用于 ImageNet 1024²。Stable Diffusion 的 VAE 把 512×512×3 → 64×64×4（空间面积下采样 32 倍，通道 32 倍）。

## 使用它

2026 年 VAE 技术栈：

| 情况 | 选择 |
|-----------|------|
| 扩散的图像潜在编码器 | Stable Diffusion VAE（`sd-vae-ft-ema`）或 Flux VAE |
| 音频潜在编码器 | Encodec（Meta）、SoundStream 或 DAC（Descript） |
| 视频潜在变量 | Sora 的时空 patch、Latent VAE、WAN VAE |
| Disentangled 表示学习 | β-VAE、FactorVAE、TCVAE |
| 离散潜在变量（用于 transformer 建模）| VQ-VAE、RVQ（ResidualVQ）|
| 用于生成的连续潜在变量 | 纯 VAE，然后在该潜在空间中调节一个流/扩散模型 |

潜在扩散模型是一个在编码器和解码器之间生活着扩散模型的 VAE。VAE 做粗压缩，扩散模型做重活。视频（VAE + 视频扩散 DiT）和音频（Encodec + MusicGen transformer）也是同样的模式。

## 交付它

保存 `outputs/skill-vae-trainer.md`。

技能接收：数据集概况 + 目标潜在维数 + 下游用途（重建、采样或潜在扩散输入），并输出：架构选择（纯/β/VQ/RVQ）、β 调度、潜在维数、解码器似然（高斯 vs 分类），以及评估计划（重建 MSE、每维 KL、`q(z|x)` 与 `N(0, I)` 之间的 Fréchet 距离）。

## 练习

1. **简单。** 把 `code/main.py` 中的 `β` 改为 `0.01`、`0.1`、`1.0`、`5.0`。记录最终的重建 MSE 和 KL。哪个 β 对你的合成数据是帕累托最优的？
2. **中等。** 用伯努利似然（交叉熵损失）替换高斯解码器似然。在相同合成数据的二值化版本上比较样本质量。
3. **困难。** 把 `code/main.py` 扩展为 mini VQ-VAE：用 K=32 条目的码本中的最近邻查找替换连续 `z`。比较重建 MSE 并报告使用了多少码本条目（码本崩溃是真实存在的）。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 自编码器 | 编码器-解码器网络 | `x → z → x̂`，学习 MSE。不是生成式的。 |
| VAE | 带采样器的 AE | 编码器输出一个分布，KL 惩罚塑造代码空间。 |
| ELBO | 证据下界 | `log p(x) ≥ recon - KL[q(z|x) || p(z)]`；当 `q = p(z|x)` 时紧。 |
| 重参数化 | `z = μ + σ·ε` | 把随机节点重写为确定性 + 纯噪声。使得可以通过采样反向传播。 |
| 先验 | `p(z)` | 潜在变量的目标分布，通常是 `N(0, I)`。 |
| 后验崩溃 | "KL 项赢了" | 编码器忽略 `x`，输出先验；解码器必须幻觉。 |
| β-VAE | 可调 KL 权重 | `loss = recon + β·KL`。更高的 β = 更解耦但更模糊。 |
| VQ-VAE | 离散潜在变量 | 用最近的码本向量替换连续 `z`；支持 transformer 建模。 |

## 生产笔记：VAE 是扩散服务器中最热的路径

在 Stable Diffusion / Flux / SD3 管道中，VAE 每个请求被调用两次 — 一次编码（如果做 img2img / inpainting）和一次解码。在 1024² 时，解码器通道通常是整个管道中最大的激活内存峰值，因为它把 `128×128×16` 潜在变量上采样回 `1024×1024×3`。两个实际后果：

- **对解码进行切片或平铺。** `diffusers` 暴露了 `pipe.vae.enable_slicing()` 和 `pipe.vae.enable_tiling()`。平铺用一点接缝伪影换取 `O(tile²)` 内存而不是 `O(H·W)`。在消费级 GPU 上处理 1024²+ 时必不可少的。
- **解码器用 bf16，最终 resize 用 fp32 数值。** SD 1.x VAE 以 fp32 发布，在 1024² 时转换为 fp16 时*悄无声息地产生 NaN*。SDXL 发布了 `madebyollin/sdxl-vae-fp16-fix` — 始终优先使用 fp16-fix 变体或使用 bf16。

## 延伸阅读

- [Kingma & Welling（2013）。自动编码变分贝叶斯](https://arxiv.org/abs/1312.6114) — VAE 论文。
- [Higgins 等人（2017）。β-VAE：使用约束变分框架学习基本视觉概念](https://openreview.net/forum?id=Sy2fzU9gl) — disentangled β-VAE。
- [van den Oord 等人（2017）。神经离散表示学习](https://arxiv.org/abs/1711.00937) — VQ-VAE。
- [Vahdat & Kautz（2021）。NVAE：一种深度层次变分自编码器](https://arxiv.org/abs/2007.03898) — 图像 VAE 的最新技术。
- [Rombach 等人（2022）。使用潜在扩散模型的高分辨率图像合成](https://arxiv.org/abs/2112.10752) — Stable Diffusion；VAE 作为编码器。
- [Défossez 等人（2022）。高保真神经音频压缩](https://arxiv.org/abs/2210.13438) — Encodec，音频 VAE 标准。