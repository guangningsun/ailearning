# GANs —— 生成器与判别器

> Goodfellow 在 2014 年的技巧是完全跳过密度估计。两个网络。一个生成假样本。一个识别它们。它们对抗，直到假样本与真实样本无法区分。这本不该有效。很多时候确实无效。但当它有效时，样本质量在窄领域内仍然是文献中最清晰的。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 3 · 02（反向传播）、阶段 3 · 08（优化器）、阶段 8 · 02（VAE）
**时间：** 约 75 分钟

## 问题

VAE 的样本模糊，因为它的 MSE 解码器损失是*均值图像*的贝叶斯最优解——而众多可信数字的均值就是一张模糊的数字。你需要的是一种奖励*可信度*而不是像素与某个目标的接近程度的损失。可信度没有闭合形式。你必须去学习它。

Goodfellow 的思路：训练一个分类器 `D(x)` 来区分真实图像和假图像。训练一个生成器 `G(z)` 来欺骗 `D`。`G` 的损失信号就是 `D` 当前认为某样本看起来真实的原因。这个信号随着 `G` 的提升而更新，追赶一个移动的目标。如果两个网络都收敛，`G` 就在从未写下 `log p(x)` 的情况下学会了数据分布。

这就是对抗训练。数学形式是一个最小最大博弈：

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

到了 2026 年，GAN 不再是生成模型的 SOTA（扩散模型和流匹配已经夺走了这顶王冠）。但 StyleGAN 2/3 仍然是已发布的锐度最高的人脸模型，GAN 判别器被用作扩散训练中的*感知损失*，对抗训练还驱动了高速单步蒸馏（SDXL-Turbo、SD3-Turbo、LCM），让你可以部署实时扩散。

## 概念

![GAN 训练：生成器和判别器的最小最大博弈](../assets/gan.svg)

**生成器 `G(z)`。** 将噪声向量 `z ~ N(0, I)` 映射为样本 `x̂`。一个类解码器的网络（密集层或转置卷积）。

**判别器 `D(x)`。** 将样本映射为一个标量概率（或分数）。真实 → 1，假 → 0。

**损失。** 两次交替更新：

- **训练 `D`：** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`。二元交叉熵，真实=1，假=0。
- **训练 `G`：** `loss_G = -log D(G(z))`。这是 Goodfellow 使用的*非饱和*形式（原始的 `log(1 - D(G(z)))` 会饱和并在 `D` 自信时杀死梯度）。

**训练循环。** 一步 `D`，一步 `G`。重复。

**为什么有效。** 如果 `G` 完美匹配 `p_data`，那么 `D` 最多只能做到随机猜测，在各处输出 0.5；`G` 不会再收到梯度。均衡。

**为什么失效。** 模式崩塌（`G` 找到一个 `D` 无法分类的模式并永远生成它）、梯度消失（`D` 学得太快，`log D` 饱和）、训练不稳定（学习率、批量大小、任何因素）。

## 让 GAN 有效工作的变体

| 年份 | 创新 | 解决方案 |
|------|------------|-----|
| 2015 | DCGAN | 卷积/转置卷积、批归一化、LeakyReLU——第一个稳定架构。 |
| 2017 | WGAN、WGAN-GP | 用 Wasserstein 距离 + 梯度惩罚替代 BCE。修复梯度消失。 |
| 2017 | 谱归一化 | 对判别器做 Lipschitz 约束。2026 年的判别器仍在使用。 |
| 2018 | Progressive GAN | 先训练低分辨率，逐步添加层。首个百万像素结果。 |
| 2019 | StyleGAN / StyleGAN2 | 映射网络 + 自适应实例归一化。固定领域逼真图像的 SOTA。 |
| 2021 | StyleGAN3 | 无别名、平移等变——2026 年仍是人脸金标准。 |
| 2022 | StyleGAN-XL | 条件式、类别感知、更大规模。 |
| 2024 | R3GAN | 以更强正则化重新品牌；无需 tricks 即可在 1024² 上工作。 |

## 构建它

`code/main.py` 在 1 维数据上训练一个小 GAN：两个高斯混合。生成器和判别器都是单隐藏层 MLP。我们手动实现前向、反向和最小最大循环。目标是观察两个关键失效模式（模式崩塌 + 梯度消失）在发生时是什么样子。

### 第 1 步：非饱和损失

原始 Goodfellow 损失 `log(1 - D(G(z)))` 当 D 以高置信度将 G 的假样本分类为假时趋近于 0。此时 G 的梯度基本为零——G 无法改进。非饱和形式 `-log D(G(z))` 有相反的渐近线：当 D 自信时它发散，给 G 一个强信号。

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### 第 2 步：每一步生成器对应一步判别器

```python
for step in range(steps):
    # 训练 D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # 训练 G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # 新的假样本
    update_G(fake_batch)
```

G 使用新的假样本，否则梯度会过时。

### 第 3 步：观察模式崩塌

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] 模式崩塌：一个模式被饿死了")
```

典型症状：两个真实模式中的一个停止被生成。判别器不再纠正它，因为它从未被视为假样本。

## 陷阱

- **判别器太强。** 将 D 的学习率降低 2-5 倍，或添加实例/层噪声。如果 D 达到 >95% 准确率，G 就死了。
- **生成器记住了一个模式。** 向 D 输入添加噪声，使用 minibatch-discriminator 层，或切换到 WGAN-GP。
- **批归一化泄露统计量。** 真实批次 + 假批次流经同一个 BN 层会混合它们的统计量。改用实例归一化或谱归一化。
- **Inception Score 作弊。** FID 和 IS 在低样本量时噪声很大。评估时使用 ≥10k 样本。
- **单步采样在条件任务中是谎言。** 你仍然需要 CFG 比例、截断技巧和重采样来获得可用的输出。

## 使用它

2026 年的 GAN 技术栈：

| 场景 | 选择 |
|-----------|------|
| 逼真人脸，固定姿态 | StyleGAN3（最锐利、最小） |
| 动漫/风格化人脸 | StyleGAN-XL 或 Stable Diffusion LoRA |
| 图像到图像翻译 | Pix2Pix / CycleGAN（阶段 8 · 04）或 ControlNet（阶段 8 · 08） |
| 高速单步文生图 | 扩散的对抗蒸馏（SDXL-Turbo、SD3-Turbo） |
| 扩散训练中的感知损失 | 图像裁剪上的小 GAN 判别器 |
| 任何多模态、开放式的任务 | 别用——用扩散或流匹配 |

GAN 锐利但狭窄。一旦你的领域打开——照片、任意文本提示、视频——切换到扩散。对抗技巧作为组件（感知损失、蒸馏）继续存在，而不是作为独立生成器。

## 交付它

保存 `outputs/skill-gan-debugger.md`。Skill 接收一个失败的 GAN 运行（损失曲线、样本网格、数据集大小）并输出按可能性排序的故障原因列表、一行修复方案和重新运行协议。

## 练习

1. **简单。** 用默认设置运行 `code/main.py`。然后设 `D_LR = 5 * G_LR` 并重新运行。G 的损失以多快的速度崩塌为常数？
2. **中等。** 将 Goodfellow BCE 损失替换为 WGAN 损失：`loss_D = E[D(fake)] - E[D(real)]`，`loss_G = -E[D(fake)]`，并将 D 的权重裁剪到 `[-0.01, 0.01]`。训练更稳定吗？比较时钟收敛时间。
3. **困难。** 将 1 维示例扩展到 2 维数据（环上 8 个高斯混合）。在步数 1k、5k、10k 时跟踪生成器捕获了多少个模式。实现 minibatch discrimination 并重新测量。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 生成器 | "G" | 噪声到样本的网络，`G: z → x̂`。 |
| 判别器 | "D" | 分类器 `D: x → [0, 1]`，真实 vs 假。 |
| 最小最大 | "这场博弈" | 联合目标的 `min_G max_D`。 |
| 非饱和损失 | "那个修复" | 对 G 使用 `-log D(G(z))` 而不是 `log(1 - D(G(z)))`。 |
| 模式崩塌 | "G 记住了一件事" | 尽管数据多样，生成器只产生少量不同输出。 |
| WGAN | "Wasserstein" | 用 Earth-Mover 距离 + 梯度惩罚替代 BCE；更平滑的梯度。 |
| 谱归一化 | "Lipschitz 技巧" | 约束 D 的权重范数以绑定其斜率；稳定训练。 |
| StyleGAN | "那个有效的" | 映射网络 + AdaIN；人脸领域的最佳，至今仍是 2026 年的标准。 |

## 生产注记：单步推理是 GAN 的持久优势

GAN 在开放领域生成的样本质量上不再领先，但在推理成本上仍然领先。在生产推理文献词汇中，GAN 具有：

- **无 prefill，无 decode 阶段。** 单一 `G(z)` 前向传播。TTFT ≈ 总延迟。
- **无 KV-cache 压力。** 唯一的状态就是权重。批量大小受激活内存限制，而非缓存。
- **平凡的连续批处理。** 由于每个请求消耗相同的固定 FLOPs，服务器目标occupancy的静态批量通常是最优的。不需要飞行内调度器。

这就是为什么 GAN 蒸馏（SDXL-Turbo、SD3-Turbo、ADD、LCM）是 2026 年快速文生图的主导技术：它将 20-50 步的扩散管道压缩为 1-4 步 GAN 风格前向传播，同时保持扩散基座的分布。对抗损失作为训练时旋钮存活下来，用于将慢生成器变成快生成器。

## 延伸阅读

- [Goodfellow 等 (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) —— 原始 GAN 论文。
- [Radford 等 (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) —— 第一个稳定架构。
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) —— WGAN。
- [Miyato 等 (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) —— SN。
- [Karras 等 (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) —— StyleGAN2。
- [Karras 等 (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) —— StyleGAN3。
- [Sauer 等 (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) —— SDXL-Turbo。