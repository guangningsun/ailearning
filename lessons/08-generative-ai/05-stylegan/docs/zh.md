# StyleGAN

> 大多数生成器把 `z` 同时混入每一层。StyleGAN 把它拆开了：先通过映射网络把 `z` 映射到中间空间 `w`，然后通过 AdaIN 在每个分辨率层级*注入* `w`。这一个改变解开了隐空间的纠缠，并使人脸 photorealism 在接下来七年中成为一个已解决的问题。

**类型：** 动手构建
**语言：** Python
**前置条件：** 阶段 8 · 03（GAN）、阶段 4 · 08（归一化）、阶段 3 · 07（CNN）
**时间：** 约 45 分钟

## 问题

DCGAN 通过一堆转置卷积把 `z` 映射到图像。问题是：`z` 同时控制一切——姿态、光照、身份、背景——纠缠在一起。沿着 `z` 的一个轴移动，四个都会变。你无法问模型"同样的人，不同的姿态"，因为表征不是这样因式分解的。

Karras 等（2019，NVIDIA）提议：不要把 `z` 直接送入卷积层。用一个常数张量 `4×4×512` 作为网络输入。学习一个 8 层 MLP 将 `z ∈ Z` 映射到 `w ∈ W`。通过*自适应实例归一化*（AdaIN）在每个分辨率注入 `w`：归一化每个卷积特征图，然后用 `w` 的仿射投影做缩放和偏移。为随机细节（皮肤毛孔、发丝）添加逐层噪声。

结果：`W` 空间的大致正交轴分别对应"高级风格"（姿态、身份）和"精细风格"（光照、颜色）。你可以通过在一张图像的低分辨率层使用图像 A 的 `w`，在高分辨率层使用图像 B 的 `w` 来交换风格。这解锁了编辑、跨域风格化，以及整个"StyleGAN 反演"研究方向。

## 概念

![StyleGAN：映射网络 + AdaIN + 逐层噪声](../assets/stylegan.svg)

**映射网络。** `f: Z → W`，一个 8 层 MLP。`Z = N(0, I)^512`。`W` 不被强制为高斯分布——它学习一个数据自适应的形状。

**合成网络。** 从一个可学习的常数 `4×4×512` 开始。每个分辨率块：`上采样 → 卷积 → AdaIN(w_i) → 噪声 → 卷积 → AdaIN(w_i) → 噪声`。分辨率翻倍：4、8、16、32、64、128、256、512、1024。

**AdaIN。**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

其中 `y_scale` 和 `y_bias` 来自 `w` 的仿射投影。逐特征图归一化，然后重新赋予风格。这里的"风格"是特征图的一阶和二阶统计量。

**逐层噪声。** 单通道高斯噪声添加到每个特征图，按一个可学习的逐通道因子缩放。控制随机细节而不影响全局结构。

**截断技巧。** 推理时，采样 `z`，计算 `w = mapping(z)`，然后 `w' = ŵ + ψ·(w - ŵ)`，其中 `ŵ` 是大量样本的均值 `w`。`ψ < 1` 以多样性换取质量。几乎每个 StyleGAN 演示都使用 `ψ ≈ 0.7`。

## StyleGAN 1 → 2 → 3

| 版本 | 年份 | 创新点 |
|---------|------|------------|
| StyleGAN | 2019 | 映射网络 + AdaIN + 噪声 + 渐进式增长。 |
| StyleGAN2 | 2020 | 权重去调制替代 AdaIN（修复液滴伪影）；跳跃/残差架构；路径长度正则化。 |
| StyleGAN3 | 2021 | 无别名卷积 + 等变核；消除纹理在像素网格上的粘滞。 |
| StyleGAN-XL | 2022 | 类条件，1024²，ImageNet。 |
| R3GAN | 2024 | 重新品牌，更强正则化；在 FFHQ-1024 上用少 20 倍参数追平扩散。 |

在 2026 年，StyleGAN3 仍然是以下场景的默认选择：（a）高 FPS 窄域 photorealism，（b）少样本域适应（用 100 张图像在新数据集上训练，冻结映射网络），（c）基于反演的编辑（找到重建真实照片的 `w`，然后编辑那个 `w`）。对于开放域文生图，它不是合适的工具——扩散模型才是。

## 动手构建

`code/main.py` 在 1 维上实现了一个"StyleGAN 简化版"：一个映射 MLP、一个合成函数（接收一个可学习常数向量并用 `w` 衍生的缩放/偏置调制它）和逐层噪声。它表明通过仿射调制注入 `w` 与将 `z` 连接到生成器输入的效果相当或更好。

### 第 1 步：映射网络

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### 第 2 步：自适应实例归一化

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

逐特征图的缩放和偏置通过线性投影从 `w` 获取。

### 第 3 步：逐层噪声

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

每个通道的 sigma 是可学习的。

## 陷阱

- **液滴伪影。** StyleGAN 1 在特征图中产生了 Blob 状液滴，因为 AdaIN 把均值清零了。StyleGAN 2 的权重去调制通过缩放卷积权重而不是激活来修复它。
- **纹理粘滞。** StyleGAN 1 和 2 的纹理跟随像素坐标而不是物体坐标（插值时可见）。StyleGAN 3 的无别名卷积用加窗 sinc 滤波器修复了这个问题。
- **模态覆盖。** 截断 `ψ < 0.7` 看起来干净但只从狭窄锥区采样；如果需要多样性，使用 `ψ = 1.0`。
- **反演有损。** 将真实照片反演到 `W` 通常通过优化或编码器（e4e、ReStyle、HyperStyle）完成。多轮迭代后结果会漂移。

## 使用场景

| 使用案例 | 方法 |
|----------|----------|
| Photoreal 人脸（动漫、产品、窄域） | StyleGAN3 FFHQ / 自定义微调 |
| 从照片编辑人脸 | e4e 反演 + StyleSpace / InterFaceGAN 方向 |
| 换脸 / 重演 | StyleGAN + 编码器 + 混合 |
| Avatar 管线 | StyleGAN3 w/ ADA 用于低数据微调 |
| 少样本域适应 | 冻结映射网络，微调合成网络 |
| 多模态或文本条件生成 | 不要用 StyleGAN——用扩散模型 |

对于答案是"某人脸部的照片"的生产级演示，StyleGAN 在推理成本（单次前向传播，4090 上 <10ms）和同等质量下的锐利度上胜于扩散模型。

## 交付

保存 `outputs/skill-stylegan-inversion.md`。技能接收一张真实照片并输出：反演方法（e4e / ReStyle / HyperStyle）、预期隐空间损失、编辑预算（在产生伪影之前可以在 `W` 中移动多远），以及已知良好的编辑方向列表（年龄、表情、姿态）。

## 练习

1. **简单。** 分别用 `adain_on=True` 和 `adain_on=False` 运行 `code/main.py`。比较固定隐向量 vs 扰动隐向量的输出分布范围。
2. **中等。** 实现混合正则化：对一个训练批，计算 `w_a`、`w_b`，在前半段合成应用 `w_a`，在后半段应用 `w_b`。解码器是否学会了解缠的风格？
3. **困难。** 取一个预训练 StyleGAN3 FFHQ 模型（ffhq-1024.pkl）。通过在标记样本上训练 SVM 找到控制"微笑"的 `w` 方向；报告在身份漂移之前能推多远。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Mapping network | "MLP" | `f: Z → W`，8 层，将隐空间几何与数据统计解耦。 |
| W 空间 | "风格空间" | 映射网络的输出；大致解缠。 |
| AdaIN | "自适应实例归一化" | 归一化特征图，然后用 `w` 投影做缩放 + 位移。 |
| Truncation trick | "Psi" | `w = mean + ψ·(w - mean)`，ψ<1 以多样性换取质量。 |
| Path-length regularization | "PL 正则化" | 惩罚每单位 `w` 变化下图像的大变化；使 `W` 更平滑。 |
| Weight demodulation | "StyleGAN2 修复" | 归一化卷积权重而不是激活；消除液滴伪影。 |
| Alias-free | "StyleGAN3 的技巧" | 加窗 sinc 滤波器；消除纹理在像素网格上的粘滞。 |
| Inversion | "为真实图像找 w" | 优化或编码 `x → w` 使得 `G(w) ≈ x`。 |

## 生产注意事项：为什么 StyleGAN 在 2026 年仍在使用

StyleGAN3 在 4090 上生成 1024² FFHQ 人脸只需不到 10ms——`num_steps = 1`，无需 VAE 解码，无需交叉注意力传递。从生产角度说，这是任何图像生成器的最低延迟。同分辨率下 50 步 SDXL + VAE 解码管线约 3 秒。这是 **300 倍差距**，对于窄域产品（头像服务、身份证文件管线、库存人脸生成）来说它在 TCO 上胜出。

两个操作后果：

- **无需调度器，无须批处理器。** 静态批处理以目标利用率为最优。连续批处理（对 LLM 和扩散模型必不可少）毫无益处，因为每个请求的 FLOPs 相同。
- **截断 `ψ` 是安全阀。** `ψ < 0.7` 从映射网络范围的狭窄锥区采样。这是服务层控制样本方差的唯一手段。在峰值负载时降低 `ψ`，为高级用户提升。

## 延伸阅读

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3。
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) — e4e 反演。
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Databases](https://arxiv.org/abs/2202.00273) — StyleGAN-XL。
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) — 现代极简 GAN 配方。