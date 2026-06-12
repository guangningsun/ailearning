# 潜在扩散与 Stable Diffusion

> 在 512×512 图像上进行像素空间扩散是一种计算犯罪。Rombach 等人（2022）注意到，生成图像不需要全部 786k 维度——你只需要足够的维度来捕捉语义结构，剩余部分交给一个独立的解码器。在 VAE 的潜在空间内运行扩散。这一思路就是 Stable Diffusion。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 8 · 02（VAE）、阶段 8 · 06（DDPM）、阶段 7 · 09（ViT）
**时间：** 约 75 分钟

## 问题

在 512² 的像素空间扩散意味着 U-Net 运行在形状为 `[B, 3, 512, 512]` 的张量上。对于一个 500M 参数的 U-Net，每采样一步约 100 GFLOPS。五十步就是每张图像 5 TFLOPS。在十亿张图像上训练，计算费用是荒谬的。

这些 FLOPs 大部分花在了把感知上不重要的细节推过网络——高频纹理，有损 VAE 可以压缩掉的东西。Rombach 的思路：训练一个 VAE 一次（*第一阶段*），冻结它，然后在 4 通道 64×64 潜在空间（*第二阶段*）中完全运行扩散。相同的 U-Net。1/16 的像素。约 64 倍更少的 FLOPs，质量相当。

这就是 Stable Diffusion 的配方。SD 1.x / 2.x 在 `64×64×4` 潜在空间上使用 860M U-Net，SDXL 在 `128×128×4` 上使用 2.6B U-Net，SD3 将 U-Net 换成了带流匹配的扩散 Transformer（DiT）。Flux.1-dev（Black Forest Labs，2024）发货的是一个 12B 参数的 DiT-MMDiT。全部运行在相同的两阶段基底层上。

## 概念

![潜在扩散：VAE 压缩 + 潜在空间中的扩散](../assets/latent-diffusion.svg)

**两个阶段，分开训练。**

1. **阶段 1 —— VAE。** 编码器 `E(x) → z`，解码器 `D(z) → x`。目标压缩：每个空间轴 8× 下采样 + 调整通道，使总潜在大小约为像素数的 1/16。损失 = 重建（L1 + LPIPS 感知）+ KL（权重较小，这样 `z` 不被强制太接近高斯分布，因为我们不需要从 `z` 精确采样）。通常用对抗损失训练，这样解码后的图像是清晰的。

2. **阶段 2 —— 在 `z` 上扩散。** 把 `z = E(x_real)` 当作数据。训练一个 U-Net（或 DiT）来对 `z_t` 去噪。推理时：通过扩散采样 `z_0`，然后 `x = D(z_0)`。

**文本条件化。** 两个额外组件。一个冻结的文本编码器（SD 1.x 用 CLIP-L，SD 2/XL 用 CLIP-L+OpenCLIP-G，SD3 和 Flux 用 T5-XXL）。一个交叉注意力注入：每个 U-Net 块接受 `[Q = 图像特征，K = V = 文本标记]` 并将它们混合。标记是文本影响图像的唯一方式。

**损失函数与第 06 课完全相同。** 相同的 DDPM / 流匹配 MSE 噪声。你只是换了数据域。

## 架构变体

| 模型 | 年份 | 骨干网络 | 潜在形状 | 文本编码器 | 参数 |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L（77 个标记） | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + refiner | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | 蒸馏版 | 128×128×4 | same | 1-4 步采样 |
| SD3 | 2024 | MMDiT（多模态 DiT） | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT 蒸馏版 | 128×128×16 | T5-XXL + CLIP-L | 12B，1-4 步 |

趋势：用 DiT 替换 U-Net（transformer over 潜在 patches），扩大文本编码器（T5 在提示 adherence 上击败 CLIP），增加潜在通道（4 → 16 给出更多细节 headroom）。

## 构建它

`code/main.py` 堆叠了一个玩具 1-D "VAE"（恒等编码器 + 解码器，用于演示；真实的 VAE 会是卷积网络）在第 06 课的 DDPM 之上，并添加了带无分类器引导的类别条件化。它表明相同的扩散损失无论是在原始 1-D 值还是在编码值上运行都有效——这是关键洞察。

### 第 1 步：编码器/解码器

```python
def encode(x):    return x * 0.5          # 玩具"压缩"到更小尺度
def decode(z):    return z * 2.0
```

一个真实的 VAE 有训练好的权重。为了教学，这个线性映射足以展示扩散在 `z` 上操作，而不关心原始数据空间。

### 第 2 步：在 `z` 空间中的扩散

与第 06 课相同的 DDPM。网络看到的数据是 `z = E(x)`。采样 `z_0` 后，用 `D(z_0)` 解码。

### 第 3 步：无分类器引导

训练期间，10% 的时间丢弃类别标签（替换为空标记）。在推理时，计算 `ε_cond` 和 `ε_uncond`，然后：

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = 无引导（完全多样性），`w = 3` = 默认值，`w = 7+` = 饱和 / 过度锐化。

### 第 4 步：文本条件化（概念，非代码）

用冻结文本编码器的输出替换类别标签。通过交叉注意力将文本嵌入馈送到 U-Net：

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

这是一个类别条件扩散模型与 Stable Diffusion 之间唯一的实质性差异。

## 陷阱

- **VAE 尺度不匹配。** SD 1.x VAE 有一个缩放常数（`scaling_factor ≈ 0.18215`）在编码后应用。忘记它会使 U-Net 在方差完全错误的潜在空间上训练。每个 checkpoint 都附带的。
- **文本编码器悄悄出错。** SD3 需要 >=128 个标记的 T5-XXL，回退到仅 CLIP 会损失很大。始终检查 `use_t5=True`，否则提示保真度会崩溃。
- **混合潜在空间。** SDXL、SD3、Flux 都使用不同的 VAE。在 SDXL 潜在空间上训练的 LoRA 在 SD3 上不工作。Hugging Face diffusers 0.30+ 拒绝加载不匹配的 checkpoint。
- **CFG 太高。** `w > 10` 产生饱和、油腻的图像，并且在多样性代价下过度拟合提示。最优点是 `w = 3-7`。
- **负提示泄露。** 空的负提示变成空标记；填充的负提示变成 `ε_uncond`。这些不一样；一些 pipeline 悄悄默认为空。

## 使用它

2026 年的生产技术栈：

| 目标 | 推荐骨干网络 |
|--------|----------------------|
| 狭窄领域、成对数据、从头训练模型 | SDXL 微调（LoRA / 全量）—— 发货最快 |
| 开放领域文生图、开放权重 | Flux.1-dev（12B，Apache / 非商业）或 SD3.5-Large |
| 最快推理、开放权重 | Flux.1-schnell（1-4 步，Apache）或 SDXL-Lightning |
| 最佳提示 adherence、托管服务 | GPT-Image / DALL-E 3（仍然）、Midjourney v7、Imagen 4 |
| 编辑工作流 | Flux.1-Kontext（2024 年 12 月）—— 原生接受图像 + 文本 |
| 研究、基线 | SD 1.5 —— 古老了但被充分研究过 |

## 交付它

保存 `outputs/skill-sd-prompter.md`。技能接受文本提示 + 目标风格并输出：模型 + checkpoint、CFG 量、采样器、负提示、分辨率、可选的 ControlNet/IP-Adapter 组合，以及每步 QA 检查清单。

## 练习

1. **简单。** 用引导 `w ∈ {0, 1, 3, 7, 15}` 运行 `code/main.py`。记录每个类的平均样本。在哪个 `w` 下类均值偏离真实数据均值？
2. **中等。** 将玩具线性编码器换成带重建损失的 tanh-MLP 编码器/解码器对。在新的潜在空间上重新训练扩散。样本质量会改变吗？
3. **困难。** 用 diffusers 设置一个真实的 Stable Diffusion 推理：加载 `sdxl-base`，用 CFG=7 运行 30 步 Euler，记录时间。现在切换到 `sdxl-turbo` 用 4 步和 CFG=0。相同主体，不同质量——描述改变了什么以及为什么。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|-----------------------|
| 第一阶段 | "VAE" | 训练好的编码器/解码器对；将 512² 压缩到 64²。 |
| 第二阶段 | "U-Net" | 潜在空间上的扩散模型。 |
| CFG | "引导量" | `(1+w)·ε_cond - w·ε_uncond`；调节条件化强度。 |
| 空标记 | "空提示嵌入" | 用于 `ε_uncond` 的非条件嵌入。 |
| 交叉注意力 | "文本如何进入" | 每个 U-Net 块作为 K 和 V 关注文本标记。 |
| DiT | "扩散 Transformer" | 用 transformer over 潜在 patches 替换 U-Net；更好扩展。 |
| MMDiT | "多模态 DiT" | SD3 的架构：文本和图像流与联合注意力。 |
| VAE 缩放因子 | "魔法数字" | 将潜在空间除以约 5.4，这样扩散在单位方差空间操作。 |

## 生产笔记：在 8GB 消费级 GPU 上运行 Flux-12B

the reference Flux integration 是"我有一块消费级 GPU，我能发货吗？"的典范配方。技巧与生产推理文献中列出的三个旋钮配方相同应用于扩散 DiT：

1. **交错加载。** Flux 有三个永远不需要共存于 VRAM 中的网络：T5-XXL 文本编码器（约 10 GB fp32）、CLIP-L（小）、12B MMDiT 和 VAE。首先编码提示，*删除*编码器，加载 DiT，去噪，*删除* DiT，加载 VAE，解码。消费级 8GB GPU 一次只装得下一个阶段。
2. **通过 bitsandbytes 的 4 位量化。** 在 T5 编码器和 DiT 上使用 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`。将内存减少 8×，质量下降在文生图上根据 Aritra 的基准测试（链接在 notebook 中）是不可察觉的。
3. **CPU 卸载。** `pipe.enable_model_cpu_offload()` 在每个前向传播推进时自动在 CPU 和 GPU 之间交换模块。增加 10-20% 延迟，但使 pipeline 得以运行。

内存计算：`10 GB T5 / 8 = 1.25 GB` 量化后，`12 B 参数 × 0.5 字节 = ~6 GB` 量化 DiT，加上激活值。用 stas00 的话说，这是 TP=1 推理的极端情况——无模型并行，最大量化。对于生产你会在 H100 上运行 TP=2 或 TP=4；对于一台开发笔记本，这就是配方。

## 延伸阅读

- [Rombach 等人（2022）。基于潜在扩散模型的高分辨率图像合成](https://arxiv.org/abs/2112.10752) —— Stable Diffusion。
- [Podell 等人（2023）。SDXL：改进高分辨率图像合成的潜在扩散模型](https://arxiv.org/abs/2307.01952) —— SDXL。
- [Peebles & Xie（2023）。使用 Transformer 的可扩展扩散模型（DiT）](https://arxiv.org/abs/2212.09748) —— DiT。
- [Esser 等人（2024）。缩放整流流 Transformer 用于高分辨率图像合成](https://arxiv.org/abs/2403.03206) —— SD3，MMDiT。
- [Ho & Salimans（2022）。无分类器扩散引导](https://arxiv.org/abs/2207.12598) —— CFG。
- [Labs（2024）。Flux.1 —— Black Forest Labs 公告](https://blackforestlabs.ai/announcing-black-forest-labs/) —— Flux.1 家族。
- [Hugging Face Diffusers 文档](https://huggingface.co/docs/diffusers/index) —— 上述每个 checkpoint 的参考实现。