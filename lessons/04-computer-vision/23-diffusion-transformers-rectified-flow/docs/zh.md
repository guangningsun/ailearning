# Diffusion Transformers 与 Rectified Flow

> U-Net 不是扩散模型的核心秘密。把它换成 transformer，把噪声调度换成直线流，突然你就有了 SD3、FLUX，以及 2026 年所有文生图模型。

**类型：** 学习 + 构建
**语言：** Python
**前置条件：** 阶段 4 第 10 课（Diffusion DDPM）、阶段 4 第 14 课（ViT）、阶段 7 第 2 课（Self-Attention）
**时间：** 约 75 分钟

## 学习目标

- 梳理从 U-Net DDPM（第 10 课）到 Diffusion Transformer（DiT）、MMDiT（SD3）、单双流 DiT（FLUX）的演进脉络
- 解释 rectified flow：为什么噪声到数据的直线轨迹能让模型用 20 步采样而非 1000 步
- 实现一个微型 DiT block 和一个 rectified flow 训练循环，每个都在 100 行以内
- 从架构、参数量、许可证角度区分模型变体（SD3、FLUX.1-dev、FLUX.1-schnell、Z-Image、Qwen-Image）

## 问题

第 10 课用 U-Net 去噪器构建了一个 DDPM。这个配方在 2020-2023 年占据主导：U-Net + beta 调度 + 噪声预测损失。它催生了 Stable Diffusion 1.5 和 2.1 以及 DALL-E 2。

2026 年每一个最先进的文生图模型都已经超越了这个范式。Stable Diffusion 3、FLUX、SD4、Z-Image、Qwen-Image、Hunyuan-Image —— 都不使用 U-Net。它们用的是 Diffusion Transformer（DiT）。SD3 和 FLUX 还把 DDPM 的噪声调度换成了 rectified flow，把噪声到数据的路径拉直，使得 1-4 步推理成为可能（通过一致性或蒸馏变体）。

这个转变很重要，因为它正是扩散式图像生成变得可控、提示词准确（SD3/SD4 解决了文本渲染）、生产速度快的根本原因。理解 DiT + rectified flow，就是理解 2026 年生成式图像技术栈。

## 概念

### 从 U-Net 到 Transformer

```mermaid
flowchart LR
    subgraph UNET["DDPM U-Net (2020)"]
        U1["Conv encoder"] --> U2["Conv bottleneck"] --> U3["Conv decoder"]
    end
    subgraph DIT["DiT (2023)"]
        D1["Patch embed"] --> D2["Transformer blocks"] --> D3["Unpatchify"]
    end
    subgraph MMDIT["MMDiT (SD3, 2024)"]
        M1["Text stream"] --> M3["Joint attention<br/>(separate weights per modality)"]
        M2["Image stream"] --> M3
    end
    subgraph FLUX["FLUX (2024)"]
        F1["Double-stream blocks<br/>(text + image separate)"] --> F2["Single-stream blocks<br/>(concat + shared weights)"]
    end

    style UNET fill:#e5e7eb,stroke:#6b7280
    style DIT fill:#dbeafe,stroke:#2563eb
    style MMDIT fill:#fef3c7,stroke:#d97706
    style FLUX fill:#dcfce7,stroke:#16a34a
```

- **DiT**（Peebles & Xie，2023）—— 用类 ViT 的 transformer 替代 U-Net，处理潜空间 patches。条件化通过自适应层归一化（AdaLN）实现。
- **MMDiT**（SD3，Esser et al.，2024）—— 文本和图像 token 各有一条独立权重流，在联合注意力层共享。
- **FLUX**（Black Forest Labs，2024）—— 前 N 个 block 是双流（类似 SD3），后续 block 拼接权重并共享（单流），以在更高深度下提升效率。
- **Z-Image**（2025）—— 一个高效的 6B 参数单流 DiT，挑战"不惜一切代价 scale"的路线。

### Rectified flow 一段话解释

DDPM 把前向过程定义为一个噪声 SDE，其中 `x_t` 逐渐被污染。学习到的反向是第二个 SDE，需要 1000 小步来求解。

Rectified flow 定义的是纯净数据和纯噪声之间的**直线**插值：

```
x_t = (1 - t) * x_0 + t * epsilon,     t in [0, 1]
```

训练一个网络预测速度 `v_theta(x_t, t) = epsilon - x_0` —— 沿直线从干净数据到噪声的方向（`dx_t/dt`）。采样时，通过对这个速度做反向积分，从噪声步进到数据。这个 ODE 更接近直线，所以需要少得多的积分步数来采样。

SD3 称之为**Rectified Flow Matching**。FLUX、Z-Image 和大多数 2026 年模型使用相同的目标。典型推理：20-30 步 Euler（确定性），对比旧 DDPM regime 下的 50+ 步 DDIM。蒸馏 / turbo / schnell / LCM 变体把这个数字降到 1-4 步。

### AdaLN 条件化

DiT 通过**自适应层归一化**对时间步和类别/文本做条件化：从条件向量预测 `scale` 和 `shift`，在 LayerNorm 之后应用。比 U-Net 中的 FiLM 风格调制更简洁，是每个现代 DiT 的默认选择。

```
cond -> MLP -> (scale, shift, gate)
norm(x) * (1 + scale) + shift, then residual add * gate
```

### SD3 和 FLUX 中的文本编码器

- **SD3** 使用三个文本编码器：两个 CLIP 模型 + T5-XXL。embedding 拼接后输入图像流作为文本条件。
- **FLUX** 使用一个 CLIP-L + T5-XXL。
- **Qwen-Image / Z-Image** 变体使用与各自基础 LLM 对齐的自研文本编码器。

文本编码器是 SD3/FLUX 比 SD1.5 在提示词推理上好得多的重要原因。仅 T5-XXL 就有 4.7B 参数。

### 无分类器引导仍然有效

Rectified flow 改变的是采样器，不是条件化。无分类器引导（在训练时以 10% 概率丢弃文本，推理时混合条件和非条件预测）在 rectified flow 中效果完全相同。大多数 2026 年模型使用引导强度 3.5-5 —— 比 SD1.5 的 7.5 低，因为 rectified flow 模型默认对提示词的跟随性更好。

### Consistency、Turbo、Schnell、LCM

四个名字，同一个思路：把一个慢速多步模型蒸馏成一个快速少步模型。

- **LCM（Latent Consistency Model）** —— 训练一个学生网络，从任意中间 `x_t` 一步预测最终 `x_0`。
- **SDXL Turbo / FLUX schnell** —— 用对抗扩散蒸馏训练的 1-4 步模型。
- **SD Turbo** —— 适配到潜扩散的 OpenAI 风格一致性模型。

任何新模型的生产发布都同时提供"全质量"检查点和"turbo / schnell"变体。Schnell（德语"快速"，Black Forest Labs 的命名惯例）1-4 步运行，适合实时流水线。

### 2026 年模型版图

| 模型 | 规模 | 架构 | 许可证 |
|-------|------|--------------|---------|
| Stable Diffusion 3 Medium | 2B | MMDiT | SAI Community |
| Stable Diffusion 3.5 Large | 8B | MMDiT | SAI Community |
| FLUX.1-dev | 12B | 双流 + 单流 DiT | 非商业 |
| FLUX.1-schnell | 12B | 同上，蒸馏 | Apache 2.0 |
| FLUX.2 | — | 迭代 FLUX.1 | 混合 |
| Z-Image | 6B | S3-DiT（Scalable Single-Stream）| 宽松 |
| Qwen-Image | ~20B | DiT + Qwen 文本塔 | Apache 2.0 |
| Hunyuan-Image-3.0 | ~80B | DiT | 研究 |
| SD4 Turbo | 3B | DiT + 蒸馏 | SAI Commercial |

FLUX.1-schnell 是 2026 年开源默认选择。Z-Image 是效率领导者。FLUX.2 和 SD4 是当前质量巅峰。

### 为什么这次相变很重要

DDPM + U-Net 有效。DiT + rectified flow **更好、更快、更干净地 scale**。这个转变类似于 NLP 中从 RNN 到 transformer 的转变：两种架构解决了同一个问题，但 transformer 可以 scale，现在占据主导。2026 年每一篇关于图像、视频或 3D 生成的论文都使用 DiT 形状的去噪器，通常也使用 rectified flow 目标。U-Net DDPM 现在主要是教学用途（第 10 课）。

## 构建

### 第 1 步：带 AdaLN 的 DiT block

```python
import torch
import torch.nn as nn


class AdaLNZero(nn.Module):
    """
    Adaptive LayerNorm with a gate. Predicts (scale, shift, gate) from the conditioning.
    Init such that the whole block starts as identity ("zero init").
    """

    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Linear(cond_dim, dim * 3)
        nn.init.zeros_(self.mlp.weight)
        nn.init.zeros_(self.mlp.bias)

    def forward(self, x, cond):
        scale, shift, gate = self.mlp(cond).chunk(3, dim=-1)
        h = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return h, gate.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, dim=192, heads=3, mlp_ratio=4, cond_dim=192):
        super().__init__()
        self.adaln1 = AdaLNZero(dim, cond_dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.adaln2 = AdaLNZero(dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, cond):
        h, gate1 = self.adaln1(x, cond)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1 * a
        h, gate2 = self.adaln2(x, cond)
        x = x + gate2 * self.mlp(h)
        return x
```

`AdaLNZero` 因为 MLP 权重初始化为零，所以开始时是一个恒等映射。训练推动 block 远离恒等映射；这极大地稳定了深度 transformer 扩散模型。

### 第 2 步：一个微型 DiT

```python
def timestep_embedding(t, dim):
    import math
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class TinyDiT(nn.Module):
    def __init__(self, image_size=16, patch_size=2, in_channels=3, dim=96, depth=4, heads=3):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        self.blocks = nn.ModuleList([DiTBlock(dim, heads, cond_dim=dim) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.head = nn.Linear(dim, patch_size * patch_size * in_channels)

    def forward(self, x, t):
        n = x.size(0)
        x = self.patch(x)
        x = x.flatten(2).transpose(1, 2) + self.pos
        t_emb = self.time_mlp(timestep_embedding(t, self.pos.size(-1)))
        for blk in self.blocks:
            x = blk(x, t_emb)
        x = self.norm_out(x)
        x = self.head(x)
        return self._unpatchify(x, n)

    def _unpatchify(self, x, n):
        p = self.patch_size
        h = w = int(self.num_patches ** 0.5)
        x = x.view(n, h, w, p, p, -1).permute(0, 5, 1, 3, 2, 4).reshape(n, -1, h * p, w * p)
        return x
```

### 第 3 步：Rectified flow 训练

```python
import torch.nn.functional as F

def rectified_flow_train_step(model, x0, optimizer, device):
    model.train()
    x0 = x0.to(device)
    n = x0.size(0)
    t = torch.rand(n, device=device)
    epsilon = torch.randn_like(x0)
    x_t = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * epsilon

    target_velocity = epsilon - x0
    pred_velocity = model(x_t, t)

    loss = F.mse_loss(pred_velocity, target_velocity)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

与 DDPM 的噪声预测损失（第 10 课）对比：结构相同，目标不同。不是预测噪声 `epsilon`，而是预测从数据指向噪声的**速度** `epsilon - x_0`。

### 第 4 步：Euler 采样器

Rectified flow 是一个 ODE。Euler 方法是最简单的，对于训练良好的 rectified flow 模型，在 20+ 步下几乎与高阶求解器一样准确。

```python
@torch.no_grad()
def rectified_flow_sample(model, shape, steps=20, device="cpu"):
    model.eval()
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    t = torch.ones(shape[0], device=device)
    for _ in range(steps):
        v = model(x, t)
        x = x - dt * v
        t = t - dt
    return x
```

20 步。在一个训练好的模型上，这产生的样本可与 1000 步 DDPM 媲美。

### 第 5 步：端到端冒烟测试

```python
import numpy as np

def synthetic_blobs(num=200, size=16, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros((num, 3, size, size), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    for i in range(num):
        cx, cy = rng.uniform(4, size - 4, size=2)
        r = rng.uniform(2, 4)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        colour = rng.uniform(-1, 1, size=3)
        for c in range(3):
            out[i, c][mask] = colour[c]
    return torch.from_numpy(out)
```

用 rectified flow 训练 `TinyDiT`。500 步后，采样输出应该看起来像模糊的彩色斑点。

## 使用

对于 FLUX / SD3 / Z-Image 的真实图像生成，`diffusers` 为每个模型提供了统一 API：

```python
from diffusers import FluxPipeline, StableDiffusion3Pipeline
import torch

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
).to("cuda")

out = pipe(
    prompt="a golden retriever surfing a tsunami, hyperrealistic, studio lighting",
    guidance_scale=0.0,           # schnell 在无 CFG 下训练
    num_inference_steps=4,
    max_sequence_length=256,
).images[0]
out.save("surf.png")
```

三行代码。`FLUX.1-schnell` 四步完成。把 model id 换成 `black-forest-labs/FLUX.1-dev` 可获得更高质量，20-30 步，带 CFG。

对于 SD3：

```python
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    torch_dtype=torch.bfloat16,
).to("cuda")
out = pipe(prompt, guidance_scale=3.5, num_inference_steps=28).images[0]
```

## 交付

本课产出：

- `outputs/prompt-dit-model-picker.md` —— 根据质量、延迟和许可证约束在 SD3、FLUX.1-dev、FLUX.1-schnell、Z-Image、SD4 Turbo 之间选择。
- `outputs/skill-rectified-flow-trainer.md` —— 写一个完整的带 AdaLN DiT 和 Euler 采样的 rectified flow 训练循环。

## 练习

1. **（简单）** 在合成 blob 数据集上训练上面的 TinyDiT 500 步。用 10、20 和 50 步 Euler 采样，比较生成的样本。
2. **（中等）** 通过将学习到的类别 embedding 拼接到时间 embedding 来添加文本条件（按颜色分为 10 个 blob"类别"）。用类别 0、5 和 9 采样，验证颜色匹配。
3. **（困难）** 计算同尺寸网络在相同数据上训练相同步数的 rectified flow 和 DDPM 版本之间生成样本的 Fréchet 距离（FID 代理）。报告哪个收敛更快。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|----------------------|
| DiT | "Diffusion transformer" | 用 transformer 替代 U-Net 作为扩散去噪器；在 patch 化后的潜空间上操作 |
| AdaLN | "Adaptive layer norm" | 通过学习到的 scale、shift、gate 对时间步/文本做条件化，在 LayerNorm 后应用；每个现代 DiT 的标准配置 |
| MMDiT | "Multi-modal DiT (SD3)" | 文本和图像 token 有独立权重流，在联合自注意力中共享 |
| 单流 / 双流 | "FLUX 技巧" | 前 N 个 block 双流（每种模态独立权重），后续 block 单流（拼接 + 共享权重）以提升效率 |
| Rectified flow | "从噪声到数据的直线" | 数据和噪声之间的线性插值；网络预测速度；推理时需要更少的 ODE 步数 |
| 速度目标 | "epsilon - x_0" | Rectified flow 中的回归目标；从干净数据指向噪声 |
| CFG 引导 | "classifier-free guidance" | 混合条件和非条件预测；在 rectified flow 模型中仍然使用 |
| Schnell / turbo / LCM | "1-4 步蒸馏" | 从全质量模型蒸馏的小步变体；生产级实时推理 |

## 延伸阅读

- [Scalable Diffusion Models with Transformers（Peebles & Xie，2023）](https://arxiv.org/abs/2212.09748) —— DiT 论文
- [Scaling Rectified Flow Transformers（Esser et al.，SD3 论文）](https://arxiv.org/abs/2403.03206) —— MMDiT 和规模化 rectified flow
- [FLUX.1 model card and technical report（Black Forest Labs）](https://huggingface.co/black-forest-labs/FLUX.1-dev) —— 双流 + 单流详情
- [Z-Image: Efficient Image Generation Foundation Model（2025）](https://arxiv.org/html/2511.22699v1) —— 6B 单流 DiT
- [Elucidating the Design Space of Diffusion（Karras et al.，2022）](https://arxiv.org/abs/2206.00364) —— 每个扩散设计权衡的参考
- [Latent Consistency Models（Luo et al.，2023）](https://arxiv.org/abs/2310.04378) —— LCM-LoRA 如何实现 4 步推理