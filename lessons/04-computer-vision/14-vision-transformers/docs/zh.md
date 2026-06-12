# 视觉 Transformer（ViT）

> 把图像切成图块，把每个图块当作一个词，运行标准 Transformer。不要回头看。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 7 第 02 课（自注意力）、阶段 4 第 04 课（图像分类）
**时间：** 约 45 分钟

## 学习目标

- 从零实现图块嵌入、学到的位置嵌入、类别 token 和 Transformer 编码器块，构建一个最小 ViT
- 解释为什么 ViT 曾被认为需要大规模预训练数据，直到 DeiT 和 MAE 证明了并非如此
- 比较 ViT、Swin 和 ConvNeXt 在架构先验上的差异（无、局部窗口注意力、卷积主干）
- 使用 `timm` 和标准线性探测 / 微调流程，在小数据集上微调预训练 ViT

## 问题

十年来，卷积一直是计算机视觉的代名词。CNN 有很强的归纳偏置 —— 局部性、平移等变性 —— 没有人认为你能取代它们。然后 Dosovitskiy 等人（2020）表明，应用普通 Transformer 到展平的图像图块上，不使用任何卷积机制，在规模上可以匹配或击败最好的 CNN。

美中不足的是"在规模上"。ImageNet-1k 上的 ViT 输给了 ResNet。在 ImageNet-21k 或 JFT-300M 上预训练然后在 ImageNet-1k 上微调的 ViT 击败了它。结论是 Transformer 缺乏有用的先验，但可以从足够多的数据中学习到。后续工作（DeiT、MAE、DINO）表明，通过正确的训练配方 —— 强增强、自监督预训练、知识蒸馏 —— ViT 也可以在小数据上训练。

到 2026 年，纯 CNN 在边缘设备上仍然有竞争力（ConvNeXt 最强），但 Transformer 主导着其他一切：分割（Mask2Former、SegFormer）、检测（DETR、RT-DETR）、多模态（CLIP、SigLIP）、视频（VideoMAE、VJEPA）。ViT 块结构是必须了解的。

## 概念

### 流程

```mermaid
flowchart LR
    IMG["图像<br/>(3, 224, 224)"] --> PATCH["图块嵌入<br/>conv 16x16 s=16<br/>-> (768, 14, 14)"]
    PATCH --> FLAT["展平为<br/>(196, 768) token"]
    FLAT --> CAT["前置<br/>[CLS] token"]
    CAT --> POS["加上学到的<br/>位置嵌入"]
    POS --> ENC["N 个 transformer<br/>编码器块"]
    ENC --> CLS["取 [CLS]<br/>token 输出"]
    CLS --> HEAD["MLP 分类器"]

    style PATCH fill:#dbeafe,stroke:#2563eb
    style ENC fill:#fef3c7,stroke:#d97706
    style HEAD fill:#dcfce7,stroke:#16a34a
```

七个步骤。图块 -> token -> 注意力 -> 分类器。每个变体（DeiT、Swin、ConvNeXt、MAE 预训练）改变其中一到步，其余保持不变。

### 图块嵌入

第一个卷积是秘密。核大小 16，步长 16，所以 224x224 图像变成 14x14 的 16x16 图块网格，每个投影到 768 维嵌入。单层卷积同时完成图块化和线性投影。

```
输入:  (3, 224, 224)
卷积 (3 -> 768, k=16, s=16, 无填充):
输出: (768, 14, 14)
空间展平: (196, 768)
```

196 个图块 = 196 个 token。每个 token 的特征维度是 768（ViT-B）、1024（ViT-L）或 1280（ViT-H）。

### 类别 token

一个学到的向量被前置到序列中：

```
tokens = [CLS; patch_1; patch_2; ...; patch_196]   shape (197, 768)
```

经过 N 个 transformer 块后，`[CLS]` 输出是全局图像表示。分类头只读取这一个向量。

### 位置嵌入

Transformer 没有内置的空间位置概念。为每个 token 添加一个学到的向量：

```
tokens = tokens + learned_pos_embedding   (同样 shape (197, 768))
```

嵌入是模型的参数；基于梯度的训练使其适应 2D 图像结构。正弦 2D 替代方案存在，但实际上很少使用。

### Transformer 编码器块

标准结构。多头自注意力、MLP、残差连接、前置 LayerNorm。

```
x = x + MSA(LN(x))
x = x + MLP(LN(x))

MLP 是两层 GELU: Linear(d -> 4d) -> GELU -> Linear(4d -> d)
```

ViT-B/16 堆叠 12 个这样的块，每个块有 12 个注意力头，共 8600 万参数。

### 为什么用 Pre-LN

早期 transformer 使用后置 LN（`x = LN(x + sublayer(x))`），训练超过 6-8 层时没有预热就无法稳定。前置 LN（`x = x + sublayer(LN(x))`）可以稳定地训练更深的网络，无需预热。每个 ViT 和每个现代 LLM 都使用前置 LN。

### 图块大小权衡

- 16x16 图块 -> 196 个 token，标准。
- 32x32 图块 -> 49 个 token，更快但分辨率更低。
- 8x8 图块 -> 784 个 token，更细但 O(n^2) 注意力成本扩展性差。

更大的图块 = 更少的 token = 更快但空间细节更少。SwinV2 在分层窗口中使用 4x4 图块。

### DeiT 在 ImageNet-1k 上训练 ViT 的配方

原始 ViT 需要 JFT-300M 才能击败 CNN。DeiT（Touvron 等，2020）仅用四个改动就将 ViT-B 训练到 ImageNet-1k 上 81.8% 的 top-1 准确率：

1. 强增强：RandAugment、Mixup、CutMix、随机擦除。
2. 随机深度（训练时随机丢弃整个块）。
3. 重复增强（同一图像每批采样 3 次）。
4. 从 CNN 教师蒸馏（可选，进一步提升准确率）。

每个现代 ViT 训练配方都源自 DeiT。

### Swin 与 ConvNeXt

- **Swin**（Liu 等，2021）—— 基于窗口的注意力。每个块在局部窗口内进行注意力计算；交替的块移动窗口以在窗口间混合信息。恢复了类似 CNN 的局部性先验，同时保留了注意力算子。
- **ConvNeXt**（Liu 等，2022）—— 重新设计的 CNN，匹配 Swin 的架构选择（深度可分离卷积、LayerNorm、GELU、倒置瓶颈）。表明差距不在"注意力 vs 卷积"而在"现代训练配方 + 架构"。

2026 年，ConvNeXt-V2 和 Swin-V2 都是生产级的；正确选择取决于你的推理栈（ConvNeXt 对边缘编译更好）和预训练语料。

### MAE 预训练

掩码自编码器（He 等，2022）：随机掩码 75% 的图块，训练编码器只处理可见的 25%，训练一个小解码器从编码器输出重建被掩码的图块。预训练后丢弃解码器，微调编码器。

MAE 使 ViT 可以在 ImageNet-1k 上单独训练，达到 SOTA，是当前自监督预训练的默认配方。

## 构建

### 第 1 步：图块嵌入

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, dim=192, image_size=64):
        super().__init__()
        assert image_size % patch_size == 0
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)
```

一层卷积、一层展平、一层转置。这就是图像到 token 的全部步骤。

### 第 2 步：Transformer 块

前置 LN、多头自注意力、带 GELU 的 MLP、残差连接。

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x
```

`nn.MultiheadAttention` 处理头拆分、缩放点积和输出投影。`batch_first=True` 使形状为 `(N, seq, dim)`。

### 第 3 步：ViT

```python
class ViT(nn.Module):
    def __init__(self, image_size=64, patch_size=16, in_channels=3,
                 num_classes=10, dim=192, depth=6, num_heads=3, mlp_ratio=4):
        super().__init__()
        self.patch = PatchEmbedding(in_channels, patch_size, dim, image_size)
        num_patches = self.patch.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x[:, 0])
        return self.head(x)

vit = ViT(image_size=64, patch_size=16, num_classes=10, dim=192, depth=6, num_heads=3)
x = torch.randn(2, 3, 64, 64)
print(f"output: {vit(x).shape}")
print(f"params: {sum(p.numel() for p in vit.parameters()):,}")
```

约 280 万参数 —— 一个可在 CPU 上运行的微型 ViT。真正的 ViT-B 是 8600 万；相同的类定义，`dim=768, depth=12, num_heads=12`。

### 第 4 步：完整性检查 —— 单张图像推理

```python
logits = vit(torch.randn(1, 3, 64, 64))
print(f"logits: {logits}")
print(f"probs:  {logits.softmax(-1)}")
```

应该能无错误运行。概率和为 1。

## 使用

`timm` 包含每个 ViT 变体及 ImageNet 预训练权重。一行代码：

```python
import timm

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
```

`timm` 是 2026 年视觉 transformer 的生产默认。支持 ViT、DeiT、Swin、Swin-V2、ConvNeXt、ConvNeXt-V2、MaxViT、MViT、EfficientFormer 等数十种模型，API 统一。

对于多模态工作（图像 + 文本），`transformers` 提供 CLIP、SigLIP、BLIP-2、LLaVA。这些中的图像编码器都是 ViT 变体。

## 交付

本课产出：

- `outputs/prompt-vit-vs-cnn-picker.md` —— 根据数据集大小、算力和推理栈在 ViT、ConvNeXt 或 Swin 之间选择的提示词。
- `outputs/skill-vit-patch-and-pos-embed-inspector.md` —— 验证 ViT 的图块嵌入和位置嵌入形状与模型期望的序列长度匹配的技能，捕捉最常见的移植 bug。

## 练习

1. **（简单）** 打印微型 ViT 前向传播中每个中间张量的形状。确认：输入 `(N, 3, 64, 64)` -> 图块 `(N, 16, 192)` -> 加 CLS `(N, 17, 192)` -> 分类器输入 `(N, 192)` -> 输出 `(N, num_classes)`。
2. **（中等）** 在第 4 课的合成 CIFAR 数据集上微调预训练的 `timm` ViT-S/16。与在同一数据上的 ResNet-18 微调比较。报告训练时间和最终准确率。
3. **（困难）** 为微型 ViT 实现 MAE 预训练：掩码 75% 的图块，训练编码器 + 小解码器重建被掩码的图块。在微调前和微调后评估合成数据上的线性探测准确率。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|----------------------|
| 图块嵌入 | "第一层卷积" | 核大小 = 步长 = 图块大小的卷积；将图像转换为 token 嵌入的网格 |
| 类别 token | "[CLS]" | 前置到 token 序列的学到的向量；其最终输出是全局图像表示 |
| 位置嵌入 | "学到的位置" | 添加到每个 token 的学到的向量，使 transformer 知道每个图块来自哪里 |
| Pre-LN | "在子层之前做 LayerNorm" | 稳定的 transformer 变体：`x + sublayer(LN(x))` 而不是 `LN(x + sublayer(x))` |
| 多头注意力 | "并行注意力" | 标准 transformer 注意力拆分为 num_heads 个独立子空间，之后拼接 |
| ViT-B/16 | "基础，16 图块" | 规范尺寸：dim=768, depth=12, heads=12, patch_size=16, image=224；约 8600 万参数 |
| DeiT | "数据高效 ViT" | 仅在 ImageNet-1k 上训练的 ViT，配合强增强；证明大规模预训练数据集并非严格必需 |
| MAE | "掩码自编码器" | 自监督预训练：掩码 75% 的图块，重建；主导的 ViT 预训练配方 |

## 延伸阅读

- [一张图值得 16x16 个词（Dosovitskiy 等，2020）](https://arxiv.org/abs/2010.11929) —— ViT 论文
- [DeiT：数据高效图像 Transformer（Touvron 等，2020）](https://arxiv.org/abs/2012.12877) —— 如何仅在 ImageNet-1k 上训练 ViT
- [掩码自编码器是可扩展的视觉学习器（He 等，2022）](https://arxiv.org/abs/2111.06377) —— MAE 预训练
- [timm 文档](https://huggingface.co/docs/timm) —— 你将在生产中使用的每个视觉 transformer 的参考
