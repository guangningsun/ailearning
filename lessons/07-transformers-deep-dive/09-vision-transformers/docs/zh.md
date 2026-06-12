# 视觉 Transformer（ViT）

> 一张图像是一组 patch 的网格。一句话是一组 token 的网格。同一个 transformer 两者通吃。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 7 · 05（完整 Transformer）、阶段 4 · 03（CNN）、阶段 4 · 14（视觉 Transformer 简介）
**时间：** 约 45 分钟

## 问题

在 2020 年之前，计算机视觉意味着卷积。ImageNet、COCO 和检测基准上的每个 SOTA 都使用 CNN 主干。Transformer 用于语言。

Dosovitskiy et al.（2020）——"一张图像价值 16×16 个词"——表明你可以完全去掉卷积。将图像切成固定大小的 patch，将每个 patch 线性投影为 embedding，然后将序列送入 vanilla transformer 编码器。在足够大规模（ImageNet-21k 预训练或更大）下，ViT 匹配或击败基于 ResNet 的模型。

ViT 是 2026 年更广泛模式的开始：一种架构，多种模态。Whisper 对音频进行 token 化。ViT 对图像进行 token 化。机器人用的 action token。视频用的 pixel token。Transformer 不在乎——给它一个序列，它就会学习。

到 2026 年，ViT 及其后代（DeiT、Swin、DINOv2、ViT-22B、SAM 3）占据了大多数视觉领域。CNN 在边缘设备和延迟敏感任务上仍然胜出。其他一切都在栈中某处有一个 ViT。

## 概念

![图像 → patches → tokens → transformer](../assets/vit.svg)

### 第 1 步——patchify

将 `H × W × C` 的图像分割成 `N × (P·P·C)` 的扁平 patches 序列。典型设置：`224 × 224` 图像，`16 × 16` patches → 196 个 patches，每个 768 值。

```
image (224, 224, 3) → 14 × 14 个 16x16x3 patches 的网格 → 196 个长度为 768 的向量
```

Patch 大小是杠杆。更小的 patches = 更多的 tokens，更好的分辨率，二次方注意力成本。更小的 patches = 更粗糙，更便宜。

### 第 2 步——线性 embedding

单个学习到的矩阵将每个扁平 patch 投影到 `d_model`。等价于卷积核大小 `P` 和步长 `P` 的卷积。在 PyTorch 中这 literally 是 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)`——两行代码的实现。

### 第 3 步——前置 `[CLS]` token，添加位置 embedding

- 前置一个可学习的 `[CLS]` token。它的最终隐藏状态是用于分类的图像表示。
- 添加可学习的位置 embedding（原始 ViT）或正弦 2D（后来的变体）。
- 在 2024+ RoPE 扩展到 2D 用于位置，有时不带显式 embedding。

### 第 4 步——标准 transformer 编码器

堆叠 L 个 `LayerNorm → 自注意力 → + → LayerNorm → MLP → +` 块。与 BERT 相同。没有视觉特定的层。这是论文的教学要点。

### 第 5 步——头

对于分类：取 `[CLS]` 隐藏状态 → linear → softmax。对于 DINOv2 或 SAM，丢弃 `[CLS]`，直接使用 patch embedding。

### 重要的变体

| 模型 | 年份 | 变化 |
|-------|------|--------|
| ViT | 2020 | 原始。固定 patch 大小，全局注意力。 |
| DeiT | 2021 | 蒸馏；只需在 ImageNet-1k 上训练。 |
| Swin | 2021 | 带移位窗口的分层。固定亚二次方成本。 |
| DINOv2 | 2023 | 自监督（无标签）。最好的通用视觉特征。 |
| ViT-22B | 2023 | 22B 参数；扩展定律适用。 |
| SigLIP | 2023 | ViT + 语言对，sigmoid 对比损失。 |
| SAM 3 | 2025 | 分割一切；ViT-Large + 可提示掩码解码器。 |

### 为什么花了一段时间

ViT 需要*大量*数据才能匹配 CNN，因为它没有 CNN 的归纳偏置（平移不变性、局部性）。没有 >100M 标记图像或强大的自监督预训练，CNN 在匹配算力下仍然胜出。DeiT 在 2021 年用蒸馏技巧解决了这个问题；DINOv2 在 2023 年用自监督永久解决了这个问题。

## 从零实现

参见 `code/main.py`。纯标准库 patchify + 线性 embedding + 完整性检查。无训练——任何现实规模的 ViT 需要 PyTorch 和数小时的 GPU 时间。

### 第 1 步：假图像

一张 24 × 24 的 RGB 图像，作为 `(R, G, B)` 元组行的列表。我们使用 6×6 patches → 16 个 patches，每个 108 维 embedding 向量。

### 第 2 步：patchify

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

光栅顺序：沿网格的行主序。每个 ViT 都使用这个顺序。

### 第 3 步：线性 embed

将每个扁平 patch 乘以随机 `(patch_flat_size, d_model)` 矩阵。验证输出形状在前置 `[CLS]` 后是 `(N_patches + 1, d_model)`。

### 第 4 步：为一个现实的 ViT 统计参数量

打印 ViT-Base 的参数量：12 层，12 个头，d=768，patch=16。与 ResNet-50（约 25M）比较。ViT-Base 约 86M。ViT-Large 约 307M。ViT-Huge 约 632M。

## 实际使用

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # 图像表示
```

**DINOv2 embedding 是 2026 年图像特征的默认选择。** 冻结主干，训练一个很小的头。适用于分类、检索、检测、描述。Meta 的 DINOv2 检查点在每个非文本视觉任务上优于 CLIP。

**Patch 大小选择。** 小模型使用 16×16（ViT-B/16）。密集预测（分割）使用 8×8 或 14×14（SAM、DINOv2）。非常大的模型使用 14×14。

## 交付物

参见 `outputs/skill-vit-configurator.md`。这个 skill 根据数据集大小、分辨率和算力预算为一个新视觉任务选择 ViT 变体和 patch 大小。

## 练习

1. **简单。** 运行 `code/main.py`。验证 patch 数等于 `(H/P) * (W/P)` 且扁平 patch 维度等于 `P*P*C`。
2. **中等。** 实现 2D 正弦位置 embedding——两个独立的正弦码用于每个 patch 的 `row` 和 `col`，拼接。将它们送入一个小型 PyTorch ViT 并比较在 CIFAR-10 上与可学习位置 embedding 的准确率。
3. **困难。** 构建一个 3 层 ViT（PyTorch），在 1,000 张 MNIST 图像上训练，使用 4×4 patches。测量测试准确率。现在在同一 1,000 张图像上添加 DINOv2 预训练（简化：只训练编码器从掩码 patches 预测 patch embedding）。准确率有提高吗？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| Patch | "视觉 transformer 的 token" | 图像 `P × P × C` 区域的像素值的扁平向量。 |
| Patchify | "切 + 扁平化" | 将图像切片为不重叠的 patches，扁平化每个为向量。 |
| `[CLS]` token | "图像摘要" | 前置的可学习 token；其最终 embedding 是图像表示。 |
| 归纳偏置 | "模型假设什么" | ViT 比 CNN 拥有更少的先验；需要更多数据来弥补差距。 |
| DINOv2 | "自监督 ViT" | 使用图像增强 + 动量教师训练，无需标签。2026 年最好的通用图像特征。 |
| SigLIP | "CLIP 的继任者" | ViT + 文本编码器，用 sigmoid 对比损失训练；在匹配算力下优于 CLIP。 |
| Swin | "窗口化 ViT" | 带局部注意力和移位窗口的分层 ViT；亚二次方。 |
| Register tokens | "2023 年技巧" | 几个额外的可学习 token，吸收注意力 sinks；改善 DINOv2 特征。 |

## 延伸阅读

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT 论文。
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT。
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin。
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2。
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — DINOv2 的 register-token 修复。