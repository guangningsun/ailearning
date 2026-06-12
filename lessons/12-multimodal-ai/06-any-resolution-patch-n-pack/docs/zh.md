# 任意分辨率视觉：Patch-n'-Pack 与 NaFlex

> 真实世界的图像不是 224x224 的正方形。收据是 9:16，图表是 16:9，医学扫描图可能是 4096x4096，手机截图是 9:19.5。2024 年之前的 VLM 方案——把所有图像缩放到固定正方形——丢弃了让 OCR、文档理解和超高分辨率场景解析得以成功的信号。NaViT（Google，2023）证明了你可以用块对角遮罩将可变分辨率的 patch 打包进单个 transformer batch 中。Qwen2-VL 的 M-RoPE（2024）彻底抛弃了绝对位置表。LLaVA-NeXT 的 AnyRes 将高分辨率图像切分为 base + 子图像的瓦片格式。SigLIP 2 的 NaFlex 变体（2025）现在成了追求单一 checkpoint 即可服务任意宽高比的开源 VLM 默认编码器。本节从头实现 patch-n'-pack。

**类型：** 构建
**语言：** Python（标准库，patch packer + 块对角遮罩）
**前置条件：** 阶段 12 · 01（ViT patches）、阶段 12 · 05（LLaVA）
**时间：** 约 120 分钟

## 学习目标

- 将一批可变分辨率图像的 patches 打包成一个序列，并构建块对角注意力遮罩。
- 根据任务选择 AnyRes 瓦片（LLaVA-NeXT）、NaFlex（SigLIP 2）或 M-RoPE（Qwen2-VL）。
- 计算 OCR、图表和摄影场景下的 token 预算，不依赖缩放。
- 说出方形缩放的三个失败模式：压扁的文本、截断的内容、填充造成的 token 浪费。

## 问题

Transformer 接收的是一个序列。Batch 是一叠等长的序列。如果你的图像都是 224x224，每次得到 196 个 patch tokens，填充不需要，问题解决。在 224 上训练，在 224 上推理，不再考虑分辨率。

但现实世界不配合。文档是竖版的（8.5×11 英寸，约 2:3）。图表截图是横版的（16:9）。收据又高又窄（1:3）。医学影像以 2048×2048 或更大尺寸输出。手机截图是 1170×2532（0.46:1）。

三种 2024 年前的方案及其各自的失败原因：

1. 缩放到固定正方形（224×224 或 336×336）。压扁会扭曲文本和人脸。下采样会摧毁图表标签和 OCR 内容。直到 LLaVA-1.5 都是行业惯例。
2. 裁剪到固定宽高比。你丢弃了大部分图像，而裁剪位置的选择本身就是一个独立的视觉问题。
3. 填充到最长边。解决了扭曲但对竖版图像浪费 50%+ 的 tokens。所有那些填充 tokens 上的二次方注意力成本。

2024-2025 年的答案：让 transformer 以图像的原始分辨率接收 patches，然后想办法将异构 batch 打包进一个序列，不浪费算力。

## 概念

### NaViT 与 patch-n'-pack

NaViT（Dehghani 等，2023）是一篇证明了此方案可以规模化运行的论文。思路很机械：

1. 对 batch 中的每张图像，在选定的 patch size（如 14）下计算其原始 patch 网格。
2. 将每张图像的 patches 展平为其自身的变长序列。
3. 将所有图像的 patches 拼接成一个长序列。
4. 构建块对角注意力遮罩，使图像 A 的 patches 只在图像 A 内部进行注意力计算。
5. 携带每个 patch 的位置信息（2D RoPE 或分数位置嵌入）。

三张图像的 batch——336×336（576 tokens）、224×224（256 tokens）和 448×336（768 tokens）——变成一个 1600-token 的序列，配以 1600×1600 的块对角遮罩。无填充。无算力浪费。Transformer 处理任意宽高比。

NaViT 还引入了训练时的分数 patch dropping——在 batch 中随机丢弃 50% 的 patches——这既做了正则化又加速了训练。SigLIP 2 继承了这一设计。

### AnyRes（LLaVA-NeXT）

LLaVA-NeXT 的 AnyRes 是务实的替代方案。给定一张高分辨率图像和一个固定编码器（CLIP 或 SigLIP，分辨率 336），将图像进行瓦片切分：

1. 从预定义的网格布局集合——(1x1)、(1x2)、(2x1)、(1x3)、(3x1)、(2x2) 等——中选择最适合图像宽高比的网格。
2. 将整张图像瓦片切分到网格中；每个瓦片成为一个 336×336 的 crop。
3. 同时生成一张缩略图：将整张图像缩放到 336×336 作为全局上下文 token。
4. 用冻结的 336 编码器编码每个瓦片。拼接瓦片 tokens 和缩略图 tokens。

对于 672×672 图像采用 2x2 网格加缩略图：4 × 576 + 576 = 2880 个视觉 tokens。代价高昂但有效——LLM 同时看到局部细节和全局上下文。

当编码器被冻结且只支持一种分辨率时，AnyRes 是首选方案。它会让大图像的 token 数量爆炸（1344×1344 图像在 4x4 网格下是 9216 + 576 ≈ 9800 tokens，几乎占满 8k LLM 上下文）。

### M-RoPE（Qwen2-VL）

Qwen2-VL 引入了多模态旋转位置嵌入。与 NaViT 的分数位置或 AnyRes 的瓦片+缩略图不同，每个 patch 携带一个 3D 位置（时间、高度、宽度）。Query/Key 旋转处理任意的 H、W 和时间长度。

M-RoPE 原生支持动态分辨率，无需重新训练。推理时输入任意 HxW 图像，patch embedder 产生 H/14 × W/14 个 tokens，每个 token 获得其 (t=0, r=row, c=col) 位置，RoPE 用正确的频率旋转注意力，完成。Qwen2.5-VL 和 Qwen3-VL 延续了此设计。InternVL3 的 V2PE 是同一理念，但每个模态使用不同的变量编码。

与 AnyRes 不同，M-RoPE 的 token 数量为 O(H × W / P²)，即原生分辨率——没有倍增的瓦片开销。与 NaViT 不同，它仍然期望每次前向传播处理单张图像。跨分辨率的 batch 仍需要在 patch-n'-pack 之上实现。

### NaFlex（SigLIP 2）

NaFlex 是 SigLIP 2 checkpoint 的原生灵活模式。单个模型在推理时服务多种序列长度（256、729、1024 tokens）。内部在训练时使用 NaViT 风格的 patch-n'-pack，并为每个 patch 使用绝对分数位置。卖点：一个 checkpoint，根据任务在推理时选择 token 预算。

对于语义任务（分类、检索），256 tokens。对于 OCR 或图表理解，1024 tokens。无需重新训练。

### 打包遮罩

块对角遮罩是大多数实现卡住的地方。对于覆盖图像 i=0..B-1、总长度为 N_total 的打包序列，长度为 n_i，遮罩 M 形状为 (N_total, N_total)：当两个索引都落在同一图像的块内时为 1，否则为 0。你可以通过累计长度列表来构建它：

```
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 iff there exists b where offsets[b] <= i < offsets[b+1] and offsets[b] <= j < offsets[b+1]
```

在 PyTorch 中用 `torch.block_diag` 或显式 gather 一行就能搞定。FlashAttention 的变长路径（`cu_seqlens`）完全跳过遮罩，直接用累计长度 tensor 在序列内做注意力——对典型 batch 比密集遮罩快约 10 倍。

### Token 预算

根据任务选择策略：

- OCR / 文档：1024-4096 tokens。SigLIP 2 NaFlex 在 1024，或 AnyRes 3x3 + 缩略图。
- 图表和 UI：729-1024 tokens，在 384-448 原生分辨率。Qwen2.5-VL 动态分辨率加上 max pixels 上限。
- 自然照片：256-576 tokens 就够了。下游 LLM 看到的信息足够。在内容密度高的地方付费买 tokens。
- 视频：空间池化后每帧 64-128 tokens，2-8 FPS。本节 12.17 涵盖此内容。

2026 年的生产规则：选择每个任务的 max-pixels 上限，在该上限内以原生宽高比编码，打包 batch，跳过填充。Qwen2.5-VL 暴露了 `min_pixels` 和 `max_pixels` 来精确控制这个旋钮。

## 使用它

`code/main.py` 实现了异构图像 batch 的 patch-n'-pack，使用整数像素坐标。它：

- 接收一系列 (H, W) 图像尺寸。
- 计算每张图像在 patch size 14 下的 patch 序列长度。
- 将它们打包成一个总长度为 `sum(n_i)` 的序列。
- 构建块对角注意力遮罩（密集版，为清晰起见）。
- 比较打包方案 vs 方形缩放和 AnyRes 瓦片的成本。
- 打印混合 batch（收据、图表、截图、照片）的 token 预算表。

运行它。输出的数字就是每个 2026 年开源 VLM 使用 patch-n'-pack 的原因。

## 交付它

本节产出 `outputs/skill-resolution-budget-planner.md`。给定混合宽高比的工作负载（OCR、图表、照片、视频帧）和总 token 预算，它选择正确的策略（NaFlex、AnyRes、M-RoPE 或固定正方形）并发出每个请求的配置。在为产品评估 VLM 时使用此 skill——它防止让延迟预算崩溃的静默 10 倍 token 爆炸。

## 练习

1. 一张收据是 600×1500（1:2.5）。在 patch size 14 下，原生分辨率有多少个 tokens？方形缩放到 336 后有多少？哪个在实践中丢失了更多 OCR 精度？
2. 为一批四张图像构建块对角遮罩，长度分别为 256、576、729、1024。验证注意力矩阵是 2585×2585，且恰好有 `256² + 576² + 729² + 1024²` 个非零元素。
3. 对于一张 1792×896 的图像在 patch 14 下，比较：(a) 缩放到 336 后编码，(b) AnyRes 2x1 + 缩略图，(c) 原生分辨率的 M-RoPE。哪个用的 tokens 最少？哪个保留了最多细节？
4. 实现分数 patch dropping：给定一个打包序列，随机均匀丢弃 50% 的 tokens，并相应更新块对角遮罩。测量遮罩的稀疏度变化。
5. 阅读 Qwen2-VL 论文第 3.2 节（arXiv:2409.12191）。用两句话描述 `min_pixels` 和 `max_pixels` 控制什么，以及为什么两个边界都很重要。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|----------------------|
| Patch-n'-pack | "NaViT 风格打包" | 将来自不同图像的变长 patch 序列拼接成一个 batch 维度 |
| 块对角遮罩 | "打包遮罩" | 将每张图像的 patches 限制为只能与自己 attend、而不能与 pack 中相邻图像 attend 的注意力遮罩 |
| AnyRes | "LLaVA-NeXT 瓦片" | 将高分辨率图像切分为固定大小瓦片网格加全局缩略图；用固定编码器编码每个瓦片 |
| NaFlex | "SigLIP 2 原生灵活" | 单一 SigLIP 2 checkpoint，在推理时可服务 256/729/1024-token 预算，无需重新训练 |
| M-RoPE | "多模态 RoPE" | 3D 旋转位置编码（时间、行、列），处理任意的 H、W、T，无需位置表 |
| cu_seqlens | "FlashAttention 打包" | FlashAttention varlen 路径使用的累计长度 tensor，替代密集块对角遮罩 |
| min_pixels / max_pixels | "分辨率边界" | Qwen2.5-VL 每个请求的旋钮，对非常小或非常大的输入限制 token 数量 |
| 视觉 token 预算 | "每张图像多少 tokens" | 每张图像发出的 patch tokens 的粗略计数；决定 LLM 的 prompt 预算和注意力成本 |

## 延伸阅读

- [Dehghani 等 — Patch n' Pack: NaViT（arXiv:2307.06304）](https://arxiv.org/abs/2307.06304)
- [Wang 等 — Qwen2-VL（arXiv:2409.12191）](https://arxiv.org/abs/2409.12191)
- [Laurençon 等 — 构建视觉语言模型时什么很重要？（Idefics2，arXiv:2405.02246）](https://arxiv.org/abs/2405.02246)
- [Tschannen 等 — SigLIP 2（arXiv:2502.14786）](https://arxiv.org/abs/2502.14786)
- [Qwen 团队 — Qwen2.5-VL 技术报告（arXiv:2502.13923）](https://arxiv.org/abs/2502.13923）