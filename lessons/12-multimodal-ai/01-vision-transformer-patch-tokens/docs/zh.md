# Vision Transformer 与 Patch-Token 原语

> 在任何多模态之前，一张图像必须先变成 transformer 能消化的一串 token。2020 年的 ViT 论文用 16×16 像素块、线性投影和位置嵌入回答了这个问题。五年后的 2026 年，每一款前沿模型（1536px 原生的 Claude Opus 4.7、Gemini 3.1 Pro、Qwen3.5-Omni）仍然以此开头——编码器从 ViT 换成了 DINOv2 再换成 SigLIP 2，加入了 register token，位置编码变成了 2D-RoPE，但这个原语没有变。本课从头到尾走一遍 patch-token 流水线，并用标准库 Python 从零实现它，这样 Phase 12 的其余内容就有了"视觉 token"的具体 mental model。

**类型：** 学习型
**语言：** Python（标准库，patch tokenizer + 几何计算器）
**前置条件：** Phase 7（Transformer）、Phase 4（计算机视觉）
**时间：** 约 120 分钟

## 学习目标

- 将 H×W×3 的图像转换为带有正确位置编码的 patch token 序列。
- 给定（patch size、resolution、hidden dim、depth），计算序列长度、参数量和 FLOPs。
- 说出让 ViT 从 2020 研究到 2026 生产环境的三次升级：自监督预训练（DINO / MAE）、register token、原生分辨率打包。
- 为下游任务在 CLS pooling、mean pooling 和 register token 之间做出选择。

## 问题

Transformer 操作的是向量序列。文本天然是序列（字节或 token）。图像是带有三个颜色通道的 2D 像素网格——不是序列。如果你把每个像素展平，224×224 的 RGB 图像会变成 150,528 个 token，在这个长度上做自注意力简直是非主流（序列长度的二次方）。

2020 年之前的方案是在前面接一个 CNN 特征提取器：ResNet 产生一个 7×7 的 2048 维向量特征图，把这 49 个 token 送进 transformer。这能跑，但继承了 CNN 的归纳偏置（平移等变性、局部感受野），也丢掉了 transformer 的规模胃口。

Dosovitskiy 等人（2020）问了个直白的问题：如果跳过 CNN 呢？把图像切成固定大小的 patch（如 16×16 像素），对每个 patch 做线性投影得到一个向量，加上位置嵌入，然后把序列喂给普通 transformer。当时这是异端——视觉不用卷积。但有了足够的数据（JFT-300M，然后是 LAION），它在 ImageNet 上击败了 ResNet，而且还在继续进步。

到 2026 年，ViT 原语已经是不容置疑的基础。每个开源权重的 VLM 的视觉塔都是某种后代（DINOv2、SigLIP 2、CLIP、EVA、InternViT）。问题不再是"要不要用 patch"，而是"用多大 patch size、什么分辨率策略、什么预训练目标、什么位置编码"。

## 概念

### Patch 即 token

给定形状为 `(H, W, 3)` 的图像 `x` 和 patch size `P`，把图像切分成 `(H/P) × (W/P)` 个不重叠的 patch。每个 patch 是一个 `P × P × 3` 的像素立方体。将每个立方体展平为 `3P²` 维向量。施加一个共享的线性投影 `W_E`（形状 `(3P², D)`），将每个 patch 映射到模型的隐藏维度 `D`。

以 ViT-B/16 标准配置为例：
- 分辨率 224，patch size 16 → 网格 14×14 → 196 个 patch token。
- 每个 patch 是 `16×16×3 = 768` 个像素值，投影到 `D = 768`。
- 加一个可学习的 `[CLS]` token → 序列长度 197。

Patch 投影在数学上等价于一个 2D 卷积，kernel size 为 `P`，stride 为 `P`，输出通道为 `D`。这才是生产代码的真正实现方式——`nn.Conv2d(3, D, kernel_size=P, stride=P)`。"线性投影"的表述是概念层面的；kernel 的表述是效率层面的。

### 位置嵌入

Patch 本身没有固定顺序——transformer 把它们看成是一个 bag。早期的 ViT 加了一个可学习的 1D 位置嵌入（每个位置一个 768 维向量，共 197 个）。能用，但把模型绑死在了训练分辨率上：如果你在推理时改变网格，就必须对位置表做插值。

现代视觉 backbone 使用 2D-RoPE（Qwen2-VL 的 M-RoPE，SigLIP 2 的默认选项）或分解的 2D 位置。2D-RoPE 根据 patch 的（行号，列号）索引旋转 query 和 key 向量，这样模型从旋转角度推断出相对的 2D 位置。无需位置表。模型可以在推理时处理任意大小的网格。

### CLS token、池化输出与 Register Token

图像级的表示是什么？三种选择并存：

1. `[CLS]` token。在 patch 序列前面加一个可学习向量。经过所有 transformer 块后，CLS token 的隐藏状态就是图像表示。继承自 BERT。由原始 ViT、CLIP 使用。
2. Mean pool。把 patch token 输出隐藏状态做平均。由 SigLIP、DINOv2、大多数现代 VLM 使用。
3. Register token。Darcet 等人（2023）观察到，没有显式 sink token 训练的 ViT 会产生高 norm 的"伪影"patch，劫持自注意力。加入 4–16 个可学习 register token 吸收这个负担，提升了密集预测质量（分割、深度）。DINOv2 和 SigLIP 2 都带了 register。

这个选择对下游任务很重要。CLS 适合分类。对于把 patch token 输入 LLM 的 VLM，你根本不做池化——每个 patch 都成为 LLM 的一个输入 token。Register 在移交前被丢弃（它们是脚手架，不是内容）。

### 预训练：监督、对比、掩码、自蒸馏

2020 年的 ViT 在 JFT-300M 上用监督分类做预训练。很快被取代：

- CLIP（2021）：在 4 亿对图像-文本对上做对比。参考 12.02 课。
- MAE（2021，He 等）：掩码 75% 的 patch，重建像素。自监督，在纯图像上有效。
- DINO（2021）/ DINOv2（2023）：学生-教师自蒸馏，无需标签，无需描述。2023 年的 DINOv2 ViT-g/14 是最强的纯视觉 backbone，也是"密集特征"用例的默认选择。
- SigLIP / SigLIP 2（2023，2025）：用 sigmoid 损失和 NaFlex 做 CLIP，配合原生宽高比。2026 年开源 VLM（Qwen、Idefics2、LLaVA-OneVision）的主导视觉塔。

预训练方式的选择决定了 backbone 的用途：CLIP/SigLIP 用于与文本的语义匹配，DINOv2 用于密集视觉特征，MAE 作为下游微调的起点。

### 扩展定律

ViT 扩展定律（Zhai 等，2022）确定了一个 ViT 的质量在模型规模、数据规模和计算量上服从可预测的定律。在固定计算量下：
- 更大的模型 + 更多数据 → 更好的质量。
- Patch size 是序列长度与保真度之间的杠杆。Patch 14（DINOv2/SigLIP SO400m 的典型值）比 patch 16 每张图像产生更多 token；对 OCR 和密集任务更好，对速度更差。
- 分辨率是另一个大杠杆。从 224 到 384 再到 512 几乎总是有帮助的，但 FLOPs 是二次方增长。

ViT-g/14（1B 参数，patch 14，分辨率 224 → 256 token）和 SigLIP SO400m/14（400M 参数，patch 14）是 2026 年开源 VLM 的两个主力编码器。

### ViT 的参数量计算

完整计算在 `code/main.py` 中。以 224 分辨率的 ViT-B/16 为例：

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

加载 checkpoint 前先用这种方式估算每个 ViT。Backbone 大小决定了任何下游 VLM 的 VRAM 下限。

### 2026 生产配置

2026 年大多数开源 VLM 附带的编码器是 SigLIP 2 SO400m/14，采用原生分辨率（NaFlex）。它有：
- 4 亿参数。
- Patch size 14，默认分辨率 384 → 每张图像 729 个 patch token。
- 用于图像级任务时做 mean pool；做 VQA 时全部 729 个 patch 送入 LLM。
- 4 个 register token，在移交 LLM 前丢弃。
- 2D-RoPE，带图像级缩放以适应原生宽高比。

这个配置中的每一个决策都可以追溯到一篇你可以阅读的论文。

## 使用方法

`code/main.py` 是一个 patch tokenizer 和几何计算器。它接收（图像 H、W，patch P，hidden D，depth L）并报告：

- 补片后的网格形状和序列长度。
- 合成 8×8 像素玩具图像的 token 序列（走一遍展平 + 投影路径）。
- 按 patch embed、position embed、transformer blocks 和 head 分拆的参数量。
- 在目标分辨率下每次前向的 FLOPs。
- ViT-B/16 @ 224、ViT-L/14 @ 336、DINOv2 ViT-g/14 @ 224、SigLIP SO400m/14 @ 384 的对比表。

运行它。把参数量与已发布数字对照。用不同的 patch size 和分辨率来感受 token 数量的代价。

## 交付物

本课产出 `outputs/skill-patch-geometry-reader.md`。给定 ViT 配置（patch size、resolution、hidden dim、depth），它生成 token 数量、参数量和 VRAM 估算，并附带理由。每当你为 VLM 选择视觉 backbone 时使用这个 skill——它防止"token 爆炸、LLM context 溢出"的意外。

## 练习

1. 计算 Qwen2.5-VL 在原生 1280×720 输入、patch size 14 下的 patch-token 序列长度。与仅用 CLS 表示相比如何？

2. 一帧 1080p（1920×1080）在 patch 14 下产生多少 token？以 30 FPS 处理 5 分钟视频，总视觉 token 是多少？哪种节省最大：池化、帧采样还是 token 合并？

3. 用纯 Python 实现 patch token 上的 mean pooling。验证对 196 个 DINOv2 输出的 token 做 mean pool 得到的结果与模型在请求 pooled embedding 时的 `forward` 返回值一致。

4. 阅读"Vision Transformers Need Registers"（arXiv:2309.16588）第 3 节。用两句话描述 register 吸收了什么伪影，以及为什么这对下游密集预测重要。

5. 修改 `code/main.py` 以支持 patch-n'-pack：给定不同分辨率的图像列表，产生一个打包序列和块对角注意力掩码。等你学到 12.06 课时用它做验证。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| Patch | "16×16 像素方块" | 输入图像的一个固定大小不重叠区域；成为一个 token |
| Patch embedding | "线性投影" | 一个共享学习矩阵（或 stride=P 的 Conv2d），将展平的 patch 像素映射到 D 维向量 |
| CLS token | "类别 token" | 前置的可学习向量，其最终隐藏状态代表整张图像；在 2026 年已可选 |
| Register token | "Sink token" | 额外的可学习 token，吸收 ViT 在预训练过程中产生的高 norm 注意力伪影 |
| Position embedding | "位置信息" | 每个位置的向量或旋转，使序列具有顺序感知；2D-RoPE 是现代默认 |
| Grid | "Patch 网格" | 给定分辨率和 patch size 下的 (H/P) × (W/P) 2D patch 数组 |
| NaFlex | "原生灵活分辨率" | SigLIP 2 特性：一个模型无需重训练即可处理多种宽高比和分辨率 |
| Backbone | "视觉塔" | 预训练的图像编码器，其 patch-token 输出在 VLM 中馈入 LLM |
| Pooling | "图像级摘要" | 将 patch token 转化为一个向量的策略：CLS、mean、attention pool 或 register-based |
| Patch 14 vs 16 | "更细 vs 更粗网格" | Patch 14 每张图像产生更多 token，对 OCR 保真度更好但更慢；patch 16 是经典默认 |

## 延伸阅读

- [Dosovitskiy 等 — An Image is Worth 16x16 Words（arXiv:2010.11929）](https://arxiv.org/abs/2010.11929) — 原始 ViT。
- [He 等 — Masked Autoencoders Are Scalable Vision Learners（arXiv:2111.06377）](https://arxiv.org/abs/2111.06377) — MAE，自监督预训练。
- [Oquab 等 — DINOv2（arXiv:2304.07193）](https://arxiv.org/abs/2304.07193) — 大规模自蒸馏，无需标签。
- [Darcet 等 — Vision Transformers Need Registers（arXiv:2309.16588）](https://arxiv.org/abs/2309.16588) — register token 与伪影分析。
- [Tschannen 等 — SigLIP 2（arXiv:2502.14786）](https://arxiv.org/abs/2502.14786) — 2026 年默认视觉塔。
- [Zhai 等 — Scaling Vision Transformers（arXiv:2106.04560）](https://arxiv.org/abs/2106.04560) — 经验扩展定律。