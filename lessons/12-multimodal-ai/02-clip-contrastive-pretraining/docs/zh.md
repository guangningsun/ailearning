# CLIP 与对比视觉-语言预训练

> OpenAI 的 CLIP（2021）证明了一个点子足够大，可以支撑接下来五年：仅用嘈杂的网络图文对和对比损失，将图像编码器和文本编码器对齐到同一向量空间。零监督标签。4 亿对。最终的 embedding 空间可以做零样本分类、图文检索，并作为每个 2026 年 VLM 的视觉塔插入。SigLIP 2（2025）用 sigmoid 替换 softmax，以更低成本超越了 CLIP。本课从 InfoNCE 到 sigmoid pairwise loss 走一遍数学，并用标准库 Python 实现训练步骤。

**类型：** 构建型
**语言：** Python（标准库，InfoNCE + sigmoid loss 实现）
**前置条件：** Phase 12·01（ViT patches）、Phase 7（Transformer）
**时间：** 约 180 分钟

## 学习目标

- 从互信息推导 InfoNCE 损失并实现一个数值稳定的向量化版本。
- 解释为什么 sigmoid pairwise loss（SigLIP）可以扩展到 batch 32768+ 而无需 softmax 所需的 all-gather 开销。
- 通过构造文本模板（`a photo of a {class}`）并对余弦相似度取 argmax，运行零样本 ImageNet 分类。
- 说出 CLIP / SigLIP 预训练给你的四个杠杆：batch size、温度、prompt 模板、数据质量。

## 问题

CLIP 之前的视觉是监督的。收集带标签数据集（ImageNet：120 万图像，1000 类），训练 CNN，交付。标签昂贵、标签有偏（偏向标注者能达成共识的东西）、标签不经过微调无法迁移到新任务。

网络上的图文对有超过十亿个松散标注的配对。一张带着 alt 文本"my dog Max in the park"的金毛照片带着监督信号——文本描述了图像。问题：你能把它转化成有用的训练吗？

CLIP 的回答：把图文对当作匹配任务。给定 N 张图像和 N 条描述，学习把每张图像匹配到它自己的描述，对抗 N-1 个干扰项。监督信号是"这两个是一伙的；这 N-1 个不是"。没有类标签。没有人工标注。只有一个对比损失。

由此产生的 embedding 空间做的事比 CLIP 训练时更多。ImageNet 零样本有效是因为"a photo of a cat"与从未被显式标注为猫的猫图片 embedding 在空间中接近。这赌注催生了 2026 年的每一个 VLM。

## 概念

### 双编码器

CLIP 有两个塔：

- 图像编码器 `f`：ViT 或 ResNet，每张图像输出一个 D 维向量。
- 文本编码器 `g`：小型 transformer，每条描述输出一个 D 维向量。

两个塔都把输出归一化到单位长度。相似度是 `cos(f(x), g(y)) = f(x)^T g(y)`，因为两者都是单位范数。

对于一个 N 对（图像，描述）的批次，构建形状为 `(N, N)` 的相似度矩阵 `S`：

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

其中 `tau` 是学习到的温度（CLIP 初始化为 0.07；在对数空间学习）。

### InfoNCE 损失

CLIP 使用对称交叉熵，行列各一份：

```
loss_i2t = CE(S, labels=identity)     # 每张图像的正样本是它自己的描述
loss_t2i = CE(S^T, labels=identity)   # 每条描述的正样本是它自己的图像
loss = (loss_i2t + loss_t2i) / 2
```

这就是 InfoNCE。CE 中的 softmax 迫使每张图像匹配自己的描述，而匹配其他描述的概率更低。"负样本"是批次中的所有其他样本。更大的 batch = 更多的负样本 = 更强的信号。CLIP 在 batch 32k 上训练；规模很关键。

### 温度

`tau` 控制 softmax 的锐度。低 tau → 尖锐分布，硬负样本挖掘效果。高 tau → 柔和，所有样本都有贡献。CLIP 学习 log(1/tau)，做截断以防止崩溃到接近零的 tau。SigLIP 2 固定初始 tau，改用学习到的偏置。

### 为什么 sigmoid 扩展更好（SigLIP）

Softmax 需要整个相似度矩阵同步。在分布式训练中，你必须把所有 embedding all-gather 到每个副本，然后做 softmax。这在通信量上是二次方增长（按 world size）。

SigLIP 把 softmax 替换为逐元素 sigmoid：对于每对 `(i, j)`，损失是"这是否为匹配对"的二元分类——正类标签是对角线，其他都是负类。损失是：

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` 当且仅当 `i == j`，否则为 0。每对的损失是独立的。无需 all-gather。每个 GPU 计算其本地块然后求和。SigLIP 2 以低成本扩展到 batch 32k-512k，而 CLIP 需要按比例更多的通信。

### 零样本分类

给定 N 个类名，对每个类构造文本模板：

```
"a photo of a {class}"
```

用文本编码器 embedding 每个模板。用图像编码器 embedding 你的图像。余弦相似度取 argmax = 预测类别。不在目标类上做任何训练。

Prompt 模板很重要。CLIP 原始论文每类用了 80 个模板（plain、artistic、photo、painting 等）并对 embedding 做平均。+3 ImageNet 分数。现代用法通常只选一到两个模板。

### 线性探测与微调

零样本是基线。线性探测（在冻结 CLIP 特征上为你的目标类训练一个线性层）在域内任务上优于零样本。全量微调在域内优于线性探测，但可能损害零样本迁移。三个 regime，三种权衡。

### SigLIP 2：NaFlex 与密集特征

SigLIP 2（2025）新增：
- NaFlex：单一模型处理可变宽高比和分辨率。
- 更好的密集特征用于分割和深度估计，目标是在 VLM 中作为冻结 backbone 使用。
- 多语言：在 100 多种语言上训练，而 CLIP 只支持英语。
- 10 亿参数规模，而 CLIP 最多 4 亿。

在 2026 年的开源 VLM 中，SigLIP 2 SO400m/14 是默认视觉塔。CLIP 仍然是纯图文检索的默认选择，因为特定的 LAION-2B 训练分布与你的查询模式匹配。

### ALIGN、BASIC、OpenCLIP、EVA-CLIP

ALIGN（Google，2021）：与 CLIP 相同的思路，18 亿对规模，90% 噪声。证明了噪声数据可以扩展。OpenCLIP（LAION）：在 LAION-400M / 2B 上开源复现 CLIP，多种规模，是首选开源 checkpoint。EVA-CLIP：从掩码图像建模初始化；在 VLM 中是强 backbone。BASIC：Google 的 CLIP+ALIGN 混合体。同一个家族，不同的数据和调优。

### 零样本天花板

CLIP 类模型在 ImageNet 零样本上上限约为 76%（CLIP-G、OpenCLIP-G）。超越需要更大规模的数据（SigLIP 2 达到 80%+）或架构变化（监督头、更多参数）。基准在饱和；真正的价值在于下游 VLM 消耗的 embedding 空间。

## 使用方法

`code/main.py` 实现：

1. 一个玩具双编码器（基于哈希的图像特征，文本字符特征），让你可以看到 InfoNCE 的形状而不依赖 numpy。
2. 纯 Python 的 InfoNCE 损失（通过 log-sum-exp 实现数值稳定）。
3. 用于对比的 sigmoid pairwise loss。
4. 零样本分类例程：计算与一组文本 prompt 的余弦相似度，取 argmax 得到预测。

运行它并观察损失曲线。绝对数字是玩具级的；但形状与真实 CLIP 训练器发出的曲线一致。

## 交付物

本课产出 `outputs/skill-clip-zero-shot.md`。给定一组图像（通过路径）和目标类列表，用 CLIP 模板构造文本 prompt，用指定 checkpoint（如 `openai/clip-vit-large-patch14`）embedding 两端，返回 top-1 / top-5 预测及相似度分数。该 skill 拒绝对不在 prompt 列表中的类做任何声称。

## 练习

1. 手算 4 对 batch 的 InfoNCE。构造 4×4 相似度矩阵，跑 softmax，取对角线，计算交叉熵。用手算验证你的 Python 实现。

2. SigLIP 额外地使用一个偏置参数 `b` 而不只是温度：`S'[i,j] = S[i,j]/tau + b`。当批次有大的类不平衡（每行负样本远多于正样本）时，`b` 扮演什么角色？阅读 SigLIP 第 3 节（arXiv:2303.15343）。

3. 为猫 vs 狗构建零样本分类器。尝试两个 prompt 模板：`a photo of a {class}` 和 `a picture of a {class}`。在 100 张测试图像上测量准确率。模板集成是否胜过单个？

4. 计算 512-GPU 运行在 batch 32k 下 softmax InfoNCE vs sigmoid pairwise 的通信成本。哪个是 O(N)，哪个是 O(N²)？引用 SigLIP 第 4 节。

5. 阅读 OpenCLIP 扩展定律论文（arXiv:2212.07143，Cherti 等）。从图中复现他们的结论：在固定模型规模下，ImageNet 零样本准确率与训练数据规模之间的对数线性关系是什么？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| InfoNCE | "对比损失" | 在批次相似度矩阵上的交叉熵；每个样本的正样本是其配对样本，负样本是其他所有 |
| Sigmoid loss | "SigLIP 损失" | 逐对二元交叉熵；无 softmax，无 all-gather，在分布式训练中扩展成本低 |
| Temperature | "tau" | 在 softmax/sigmoid 前缩放 logit 的标量；控制分布的锐度 |
| Zero-shot | "无微调分类" | 用文本 prompt 构造类 embedding，通过余弦相似度分类；不在目标类上训练 |
| Prompt template | "a photo of a ..." | 围绕类名的文本框架；影响零样本准确率 1-5 个百分点 |
| Dual encoder | "双塔" | 一个图像编码器 + 一个文本编码器，输出在共享 D 维空间 |
| Hard negative | "难缠的干扰项" | 一个与正样本足够相似的负样本，模型必须努力才能将它们分开 |
| Linear probe | "冻结 + 一层" | 只在冻结特征上训练一个线性分类器；衡量特征质量 |
| NaFlex | "原生灵活分辨率" | SigLIP 2 的能力，可以以任何宽高比和分辨率摄入图像，无需 resize |
| Temperature scaling | "对数参数化 tau" | CLIP 参数化 `log(1/tau)` 使梯度行为合理；做截断以防止 tau 崩溃到接近零 |

## 延伸阅读

- [Radford 等 — Learning Transferable Visual Models From Natural Language Supervision（arXiv:2103.00020）](https://arxiv.org/abs/2103.00020) — CLIP 论文。
- [Zhai 等 — Sigmoid Loss for Language Image Pre-Training（arXiv:2303.15343）](https://arxiv.org/abs/2303.15343) — SigLIP。
- [Tschannen 等 — SigLIP 2（arXiv:2502.14786）](https://arxiv.org/abs/2502.14786) — 多语言 + NaFlex。
- [Jia 等 — ALIGN（arXiv:2102.05918）](https://arxiv.org/abs/2102.05918) — 用嘈杂网络数据扩展。
- [Cherti 等 — Reproducible scaling laws for contrastive language-image learning（arXiv:2212.07143）](https://arxiv.org/abs/2212.07143) — OpenCLIP 扩展定律。