# 从 CLIP 到 BLIP-2 — Q-Former 作为模态桥

> CLIP 对齐了图像和文本，但无法生成描述、回答问题或进行对话。BLIP-2（Salesforce，2023）用一个小型可训练桥解决了这个问题：32 个可学习 query 向量通过交叉注意力关注冻结 ViT 的特征，然后直接插入冻结 LLM 的输入流。1.88 亿参数的桥接器将一个 11B LLM 连接到 ViT-g/14。2026 年之前的每一个基于 adapter 的 VLM——MiniGPT-4、InstructBLIP、LLaVA 的表亲们——都是它的后代。本课阅读 Q-Former 的架构，解释它的两阶段训练，并构建一个玩具版本，将视觉 token 喂给冻结的文本解码器。

**类型：** 构建型
**语言：** Python（标准库，交叉注意力 + 可学习 query 演示）
**前置条件：** Phase 12·02（CLIP）、Phase 7（Transformer）
**时间：** 约 180 分钟

## 学习目标

- 解释为什么冻结视觉编码器和冻结 LLM 之间有一个可训练瓶颈，比端到端微调在成本和稳定性上更优。
- 实现一个交叉注意力块，其中固定的一组可学习 queries 关注外部图像特征。
- 走一遍 BLIP-2 的两阶段预训练：表示学习（ITC + ITM + ITG）然后生成学习（LM 损失，冻结解码器）。
- 将 Q-Former 与 LLaVA 使用的更简单的 MLP projector 比较，并论证各自何时胜出。

## 问题

你有一个产生每张图像 256 个 1408 维 patch token 的冻结 ViT。你有一个期望 4096 维 token embedding 的冻结 7B LLM。明显的桥——从 1408 到 4096 的线性层——能用，但把所有 256 个 patch token 送入 LLM 的 context 每张图像要多花 256 个 token。32 张图像一个批次，视觉模态单独就消耗 8192 个 token。

BLIP-2 的问题：能把 256-token 的图像表示压缩成少得多 token（比如 32 个）同时保留足够信息让 LLM 做描述、回答问题和推理图像吗？而且能不碰冻结的 backbone 来训练这个桥，把训练成本控制在只有桥的参数？

答案：Q-Former。32 个可学习的"query"向量对 ViT 的 patch token 做交叉注意力，产生 32 个 token 的视觉摘要供 LLM 消耗。共 1.88 亿参数。在接触 LLM 之前，用对比、匹配和生成目标来训练。

## 概念

### 可学习 queries

Q-Former 的核心技巧：不让 LLM 的文本 token 去关注图像 patch，而是引入一组新的 32 个可学习 query 向量 `Q`，让*它们*去关注图像 patch。Queries 是模型的参数——它们在训练中学习，对每张图像使用相同的 32 个 queries。

经过交叉注意力后，每个 query 持有一个图像的压缩摘要——"描述主要物体"、"描述背景"、"数物体数量"等。Queries 不会字面上专门化到语义标签上；它们学习的是能让下游损失下降的任何编码。

### 架构

Q-Former 是一个小型 transformer（12 层，约 1 亿参数），有两条路径：

1. Query 路径：32 个 query 向量流经自注意力（彼此之间），然后对冻结 ViT 的 patch token 做交叉注意力，然后是 FFN。
2. Text 路径：类似 BERT 的文本编码器，与 query 路径共享自注意力和 FFN 权重。Text 路径禁用交叉注意力。

训练时两条路径都运行。Queries 和文本通过共享自注意力交互，这意味着 queries 可以根据任务需要（ITM、ITG）以文本为条件。在 VLM 移交的推理时，只有 queries 流过，产生 32 个视觉 token。

### 两阶段训练

BLIP-2 预训练分两个阶段：

阶段 1：表示学习（无 LLM）。三个损失：
- ITC（图文对比）：池化 query token 与文本 CLS token 之间的 CLIP 式对比。
- ITM（图文匹配）：二元分类器——这个图文对是否匹配？硬负样本挖掘。
- ITG（图生文本生成）：文本上的因果 LM head，以 queries 为条件。迫使 queries 编码可生成文本的内容。

只有 Q-Former 训练。ViT 冻结。没有 LLM 参与。

阶段 2：生成学习。接上一个冻结 LLM（OPT-2.7B 或 Flan-T5-XL 等）。通过一个小型线性层将 32 个 query 输出投影到 LLM 的 embedding 维度。将其 prepend 到文本 prompt 上。只在拼接后的 prompt + 图像 + 描述序列上用 LM 损失训练线性投影和 Q-Former。

阶段 2 之后，Q-Former + 投影是完整的视觉 adapter。推理时：图像 → ViT → Q-Former → 线性投影 → prepend 到文本 → 冻结 LLM 发出输出。

### 参数经济

BLIP-2 + ViT-g/14（1.1B，冻结）+ OPT-6.7B（67 亿，冻结）+ Q-Former（1.88 亿，训练）= 80 亿总计，1.88 亿训练。Q-Former 单独约占整个栈参数的 2.4%。训练成本反映了这一点：在少数 A100 上几天 vs 端到端几周。

质量：BLIP-2 在零样本 VQA 上匹配或击败 Flamingo-80B，同时体积小 50 倍。桥是有效的。

### InstructBLIP 与指令感知的 Q-Former

InstructBLIP（2023）用额外输入扩展了 Q-Former：指令文本本身。在交叉注意力时，queries 现在同时获得图像 patch 和指令。Queries 可以按指令专门化（"数车"、"描述情绪"），而不是学习一个固定摘要。在保留任务上基准有提升。

### MiniGPT-4 与纯 projector 方法

MiniGPT-4 保留了 Q-Former，但只训练输出线性投影，冻结其他所有。便宜，但代价是质量——queries 是 BLIP-2 的，不是你自己的。适合快速迭代，不是最佳架构。

### 为什么 LLaVA 选得更简单

LLaVA（2023，12.05 课）用普通的 2 层 MLP 替换了 Q-Former，将每个 ViT patch token 投影到 LLM 空间——每张图像 576 个 token（24×24 网格），全部送入 LLM。压缩更差，但让 LLM 能关注原始 patch。当时这有争议；到 2023 年底它已成为主流，因为视觉指令数据（LLaVA-Instruct-150k）证明 MLP 可以被训练来保留足够信号。权衡：LLaVA 的 context 填充更快，但它自然扩展到多图像和视频。

到 2026 年领域分裂：Q-Former 在 token 预算紧张时（长视频、多个图像）存活；MLP projector 在质量-per-token 是优先时占主导。

### 门控交叉注意力：Flamingo，祖先

Flamingo（12.04 课）比 BLIP-2 更早，使用了相同的交叉注意力思路，但在每个冻结 LLM 层上都做，不只是作为一个桥。BLIP-2 表明你可以只压缩到输入层然后仍然有效。Gemini 和 Idefics 两者结合：交织输入 token 加上可选的门控交叉注意力用于上下文少样本。

### 2026 年的后代

- Q-Former：BLIP-2、InstructBLIP、MiniGPT-4，以及大多数出于 token 预算原因的视频-语言模型。
- Perceiver resampler：Flamingo 的变体（12.04 课）；Idefics 家族、Eagle、OmniMAE。
- MLP projector：LLaVA、LLaVA-NeXT、LLaVA-OneVision、Cambrian-1。
- Attention pool：VILA、PaliGemma。

四种都有效。决定性问题是你的约束是 token 预算还是质量-per-token。

## 使用方法

`code/main.py` 构建一个标准库 Q-Former 风格的交叉注意力：

1. 模拟 256 个图像 patch token（dim 128）。
2. 实例化 32 个可学习 queries（dim 128）。
3. 运行缩放点积交叉注意力（Q 来自 queries，K/V 来自 patches）。
4. 通过线性层投影到 LLM-dim（512）。
5. 输出 32 个 LLM 就绪的视觉 token。

全部数学在纯 Python 中（嵌套向量循环）。是玩具但形状正确。打印注意力权重矩阵，这样你可以看到每个 query 从哪些 patch 拉取。

## 交付物

本课产出 `outputs/skill-modality-bridge-picker.md`。给定目标 VLM 配置（视觉编码器 token 数量、LLM context 预算、部署约束、质量目标），它推荐 Q-Former vs MLP vs Perceiver resampler，并附简短理由和每个桥的参数量估算。

## 练习

1. 在 PyTorch 中实现交叉注意力块。验证用 32 个 queries 和 256 个 keys/values 时，注意力权重矩阵是 32 × 256，且每行在 softmax 后和为 1。

2. 在 BLIP-2 阶段 1，Q-Former 同时跑三个损失：ITC、ITM、ITG。写出每个的前向签名伪代码。哪个需要 text encoder 路径是激活的？

3. 比较参数量：Q-Former（12 层，768 hidden）vs 2 层 MLP projector（1408 → 4096，两层）。在什么 LLM 规模下 1.88 亿 Q-Former 的成本在训练效率上值得？

4. 阅读 BLIP-2 论文（arXiv:2301.12597）第 3.2 节关于 Q-Former 如何初始化。解释为什么从 BERT-base 初始化（而非随机）加速收敛。

5. 对于一段 10 分钟视频，以 1 FPS 采样到 60 帧，计算每帧 token 成本在（Q-Former → 32 tokens/帧）vs（MLP projector → 576 tokens/帧）。哪个能放进 128k-token LLM context 窗口？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| Q-Former | "Querying transformer" | 带 32 个可学习 query 向量的小型 transformer，通过交叉注意力关注冻结 ViT 特征 |
| Learnable queries | "视觉软 prompt" | 一组固定参数，作为交叉注意力的 query 侧；每个模型学自己的，对所有输入共享 |
| Cross-attention | "Q 来自这里，K/V 来自那里" | query、key、value 来自不同来源的注意力；queries 如何从 ViT patch 拉取 |
| ITC | "图文对比" | 将 Q-Former 池化 queries 与文本 CLS 应用 CLIP 式损失 |
| ITM | "图文匹配" | 在硬负样本挖掘对上做二元分类器；迫使 queries 区分细粒度不匹配 |
| ITG | "图生文本生成" | 以 queries 为条件的因果 LM 损失，生成文本；迫使 queries 编码可文本解码的内容 |
| Two-stage pretraining | "先表示后生成" | 阶段 1 只训练 Q-Former（ITC/ITM/ITG）；阶段 2 接上冻结 LLM，只训练投影 + Q-Former |
| Frozen backbone | "不要微调" | 视觉编码器和 LLM 权重固定；只有桥训练 |
| Projection head | "线性到 LLM dim" | 将 Q-Former 输出映射到 LLM embedding 维度的最终线性层 |
| Perceiver resampler | "Flamingo 的版本" | 类似的可学习 query 交叉注意力，在 Flamingo 中用于每一层而非单一桥 |

## 延伸阅读

- [Li 等 — BLIP-2（arXiv:2301.12597）](https://arxiv.org/abs/2301.12597) — 核心论文。
- [Li 等 — BLIP（arXiv:2201.12086）](https://arxiv.org/abs/2201.12086) — 前身，带 ITC/ITM/ITG 三件套。
- [Li 等 — ALBEF（arXiv:2107.07651）](https://arxiv.org/abs/2107.07651) — "先对齐再融合"——阶段 1 训练的概念祖先。
- [Dai 等 — InstructBLIP（arXiv:2305.06500）](https://arxiv.org/abs/2305.06500) — 指令感知的 Q-Former。
- [Zhu 等 — MiniGPT-4（arXiv:2304.10592）](https://arxiv.org/abs/2304.10592) — 纯 projector 方法。
- [Jaegle 等 — Perceiver IO（arXiv:2107.14795）](https://arxiv.org/abs/2107.14795) — 可学习 query 交叉注意力的一般架构。