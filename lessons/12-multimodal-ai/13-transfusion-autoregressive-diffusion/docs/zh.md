# Transfusion：自回归文本与扩散图像的单一Transformer融合

> Chameleon 和 Emu3 押注于离散 Token。它们有效，但量化瓶颈肉眼可见——图像质量在连续空间扩散模型面前遇到了天花板。Transfusion（Meta，Zhou 等，2024年8月）则反向押注：保持图像连续，完全去掉 VQ-VAE，用两个损失函数训练一个 transformer。文本 Token 用下一个 Token 预测。图像块用流匹配 / 扩散损失。两个目标函数优化同一套权重。Stable Diffusion 3（MMDiT）的底层架构是一个近亲。本文将解读 Transfusion 论文，构建一个双损失训练器的 MNIST 规模示例，并追踪让一个 transformer 同时完成两项工作的注意力掩码。

**类型：** 构建型
**语言：** Python（标准库，双损失训练器 MNIST 规模示例）
**前置条件：** 阶段 12 · 11（Chameleon）、阶段 8（生成式 AI）
**时间：** 约 180 分钟

## 学习目标

- 构建一个同时运行两种损失（文本 Token 的 NTP、图像块的扩散 MSE）的 transformer，共享同一套主干网络。
- 解释为什么图像块的双向注意力 + 文本的因果注意力是正确的掩码选择。
- 从计算量、质量和代码复杂度三个维度对比 Transfusion 风格（连续图像、扩散损失）与 Chameleon 风格（离散图像、NTP）。
- 说出 MMDiT 的贡献：每个 block 有模态专属权重，在残差流做联合注意力。

## 问题

离散与连续的图像 Token 之争早于 LLM 时代就已存在。连续表示（原始像素、VAE 潜变量）保留细节。离散 Token（VQ 索引）适配 transformer 的原生词表，但在量化步骤丢失细节。

Chameleon / Emu3 走了离散路线：一个损失、一个架构，但图像保真度受限于分词器质量。

扩散模型走了连续路线：卓越的图像质量，但与 LLM 是独立模型，噪声调度工程复杂，与文本生成没有干净的集成。

Transfusion 问：能否两者兼得？保持图像连续，仍训练一个模型，把两个损失缝合到一次梯度更新中。

## 概念

### 双损失架构

单一解码器-only transformer 处理一个包含以下内容的序列：

- 文本 Token（离散的，来自 BPE 词表）。
- 图像块（连续的，16×16 像素块通过线性嵌入投影到隐藏维度——与 ViT 编码器的输入相同）。
- `<image>` 和 `</image>` 标签，标记连续块的位置。

前向传播只跑一次。损失函数根据每个 Token 类型选择两个头之一：

- 对于文本 Token：标准交叉熵，在词表-logits 头上计算。
- 对于图像块：在连续块上计算扩散损失——预测每个块添加的噪声。

梯度通过共享的 transformer 主干流动。两个损失同时优化共享权重。

### 注意力掩码：因果文本 + 双向图像

文本 Token 必须是因果的——不能让文本 Token attend 到未来的文本，否则 teacher forcing 就失效了。然而图像块代表一个快照；它们应该在同一图像块内双向 attend 彼此。

掩码规则：

```
M[i, j] = 1 条件：
  (i 是文本且 j 是文本且 j <= i)   # 文本因果
  OR (i 是图像且 j 是图像且 same_image_block(i, j))   # 图像块内双向
  OR (i 是文本且 j 是图像且 j < i_image_end)   # 文本 attend 到前面的图像
  OR (i 是图像且 j 是文本且 j < i_image_start)   # 图像 attend 到前面的文本
```

在训练和推理时实现为块三角掩码。

### Transformer 内部的扩散损失

扩散损失是标准做法：向图像块添加噪声，要求模型预测噪声（或等效地预测干净块）。Transfusion 版本使用流匹配——从噪声预测到干净数据的 velocity 场。

训练时：
1. 对于每个图像块 x0，采样一个随机时间步 t。
2. 采样噪声 ε，计算 xt = (1-t) * x0 + t * ε（流匹配的线性插值）。
3. Transformer 预测 v_theta(xt, t)；损失 = MSE(v_theta(xt, t), ε - x0)。
4. 与同一序列的文本 NTP 损失一起反向传播。

推理时生成：
- 文本 Token：标准自回归采样。
- 图像块：在先验文本 Token 条件下，运行扩散采样循环（典型 10-30 步）。

### MMDiT：Stable Diffusion 3 的变体

Stable Diffusion 3（Esser 等，2024年3月）在 Transfusion 同期推出了 MMDiT（多模态扩散 Transformer）。两种架构是兄弟关系。

MMDiT 的关键差异：

- 每个 block 有模态专属权重。每个 transformer block 对文本 Token 和图像块有独立的 Q、K、V 和 MLP 权重。注意力是联合的（跨模态）；其他都是模态专属的。
- 矫正流训练。一种特定的流匹配变体，采样公式已知，比 DDPM 的数学更简单。
- 规模。MMDiT 是 SD3（2B 和 8B 参数变体）的主干。Transfusion 论文扩展到 7B。

两者收敛到同一个核心思想：一个 transformer 在文本上运行 NTP，在连续图像表示上运行扩散。

### 为什么这比 Chameleon 风格更好

连续扩散与离散 NTP 在图像生成上的质量差距是可测量的。Transfusion 论文报告：

- 在 7B 参数下，在 FID 上比同规模的 Chameleon 风格模型高 3-5 分。
- 不需要训练分词器——图像编码器更简单（线性投影到隐藏维度，与 ViT 的输入层相同）。
- 推理时可以并行化图像块去噪，不像自回归图像 Token 那样顺序执行。

缺点：Transfusion 是双损失模型，训练动态更复杂。损失权重需要调优。NTP 和扩散之间的调度不匹配可能导致其中一个头主导训练。

### 下游应用

Janus-Pro（第 12.15 课）通过解耦视觉编码器来理解生成——一个用 SigLIP，另一个用 VQ——同时共享 transformer 主干，改进了 Transfusion 的思想。Show-o（第 12.14 课）将扩散替换为离散扩散（掩码预测）。统一生成家族在 Transfusion 之后迅速分支。

2026 年生产级发出图像的 VLM——Gemini 3 Pro、GPT-5、Claude Opus 4.7 的图像生成路径——几乎肯定使用这个家族的某个后代。细节是专有的。

## 使用它

`code/main.py` 构建了一个小型 MNIST 类问题的玩具 Transfusion：

- 文本描述是描述数字（0-9）的短整数序列。
- 图像是 4×4 的字节网格。
- 一对共享权重的线性投影作为 transformer 的替代；文本上 NTP 损失，有噪声块上 MSE 损失。
- 训练循环交替使用两种损失，注意力掩码是显式构造的。
- 生成在一个前向传播中产生文本描述和 4×4 图像。

这个 transformer 是玩具级的。真正的产物是双损失的管道连接、注意力掩码构造和推理循环。

## 交付它

本课产出 `outputs/skill-two-loss-trainer-designer.md`。给定一个新的多模态训练任务（文本+图像、文本+音频、文本+视频），它设计双损失调度（损失权重、掩码形状、共享 vs 模态专属 block）并标记实现风险。

## 练习

1. 一个 Transfusion 风格模型训练 70% 文本 Token 和 30% 图像块。图像扩散损失的量级大约是文本 NTP 损失的 10 倍。需要什么损失权重来平衡它们？

2. 实现序列 `[T, T, <image>, P, P, P, P, </image>, T]` 的块三角掩码。将每个条目标记为 0 或 1。

3. MMDiT 有模态专属的 QKV 权重。与 Transfusion 的全共享 transformer 相比，这增加了多少参数开销？在 7B 参数下，值得吗？

4. 生成：给定文本提示符，模型运行 50 个 Token 的 NTP，然后遇到 `<image>`，然后在 256 个块上运行 20 步去噪。总共需要多少次前向传播？

5. 阅读 SD3 论文第 3 节。描述矫正流及其为什么比 DDPM 用更少的推理步数收敛。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 双损失训练 | "NTP + 扩散" | 单一 transformer 在同一次梯度更新中优化文本 Token 的交叉熵和连续图像块的 MSE |
| 流匹配 | "矫正流" | 扩散变体，预测从噪声到数据的 velocity 场；比 DDPM 数学更简单 |
| MMDiT | "多模态 DiT" | Stable Diffusion 3 的架构：联合注意力，模态专属的 MLP 和归一化层 |
| 块三角掩码 | "因果文本 + 双向图像" | 对文本因果但在图像区域内双向的注意力掩码 |
| 连续图像表示 | "无 VQ" | 图像块作为实值向量，而非整数码本索引 |
| 速度预测 | "v 参数化" | 网络输出是噪声和数据之间的 velocity 场，而非噪声本身 |

## 延伸阅读

- [Zhou 等 — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser 等 — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao 等 — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie 等 — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)