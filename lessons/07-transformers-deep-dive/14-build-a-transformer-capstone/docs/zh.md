# 从零构建 Transformer — 顶点设计

> 十三节课。一个模型。没有捷径。

**类型：** 构建型
**语言：** Python
**前置条件：** 第 7 阶段 · 01 至 13。不要跳过。
**时间：** 约 120 分钟

## 问题

你已经读完了所有论文。你已经实现了注意力机制、多头分割、位置编码、编码器和解码器块、BERT 和 GPT 损失、MoE、KV 缓存。现在把它们放在一起，让它们在一个真实任务上协同工作。

顶点设计：在一个字符级语言建模任务上从头训练一个小型纯解码器 transformer。它读莎士比亚。它生成新的莎士比亚作品。它小到可以在笔记本上在 10 分钟内训练完成。它正确到可以换用更大的数据集和更长的训练时间来获得一个真正的语言模型。

这是课程的"nanoGPT"。它不是原创的——Karpathy 的 2023 nanoGPT 教程是每个学生至少写一次的参考实现。我们提取了它的结构，并围绕我们已经讲过的内容重新调整它。

## 概念

![从零构建 Transformer 模块图](../assets/capstone.svg)

带注释的架构：

```
输入 token (B, N)
   │
   ▼
token embedding + 位置 embedding  ◀── 第 04 节（RoPE 选项）
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── 第 05 节
│  MultiHeadAttention (因果)         │  ◀── 第 03 + 07 节（因果 mask）
│  残差                              │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── 第 05 节
│  残差                              │
└────────────────────────────────── ┘
   │
   ▼
最终 RMSNorm
   │
   ▼
lm_head（绑定到 token embedding）
   │
   ▼
logits (B, N, V)
   │
   ▼
移位一位的交叉熵                     ◀── 第 07 节
```

### 我们交付的内容

- `GPTConfig` — 一个地方配置所有超参数。
- `MultiHeadAttention` — 因果的、批处理的，带可选的 Flash 式通路（PyTorch 的 `scaled_dot_product_attention`）。
- `SwiGLUFFN` — 现代 FFN。
- `Block` — pre-norm、残差包装的注意力 + FFN。
- `GPT` — embedding、堆叠块、LM head、generate()。
- 带 AdamW、余弦 LR、梯度裁剪的训练循环。
- 莎士比亚文本上的字符级分词器。

### 我们没有交付的内容

- RoPE — 在第 04 节中从概念上实现过。这里我们使用学习到的位置 embedding 以保持简单。练习要求你换入 RoPE。
- 生成时的 KV 缓存 — 每个生成步骤都重新计算对完整前缀的注意力。更慢但更简单。练习要求你添加 KV 缓存。
- Flash Attention — PyTorch 2.0+ 在输入匹配时自动分派；我们使用 `F.scaled_dot_product_attention`。
- MoE — 每个 block 一个 FFN。你在第 11 节见过 MoE。

### 目标指标

在 Mac M2 笔记本上，一个 4 层、4 头、d_model=128 的 GPT 在 `tinyshakespeare.txt` 上训练 2,000 步：

- 训练损失从约 4.2（随机）收敛到约 1.5，大约需要 6 分钟。
- 采样输出看起来像莎士比亚的风格：古英语词汇、换行符、"ROMEO:" 这样的角色名出现了。
- 验证损失（保留文本最后 10% 作为验证）在训练损失附近紧密跟踪；在当前规模/预算下没有过拟合。

## 从零实现

这节课使用 PyTorch。安装 `torch`（CPU 版本即可）。见 `code/main.py`。脚本处理：

- 如果缺失则下载 `tinyshakespeare.txt`（或读取本地副本）。
- 字节级字符分词器。
- 90/10 划分训练/验证集。
- 在支持硬件上使用 bf16 autocast 的训练循环。
- 训练完成后采样。

### 第 1 步：数据

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 个唯一字符。极小词汇表。适合 4 字节的 vocab_size。没有 BPE，没有分词器麻烦。

### 第 2 步：模型

见 `code/main.py`。Block 是第 05 节教科书式的——pre-norm、RMSNorm、SwiGLU、因果 MHA。4/4/128 的参数量：约 800K。

### 第 3 步：训练循环

获取长度为 256 的随机 token 窗口批次。前向传播。移位一位交叉熵。反向传播。AdamW 步。日志。重复。

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 第 4 步：采样

给定一个提示，反复前向传播，从 top-p logits 中采样，附加，继续。500 token 后停止。

### 第 5 步：读输出

2,000 步后：

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

不是莎士比亚。但有莎士比亚的形状。对于约 800K 参数和笔记本上的 6 分钟来说，是一个明显的胜利。

## 实际使用

这个顶点设计是一个参考架构。三个扩展可以把它交付成真实的东西：

1. **换分词器。** 使用 BPE（例如 `tiktoken.get_encoding("cl100k_base")`）。词汇表大小从 65 跃升至约 50,000。模型容量需要相应扩大来补偿。
2. **在更大的语料上训练。** 使用 `OpenWebText` 或 `fineweb-edu`（HuggingFace）。125M 参数的 GPT 在单个 A100 上训练 10B token 约需 24 小时。
3. **添加 RoPE + KV 缓存 + Flash Attention。** 下面的练习会带你完成每一个。

最终你会得到一个 125M 参数的 GPT，能生成流畅的英语。不是前沿模型。但同样的代码路径——只是更大——是 Karpathy、EleutherAI 和 Allen Institute 在 2026 年用来训练研究检查点的工具。

## 交付物

见 `outputs/skill-transformer-review.md`。该技能审查一个从零实现 transformer 的正确性，涵盖所有 13 节先前课程。

## 练习

1. **简单。** 运行 `code/main.py`。验证你训练模型的最终步验证损失低于 2.0。将 `max_steps` 从 2,000 改为 5,000——验证损失还在继续改善吗？
2. **中等。** 将学习到的位置 embedding 替换为 RoPE。在 `MultiHeadAttention` 内部对 Q 和 K 应用旋转。训练并验证验证损失至少一样低。
3. **中等。** 在采样循环中实现 KV 缓存。用和不用缓存分别生成 500 个 token。笔记本上的墙上时钟时间应该改善 5–20×。
4. **困难。** 给模型添加第二个预测下一个加一个 token 的头（MTP——来自 DeepSeek-V3 的多 token 预测）。联合训练。有帮助吗？
5. **困难。** 将每个 block 的单一 FFN 替换为 4 专家 MoE。路由器 + top-2 路由。看在匹配活跃参数下验证损失如何变化。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| nanoGPT | "Karpathy 的教程仓库" | 最小化解码器 only transformer 训练代码，约 300 行；经典参考。 |
| tinyshakespeare | "标准的玩具语料" | 约 1.1 MB 文本；自 2015 年以来每个字符级 LM 教程都用它。 |
| 绑定 embedding | "共享输入/输出矩阵" | LM head 权重 = token embedding 矩阵的转置；节省参数，改善质量。 |
| bf16 autocast | "训练精度技巧" | 前向/反向用 bf16运行，优化器状态保持 fp32；自 2021 年以来的标准。 |
| 梯度裁剪 | "阻止尖峰" | 将全局梯度范数上限设为 1.0；防止训练崩溃。 |
| 余弦 LR 调度 | "2020 年后的默认" | LR 先线性上升（warmup）然后余弦衰减到峰值的 10%。 |
| MFU | "模型 FLOP 利用率" | 实际 FLOPs / 理论峰值；2026 年 40% 密集、30% MoE 是强项。 |
| 验证损失 | "保留损失" | 模型从未见过的数据的交叉熵；过拟合检测器。 |

## 延伸阅读

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — 经典带注释实现。