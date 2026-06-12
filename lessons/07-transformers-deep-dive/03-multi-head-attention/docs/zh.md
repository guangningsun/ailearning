# 多头注意力

> 一个注意力头一次学习一种关系。八个头学习八种。头是免费的。多拿几个。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 7 · 02（自注意力从零实现）
**时间：** 约 75 分钟

## 问题

单个自注意力头计算一个注意力矩阵。该矩阵捕获一种关系——通常是使训练信号上的损失最小化的那种。如果你的数据有主谓一致、共指、长距离话语和句法分块都纠缠在一起，单个头将它们混合成一个 softmax 分布并丢失一半信号。

2017 年 Vaswani 论文的修复方法：并行运行几个注意力函数，每个有自己的 Q、K、V 投影，然后拼接输出。每个头在维度 `d_model / n_heads` 的较小子空间中操作。参数总量保持不变。表达能力提升。

多头注意力是 2026 年每个 Transformer 的默认配置。唯一的争论是*有多少*个头，以及 key 和 value 是否共享投影（分组查询注意力、多查询注意力、多头潜在注意力）。

## 概念

![多头注意力拆分、关注、拼接](../assets/multi-head-attention.svg)

**拆分。** 取 shape 为 `(N, d_model)` 的 `X`。投影到 Q、K、V，每个 shape 为 `(N, d_model)`。重塑为 `(N, n_heads, d_head)`，其中 `d_head = d_model / n_heads`。转置为 `(n_heads, N, d_head)`。

**并行关注。** 在每个头内运行缩放点积注意力。每个头产生 `(N, d_head)`。头在 embedding 的不同子空间中操作，在注意力计算本身期间从不交流。

**拼接和投影。** 将头堆叠回 `(N, d_model)`，然后乘以一个学习到的输出矩阵 `W_o`，shape 为 `(d_model, d_model)`。`W_o` 是头混合的地方。

**为什么它有效。** 每个头可以专门化而不与其他头竞争表示预算。2019–2024 年的探测研究表明了不同的头角色：位置头、关注前一个 token 的头、复制头、命名实体头、归纳头（这是上下文学习的基础）。

**2026 年的变体谱系：**

| 变体 | Q 头 | K/V 头 | 使用者 |
|---------|---------|-----------|---------|
| 多头 (MHA) | N | N | GPT-2, BERT, T5 |
| 多查询 (MQA) | N | 1 | PaLM, Falcon |
| 分组查询 (GQA) | N | G (例如 N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| 多头潜在 (MLA) | N | 压缩到低秩 | DeepSeek-V2, V3 |

GQA 是现代默认配置，因为它将 KV-cache 内存减少 `N/G` 倍，同时保持几乎全部质量。MLA 通过将 K/V 压缩到潜在空间，然后在计算时投影回来——消耗 FLOPs，节省更多内存。

## 从零构建

### 第 1 步：从我们已经有的单头注意力拆分头

取第 02 课中的 `SelfAttention`，用拆分/拼接对包装它。参见 `code/main.py` 的 numpy 实现；逻辑是：

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

一次重塑和一次转置。没有循环。这正是 PyTorch 在 `nn.MultiheadAttention` 下所做的。

### 第 2 步：每头运行缩放点积注意力

每个头获得自己的 Q、K、V 切片。注意力变成批处理矩阵乘法：

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

在真实硬件上 `Qh @ Kh.transpose(...)` 是一次 `bmm`。GPU 看到一个 shape 为 `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)` 的单次批处理矩阵乘法。添加头是免费的。

### 第 3 步：分组查询注意力变体

只有 key 和 value 投影改变。Q 得到 `n_heads` 组；K 和 V 得到 `n_kv_heads < n_heads` 组并重复以匹配：

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

在推理时这节省了内存，因为只有 `n_kv_heads` 个副本存在于 KV cache 中，而不是 `n_heads`。Llama 3 70B 使用 64 个查询头和 8 个 KV 头——8 倍的缓存缩小。

### 第 4 步：探测每个头学到了什么

用 4 个头在短句子上运行 MHA。对于每个头，打印 `(N, N)` 注意力矩阵。你会看到不同的头即使在随机初始化下也能挑出不同的结构——部分是信号，部分是子空间中的旋转对称性。

## 实际使用

在 PyTorch 中，一行版本：

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

截至 PyTorch 2.5+ 的 GQA：

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention 自动在 CUDA 上调度 Flash Attention。
# 对于 GQA，传递 shape 为 (B, n_heads, N, d_head) 的 Q 和 shape 为
# (B, n_kv_heads, N, d_head) 的 K、V。PyTorch 处理重复。
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**有多少个头？** 2026 年生产模型的的经验法则：

| 模型大小 | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| 小 (~125M) | 768 | 12 | 64 |
| 基座 (~350M) | 1024 | 16 | 64 |
| 大 (~1B) | 2048 | 16 | 128 |
| 前沿 (~70B) | 8192 | 64 | 128 |

`d_head` 几乎总是落在 64 或 128。它是一个头能"看到"的单位量。低于 32，头开始与缩放因子 `sqrt(d_head)` 斗争；高于 256，你失去了"许多小专家"的好处。

## 交付物

参见 `outputs/skill-mha-configurator.md`。该技能在给定参数预算、序列长度和部署目标的情况下，为新的 transformer 推荐头数、kv 头数和投影策略。

## 练习

1. **简单。** 取 `code/main.py` 中的 MHA，将 `n_heads` 从 1 改为 16，同时固定 `d_model=64`。在合成复制任务上绘制 tiny 单层模型的损失。更多的头有帮助、 plateau 还是有害？
2. **中等。** 实现 MQA（所有查询头共享一个 KV 头）。测量参数数量比完整 MHA 下降了多少。计算在 N=2048 推理时 KV-cache 大小缩小了多少。
3. **困难。** 实现一个微型版本的多头潜在注意力：将 K,V 压缩到秩-`r` 潜在，将潜在存储在 KV cache 中，在注意力时解压缩。在什么 `r` 下缓存内存跨越低于完整 MHA 的 1/8，同时验证 ppl 质量保持在 1 bit 内？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 头 (Head) | "单个注意力电路" | 一个维度为 `d_head = d_model / n_heads` 的 Q/K/V 投影，有自己的注意力矩阵。 |
| d_head | "头维度" | 每头隐藏宽度；在生产中几乎总是 64 或 128。 |
| 拆分 / 拼接 | "重塑技巧" | `(N, d_model) ↔ (n_heads, N, d_head)` 在注意力周围的重塑+转置。 |
| W_o | "输出投影" | `(d_model, d_model)` 矩阵，在拼接头之后应用；头混合的地方。 |
| MQA | "一个 KV 头" | 多查询注意力：单个共享 K/V 投影。最小的 KV cache，有一些质量损失。 |
| GQA | "自 Llama 2 以来的默认" | 分组查询注意力，`n_kv_heads < n_heads`；重复以匹配 Q。 |
| MLA | "DeepSeek 的技巧" | 多头潜在注意力：K,V 压缩到低秩潜在，在注意力时解压缩。 |
| 归纳头 | "上下文学习背后的电路" | 一对检测先前出现并复制其后发生内容的头。 |

## 延伸阅读

- [Vaswani 等 (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — 原始多头规范。
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — MQA 论文。
- [Ainslie 等 (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — 如何在训练后将 MHA 转换为 GQA。
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA 以及为什么它在缓存内存上胜过 MHA/GQA。
- [Olsson 等 (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — 机械地看头实际上在做什么。