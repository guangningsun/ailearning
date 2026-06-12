# 完整 Transformer — 编码器与解码器

> 注意力是明星。其他一切 —— 残差、归一化、前馈网络、交叉注意力 —— 都是让你能堆叠多层的脚手架。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 7 · 02（自注意力）、阶段 7 · 03（多头注意力）、阶段 7 · 04（位置编码）
**时间：** 约 75 分钟

## 问题

单层注意力是一个特征提取器，而不是一个模型。每层一次矩阵乘法对语言来说容量不够。你需要深度 —— 而深度没有正确的管道工程是会断裂的。

2017 年的 Vaswani 论文打包了六个设计决策，将一层注意力变成一个可堆叠的块。此后的每一个 transformer —— 纯编码器（BERT）、纯解码器（GPT）、编码器-解码器（T5） —— 都继承了相同的骨架。2026 年这些块已经被优化（RMSNorm、SwiGLU、pre-norm、RoPE），但骨架是完全相同的。

本课讲的是骨架。后续课程将它专门化 —— 06 讲编码器，07 讲解码器，08 讲编码器-解码器。

## 概念

![编码器和解码器块内部结构，连线图](../assets/full-transformer.svg)

### 六个组成部分

1. **Embedding + 位置信号。** Token → 向量。位置通过 RoPE（现代）或正弦编码（经典）注入。
2. **自注意力。** 每个位置attend到所有其他位置。在解码器中是带掩码的。
3. **前馈网络（FFN）。** 逐位置的两层 MLP：`W_2 · activation(W_1 · x)`。扩展比率默认 4×。
4. **残差连接。** `x + sublayer(x)`。没有它，梯度在约 6 层之后就会消失。
5. **层归一化。** `LayerNorm` 或 `RMSNorm`（现代）。稳定残差流。
6. **交叉注意力（仅解码器）。** Query 来自解码器，Key 和 Value 来自编码器输出。

观察一个向量流经一个块的过程：注意力混合各位置，残差将它向前传递，FFN 变换它，Norm 保持流稳定。

```figure
transformer-block
```

### 编码器块（BERT、T5 编码器使用）

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

编码器是双向的。无掩码。所有位置都能看到所有位置。

### 解码器块（GPT、T5 解码器使用）

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

解码器每个块有三个子层。中间那个 —— 交叉注意力 —— 是编码器信息流向解码器的唯一通道。在纯解码器架构（GPT）中，交叉注意力被省略，只剩下带掩码的自注意力 + FFN。

### Pre-norm vs Post-norm

原始论文：`x + sublayer(LN(x))` vs `LN(x + sublayer(x))`。Post-norm 在 2019 年左右失宠 —— 没有仔细的预热就很难训练深度。Pre-norm（`LN` *在*子层*之前*）是 2026 年的默认：Llama、Qwen、GPT-3+、Mistral 都使用它。

### 2026 年现代化后的块

Vaswani 2017 年用的是 LayerNorm + ReLU。现代堆栈两者都替换了。生产环境中的块实际长这样：

| 组件 | 2017 年 | 2026 年 |
|-----------|------|------|
| 归一化 | LayerNorm | RMSNorm |
| FFN 激活函数 | ReLU | SwiGLU |
| FFN 扩展 | 4× | 2.6×（SwiGLU 使用三个矩阵，总参数量匹配） |
| 位置编码 | 绝对正弦编码 | RoPE |
| 注意力 | 完整 MHA | GQA（或 MLA） |
| 偏置项 | 有 | 无 |

RMSNorm 去掉了 LayerNorm 的均值中心化（少做一次减法），节省了计算量，且经验上稳定性相当。SwiGLU（`Swish(W1 x) ⊙ W3 x`）在 Llama、PaLM 和 Qwen 论文中持续以约 0.5 个 ppl 的优势超越 ReLU/GELU FFN。

### 参数量

对于一个 `d_model = d`、FFN 扩展 `r` 的块：

- MHA：`4 · d²`（Q、K、V、O 投影）
- FFN（SwiGLU）：`3 · d · (r · d)` ≈ `3rd²`
- 归一化：可忽略

当 `d = 4096, r = 2.6, layers = 32`（约等于 Llama 3 8B），总计：`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B 每层参数 × 32 ≈ 7B`（加上 embedding 和 head）。与已发布的数据吻合。

## 动手实现

### 第 1 步：构建块

使用第 03 课中的 tiny `Matrix` 类（复制到本文件以保证独立性）：

- `layer_norm(x, eps=1e-5)` — 减去均值，除以标准差。
- `rms_norm(x, eps=1e-6)` — 除以 RMS。不减去均值。
- `gelu(x)` 和 `silu(x) * W3 x`（SwiGLU）。
- `ffn_swiglu(x, W1, W2, W3)`。
- `encoder_block(x, params)` 和 `decoder_block(x, enc_out, params)`。

参见 `code/main.py` 的完整连线。

### 第 2 步：连接 2 层编码器和 2 层解码器

堆叠它们。将编码器输出传入每个解码器的交叉注意力。在输出投影前加一个最终的 LN。

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### 第 3 步：在玩具示例上运行前向传播

输入 6 个 token 的源序列和 5 个 token 的目标序列。验证输出形状是 `(5, vocab)`。不涉及训练 —— 本课讲的是架构，不是损失函数。

### 第 4 步：换入 RMSNorm + SwiGLU

用 RMSNorm 和 SwiGLU 替换 LayerNorm 和 ReLU-FFN。确认形状仍然匹配。这是一次函数替换就完成的 2026 年现代化改造。

## 实际使用

PyTorch/TF 参考实现：`nn.TransformerEncoderLayer`、`nn.TransformerDecoderLayer`。但大多数 2026 年的生产代码都自己写块，因为：

- Flash Attention 在注意力内部调用，不通过 `nn.MultiheadAttention`。
- GQA / MLA 不在标准库参考实现中。
- RoPE、RMSNorm、SwiGLU 不是 PyTorch 的默认值。

HF `transformers` 有干净可读的参考块：`modeling_llama.py` 是 2026 年标准的纯解码器块。大约 500 行，值得通读一遍。

**编码器 vs 解码器 vs 编码器-解码器 —— 何时选哪个：**

| 需求 | 选择 | 示例 |
|------|------|---------|
| 文本分类、embedding、QA | 纯编码器 | BERT、DeBERTa、ModernBERT |
| 文本生成、聊天、代码、推理 | 纯解码器 | GPT、Llama、Claude、Qwen |
| 结构化输入 → 结构化输出（翻译、摘要） | 编码器-解码器 | T5、BART、Whisper |

纯解码器在语言任务上胜出，因为它扩展最干净，同时处理理解和生成。编码器-解码器在输入有明确的"源序列"身份时仍然是最佳选择（翻译、语音识别、结构化任务）。

## 交付物

参见 `outputs/skill-transformer-block-reviewer.md`。该技能对照 2026 年的默认设置审查新的 transformer 块实现，并标记缺失的部分（pre-norm、RoPE、RMSNorm、GQA、FFN 扩展比率）。

## 练习

1. **简单。** 计算你的 `encoder_block` 在 `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True` 时的参数量。通过实现该块并使用 `sum(p.numel() for p in block.parameters())` 来验证。
2. **中等。** 从 post-norm 切换到 pre-norm。初始化两者，在随机输入上测量 12 层堆叠后的激活范数。Post-norm 的激活应该会爆炸；pre-norm 的应该保持有界。
3. **困难。** 在一个玩具复制任务（复制并反转 `x`）上实现 4 层编码器-解码器。训练 100 步。报告损失。换入 RMSNorm + SwiGLU + RoPE —— 损失有下降吗？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 块 (Block) | "一个 transformer 层" | Norm + 注意力 + Norm + FFN 的堆叠，外裹残差连接。 |
| 残差 (Residual) | "跳跃连接" | `x + f(x)` 输出；使梯度能穿过深层堆叠流动。 |
| Pre-norm | "先归一化，而不是后归一化" | 现代做法：`x + sublayer(LN(x))`。无需预热技巧就能训练更深的层。 |
| RMSNorm | "LayerNorm 去掉均值" | 除以 RMS；少一个操作，经验稳定性相同。 |
| SwiGLU | "大家都换成的 FFN" | `Swish(W1 x) ⊙ W3 x → W2`。在 LM ppl 上超越 ReLU/GELU。 |
| 交叉注意力 (Cross-attention) | "解码器如何看到编码器" | MHA，其中 Q 来自解码器，K/V 来自编码器输出。 |
| FFN 扩展 (FFN expansion) | "中间 MLP 有多宽" | 隐藏维度与 d_model 的比率，通常是 4（LayerNorm）或 2.6（SwiGLU）。 |
| 无偏置 (Bias-free) | "去掉 +b 项" | 现代堆栈在线性层中省略偏置；略微改善 ppl，模型更小。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始块规范。
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — 为什么 pre-norm 在深层优于 post-norm。
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm。
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU 论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 2026 年标准的纯解码器块。