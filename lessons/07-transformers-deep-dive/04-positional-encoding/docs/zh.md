# 位置编码 — Sinusoidal、RoPE 与 ALiBi

> 注意力机制对置换是不变的。"The cat sat on the mat" 和 "mat the on sat cat the" 在没有位置信号的情况下会产生完全相同的输出。三种算法解决了这个问题 —— 每一种都对"位置"的含义做出了不同的押注。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 7 · 02（自注意力）、阶段 7 · 03（多头注意力）
**时间：** 约 45 分钟

## 问题

缩放点积注意力是"顺序盲"的。注意力矩阵 `softmax(Q K^T / √d) V` 是由成对相似度计算得出的。将 `X` 的行打乱，输出的行也会以同样的方式被打乱。注意力机制内部没有任何东西关心位置。

在词袋模型里这不是 bug。但对于语言、代码、音频、视频 —— 任何顺序承载意义的场景 —— 这是致命的。

解决办法是将位置以某种方式注入到 embedding 中。三代解决方案：

1. **绝对位置正弦编码**（Vaswani 2017）。将位置的 `sin/cos` 加到 embedding 上。简单、无需学习，但在训练长度之外的外推能力很差。
2. **RoPE — 旋转位置编码**（Su 2021）。将 Q 和 K 向量按与位置成比例的角度旋转。在点积中直接编码*相对*位置。2026 年的主流方案。
3. **ALiBi — 带线性偏置的注意力**（Press 2022）。完全跳过 embedding；根据距离在注意力分数上为每个头添加线性惩罚。外推能力极强。

截至 2026 年，本质上每一个前沿开源模型都在用 RoPE：Llama 2/3/4、Qwen 2/3、Mistral、Mixtral、DeepSeek-V3、Kimi。少量长上下文模型使用 ALiBi 或其现代变体。绝对正弦编码已成为历史。

## 概念

![正弦绝对编码 vs RoPE 旋转 vs ALiBi 距离偏置](../assets/positional-encoding.svg)

### 绝对位置正弦编码

预计算一个形状为 `(max_len, d_model)` 的固定矩阵 `PE`：

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

然后在注意力计算前将 `X' = X + PE[:N]` 加到 embedding 上。每一个维度都是不同频率的正弦波。模型从相位模式中学习读取位置。在 `max_len` 之外就会失效：当模型只见过位置 0–2047，没有任何东西告诉它位置 2048 会发生什么。

### RoPE

对 Q 和 K 向量（而非 embedding）进行旋转。对于一对维度 `(2i, 2i+1)`：

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

对位置为 `pos_k` 的 keys 施加相同的旋转。点积 `q'_m · k'_n` 变成了关于 `(m - n)` 的函数。这就是：**注意力分数只取决于相对距离**，尽管旋转依据的是绝对位置。很妙的技巧。

RoPE 的扩展：`base` 可以缩放（NTK-aware、YaRN、LongRoPE），从而在不重新训练的情况下外推到更长的上下文。Llama 3 就是通过这种方式将上下文从 8K 扩展到 128K 的。

### ALiBi

跳过 embedding 这个把戏。直接偏置注意力分数：

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

其中 `m_h` 是每个头特有的斜率（例如 `1 / 2^(8·h/H)`）。近的 token 得到增强；远的 token 受到惩罚。没有训练时间成本。论文表明，外推能力超越正弦编码，且在其原始训练长度上与 RoPE 持平。

### 2026 年该选哪个

| 变体 | 外推能力 | 训练成本 | 使用者 |
|---------|---------------|---------------|---------|
| 绝对正弦编码 | 差 | 免费 | 原始 transformer、早期 BERT |
| 学习式绝对编码 | 无 | 很小 | GPT-2、GPT-3 |
| RoPE | 配合缩放效果好 | 免费 | Llama 2/3/4、Qwen 2/3、Mistral、DeepSeek-V3、Kimi |
| RoPE + YaRN | 优秀 | 微调阶段 | Qwen2-1M、Llama 3.1 128K |
| ALiBi | 优秀 | 免费 | BLOOM、MPT、Baichuan |

RoPE 胜出是因为它插入注意力机制而不改变架构编码相对位置，且其 `base` 超参数为长上下文微调提供了一个干净的调节旋钮。

## 动手实现

### 第 1 步：正弦编码

参见 `code/main.py`。4 行代码：

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

在第一个注意力层之前将它加到 embedding 矩阵上。

### 第 2 步：RoPE 应用于 Q 和 K

RoPE 就地作用于 Q 和 K。对于每对维度：

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

关键：对位置 `m` 的 Q 和位置 `n` 的 K 施加相同的函数。它们的点积在每一对坐标上都会带上一个 `cos((m-n)·θ_i)` 因子。注意力免费学会了相对位置。

### 第 3 步：ALiBi 斜率与偏置

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

将 `bias[h]` 加到头 `h` 的 `(seq_len, seq_len)` 注意力分数矩阵上，然后 softmax。

### 第 4 步：验证 RoPE 的相对距离性质

取两个随机向量 `a, b`。先用 `(pos_a, pos_b)` 旋转，再用 `(pos_a + k, pos_b + k)` 旋转。两个点积必须在浮点误差范围内相等。这个性质是 RoPE 的全部意义 —— 它对绝对偏移不变，只关心相对差距。

## 实际使用

PyTorch 2.5+ 在 `torch.nn.functional` 中提供了 RoPE 工具。大多数生产代码使用 `flash_attn` 或 `xformers`，其中 RoPE 在注意力内核内部应用。

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026 年的长上下文技巧：**

- **NTK-aware 插值。** 从 4K 扩展到 16K+ 时，将 `base` 重新缩放为 `base * (scale_factor)^(d/(d-2))`。
- **YaRN。** 更智能的插值，在长上下文上保持注意力熵。Llama 3.1 128K 使用了它。
- **LongRoPE。** 微软 2024 年的方法，使用进化搜索来选取每维缩放因子。Phi-3-Long 使用了它。
- **位置插值 + 微调。** 只需将位置按扩展因子缩小，然后对 1–5B 个 token 进行微调。出人意料地有效。

## 交付物

参见 `outputs/skill-positional-encoding-picker.md`。该技能根据目标上下文长度、外推需求和训练预算为一个新模型选择编码策略。

## 练习

1. **简单。** 将正弦 `PE` 矩阵绘制为 `max_len=512, d=128` 的热力图。确认"条纹随维度索引增长而变宽"的模式。
2. **中等。** 实现 NTK-aware RoPE 缩放。在长度为 256 的序列上训练一个小 LM，然后在长度为 1024 下有/无缩放两种情况测试。测量困惑度。
3. **困难。** 在同一个注意力模块中实现 ALiBi 和 RoPE。在长度为 512 的复制任务上训练一个 4 层 transformer。在测试时外推到 2048。比较性能退化情况。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 位置编码 (Positional encoding) | "告诉注意力顺序" | 任何添加到 embedding 或注意力中的、用于编码位置的信号。 |
| 正弦编码 (Sinusoidal) | "最初的那个" | 以几何级数排列的 `sin/cos` 加到 embedding 上；不能外推。 |
| RoPE | "旋转编码" | 按位置相关角度旋转 Q、K；点积编码相对距离。 |
| ALiBi | "线性偏置技巧" | 在注意力分数上添加 `-m·|i-j|`；不需要 embedding，外推能力极强。 |
| base | "RoPE 的旋钮" | RoPE 中的频率缩放器；增加它可以延长推理时的上下文。 |
| NTK-aware | "一种 RoPE 缩放技巧" | 重新缩放 `base`，使高频维度在上下文扩展时不被挤压。 |
| YaRN | "更花哨的那个" | 保持注意力熵的逐维插值+外推。 |
| 外推 (Extrapolation) | "在训练长度之外也能工作" | 该位置编码方案能否在训练时见过的 `max_len` 之后仍给出正确输出？ |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 原始正弦编码。
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 论文。
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi。
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — RoPE 缩放的最先进方案。
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Meta 的 Llama 2 长上下文论文。
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — 微软方法，被 Phi-3-Long 使用，也在"实际使用"一节中引用。
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — 每一种 RoPE 缩放方案的生产级实现（默认、线性、动态、YaRN、LongRoPE、Llama-3）。