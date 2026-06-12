# 注意力变体 — 滑动窗口、稀疏化、双差分

> 全注意力的代价是一个圆。每个 token 都看到每个 token，内存来付账。四种变体改变了圆的形状，省下一半成本。

**类型：** 动手构建
**语言：** Python
**前置条件：** 阶段 7 · 02（自注意力）、阶段 7 · 03（多头注意力）、阶段 7 · 12（KV Cache / Flash Attention）
**时间：** 约 60 分钟

## 问题

全注意力的内存开销是 `O(N²)`，计算量也是 `O(N²)`（按序列长度计）。对于 128K 上下文的 Llama 3 70B，每层有 160 亿个注意力条目，乘以 80 层。Flash Attention（第十二课）隐藏了 `O(N²)` 的激活内存，但没有改变算术成本——每个 token 仍然要attend到其他所有 token。

有三类变体改变了注意力矩阵本身的拓扑结构：

1. **滑动窗口注意力（SWA）。** 每个 token 只attend到一个固定的邻域窗口，而不是完整前缀。内存和计算降到 `O(N · W)`，其中 `W` 是窗口大小。Gemma 2/3、Mistral 7B 的前几层、Phi-3-Long。
2. **稀疏 / 分块注意力。** 只有选定的配对 `(i, j)` 获得评分；其余被强行置为零权重。Longformer、BigBird、OpenAI 稀疏 transformer。
3. **双差分注意力。** 用两套独立的 Q/K 投影计算两张注意力图，然后相减。消除了"注意力汇"问题——权重不该泄漏到前几个 token 上。微软的 DIFF Transformer（2024）。

这些可以共存。2026 年的前沿模型通常将它们混合使用：大多数层用 SWA-1024，每隔五层有一个全局全注意力，还有几个双差分头来清理检索结果。Gemma 3 的 5:1 SWA-to-global 比例是当前的教科书级默认配置。

## 概念

### 滑动窗口注意力（SWA）

位置 `i` 处的每个 query 只attend到 `[i - W, i]`（因果 SWA）或 `[i - W/2, i + W/2]`（双向）范围内的位置。窗口外的 token 在 score 矩阵中得到 `-inf`。

```
全因果：               滑动窗口（W=4）：
positions 0-7          positions 0-7，W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

对于 `N = 8192` 和 `W = 1024`，score 矩阵在期望意义下有 1024 × 8192 个非零行——减少 8 倍。

**KV cache 随 SWA 缩小。** 每层只需保留 K 和 V 的最后 `W` 个 token。对于类似 Gemma-3 的配置（1024 窗口、128K 上下文），KV cache 减少 128 倍。

**质量代价。** 纯 SWA 的 transformer 在长程检索上表现吃力。解决办法：将 SWA 层与全注意力层交错排列。Gemma 3 使用 5:1 SWA:global。Mistral 7B 使用了一个因果-SWA 堆栈，信息通过重叠窗口"向前流动"——每层将有效感受野扩展 `W`，经过 `L` 层后模型可以attend到 `L × W` 个 token 之前的内容。

### 稀疏 / 分块注意力

预先选择一个 `N × N` 的稀疏模式。三种经典形状：

- **局部 + 步进（OpenAI 稀疏 transformer）。** attend到最后 `W` 个 token，再加上下标为 `stride` 倍数的更早 token。兼顾局部和长程，`O(N · sqrt(N))` 计算量。
- **Longformer / BigBird。** 局部窗口 + 一小部分全局 token（如 `[CLS]`），这些全局 token attend所有人也被所有人attend + 随机稀疏连接。实证达到 2 倍上下文且质量持平。
- **原生稀疏注意力（DeepSeek，2025）。** 学习哪些 `(Q, K)` 分块是重要的；在内核层面跳过零分块。与 FlashAttention 兼容。

稀疏注意力是一个内核工程的故事。数学很简单（mask score 矩阵）；收益来自从不把零条目加载到 SRAM。FlashAttention-3 和 2026 年的 FlexAttention API 使自定义稀疏模式成为 PyTorch 中的一等公民。

### 双差分注意力（DIFF Transformer，2024）

常规注意力存在"注意力汇"问题：softmax 强制每行和为 1，因此没有特别想attend内容的 token 把权重倾倒在第一个 token（或前几个）上。这窃取了本应流向真实内容的容量。

双差分注意力通过计算**两张**注意力图并相减来修复：

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

其中 `λ` 是一个学习到的标量（通常 0.5–0.8）。A1 捕获真实内容权重；A2 捕获汇。相减抵消了汇，将权重重新分配给相关 token。

报告结果（微软 2024）：困惑度降低 5–10%，在相同训练长度下有效上下文延长 1.5–2 倍，大海捞针检索更锐利。

### 变体对比

| 变体 | 计算量 | KV cache | 质量 vs 全注意力 | 生产使用 |
|-------------|---------|----------|-------------------|----------------|
| 全注意力 | O(N²) | 每层 O(N) | 基线 | 每个模型的默认层 |
| SWA（窗口 1024） | O(N·W) | 每层 O(W) | -0.1 ppl，配合全局层使用良好 | Gemma 2/3、Phi-3-Long |
| 局部 + 步进稀疏 | O(N·√N) | 混合 | 与 SWA 类似 | OpenAI 稀疏 transformer、Longformer |
| BigBird（局部 + 全局 + 随机） | O(N) 近似 | 混合 | 2 倍上下文下与全注意力持平 | 早期长上下文 BERT |
| 原生稀疏（DeepSeek-V3.2） | O(N · 激活比例) | O(N) | 困惑度差异在 0.05 以内 | DeepSeek-V3.2，2025 |
| 双差分 | O(2·N²) | O(2N) | ppl 降低 5–10% | DIFF Transformer，2026 年初模型 |

## 动手构建

参见 `code/main.py`。我们实现了一个因果 mask 比较器，在一个小序列上并列展示全注意力、SWA、局部+步进和双差分注意力。

### 第 1 步：全因果 mask（基线）

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

基线来自第七课。下三角；主对角线以上权重为零。

### 第 2 步：滑动窗口因果 mask

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

一个参数——`window`。当 `window >= n` 时，恢复全因果注意力。当 `window = 1` 时，每个 token 只attend自己。

### 第 3 步：局部 + 步进稀疏 mask

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

密集的局部窗口，加上回溯到序列开头的每个 `stride`-倍位置。感受野随额外层数呈对数增长。

### 第 4 步：双差分注意力

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

两次注意力传递，用一个学习到的混合系数相减。在代码中我们比较单注意力和双差分的注意力汇热图，观察汇的消失。

### 第 5 步：KV cache 大小

打印每个变体在 `N = 131072` 时每层的 cache 大小。SWA 和稀疏变体减少 10–100 倍。双差分翻倍。明明白白地为内存账单买单。

## 实际使用

2026 年生产模式：

```python
from transformers import AutoModelForCausalLM
# Gemma 3 混合 SWA（window=1024）和全局层，比例 5:1。
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+ 的 FlexAttention 接受一个 mask 函数：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

这会编译成自定义的 Triton 内核。对于常见模式，在 FlashAttention-3 速度的 10% 以内，且 mask 函数是一个 Python 可调用对象。

**何时选择哪种：**

- **纯全注意力**——每层最高达约 16K 上下文，或当检索质量至关重要时。
- **SWA + 全局混合**——长上下文（>32K），训练和推理内存受限。2026 年 32K 以上的默认配置。
- **稀疏分块注意力**——自定义内核、自定义模式。专为专用工作负载保留（检索、音频）。
- **双差分注意力**——任何注意力汇污染有影响的工作负载（长上下文 RAG、大海捞针）。

## 交付物

参见 `outputs/skill-attention-variant-picker.md`。该 skill 根据目标上下文长度、检索需求和训练/推理计算档案，为新模型选择注意力拓扑结构。

## 练习

1. **简单。** 运行 `code/main.py`。验证 `window=4` 时 SWA 将每行最后 4 个 token 之外的全部清零。验证 `window=n` 逐位重现全因果注意力。
2. **中等。** 在第七课 capstone 上实现因果 SWA（`window=1024`）。在 tinyshakespeare 上训练 1,000 步。全注意力相比，验证损失回退多少？峰值内存下降多少？
3. **困难。** 在 capstone 模型中实现 Gemma-3 风格的 5:1 层混合（5 个 SWA，1 个全局）。在匹配参数下与纯 SWA 和纯全局基线比较损失、内存和生成质量。
4. **困难。** 实现带每个头学习到的 `λ` 的双差分注意力。在合成检索任务上训练（一个针，2000 个干扰项）。在匹配参数下测量检索准确率 vs 单注意力基线。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 滑动窗口注意力（SWA） | "局部注意力" | 每个 query attend到其最后 `W` 个 token；KV cache 缩小到 `O(W)`。 |
| 有效感受野 | "模型能看到多远" | 在窗口为 `W` 的 `L` 层 SWA 堆栈中，最远可达 `L × W` 个 token。 |
| Longformer / BigBird | "局部 + 全局 + 随机" | 带有少数始终attend的全局 token 的稀疏模式；早期长上下文方法。 |
| 原生稀疏注意力 | "DeepSeek 的内核技巧" | 学习分块级稀疏度；在内核层面跳过零分块同时保持质量。 |
| 双差分注意力 | "两张图，相减" | DIFF Transformer：从第一张注意力图中减去学习到的 `λ` 倍的第二张注意力图，以消除注意力汇。 |
| 注意力汇 | "权重泄漏到 token 0" | Softmax 归一化强制每行和为 1；无信息的 query 将权重倾倒在位置 0。 |
| FlexAttention | "Mask 即 Python" | PyTorch 2.5+ API，将任意 mask 函数编译为 FlashAttention 形状的内核。 |
| 层类型混合 | "5:1 SWA-to-global" | 在堆栈中交错稀疏和全注意力层，在更低内存下保持质量。 |

## 延伸阅读

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — 经典的滑动窗口 + 全局 token 论文。
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — 局部 + 全局 + 随机。
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — OpenAI 的局部+步进模式。
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — 1:1 SWA:global 混合。
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — 5:1 混合，窗口 1024，现在是教科书级默认配置。
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF Transformer 论文。
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-V3.2 的学习稀疏度注意力。
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) — 实际使用中 mask-as-callable 模式的 API 参考。