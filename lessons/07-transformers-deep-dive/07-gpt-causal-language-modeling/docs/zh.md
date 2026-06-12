# GPT — 因果语言建模

> BERT 看到两侧。GPT 只看过去。三角掩码是现代 AI 中最具影响力的一行代码。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 7 · 02（自注意力）、阶段 7 · 05（完整 Transformer）、阶段 7 · 06（BERT）
**时间：** 约 75 分钟

## 问题

语言模型回答一个问题：已知前 `t-1` 个 token，第 `t` 个 token 的概率分布是什么？在这个信号上训练——下一个 token 预测——你得到一个能一次一个 token 生成任意文本的模型。

要在整个序列上端到端并行训练，你需要让每个位置的预测只依赖于更早的位置。否则模型会通过查看答案而轻松作弊。

因果掩码做到了这一点。它是一个上三角矩阵，填充 `-inf`，在 softmax 之前加到注意力分数上。经过 softmax 后，这些位置变成 0。每个位置只能 attending 到自己和更早的位置。而且因为你一次性作用于整个序列，你在一轮前向传播中得到 N 个并行的下一个 token 预测。

GPT-1（2018）、GPT-2（2019）、GPT-3（2020）、GPT-4（2023）、GPT-5（2024）、Claude、Llama、Qwen、Mistral、DeepSeek、Kimi——它们都是纯解码器的因果 transformer，有着相同的核心循环。只是更大、更好的数据、更好的 RLHF。

## 概念

![因果掩码创建三角注意力矩阵](../assets/causal-attention.svg)

### 掩码

给定一个长度为 `N` 的序列，构建一个 `N × N` 矩阵：

```
M[i, j] = 0       如果 j <= i
M[i, j] = -inf    如果 j > i
```

在 softmax 之前将 `M` 加到原始注意力分数上。`exp(-inf) = 0`，所以被掩码的位置贡献零权重。注意力矩阵的每一行都是只关于之前位置的概率分布。

实现代价：一次 `torch.tril()` 调用。计算时间：纳秒级。对领域的影响：一切。

### 并行训练，串行推理

训练：将整个 `(N, d_model)` 序列前向传播一次，计算 N 个交叉熵损失（每个位置一个），求和，反向传播。沿序列方向并行。这就是 GPT 训练能够扩展的原因——你可以在一次 GPU 传递中处理 100 万个 token 的批次。

推理：逐 token 生成。输入 `[t1, t2, t3]`，得到 `t4`。输入 `[t1, t2, t3, t4]`，得到 `t5`。输入 `[t1, t2, t3, t4, t5]`，得到 `t6`。KV 缓存（第 12 课）保存 `t1…tn` 的隐藏状态，这样你不必在每一步重新计算它们。但推理时的串行深度 = 输出长度。这就是自回归的代价，也是为什么解码是每个 LLM 的延迟瓶颈。

### 损失——偏移一位

给定 token `[t1, t2, t3, t4]`：

- 输入：`[t1, t2, t3]`
- 目标：`[t2, t3, t4]`

对每个位置 `i`，计算 `-log P(target_i | inputs[:i+1])`。求和。这就是整个序列的交叉熵。

你听说过的每个 transformer 语言模型都用这个损失训练。预训练、微调、SFT——相同的损失，不同的数据。

### 解码策略

训练后，采样选择比人们想象的更重要。

| 方法 | 做什么 | 何时使用 |
|--------|--------------|-------------|
| 贪心 | 每步取 argmax | 确定性任务、代码补全 |
| 温度 | 将 logits 除以 T，再采样 | 创造性任务，T 越高越多样化 |
| Top-k | 只从 top-k 个 token 中采样 | 去掉低概率尾部 |
| Top-p（核） | 从累积概率 ≥ p 的最小集合中采样 | 2020+ 默认；适应分布形状 |
| Min-p | 保留 `p > min_p * max_p` 的 token | 2024+；比 top-p 更好地拒绝长尾 |
| 投机解码 | 草稿模型提议 N 个 token，大模型验证 | 相同质量下延迟降低 2–3 倍 |

在 2026 年，min-p + 温度 0.7 是开源模型的一个合理默认。投机解码是任何生产推理栈的基本要求。

### 什么让"GPT 配方"奏效

1. **纯解码器。** 没有编码器开销。每层一次注意力 + FFN。
2. **扩展。** 124M → 1.5B → 175B → 万亿。Chinchilla 扩展定律（第 13 课）告诉你如何分配算力。
3. **上下文学习。** 在约 6B–13B 时涌现。模型无需微调就能遵循 few-shot 示例。
4. **RLHF。** 在人类偏好上进行后训练，将原始预训练文本转化为聊天助手。
5. **Pre-norm + RoPE + SwiGLU。** 大规模稳定训练。

自 GPT-2 以来，核心架构没有太大变化。所有有趣的事情都发生在数据、规模和后训练中。

## 从零实现

### 第 1 步：因果掩码

参见 `code/main.py`。一行代码：

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

在 softmax 之前将它加到注意力分数上。这就是整个机制。

### 第 2 步：一个 2 层 GPT 风格的模型

堆叠两个解码器块（带掩码的自注意力 + FFN，无交叉注意力）。添加 token embedding、位置编码和一个 unembedding（绑定到 token embedding 矩阵——自 GPT-2 以来的标准技巧）。

### 第 3 步：下一个 token 预测，端到端

在 20 token 的玩具词表上，在每个位置输出 logits。根据偏移一位的目标计算交叉熵损失。无梯度——这是前向传播的完整性检查。

### 第 4 步：采样

实现贪心、温度、top-k、top-p、min-p。在固定 prompt 上运行每个并比较输出。采样函数 10 行代码。

## 实际使用

PyTorch，2026 年惯用法：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

在底层，`generate()` 运行前向传播，取最终位置的 logits，采样下一个 token，附加它，然后重复。每个生产 LLM 推理栈（vLLM、TensorRT-LLM、llama.cpp、Ollama、MLX）都用大量优化实现了相同的循环——批处理 prefill、连续批处理、KV 缓存分页、投机解码。

**GPT vs BERT，各一行：** GPT 预测 `P(x_t | x_{<t})`。BERT 预测 `P(x_masked | x_unmasked)`。损失函数决定了模型是否能生成。

## 交付物

参见 `outputs/skill-sampling-tuner.md`。这个 skill 为新的生成任务选择采样参数，并在需要确定性解码时发出标志。

## 练习

1. **简单。** 运行 `code/main.py` 并验证因果注意力矩阵在 softmax 后是下三角的。抽查：第 3 行应该只在第 0–3 列有权重。
2. **中等。** 实现宽度为 4 的束搜索。在 10 个短 prompt 上比较束宽-4 与贪心的困惑度。束搜索总是赢吗？（提示：对翻译通常如此，对开放式聊天则不然。）
3. **困难。** 实现投机解码：用 2 层模型作为草稿，6 层模型作为验证器。在 100 个长度为 64 的补全上测量墙上时钟加速。确认输出与验证器的贪心结果一致。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 因果掩码 | "三角" | 上三角 `-inf` 矩阵，加到注意力分数上，使位置 `i` 只能看到位置 `≤ i`。 |
| 下一个 token 预测 | "损失" | 在每个位置，模型分布相对于真实下一个 token 的交叉熵。 |
| 自回归 | "一次生成一个" | 将输出反馈作为输入；只在训练时并行，生成时不并行。 |
| Logits | "softmax 前的分数" | LM head 在 softmax 之前的原始输出；采样基于这些。 |
| 温度 | "创造力旋钮" | 将 logits 除以 T；T→0 = 贪心，T→∞ = 均匀。 |
| Top-p | "核采样" | 将分布截断到累积和 ≥p 的最小集合；从剩余部分采样。 |
| Min-p | "比 top-p 更好" | 保留 `p ≥ min_p × max_p` 的 token；截断适应分布的锐度。 |
| 投机解码 | "草稿 + 验证" | 便宜模型提议 N 个 token；大模型并行验证。 |
| 教师强制 | "训练技巧" | 训练时，喂入真实的上一个 token，而不是模型的预测。每个 seq2seq LM 的标准做法。 |

## 延伸阅读

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1。
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2。
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 和上下文学习。
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 投机解码论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 标准的因果 LM 参考代码。