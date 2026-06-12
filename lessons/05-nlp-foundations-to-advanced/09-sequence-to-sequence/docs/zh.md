# 序列到序列模型

> 两个 RNN 假装自己是翻译器。它们遭遇的瓶颈，正是注意力机制诞生的原因。

**类型：** 动手构建
**语言：** Python
**前置条件：** 阶段 5 · 08（用于文本的 CNN + RNN）、阶段 3 · 11（PyTorch 入门）
**时间：** 约 75 分钟

## 问题

分类将变长序列映射为单个标签。翻译将变长序列映射为另一个变长序列。输入和输出属于不同词表，可能是不同语言，长度也不保证相等。

Seq2seq 架构（Sutskever, Vinyals, Le, 2014）用一条刻意简单的配方破解了这个难题。两个 RNN。一个读取源句子并生成一个固定大小的上下文向量。另一个读取该向量并逐个 token 生成目标句子。 Lesson 08 里写的同一套代码，只是粘合方式不同。

研究这个架构有两个理由。第一，上下文字向量瓶颈是 NLP 中最具教学意义的失败案例。它解释了注意力机制和 Transformer 所擅长的一切。第二，训练配方（教师强制、计划采样、推理时的束搜索）至今仍适用于包括 LLM 在内的所有现代生成系统。

## 概念

**编码器。** 一个读取源句子的 RNN。它的最终隐藏状态就是**上下文向量**——对整个输入的固定大小摘要。理论上，除了源句子什么都丢了。

**解码器。** 另一个从上下文向量初始化的 RNN。每一步它以上一步生成的 token 为输入，生成目标词表上的一个分布。用采样或 argmax 选下一个 token。反馈回去。重复直到产生 `<EOS>` token 或达到最大长度。

**训练：** 每个解码步骤的交叉熵损失，对整个序列求和。标准的时间反向传播穿越两个网络。

**教师强制。** 训练时，解码器在步骤 `t` 的输入是位置 `t-1` 的*真实* token，而不是解码器自己的前一个预测。这稳定了训练；没有它，早期错误会级联放大，模型永远学不会。推理时，你只能用模型自己的预测，所以总存在训练/推理分布差异。这个差异叫做**曝光偏差**。

**瓶颈。** 编码器学到的关于源句子的所有信息，必须被压缩进一个上下文向量。长句子会丢失细节。稀有词会模糊。语序调整（chat noir vs. black cat）必须被记住，而不是被计算出来。

注意力机制（Lesson 10）通过让解码器看到*每一个*编码器隐藏状态来修复这个问题，而不仅仅是最后一个。这就是全部卖点。

## 动手构建

### 第 1 步：编码器

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` 的形状是 `[batch, seq_len, hidden_dim]`——每个输入位置一个隐藏状态。`hidden` 的形状是 `[1, batch, hidden_dim]`——最后一步。 Lesson 08 说"对 outputs 做池化用于分类"。这里我们保留最后一个隐藏状态作为上下文向量，忽略每步的 outputs。

### 第 2 步：解码器

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

解码器每次只被调用一步。输入：一批单个 token 和当前隐藏状态。输出：下一个 token 的词表 logit 和更新后的隐藏状态。

### 第 3 步：带教师强制训练循环

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

有两个值得注意的旋钮。`ignore_index=0` 跳过填充 token 的损失。`teacher_forcing_ratio` 是每一步使用真实 token 与模型预测的概率。从 1.0（完全教师强制）开始，在训练过程中退火到约 0.5，以缩小曝光偏差差距。

### 第 4 步：推理循环（贪心）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

贪心解码每一步都选择最高概率的 token。它可能会走偏：一旦你提交了一个 token，就无法收回。**束搜索**保留 top-`k` 个部分序列到最后，选择得分最高的完整序列。束宽 3-5 是标准配置。

### 第 5 步：瓶颈演示

在一个玩具复制任务上训练模型：源 `[a, b, c, d, e]`，目标 `[a, b, c, d, e]`。增加序列长度。观察准确率。

```
seq_len=5   复制准确率: 98%
seq_len=10  复制准确率: 91%
seq_len=20  复制准确率: 62%
seq_len=40  复制准确率: 23%
```

一个单独的 GRU 隐藏状态无法无损地存储 40 个 token 的输入。信息在每个编码器步骤都存在，但解码器只能看到最后一个状态。注意力机制直接解决了这个问题。

## 实际使用

PyTorch 有 `nn.Transformer` 和基于 `nn.LSTM` 的 seq2seq 模板。 Hugging Face 的 `transformers` 库提供了在数十亿 token 上训练过的完整编码器-解码器模型（BART, T5, mBART, NLLB）。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代编码器-解码器用 Transformer 取代了 RNN。每个块内部的机制不同，但高层形状（编码器、解码器、逐 token 生成）与 2014 年的 seq2seq 论文完全一致。

### 何时仍应使用基于 RNN 的 seq2seq

对于新项目，几乎从不。只有特定例外：

- 流式翻译，需要以有限内存逐个 token 消费输入。
- 设备端文本生成，Transformer 的内存成本过高。
- 教学目的。理解编码器-解码器瓶颈是理解 Transformer 为何获胜的最快路径。

### 曝光偏差及其缓解方法

- **计划采样。** 在训练期间退火教师强制比例，让模型学会从自己的错误中恢复。
- **最小风险训练。** 用句子级 BLEU 分数而不是 token 级交叉熵来训练。更接近你真正想要的东西。
- **强化学习微调。** 用一个指标奖励序列生成器。用于现代 LLM 的 RLHF。

这三种方法至今仍适用于基于 Transformer 的生成。

## 交付

保存为 `outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: 为给定任务设计一个序列到序列流水线。
phase: 5
lesson: 09
---

给定一个任务（翻译、摘要、改写、问题重述），输出：

1. 架构。预训练 Transformer 编码器-解码器（BART, T5, mBART, NLLB）是默认选项。仅在特定约束下使用基于 RNN 的 seq2seq。
2. 起始检查点。命名它（`facebook/bart-base`、`google/flan-t5-base`、`facebook/nllb-200-distilled-600M`）。将检查点与任务和语言覆盖范围匹配。
3. 解码策略。贪心用于确定性输出，束搜索（宽度 4-5）用于质量，温度采样用于多样性。一句话说明理由。
4. 上线前要验证的一个失败模式。曝光偏差表现为较长输出上的生成漂移；对第 90 百分位长度采样 20 个输出，肉眼检查。

拒绝推荐在少于一百万个平行样例的情况下从零训练 seq2seq。将任何对用户面向内容使用贪心解码的流水线标记为脆弱（贪心会重复和循环）。
```

## 练习

1. **简单。** 实现玩具复制任务。在输入输出相同（目标等于源）的对上训练 GRU seq2seq。在长度 5、10、20 处测量准确率。复现瓶颈。
2. **中等。** 添加束宽为 3 的束搜索解码。在小型平行语料上用 BLEU 测量与贪心的对比。记录束搜索何时获胜（通常在最后几个 token）和何时没有区别。
3. **困难。** 在 10k 对改写数据集上微调 `facebook/bart-base`。将微调模型的束宽-4 输出与基础模型在保留输入上的输出进行比较。报告 BLEU 并挑选 10 个定性示例。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 编码器 | 输入 RNN | 读取源句子。生成每步隐藏状态和最终上下文向量。 |
| 解码器 | 输出 RNN | 从上下文向量初始化。逐个 token 生成目标 token。 |
| 上下文向量 | 摘要 | 最终编码器隐藏状态。固定大小。注意力机制解决的瓶颈。 |
| 教师强制 | 使用真实 token | 训练时输入真实的前一个 token。稳定学习。 |
| 曝光偏差 | 训练/测试差距 | 在真实 token 上训练的模型从未练习过从自己的错误中恢复。 |
| 束搜索 | 更好的解码 | 每一步保留 top-k 个部分序列，而不是贪心地提交。 |

## 延伸阅读

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 原始 seq2seq论文。四页。
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — 引入了 GRU 和编码器-解码器框架。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 注意力论文。学完本课后立即阅读。
- [PyTorch NLP 从零开始教程](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — 可构建的 seq2seq + 注意力代码。