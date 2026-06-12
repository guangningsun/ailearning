# T5、BART — 编码器-解码器模型

> 编码器理解。解码器生成。把它们放在一起，你就得到一个为输入→输出任务而构建的模型：翻译、摘要、改写、转录。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 7 · 05（完整 Transformer）、阶段 7 · 06（BERT）、阶段 7 · 07（GPT）
**时间：** 约 45 分钟

## 问题

纯解码器 GPT 和纯编码器 BERT 各自为不同目标精简了 2017 年的架构。但很多任务天然是输入-输出的：

- 翻译：英语 → 法语。
- 摘要：5000 token 的文章 → 200 token 的摘要。
- 语音识别：音频 token → 文本 token。
- 结构化抽取：散文 → JSON。

对这些任务，编码器-解码器是最简洁的适配。编码器产生源的密集表示。解码器在每一步生成输出时交叉 attending 到该表示。训练时在输出端偏移一位。与 GPT 相同的损失，只是以编码器输出为条件。

两篇论文定义了现代 playbook：

1. **T5**（Raffel et al. 2019）。"Text-to-Text Transfer Transformer"。将每个 NLP 任务重新表述为文本输入、文本输出。单一架构、单一词表、单一损失。预训练于掩码跨度预测（破坏输入中的跨度，在输出中解码它们）。
2. **BART**（Lewis et al. 2019）。"Bidirectional and Auto-Regressive Transformer"。去噪自编码器：用多种方式破坏输入（打乱、掩码、删除、旋转），让解码器重建原始内容。

在 2026 年，编码器-解码器格式在输入结构重要的地方仍然存在：

- Whisper（语音 → 文本）。
- 谷歌的翻译栈。
- 一些具有独特上下文和编辑结构的代码补全/修复模型。
- Flan-T5 及其变体用于结构化推理任务。

纯解码器抢尽了风头，但编码器-解码器从未消失。

## 概念

![带交叉注意力的编码器-解码器](../assets/encoder-decoder.svg)

### 前向循环

```
源 token ─▶ 编码器 ─▶ (N_src, d_model)  ──┐
                                            │
目标 token ─▶ 解码器块                       │
              ├─▶ 带掩码的自注意力           │
              ├─▶ 交叉注意力 ◀───────────┘
              └─▶ FFN
             ↓
           下一个 token logits
```

关键的是，编码器对每个输入只运行一次。解码器是自回归的，但在每一步都交叉 attending 到*相同的*编码器输出。缓存编码器输出对长输入是免费的加速。

### T5 预训练——跨度破坏

随机选取输入中的跨度（平均长度 3 token，总量 15%）。用唯一的哨兵 token 替换每个跨度：`<extra_id_0>`、`<extra_id_1>` 等。解码器只输出带有哨兵前缀的被破坏跨度：

```
源：The quick <extra_id_0> fox jumps <extra_id_1> dog
目标：<extra_id_0> brown <extra_id_1> over the lazy
```

比预测整个序列更便宜。在 T5 论文的消融实验中与 MLM（BERT）和前缀-LM（UniLM）具有竞争力。

### BART 预训练——多噪声去噪

BART 尝试五种噪声函数：

1. Token 掩码。
2. Token 删除。
3. 文本填充（掩码一个跨度，解码器插入正确长度）。
4. 句子排列。
5. 文档旋转。

文本填充 + 句子排列的组合产生了最好的下游数字。解码器始终重建原始内容。BART 的输出是完整序列，而不仅仅是破坏的跨度——所以预训练算力高于 T5。

### 推理

与 GPT 相同的自回归生成。贪心/束/top-p 采样适用。束搜索（宽度 4–5）是翻译和摘要的标准，因为输出分布比聊天更窄。

### 2026 年何时选择各变体

| 任务 | 编码器-解码器？ | 为什么 |
|------|------------------|-----|
| 翻译 | 是，通常 | 清晰的源序列；固定的输出分布；束搜索有效 |
| 语音转文本 | 是（Whisper） | 输入模态与输出不同；编码器塑造音频特征 |
| 聊天/推理 | 否，纯解码器 | 没有持久的"输入"——对话本身就是序列 |
| 代码补全 | 通常否 | 纯解码器加长上下文胜出；代码模型如 Qwen 2.5 Coder 是纯解码器 |
| 摘要 | 两者皆可 | BART、PEGASUS 击败了早期的纯解码器基线；现代纯解码器 LLM 与之匹敌 |
| 结构化抽取 | 两者皆可 | T5 很简洁，因为"文本 → 文本"吸收任何输出格式 |

自 ~2022 年以来的趋势：纯解码器接管了编码器-解码器曾经占据的任务，因为 (a) 指令调优的纯解码器 LLM 通过提示泛化到任何任务，(b) 一种架构比两种更容易扩展，(c) RLHF 假定一个解码器。编码器-解码器在输入模态不同（语音、图像）或束搜索质量重要的地方保留下来。

## 从零实现

参见 `code/main.py`。我们为玩具语料库实现 T5 风格的跨度破坏——这是本课最有用的部分，因为它出现在此后的每个编码器-解码器预训练配方中。

### 第 1 步：跨度破坏

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """选取总和约为 mask_rate 的跨度。返回 (被破坏的输入, 目标)。"""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

目标格式是 T5 惯例：`<sent0> span0 <sent1> span1 ...`。被破坏的输入将未更改的 token 与跨度位置的哨兵 token 交织在一起。

### 第 2 步：验证往返

给定被破坏的输入和目标，重建原始句子。如果你的破坏是可逆的，前向传播就是定义良好的。这是一个完整性检查——真实训练从不这样做，但测试便宜，能捕获跨度簿记中的 off-by-one bug。

### 第 3 步：BART 噪声

五个函数：`token_mask`、`token_delete`、`text_infill`、`sentence_permute`、`document_rotate`。组合其中两个并展示结果。

## 实际使用

HuggingFace 参考：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5 的技巧：任务名称进入输入文本。相同的模型处理数十个任务，因为每个任务都是文本输入、文本输出。在 2026 年这个模式已被指令调优的纯解码器模型泛化，但 T5 是第一个将其正式化的。

## 交付物

参见 `outputs/skill-seq2seq-picker.md`。这个 skill 根据输入-输出结构、延迟和质量目标为一个新任务在编码器-解码器和纯解码器之间选择。

## 练习

1. **简单。** 运行 `code/main.py`，对一个 30 token 的句子应用跨度破坏，验证连接非哨兵源 token 与解码目标跨度可以重现原始句子。
2. **中等。** 实现 BART 的 `text_infill` 噪声：用单个 `<mask>` token 替换随机跨度，解码器必须推断出正确的跨度长度和内容。展示一个例子。
3. **困难。** 在一个小型英语 → 猪拉丁语语料库（200 对）上微调 `flan-t5-small`。在一个保留的 50 对集合上测量 BLEU。与在相同数据上用相同算力微调 `Llama-3.2-1B` 进行比较。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 编码器-解码器 | "Seq2seq transformer" | 两个堆栈：用于输入的双向编码器，用于输出的带交叉注意力的因果解码器。 |
| 交叉注意力 | "源对目标说话的地方" | 解码器的 Q × 编码器的 K/V。编码器信息进入解码器的唯一地方。 |
| 跨度破坏 | "T5 的预训练技巧" | 用哨兵 token 替换随机跨度；解码器输出这些跨度。 |
| 去噪目标 | "BART 的玩法" | 对输入应用噪声函数，训练解码器重建干净序列。 |
| 哨兵 token | "`<extra_id_N>` 占位符" | 特殊 token，在源中标记被破坏的跨度并在目标中重新标记。 |
| Flan | "指令调优的 T5" | 在 >1,800 个任务上微调的 T5；使编码器-解码器在指令跟随上具有竞争力。 |
| 束搜索 | "解码策略" | 在每步保留 top-k 个部分序列；翻译/摘要的标准。 |
| 教师强制 | "训练时的输入" | 训练时，喂入真实的上一个输出 token，而不是采样的那个。 |

## 延伸阅读

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART。
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5。
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper，2026 年标准的编码器-解码器。
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — 参考实现。