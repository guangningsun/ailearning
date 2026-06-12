# 子词分词 — BPE、WordPiece、Unigram、SentencePiece

> 词级分词器遇到未登录词就卡死。字符级分词器让序列长度爆炸。子词分词器取两者之长。每一台前沿 LLM 都运行在上面。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 5 · 01（文本处理）、阶段 5 · 04（GloVe / FastText / 子词）
**时间：** 约 60 分钟

## 问题

你的词表有 50,000 个词。用户输入"untokenizable"。你的分词器返回 `[UNK]`。模型现在对这个词完全没有信号。更糟的是：语料库中第 90 百分位的文档有 40 个生僻词，意味着每个文档丢失 40 比特的信息。

子词分词器解决了这个问题。常用词保持为单一 token。生僻词分解成有意义的片段：`untokenizable` → `un`, `token`, `izable`。训练数据覆盖一切，因为任何字符串归根结底都是字节序列。

2026 年的每一台前沿 LLM 都运行在三种算法之一（BPE、Unigram、WordPiece）上，并被包裹在三个库之一中（tiktoken、SentencePiece、HF Tokenizers）。选择分词器是发布语言模型的必经之路。

## 概念

![BPE vs Unigram vs WordPiece，逐字符对比](../assets/subword-tokenization.svg)

**BPE（Byte-Pair Encoding，字节对编码）。** 从字符级词表开始。统计所有相邻字符对。合并出现频率最高的字符对为一个新 token。重复直到达到目标词表大小。主导算法：GPT-2/3/4、Llama、Gemma、Qwen2、Mistral。

**字节级 BPE。** 相同算法，但操作原始字节（256 个基础 token）而非 Unicode 字符。保证零 `[UNK]` token——任何字节序列都可以编码。GPT-2 使用 50,257 个 token（256字节 + 50,000 次合并 + 1 个特殊 token）。

**Unigram。** 从一个大词表开始。为每个 token 分配一个 Unigram 概率。迭代剪枝那些移除后对语料库对数似然影响最小的 token。推理时是概率性的：可以对分词结果采样（通过子词正则化进行数据增强很有用）。被 T5、mBART、ALBERT、XLNet、Gemma 使用。

**WordPiece。** 合并能最大化训练语料库似然度的字符对，而非原始频率。被 BERT、DistilBERT、ELECTRA 使用。

**SentencePiece vs tiktoken。** SentencePiece 是一个对原始 Unicode 文本直接*训练*词表的库（ BPE 或 Unigram），将空格编码为 `▁`。tiktoken 是 OpenAI 针对预建词表的快速*编码器*；它不训练。

经验法则：

- **训练新词表：** SentencePiece（多语言，无需预分词）或 HF Tokenizers。
- **针对 GPT 词表快速推理：** tiktoken（cl100k_base、o200k_base）。
- **两者都要：** HF Tokenizers — 一个库，训练和服务都能用。

## 动手实现

### 第 1 步：从零实现 BPE

参见 `code/main.py`。循环逻辑：

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

算法编码了三个要点。`</w>` 标记词尾，这样"low"（后缀）和"lower"（前缀）就能区分。频率加权使得高频字符对优先被合并。合并列表是有序的——推理时按训练顺序应用合并。

### 第 2 步：用学到的合并规则编码

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

朴素实现 O(n·|merges|)。生产级实现（tiktoken、HF Tokenizers）使用合并等级查找加优先队列，运行时间接近线性。

### 第 3 步：实际使用 SentencePiece

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # 或 "unigram"
    character_coverage=0.9995, # CJK 语言设低一些（如日语 0.995，英语 0.9995）
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注意：无需预分词，空格编码为 `▁`，`character_coverage` 控制生僻字符是保留还是映射到 `<unk>`。

### 第 4 步：用 tiktoken 处理 OpenAI 兼容词表

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

仅编码。快速（Rust 后端）。与 GPT-4/5 的分词结果在字节计数、成本估算、上下文窗口预算方面完全一致。

## 2026 年仍会踩到的坑

- **分词器漂移。** 用词表 A 训练，部署时用词表 B。Token ID 不同，模型输出垃圾。在 CI 中检查 `tokenizer.json` 的哈希值。
- **空格歧义。** BPE 中"hello"和" hello"产生不同的 token。总要明确指定 `add_special_tokens` 和 `add_prefix_space`。
- **多语言训练不足。** 英语为主的语料库产生的词表会将非拉丁文字分割成 5-10 倍的 token。同样的提示在日语/阿拉伯语上消耗 GPT-3.5 的成本是英语的 5-10 倍。o200k_base 部分解决了这个问题。
- **Emoji 分割。** 一个 Emoji 可以占用 5 个 token。在预算上下文时检查 Emoji 处理。

## 实际使用

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| 从零训练单语模型 | HF Tokenizers（BPE） |
| 训练多语言模型 | SentencePiece（Unigram，`character_coverage=0.9995`） |
| 提供 OpenAI 兼容 API | tiktoken（GPT-4+ 用 `o200k_base`） |
| 领域特定词表（代码、数学、蛋白质） | 在领域语料库上训练自定义 BPE，与基础词表合并 |
| 边缘推理，小模型 | Unigram（更小的词表效果更好） |

词表大小是一个缩放决策，不是常数。粗略启发式：<1B 参数用 32k，1-10B 用 50-100k，多语言/前沿模型用 200k+。

## 交付物

保存为 `outputs/skill-bpe-vs-wordpiece.md`：

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## 练习

1. **简单。** 在 `code/main.py` 的小语料库上训练一个 500 次合并的 BPE。编码三个留出词。有多少恰好产生 1 个 token vs >1 个 token？
2. **中等。** 在 100 句英文维基百科句子上比较 `cl100k_base`、`o200k_base` 和你自己用 vocab=32k 训练的 SentencePiece BPE 的 token 数量。报告每种的压缩率。
3. **困难。** 用 BPE、Unigram 和 WordPiece 训练同一语料库。在小型情感分类器上测量每种分词方式的下游准确率。这个选择会让 F1 移动超过 1 分吗？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding（字节对编码） | 贪婪合并出现频率最高的相邻字符对，直到达到目标词表大小。 |
| 字节级 BPE | 永远不会有未知 token | 在原始 256 字节上做 BPE；GPT-2 / Llama 使用这个。 |
| Unigram | 概率分词器 | 用对数似然度从大候选集中剪枝；被 T5、Gemma 使用。 |
| SentencePiece | 那个处理空格的 | 在原始文本上训练 BPE/Unigram 的库；空格编码为 `▁`。 |
| tiktoken | 那个快的 | OpenAI 的 Rust 后端 BPE 编码器，用于预建词表。不训练。 |
| 合并列表 | 那些神奇数字 | 有序的 `(a, b) → ab` 合并列表；推理时按顺序应用。 |
| 字符覆盖率 | 多罕见算太罕见？ | 分词器必须覆盖的训练语料库中字符的分数；~0.9995 是典型值。 |

## 延伸阅读

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPE 论文。
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — Unigram 论文。
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — 这个库。
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — 简洁参考。
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) —  cookbook + 编码列表。