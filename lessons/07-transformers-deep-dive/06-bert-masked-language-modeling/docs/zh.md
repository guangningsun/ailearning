# BERT — 掩码语言建模

> GPT 预测下一个词。BERT 预测缺失的词。一句话的差别 —— 然后是整整五年里所有与 embedding 相关的一切。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 7 · 05（完整 Transformer）、阶段 5 · 02（文本表示）
**时间：** 约 45 分钟

## 问题

2018 年，每个 NLP 任务 —— 情感分析、命名实体识别、问答、文本蕴含 —— 都要在各自的标注数据上从头训练自己的模型。当时没有预训练的"理解英语"的检查点可供微调。ELMo（2018）表明可以用双向 LSTM 预训练上下文 embedding；它有帮助但不能泛化。

BERT（Devlin et al. 2018）提出了一个问题：如果我们用一个 transformer 编码器，在互联网上每一个句子上训练它，强迫它从双侧上下文预测缺失的词会怎样？然后你在下游任务上微调一个 head。参数效率带来了突破。

结果：18 个月内，BERT 及其变体（RoBERTa、ALBERT、ELECTRA）横扫了所有存在的 NLP 排行榜。到 2020 年，地球上每一个搜索引擎、内容审核管道和语义搜索系统里都有一个 BERT 在里面。

2026 年，纯编码器模型仍然是分类、检索和结构化抽取的正确工具 —— 它们每个 token 的运行速度比解码器快 5–10×，且它们的 embedding 是每一个现代检索栈的支柱。ModernBERT（2024 年 12 月）通过 Flash Attention + RoPE + GeGLU 将架构推至 8K 上下文。

## 概念

![掩码语言建模：选取 token，掩码它们，预测原始词](../assets/bert-mlm.svg)

### 训练信号

取一个句子：`the quick brown fox jumps over the lazy dog`。

随机掩码 15% 的 token：

```
输入：  the [MASK] brown fox jumps [MASK] the lazy dog
目标：  the  quick brown fox jumps  over  the lazy dog
```

训练模型预测被掩码位置上的原始 token。因为编码器是双向的，在位置 1 预测 `[MASK]` 时可以使用位置 2+ 的 `brown fox jumps`。这就是 GPT 做不到的事。

### BERT 的掩码规则

在被选中用于预测的 15% 的 token 中：

- 80% 被替换为 `[MASK]`。
- 10% 被替换为一个随机 token。
- 10% 保持不变。

为什么不总是用 `[MASK]`？因为 `[MASK]` 在推理时永远不会出现。将模型训练成 100% 的掩码位置都期望看到 `[MASK]`会在预训练和微调之间造成分布偏移。10% 随机 + 10% 不变使模型保持诚实。

### 下一句预测（NSP）—— 以及为什么它被抛弃了

原始 BERT 也在 NSP 上训练：给定两个句子 A 和 B，预测 B 是否跟在 A 后面。RoBERTa（2019）消融了它，表明 NSP 有害无益。现代编码器跳过它。

### 2026 年的变化：ModernBERT

2024 年的 ModernBERT 论文用 2026 年的原语重建了这个块：

| 组件 | 原始 BERT (2018) | ModernBERT (2024) |
|-----------|----------------------|-------------------|
| 位置编码 | 学习式绝对编码 | RoPE |
| 激活函数 | GELU | GeGLU |
| 归一化 | LayerNorm | Pre-norm RMSNorm |
| 注意力 | 全密集型 | 交替局部（128）+ 全局 |
| 上下文长度 | 512 | 8192 |
| 分词器 | WordPiece | BPE |

而且与 2018 年的堆栈不同，它是原生支持 Flash Attention 的。在序列长度 8K 下，推理速度比 DeBERTa-v3 快 2–3 倍，GLUE 分数更优。

### 2026 年仍然选择编码器的场景

| 任务 | 为什么编码器优于解码器 |
|------|---------------------------|
| 检索 / 语义搜索 embedding | 双向上下文 = 每个 token 的 embedding 质量更高 |
| 分类（情感、意图、毒性） | 一次前向传播；无生成开销 |
| NER / token 标注 | 逐位置输出，原生双向 |
| 零样本蕴含（NLI） | 编码器之上的分类 head |
| RAG 排序器 | 交叉编码器评分，比 LLM 排序器快 10 倍 |

## 动手实现

### 第 1 步：掩码逻辑

参见 `code/main.py`。函数 `create_mlm_batch` 接收 token ID 列表、词表大小和掩码概率。返回输入 ID（已应用掩码）和标签（仅在被掩码位置有值，其余为 -100 —— PyTorch 的忽略索引约定）。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### 第 2 步：在小规模语料上运行 MLM 预测

在一个 20 词词表、200 个句子的数据集上训练一个 2 层编码器 + MLM head。不涉及梯度 —— 我们做前向传播的合理性检查。完整训练需要 PyTorch。

### 第 3 步：比较掩码类型

展示三路规则如何在没有 `[MASK]` 的情况下保持模型可用。在一个未掩码的句子和一个已掩码的句子上预测。两者都应该产生合理的 token 分布，因为模型在训练中见过这两种模式。

### 第 4 步：微调 head

用一个分类 head 替换 MLM head，在玩具情感数据集上微调。只有 head 训练；编码器被冻结。这是每一个 BERT 应用都遵循的模式。

## 实际使用

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**Embedding 模型是微调后的 BERT。** 像 `all-MiniLM-L6-v2` 这样的 `sentence-transformers` 模型是用对比损失训练的 BERT。编码器是一样的，只是损失函数变了。

**交叉编码器排序器也是微调后的 BERT。** 在 `[CLS] query [SEP] doc [SEP]` 上的配对分类。查询和文档之间的双向注意力正是交叉编码器相对于双编码器的质量优势所在。

**2026 年什么时候不选 BERT。** 任何生成性任务。编码器没有合理的方式来自回归地生成 token。还有：当参数小于 1B 时，一个小型解码器可以以更大的灵活性匹配质量（Phi-3-Mini、Qwen2-1.5B）。

## 交付物

参见 `outputs/skill-bert-finetuner.md`。该技能为一个新的分类或抽取任务规划 BERT 微调（backbone 选择、head 规范、数据、评估、停止条件）。

## 练习

1. **简单。** 运行 `code/main.py` 并打印 10,000 个 token 上的掩码分布。确认约 15% 被选中，其中约 80% 变成 `[MASK]`。
2. **中等。** 实现全词掩码：如果一个词被切分成子词，则将所有子词一起掩码或不掩码。测量这是否提高了 500 句语料上的 MLM 准确率。
3. **困难。** 在公共数据集的 10,000 个句子上训练一个微型的（2 层，d=64）BERT。用 `[CLS]` token 微调 SST-2 情感。与匹配参数量的纯解码器基线比较 —— 哪个更好？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| MLM | "掩码语言建模" | 训练信号：随机将 15% 的 token 替换为 `[MASK]`，预测原始 token。 |
| 双向 (Bidirectional) | "双向看" | 编码器注意力没有因果掩码 —— 每个位置都能看到所有其他位置。 |
| `[CLS]` | "池化 token" | 一个特殊 token，被 prepend 到每个序列的开头；其最终 embedding 用作句子级表示。 |
| `[SEP]` | "分段分隔符" | 分隔成对的序列（例如查询/文档、句子 A/B）。 |
| NSP | "下一句预测" | BERT 的第二个预训练任务；在 RoBERTa 中被证明无用，2019 年后被抛弃。 |
| 微调 (Fine-tuning) | "适配到某个任务" | 保持编码器大部分冻结；在顶部训练一个小的 head 用于下游任务。 |
| 交叉编码器 (Cross-encoder) | "排序器" | 一个 BERT，同时将查询和文档作为输入，输出一个相关性分数。 |
| ModernBERT | "2024 年更新版" | 用 RoPE、RMSNorm、GeGLU、交替局部/全局注意力、8K 上下文重建的编码器。 |

## 延伸阅读

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — 原始论文。
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — 如何正确训练 BERT；杀死了 NSP。
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — 替换 token 检测在匹配计算量下超越 MLM。
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — ModernBERT 论文。
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — 标准编码器参考。