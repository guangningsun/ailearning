# 多语言 NLP

> 一个模型，100+ 种语言，绝大多数语言零训练数据。跨语言迁移是 2020 年代最实用的奇迹。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 5 · 04（GloVe、FastText、子词）、阶段 5 · 11（机器翻译）
**时间：** 约 45 分钟

## 问题

英语有数十亿条标注样本。乌尔都语有数千条。迈蒂利语几乎没有。任何面向全球用户的实用 NLP 系统都必须应对这样的长尾语言——这些语言没有针对特定任务的训练数据。

多语言模型通过在多种语言上同时训练一个模型来解决这个问题。共享表示让模型将在高资源语言上学到的技能迁移到低资源语言。在英语情感分析上微调模型，它开箱就能对乌尔都语产生惊人的良好情感预测。这就是零样本跨语言迁移，它彻底改变了 NLP 交付给世界的方式。

本课讲述其中的权衡、经典模型，以及刚接触多语言工作的团队最容易栽跟头的一个决策：选择迁移的源语言。

## 概念

![通过共享多语言 embedding 空间实现跨语言迁移](../assets/multilingual.svg)

**共享词表。** 多语言模型使用在所有目标语言文本上训练的 SentencePiece 或 WordPiece 分词器。词表是共享的：相同的子词单元在不同相关语言中代表相同的词素。英语和意大利语中的 `anti-` 获得相同的 token。

**共享表示。** 一个在多种语言上进行掩码语言建模预训练的 transformer，会学习到不同语言中语义相似的句子会产生相似的隐藏状态。mBERT、XLM-R 和 NLLB 都具备这一特点。"cat" 的英语 embedding 与法语 "chat"、西班牙语 "gato" 的 embedding 聚集在一起，完整句子的 embedding 也是如此。

**零样本迁移。** 在一种语言（通常是英语）的标注数据上微调模型。推理时，用它处理模型支持的任何其他语言。无需目标语言标签。对于类型学上相关的语言效果很强，对于遥远的语言则较弱。

**少样本微调。** 在目标语言中添加 100-500 条标注样本。分类任务的准确率跃升至英语基线的 95-98%。这是多语言 NLP 中性价比最高的一个杠杆。

## 模型

| 模型 | 年份 | 覆盖语言 | 说明 |
|-------|------|----------|-------|
| mBERT | 2018 | 104 种语言 | 在维基百科上训练。第一个实用的多语言语言模型。低资源语言上较弱。 |
| XLM-R | 2019 | 100 种语言 | 在 CommonCrawl（比维基百科大得多）上训练。确立了跨语言基线。Base 270M，Large 550M。 |
| XLM-V | 2023 | 100 种语言 | XLM-R 配备 100 万 token 词表（vs 25 万）。低资源语言上表现更好。 |
| mT5 | 2020 | 101 种语言 | 用于多语言生成的 T5 架构。 |
| NLLB-200 | 2022 | 200 种语言 | Meta 的翻译模型；包含 55 种低资源语言。 |
| BLOOM | 2022 | 46 种语言 + 13 种编程语言 | 开源 176B LLM，多语言训练。 |
| Aya-23 | 2024 | 23 种语言 | Cohere 的多语言 LLM。阿拉伯语、印地语、斯瓦希里语表现出色。 |

根据使用场景选择。分类任务以 XLM-R-base 作为合理的默认选择。生成任务视翻译还是开放生成而定，选用 mT5 或 NLLB。LLM 式工作配合 Aya-23 或 Claude，并使用显式多语言提示。

## 源语言决策（2026 年研究）

大多数团队默认使用英语作为微调源。近期研究（2026 年）表明这往往是错误的。

语言相似度比原始语料库大小更能预测迁移质量。对于斯拉夫语目标，德语或俄语往往优于英语。对于印度语目标，印地语往往优于英语。**qWALS** 相似度指标（2026 年，基于世界语言结构地图集特征）可量化这一点。**LANGRANK**（Lin 等，ACL 2019）是另一种更早的方法，从语言相似度、语料库大小和遗传相关性等多方面对候选源语言进行排序。

实用规则：如果目标语言有一个类型学上接近的高资源亲属语言，先尝试在该语言上微调，然后与英语微调比较。

## 动手构建

### 第 1 步：零样本跨语言分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

一个模型，三种语言，同一套 API。在 NLI 数据上训练的 XLM-R 通过蕴含技巧很好地迁移到分类任务。

### 第 2 步：多语言 embedding 空间

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

翻译结果在 embedding 空间中位置接近。不同的英语句子则距离更远。这正是跨语言检索、聚类和相似度工作的基础。

### 第 3 步：少样本微调策略

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

对于 100-500 条目标语言样本，`num_train_epochs=5` 和 `learning_rate=2e-5` 是安全的默认配置。更高的学习率会导致多语言对齐崩溃，得到一个只有英语能力的模型。

## 真正有效的评估

- **各语言在保留集上的准确率。** 不要汇总。汇总数字会掩盖长尾问题。
- **与单语言基线对比。** 对于有足够数据的语言，从头训练的单语言模型有时会击败多语言模型。要测试。
- **实体级测试。** 目标语言中的命名实体。多语言模型往往对远离拉丁字母体系的文字体系分词能力较弱。
- **跨语言一致性。** 相同含义的两种语言应该产生相同的预测。衡量这个差距。

## 实际使用

2026 年技术栈：

| 任务 | 推荐方案 |
|-----|-------------|
| 分类，100 种语言 | XLM-R-base（~270M）微调 |
| 零样本文本分类 | `joeddav/xlm-roberta-large-xnli` |
| 多语言句子 embedding | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 翻译，200 种语言 | `facebook/nllb-200-distilled-600M`（见第 11 课） |
| 生成式多语言 | Claude、GPT-4、Aya-23、mT5-XXL |
| 低资源语言 NLP | XLM-V 或在相关高资源语言上进行领域特定微调 |

如果关注性能，总要在目标语言上预留微调预算。零样本是起点，不是终点答案。

### 分词税（低资源语言会出什么问题）

多语言模型在所有语言间共享一个分词器。这个词表在以英语、法语、西班牙语、汉语、德语为主的语料库上训练。对于主流语言集之外的任何语言，三种税会悄然叠加：

- **fertility 税。** 低资源语言文本分词后每词 token 数远多于英语。一个印地语句子可能需要相当于等效英语句子的 3-5 倍 token 数。这 3-5 倍会蚕食你的上下文窗口、训练效率和延迟。
- **变体恢复税。** 每个拼写错误、音标变体、Unicode 规范化不匹配或大小写变体都会成为冷启动的不相关序列，在 embedding 空间中孤立存在。模型无法学习母语者认为显而易见的正字法对应关系。
- **容量溢出税。** 税 1 和税 2 消耗上下文位置、层深和 embedding 维度。剩余给实际推理的容量系统性地小于高资源语言从同一模型获得的容量。

实际症状：模型在印地语上训练正常，损失曲线看起来正确，评估困惑度也合理，但生产输出微妙地错误。形态学在句中崩溃。稀有屈折形式无法恢复。**你无法通过数据规模来弥补一个坏掉的分词器。**

缓解方法：为目标语言选择分词覆盖率好的分词器（XLM-V 的 100 万 token 词表是直接解决方案）；训练前在保留的目标语文本上验证分词 fertility；对于真正长尾的文字体系使用字节级后备（SentencePiece `byte_fallback=True`，GPT-2 风格的字节级 BPE），确保没有任何内容是 OOV。

## 交付

保存为 `outputs/skill-multilingual-picker.md`：

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## 练习

1. **简单。** 在英语、法语、印地语和阿拉伯语各 10 句话上运行零样本分类流水线。报告每种语言的准确率。法语应该很强，印地语尚可，阿拉伯语参差不齐。
2. **中等。** 使用 `paraphrase-multilingual-MiniLM-L12-v2` 在一个小规模混合语言语料库上构建跨语言检索器。用英语查询，从任意语言检索文档。测量 recall@5。
3. **困难。** 比较英语源和印地语源的微调对印地语分类任务的效果。使用 500 条目标语言样本进行少样本微调，两种方案均如此。报告哪种源语言产生了更好的印地语准确率，以及好多少。这就是 LANGRANK 论文的迷你版。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 多语言模型 | 一个模型，多种语言 | 跨语言共享词表和参数。 |
| 跨语言迁移 | 用一种语言训练，用另一种语言运行 | 在源语言上微调，在目标语言上评估，无需目标语言标签。 |
| 零样本 | 无目标语言标签 | 不在目标语言上微调直接迁移。 |
| 少样本 | 小规模目标语言标签 | 使用 100-500 条目标语言样本进行微调。 |
| mBERT | 第一个多语言 LM | 在维基百科上预训练的 104 种语言 BERT。 |
| XLM-R | 标准跨语言基线 | 在 CommonCrawl 上预训练的 100 种语言 RoBERTa。 |
| NLLB | Meta 的 200 语言 MT | No Language Left Behind。包含 55 种低资源语言。 |

## 延伸阅读

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — XLM-R 论文。
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — 开启跨语言迁移研究线的分析论文。
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — NLLB-200 论文。
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Aya，Cohere 的多语言 LLM。
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — qWALS / LANGRANK 源语言论文。