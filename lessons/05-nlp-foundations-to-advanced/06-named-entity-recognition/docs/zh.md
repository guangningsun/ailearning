# 命名实体识别

> 把名字抽出来。听起来容易，直到你遇到模糊边界、嵌套实体和领域 jargon。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 5 · 02（BoW + TF-IDF）、阶段 5 · 03（词嵌入）
**时间：** 约 75 分钟

## 问题

"Apple sued Google over its iPhone search deal in the US." 五个实体：Apple（ORG）、Google（ORG）、iPhone（PRODUCT）、search deal（也许）、US（GPE）。一个好的 NER 系统提取所有实体并给出正确类型。一个差的一个漏掉 iPhone，把 Apple 水果和 Apple 公司混淆，把 "US" 标记为 PERSON。

NER 是每个结构化抽取 pipeline下面的工作马。简历解析、合规日志扫描、医疗记录匿名化、搜索查询理解、聊天机器人响应的 grounding、法律合同抽取。你从来看不到它；但你一直依赖它。

本课走过经典路径（基于规则、HMM、CRF）进入现代路径（BiLSTM-CRF，然后是 transformer）。每一步都解决了前一步的特定限制。这个模式就是本课要讲的内容。

## 概念

**BIO 标记**（或 BILOU）把实体抽取变成序列标注问题。用 `B-TYPE`（实体开头）、`I-TYPE`（实体内部）或 `O`（不在任何实体中）标记每个 token。

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

多 token 实体链：`New B-GPE`、`York I-GPE`、`City I-GPE`。理解 BIO 的模型可以抽取任意跨度。

架构演进：

- **基于规则。** 正则 + 词典查找。对已知实体高精确率，对新实体零覆盖。
- **HMM。** 隐马尔可夫模型。给定标签的 token 发射概率，标签到标签的转移概率。Viterbi 解码。在标注数据上训练。
- **CRF。** 条件随机场。像 HMM 但是判别的，所以你可以混合任意特征（词形、大小写、邻词）。2026 年仍然是低资源部署的经典生产工作马。
- **BiLSTM-CRF。** 用神经特征代替手工特征。LSTM 双向读取句子，CRF 层在顶部强制一致的标签序列。
- **基于 Transformer。** 用 token分类头微调 BERT。最高准确率。最高算力。

## 构建

### 第 1 步： BIO 标记辅助函数

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### 第 2 步：手工特征

对于经典（非神经）NER，特征就是一切。有用的特征：

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` 返回 `xXxxxx`。`word_shape("USA-2024")` 返回 `XXX-dddd`。大写模式对专有名词是高信号。

### 第 3 步：简单基于规则 + 词典基线

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

生产级词典有数百万条从 Wikipedia 和 DBpedia 抓取的条目。覆盖率很好。消歧（公司 Apple 对比水果 Apple）很糟糕。这就是统计模型胜出的原因。

### 第 4 步：CRF 步骤（概述，非完整实现）

没有概率论基础，50 行从零实现完整 CRF不会有启发性。使用 `sklearn-crfsuite` 代替：

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` 和 `c2` 是 L1 和 L2 正则化。`all_possible_transitions=True` 让模型学习非法序列（如 `O` 后面跟 `I-ORG`）是不可能的，这就是 CRF 如何强制执行 BIO 一致性而不需要你写约束。

### 第 5 步：BiLSTM-CRF 增加了什么

特征变成学习到的。输入：token embedding（GloVe 或 fastText）。LSTM 从左到右和从右到左读取。拼接的隐藏状态通过 CRF 输出层。CRF 仍然强制执行标签序列一致性；LSTM 用学习到的特征替代手工特征。

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

对于 CRF 层，使用 `torchcrf.CRF`（pip install pytorch-crf）。相比手工 CRF 的提升可测量，但没有你期望的那么大，除非你有数万条标注句子。

## 使用

spaCy 开箱即用生产级 NER。

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

注意 `iPhone` 被标记为 `ORG` 而不是 `PRODUCT` — spaCy 的小模型产品实体覆盖弱。大模型（`en_core_web_lg`）更好。Transformer 模型（`en_core_web_trf`）更好。

Hugging Face 用于基于 BERT 的 NER：

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` 将连续的 B-X、I-X token 合并为一个 span。没有它，你得到 token 级标签，需要自己合并。

### 基于 LLM 的 NER（2026 年的选项）

零样本和少样本 LLM NER 现在在许多领域与微调模型具有竞争力，当标注数据稀缺时更是如此。

- **零样本提示。** 给 LLM 一个实体类型列表和一个示例 schema。要求 JSON 输出。开箱即用；在新领域上准确率中等。
- **ZeroTuneBio 风格提示。** 将任务分解为候选抽取 → 含义解释 → 判断 → 重新检查。多阶段提示（不是单次）显著提升生物医学 NER 的准确率。同样的模式适用于法律、金融和科学领域。
- **带 RAG 的动态提示。** 从小型标注种子集中为每次推理调用检索最相似的标注样本；动态构建少样本提示。在 2026 年基准上，这使 GPT-4 生物医学 NER F1 比静态提示提升 11-12%。
- **按实体类型分解。** 对于长文档，单次调用抽取所有实体类型会在长度增长时丢失召回率。每个实体类型运行一次抽取 pass。更高的推理成本，明显更高的准确率。这是临床笔记和法律合同的标准模式。

截至 2026 年的生产建议：在收集训练数据之前先用 LLM 零样本基线。通常 F1 已经足够好，你永远不需要微调。

### 经典 NER 仍然胜出的地方

即使有 LLM 可用，经典 NER 在以下情况下胜出：

- 延迟预算低于 50ms。
- 你有数千条标注样本且需要 98%+ F1。
- 领域有稳定本体论，预训练 CRF 或 BiLSTM 可以很好地迁移。
- 监管约束要求本地、非生成式模型。

### 它失效的地方

- **领域偏移。** 在法律合同上用 CoNLL 训练的 NER 表现比词典差。在你的领域上微调。
- **嵌套实体。** "Bank of America Tower" 同时是 ORG 和 FACILITY。标准 BIO 无法表示重叠的 span。你需要嵌套 NER（多 pass 或基于 span 的模型）。
- **长实体。** "United States Federal Deposit Insurance Corporation." Token 级模型有时会切分它。使用 `aggregation_strategy` 或后处理。
- **稀疏类型。** 医疗 NER 标签如 DRUG_BRAND、ADVERSE_EVENT、DOSE。通用模型一无所知。ScispaCy 和 BioBERT 是那里的起点。

## 交付

保存为 `outputs/skill-ner-picker.md`：

```markdown
---
name: ner-picker
description: 为给定的抽取任务选择正确的 NER 方法。
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

给定任务描述（领域、标签集、语言、延迟、数据量），输出：

1. 方法。基于规则 + 词典、CRF、BiLSTM-CRF 或 transformer 微调。
2. 起始模型。命名它（spaCy 模型 ID、Hugging Face checkpoint ID 或"自定义，从零训练"）。
3. 标注策略。 BIO、BILOU 或基于 span。一句话说明理由。
4. 评估。使用 `seqeval`。始终报告实体级 F1（不是 token 级）。

如果标注样本少于 500 条，除非用户已有预训练的领域模型，否则拒绝推荐微调 transformer。标记嵌套实体需要基于 span 或多 pass 模型。如果用户提到"生产规模"且标签与 CoNLL-2003 相同，要求进行词典审计。
```

## 练习

1. **简单。** 实现 `bio_to_spans`（`spans_to_bio` 的逆函数），在 10 个句子上验证往返一致性。
2. **中等。** 在 CoNLL-2003 英语 NER 数据集上训练上面的 sklearn-crfsuite CRF。使用 `seqeval` 报告每个实体的 F1。典型结果：约 84 F1。
3. **困难。** 在特定领域 NER 数据集（医疗、法律或金融）上微调 `distilbert-base-cased`。与 spaCy 小模型比较。记录数据泄露检查并写出让你惊讶的地方。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| NER | 抽取名字 | 用类型（PERSON、ORG、GPE、DATE……）标记 token span。 |
| BIO | 标注方案 | `B-X` 开始，`I-X` 继续，`O` 在外面。 |
| BILOU | 更好的 BIO | 添加 `L-X`（最后）、`U-X`（单元）以获得更清晰的边界。 |
| CRF | 结构化分类器 | 对标签之间的转移建模，而不只是发射。强制有效序列。 |
| 嵌套 NER | 重叠实体 | 一个 span 是另一个 span 的不同实体。 BIO 无法表达这一点。 |
| 实体级 F1 | 正确的 NER 指标 | 预测的 span 必须与真实 span 完全匹配。 Token 级 F1 高估准确率。 |

## 延伸阅读

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — BiLSTM-CRF 论文。经典。
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — 引入了成为标准的 token 分类模式。
- [spaCy 语言学特性 — 命名实体](https://spacy.io/usage/linguistic-features#named-entities) — `Doc.ents` 和 `Span` 上每个属性的实用参考。
- [seqeval](https://github.com/chakki-works/seqeval) — 正确的指标库。始终使用它。