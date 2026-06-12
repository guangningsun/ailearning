# 自然语言推理 — 文本蕴含

> "t 蕴含 h" 意味着人类在阅读 t 后会得出 h 为真的结论。NLI 的任务就是预测这种蕴含 / 矛盾 / 中立。表面平淡无奇，生产环境中却举足轻重。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 5 · 05（情感分析）、阶段 5 · 13（问答系统）
**时间：** 约 60 分钟

## 问题

你做了一个摘要模型。它生成了一段摘要。你怎么知道摘要中没有幻觉？

你做了一个聊天机器人。它回答了"是的"。你怎么知道这个回答被检索到的段落所支持？

你需要对 10,000 篇新闻文章按主题分类。你没有训练标签。能复用某个模型吗？

这三个问题都可以归结为自然语言推理。NLI 问的是：给定一个前提 `t` 和一个假设 `h`，`h` 是被 `t` 蕴含、被矛盾，还是中立（无关）？

- **幻觉检测：** `t` = 源文档，`h` = 摘要中的断言。不是蕴含 = 幻觉。
- **有据可查的 QA：** `t` = 检索到的段落，`h` = 生成的答案。不是蕴含 = 捏造。
- **零样本分类：** `t` = 文档，`h` = 标签的语言化表达（"这是关于体育的"）。蕴含 = 预测标签。

一个任务，三种生产用途。这就是为什么每个 RAG 评估框架底层都塞了一个 NLI 模型。

## 概念

![NLI：三类分类，前提 vs 假设](../assets/nli.svg)

**三个标签。**

- **蕴含。** `t` → `h`。"猫在垫子上"蕴含"有一只猫。"
- **矛盾。** `t` → ¬`h`。"猫在垫子上"与"没有猫"矛盾。
- **中立。** 两种推理都不成立。"猫在垫子上"与"猫饿了"中立。

**不是逻辑蕴含。** NLI 是*自然*语言推理 —— 即一个普通读者会推理出什么，而不是严格逻辑。如果把所有权公理化，"John 遛了他的狗"在 NLI 中蕴含"John 有一只狗"，但严格的一阶逻辑只有在你显式公理化所有权时才会承认这一点。

**数据集。**

- **SNLI**（2015）。57 万个人类标注的对，图像描述作为前提。领域较窄。
- **MultiNLI**（2017）。43.3 万对，涵盖 10 个领域。2026 年的标准训练语料。
- **ANLI**（2019）。对抗性 NLI。人类专门编写了旨在击穿现有模型的例子。更难。
- **DocNLI、ConTRoL**（2020–21）。文档级前提。测试多跳和长距离推理。

**架构。** Transformer 编码器（BERT、RoBERTa、DeBERTa）读取 `[CLS] 前提 [SEP] 假设 [SEP]`。`[CLS]` 表示喂入一个 3 类 softmax。在 MNLI 上训练，在保留基准上评估，在同分布对上的准确率可达 90%+。

**通过 NLI 实现零样本。** 给定一个文档和候选标签，将每个标签转化为一个假设（"这段文本是关于体育的"）。计算每个的蕴含概率，取最大值。这就是 Hugging Face `zero-shot-classification` pipeline 背后的机制。

## 构建

### 第 1 步：运行预训练 NLI 模型

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

对于生产级 NLI，`facebook/bart-large-mnli` 和 `microsoft/deberta-v3-large-mnli` 是开源默认选择。DeBERTa-v3 在排行榜上名列前茅。

### 第 2 步：零样本分类

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

默认模板是"This example is about {label}."。可以用 `hypothesis_template` 自定义。无需训练数据。无需微调。开箱即用。

### 第 3 步：RAG 的可信度检查

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

这就是 RAGAS 可信度的核心。将生成的答案拆解为原子主张。用每个主张与检索到的上下文做 NLI 检查。报告蕴含的部分比例。

### 第 4 步：手写 NLI 分类器（概念层面）

见 `code/main.py`：一个仅用标准库的小玩具：前提和假设通过词汇重叠 + 否定检测来比较。与 transformer 模型没有竞争力 —— 但它展示了任务的模样：两个文本输入，3 类标签输出，损失 = 在 `{entail, contradict, neutral}` 上的交叉熵。

## 陷阱

- **仅假设的捷径。** 模型仅从假设就能以 ~60% 的准确率预测标签（SNLI 上），因为"not"、"nobody"、"never"与矛盾相关。检测标签泄露的强基线。
- **词汇重叠启发式。** 子序列启发式（"每个子序列都被蕴含"）能通过 SNLI 但在 HANS/ANLI 上失败。使用对抗基准。
- **文档级性能下降。** 单句 NLI 模型在文档级前提上 F1 下降 20+。长上下文请用 DocNLI 训练的模型。
- **零样本模板敏感性。** "This example is about {label}" vs "{label}" vs "The topic is {label}" 可以带来 10+ 分的准确率波动。调优模板。
- **领域不匹配。** MNLI 基于通用英语训练。法律、医疗、科学文本需要领域特定的 NLI 模型（如 SciNLI、MedNLI）。

## 使用

2026 技术栈：

| 使用场景 | 模型 |
|---------|-------|
| 通用 NLI | `microsoft/deberta-v3-large-mnli` |
| 快速 / 边缘部署 | `cross-encoder/nli-deberta-v3-base` |
| 零样本分类（轻量） | `facebook/bart-large-mnli` |
| 文档级 NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 多语言 | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAG 中的幻觉检测 | RAGAS / DeepEval 内部的 NLI 层 |

2026 元模式：NLI 是文本理解的" duct tape"。每当需要"A 支持 B 吗？"或"A 与 B 矛盾吗？"时 —— 先用 NLI，别急着再调一次 LLM。

## 交付

保存为 `outputs/skill-nli-picker.md`：

```markdown
---
name: nli-picker
description: 为分类 / 可信度检查 / 零样本任务选择 NLI 模型、标签模板和评估设置。
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

给定一个使用场景（可信度检查、零样本分类、文档级推理），输出：

1. 模型。命名的 NLI 检查点。理由与领域、长度、语言相关。
2. 模板（如果是零样本）。语言化模式。示例。
3. 阈值。决策规则的蕴含截断值。基于校准的理由。
4. 评估。在保留标注集上的准确率、仅假设基线、对抗子集。

没有 100 个示例标注的合理性检查，拒绝交付零样本分类。没有文档级前提的情况下拒绝使用句级 NLI 模型。标记任何声称 NLI 能解决幻觉的说法 —— 它能减少幻觉，但不能消除它。
```

## 练习

1. **简单。** 在 20 个人工制作的（前提、假设、标签）三元组上运行 `facebook/bart-large-mnli`，覆盖所有三个类别。测量准确率。加入对抗性"子序列启发式"陷阱（"我没吃蛋糕" vs "我吃了蛋糕"），看是否会被击穿。
2. **中等。** 在 100 条 AG News 标题上比较零样本模板 `"This text is about {label}"` vs `"The topic is {label}"` 和 `"{label}"`。报告准确率波动。
3. **困难。** 构建一个 RAG 可信度检查器：原子主张分解 + 每主张的 NLI。在 50 个带黄金上下文的 RAG 生成答案上评估。与人工标签相比，测量假阳性和假阴性率。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| NLI | 自然语言推理 | 前提-假设关系的三类分类。 |
| RTE | 识别文本蕴含 | NLI 的旧称；同一任务。 |
| 蕴含 | "t 蕴含 h" | 一个普通读者在给定 t 的情况下会得出 h 为真。 |
| 矛盾 | "t 排除 h" | 一个普通读者在给定 t 的情况下会得出 h 为假。 |
| 中立 | "未决定" | 从 t 到 h 两种推理都不成立。 |
| 零样本分类 | 用 NLI 做分类器 | 将标签语言化为假设，取最大蕴含。 |
| 可信度 | 答案是否被支持？ | 在（检索上下文，生成答案）上的 NLI。 |

## 延伸阅读

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI。
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI。
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — ANLI 基准。
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — 用 NLI 做分类器。
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — 2026 年 NLI 的主力模型。
