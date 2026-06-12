# 机器翻译

> 翻译是 NLP 研究三十年来的主要资金来源，至今仍是。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 5 · 10（注意力机制）、阶段 5 · 04（GloVe、FastText、子词）
**时间：** 约 75 分钟

## 问题

一个模型读取一种语言的句子，输出另一种语言的句子。长度不一致。词序不一致。源语言中的一些词可能对应目标语言中的多个词，反之亦然。惯用语拒绝一对一映射。"I miss you" 在法语中是 "tu me manques"——字面意思是 "you are lacking to me"。没有词级对齐能承受这样的挑战。

机器翻译是迫使 NLP 发明编码器-解码器、注意力机制、Transformer，最终催生整个 LLM 范式的任务。每一次进步都源于翻译质量可衡量，而人机之间的差距始终顽固存在。

本课跳过历史回顾，直接教授 2026 年的工作流程：预训练多语言编码器-解码器（NLLB-200 或 mBART）、子词分词、束搜索、BLEU 和 chrF 评估，以及少数至今仍在生产环境中未被捕获的失败模式。

## 概念

![MT 流程：分词 → 编码 → 带注意力的解码 → 去除分词](../assets/mt-pipeline.svg)

现代 MT 是一个基于 Transformer 编码器-解码器的模型，在平行文本上训练。编码器以源语言的分词方式读取源文本。解码器通过交叉注意力（第十课）使用编码器的输出，逐个子词生成目标文本。解码使用束搜索以避免贪婪解码的陷阱。输出经过去除分词、还原大小写，然后与参考译文进行评分。

三个操作选择决定了实际 MT 质量。

- **分词器。** 在混合语言语料库上训练的 SentencePiece BPE。跨语言共享词汇表是 NLLB 实现零样本语言对的原因。
- **模型大小。** NLLB-200 蒸馏版 600M 可在笔记本电脑上运行。NLLB-200 3.3B 是发布的生产默认值。54.5B 是研究天花板。
- **解码。** 通用内容的束宽为 4-5。长度惩罚以避免输出过短。当需要术语一致性时使用约束解码。

## 构建

### 第 1 步：预训练 MT 调用

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

这里有三个关键点。`src_lang` 告诉分词器使用哪种文字体系和分词规则。`forced_bos_token_id` 告诉解码器生成哪种语言。两者都是 NLLB 特有的技巧；mBART 和 M2M-100 使用各自的约定，不能互换。

### 第 2 步：BLEU 和 chrF

BLEU 衡量输出与参考之间的 n-gram 重叠。四个参考 n-gram 大小（1-4），精度的几何均值，对过短输出的简短惩罚。分数范围为 [0, 100]。常用。但解释起来令人沮丧：30 BLEU 是 "可用"；40 是 "良好"；50 是 "卓越"；差异小于 1 BLEU 就是噪声。

chrF 衡量字符级 F-score。对形态丰富的语言更敏感，因为 BLEU 会低估匹配数。常与 BLEU 一起报告。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

始终使用 `sacrebleu`。它标准化了分词方式，使分数在不同论文之间具有可比性。自己计算 BLEU 是导致误导性基准测试的常见原因。

### 三层评估体系（2026 年）

现代 MT 评估使用三种互补的指标家族。至少使用两种再发布。

- **启发式**（BLEU、chrF）。快速、基于参考、可解释、对释义不敏感。用于遗留比较和回归检测。
- **学习型**（COMET、BLEURT、BERTScore）。在人类判断上训练的神经模型；比较翻译与源文本和参考的语义相似度。COMET 自 2023 年以来与 MT 研究的关联度最高，在 2026 年是质量优先场景的生产默认值。
- **LLM 即评判**（无参考）。提示大型模型在流畅性、充分性、语气、文化适宜性方面对翻译进行评分。当评分标准设计良好时，GPT-4 即评判与人类一致率约 80%。用于没有参考的开放式内容。

实用的 2026 技术栈：`sacrebleu` 用于 BLEU 和 chrF，`unbabel-comet` 用于 COMET，以及提示 LLM 用于最终面向人类的信号。在将每个指标用于生产数据之前，用 50-100 个人工标注的示例进行校准。

无参考指标（COMET-QE、BLEURT-QE、LLM 即评判）允许你在没有参考的情况下评估翻译，这对于不存在参考翻译的长尾语言对非常重要。

### 第 3 步：生产中会出什么错

上述工作流程 80% 的时间能流利翻译，其余 20% 会静默失败。已命名的失败模式：

- **幻觉。** 模型发明了源文本中没有的内容。在不熟悉的领域词汇中常见。症状：输出流畅，但声称了源文本未陈述的事实。缓解措施：对领域术语使用约束解码，对受监管内容进行人工审查，监控输出是否比输入长得多。
- **目标语言错误生成。** 模型翻译成错误的语言。NLLB 在稀有语言对上出奇地容易出现这个问题。缓解措施：验证 `forced_bos_token_id`，始终在输出上运行语言 ID 模型检查。
- **术语漂移。** "Sign up" 在文档 1 中变成 "s'inscrire"，在文档 2 中变成 "créer un compte"。对于 UI 文本和面向用户的字符串，一致性比原始质量更重要。缓解措施：词汇表约束解码或后编辑字典。
- **正式程度不匹配。** 法语的 "tu" vs "vous"，日语的礼貌级别。模型选择训练中最常见的形式。对于面向客户的内容，这通常是错误的。缓解措施：如果模型支持，用正式程度 token 作为提示前缀，或在正式语料库上微调小型模型。
- **短输入的长度爆炸。** 很短的输入句子往往产生过长的翻译，因为长度惩罚在 ~5 个源 token 以下急剧下降。缓解措施：硬性最大长度上限，与源长度成正比。

### 第 4 步：领域微调

预训练模型是通才。在领域平行数据上微调对法律、医学或游戏对话翻译有明显帮助。方法不难：

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

几千个高质量的平行示例胜过几十万个嘈杂的网络抓取示例。训练数据质量是最大的生产杠杆。

## 使用

2026 年 MT 的生产技术栈：

| 使用场景 | 推荐起点 |
|---------|---------------------------|
| 任意语言对、200 种语言 | `facebook/nllb-200-distilled-600M`（笔记本电脑）或 `nllb-200-3.3B`（生产） |
| 以英语为中心、高质量、50 种语言 | `facebook/mbart-large-50-many-to-many-mmt` |
| 短时运行、廉价推理、英语-法语/德语/西班牙语 | Helsinki-NLP / Marian 模型 |
| 延迟敏感的浏览器端 | ONNX 量化的 Marian（~50 MB） |
| 最高质量、愿意付费 | GPT-4 / Claude / Gemini + 翻译提示 |

截至 2026 年，LLM 在多个语言对上已超越专业 MT 模型，特别是在惯用语内容和长上下文方面。权衡是每 token 成本和延迟。当上下文长度、风格一致性或通过提示的领域适应性比吞吐量更重要时，选择 LLM。

## 发布

保存为 `outputs/skill-mt-evaluator.md`：

```markdown
---
name: mt-evaluator
description: 评估机器翻译输出是否可以发布。
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

给定源文本和候选翻译，输出：

1. 自动评分估计。你期望的 BLEU 和 chrF 范围。说明是否有参考译文可用。
2. 五点人工核查清单：(a) 内容保留（无幻觉），(b) 语言正确，(c) 语域/正式程度匹配，(d) 术语与提供的词汇表一致，(e) 无截断或长度爆炸。
3. 一个需要探究的领域特定问题。例如：法律方面——命名实体和法规引用；医学方面——药物名称和剂量；UI 方面——占位符变量 `{name}`。
4. 置信度标志。"发布" / "带审查发布" / "不可发布"。与第 2 步中发现的问题严重程度挂钩。

未经输出上的语言 ID 检查，拒绝发布翻译。未经参考译文，拒绝评估，除非用户明确选择无参考评分（COMET-QE、BLEURT-QE）。标记超过 1000 token 的内容为可能需要分块翻译。
```

## 练习

1. **简单。** 使用 `nllb-200-distilled-600M` 将一个 5 句英文段落翻译成法语，再译回英语。测量往返翻译与原文的接近程度。你应该看到语义保留但用词会有漂移。
2. **中等。** 使用 `fasttext lid.176` 或 `langdetect` 对翻译输出实现语言 ID 检查。集成到 MT 调用中，以便在返回之前捕获目标语言错误生成。
3. **困难。** 在你选择的 5,000 对领域语料库上微调 `nllb-200-distilled-600M`。在微调前后在保留集上测量 BLEU。报告哪些类型的句子有所改进，哪些退化了。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| BLEU | 翻译分数 | 带简短惩罚的 n-gram 精度。[0, 100]。 |
| chrF | 字符 F-score | 字符级 F-score。对形态丰富的语言更敏感。 |
| NMT | 神经机器翻译 | 基于平行文本训练的 Transformer 编码器-解码器。2017 年以来的默认值。 |
| NLLB | 不让任何语言掉队 | Meta 的 200 种语言 MT 模型系列。 |
| 约束解码 | 受控输出 | 强制特定 token 或 n-gram 出现在输出中或不出现。 |
| 幻觉 | 虚构内容 | 源文本不支持的模型输出。 |

## 延伸阅读

- [Costa-jussà 等 (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — NLLB 论文。
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — 为什么 `sacrebleu` 是报告 BLEU 的唯一正确方式。
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — chrF 论文。
- [Hugging Face MT 指南](https://huggingface.co/docs/transformers/tasks/translation) — 实用的微调演练。
