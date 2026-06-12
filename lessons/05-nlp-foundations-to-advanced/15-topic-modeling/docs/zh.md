# 主题建模 — LDA 与 BERTopic

> LDA：文档是主题的混合，主题是词的分布。BERTopic：文档聚类在 embedding 空间中，聚类是主题。目标相同，分解方式不同。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 5 · 02（BoW + TF-IDF）、阶段 5 · 03（Word2Vec）
**时间：** 约 45 分钟

## 问题

你有 10,000 张客服工单、50,000 篇新闻文章或 200,000 条推文。你需要在不阅读的情况下知道整个语料库讲了什么。你没有标注好的分类。甚至不知道有多少个分类。

主题建模在没有监督的情况下回答这个问题。输入一个语料库，输出一小组连贯的主题，以及每个文档在这些主题上的分布。

两大算法家族主导。LDA（2003）将每个文档视为潜在主题的混合，每个主题视为词的分布。推理是贝叶斯的。它仍在生产中运行，当你需要混合成员主题分配和可解释的词级概率分布时。

BERTopic（2020）用 BERT 对文档编码，用 UMAP 降维，用 HDBSCAN 聚类，用类别 TF-IDF 提取主题词。它在短文本、社交媒体和语义相似度比词重叠更重要的场景中胜出。每个文档一个主题，这对长文本内容是一个局限。

本课建立对两者的直觉，并指出对于给定语料库应该选哪个。

## 概念

![LDA 混合模型 vs BERTopic 聚类](../assets/topic-modeling.svg)

**LDA 生成故事。** 每个主题是词的分布。每个文档是主题的混合。要在文档中生成一个词，先从文档的混合分布中采样一个主题，再从该主题的分布中采样一个词。推理反过来：给定观察到的词，推断每个文档的主题分布和每个主题的词分布。折叠吉布斯采样或变分贝叶斯完成数学计算。

关键 LDA 输出：

- `doc_topic`：矩阵 `(n_docs, n_topics)`，每行和为 1（文档的主题混合）。
- `topic_word`：矩阵 `(n_topics, vocab_size)`，每行和为 1（主题的词分布）。

**BERTopic 流程。**

1. 用句子转换器（如 `all-MiniLM-L6-v2`）编码每个文档。384 维向量。
2. 用 UMAP 降维到约 5 维。BERT embedding 维数太高，无法直接聚类。
3. 用 HDBSCAN 聚类。基于密度，生成可变大小的聚类和一个"离群值"标签。
4. 对每个聚类，计算该聚类中文档的类别 TF-IDF 来提取顶部词汇。

输出是每个文档一个主题（加上 -1 离群值标签）。可选地，通过 HDBSCAN 的概率向量获得软成员关系。

## 构建

### 第 1 步：通过 scikit-learn 实现 LDA

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

注意：移除停用词，min_df 和 max_df 过滤罕见词和普遍词，使用 CountVectorizer（不是 TfidfVectorizer），因为 LDA期望原始计数。

### 第 2 步：BERTopic（生产环境）

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

对 `Topic != -1` 的过滤掉 BERTopic 的离群值桶（HDBSCAN 无法聚类的文档）。`min_topic_size` 控制 HDBSCAN 的最小聚类大小；BERTopic 库默认值是 10。本例中设置为 15 是针对本课的规模。对于超过 10,000 个文档的语料库，增加到 50 或 100。

### 第 3 步：评估

两种方法都输出主题词。问题是这些词是否连贯。

- **主题连贯性（c_v）。** 结合滑动窗口上下文中顶部词对的 NPMI（归一化逐点互信息），将分数聚合成主题向量，并通过余弦相似度比较这些向量。越高越好。使用 `gensim.models.CoherenceModel` 的 `coherence="c_v"`。
- **主题多样性。** 所有主题顶部词中独特词的占比。越高越好（主题不重叠）。
- **定性检查。** 阅读每个主题的顶部词。它们命名了一个真实的事物吗？人类判断仍然是最后一道防线。

## 何时选哪个

| 场景 | 选择 |
|-----------|------|
| 短文本（推文、评论、标题） | BERTopic |
| 有主题混合的长文档 | LDA |
| 无 GPU / 有限算力 | LDA 或 NMF |
| 需要文档级多主题分布 | LDA |
| LLM 集成用于主题标注 | BERTopic（直接支持） |
| 资源受限的边缘部署 | LDA |
| 最大语义连贯性 | BERTopic |

最大的实际考虑是文档长度。BERT embedding 会截断；LDA 计数适用于任何长度。对于长于 embedding 模型上下文的文档，要么分块+聚合，要么使用 LDA。

## 使用

2026 年的技术栈：

- **BERTopic。** 短文本和语义优先场景的默认选择。
- **`gensim.models.LdaModel`。** 生产环境中的经典 LDA，成熟、经过实战检验。
- **`sklearn.decomposition.LatentDirichletAllocation`。** 实验的简单 LDA。
- **NMF。** 非负矩阵分解。LDA 的快速替代方案，在短文本上质量相当。
- **Top2Vec。** 与 BERTopic 类似的设计。社区较小，但在某些基准测试上表现良好。
- **FASTopic。** 较新，在非常大的语料库上比 BERTopic 快。
- **基于 LLM 的标注。** 运行任何聚类，然后提示模型命名每个聚类。

## 交付

保存为 `outputs/skill-topic-picker.md`：

```markdown
---
name: topic-picker
description: 为语料库选择 LDA 或 BERTopic。指定库、参数、评估方法。
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

给定语料库描述（文档数量、平均长度、领域、语言、算力预算），输出：

1. 算法。LDA / NMF / BERTopic / Top2Vec / FASTopic。一句话理由。
2. 配置。主题数量：`recommended = max(5, round(sqrt(n_docs)))`，对于 40,000 以下文档的语料库限制在 200 以内；只有当语料库确实很大（>40k）时才允许 >200，并注明增加的算力成本。`min_df` / `max_df` 过滤器和神经方法的 embedding 模型也属于此处。
3. 评估。通过 `gensim.models.CoherenceModel` 的主题连贯性（c_v）、主题多样性和 20 样本人工阅读。
4. 需要探测的失败模式。对于 LDA，"垃圾主题"吸收停用词和高频词。对于 BERTopic，-1 离群聚类吞没歧义文档。

对于长于 embedding 模型上下文窗口的文档，如果没有分块策略则拒绝使用 BERTopic。对于非常短的文本（推文、10 个词以下的评论）拒绝使用 LDA，因为连贯性会崩溃。将任何低于 5 的 n_topics 选择标记为可能错误；对于40k 以下文档的语料库将 >200 标记为可能过度分割。
```

## 练习

1. **简单。** 在 20 Newsgroups 数据集上用 5 个主题拟合 LDA。打印每个主题的前 10 个词。手工标注每个主题。该算法找到了真实分类吗？
2. **中等。** 在同一 20 Newsgroups 子集上拟合 BERTopic。比较发现的主题数量、顶部词和定性连贯性，与 LDA 对比。哪个更清晰地揭示了真实分类？
3. **困难。** 在你的语料库上计算 LDA 和 BERTopic 的 c_v 连贯性。分别用 5、10、20、50 个主题运行每个模型。绘制连贯性 vs 主题数量的图。报告哪种方法在主题数量变化时更稳定。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 主题 (Topic) | 语料库所关于的事物 | 词的概率分布（LDA）或相似文档的聚类（BERTopic）。 |
| 混合成员 (Mixed membership) | 文档是多个主题 | LDA 为每个文档分配所有主题上的分布。 |
| UMAP | 降维 | 流形学习，保留局部结构；用于 BERTopic。 |
| HDBSCAN | 密度聚类 | 找到可变大小的聚类；为离群值产生"噪声"标签（-1）。 |
| c_v 连贯性 | 主题质量指标 | 滑动窗口内顶部主题词的逐点互信息的平均值。 |

## 延伸阅读

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — LDA论文。
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — BERTopic 论文。
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — 引入 c_v 的论文。
- [BERTopic文档](https://maartengr.github.io/BERTopic/) — 生产环境参考。优秀的示例。