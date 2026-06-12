# 预训练数据流水线

> 模型是一面镜子。你喂什么数据，它就反映什么数据。喂垃圾，它就以流利的语言反映垃圾。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 10，第 01-02 课（分词器、分词器构建）
**时间：** 约 90 分钟

## 学习目标

- 构建一个流式数据流水线，在不将所有数据加载到内存的情况下，对 TB 级文本进行分词、分块、打乱和批处理
- 实现真实预训练流水线中使用的数据质量过滤器（去重、语言检测、内容过滤）
- 创建具有正确注意力掩码和文档边界处理的固定长度训练序列
- 分析流水线吞吐量，确保数据加载器跟上 GPU 训练速度

## 问题

你有一个分词器。现在你需要数据。

不是数据集。不是 CSV 文件。是 TB 级的文本——清洗、去重、质量过滤、分词成固定长度序列，并以随机批次快速送达，使你的 8-GPU 集群无需等待下一批次。

大多数人都以为训练 LLM 就是模型架构的事。其实不是。Llama 3 使用了 15.6 万亿个 token。GPT-3 使用了 3000 亿个。DeepSeek-V2 使用了 8.1 万亿个。三者的架构大致相同：堆叠的 Transformer 块，带有注意力层和前馈层。输出质量的差异压倒性地来自数据。

DeepMind 的 Chinchilla 论文将这一点精确化了。对于给定的计算预算，模型参数与训练 token 数之间存在一个最优比例。Chinchilla 表明，2022 年的大多数模型都严重训练不足——相对于它们所见的数据量，参数太多了。一个在 1.4 万亿个 token 上训练的 70B 参数模型（Chinchilla 最优），表现优于一个在 3000 亿个 token 上训练的 280B 模型（Gopher）。

你的数据流水线决定了模型学到的是语言还是噪声。

## 概念

### 数据从哪里来

每个大语言模型都是在多种来源的混合数据上训练的。确切的组成对大多数实验室来说都是严格保密的，但我们足以了解其分类。

| 来源 | 规模 | 质量 | 使用者 |
|--------|------|---------|---------|
| Common Crawl | 原始约 250 TB | 低（需要大量过滤） | GPT-3、Llama、大多数开源模型 |
| Wikipedia | 约 20 GB | 高 | 每个主要 LLM |
| GitHub 代码 | 约 1 TB+ | 中等（大量重复、死代码） | StarCoder、CodeLlama、DeepSeek-Coder |
| 书籍（BookCorpus、Pile） | 约 100 GB | 高 | GPT-2、GPT-3、早期模型 |
| 学术论文（arXiv、S2ORC） | 约 100 GB | STEM 领域高 | Llama、Galactica |
| StackOverflow、Reddit | 约 100 GB | 中等 | Llama、 Falcon |
| 精选网络数据（C4、RefinedWeb） | 约 5 TB | 中高（预过滤） | T5、Falcon |

Llama 3 披露了其数据混合比：大约 50% 网络数据、25% 代码、13% 书籍和学术论文、8% 数学数据、4% 多语言网络数据。总计来自超过 5 TB 原始文本源的 15.6 万亿个 token。

比例与总量同样重要。太多网络数据，模型就会变成 Reddit 复读机。太少代码，它就不会编程。太少数学，它就在推理上拉胯。要把这个混合比调好，是训练 LLM 最困难的部分之一，没有公式——需要实验和评估。

### 数据清洗

原始网络数据非常脏。一个典型的 Common Crawl 转储包含：

- HTML 标签和 JavaScript
- 样板页眉、页脚、导航菜单
- 重复页面（精确重复和近似重复）
- 机器生成的垃圾信息
- 个人身份信息（PII）
- 低质量文本（关键词列表、SEO 垃圾信息）
- 编码为文本的非文本内容

清洗这些不是可选项。这是生成连贯段落的模型和输出混有产品列表的 HTML 标签的模型之间的区别。

```mermaid
graph TD
    A[原始文本] --> B[HTML 剥离]
    B --> C[语言检测]
    C --> D[质量过滤]
    D --> E[去重]
    E --> F[PII 移除]
    F --> G[清洗后文本]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
    style G fill:#1a1a2e,stroke:#e94560,color:#fff
```

每个步骤消除一类噪声：

**HTML 剥离：** 移除所有标记。只保留可见的文本内容。`trafilatura` 或 `readability` 等库在提取文章内容的同时丢弃导航、广告和样板内容。

**语言检测：** 使用 fastText 的语言识别模型（lid.176.bin）对每个文档进行分类。过滤到你的目标语言。一个被分类为英语但置信度低于 0.8 的文档可能不是干净的英语。

**质量过滤：** 这是一个有趣的环节。RefinedWeb（Falcon 背后的数据集）使用基于困惑度的过滤器：在 Wikipedia 上训练一个小语言模型，然后对每个文档打分。高困惑度意味着该文档不像 Wikipedia——可能是垃圾信息、关键词列表或机器生成的内容。困惑度超过阈值的文档会被移除。

**去重：** 最具影响力的单一清洗步骤。Common Crawl 包含大量重复页面——法律免责声明、Cookie 通知、服务条款。在重复数据上训练会浪费计算资源，并可能导致模型记住并逐字 regurgitate 特定段落。

**PII 移除：** 姓名、电子邮件地址、电话号码、社会安全号码。基于正则表达式的结构化 PII 检测，上下文中的命名实体识别模型。

### 使用 MinHash 进行去重

精确去重很简单：对每个文档进行哈希，移除重复项。但近似重复才是真正的问题。同一篇新闻文章的两个副本，周围环绕着略有不同的广告，就是近似重复。内容 95% 相同，但逐字节比较它们是不同的。

MinHash + 局部敏感哈希（LSH）可以高效解决这个问题。

```mermaid
graph LR
    A[文档] --> B[分片]
    B --> C[MinHash 签名]
    C --> D[LSH 桶]
    D --> E[候选对]
    E --> F[Jaccard 相似度]
    F --> G[去重后集合]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
    style G fill:#1a1a2e,stroke:#e94560,color:#fff
```

思路：

1. **分片（Shingling）：** 将每个文档转换为 n-gram 集合（例如，单词或字符的 5-gram）。"the quick brown fox" 用 3 词分片变成 {"the quick brown", "quick brown fox"}。

2. **MinHash：** 对于每个文档的分片集合，计算 k 个哈希值。每个哈希值是所有分片在不同哈希函数下的最小哈希。这创建了一个固定大小的"签名"，近似任意两个文档之间的 Jaccard 相似度。

3. **LSH：** 根据 MinHash 签名的条带将文档分组到桶中。同一桶中的文档是近似重复的候选。这避免了每对比较——你只需比较候选。

4. **验证：** 对于每对候选对，计算精确的 Jaccard 相似度。如果相似度超过阈值（通常为 0.8），则移除其中一个副本。

Llama 团队报告说，通过去重移除了大约 38% 的网络数据。这不是一个小数字。Common Crawl 中超过三分之一是重复或近似重复内容。

### 序列打包

你的模型期望固定长度的输入序列。你的文档长度可变。有些 50 个 token。有些 50,000 个 token。

朴素方法：将每个文档填充到最大序列长度。这在 contribute 学习毫无意义的填充 token 上浪费了大量计算。

更好的方法：将多个文档打包到一个序列中，用序列结束 token 分隔。一个 2048 token 的序列可能包含三个短文档，用 [EOS] token 连接。

```mermaid
graph TD
    subgraph Naive Packing[朴素打包]
        A1["文档 A (200 token)"] --> P1["[PAD] x 1848"]
        A2["文档 B (500 token)"] --> P2["[PAD] x 1548"]
        A3["文档 C (100 token)"] --> P3["[PAD] x 1948"]
    end

    subgraph Efficient Packing[高效打包]
        B1["文档 A (200) | 文档 B (500) | 文档 C (100) | 文档 D (400) | 文档 E (848)"]
    end

    style A1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style A2 fill:#1a1a2e,stroke:#e94560,color:#fff
    style A3 fill:#1a1a2e,stroke:#e94560,color:#fff
    style P1 fill:#333,stroke:#666,color:#999
    style P2 fill:#333,stroke:#666,color:#999
    style P3 fill:#333,stroke:#666,color:#999
    style B1 fill:#1a1a2e,stroke:#16c784,color:#fff
```

注意力掩码必须正确设置。文档 A 的 token 不应 attend 到同一打包序列中文档 B 的 token。这需要一个块对角线注意力掩码。

长文档在序列边界处被截断或分块。分割点很重要：在句子中间分割迫使模型看到不完整的 thoughts。一些流水线在可能的情况下将分割对齐到段落或句子边界。

### Chinchilla 扩展定律

对于固定的计算预算 C（以 FLOPs 衡量），最优模型大小 N 和数据集大小 D 遵循：

```
N_opt ~ C^0.5
D_opt ~ C^0.5
```

在实践中，这意味着你应该大致等量地扩展模型大小和数据集大小。一个参数多 10 倍的模型需要大约 10 倍的训练 token 才能达到相同的损失。

| 模型 | 参数 | 训练 Token | Chinchilla 最优？ |
|-------|-----------|----------------|-------------------|
| GPT-3 | 175B | 300B | 否（训练不足 3-4 倍） |
| Chinchilla | 70B | 1.4T | 是（按设计） |
| Llama 2 | 70B | 2T | 训练过度（有意为之） |
| Llama 3 | 70B | 15T | 严重训练过度 |

Llama 3 有意违反 Chinchilla 定律。Meta 发现，在更多数据上训练过度——远远超出计算最优比例——会产生更好的推理模型。额外的训练成本一次性支付，但更小的模型服务成本永远更低。这有时被称为"推理最优"扩展方法，自 2024 年以来已成为行业标准。

## 构建

### 第 1 步：文本清洗

剥离 HTML、规范化空格、移除非文本内容。我们将使用公共领域文本（Gutenberg 项目）作为我们的小型语料库。

```python
import re

def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def quality_filter(text, min_words=50, max_ratio_caps=0.3, max_ratio_special=0.1):
    words = text.split()
    if len(words) < min_words:
        return False
    caps_ratio = sum(1 for w in words if w.isupper()) / len(words)
    if caps_ratio > max_ratio_caps:
        return False
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if special_chars / max(len(text), 1) > max_ratio_special:
        return False
    return True
```

质量过滤器捕获 SEO 垃圾信息（全大写）、机器生成的噪声（高特殊字符比）和残缺页面（太短）。仅这三个检查就能从网络爬取中移除大量垃圾。

### 第 2 步：MinHash 去重

从零开始实现 MinHash。不需要外部库——只需要 `hashlib`。

```python
import hashlib
from collections import defaultdict

def get_shingles(text, k=5):
    words = text.lower().split()
    if len(words) < k:
        return set()
    return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}

def minhash_signature(shingles, num_hashes=128):
    signature = []
    for i in range(num_hashes):
        min_hash = float("inf")
        for shingle in shingles:
            h = int(hashlib.sha256(f"{i}:{shingle}".encode()).hexdigest(), 16)
            min_hash = min(min_hash, h)
        signature.append(min_hash)
    return signature

def lsh_buckets(signature, bands=16):
    rows_per_band = len(signature) // bands
    buckets = []
    for b in range(bands):
        start = b * rows_per_band
        band_data = tuple(signature[start:start + rows_per_band])
        bucket_hash = hashlib.md5(str(band_data).encode()).hexdigest()
        buckets.append((b, bucket_hash))
    return buckets

def deduplicate(documents, threshold=0.8, num_hashes=128, bands=16):
    signatures = []
    shingle_sets = []
    for doc in documents:
        shingles = get_shingles(doc)
        shingle_sets.append(shingles)
        signatures.append(minhash_signature(shingles, num_hashes))

    bucket_map = defaultdict(list)
    for doc_idx, sig in enumerate(signatures):
        for band_id, bucket_hash in lsh_buckets(sig, bands):
            bucket_map[(band_id, bucket_hash)].append(doc_idx)

    duplicate_pairs = set()
    for bucket_docs in bucket_map.values():
        if len(bucket_docs) < 2:
            continue
        for i in range(len(bucket_docs)):
            for j in range(i + 1, len(bucket_docs)):
                duplicate_pairs.add((bucket_docs[i], bucket_docs[j]))

    removed = set()
    for i, j in duplicate_pairs:
        if i in removed or j in removed:
            continue
        s1, s2 = shingle_sets[i], shingle_sets[j]
        if not s1 or not s2:
            continue
        jaccard = len(s1 & s2) / len(s1 | s2)
        if jaccard >= threshold:
            removed.add(j)

    return [doc for idx, doc in enumerate(documents) if idx not in removed], len(removed)
```

`num_hashes=128` 和 `bands=16` 参数控制精确度-召回率权衡。更多哈希给出更准确的相似度估计。更多条带增加召回率（捕获更多重复），代价是更多误报。这些值对典型网络文本效果很好。

### 第 3 步：分词并打包序列

获取清洗、去重后的文本，对其进行分词，并打包成固定长度序列用于训练。

```python
def tokenize_corpus(documents, tokenizer):
    all_tokens = []
    for doc in documents:
        tokens = tokenizer.encode(doc)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_id)
    return all_tokens

def pack_sequences(token_ids, seq_length, pad_id=0):
    sequences = []
    attention_masks = []
    for i in range(0, len(token_ids), seq_length):
        seq = token_ids[i:i + seq_length]
        mask = [1] * len(seq)
        if len(seq) < seq_length:
            pad_count = seq_length - len(seq)
            seq = seq + [pad_id] * pad_count
            mask = mask + [0] * pad_count
        sequences.append(seq)
        attention_masks.append(mask)
    return sequences, attention_masks
```

### 第 4 步：训练用 DataLoader

生成随机化的打包序列批次。这是训练循环所消费的。

```python
import random

class PreTrainingDataLoader:
    def __init__(self, sequences, attention_masks, batch_size, shuffle=True):
        self.sequences = sequences
        self.attention_masks = attention_masks
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.sequences) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.sequences)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            batch_seqs = [self.sequences[i] for i in batch_idx]
            batch_masks = [self.attention_masks[i] for i in batch_idx]
            yield batch_seqs, batch_masks
```

### 第 5 步：数据集统计

计算重要的数字：总 token 数、唯一 token 数、压缩比、文档长度分布。

```python
from collections import Counter

def compute_statistics(documents, token_ids, sequences, tokenizer_vocab_size):
    total_chars = sum(len(d) for d in documents)
    total_tokens = len(token_ids)
    unique_tokens = len(set(token_ids))
    compression_ratio = total_chars / total_tokens

    doc_lengths = [len(d.split()) for d in documents]
    avg_doc_length = sum(doc_lengths) / max(len(doc_lengths), 1)
    max_doc_length = max(doc_lengths) if doc_lengths else 0
    min_doc_length = min(doc_lengths) if doc_lengths else 0

    token_counts = Counter(token_ids)
    top_tokens = token_counts.most_common(10)

    non_pad_tokens = sum(sum(1 for t in seq if t != 0) for seq in sequences)
    total_positions = sum(len(seq) for seq in sequences)
    utilization = non_pad_tokens / max(total_positions, 1)

    stats = {
        "total_documents": len(documents),
        "total_characters": total_chars,
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "vocab_utilization": unique_tokens / tokenizer_vocab_size,
        "compression_ratio": compression_ratio,
        "avg_doc_length_words": avg_doc_length,
        "max_doc_length_words": max_doc_length,
        "min_doc_length_words": min_doc_length,
        "num_sequences": len(sequences),
        "sequence_utilization": utilization,
        "top_10_tokens": top_tokens,
    }
    return stats
```

压缩比告诉你分词器在这个语料库上的效率。英语文本通常压缩到每个 token 约 3-4 个字符。如果你看到每个 token 1.5 个字符，你的分词器拆分过于激进。如果看到 8+，它学到了非常特定领域的合并。

序列利用率告诉你打包序列中多少是真实数据，多少是填充。低于 90% 意味着你的打包效率低下——你在填充 token 上浪费计算。

## 使用

### 与 HuggingFace Datasets 对比

通过 HuggingFace 的 datasets 库加载相同语料库并比较流水线速度。

```python
from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

import time

start = time.time()
tokenized = ds.map(
    lambda x: tokenizer(x["text"], truncation=True, max_length=2048),
    batched=True,
    num_proc=4,
)
hf_time = time.time() - start
total_tokens = sum(len(t) for t in tokenized["input_ids"])
print(f"HuggingFace: {total_tokens:,} tokens in {hf_time:.2f}s ({total_tokens/hf_time:,.0f} tokens/sec)")
```

HuggingFace 流水线在幕后使用 Rust 分词器并跨 4 个核心并行处理。你的纯 Python 流水线会慢 10-50 倍。这个差距就是生产团队使用编译型分词器的原因。算法是一样的。实现语言的差异才是关键。

## 交付

本课产出一个用于验证和调试 LLM 训练流水线数据质量的提示词。参见 `outputs/prompt-data-quality-checker.md`。

## 练习

1. **简单：** 使用简单启发式方法（字符集分析）将语言检测添加到清洗流水线。过滤到仅英语文档并测量有多少文档被移除。
2. **中等：** 在 MinHash 近去重之外实现使用 SHA-256 哈希的精确去重。在网络抓取的语料库上比较每种方法捕获的重复数。
3. **困难：** 构建基于困惑度的质量过滤器。在 Wikipedia 文本上训练一个小二元语言模型，用困惑度对每个文档打分，并移除底部 20%。比较在过滤与未过滤数据上训练时的模型输出质量。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|----------------------|
| Common Crawl | "互联网" | 一个非营利组织，每月爬取网络——原始约 250TB，是大多数 LLM 训练数据的起点 |
| MinHash | "某种哈希技巧" | 使用固定大小签名估计集合间 Jaccard 相似度的技术——实现大规模近似重复检测 |
| LSH | "局部敏感哈希" | 将相似项分组到同一桶中的方法——将成对比较从 O(n^2) 减少到近线性 |
| 序列打包 | "连接文档" | 将多个文档放入固定长度序列并配合正确的注意力掩码——消除填充浪费 |
| Chinchilla 扩展 | "用更多数据训练" | 对于固定计算预算，最优性能要求模型大小和训练 token 大致等量扩展 |
| Fertility | "每个词的 token 数" | 每个词的平均 token 数——GPT-4 中英语为 1.3，非拉丁字母文字更高 |
| 数据混合 | "选择训练数据" | 代码 vs 文本 vs 数学 vs 多语言数据的比例——没有公式，需要实验 |
| 困惑度过滤器 | "质量打分" | 使用小语言模型对文档打分——高困惑度意味着文本不像干净的参考数据 |
| 去重 | "移除副本" | 消除精确和近似重复文档——通常移除原始网络数据的 30-40% |
| 注意力掩码 | "哪些 token 要关注" | 一个二元掩码，防止打包序列中文档边界之间的注意力 |

## 延伸阅读

- [Hoffmann et al., 2022 -- Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556) —— 这篇论文改变了我们对数据规模的思考方式
- [Penedo et al., 2023 -- The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116) —— 如何将 Common Crawl 过滤为高质量数据
- [Touvron et al., 2023 -- Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) —— Llama 2 的数据流水线细节
- [Lee et al., 2022 -- Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) —— 为什么去重比你想象的更重要
- [Broder, 1997 -- On the Resemblance and Containment of Documents](https://ieeexplore.ieee.org/document/666900) —— 原始 MinHash 论文
- [Meta, 2024 -- Llama 3 Technical Report](https://arxiv.org/abs/2407.21783) —— 15.6T token、数据混合比例、过滤流水线