# 音频分类 — 从 MFCC 上的 k-NN 到 AST 和 BEATs

> 从"狗叫 vs 警笛"到"这是哪种语言"，都是音频分类。特征是 Mel。架构每个十年演进一次。评估指标是 AUC、F1 和每类召回率。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 6 · 02（频谱图 & Mel）、阶段 3 · 06（CNN）、阶段 5 · 08（文本用 CNN & RNN）
**时间：** 约 75 分钟

## 问题

你得到一段 10 秒的片段。你想知道："这是什么？"城市声音（警笛、钻机、狗）、语音命令（是/否/停）、语言识别（英/西/阿）、说话人情绪（愤怒/中性），还是环境声音（室内/室外、人声嘈杂）。这些都是*音频分类*，而 2026 年的基线架构已经成熟：log-mel → CNN 或 Transformer → softmax。

核心难点不在网络，而在数据。音频数据集有严重的类别不平衡、强域偏移（干净 vs 噪声）和标签噪声（谁决定的"城市嘈杂" vs "餐厅噪声"？）。80% 的问题在于整理、增强和评估，而不是把 CNN 换成 Transformer。

## 概念

![音频分类阶梯：从 MFCC 上的 k-NN 到 AST 再到 BEATs](../assets/audio-classification.svg)

**MFCC 上的 k-NN（1990 年代基线）。** 将每段 MFCC 展平，计算与标记库的余弦相似度，返回前 K 个的多数投票。在干净的小数据集上出人意料地强（Speech Commands、ESC-50）。无需 GPU 运行。

**对数 Mel 上的 2D CNN（2015-2019）。** 将 `(T, n_mels)` 对数 Mel 视为图像。应用 ResNet-18 或 VGG 风格。在时间轴上做全局平均池化。Softmax 输出类别。仍是 2026 年大多数 Kaggle 竞赛的基线。

**音频频谱图 Transformer，AST（2021-2024）。** 将对数 Mel 分块（例如 16×16 patch），添加位置 embedding，输入 ViT。在监督学习 AudioSet（mAP 0.485）上达到 SOTA。

**BEATs 和 WavLM-base（2024-2026）。** 在数百万小时的自监督预训练。用 1-10% 的监督数据微调目标任务。2026 年这是非语音音频的默认起点。BEATs-iter3 在 AudioSet 上以 1/4 的计算量超越 AST 1-2 mAP。

**Whisper 编码器作为冻结主干（2024）。** 取 Whisper 的编码器，丢弃解码器，接一个线性分类器。在语言识别和简单事件分类上接近 SOTA，无需音频增强。"免费午餐"基线。

### 类别不平衡才是真正的挑战

ESC-50：50 类，每类 40 片段——平衡、简单。UrbanSound8K：10 类，10:1 不平衡。AudioSet：632 类，long tail 100,000:1。有效的技术：

- 训练时平衡采样（评估时不用）。
- Mixup：线性插值两个片段（及其标签）作为增强。
- SpecAugment：随机遮蔽时间和频率带。简单；关键。

### 评估

- 多分类互斥（Speech Commands）：top-1 准确率，top-5 准确率。
- 多分类多标签（AudioSet、UrbanSound 风格）：平均精度（mAP）。
- 严重不平衡：每类召回率 + macro F1。

2026 年你需要知道的数字：

| 基准 | 基线 | SOTA 2026 | 来源 |
|-----------|----------|-----------|--------|
| ESC-50 | 82%（AST） | 97.0%（BEATs-iter3） | BEATs 论文（2024） |
| AudioSet mAP | 0.485（AST） | 0.548（BEATs-iter3） | HEAR 排行榜 2026 |
| Speech Commands v2 | 98%（CNN） | 99.0%（Audio-MAE） | HEAR v2 结果 |

## 动手实现

### 第 1 步：特征化

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### 第 2 步：固定长度摘要

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

简单但强：沿时间取 mean + variance，为 13 系数 MFCC 产生 26 维固定 embedding。运行极快。直到 2017 年还能打败 ESC-50 上的 SOTA 神经网络基线。

### 第 3 步：k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### 第 4 步：升级到对数 Mel 上的 CNN

用 PyTorch：

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

300 万参数。在单张 RTX 4090 上训练 ESC-50 约 10 分钟。准确率 80%+。

### 第 5 步：2026 年默认——微调 BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

BEATs 用 `microsoft/BEATs-base`，通过 `beats` 库；transformers API 形状相同。

## 实际使用

2026 年的技术栈：

| 情况 | 从这里开始 |
|-----------|-----------|
| 小数据集（<1000 片段） | MFCC 均值的 k-NN（你的基线）+ 音频增强 |
| 中等数据集（1K–100K） | BEATs 或 AST 微调 |
| 大数据集（>100K） | 从头训练或微调 Whisper 编码器 |
| 实时、边缘 | 40-MFCC CNN，量化为 int8（KWS 风格） |
| 多标签（AudioSet） | BEATs-iter3 + BCE loss + mixup + SpecAugment |
| 语言识别 | MMS-LID、SpeechBrain VoxLingua107 基线 |

决策规则：**从冻结主干开始，而非从头模型**。微调 BEATs head 能在数小时内达到 SOTA 的 95%，而非数周。

## 交付

保存为 `outputs/skill-classifier-designer.md`。为给定的音频分类任务选择架构、增强、类别平衡策略和评估指标。

## 练习

1. **简单。** 运行 `code/main.py`。它在 4 类合成数据集（不同音高的纯音）上训练 k-NN MFCC 基线。报告混淆矩阵。
2. **中等。** 将 `summarize` 替换为 [mean, var, skew, kurtosis]。4 矩池化在同一合成数据集上是否优于 mean+var？
3. **困难。** 用 `torchaudio` 在 ESC-50 fold 1 上训练 2D CNN。报告 5 折交叉验证准确率。添加 SpecAugment（时间遮蔽 = 20，频率遮蔽 = 10）并报告增量。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|-----------------------|
| AudioSet | 音频的 ImageNet | Google 的 200 万片段、632 类弱标注 YouTube 数据集。 |
| ESC-50 | 小型分类基准 | 50 类 × 40 片段环境声音。 |
| AST | 音频频谱图 Transformer | 在对数 Mel patch 上的 ViT；2021 SOTA。 |
| BEATs | 自监督音频 | Microsoft 模型，iter3 在 2026 年 AudioSet 领先。 |
| Mixup | 配对增强 | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`。 |
| SpecAugment | 遮蔽增强 | 将频谱图的随机时间和频率带置零。 |
| mAP | 主要多标签指标 | 跨类别和阈值的平均精度。 |

## 延伸阅读

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — 2021–2024 年的创纪录架构。
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — 2024+ 年的默认选择。
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — 主导的音频增强。
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — 50 类基准数据集。
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — 632 类 YouTube 分类法；仍是黄金标准。