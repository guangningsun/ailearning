# 语音识别 (ASR) — CTC、RNN-T 与注意力机制

> 语音识别就是在每个时间步做音频分类，再用序列模型把它们粘起来——这个模型懂英文，也懂静音。CTC、RNN-T 和注意力是三种实现方式。选一种，然后理解它为什么有效。

**类型：** 动手构建
**语言：** Python
**前置条件：** 阶段 6 · 02（频谱图与 Mel 特征）、阶段 5 · 08（用于文本的 CNN 与 RNN）、阶段 5 · 10（注意力机制）
**时间：** 约 45 分钟

## 问题

你有 10 秒的 16 kHz 音频片段。想要得到一句话："turn on the kitchen lights"。挑战在于结构层面：音频帧与字符之间不是一对一的关系。"okay" 这个词可能占用 200 毫秒，也可能占用 1200 毫秒。静音打断话语，某些音素比其他音素更长。输出 token 的数量事先未知。

三种建模方法可以解决这个问题：

1. **CTC（连接时序分类）。** 每帧输出 token 概率分布，外加一个特殊的*空 token*。解码时合并重复项并去除空 token。非自回归，速度快。被 wav2vec 2.0、MMS 采用。
2. **RNN-T（循环神经网络转录机）。** 联合网络根据编码器帧和先前 token 预测下一个 token。可流式处理。被 Google 端侧 ASR、NVIDIA Parakeet 采用。
3. **注意力编码器-解码器。** 编码器将音频压缩为隐藏状态，解码器通过交叉注意力自回归生成 token。被 Whisper、SeamlessM4T 采用。

到 2026 年，LibriSpeech test-clean 上的 SOTA WER 为 1.4%（Parakeet-TDT-1.1B，NVIDIA）和 1.58%（Whisper-Large-v3-turbo）。差异很小，但部署差异巨大。

## 概念

![三种 ASR 建模方法：CTC、RNN-T、注意力编码器-解码器](../assets/asr-formulations.svg)

**CTC 直觉。** 编码器输出 `T` 个帧级分布，涵盖 `V+1` 个 token（V 个字符 + 空 token）。对于目标字符串 `y`（长度 `U < T`），任何能折叠为 `y` 的帧对齐方式都计入。CTC 损失对所有此类对齐方式求和。推理：逐帧取 argmax，合并重复项，去除空 token。

优势：非自回归、可流式处理、零前瞻。缺点：*条件独立假设*——每帧预测独立于其他帧，因此没有内部语言模型。可通过外部 LM 结合束搜索或浅层融合来修复。

**RNN-T 直觉。** 增加一个*预测器*网络来嵌入 token 历史，以及一个*合并器*将预测器状态与编码器帧结合为 `V+1`（`+1` 是空 / 无发射）的联合分布。显式建模了 CTC 所忽略的条件依赖。可流式处理，因为每一步只依赖过去的帧和过去的 token。

优势：可流式 + 内部 LM。缺点：训练更复杂，内存消耗大（3D 损失网格）；RNN-T 损失核是独立的一整个库类别。

**注意力编码器-解码器。** 编码器（6-32 层 transformer）对 log-mel 帧进行处理。解码器（6-32 层 transformer）交叉关注编码器输出来自回归生成 token。没有对齐约束——注意力可以看向音频中的任何位置。非流式处理，除非限制注意力（分块 Whisper-Streaming，2024）。

优势：离线 ASR 质量最高，使用标准 seq2seq 工具容易训练。缺点：自回归延迟与输出长度成正比；不经过工程改造无法流式处理。

### WER：那个数字

**词错误率** = `(S + D + I) / N`，其中 S=替换，D=删除，I=插入，N=参考词数。在词级别匹配 Levenshtein 编辑距离。越低越好。WER 超过 20% 通常无法使用；低于 5% 对于朗读语音已达到人类水平。2026 年标准基准测试数据：

| 模型 | LibriSpeech test-clean | LibriSpeech test-other | 规模 |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B 参数 |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

以上全部是编码器-解码器或 RNN-T 架构。纯 CTC 系统（wav2vec 2.0）在 test-clean 上约 1.8–2.1%。

## 动手构建

### 第 1 步：贪心 CTC 解码

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: 每帧概率向量的列表
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

两条规则：合并连续重复项，去除空 token。示例：`a a _ _ a b b _ c` → `a a b c`。

### 第 2 步：束搜索 CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

生产环境使用带 LM 融合的前缀树束搜索；这是概念骨架。

### 第 3 步：WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### 第 4 步：针对 Whisper 的推理

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

2026 年最强的通用 ASR 一行搞定。在 24 GB GPU 上以约 20× 实时速度运行。

### 第 5 步：使用 Parakeet 或 wav2vec 2.0 流式处理

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

流式 ASR 需要分块编码器注意力和携带状态；使用支持此功能的库（Parakeet 用 NeMo，`transformers` pipeline 用 `chunk_length_s`）。

## 使用

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| 英语、离线、最高质量 | Whisper-large-v3-turbo |
| 多语言、鲁棒 | SeamlessM4T v2 |
| 流式处理、低延迟 | Parakeet-TDT-1.1B 或 Riva |
| 边缘、移动端、<500ms 延迟 | Whisper-Tiny 量化版或 Moonshine（2024） |
| 长音频 | 带 VAD 分块的 Whisper（WhisperX） |
| 领域特定（医疗、法律） | 微调 wav2vec 2.0 + 领域 LM 融合 |

## 2026 年仍然在发货的坑

- **没有 VAD。** 在静音上运行 Whisper 会产生幻觉（"Thanks for watching!"）。务必用 VAD 门控。
- **字符 vs 词 vs 子词 WER。** 报告规范化后的词级 WER（转小写、去除标点）。
- **语言 ID 漂移。** Whisper 的自动 LID 会把噪声片段误路由到日语或威尔士语；当你确定语言时强制使用 `language="en"`。
- **长片段没有分块。** Whisper 有 30 秒窗口。任何更长的内容使用 `chunk_length_s=30, stride=5`。

## 交付

保存为 `outputs/skill-asr-picker.md`。为给定的部署目标选择模型、解码策略、分块方式和 LM 融合。

## 练习

1. **简单。** 运行 `code/main.py`。它对手工制作的 CTC 输出进行贪心解码，并与参考计算 WER。
2. **中等。** 正确实现前缀树束搜索（考虑空合并规则）。在 10 个示例的综合数据集上与贪心方法比较。
3. **困难。** 在 [LibriSpeech test-clean](https://www.openslr.org/12) 上使用 `whisper-large-v3-turbo`。计算前 100 个句子的 WER。与已发布数字比较。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| CTC | 空 token 损失 | 所有帧到 token 对齐的边缘化；非自回归。 |
| RNN-T | 流式损失 | CTC + 下一 token 预测器；处理词序。 |
| 注意力编码器-解码器 | Whisper 风格 | 编码器 + 交叉注意力解码器；离线质量最佳。 |
| WER | 你报告的数字 | 词级别的 `(S+D+I)/N`。 |
| 空 token | 空无 | CTC 中的特殊 token，表示"本帧无发射"。 |
| LM 融合 | 外部语言模型 | 束搜索期间添加加权的 LM log 概率。 |
| VAD | 静音门 | 语音活动检测器；裁剪非语音部分。 |

## 延伸阅读

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — CTC 原始论文。
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — RNN-T 原始论文。
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 2022 年经典论文；2024 年 v3-turbo 扩展。
- [NVIDIA NeMo — Parakeet-TDT 卡片](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — 2026 年 Open ASR 排行榜领先者。
- [Hugging Face — Open ASR 排行榜](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 25+ 模型的实时基准测试。