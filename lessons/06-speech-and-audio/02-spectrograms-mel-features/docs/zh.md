# 频谱图、Mel 尺度与音频特征

> 神经网络不适合直接消费原始波形。频谱图可以。Mel 频谱图更适合。2026 年的每一个 ASR、TTS 和音频分类器都因这一预处理选择而兴衰。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 6 · 01（音频基础）
**时间：** 约 45 分钟

## 问题

拿一段 10 秒 16 kHz 的片段。这是 160,000 个浮点数，范围 `[-1, 1]`，与"狗在叫"或"单词 cat"这个标签几乎完美地无关。原始波形里有这些信息，但形式让模型难以提取。两个相同的音素相隔 100 ms 说出来，原始采样会完全不同。

频谱图解决了这个问题。它把人类感知忽略的时间细节（微秒级抖动）压缩掉，保留感知关注的部分（哪些频率有能量，随时间窗口约 10–25 ms 的变化）。

Mel 频谱图更进一步。人类对音高的感知是对数的：100 Hz 到 200 Hz 听起来与 1000 Hz 到 2000 Hz "距离相同"。Mel 尺度对频率轴做了弯曲以匹配这种感知。从 2010 年到 2026 年，Mel 尺度频谱图是语音 ML 中最重要的特征。

## 概念

![波形 → STFT → Mel 频谱图 → MFCC 阶梯](../assets/mel-features.svg)

**STFT（短时傅里叶变换）。** 将波形切成重叠帧（典型：25 ms 窗，10 ms 步长 = 16 kHz 下 400 采样 / 160 采样）。每帧乘以窗函数（Hann 是默认值；Hamming 略有不同的权衡）。对每帧做 FFT。将幅值谱堆叠成形状为 `(n_frames, n_freq_bins)` 的矩阵。这就是你的频谱图。

**对数幅值。** 原始幅值跨越 5-6 个数量级。取 `log(|X| + 1e-6)` 或 `20 * log10(|X|)` 来压缩动态范围。每个生产流水线都使用对数幅值，而非原始幅值。

**Mel 尺度。** 频率 `f`（Hz）到 Mel `m` 的映射：`m = 2595 * log10(1 + f / 700)`。该映射在 1 kHz 以下大致线性，在以上大致对数。覆盖 0–8 kHz 的 80 个 Mel bins 是标准 ASR 输入。

**Mel 滤波器组。** 一组在 Mel 尺度上等距分布的三角滤波器。每个滤波器是相邻 FFT bins 的加权和。用滤波器组矩阵乘以 STFT 幅值，一次 matmul 就得到 Mel 频谱图。

**对数 Mel 频谱图。** `log(mel_spec + 1e-10)`。Whisper 的输入。Parakeet 的输入。SeamlessM4T 的输入。2026 年通用的音频前端。

**MFCC。** 取得对数 Mel 频谱图，应用 DCT（II 型），保留前 13 个系数。去相关特征并进一步压缩。直到 2015 年 CNN/Transformer 在原始对数 Mel 上赶上来之前的主导特征。仍在说话人识别中使用（x-vectors、ECAPA）。

**分辨率权衡。** FFT 越大 = 频率分辨率越好但时间分辨率越差。25 ms / 10 ms 是音频-ML 默认；50 ms / 12.5 ms 用于音乐；5 ms / 2 ms 用于瞬态检测（鼓点、爆破音）。

## 动手实现

### 第 1 步：分帧

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

10 秒 16 kHz 片段，`frame_len=400, hop=160` 产生 998 帧。

### 第 2 步：Hann 窗

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

FFT 前逐元素相乘。消除了在非零端点截断导致频谱泄漏。

### 第 3 步：STFT 幅值

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

生产环境用 `torch.stft` 或 `librosa.stft`（FFT 驱动、矢量化）。这里的循环是教学用的；它运行在 `code/main.py` 的短片段上。

### 第 4 步：Mel 滤波器组

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

`n_fft=400` 下覆盖 0–8 kHz 的 80 个 mels 给出 `(80, 201)` 矩阵。用转置乘以 `(n_frames, 201)` STFT 幅值得到 `(n_frames, 80)` Mel 频谱图。

### 第 5 步：对数 Mel

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

常见替代：`librosa.power_to_db`（参考归一化 dB）、`10 * log10(power + eps)`。Whisper 使用更复杂的裁剪 + 归一化例程（见 Whisper 的 `log_mel_spectrogram`）。

### 第 6 步：MFCC

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

对每帧对数 Mel 做 DCT，保留前 13 个系数。这就是你的 MFCC 矩阵。第一个系数通常被丢弃（它编码整体能量）。

## 实际使用

2026 年的技术栈：

| 任务 | 特征 |
|------|----------|
| ASR（Whisper、Parakeet、SeamlessM4T） | 80 个对数 Mel，10 ms 步长，25 ms 窗 |
| TTS 声学模型（VITS、F5-TTS、Kokoro） | 80 个 Mel，5–12 ms 步长以实现精细时间控制 |
| 音频分类（AST、PANNs、BEATs） | 128 个对数 Mel，10 ms 步长 |
| 说话人 embedding（ECAPA-TDNN、WavLM） | 80 个对数 Mel 或原始波形 SSL |
| 音乐（MusicGen、Stable Audio 2） | EnCodec 离散 token（非 Mel） |
| 关键词检测 | Tiny 设备用 40 个 MFCC |

经验法则：**如果你不是在处理音乐，从 80 个对数 Mel 开始**。任何偏离都需要自证其合理性。

## 2026 年仍会发货的陷阱

- **Mel 数量不匹配。** 训练用 80 个 mels，推理用 128 个。静默失败。在两端都记录特征形状。
- **上游采样率不匹配。** 在 22.05 kHz 计算的 Mel 与 16 kHz 的看起来不同。在特征化之前先修正 SR。
- **dB 与对数。** Whisper 期望对数 Mel，不是 dB-Mel。一些 HF 流水线会自动检测；你的自定义代码不会。
- **归一化漂移。** 训练时逐 utterance 归一化，推理时全局归一化。生产 bug，会使 WER 加倍。
- **填充泄漏。** 对片段末尾做零填充会在尾部帧产生平坦频谱。对称填充或复制。

## 交付

保存为 `outputs/skill-feature-extractor.md`。该 skill 为给定的模型目标选择特征类型、Mel 数量、帧/步长和归一化方式。

## 练习

1. **简单。** 运行 `code/main.py`。它合成一个扫频 chirp（频率从 200 → 4000 Hz），打印每帧的 argmax Mel bin。可视化（可选）并确认它匹配扫频。
2. **中等。** 用 `n_mels` 在 `{40, 80, 128}` 和 `frame_len` 在 `{200, 400, 800}` 中重新运行。测量时间轴上的尖峰带宽。哪个组合对 chirp 分辨率最好？
3. **困难。** 实现 `power_to_db`，并用以下三种特征在 AudioMNIST 上比较一个小型 CNN 分类器的 ASR 准确率：(a) 原始对数 Mel，(b) `ref=max` 的 dB-Mel，(c) MFCC-13 + delta + delta-delta。报告 top-1 准确率。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|-----------------------|
| 帧 | 一切片 | 一次 FFT 输入的 25 ms 波形块。 |
| 步长 | 跨度 | 连续帧之间的采样数；10 ms 是 ASR 默认。 |
| 窗 | Hann/Hamming 那个东西 | 点态乘法器，将帧边缘渐变到零。 |
| STFT | 频谱图生成器 | 分帧 + 加窗 FFT；产生时间 × 频率矩阵。 |
| Mel | 弯曲的频率 | 对数感知尺度；`m = 2595·log10(1 + f/700)`。 |
| 滤波器组 | 那个矩阵 | 将 STFT 投影到 Mel bins 的三角滤波器。 |
| 对数 Mel | Whisper 的输入 | `log(mel_spec + eps)`；2026 年标准化。 |
| MFCC | 老派特征 | 对数 Mel 的 DCT；13 个系数，去相关。 |

## 延伸阅读

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — MFCC 论文。
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — 原始 Mel 尺度。
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — 阅读参考实现。
- [librosa feature extraction docs](https://librosa.org/doc/main/feature.html) — `mfcc`、`melspectrogram` 和 hop/窗的参考。
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — Parakeet + Canary 模型的生产规模流水线。