# 音频基础 — 波形、采样与傅里叶变换

> 波形是原始信号。频谱图是表征形式。Mel 特征是 ML 友好形态。现代 ASR 和 TTS 流水线都沿着这把梯子往上走，而第一级就是理解采样与傅里叶。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 1 · 06（向量与矩阵）、阶段 1 · 14（概率分布）
**时间：** 约 45 分钟

## 问题

麦克风产生的是压力-时间信号。神经网络消耗的是张量。两者之间是一系列约定，违反它们就会产生静默 bug：模型训练正常但 WER 翻倍，TTS 发出嘶嘶声，语音克隆系统记住的是麦克风而非说话人。

语音系统的所有 bug 都能追溯到三个问题之一：

1. 数据的采样率是多少，模型期望的又是多少？
2. 信号是否发生了混叠？
3. 你是在原始采样上操作还是在频率表示上操作？

把这三个问题搞对了，阶段 6 的其余内容就 tractable。搞错了，就连 Whisper-Large-v4 也会输出垃圾。

## 概念

![波形、采样、DFT 与频率 bins 可视化](../assets/audio-fundamentals.svg)

**波形。** 一个一维浮点数数组，范围 `[-1.0, 1.0]`。以采样编号为索引。转换为秒需除以采样率：`t = n / sr`。一段 10 秒、16 kHz 的片段是 160,000 个浮点数的数组。

**采样率（sr）。** 每秒多少个采样。2026 年的常用采样率：

| 采样率 | 用途 |
|--------|------|
| 8 kHz | 电话、传统 VOIP。奈奎斯特 4 kHz 会抹掉辅音。ASR 应避免。 |
| 16 kHz | ASR 标准。Whisper、Parakeet、SeamlessM4T v2 都输入 16 kHz。 |
| 22.05 kHz | 旧模型 TTS 声码器训练。 |
| 24 kHz | 现代 TTS（Kokoro、F5-TTS、xTTS v2）。 |
| 44.1 kHz | CD 音频、音乐。 |
| 48 kHz | 电影、专业音频、高保真 TTS（VALL-E 2、NaturalSpeech 3）。 |

**奈奎斯特-香农。** 采样率 `sr` 能无歧义地表示最高 `sr/2` 的频率。`sr/2` 边界即*奈奎斯特频率*。高于奈奎斯特的能量会被*混叠*——折叠到低频——从而破坏信号。下采样前务必做低通滤波。

**位深。** 16 位 PCM（有符号 int16，范围 ±32,767）是通用交换格式。24 位用于音乐，32 位浮点用于内部 DSP。`soundfile` 等库读取 int16 但暴露为 `[-1, 1]` 的 float32 数组。

**傅里叶变换。** 任何有限信号都是不同频率正弦波的叠加。离散傅里叶变换（DFT）对 `N` 个采样计算 `N` 个复系数——每频率 bin 一个。频率 bin `k` 对应频率 `k · sr / N` Hz。幅值是该频率的能量，相角是相位。

**FFT。** 快速傅里叶变换：当 `N` 是 2 的幂时，DFT 的 `O(N log N)` 算法。每个音频库底层都使用 FFT。16 kHz 下 1024 采样的 FFT 给出 512 个可用频率 bins，覆盖 0–8 kHz，分辨率 15.6 Hz。

**分帧 + 加窗。** 我们不对整个片段做 FFT，而是将它切成重叠的*帧*（通常 25 ms，步长 10 ms），每帧乘以窗函数（Hann 或 Hamming）以消除边缘不连续，然后对每帧做 FFT。这就是短时傅里叶变换（STFT）。第 02 课从这里继续。

## 动手实现

### 第 1 步：读取片段并绘制波形

`code/main.py` 只用标准库 `wave` 模块以保持演示无依赖。生产环境你会用 `soundfile` 或 `torchaudio.load`（都返回 `(waveform, sr)` 元组）：

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### 第 2 步：从第一性原理合成正弦波

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

一个 440 Hz 正弦波（标准音 A）在 16 kHz 下持续 1 秒是 16,000 个浮点数。用 `wave.open(..., "wb")` 和 16 位 PCM 编码写入。

### 第 3 步：手工计算 DFT

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)`——`N=256` 时用来验证正确性尚可，实际音频毫无用处。真代码调用 `numpy.fft.rfft` 或 `torch.fft.rfft`。

### 第 4 步：找主导频率

幅值峰值索引 `k_star` 对应频率 `k_star * sr / N`。在 440 Hz 正弦波上运行应返回 bin `440 * N / sr` 处的峰值。

### 第 5 步：演示混叠

用 10 kHz 采样 7 kHz 正弦波（奈奎斯特 = 5 kHz）。7 kHz 音调高于奈奎斯特，折叠到 `10 − 7 = 3 kHz`。FFT 峰值出现在 3 kHz。这是经典的混叠演示，也是每个 DAC/ADC 都附带砖墙式低通滤波器的原因。

## 实际使用

你 2026 年真正会发货的技术栈：

| 任务 | 库 | 为什么 |
|------|------|--------|
| 读写 WAV/FLAC/OGG | `soundfile`（libsndfile 封装） | 最快、最稳定，返回 float32。 |
| 重采样 | `torchaudio.transforms.Resample` 或 `librosa.resample` | 内置正确的抗混叠。 |
| STFT / Mel | `torchaudio` 或 `librosa` | GPU 友好；PyTorch 生态。 |
| 实时流 | `sounddevice` 或 `pyaudio` | 跨平台 PortAudio 绑定。 |
| 检查文件 | `ffprobe` 或 `soxi` | CLI，快速，报告 sr/通道/编解码器。 |

决策规则：**先匹配采样率，再匹配其他任何东西**。Whisper 期望 16 kHz 单声道 float32。传入 44.1 kHz 立体声，你会得到看起来像模型 bug 的垃圾。

## 交付

保存为 `outputs/skill-audio-loader.md`。这个 skill 帮助检查音频输入是否匹配下游模型的预期，并在不匹配时正确重采样。

## 练习

1. **简单。** 合成 1 秒 220 Hz + 440 Hz + 880 Hz 混音，16 kHz。做 DFT。确认在预期 bins 处有三个峰值。
2. **中等。** 用 48 kHz 录制自己声音的 3 秒 WAV。用 `torchaudio.transforms.Resample`（带抗混叠）下采样到 16 kHz，再用朴素抽取（每三个采样取一个）。对两者做 FFT。混叠出现在哪里？
3. **困难。** 只用 `math` 和第 3 步的 DFT 从零构建 STFT。帧长 400，步长 160，Hann 窗。用 `matplotlib.pyplot.imshow` 绘制幅值。这就是第 02 课的频谱图。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|-----------------------|
| 采样率 | 每秒多少个采样 | ADC 测量信号的频率（Hz）。 |
| 奈奎斯特 | 能表示的最高频率 | `sr/2`；高于此的能量会混叠回来。 |
| 位深 | 每个采样的分辨率 | `int16` = 65,536 个等级；`float32` = 24 位精度，范围 `[-1, 1]`。 |
| DFT | 序列的傅里叶变换 | `N` 个采样 → `N` 个复频率系数。 |
| FFT | 快速 DFT | `O(N log N)` 算法，要求 `N` 为 2 的幂。 |
| Bin | 频率柱 | `k · sr / N` Hz；分辨率 = `sr / N`。 |
| STFT | 频谱图的底层 | 分帧 + 加窗 FFT，按时间排列。 |
| 混叠 | 奇怪的频率鬼影 | 高于奈奎斯特的能量镜像到低 bins。 |

## 延伸阅读

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — 采样定理背后的论文。
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — 免费、权威的 DSP 教科书。
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html) — 带代码的实用教程。
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) — 真实世界音频不是干净正弦波的原因。
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — 10 分钟搞懂频率 bin 直觉。