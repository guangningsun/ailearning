# 文本转语音 (TTS) —— 从 Tacotron 到 F5 和 Kokoro

> ASR 把语音转成文本；TTS 把文本转成语音。2026 年的技术栈分三层：文本 → token，token → mel，mel → 波形。每层都有一个跑在笔记本上的默认模型。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 6 · 02（频谱图与 Mel）、阶段 5 · 09（Seq2Seq）、阶段 7 · 05（完整 Transformer）
**时间：** 约 75 分钟

## 问题

你有一段文字："Please remind me to water the plants at 6 pm."需要生成一段 3 秒的自然音频，正确处理韵律（停顿、重音），"plants" 发音正确，且在 CPU 上跑一个实时语音助手时延迟低于 300 ms。此外还需要切换音色、处理双语切换输入（"remind me at 6 pm, daijoubu?"），以及正确读出人名。

现代 TTS 流程如下：

1. **文本前端。** 规范化文本（日期、数字、邮箱），转换为音素或子词 token，预测韵律特征。
2. **声学模型。** 文本 → mel 频谱图。Tacotron 2（2017）、FastSpeech 2（2020）、VITS（2021）、F5-TTS（2024）、Kokoro（2024）。
3. **声码器。** mel → 波形。WaveNet（2016）、WaveRNN、HiFi-GAN（2020）、BigVGAN（2022）、2024+ 的神经编解码器声码器。

到 2026 年，声学模型 + 声码器的边界已经因端到端扩散和流匹配模型而模糊。但三层 mental model 对调试仍然适用。

## 概念

![Tacotron、FastSpeech、VITS、F5/Kokoro 对比](../assets/tts.svg)

**Tacotron 2（2017）。** Seq2seq：字符 embedding → BiLSTM 编码器 → 位置敏感注意力 → 自回归 LSTM 解码器逐帧输出 mel。慢（AR），长文本容易不稳定。至今仍作为 baseline 被引用。

**FastSpeech 2（2020）。** 非自回归。时长预测器输出每个音素对应多少帧 mel。1 次前向，10 倍快于 Tacotron。自然度略有损失（单调对齐），但部署广泛。

**VITS（2021）。** 编码器 + 流式时长预测 + HiFi-GAN 声码器端到端联合训练，带变分推断。高质量，单模型。2022–2024 年开源 TTS 主导方案。变体：YourTTS（多说话人零样本）、XTTS v2（2024，Coqui）。

**F5-TTS（2024）。** 基于流匹配的扩散 Transformer。自然韵律，5 秒参考音频零样本语音克隆。占据 2026 年开源 TTS 排行榜榜首。3.35 亿参数。

**Kokoro（2024）。** 小模型（8200 万），可 CPU 运行，实时场景英语 TTS 最佳。闭词汇英语专用，Apache-2.0。

**OpenAI TTS-1-HD、ElevenLabs v2.5、Google Chirp-3。** 商业领域前沿。ElevenLabs v2.5 的情感标签（"[whispered]"、"[laughing]"）和角色音色主导 2026 年有声书制作。

### 声码器演进

| 时代 | 声码器 | 延迟 | 质量 |
|------|--------|------|------|
| 2016 | WaveNet | 仅离线 | 发布时 SOTA |
| 2018 | WaveRNN | 约实时 | 良好 |
| 2020 | HiFi-GAN | 100 倍实时 | 接近人类 |
| 2022 | BigVGAN | 50 倍实时 | 跨说话人/语言泛化 |
| 2024 | SNAC、DAC（神经编解码器） | 与 AR 模型集成 | 离散 token，位高效 |

到 2026 年，大多数"TTS"模型已是文本到波形的端到端；mel 频谱图成为内部表示。

### 评估方法

- **MOS（平均意见分）。** 1–5 分制，众包打分。仍是黄金标准，但速度很慢。
- **CMOS（比较 MOS）。** A vs B 偏好。每条标注置信区间更紧。
- **UTMOS、DNSMOS。** 无参考神经 MOS 预测器。用于排行榜。
- **通过 ASR 的 CER（字符错误率）。** 将 TTS 输出过 Whisper，计算与输入文本的 CER。反映可懂度。
- **SECS（说话人 embedding 余弦相似度）。** 语音克隆质量指标。

2026 年 LibriTTS test-clean 上的数字：

| 模型 | UTMOS | CER（过 Whisper） | 参数量 |
|------|-------|--------------------|--------|
| Ground truth | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 3.35 亿 |
| XTTS v2 | 3.81 | 3.5% | 4.70 亿 |
| VITS | 3.62 | 3.1% | 2500 万 |
| Kokoro v0.19 | 3.87 | 1.8% | 8200 万 |
| Parler-TTS Large | 3.76 | 2.8% | 23 亿 |

## 构建

### 第 1 步：音素化输入

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

音素是通用桥梁。不要向 VITS 级别以下的模型输入原始文本。

### 第 2 步：运行 Kokoro（2026 CPU 默认方案）

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = 美国英语
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 张量, sr=24000
```

离线运行，单文件，8200 万参数。

### 第 3 步：用 F5-TTS 语音克隆运行

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

传入 5 秒参考音频片段及其转写稿；F5 克隆韵律和音色。

### 第 4 步：从零实现 HiFi-GAN 声码器

太大放不进教程脚本，但结构如下：

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 个上采样块，总共 256 倍，从 mel 速率升到音频速率
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> 波形
```

训练：对抗训练（判别器看短窗口）+ mel 频谱图重建损失 + 特征匹配损失。已经商品化——直接用 `hifi-gan` 仓库或 nvidia-NeMo 的预训练 checkpoint。

### 第 5 步：完整流程（伪代码）

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## 使用

2026 年技术栈：

| 场景 | 选择 |
|------|------|
| 实时英语语音助手 | Kokoro（CPU）或 XTTS v2（GPU） |
| 5 秒参考零样本克隆 | F5-TTS |
| 商业角色音色 | ElevenLabs v2.5 |
| 有声书旁白 | ElevenLabs v2.5 或 XTTS v2 + 微调 |
| 低资源语言 | 在 5–20 小时目标语言数据上训练 VITS |
| 情感/表情标签 | ElevenLabs v2.5 或 StyleTTS 2 微调 |

截至 2026 年的开源首选：**F5-TTS 主打质量，Kokoro 主打效率**。除非你是做历史研究，否则不用碰 Tacotron。

## 陷阱

- **没有文本规范化器。** "Dr. Smith" 读成 "Doctor" 还是 "Drive"？"2026" 读成 "twenty twenty six" 还是 "two zero two six"？在音素化之前先规范化。
- **OOV 专有名词。** "Ghumare" → "ghyu-mair"。为未知 token 备一个字素转音素模型。
- **削波。** 声码器输出一般不会削波，但推理时 mel 缩放不匹配可能超出 ±1.0。始终用 `np.clip(wav, -1, 1)`。
- **采样率不匹配。** Kokoro 输出 24 kHz；下游 pipeline 期望 16 kHz → 重采样，否则产生混叠。

## 交付

保存为 `outputs/skill-tts-designer.md`。为给定音色、延迟和语言目标设计一个 TTS pipeline。

## 练习

1. **简单。** 运行 `code/main.py`。从一个 toy 词表构建音素字典，估计每个音素的时长，打印一个假的"mel"调度表。
2. **中等。** 安装 Kokoro，用 `af_bella` 和 `am_adam` 音色合成同一句话。对比音频时长和主观质量。
3. **困难。** 录一段 5 秒自己的参考音频。用 F5-TTS 克隆。报告参考音频与克隆输出的 SECS。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|-----------------------|
| 音素 (Phoneme) | 发音单位 | 抽象音位类；英语有 39 个（ARPABet）。 |
| 时长预测器 (Duration predictor) | 每个音素持续多久 | 非 AR 模型输出；每个音素的整帧数。 |
| 声码器 (Vocoder) | mel → 波形 | 将 mel 频谱图映射为原始采样点的神经网络。 |
| HiFi-GAN | 标准声码器 | GAN-based；2020–2024 年主流。 |
| MOS | 主观质量 | 人类评分员打出的 1–5 分平均意见分。 |
| SECS | 语音克隆指标 | 目标与输出说话人 embedding 之间的余弦相似度。 |
| F5-TTS | 2024 开源 SOTA | 流匹配扩散；零样本克隆。 |
| Kokoro | CPU 英语首选 | 8200 万参数模型，Apache 2.0。 |

## 延伸阅读

- [Shen 等 (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — seq2seq baseline。
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — 端到端流式。
- [Chen 等 (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 当前开源 SOTA。
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — 2026 年仍在广泛部署的声码器。
- [HuggingFace 上的 Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) — 2024 年 CPU 友好英语 TTS。