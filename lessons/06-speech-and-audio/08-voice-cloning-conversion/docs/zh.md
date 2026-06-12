# 语音克隆与语音转换

> 语音克隆是用别人的声音读你写的文本。语音转换是把你的声音改写成别人的，同时保留你说的内容。两者都依赖同一个分解：将说话人身份与内容分离。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 6 · 06（说话人识别）、阶段 6 · 07（TTS）
**时间：** 约 75 分钟

## 问题

到 2026 年，一段 5 秒的音频片段就足以在消费级 GPU 上生成高质量的任意声音克隆。ElevenLabs、F5-TTS、OpenVoice v2、VoiceBox 都提供零样本或少样本克隆。这项技术是福祉（无障碍 TTS、配音、辅助语音），也是武器（诈骗电话、政治深度伪造、IP 盗用）。

两个紧密相关的任务：

- **语音克隆（TTS 侧）：** 文本 + 5 秒参考声音 → 该音色的音频。
- **语音转换（语音侧）：** 源音频（A 说 X）+ B 的参考声音 → B 说 X 的音频。

两者都将波形分解为（内容、说话人、韵律），再将内容与说话人重新组合。

2026 年必须面对的关键约束：**水印和同意验证在欧盟（AI Act，2026 年 8 月可执行）和加州（AB 2905，2025 年生效）是法律强制要求**。Pipeline 必须嵌入不可听水印，并拒绝非同意克隆。

## 概念

![语音克隆 vs 转换：分解、交换说话人、重组](../assets/voice-cloning.svg)

**零样本克隆。** 将 5 秒片段传给在数千个说话人上训练过的模型。说话人编码器将片段映射为说话人 embedding；TTS 解码器以该 embedding 和文本为条件生成。

使用者：F5-TTS（2024）、YourTTS（2022）、XTTS v2（2024）、OpenVoice v2（2024）。

**少样本微调。** 录制目标声音 5–30 分钟。用 LoRA 微调基础模型约一小时。质量从"还行"跃升到"无法区分"。Coqui 和 ElevenLabs 都支持此模式；社区用它配合 F5-TTS。

**语音转换（VC）。** 两大类：

- **识别-合成。** 运行类似 ASR 的模型提取内容表示（如软音素后验、PPG），然后用目标说话人 embedding 重新合成。对语言和口音鲁棒。使用者：KNN-VC（2023）、Diff-HierVC（2023）。
- **解耦。** 训练一个自编码器，在瓶颈处将内容、说话人和韵律在隐空间中分离。推理时交换说话人 embedding。质量较低但更快。使用者：AutoVC（2019）、VITS-VC 变体。

**基于神经编解码器的克隆（2024+）。** VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox —— 将音频视为 SoundStream / EnCodec 的离散 token，训练一个大型自回归或流匹配模型来处理编解码器 token。质量可与 ElevenLabs 在短 prompt 上比肩。

### 伦理部分，不是附加物

**水印。** PerTh（Perth）和 SilentCipher（2024）在音频中嵌入约 16–32 比特 ID，不可察觉。能经受重编码、流媒体传输和常见编辑。生产就绪的开源方案。

**同意验证。** 必须为每个克隆输出配上可验证的同意记录。"我 Rohit 于 2026-04-22 授权此声音用于 X 目的。"存储在防篡改日志中。

**检测。** AASIST、RawNet2 和 Wav2Vec2-AASIST 作为检测器出货。ASVspoof 2025 挑战赛发布的最先进检测器对 ElevenLabs、VALL-E 2 和 Bark 输出的 EER 为 0.8%–2.3%。

### 数字（2026）

| 模型 | 零样本？ | SECS（目标相似度） | WER（可懂度） | 参数量 |
|------|--------|--------------------|--------------|--------|
| F5-TTS | 是 | 0.72 | 2.1% | 3.35 亿 |
| XTTS v2 | 是 | 0.65 | 3.5% | 4.70 亿 |
| OpenVoice v2 | 是 | 0.70 | 2.8% | 2.20 亿 |
| VALL-E 2 | 是 | 0.77 | 2.4% | 3.70 亿 |
| VoiceBox | 是 | 0.78 | 2.1% | 3.30 亿 |

SECS > 0.70 对大多数听众来说通常与目标无法区分。

## 构建

### 第 1 步：通过识别-合成分解（main.py 中仅代码演示）

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

概念简单；实现复杂度在 `tts_model` 和说话人编码器。

### 第 2 步：用 F5-TTS 零样本克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

参考转写稿必须与参考音频完全匹配，包括标点；不匹配会破坏对齐。

### 第 3 步：用 KNN-VC 做语音转换

```python
import torch
from knnvc import KNNVC  # 2023 模型，https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC 运行 WavLM 提取源和目标池的每帧 embedding，然后将每个源帧替换为池中最接近的帧。无参数，对一分钟目标语音有效。

### 第 4 步：嵌入水印

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # 返回 payload 字节
```

约 32 比特 payload，经 MP3 重编码和轻度噪声后仍可检测。

### 第 5 步：同意验证

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "需要签名同意"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 使用

2026 年技术栈：

| 场景 | 选择 |
|------|------|
| 5 秒零样本克隆，开源 | F5-TTS 或 OpenVoice v2 |
| 商业生产级克隆 | ElevenLabs 即时语音克隆 v2.5 |
| 语音转换（改写） | KNN-VC 或 Diff-HierVC |
| 多说话人微调 | StyleTTS 2 + 说话人适配器 |
| 跨语言克隆 | XTTS v2 或 VALL-E X |
| 深度伪造检测 | Wav2Vec2-AASIST |

## 陷阱

- **参考转写稿对齐不当。** F5-TTS 及类似模型要求参考文本与参考音频完全匹配，包括标点。
- **参考音频有混响。** 回声毁掉克隆效果。录干声、近距离麦克风。
- **情感不匹配。** 参考"欢快"，生成的任何内容都欢快克隆。将参考情感与目标用途匹配。
- **语言泄漏。** 克隆英语说话人然后让模型说法语，往往仍带口音；用跨语言模型（XTTS、VALL-E X）。
- **无水印。** 2026 年 8 月起在欧盟法律上无法出货。

## 交付

保存为 `outputs/skill-voice-cloner.md`。设计一个带同意验证 + 水印 + 质量目标的克隆或转换 pipeline。

## 练习

1. **简单。** 运行 `code/main.py`。演示说话人 embedding 交换，通过计算交换前后两个"说话人"之间的余弦相似度。
2. **中等。** 用 OpenVoice v2 克隆自己的声音。测量参考与克隆之间的 SECS。通过 Whisper 测量 CER。
3. **困难。** 对 20 个克隆应用 SilentCipher 水印，经过 128 kbps MP3 编码-解码，检测 payload。报告比特准确率。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 零样本克隆 (Zero-shot clone) | 5 秒就够了 | 预训练模型 + 说话人 embedding；无需训练。 |
| PPG | 音素后验图 | 每帧 ASR 后验，用作语言无关的内容表示。 |
| KNN-VC | 最近邻转换 | 将每个源帧替换为目标池中最接近的帧。 |
| 神经编解码器 TTS | VALL-E 风格 | 在 EnCodec/SoundStream token 上做 AR 模型。 |
| 水印 (Watermark) | 不可听签名 | 嵌入音频的比特，经重编码仍可存活。 |
| SECS | 克隆保真度 | 目标与克隆说话人 embedding 之间的余弦相似度。 |
| AASIST | 深度伪造检测器 | 反欺骗模型；检测合成语音。 |

## 延伸阅读

- [Chen 等 (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 开源 SOTA 零样本克隆。
- [Baevski 等 / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) 及 [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — 神经编解码器 TTS。
- [Qian 等 (2019). AutoVC](https://arxiv.org/abs/1905.05879) — 基于解耦的语音转换。
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — 基于检索的 VC。
- [SilentCipher (2024) — 音频水印](https://github.com/sony/silentcipher) — 生产就绪的 32 比特音频水印。
- [ASVspoof 2025 结果](https://www.asvspoof.org/) — 检测器 vs 生成器 arms race，2026 年更新。