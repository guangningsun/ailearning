# 语音活动检测与话轮转换 — Silero、Cobra 及 Flush 技巧

> 每个语音助手都取决于两个决策：用户现在是否在说话，以及他们是否说完了？VAD 回答第一个。话轮检测（VAD + 静音延续 + 语义端点模型）回答第二个。任何一个决策出错，你的助手要么打断用户，要么永远停不下来。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 6 · 11（实时音频）、阶段 6 · 12（语音助手）
**时间：** 约 45 分钟

## 问题

语音助手在每个 20 ms 片段上做出三个不同的决策：

1. **这一帧是语音吗？** — VAD。二值，每帧判断。
2. **用户开始新的话语了吗？** — 起音检测。
3. **用户结束了吗？** — 端点检测（话轮结束）。

naive 的答案（能量阈值）在任何噪声下都会失败——交通、键盘、人群嘈杂声。2026 年的答案：Silero VAD（开源、深度学习）+ 话轮检测模型（语义端点）+ VAD 校准的静音延续。

## 概念

![VAD 级联：能量 → Silero → 话轮检测器 → Flush 技巧](../assets/vad-turn-taking.svg)

### 三层 VAD 级联

**第一层：能量门。** 最便宜。RMS 阈值在 -40 dBFS。过滤明显的静音，但对任何超过阈值的噪声都会触发。

**第二层：Silero VAD**（2020-2026，MIT）。100 万参数。在 6000+ 种语言上训练。在单个 CPU 线程上每个 30 ms 片段约 1 ms 内运行。5% FPR 下 TPR 87.7%。开源默认选择。

**第三层：语义话轮检测器。** LiveKit 的话轮检测模型（2024-2026）或你自己的小型分类器。区分"话中停顿"和"说完"。使用语言学上下文（语调 + 最近词汇），而不仅仅是静音。

### 关键参数及其默认值

- **阈值。** Silero 输出一个概率；在 > 0.5（默认）或 > 0.3（敏感）分类为语音。阈值越低 = 首词被切掉的情况越少，但假阳性越多。
- **最小语音时长。** 拒绝短于 250 ms 的语音——通常是咳嗽声或椅子声。
- **静音延续（端点检测）。** VAD 返回 0 后，等待 500-800 ms 再宣布话轮结束。太短 → 打断用户。太长 → 感觉迟钝。
- **预滚动缓冲区。** 在 VAD 触发前保持 300-500 ms 的音频。防止"嘿"被切掉。

### Flush 技巧（Kyutai 2025）

流式 STT 模型有前瞻延迟（Kyutai STT-1B 为 500 ms，STT-2.6B 为 2.5 s）。通常你在语音结束后要等那么长时间才能得到转录。Flush 技巧：当 VAD 触发语音结束时，**向 STT 发送一个 flush 信号**，强制立即输出。STT 以约 4 倍实时速度处理，所以 500 ms 缓冲区在约 125 ms 内完成。

端到端：125 ms VAD + flush STT = 对话延迟。

### 2026 年 VAD 比较

| VAD | 5% FPR 下 TPR | 延迟 | 许可证 |
|-----|--------------|------|--------|
| WebRTC VAD（Google，2013） | 50.0% | 30 ms | BSD |
| Silero VAD（2020-2026） | 87.7% | ~1 ms | MIT |
| Cobra VAD（Picovoice） | 98.9% | ~1 ms | 商业 |
| pyannote 分割 | 95% | ~10 ms | MIT-ish |

Silero 是正确的默认选择。Cobra 是合规性/准确性的升级。纯能量 VAD 在 2026 年的生产环境中没有立足之地。

## 动手实现

### 第 1 步：能量门

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### 第 2 步：Python 中的 Silero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### 第 3 步：话轮结束状态机

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### 第 4 步：Flush 技巧骨架

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT（Kyutai、Deepgram、AssemblyAI）必须支持 flush 才能工作。Whisper 流式不支持——它是基于块的，始终等待片段。

## 实际使用

| 情况 | VAD 选择 |
|------|----------|
| 开源、快速、通用 | Silero VAD |
| 商业呼叫中心 | Cobra VAD |
| 设备端（手机） | Silero VAD ONNX |
| 研究/ diarization | pyannote 分割 |
| 零依赖后备 | WebRTC VAD（传统） |
| 需要话轮结束质量 | Silero + LiveKit 话轮检测器分层 |

经验法则：除非真的没有其他选择，否则不要使用纯能量 VAD。

## 陷阱

- **固定阈值。** 在安静时有效，在噪声中失败。要么在设备上校准，要么切换到 Silero。
- **静音延续太短。** 助手在句子中间打断用户。对话语音的甜蜜点是 500-800 ms。
- **延续太长。** 感觉迟钝。与目标用户 A/B 测试。
- **没有预滚动缓冲区。** 丢失用户前 200-300 ms 的音频。始终保持滚动预滚动。
- **忽略语义端点。** "嗯，让我想想……"包含长时间停顿。用户讨厌在思考过程中被打断。使用 LiveKit 的话轮检测器或类似模型。

## 交付物

保存为 `outputs/skill-vad-tuner.md`。为工作负载选择 VAD 模型、阈值、延续、预滚动和话轮检测策略。

## 练习

1. **简单。** 运行 `code/main.py`。它模拟语音 + 静音 + 语音 + 咳嗽序列并测试三个 VAD 层。
2. **中等。** 安装 `silero-vad`，处理 5 分钟录音，调整阈值以最小化首词被切和假触发。报告精确率/召回率。
3. **困难。** 构建一个小型话轮检测器：Silero VAD + 一个 3 层 MLP（在最后 10 个词的嵌入上，使用 sentence-transformers）。在手工标注的话轮结束数据集上训练。将 Silero 单独使用提高 10% F1。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|----------|
| VAD | 语音检测器 | 二值每帧：是语音吗？ |
| 话轮检测 | 端点检测 | VAD + 静音延续 + 语义端点。 |
| 静音延续 | 语音后等待 | 宣布话轮结束前等待的时间；500-800 ms。 |
| 预滚动 | 预语音缓冲区 | 在 VAD 触发前保持 300-500 ms 音频。 |
| Flush 技巧 | Kyutai 技巧 | VAD → flush-STT → 125 ms 而不是 500 ms 延迟。 |
| 语义端点 | "他们是要停下来吗？" | 查看词汇而不仅仅是静音的 ML 分类器。 |
| 5% FPR 下 TPR | ROC 点 | 标准 VAD 基准；Silero 87.7%，WebRTC 50%。 |

## 延伸阅读

- [Silero VAD](https://github.com/snakers4/silero-vad) — 参考开源 VAD。
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — 商业准确率领先。
- [Kyutai — Unmute + flush 技巧](https://kyutai.org/stt) — 亚 200 ms 工程技巧。
- [LiveKit — 话轮检测](https://docs.livekit.io/agents/logic/turns/) — 生产中的语义端点。
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — 传统基线。
- [pyannote 分割](https://github.com/pyannote/pyannote-audio) — diarization 级分割。