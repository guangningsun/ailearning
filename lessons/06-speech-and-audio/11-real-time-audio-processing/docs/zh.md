# 实时音频处理

> 批处理流水线处理文件。实时流水线在下一次 20 毫秒到来之前处理当前这一次。每个对话式 AI、广播工作室和电话机器人都依赖这个延迟预算生存或死亡。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 6 · 02 (频谱图)、阶段 6 · 04 (ASR)、阶段 6 · 07 (TTS)
**时间：** 约 75 分钟

## 问题

你想要一个感觉"活着"的语音助手。人类对话轮次延迟约 230 ms（沉默到响应）。超过 500 ms 会感觉像机器人；超过 1500 ms 会感觉坏了。2026 年一个完整的 **听 → 理解 → 回应 → 说话** 环路的预算：

| 阶段 | 预算 |
|-------|--------|
| 麦克风 → 缓冲 | 20 ms |
| VAD | 10 ms |
| ASR（流式） | 150 ms |
| LLM（首 token） | 100 ms |
| TTS（首 chunk） | 100 ms |
| 渲染 → 扬声器 | 20 ms |
| **总计** | **~400 ms** |

Moshi (Kyutai, 2024) 实测 200 ms 全双工。GPT-4o-realtime (2024) 实测约 320 ms。2022 年的级联流水线发货时延迟为 2500 ms。10 倍的提升来自三个技术：(1) 全流式化，(2) 异步流水线与部分结果，(3) 可中断生成。

## 概念

![流式音频流水线，含环形缓冲、VAD 门控、中断](../assets/real-time.svg)

**帧 / chunk / 窗口。** 实时音频以固定大小的块流动。常见选择：20 ms（16 kHz 下 320 个样本）。下游所有组件必须跟上这个节拍。

**环形缓冲。** 固定大小的环形缓冲区。生产者线程写入新帧，消费者线程读取。防止热路径中的内存分配。大小 ≈ 最大延迟 × 采样率；2 秒 16 kHz 环形缓冲 = 32,000 个样本。

**VAD（语音活动检测）。** 当无人说话时阻断下游工作。Silero VAD 4.0 (2024) 在 CPU 上每 30 ms 帧运行 <1 ms。`webrtcvad` 是更老的替代方案。

**流式 ASR。** 在音频到达时即输出部分转录的模型。Parakeet-CTC-0.6B 流式模式（NeMo, 2024）在 320 ms 延迟下达到 2-5% WER。Whisper-Streaming (Macháček et al., 2023) 将 Whisper 分块以实现近流式，延迟约 2 秒。

**中断。** 当用户在助手说话时开口，必须 (a) 检测打断，(b) 停止 TTS，(c) 丢弃剩余的 LLM 输出。所有操作需在 100 ms 内完成，否则用户会感觉助手"聋了"。

**WebRTC Opus 传输。** 20 ms 帧，48 kHz，自适应比特率 8-128 kbps。浏览器和移动端的标准。LiveKit、Daily.co、Pion 是 2026 年构建语音应用的技术栈。

**抖动缓冲。** 网络数据包乱序或延迟到达。抖动缓冲器重新排序并平滑；太小 → 可听间隙，太大 → 延迟。典型值 60-80 ms。

### 常见陷阱

- **线程竞争。** Python 的 GIL + 重量级模型可能饿死音频线程。使用 C 回调音频库（sounddevice、PortAudio）并让 Python 远离热路径。
- **重采样延迟。** 流水线内部重采样增加 5-20 ms。要么预先重采样，要么使用零延迟重采样器（PolyPhase、`soxr_hq`）。
- **TTS 预热。** 即使是 Kokoro 这样的快速 TTS，首次请求也有 100-200 ms 预热。在第一次真实交互前缓存模型并用虚拟运行预热。
- **回声消除。** 没有 AEC，TTS 输出会重新进入麦克风并触发 ASR 处理助手自己的声音。WebRTC AEC3 是开源默认方案。

## 构建

### 第 1 步：环形缓冲

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

容量决定最大缓冲延迟。32,000 个样本（16 kHz）= 2 秒。

### 第 2 步：VAD 门控

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

生产环境替换为 Silero VAD：

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### 第 3 步：流式 ASR

```python
# 通过 NeMo 使用 Parakeet-CTC-0.6B 流式识别
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 第 4 步：中断处理器

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # 打断
        # 然后喂给流式 ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

关键在于异步 I/O 和可取消的 TTS 流式传输。WebRTC peerconnection.stop() 是停止音轨的标准方式。

## 使用

2026 年技术栈：

| 层级 | 推荐选择 |
|-------|------|
| 传输 | LiveKit (WebRTC) 或 Pion (Go) |
| VAD | Silero VAD 4.0 |
| 流式 ASR | Parakeet-CTC-0.6B 或 Whisper-Streaming |
| LLM 首 token | Groq、Cerebras、vLLM-streaming |
| 流式 TTS | Kokoro 或 ElevenLabs Turbo v2.5 |
| 回声消除 | WebRTC AEC3 |
| 端到端原生 | OpenAI Realtime API 或 Moshi |

## 陷阱

- **为安全起见缓冲 500 ms。** 缓冲器 *就是* 你的延迟地板。缩小它。
- **不固定线程。** 音频回调在优先级低于 UI 的线程上 = 负载下出现杂音。
- **TTS chunk 太小。** 低于 200 ms 的 chunk 会让声码器伪影可闻。320 ms chunk 是最佳点。
- **没有抖动缓冲。** 真实网络有抖动；不平滑会出现爆音。
- **一次性错误处理。** 音频流水线必须防崩溃。一次异常终结整个会话。

## 交付

保存为 `outputs/skill-realtime-designer.md`。设计一个实时音频流水线，给出每阶段的明确延迟预算。

## 练习

1. **简单。** 运行 `code/main.py`。模拟环形缓冲 + 能量 VAD；为假 10 秒流打印各阶段延迟。
2. **中等。** 使用 `sounddevice` 构建直通循环，以 20 ms 帧处理麦克风输入并在每帧打印 VAD 状态。
3. **困难。** 用 `aiortc` 构建全双工回声测试：浏览器 → WebRTC → Python → WebRTC → 浏览器。用 1 kHz 脉冲测量玻璃到玻璃延迟。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 环形缓冲 | 环形队列 | 固定大小、无锁（或 SPSC 加锁）FIFO，用于音频帧。 |
| VAD | 静音门 | 模型或启发式方法标记语音与非语音。 |
| 流式 ASR | 实时 STT | 音频到达时即输出部分文本；有界前瞻。 |
| 抖动缓冲 | 网络平滑器 | 对失序数据包重新排序的队列；典型值 60-80 ms。 |
| AEC | 回声消除 | 减去扬声器到麦克风的反馈路径。 |
| 打断 | 用户中断 | 系统检测到用户在中途说话；必须停止播放。 |
| 全双工 | 双向同时 | 用户和机器人可以同时说话；Moshi 是全双工。 |

## 延伸阅读

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — 分块近流式 Whisper。
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — 全双工 200 ms 延迟。
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — 生产级音频智能体编排。
- [Silero VAD 仓库](https://github.com/snakers4/silero-vad) — 子 1 ms VAD，Apache 2.0。
- [WebRTC AEC3 论文](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — 开源回声消除。