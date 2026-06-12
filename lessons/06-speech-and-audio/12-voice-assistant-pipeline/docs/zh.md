# 构建语音助手流水线 — 阶段 6 顶点项目

> 将第 01-11 课的所有内容串联起来。构建一个能听、能思考、能回话的语音助手。2026 年这已是一个工程问题而非研究问题——但集成细节决定它能否交付。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 6 · 04、05、06、07、11；阶段 11 · 09（函数调用）；阶段 14 · 01（智能体循环）
**时间：** 约 120 分钟

## 问题

构建一个端到端助手：

1. 捕获麦克风输入（16 kHz 单声道）。
2. 检测用户语音的开始和结束。
3. 流式转录。
4. 将转录文本传给能调用工具（计时器、天气、日历）的 LLM。
5. 将 LLM 文本流式传输到 TTS。
6. 将音频播放给用户。
7. 如果用户在中途打断则停止。

延迟目标：在笔记本电脑 CPU 上，用户说完话后 800 ms 内输出第一个 TTS 音频字节。质量目标：无漏词，无对沉默的幻觉字幕，无语音克隆泄漏，无提示注入成功。

## 概念

![语音助手流水线：麦克风 → VAD → STT → LLM+工具 → TTS → 扬声器](../assets/voice-assistant.svg)

### 七个组件

1. **音频捕获。** 麦克风 → 16 kHz 单声道 → 20 ms chunks。生产环境通常用 Python 的 `sounddevice` 或原生 AudioUnit/ALSA/WASAPI。
2. **VAD（第 11 课）。** Silero VAD @ threshold 0.5，min speech 250 ms，silence hang-over 500 ms。发出"开始"和"结束"信号。
3. **流式 STT（第 4-5 课）。** Whisper-streaming、Parakeet-TDT 或 Deepgram Nova-3 (API)。部分 + 最终转录。
4. **带工具调用的 LLM。** GPT-4o / Claude 3.5 / Gemini 2.5 Flash。JSON schema 定义工具。流式输出 token。
5. **流式 TTS（第 7 课）。** Kokoro-82M（最快开源）或 Cartesia Sonic（商业）。收到 20 个 LLM token 后开始 TTS。
6. **播放。** 扬声器输出；为低带宽网络做 opus 编码。
7. **中断处理器。** 如果 TTS 播放期间 VAD 触发，停止播放，取消 LLM，重新开始 STT。

### 你会遇到的三个失败模式

1. **第一个词被截。** VAD 启动晚了一拍。用户说的"嘿"没了。将启动阈值设为 0.3 而不是 0.5。
2. **中途打断时响应混乱。** LLM 在用户打断后继续生成；助手和用户一起说话。将 VAD → 取消 LLM 连接起来。
3. **沉默幻觉。** Whisper 在静音预热帧上输出"感谢观看"。始终用 VAD 门控。

### 2026 年生产参考栈

| 栈 | 延迟 | 许可证 | 备注 |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | 商业 API | 2026 年行业默认 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | 大部分开源 | DIY 友好 |
| Moshi（全双工） | 200-300 ms | CC-BY 4.0 | 单模型；不同架构，第 15 课 |
| Vapi / Retell（托管） | 300-500 ms | 商业 | 启动最快；定制受限 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | 离线 | 开源 | 隐私 / 边缘部署 |

## 构建

### 第 1 步：麦克风捕获与分块（伪代码）

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### 第 2 步：VAD 门控的轮次捕获

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### 第 3 步：流式 STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### 第 4 步：在 LLM 循环中调用工具

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### 第 5 步：中断处理

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## 使用

参见 `code/main.py` 获取一个可运行的模拟，它用 stub 模块连接所有七个组件，这样即使没有硬件也能看到流水线的形态。要做真实实现，将 stub 替换为：

- `silero-vad`（`pip install silero-vad`）
- `deepgram-sdk` 或 `openai-whisper`
- `openai`（`gpt-4o`）或 `anthropic`
- `kokoro` 或 `cartesia`
- `sounddevice` 用于 I/O

## 陷阱

- **永远记录 PII。** 完整轮次音频在大多数管辖区都是 PII。30 天保留，加密存储。
- **没有打断。** 用户会打断。助手必须停止说话。
- **TTS 阻塞。** 同步 TTS 阻塞事件循环。使用异步或单独线程。
- **没有工具调用错误处理。** 工具会失败。LLM 必须收到错误 + 重试一次，然后优雅降级。
- **过度积极的幻觉过滤器。** 过滤过度，助手重复"我无法帮助您"。过滤不足，它什么都说。在保留集上校准。
- **没有唤醒词选项。** 始终监听是隐私风险。添加唤醒词门控（Porcupine 或 openWakeWord）。

## 交付

保存为 `outputs/skill-voice-assistant-architect.md`。给定预算 + 规模 + 语言 + 合规约束，产出完整的技术栈规格。

## 练习

1. **简单。** 运行 `code/main.py`。它模拟一个端到端的完整轮次，打印各阶段延迟。
2. **中等。** 将 STT stub 替换为预录 `.wav` 上的真实 Whisper 模型。测量 WER 和端到端延迟。
3. **困难。** 添加工具调用：实现 `get_weather`（任意 API）和 `set_timer`。让 LLM 路由经过工具，并验证当用户说"设置一个 5 分钟计时器"时正确函数被触发并用口语确认。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 轮次 (Turn) | 用户 + 助手的一轮对话 | 一个 VAD 界定的用户语音 + 一个 LLM-TTS 响应。 |
| 打断 (Barge-in) | 中断 | 用户在助手说话时开口；助手停止。 |
| 唤醒词 (Wake word) | "嘿，助手" | 短关键词检测器；Porcupine、Snowboy、openWakeWord。 |
| 端点检测 (End-pointing) | 轮次结束 | VAD + 最小沉默判断用户已说完。 |
| 预滚 (Pre-roll) | 预说话缓冲 | 在 VAD 触发前保持 200-400 ms 音频以避免第一个词被截。 |
| 工具调用 (Tool call) | 函数调用 | LLM 发出 JSON；运行时分发；结果反馈到循环中。 |

## 延伸阅读

- [LiveKit — 语音助手快速入门](https://docs.livekit.io/agents/) — 生产级参考。
- [Pipecat — 语音助手示例](https://github.com/pipecat-ai/pipecat) — DIY 友好框架。
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 托管原生语音路径。
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — 全双工参考（第 15 课）。
- [Porcupine 唤醒词](https://picovoice.ai/products/porcupine/) — 唤醒词门控。
- [Anthropic — 工具使用指南](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — LLM 函数调用。