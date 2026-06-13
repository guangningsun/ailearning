# Capstone 03 — 实时语音助手（ASR → LLM → TTS）

> 一个体验良好的语音智能体端到端延迟低于 800ms，知道用户何时停止说话，能处理打断，能在不停顿的情况下调用工具。Retell、Vapi、LiveKit Agents 和 Pipecat 在 2026 年都达到了这个标准。它们用同样的形态做到了：流式 ASR、轮次检测器、流式 LLM 和流式 TTS，全部通过 WebRTC 连接，每跳都有激进的延迟预算。构建一个，测量 WER 和 MOS 以及错误截断率，并在丢包环境下运行。

**类型：** Capstone
**语言：** Python（智能体 + 管道）、TypeScript（Web 客户端）
**前置条件：** 阶段 6（语音与音频）、阶段 7（Transformer）、阶段 11（LLM 工程）、阶段 13（工具）、阶段 14（智能体）、阶段 17（基础设施）
**涉及阶段：** P6 · P7 · P11 · P13 · P14 · P17
**时间：** 30 小时

## 问题

语音是 2025-2026 年发展最快的 AI UX 类别。技术天花板每个季度都在下降。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0 和 Pipecat 0.0.70 都让 800ms 以内的首次音频输出变得触手可及。门槛不仅是延迟，而是交互感受：不截断用户、不被用户截断、从句子中途打断中恢复、在对话中调用工具而不卡住音频、在不稳定的移动网络中存活。

你无法通过拼接三个 REST 调用来达到这个目标。架构是端到端的流水线流式处理。构建它，失败模式就会显现：为一个调优用于电话音频的 VAD 在背景电视上触发，一个轮次检测器等待一个永远不会来的标点符号，一个 TTS 缓冲 400ms 才发出声音。本 capstone 是要在负载下逐个修复这些问题，并发布一份延迟和质量报告。

## 概念

管道有五个流式阶段：**音频入**（来自浏览器或 PSTN 的 WebRTC）、**ASR**（来自 Deepgram Nova-3 或 faster-whisper 的流式部分转录）、**轮次检测**（VAD 加上一个读取部分转录寻找完成线索的小型轮次检测模型）、**LLM**（一旦轮次被判定完成就流式发出令牌）、**TTS**（在第一个 LLM 令牌后约 200ms 内流式发出音频）。

三个横切关注点。**打断处理**：当用户开始说话而智能体正在说话时，取消 TTS，ASR 立即拾取。**工具使用**：对话中的函数调用（天气、日历）必须在侧通道上运行而不卡住音频；如果延迟超过 300ms，智能体会预填充一个确认令牌（"稍等一下……"）。**背压**：在丢包下，部分转录被保留，VAD 提高语音门限，智能体避免在未确认的消息上说话。

测量标准是定量的。在 15dB SNR 的 Hamming VAD 基准上 WER 低于 8%。100 次测量通话中首次音频输出 p50 低于 800ms。错误截断率低于 3%。TTS MOS 高于 4.2。单台 g5.xlarge 上 50 路并发通话。这些数字是交付物。

## 架构

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## 技术栈

- 传输： LiveKit Agents 1.0（WebRTC）加 Twilio PSTN 网关；Pipecat 0.0.70 作为备选框架
- ASR： Deepgram Nova-3（流式，部分结果首次 < 300ms）或 faster-whisper Whisper-v3-turbo 自托管（GPU）
- VAD： Silero VAD v5 加 LiveKit 轮次检测器（读取部分转录的小型 transformer）
- LLM： OpenAI GPT-4o-realtime 紧密集成，Gemini 2.5 Flash Live，或级联的 Claude Haiku 4.5（流式补全，单独音频路径）
- TTS： Cartesia Sonic-2（最低首次字节延迟）、ElevenLabs Flash v3，或开源 Orpheus 自托管
- 工具： FastMCP 侧通道天气/日历/预订；如果工具 > 300ms 未返回，智能体预发填充语
- 可观测性： OpenTelemetry 语音 span，Langfuse 语音追踪带音频回放
- 部署： 单台 g5.xlarge（24GB VRAM）用于自托管 Whisper + Orpheus；托管 API 用于最低延迟

## 构建步骤

1. **WebRTC 会话。** 启动一个 LiveKit room 和一个流式传输麦克风音频的 Web 客户端。在服务器上，连接一个加入 room 的智能体 worker。

2. **ASR 流式处理。** 将 20ms PCM 帧送入 Deepgram Nova-3（或 GPU 上的 faster-whisper）。订阅部分和最终转录。记录每个部分结果的延迟。

3. **VAD 和轮次检测器。** 在帧流上运行 Silero VAD v5。在语音结束事件时，用最新的部分转录触发 LiveKit 轮次检测器。只有当 VAD 表示静默 500ms 且轮次检测器完成分数 > 0.6 时才提交"轮次完成"。

4. **LLM 流。** 轮次完成后，用运行中的对话加最终转录启动 LLM 调用。流式发出令牌。在第一个令牌时移交给 TTS。

5. **TTS 流。** Cartesia Sonic-2 流式发回音频块。第一个块必须在第一个 LLM 令牌后 200ms 内离开服务器。将块发送到 LiveKit room；客户端通过 WebRTC 抖动缓冲播放。

6. **打断处理。** 当 VAD 在 TTS 播放时检测到新的用户语音时，立即取消 TTS 流，丢弃剩余的 LLM 输出，重新启动 ASR。发布一个 `tts_canceled` span。

7. **工具侧通道。** 将天气和日历注册为函数调用工具。调用时并发触发；如果 300ms 内未解决，让 LLM 发出"稍等一下，让我查一下"作为填充语；工具返回后恢复。

8. **评估 harness。** 录制 100 通电话。计算 WER（对照 holdout 转录）、错误截断率（TTS 在用户句子中途被取消）、首次音频输出 p50、TTS MOS（人工或 NISQA），以及抖动丢包测试（丢弃 3% 的数据包）。

9. **负载测试。** 用合成呼叫者在单台 g5.xlarge 上驱动 50 路并发通话。测量持续的首次音频输出 p95。

## 使用方法

```
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## 交付

`outputs/skill-voice-agent.md` 是交付物。给定一个领域（客户支持、日程安排或 kiosk），它架起一个 LiveKit 智能体，其 ASR/VAD/LLM/TTS 管道经过调优达到测量标准。评分标准：

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 端到端延迟 | 100 次录制通话中 p50 首次音频输出低于 800ms |
| 20 | 轮次质量 | 在 Hamming VAD 基准上错误截断率低于 3% |
| 20 | 工具使用正确性 | 对话中工具调用返回正确数据而不卡住音频 |
| 20 | 丢包可靠性 | 注入 3% 丢包时 WER 和轮次稳定性 |
| 15 | 评估 harness 完整性 | 可复现的测量，带公开配置 |
| **100** | | |

## 练习

1. 将 Deepgram Nova-3 换成 g5.xlarge 上的 faster-whisper v3 turbo。测量延迟和 WER 差距。识别 CPU vs GPU 决策在哪里起作用。

2. 添加一个打断仲裁策略：当用户在工具调用期间打断时，智能体做什么？比较三种策略（硬取消、完成工具后停止、排队下一轮）。

3. 运行对抗性轮次检测器测试：让用户在句子中间长时间停顿。调优 VAD 静默阈值和轮次检测器分数阈值，在不超出 900ms 的情况下实现最低错误截断。

4. 通过 Twilio 在 PSTN 上部署同一个智能体。比较 PSTN 首次音频输出与 WebRTC。解释抖动缓冲和编解码器差异。

5. 为非英语语言添加语音活动检测（日语、西班牙语）。测量 Silero VAD v5 与语言特定微调的误触发率。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Turn detection | "话语结束" | 给定 VAD 静默和部分转录，判断用户已说完的分类器 |
| Barge-in | "打断处理" | 当 VAD 检测到新的用户语音时取消 TTS 中途播放 |
| First-audio-out | "延迟" | 从用户停止说话到第一个音频数据包离开服务器的时间 |
| VAD | "语音门" | 将音频帧分类为语音还是静默的模型；Silero VAD v5 是 2026 年的默认模型 |
| Jitter buffer | "音频平滑" | 客户端缓冲，短暂保留数据包以吸收网络方差 |
| Filler | "确认令牌" | 当工具较慢时智能体发出的短句以避免静默 |
| MOS | "平均意见分" | 感知语音质量评分；NISQA 是自动化的代理 |

## 延伸阅读

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — 参考 WebRTC 智能体框架
- [Pipecat](https://github.com/pipecat-ai/pipecat) — 备选 Python 优先流式智能体框架
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 集成语音模型参考
- [Deepgram Nova-3 文档](https://developers.deepgram.com/docs) — 流式 ASR 参考
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VAD 参考模型
- [Cartesia Sonic-2](https://docs.cartesia.ai) — 低延迟 TTS 参考
- [Retell AI 架构](https://docs.retellai.com) — 生产语音智能体架构
- [Vapi.ai 生产栈](https://docs.vapi.ai) — 备选生产参考
