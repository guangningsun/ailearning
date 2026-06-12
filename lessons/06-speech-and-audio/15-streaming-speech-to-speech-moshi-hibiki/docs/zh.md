# 流式语音到语音 — Moshi、Hibiki 及全双工对话

> 2024-2026 年重新定义了语音 AI。Moshi 发布了一个单一模型，同时听和说，延迟为 200 ms。Hibiki 逐块进行语音到语音翻译。两者都放弃了 ASR → LLM → TTS 管道，转而使用基于 Mimi 编解码器 token 的统一全双工架构。这是新的参考设计。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 6 · 13（神经音频编解码器）、阶段 6 · 11（实时音频）、阶段 7 · 05（完整 Transformer）
**时间：** 约 75 分钟

## 问题

从第 11 + 12 课构建的每个语音助手都有一个约 300-500 ms 的基本延迟下限：VAD 触发，STT 处理，LLM 推理，TTS 生成。每个阶段都有自己的最小延迟。你可以调整和并行化，但管道形状限制了你。

Moshi（Kyutai，2024-2026）提出了一个不同的问题：如果没有管道呢？如果一个模型直接、连续地接收音频输入并输出音频，文本作为中间"内心独白"而不是必需阶段呢？

答案是**全双工语音到语音**。理论延迟 160 ms（80 ms Mimi 帧 + 80 ms 声学延迟）。实际延迟在单个 L4 GPU 上为 200 ms。这只是最佳级联语音助手的一半。

## 概念

![Moshi 架构：两个并行 Mimi 流 + 内心独白文本](../assets/moshi-hibiki.svg)

### Moshi 架构

**输入。** 两个 Mimi 编解码器流，均为 12.5 Hz × 8 个码本：

- 流 1：用户音频（Mimi 编码，持续到达）
- 流 2：Moshi 自己的音频（Moshi 生成的）

**Transformer。** 一个 70 亿参数的 Temporal Transformer 处理两个流和一个文本"内心独白"流。在每个 80 ms 步骤中，它：

1. 消费最新的用户 Mimi token（8 个码本）。
2. 消费最近的 Moshi Mimi token（8 个码本，按产生顺序）。
3. 生成下一个 Moshi 文本 token（内心独白）。
4. 生成下一个 Moshi Mimi token（8 个码本，通过小型 Depth Transformer）。

三个流——用户音频、Moshi 音频、Moshi 文本——并行运行。Moshi 可以在说话时听到用户；当用户打断时可以在内部中断；可以不中断主要话语的情况下发出背信道（"嗯嗯"）。

**Depth Transformer。** 在一帧内，8 个码本不是并行预测的——它们有码本间依赖。一个小型 2 层"深度 Transformer"在 80 ms 内顺序预测它们。这是 AR 编解码器 LM 的标准分解（也被 VALL-E、VibeVoice 使用）。

### 为什么内心独白文本有帮助

没有明确的文本，模型必须在声学流中隐式建模语言。Moshi 的洞察：强制它同时发出文本 token 和音频。文本流本质上是 Moshi 正在说的话的转录。这提高了语义连贯性，使得更容易交换语言模型头，并且免费获得转录。

### Hibiki：流式语音到语音翻译

相同的架构，在翻译对上训练。源语言音频输入，目标语言音频输出，持续进行。Hibiki-Zero（2026 年 2 月）消除了词级对齐训练数据的需求——使用句子级数据 + GRPO 强化学习来优化延迟。

最初支持四种语言对；可以用约 1000 小时适应新语言。

### 更广泛的 Kyutai 技术栈（2026 年）

- **Moshi** — 全双工对话（首先法语，英语支持良好）
- **Hibiki / Hibiki-Zero** — 同时语音翻译
- **Kyutai STT** — 流式 ASR（500 ms 或 2.5 s 前瞻）
- **Kyutai Pocket TTS** — 100M 参数 TTS 在 CPU 上运行（2026 年 1 月）
- **Unmute** — 在公共服务器上组合这些的完整管道

在 L40S GPU 上的吞吐量：64 个并发会话，3 倍实时。

### Sesame CSM — 表亲

Sesame CSM（2025）使用类似的想法——一个 Llama-3 主干加上 Mimi 编解码器头。但 CSM 是单向的（接收上下文 + 文本，产生语音），而不是全双工。它是市场上最好的"语音存在感"TTS；与 Moshi 的全双工能力不尽相同。

### 2026 年性能数据

| 模型 | 延迟 | 用途 | 许可证 |
|------|------|------|--------|
| Moshi | 200 ms（L4） | 全双工英语/法语对话 | CC-BY 4.0 |
| Hibiki | 12.5 Hz 帧率 | 法语 ↔ 英语流式翻译 | CC-BY 4.0 |
| Hibiki-Zero | 相同 | 5 种语言对，无对齐数据 | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | 上下文条件 TTS | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | 闭源，OpenAI API | 商业 |
| Gemini 2.5 Live | ~350 ms | 闭源，Google API | 商业 |

## 动手实现

### 第 1 步：接口

Moshi 暴露一个 WebSocket 服务器，接收 80 ms 的 Mimi 编码音频块并返回 80 ms 的 Mimi 编码音频块。双向往返。持续进行。

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### 第 2 步：全双工循环

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

两个方向同时运行。Python asyncio 或 Rust futures 是标准传输方式。

### 第 3 步：训练目标（概念性）

对于每个 80 ms 帧 `t`：

- 输入：`user_mimi[0..t]`, `moshi_mimi[0..t-1]`, `moshi_text[0..t-1]`
- 预测：`moshi_text[t]`，然后 `moshi_mimi[t, codebook_0..7]`

文本在音频之前预测（内心独白）；音频在深度 transformer 内按码本顺序预测。

### 第 4 步：Moshi 赢在哪里和输在哪里

Moshi 赢的地方：

- 在廉价硬件上端到端亚 250 ms。
- 自然的背信道和打断。
- 无需管道胶水代码。

Moshi 输的地方：

- 工具调用（没有为此训练；你需要单独的 LLM 路径）。
- 长推理（Moshi 是一个约 80 亿参数的对话模型，不是 Claude/GPT-4）。
- 冷门话题的事实准确性。
- 大多数生产企业用例（2026 年仍使用管道）。

## 实际使用

| 情况 | 选择 |
|------|------|
| 最低延迟语音伴侣 | Moshi |
| 实时翻译通话 | Hibiki |
| 语音演示/研究 | Moshi、CSM |
| 带工具的企业代理 | 管道（第 12 课），不是 Moshi |
| 上下文中的自定义声音 TTS | Sesame CSM |
| 任意语言的语音到语音 | GPT-4o Realtime 或 Gemini 2.5 Live（商业） |

## 陷阱

- **有限的工具调用。** Moshi 是一个对话模型，不是代理框架。结合管道使用工具。
- **特定声音条件。** Moshi 使用单一训练角色；克隆是单独的训练过程。
- **语言覆盖。** 法语 + 英语优秀；其他有限。Hibiki-Zero 有帮助，但你仍然需要训练数据。
- **资源成本。** 一个完整的 Moshi 会话占用一个 GPU 槽；不是廉价的共享租户部署模式。

## 交付物

保存为 `outputs/skill-duplex-pipeline.md`。为语音助手工作负载选择管道与全双工架构，并说明理由。

## 练习

1. **简单。** 运行 `code/main.py`。它符号化地模拟双流 + 内心独白架构。
2. **中等。** 从 HuggingFace 拉取 Moshi，运行服务器，测试一次对话。测量从用户语音结束到 Moshi 响应开始的实际时钟延迟。
3. **困难。** 取你第 12 课的管道代理，在 20 个匹配测试话语上比较 P50 延迟与 Moshi。写出管道何时在架构上仍然获胜。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|----------|
| 全双工 | 同时听和说 | 两个音频流同时在同一模型上活跃。 |
| 内心独白 | 模型的文本流 | Moshi 与音频输出一起发出文本 token。 |
| 深度 Transformer | 码本间预测器 | 小型 transformer，在 80 ms 一帧内预测 8 个码本。 |
| Mimi | Kyutai 的编解码器 | 12.5 Hz × 8 个码本；语义+声学；驱动 Moshi。 |
| 流式 S2S | 音频 → 实时音频 | 逐块翻译/对话，无需管道阶段。 |
| 背信道 | "嗯嗯"反应 | Moshi 可以在不中断自己话轮的情况下发出小确认。 |

## 延伸阅读

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — 论文。
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) — 无对齐数据的流式翻译。
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — CSM 规范。
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) — 安装 + 服务器。
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — 闭源商业同类。
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) — 底层 STT/TTS 框架。