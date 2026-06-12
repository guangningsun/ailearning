# 音频 Transformer —— Whisper 架构

> 音频是频率随时间的图像。Whisper 是一个 ViT，吃进 mel 频谱图，吐出文字。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 7 · 05（全量 Transformer）、阶段 7 · 08（编码器-解码器）、阶段 7 · 09（ViT）
**时间：** 约 45 分钟

## 问题

在 Whisper（OpenAI，Radford 等人，2022）之前，业界最先进的自动语音识别（ASR）是 wav2vec 2.0 和 HuBERT —— 自监督特征提取器加微调头。高质量，但数据流程复杂、领域脆弱。多语言语音识别需要为每个语系单独建模。

Whisper 押注了三点：

1. **海量训练。** 从互联网抓取了 97 种语言的 68 万小时弱标注音频。没有干净的学术语料库，没有音素标签。
2. **多任务单模型。** 一个解码器联合训练转录、翻译、语音活动检测、语言识别和时间戳——通过任务 token 来区分。
3. **标准编码器-解码器 Transformer。** 编码器摄入 log-mel 频谱图，解码器自回归生成文本 token。无声码器，无 CTC，无 HMM。

结果：Whisper large-v3 在口音、噪声和零干净标注数据的语言上都表现出极强的鲁棒性。它是 2026 年所有开源语音助手和大多数商业语音助手的前端默认选择。

## 概念

![Whisper 流程：音频 → mel → 编码器 → 解码器 → 文本](../assets/whisper.svg)

### 步骤 1 —— 重采样 + 窗口

音频 16 kHz。裁剪/填充到 30 秒。计算 log-mel 频谱图：80 个 mel bin，10 ms 步长 → 约 3000 帧 × 80 特征。这就是 Whisper 看到的"输入图像"。

### 步骤 2 —— 卷积干

两层 Conv1D，卷积核 3，步长 2，将 3000 帧减少到 1500。序列长度减半，参数增加不多。

### 步骤 3 —— 编码器

24 层（large 版本）Transformer 编码器，处理 1500 个时间步。正弦位置编码、自注意力、GELU 前馈网络。输出 1500 × 1280 隐藏状态。

### 步骤 4 —— 解码器

24 层 Transformer 解码器。它自回归地从 BPE 词表中生成 token——该词表是 GPT-2 词表的超集，加入了少量音频专用特殊 token。

### 步骤 5 —— 任务 token

解码器 prompt 以控制 token 开头，告诉模型要做什么：

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

或者

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

模型就是按这个约定训练的。你通过前缀来控制任务。这是 2026 年的指令微调等价物，但应用在语音上。

### 步骤 6 —— 输出

束搜索（宽度 5）加上对数概率阈值。当没有 `<|notimestamps|>` token 时，每 0.02 秒音频预测一个时间戳。

### Whisper 规模

| 模型 | 参数 | 层数 | d_model | 头数 | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB（4 层解码器） |

Large-v3-turbo（2024）将解码器从 32 层削减到 4 层。解码速度快 8 倍，WER 下降不到 1 个点。这一解码速度的突破使得 Whisper-turbo 成为 2026 年实时语音代理的默认选择。

### Whisper 不做的事

- 没有说话人分离（ diarization）。需要配合 pyannote。
- 原生不支持实时流式——30 秒窗口是固定的。现代封装（`faster-whisper`、`WhisperX`）通过 VAD + 重叠来实现流式。
- 没有 30 秒以上的长上下文需要外部分块。实际效果很好，因为人类语音转录很少需要长距离上下文。

### 2026 年格局

| 任务 | 模型 | 说明 |
|------|-------|-------|
| 英语 ASR | Whisper-turbo，Moonshine | Moonshine 在边缘设备上快 4 倍 |
| 多语言 ASR | Whisper-large-v3 | 97 种语言 |
| 流式 ASR | faster-whisper + VAD | 可实现 150 ms 延迟目标 |
| TTS | Piper，XTTS-v2，Kokoro | 编码器-解码器模式，但形状类似 Whisper |
| 音频 + 语言 | AudioLM，SeamlessM4T | 在一个 transformer 中融合文本 token 和音频 token |

## 构建它

见 `code/main.py`。我们不训练 Whisper——我们构建 log-mel 频谱图管道 + 任务 token prompt 格式化器。这些才是你在生产环境中真正要接触的部分。

### 步骤 1：合成音频

生成一个 440 Hz 的 1 秒正弦波，采样率 16 kHz。16,000 个采样点。

### 步骤 2：log-mel 频谱图（简化版）

完整的 mel 频谱图需要 FFT。我们做简化版的分帧 + 每帧能量版本，展示管道而无需 `librosa`：

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

帧 = 25 ms，步长 = 10 ms。匹配 Whisper 的窗口。每帧能量代替 mel bin 用于教学。

### 步骤 3：填充到 30 秒

Whisper 始终处理 30 秒的块。将频谱图填充（或裁剪）到 3000 帧。

### 步骤 4：构建 prompt token

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

这就是整个任务控制面。一个 4 token 的前缀。

## 使用它

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

更快、OpenAI 兼容：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026 年何时选 Whisper：**

- 多语言 ASR，一个模型搞定。
- 嘈杂、多样音频的鲁棒转录。
- 研究 / 原型 ASR——最快的起点。

**何时选其他：**

- 边缘设备的超低延迟流式——Moonshine 在同等质量下超过 Whisper。
- 需要 <200 ms 的实时对话 AI——专用的流式 ASR。
- 说话人分离——Whisper 不做这个；配合 pyannote。

## 交付它

见 `outputs/skill-asr-configurator.md`。这个 skill 为新的语音应用选择 ASR 模型、解码参数和预处理管道。

## 练习

1. **简单。** 运行 `code/main.py`。确认 16 kHz、10 ms 步长下 1 秒信号的帧数约为 100 帧。30 秒：约 3000 帧。
2. **中等。** 用 `numpy.fft` 构建完整的 log-mel 频谱图。验证 80 个 mel bin 与 `librosa.feature.melspectrogram(n_mels=80)` 在数值误差范围内一致。
3. **困难。** 实现流式推理：将音频分块为 10 秒窗口、2 秒重叠，在每个块上运行 Whisper，合并转录。在 5 分钟播客样本上测量词错误率 vs 单次通过。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| Mel 频谱图 | "音频图像" | 2D 表示：一个轴是频率 bin，另一个轴是时间帧；每个格子的对数缩放能量。 |
| Log-mel | "Whisper 看到的东西" | 经过对数处理的 mel 频谱图；近似人类对响度的感知。 |
| 帧 | "一个时间切片" | 25 ms 的采样窗口；步长 10 ms 重叠。 |
| 任务 token | "语音的 prompt 前缀" | 解码器 prompt 中的特殊 token，如 `<|transcribe|>` / `<|translate|>`。 |
| 语音活动检测（VAD） | "找到语音" | 在 ASR 之前去掉静音的门控；大幅降低成本。 |
| CTC | "连接时序分类" | 经典的无对齐 ASR 损失函数；Whisper **不使用**它。 |
| Whisper-turbo | "小解码器，大编码器" | large-v3 编码器 + 4 层解码器；解码快 8 倍。 |
| Faster-whisper | "生产级封装" | CTranslate2 重实现；int8 量化；比 OpenAI 参考实现快 4 倍。 |

## 延伸阅读

- [Radford 等人（2022）。通过大规模弱监督实现鲁棒语音识别](https://arxiv.org/abs/2212.04356) —— Whisper 论文。
- [OpenAI Whisper 仓库](https://github.com/openai/whisper) —— 参考代码 + 模型权重。读 `whisper/model.py` 可以从头到尾看清 Conv1D 干 + 编码器 + 解码器，约 400 行。
- [OpenAI Whisper —— `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) —— 步骤 5-6 描述的束搜索 + 任务 token 逻辑在这里；500 行，完全可读。
- [Baevski 等人（2020）。wav2vec 2.0：语音自监督学习框架](https://arxiv.org/abs/2006.11477) —— 先驱；在某些场景下仍是 SOTA 特征。
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) —— 生产级封装，比参考实现快 4 倍。
- [Jia 等人（2024）。Moonshine：用于实时转录和语音命令的语音识别](https://arxiv.org/abs/2410.15608) —— 2024 年边缘友好型 ASR，形状类似 Whisper 但更小。
- [HuggingFace 博客 —— "用 🤗 Transformers 微调 Whisper 实现多语言 ASR"](https://huggingface.co/blog/fine-tune-whisper) —— 包含 mel 频谱图预处理器和 token 时间戳处理的微调规范指南。
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) —— 完整实现（编码器、解码器、交叉注意力、生成），与本课架构图对应。