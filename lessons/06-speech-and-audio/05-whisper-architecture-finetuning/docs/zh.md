# Whisper — 架构与微调

> Whisper 是一个 30 秒窗口的 transformer 编码器-解码器，在 68 万小时的多语言弱监督音频-文本对上训练。一个架构，多种任务，跨越 99 种语言鲁棒运行。2026 年的参考 ASR。

**类型：** 动手构建
**语言：** Python
**前置条件：** 阶段 6 · 04（ASR）、阶段 5 · 10（注意力机制）、阶段 7 · 05（完整 Transformer）
**时间：** 约 75 分钟

## 问题

Whisper 由 OpenAI 于 2022 年 9 月发布，是第一个作为商品发货的 ASR 模型：粘贴音频，获得文本，99 种语言，对噪声鲁棒，在笔记本上运行。到 2024 年 OpenAI 已发布 Large-v3 和 Turbo 变体；到 2026 年，Whisper 已成为从播客转录到语音助手到 YouTube 字幕的所有默认基线。

但 Whisper 不是一个可以永远当作黑盒的管道。领域偏移会杀死它——技术术语、说话人口音、专有名词、短片段、静音。你需要知道：

1. 它内部实际是什么。
2. 如何正确处理分块、流式或长音频。
3. 何时微调以及如何微调。

## 概念

![Whisper 编码器-解码器、任务、分块推理、微调](../assets/whisper.svg)

**架构。** 标准 transformer 编码器-解码器。

- 输入：30 秒 log-mel 频谱图，80 mel，10 ms 跳步 → 3000 帧。更短的片段零填充，更长的片段分块。
- 编码器：卷积下采样（步长 2）+ `N` 个 transformer 块。Large-v3：32 层，1280 维，20 头。
- 解码器：`N` 个 transformer 块，带因果自注意 + 对编码器输出的交叉注意。与编码器尺寸相同。
- 输出：51,865 token 词表的 BPE token。

Large-v3 有 1.55B 参数。Turbo 使用 4 层解码器（来自 32 层），延迟降低 8 倍，WER 仅下降 <1%。

**提示格式。** Whisper 是一个多任务模型，通过解码器提示中的特殊 token 来引导：

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — 语言标签；强制翻译 vs 转录行为。
- `<|transcribe|>` 或 `<|translate|>` — 从任意语言输入翻译为英语输出，或逐字转录。
- `<|notimestamps|>` — 跳过词级时间戳（更快）。

提示使得一个模型可以执行多种任务。将 `<|en|>` 改为 `<|fr|>` 就可以转录法语。

**30 秒窗口。** 一切以 30 秒为基准。更长的片段需要分块；更短的片段需要填充。窗口本身不支持原生流式处理——这就是 WhisperX、Whisper-Streaming 和 faster-whisper 存在的原因。

**Log-mel 归一化。** `(log_mel - mean) / std`，其中统计量来自 Whisper 自己的训练语料。你*必须*使用 Whisper 的预处理（`whisper.audio.log_mel_spectrogram`），而不是 `librosa.feature.melspectrogram`。

### 2026 年变体

| 变体 | 参数 | 延迟（A100） | WER（LibriSpeech-clean） |
|---------|--------|----------------|------------------------|
| Tiny | 39M | 1× 实时 | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming（2024） | 1.55B | 流式 | 2.0% |

### 微调

2026 年的标准工作流程：

1. 收集 10–100 小时目标领域音频及对齐转录。
2. 使用 `generate_with_loss` 回调运行 `transformers.Seq2SeqTrainer`。
3. 参数高效：LoRA 应用于注意力层的 `q_proj`、`k_proj`、`v_proj`，GPU 内存降低 4 倍，WER 代价 <0.3%。
4. 如果数据少于 10 小时，冻结编码器。只微调解码器。
5. 使用 Whisper 自己的分词器和提示格式；永不更换分词器。

社区结果：在 20 小时医疗听写上微调 Medium，医疗词汇 WER 从 12% 降至 4.5%。在 4 小时冰岛语上微调 Turbo，WER 从 18% 降至 6%。

## 动手构建

### 第 1 步：开箱即用 Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # 防止失控重复
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

始终应覆盖的关键默认值：`temperature=0.0`（采样默认 0.0 → 0.2 → 0.4 … 回退链），`condition_on_previous_text=False`（防止级联幻觉问题），以及 `no_speech_threshold=0.6`（静音检测）。

### 第 2 步：分块长音频

```python
# whisperx 是 2026 年长音频带词级时间戳的参考实现
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX 添加了 (1) Silero VAD 门控，(2) 通过 wav2vec 2.0 进行词级对齐，(3) 通过 `pyannote.audio` 进行说话人分离。2026 年生产转录的主力。

### 第 3 步：使用 LoRA 微调

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M 可训练 / 809M 总计
```

然后是标准 Trainer 循环。每 1000 步保存检查点。在留出集上用 WER 评估。

### 第 4 步：检查每层学到了什么

```python
# 在解码期间获取交叉注意力权重，看解码器关注什么。
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

用热图可视化——你会看到对角线对齐，因为解码器步骤扫描编码器帧。那条对角线就是 Whisper 的词时间戳概念。

## 使用

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| 通用英语、离线 | Large-v3-turbo 通过 `whisperx` |
| 移动端 / 边缘 | Whisper-Tiny 量化（int8）或 Moonshine |
| 多语言长音频 | Large-v3 通过 `whisperx` + 说话人分离 |
| 低资源语言 | 使用 LoRA 微调 Medium 或 Turbo |
| 流式处理（2 秒延迟） | Whisper-Streaming 或 Parakeet-TDT |
| 词级时间戳 | WhisperX（通过 wav2vec 2.0 强制对齐） |

`faster-whisper`（CTranslate2 后端）是 2026 年最快的 CPU+GPU 推理运行时——比原版快 4 倍，输出完全相同。

## 坑

2026 年仍然在发货的坑：

- **静音上的幻觉文本。** Whisper 在字幕上训练，包括 "Thanks for watching!"、"Subscribe!"、歌词。调用前务必用 VAD 门控。
- **`condition_on_previous_text` 级联。** 一个幻觉污染后续窗口。除非需要跨片段的流畅性，否则设为 `False`。
- **短片段填充。** 填充到 30 秒的 2 秒片段可能在尾部静音产生幻觉。使用 `pad=False` 或 VAD 门控。
- **错误的 mel 统计量。** 使用 librosa 的 mel 而非 Whisper 的会产生接近随机的输出。使用 `whisper.audio.log_mel_spectrogram`。

## 交付

保存为 `outputs/skill-whisper-tuner.md`。为给定领域设计 Whisper 微调或推理管道。

## 练习

1. **简单。** 运行 `code/main.py`。它对一个 Whisper 风格的提示进行分词，计算解码后的形状预算，并打印 10 分钟片段的分块计划。
2. **中等。** 安装 `faster-whisper`，转录 10 分钟播客，与人工转录本比较 WER。尝试 `language="auto"` vs 强制 `language="en"`。
3. **困难。** 使用 HF `datasets`，选择 Whisper 表现较差的语言（如乌尔都语），在 2 小时数据上微调 Medium + LoRA 2 个 epoch，报告 WER 变化。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 30 秒窗口 | Whisper 的限制 | 硬输入上限；将更长音频分块。 |
| SOT | 转录开始 | `<|startoftranscript|>` 启动解码器提示。 |
| 时间戳 token | 时间对齐 | 每 0.02 秒偏移是 51k 词表中的一个特殊 token。 |
| Turbo | 快速变体 | 4 层解码器，快 8 倍，WER 下降 <1%。 |
| WhisperX | 长音频包装器 | VAD + Whisper + wav2vec 对齐 + 说话人分离。 |
| LoRA 微调 | 高效调优 | 向注意力添加低秩适配器；训练约 0.3% 的参数。 |
| 幻觉 | 静默故障 | Whisper 从噪声/静音中产生流利英语。 |

## 延伸阅读

- [Radford et al. (2022). Whisper 论文](https://arxiv.org/abs/2212.04356) — 原始架构和训练配方。
- [OpenAI (2024). Whisper Large-v3-turbo 发布](https://github.com/openai/whisper/discussions/2363) — 4 层解码器，8 倍加速。
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — 长音频、词对齐、说话人分离。
- [Systran — faster-whisper 仓库](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 后端，4 倍更快。
- [HuggingFace — Whisper 微调教程](https://huggingface.co/blog/fine-tune-whisper) — 标准 LoRA / 全量微调 walkthrough。