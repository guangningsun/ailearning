# 音频语言模型 — Qwen2.5-Omni、Audio Flamingo、GPT-4o Audio

> 2026 年的音频语言模型能够对语音、环境声音和音乐进行推理。Qwen2.5-Omni-7B 在 MMAU-Pro 上与 GPT-4o Audio 持平。Audio Flamingo Next 在 LongAudioBench 上超越了 Gemini 2.5 Pro。开源与闭源之间的差距已基本消除——唯一例外是多音频任务，所有模型都接近随机水平。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 6 · 04 (ASR)、阶段 12 · 03 (视觉语言模型)、阶段 7 · 10 (音频 Transformer)
**时间：** 约 45 分钟

## 问题

给你 5 秒音频：狗叫声、有人喊"停！"，然后是沉默。有用的问题横跨多个维度：

- **转录。** "说了什么？" —— ASR 领域。
- **语义推理。** "那个人有危险吗？" —— 需要同时理解吠叫 + 喊叫 + 沉默。
- **音乐推理。** "什么乐器演奏了旋律？"
- **长音频检索。** "在这段 90 分钟的讲座中，讲师在哪里解释了梯度下降？"

能用一个提示回答所有这些问题的单一模型就是**音频语言模型** (LALM / ALM)。与纯 ASR 的区别在于：LALM 生成自由形式的自然语言答案，而不仅仅是转录文本。

## 概念

![音频语言模型：音频编码器 + 投影层 + LLM 解码器](../assets/alm-architecture.svg)

### 三组件模板

每个 2026 年的 LALM 都遵循相同骨架：

1. **音频编码器。** Whisper 编码器 · BEATs · CLAP · WavLM · 或各模型的自定义编码器。
2. **投影层。** 线性层或 MLP，将音频编码器特征映射到 LLM 的 token embedding 空间。
3. **LLM。** 基于 Llama / Qwen / Gemma 的解码器。接收交错的文本 + 音频 token；输出文本。

训练流程：

- **阶段 1。** 冻结编码器 + LLM；仅在 ASR / 标注数据上训练投影层。
- **阶段 2。** 在指令遵循音频任务（QA、推理、音乐理解）上进行全面 / LoRA 微调。
- **阶段 3（可选）。** 语音输入 / 语音输出需加入语音解码器。Qwen2.5-Omni 和 AF3-Chat 做了这一步。

### 2026 年模型地图

| 模型 | 主干网络 | 音频编码器 | 输出模态 | 访问方式 |
|-------|----------|------------|-----------|-----------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | Custom + Whisper | 文本 + 语音 | Apache-2.0 |
| Qwen3-Omni | Qwen3 | Custom | 文本 + 语音 | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | 文本 | NVIDIA 非商业许可 |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | 文本 | NVIDIA 非商业许可 |
| SALMONN | Vicuna | Whisper + BEATs | 文本 | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | 文本 | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | 文本 | Apache-2.0 |
| Gemini 2.5 Flash/Pro（闭源） | Gemini | 专有 | 文本 + 语音 | API |
| GPT-4o Audio（闭源） | GPT-4o | 专有 | 文本 + 语音 | API |

### 基准测试现实检验 (2026)

**MMAU-Pro。** 1800 个 QA 对，覆盖语音 / 声音 / 音乐 / 混合。多音频子集包含在内。

| 模型 | 总体 | 语音 | 声音 | 音乐 | 多音频 |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | LongAudioBench SOTA | — | — | — | — |

**多音频列对所有模型都是严峻考验。** 四选一多选题随机命中率 = 25%；大多数模型得分与此相当。LALM 在比较两个音频片段方面仍然困难。

### 2026 年 LALM 的应用场景

- **呼叫中心录音合规审计。** "客服是否提及了要求的披露？"
- **无障碍辅助。** 为听障用户描述声音事件（不仅仅是转录）。
- **内容审核。** 检测暴力语言 + 威胁语气 + 背景语境。
- **播客 / 会议章节划分。** 语义摘要，而非仅说话人转换。
- **音乐目录分析。** "找出所有有 B 段转调的曲目。"

### 尚不适用（目前）的场景

- 细粒度音乐理论（低于和弦级别）。
- 长对话中带说话人归属的推理（超过 10 分钟会退化）。
- 多音频比较（22-26% 几乎等同于随机）。
- 实时流式推理（大多数是离线批处理推理）。

## 构建

### 第 1 步：查询 Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### 第 2 步：投影层模式

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

就这么简单。投影层通常只有 1-3 个线性层。在 ASR 配对数据（音频 → 转录）上训练它是阶段 1 的代理任务。

### 第 3 步：在 MMAU / LongAudioBench 上基准测试

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

按类别（语音 / 声音 / 音乐 / 多音频）分别报告。汇总数字会掩盖模型的真实失败点。

## 使用

| 任务 | 2026 年推荐 |
|------|-----------|
| 自由形式音频 QA（开放式） | Qwen2.5-Omni-7B |
| 最佳开源长音频模型 | Audio Flamingo Next |
| 最佳闭源 | Gemini 2.5 Pro |
| 语音输入 / 输出智能体 | Qwen2.5-Omni 或 GPT-4o Audio |
| 音乐推理 | Audio Flamingo 3 或 2（音乐专用 AF-CLAP） |
| 呼叫中心审计 | 通过 API 使用 Gemini 2.5 Pro，配合 RAG 访问您的策略文档 |

## 陷阱

- **对多音频过度信任。** 如果您的任务需要"哪个片段有 X"，随机水平的表现是真实存在的。
- **长音频退化。** 超过 10 分钟，大多数模型的说话人归属会出问题。先做 diarization（第 6 课），再做摘要。
- **对沉默的幻觉。** 继承自使用 Whisper 编码器的 LALM 的相同问题。用 VAD 门控。
- **基准测试选择性报告。** 供应商博客文章突出最佳类别。亲自运行 MMAU-Pro 多音频子集。

## 交付

保存为 `outputs/skill-alm-picker.md`。为给定的音频理解任务选择 LALM + 基准测试子集 + 输出模态（文本 vs 语音）。

## 练习

1. **简单。** 运行 `code/main.py` 查看toy投影层模式 + 伪 LALM 路由（音频 embedding、文本 token） → 输出 token。
2. **中等。** 在 100 个 MMAU-Pro 语音项目上评估 Qwen2.5-Omni-7B。与论文报告的数字对比。
3. **困难。** 构建一个最小的音频标注基线：BEATs 编码器 + 2 层投影层 + 冻结的 Llama-3.2-1B。仅在 AudioCaps 上微调投影层。在 Clotho-AQA 上与 SALMONN 对比。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| LALM | 音频版 ChatGPT | 音频编码器 + 投影层 + LLM 解码器。 |
| 投影层 | 适配器 | 将音频特征映射到 LLM embedding 空间的小型 MLP。 |
| MMAU | 基准测试 | 10k 个音频 QA 对，覆盖语音、声音、音乐。 |
| MMAU-Pro | 更难的 MMAU | 1800 个多音频 / 重推理问题。 |
| LongAudioBench | 长音频评估 | 带语义查询的多分钟音频片段。 |
| 语音输入 / 输出 | 原生语音 | 模型摄入语音并输出语音，无需经过文本。 |

## 延伸阅读

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — 参考架构。
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — 语音进语音出。
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — 开源长音频领导者。
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench SOTA。
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — 双编码器先驱。
- [MMAU-Pro 排行榜](https://mmaubenchmark.github.io/) — 2026 年实时排名。