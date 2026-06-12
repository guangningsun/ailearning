# 音频生成

> 音频是一种 1-D 信号，采样率 16-48 kHz。五秒钟的音频片段包含 80-240k 个采样点。没有任何 Transformer 能直接处理这么长的序列。2026 年所有生产级音频模型都采用相同方案：神经编解码器（Encodec、SoundStream、DAC）将音频压缩为 50-75 Hz 的离散 token，然后由 Transformer 或扩散模型生成 token。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 6 · 02（音频特征）、阶段 6 · 04（ASR）、阶段 8 · 06（DDPM）
**时间：** 约 45 分钟

## 问题

三类音频生成任务：

1. **语音合成（TTS）。** 输入文本，输出语音。纯净语音是窄带的且具有强语音结构——用 Transformer-over-tokens 解决得很好。VALL-E（微软）、NaturalSpeech 3、ElevenLabs、OpenAI TTS。
2. **音乐生成。** 输入提示（文本、旋律、和弦进行、风格），输出音乐。分布范围更广。MusicGen（Meta）、Stable Audio 2.5、Suno v4、Udio、Riffusion。
3. **音效 / 声音设计。** 输入提示，输出环境音或 Foley。AudioGen、AudioLDM 2、Stable Audio Open。

三者都运行在相同的基础之上：神经音频编解码器 + token 自回归或扩散生成器。

## 概念

![音频生成：编解码器 token + Transformer 或扩散](../assets/audio-generation.svg)

### 神经音频编解码器

Encodec（Meta，2022）、SoundStream（Google，2021）、Descript Audio Codec（DAC，2023）。卷积编码器将波形压缩为每时间步的向量；残差矢量量化（RVQ）将每个向量转换为 K 个码本的索引串级。解码器逆向操作。24 kHz 音频在 2 kbps 下使用 8 个 RVQ 码本以 75 Hz 运行 = 600 tokens/秒。

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### 在此之上的两种生成范式

**Token 自回归。** 将 RVQ token 展平为序列，运行纯解码器 Transformer。MusicGen 使用"延迟并行"以每流偏移量并行发出 K 个码本流。VALL-E 从文本提示 + 3 秒语音样本生成语音 token。

**潜在扩散。** 将编解码器 token 打包为连续潜在向量，或用分类扩散建模。Stable Audio 2.5 在连续音频潜在向量上使用流匹配。AudioLDM 2 使用文本到 mel 再到音频的扩散。

2024-2026 年的趋势：流匹配在音乐领域胜出（推理更快、样本更干净），而 token 自回归仍主导语音领域，因为它天然是因果的且易于流式传输。

## 生产格局

| 系统 | 任务 | 骨干网络 | 延迟 |
|--------|------|----------|---------|
| ElevenLabs V3 | TTS | Token-AR + 神经声码器 | 首 token 约 300ms |
| OpenAI GPT-4o audio | 全双工语音 | 端到端多模态 AR | 约 200ms |
| NaturalSpeech 3 | TTS | 潜在流匹配 | 非流式 |
| Stable Audio 2.5 | 音乐 / 音效 | DiT + 音频潜在向量上的流匹配 | 1 分钟片段约 10s |
| Suno v4 | 完整歌曲 | 未公开；疑似 token-AR | 每首歌约 30s |
| Udio v1.5 | 完整歌曲 | 未公开 | 每首歌约 30s |
| MusicGen 3.3B | 音乐 | Encodec 32kHz 上的 Token-AR | 实时 |
| AudioCraft 2 | 音乐 + 音效 | 流匹配 | 5 秒片段约 5s |
| Riffusion v2 | 音乐 | 频谱图扩散 | 约 10s |

## 构建它

`code/main.py` 模拟核心思想：在由两种不同"风格"生成的合成"音频 token"序列上训练一个微型 next-token Transformer（风格 A 为交替的低 token 和高 token，风格 B 为单调递增）。以风格和样本为条件。

### 第 1 步：合成音频 token

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "类语音"：交替
        return [i % vocab_size for i in range(length)]
    # "类音乐"：递增
    return [(i * 3) % vocab_size for i in range(length)]
```

### 第 2 步：训练一个微型 token 预测器

一个以风格为条件的大二元模型预测器。重点在于这个模式：编解码器 token → 交叉熵训练 → 自回归采样。

### 第 3 步：条件采样

给定风格 token 和起始 token，从预测分布中采样下一个 token。继续采样 20-40 个 token。

## 陷阱

- **编解码器质量决定输出质量上限。** 如果编解码器无法忠实地表示声音，再好的生成器也无济于事。DAC 是当前开源最佳。
- **RVQ 误差累积。** 每个 RVQ 层建模前一个的残差。第 1 层的误差会传播。在高层上使用温度 0 采样有助于缓解。
- **音乐结构。** 30 秒的 token 在 75 Hz 下超过 20k 个 token。对 Transformer 来说很难。MusicGen 使用滑动窗口 + 提示延续；Stable Audio 使用更短片段 + 交叉淡入。
- **边界处的伪影。** 生成片段之间的交叉淡入需要仔细的重叠相加。
- **对干净数据的渴求。** 音乐生成器需要数万小时的授权音乐。Suno / Udio 的 RIAA 诉讼（2024 年）让这一问题浮出水面。
- **语音克隆的伦理问题。** 3 秒样本加文本提示足以让 VALL-E / XTTS / ElevenLabs 克隆声音。每个生产模型都需要滥用检测 + 选择退出列表。

## 使用它

| 任务 | 2026 技术栈 |
|------|------------|
| 商业 TTS | ElevenLabs、OpenAI TTS 或 Azure Neural |
| 语音克隆（已获同意验证） | XTTS v2（开源）或 ElevenLabs Pro |
| 背景音乐，快速 | Stable Audio 2.5 API、Suno 或 Udio |
| 带歌词的音乐 | Suno v4 或 Udio v1.5 |
| 音效 / Foley | AudioCraft 2、ElevenLabs SFX 或 Stable Audio Open |
| 实时语音代理 | GPT-4o 实时版或 Gemini Live |
| 开源音乐研究 | MusicGen 3.3B、Stable Audio Open 1.0、AudioLDM 2 |
| 配音 / 翻译 | HeyGen、ElevenLabs Dubbing |

## 交付它

保存 `outputs/skill-audio-brief.md`。Skill 接收音频简报（任务、时长、风格、声音、许可）并输出：模型 + 托管、提示格式（风格标签、风格描述符、结构标记）、编解码器 + 生成器 + 声码器链、随机种子协议和评估计划（MOS / CLAP 分数 / TTS 的 CER / 用户 A/B 测试）。

## 练习

1. **简单。** 运行 `code/main.py` 并显式设置风格。验证生成的序列是否符合该风格的模式。
2. **中等。** 添加延迟并行解码：模拟两个必须保持 1 步偏移的 token 流。训练一个联合预测器。
3. **困难。** 使用 HuggingFace transformers 在本地运行 MusicGen-small。用三个不同的提示生成 10 秒片段；A/B 测试风格 adherence。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 编解码器 (Codec) | "神经压缩" | 用于音频的编码器 / 解码器；典型输出是 50-75 Hz 的 token。 |
| RVQ | "残差 VQ" | K 个量化器的串级；每个建模前一个的残差。 |
| Token | "一个编解码器符号" | 码本中的离散索引；典型值为 1024 或 2048。 |
| 延迟并行 (Delayed parallel) | "偏移码本" | 发出 K 个带交错偏移的 token 流以缩短序列长度。 |
| 流匹配 (Flow matching) | "2024 年音频的胜出方案" | 比扩散更笔直的替代方案；采样更快。 |
| 语音提示 (Voice prompt) | "3 秒样本" | 说话人嵌入或 token 前缀，引导克隆声音的方向。 |
| Mel 频谱图 (Mel spectrogram) | "视觉" | 对数幅度感知频谱图；被许多 TTS 系统使用。 |
| 声码器 (Vocoder) | "Mel 转波形" | 将 Mel 频谱图转换回音频的神经组件。 |

## 生产注意事项：音频是一个流式问题

音频是用户期望在**生成的同时就到达**的唯一输出模态，而不是一次性全部到达。从生产角度来说，这意味着 TPOT 很重要（每个输出 token 的时间），因为用户的听力速度才是目标吞吐量——而不是阅读速度。对于以约 75 tokens/秒（Encodec）量化的 16kHz 音频，服务器必须为每个用户生成 ≥75 tokens/秒才能保持播放流畅。

两个架构后果：

- **流匹配音频模型无法简单流式传输。** Stable Audio 2.5 和 AudioCraft 2 一次渲染固定片段长度。要流式传输，需要将片段分块并在边界处重叠——想想滑动窗口扩散——相对于编解码器 AR 模型增加了 100-300ms 的延迟开销。

如果产品是"实时语音聊天"或"实时音乐延续"，选择编解码器 AR 路径。如果是"提交时渲染 30 秒片段"，流匹配在质量和总延迟上胜出。

## 延伸阅读

- [Défossez 等（2022）。Encodec：高质量神经音频压缩](https://arxiv.org/abs/2210.13438) — 编解码器标准。
- [Zeghidour 等（2021）。SoundStream](https://arxiv.org/abs/2107.03312) — 首个广泛使用的神经音频编解码器。
- [Kumar 等（2023）。使用改进的 RVQGAN 实现高保真音频压缩（DAC）](https://arxiv.org/abs/2306.06546) — DAC。
- [Wang 等（2023）。神经编解码器语言模型是零样本文本到语音合成器（VALL-E）](https://arxiv.org/abs/2301.02111) — VALL-E。
- [Copet 等（2023）。简单可控的音乐生成（MusicGen）](https://arxiv.org/abs/2306.05284) — MusicGen。
- [Liu 等（2023）。AudioLDM 2：通过自监督预训练学习整体音频生成](https://arxiv.org/abs/2308.05734) — AudioLDM 2。
- [Stability AI（2024）。Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) — 2025 年使用流匹配的文本到音乐。