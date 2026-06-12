# 音乐生成 —— MusicGen、Stable Audio、Suno，以及许可证地震

> 2026 年音乐生成：Suno v5 和 Udio v4 主导商业领域；MusicGen、Stable Audio Open 和 ACE-Step 主导开源。技术问题基本已解决。法律问题（华纳音乐 5 亿美元和解、UMG 和解）在 2025–2026 年重塑了整个领域。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 6 · 02（频谱图）、阶段 4 · 10（扩散模型）
**时间：** 约 75 分钟

## 问题

文本 → 30 秒到 4 分钟的音乐片段，带歌词、人声和结构。三个子问题：

1. **器乐生成。** "低保真嘻哈鼓点配温暖键盘"类文本 → 音频。MusicGen、Stable Audio、AudioLDM。
2. **歌曲生成（带人声 + 歌词）。** "关于德克萨斯雨夜的乡村歌曲" → 完整歌曲。Suno、Udio、YuE、ACE-Step。
3. **条件可控生成。** 扩展现有片段、重生成桥段、切换风格、分轨或局部填充。Udio 的局部填充 + 分轨是 2026 年必须对标的功能。

## 概念

![音乐生成：token-LM vs 扩散，2026 年模型地图](../assets/music-generation.svg)

### 基于神经编解码器 token 的语言模型

Meta 的 **MusicGen**（2023，MIT）及众多衍生模型：以文本/旋律 embedding 为条件，自回归预测 EnCodec token（32 kHz，4 个码本），用 EnCodec 解码。3 亿–33 亿参数。强 baseline；超过 30 秒容易吃力。

**ACE-Step**（开源，2026 年 4 月发布 40 亿 XL）将此扩展为完整歌曲的歌词条件生成。开源社区最接近 Suno 的方案。

### 在 mel 或隐空间上扩散

**Stable Audio（2023）** 和 **Stable Audio Open（2024）**：在压缩音频上做隐扩散。擅长循环、音效设计、环境纹理。不擅长结构完整的歌曲。

**AudioLDM / AudioLDM2**：通过 T2I 风格的隐扩散做文本转音频，泛化到音乐、音效、语音。

### 混合（生产）—— Suno、Udio、Lyria

闭源。很可能是 AR 编解码器 LM + 基于扩散的声码器，带专用的人声/鼓/旋律 head。Suno v5（2026）以 ELO 1293 质量领先。Udio v4 新增局部填充 + 分轨（贝斯、鼓、人声可分别下载）。

### 评估方法

- **FAD（Fréchet 音频距离）。** 用 VGGish 或 PANNs 特征计算生成音频与真实音频分布在 embedding 层面的距离。越低越好。MusicGen small：MusicCaps 上 4.5 FAD；SOTA 约 3.0。
- **音乐性（主观）。** 人类偏好。Suno v5 ELO 1293 领先。
- **文本-音频对齐。** prompt 与输出之间的 CLAP 分数。
- **音乐性伪影。** 拍子错位、人声漂移、超过 30 秒后结构丢失。

## 2026 年模型地图

| 模型 | 参数量 | 时长 | 人声 | 许可证 |
|------|--------|------|------|--------|
| MusicGen-large | 33 亿 | 30 秒 | 无 | MIT |
| Stable Audio Open | 12 亿 | 47 秒 | 无 | Stability 非商业 |
| ACE-Step XL（2026 年 4 月） | 40 亿 | > 2 分钟 | 有 | Apache-2.0 |
| YuE | 70 亿 | > 2 分钟 | 有，多语言 | Apache-2.0 |
| Suno v5（闭源） | ？ | 4 分钟 | 有，ELO 1293 | 商业 |
| Udio v4（闭源） | ？ | 4 分钟 | 有 + 分轨 | 商业 |
| Google Lyria 3（闭源） | ？ | 实时 | 有 | 商业 |
| MiniMax Music 2.5 | ？ | 4 分钟 | 有 | 商业 API |

## 法律环境（2025–2026）

- **华纳音乐诉 Suno 和解。** 5 亿美元。WMG 现在对 Suno 上的 AI 相似性、音乐版权和用户生成曲目拥有监督权。Udio 有类似的 UMG 和解。
- **欧盟 AI Act** + **加州 SB 942**：AI 生成音乐必须披露。
- **Riffusion / MusicGen** 以 MIT 授权，无合规负担，但也没有商业人声。

可安全出货的模式：

1. 仅生成器乐（MusicGen、Stable Audio Open，MIT/CC0 输出）。
2. 使用商业 API（Suno、Udio、ElevenLabs Music）并取得逐生成授权。
3. 在自有或授权目录上训练（大多数企业最终走到这一步）。
4. 用水印 + 元数据标记生成内容。

## 构建

### 第 1 步：用 MusicGen 生成

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

三个规模：`small`（3 亿，快速）、`medium`（15 亿）、`large`（33 亿）。small 足够用来验证想法是否落地。

### 第 2 步：旋律条件

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody 接收色图并保留旋律同时更换音色。适用于"给我这个旋律的弦乐四重奏版"。

### 第 3 步：FAD 评估

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

计算 VGGish embedding 距离。对风格级回归测试有用；不能替代人类听众。

### 第 4 步：加入 LLM-音乐工作流

结合第 7–8 课的想法：

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 使用

| 目标 | 技术栈 |
|------|-------|
| 器乐音效设计 | Stable Audio Open |
| 游戏/自适应音乐 | Google Lyria RealTime（闭源） |
| 带人声完整歌曲（商业） | Suno v5 或 Udio v4，含明确授权 |
| 带人声完整歌曲（开源） | ACE-Step XL 或 YuE |
| 短视频广告配乐 | MusicGen 旋律条件（哼唱参考） |
| 音乐视频背景 | MusicGen + Stable Video Diffusion |

## 2026 年仍在踩的坑

- **版权洗白 prompt。** "Song in the style of Taylor Swift" —— 商业 Suno/Udio 现在能过滤这些，开源模型不能。自己加过滤列表。
- **超过 30 秒后重复/漂移。** AR 模型会循环。交叉淡入多个生成结果，或用 ACE-Step 保持结构连贯。
- **速度漂移。** 模型 BPM 容易跑偏。在 prompt 中用 BPM 标签，用 librosa 的 `beat_track` 后处理。
- **人声清晰度。** Suno 很强；开源模型人声往往含混。如果歌词重要，用商业 API 或微调。
- **单声道输出。** 开源模型生成单声道或假立体声。用专业立体声重建（ezst、Cartesia 的立体声扩散）升级。

## 交付

保存为 `outputs/skill-music-designer.md`。为音乐生成部署选择模型、许可证策略、时长/结构规划及披露元数据。

## 练习

1. **简单。** 运行 `code/main.py`。生成一个"生成式"和弦进行 + 鼓点，用 ASCII 符号表示 —— 音乐生成的卡通版。想用 MIDI 渲染器回放也行。
2. **中等。** 安装 `audiocraft`，用 MusicGen-small 在 4 种风格 prompt 上各生成 10 秒片段，用 FAD 对比参考风格集。
3. **困难。** 用 ACE-Step（或 MusicGen-melody）用不同音色 prompt 生成同一曲目的三个变体。计算 CLAP 相似度验证对齐。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| FAD | 音频 FID | 真实 vs 生成音频 embedding 分布之间的 Fréchet 距离。 |
| 色图 (Chromagram) | 旋律即音高 | 每帧 12 维向量；旋律条件的输入。 |
| 分轨 (Stems) | 乐器音轨 | 贝斯/鼓/人声/旋律分离开的 WAV。 |
| 局部填充 (Inpainting) | 重生成一个段落 | 遮罩时间窗口；模型只重生成该部分。 |
| CLAP | 文本-音频 CLIP | 对比音频-文本 embedding；评估文本-音频对齐。 |
| EnCodec | 音乐编解码器 | Meta 的神经编解码器，MusicGen 使用；32 kHz，4 个码本。 |

## 延伸阅读

- [Copet 等 (2023). MusicGen](https://arxiv.org/abs/2306.05284) — 开源自回归 benchmark。
- [Evans 等 (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — 音效设计默认选择。
- [ACE-Step](https://github.com/ace-step/ACE-Step) — 开源 40 亿完整歌曲生成器，2026 年 4 月。
- [Suno v5 平台文档](https://suno.com) — 商业质量领先者。
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — 音乐 + 音效的隐扩散。
- [WMG-Suno 和解报道](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) — 2025 年 11 月先例。