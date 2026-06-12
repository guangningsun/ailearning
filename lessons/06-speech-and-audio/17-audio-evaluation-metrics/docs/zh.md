# 音频评估 — WER、 MOS、UTMOS、MMAU、FAD 及公开榜单

> 无法测量就无法上线。本课列出 2026 年各音频任务的指标：ASR（WER、CER、RTFx）、TTS（MOS、UTMOS、SECS、WER-on-ASR 往返）、音频语言模型（MMAU、LongAudioBench）、音乐（FAD、CLAP）和说话人（EER）。以及用来比较的公开榜单。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 6 · 04、06、07、09、10；阶段 2 · 09（模型评估）
**时间：** 约 60 分钟

## 问题

每个音频任务都有多个指标，每个指标测量不同的维度。用错指标会导致你在仪表盘上看起来很好、在生产环境中却很糟糕。2026 年权威清单：

| 任务 | 主要指标 | 次要指标 |
|------|---------|-----------|
| ASR | WER | CER · RTFx · 首个 token 延迟 |
| TTS | MOS / UTMOS | SECS · WER-on-ASR 往返 · CER · TTFA |
| 语音克隆 | SECS（ECAPA 余弦） | MOS · CER |
| 说话人验证 | EER | minDCF · 特定工作点的 FAR / FRR |
|  diarization | DER | JER · 说话人混淆 |
| 音频分类 | top-1 · mAP | macro F1 · 每类召回率 |
| 音乐生成 | FAD | CLAP · 听评团 MOS |
| 音频语言模型 | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| 流式 S2S | 延迟 P50/P95 | WER · MOS |

## 概念

![音频评估矩阵 — 指标 vs 任务 vs 2026 榜单](../assets/eval-landscape.svg)

### ASR 指标

**WER（词错误率）。** `(S + D + I) / N`。小写化，剥除标点，计分前标准化数字。使用 `jiwer` 或 OpenAI 的 `whisper_normalizer`。< 5% = 人类水平的朗读语音。

**CER（字符错误率）。** 相同公式，字符级。用于音调语言（普通话、粤语），在这些语言中分词存在歧义。

**RTFx（实时因子的倒数）。** 每壁钟秒处理的音频秒数。越高越好。Parakeet-TDT 达到 3380×。Whisper-large-v3 约 30×。

**首个 token 延迟。** 从音频输入到第一个转录 token 的壁钟时间。对流式处理至关重要。Deepgram Nova-3：约 150 毫秒。

### TTS 指标

**MOS（平均意见得分）。** 1-5 人类评分。黄金标准但速度慢。每样本收集 20+ 位听众，每模型 100+ 样本。

**UTMOS（2022-2026）。** 学习型 MOS 预测器。在标准基准上与人类 MOS 相关性约 0.9。F5-TTS：UTMOS 3.95；真实值：4.08。

**SECS（说话人编码器余弦相似度）。** 用于语音克隆。参考音频和克隆输出之间的 ECAPA 嵌入余弦。> 0.75 = 可识别的克隆。

**WER-on-ASR 往返。** 在 TTS 输出上运行 Whisper，计算与输入文本的 WER。能捕捉可懂度的退化。2026 年最优水平：< 2% CER。

**TTFA（首音频时间）。** 壁钟延迟。Kokoro-82M：约 100 毫秒；F5-TTS：约 1 秒。

### 语音克隆专用

**SECS + MOS + CER** 三合一。高 SECS 但低 MOS 意味着音色对但不自然；反过来则意味着自然但说错了人。

### 说话人验证

**EER（等错误率）。** 误接受率等于误拒绝率的阈值。ECAPA 在 VoxCeleb1-O 上：0.87%。

**minDCF（最小检测成本）。** 特定工作点（通常 FAR=0.01）的加权成本。比 EER 更适合生产环境。

### Diarization

**DER（ diarization 错误率）。** `(FA + Miss + Confusion) / total_speaker_time`。漏检语音 + 误报语音 + 说话人混淆，每项作为分数。AMI 会议：DER ~10-20% 是合理水平。pyannote 3.1 + Precision-2 商业版：在录音良好的音频上 <10% DER。

**JER（Jaccard 错误率）。** DER 的替代方案，对短片段偏差更鲁棒。

### 音频分类

多标签：**mAP（平均精度）** 覆盖所有类别。AudioSet：BEATs-iter3 为 0.548 mAP。

多分类互斥：**top-1、top-5 准确率**。Speech Commands v2：99.0% top-1（Audio-MAE）。

不平衡：**macro F1** + **每类召回率**。报告每类 — 汇总准确率会掩盖哪些类别失败。

### 音乐生成

**FAD（Fréchet 音频距离）。** 真实音频和生成音频的 VGGish 嵌入分布之间的距离。MusicGen-small 在 MusicCaps 上：4.5。MusicLM：4.0。越低越好。

**CLAP 分数。** 使用 CLAP 嵌入的文本-音频对齐分数。> 0.3 = 合理的对齐。

**听评团 MOS。** 消费级音乐最终的评价标准。Suno v5 ELO 1293 在 TTS Arena 上（来自配对人类偏好）。

### 音频语言基准

**MMAU（大规模多音频理解）。** 10k 音频问答对。

**MMAU-Pro。** 1800 难题项，四类：语音 / 声音 / 音乐 / 多音频。四选一随机机会 25%。Gemini 2.5 Pro 总体约 60%；多音频约 22%（所有模型）。

**LongAudioBench。** 带语义查询的多分钟音频片段。Audio Flamingo Next 超越 Gemini 2.5 Pro。

**AudioCaps / Clotho。** 字幕基准。SPICE、CIDEr、FENSE 指标。

### 流式语音到语音

**延迟 P50 / P95 / P99。** 从用户语音结束到首个可听回复的壁钟时间。Moshi：200 毫秒；GPT-4o Realtime：300 毫秒。

**输出上的 WER / MOS。**

**插话响应性。** 从用户打断到助手静音的时间。目标 < 150 毫秒。

### 2026 年榜单

| 榜单 | 追踪内容 | URL |
|------------|--------|-----|
| Open ASR Leaderboard（HF） | 英语 + 多语言 + 长音频 | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena（HF） | 英语 TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT，配对投票 ELO | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM 推理 | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | 说话人识别 | `voxsrc.github.io` |
| MMAU 音乐子集 | 音乐 LALM | （在 MMAU 内） |
| HEAR benchmark | 自监督音频 | `hearbenchmark.com` |

## 动手实现

### 第 1 步：带标准化的 WER

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### 第 2 步：TTS 往返 WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### 第 3 步：语音克隆的 SECS

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### 第 4 步：音乐生成的 FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### 第 5 步：说话人验证的 EER（第 6 课同款代码）

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## 使用场景

每次部署都配上固定的评估工具链，在每次模型更新时运行。三条核心规则：

1. **计分前标准化。** 小写化、剥除标点、数字展开。报告标准化规则。
2. **报告分布，不报均值。** 延迟用 P50/P95/P99。分类用每类召回率。MMAU 用每类别。
3. **跑一个权威公开基准。** 即使你的生产数据不同，在 Open ASR / TTS Arena / MMAU 上报告也能让评审 apples-to-apples 比较。

## 陷阱

- **UTMOS 外推。** 在 VCTK 风格干净语音上训练；对噪声/克隆/情感音频评分差。
- **MOS 评审团偏差。** 20 个 Amazon Mechanical Turk 工人 ≠ 20 个目标用户。如果风险高，花钱请领域评审团。
- **FAD 依赖参考集。** 在不同模型间用相同参考分布比较。
- **汇总 WER。** 总体 5% WER 可能隐藏方言口音上 30% WER。按人口统计切片报告。
- **公开基准饱和。** 大多数前沿模型在标准基准上接近上限。构建一个反映你流量的内部留出集。

## 上线

保存为 `outputs/skill-audio-evaluator.md`。为任何音频模型发布选择指标、基准和报告格式。

## 练习

1. **简单。** 运行 `code/main.py`。在玩具输入上计算 WER / CER / EER / SECS / FAD 类 / MMAU 类指标。
2. **中等。** 构建 TTS 往返 WER 工具链。将 Kokoro 或 F5-TTS 输出通过 Whisper。在 50 个提示上计算 WER。标记 WER > 10% 的提示。
3. **困难。** 在 MMAU-Pro 语音和多音频子集上评估你第 10 课 LALM 选择（各 50 项）。报告每类准确率并与已发表数字比较。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| WER | ASR 分数 | 标准化后词级的 `(S+D+I)/N`。 |
| CER | 字符 WER | 用于音调语言或字符级系统。 |
| MOS | 人类意见 | 1-5 评分；20+ 听众 × 100 样本。 |
| UTMOS | ML MOS 预测器 | 学习模型；与人类 MOS 相关性约 0.9。 |
| SECS | 语音克隆相似度 | 参考音频和克隆之间的 ECAPA 余弦。 |
| EER | 说话人验证分数 | FAR = FRR 的阈值。 |
| DER | Diarization 分数 | (FA + Miss + Confusion) / total。 |
| FAD | 音乐生成质量 | VGGish 嵌入上的 Fréchet 距离。 |
| RTFx | 吞吐量 | 每壁钟秒的音频秒数。 |

## 延伸阅读

- [jiwer](https://github.com/jitsi/jiwer) — 带标准化工具的 WER/CER 库。
- [UTMOS（Saeki et al. 2022）](https://arxiv.org/abs/2204.02152) — 学习型 MOS 预测器。
- [Fréchet Audio Distance（Kilgour et al. 2019）](https://arxiv.org/abs/1812.08466) — 音乐生成标准。
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 2026 年实时排名。
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — 人类投票 TTS 榜单。
- [MMAU-Pro 基准](https://mmaubenchmark.github.io/) — LALM 推理榜单。
- [HEAR benchmark](https://hearbenchmark.com/) — 音频 SSL 基准。