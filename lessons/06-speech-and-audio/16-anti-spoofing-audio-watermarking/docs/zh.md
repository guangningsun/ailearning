# 语音反欺骗与音频水印 — ASVspoof 5、AudioSeal、WaveVerify

> 语音克隆跑得比防御更快。2026 年生产级语音系统需要两样东西：一个检测器（AASIST、RawNet2）用于判别真实语音与合成语音，以及一个水印（AudioSeal）能在压缩和编辑后存活。同时上线，否则不要上线语音克隆。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 6 · 06（说话人识别）、阶段 6 · 08（语音克隆）
**时间：** 约 75 分钟

## 问题

三类相关防御：

1. **反欺骗 / 深度伪造检测。** 给定一段音频，判断它是合成语音还是真实语音？ASVspoof 基准测试（ASVspoof 2019 → 2021 → 5）是行业标准。
2. **音频水印。** 在生成的音频中嵌入人耳不可感知的信号，后续检测器可提取。AudioSeal（Meta）和 WavMark 是开源方案。
3. **认证溯源。** 音频文件和元数据的密码学签名。C2PA / 内容真实性倡议。

检测对付不配合的对手。水印处理合规 — AI 生成的音频应可被识别。2026 年两者缺一不可。

## 概念

![反欺骗 vs 水印 vs 溯源 — 三层防御](../assets/spoofing-watermark.svg)

### ASVspoof 5 — 2024-2025 基准测试

相比前几版的主要变化：

- **众包数据**（非录音棚干净数据）— 更真实的条件。
- **约 2000 名说话人**（之前约 100 名）。
- **32 种攻击算法。** TTS + 语音转换 + 对抗扰动。
- **两个赛道。**  countermeasures（CM）独立检测；Spoofing-robust ASV（SASV）用于生物识别系统。

ASVspoof 5 当前最优水平：约 7.23% EER。在较老的 ASVspoof 2019 LA 上：0.42% EER。实际部署：在野外音频片段上预期 5-10% EER。

### AASIST 和 RawNet2 — 检测模型系列

**AASIST**（2021，2026 年持续更新）。图注意力机制处理频谱特征。ASVspoof 5 countermeasures 任务的当前最优模型。

**RawNet2。** 卷积前端处理原始波形 + TDNN 主干。更简单的基线；微调后仍有竞争力。

**NeXt-TDNN + SSL 特征。** 2025 年变体：ECAPA 风格 + WavLM 特征 + Focal Loss。在 ASVspoof 2019 LA 上达到 0.42% EER。

### AudioSeal — 2024 年水印默认方案

Meta 的 **AudioSeal**（2024 年 1 月，v0.2 2024 年 12 月）。核心设计：

- **局部化检测。** 以 16 kHz 采样分辨率（1/16000 秒）逐帧检测水印。
- **生成器和检测器联合训练。** 生成器学习嵌入不可感知信号；检测器学习在各种数据增强中找到它。
- **鲁棒。** 经受 MP3 / AAC 压缩、EQ、±10% 变速、+10 dB SNR 噪声混合。
- **快速。** 检测器以 485 倍实时速度运行；比 WavMark 快 1000 倍。
- **容量。** 16 位载荷（可在每次话语中嵌入模型 ID、生成时间戳、用户 ID）。

### WavMark

AudioSeal 之前的开源基线。可逆神经网络，32 比特/秒。问题：

- 同步暴力搜索速度慢。
- 可被高斯噪声或 MP3 压缩移除。
- 不友好实时处理。

### WaveVerify（2025 年 7 月）

解决 AudioSeal 的弱点 — 特别针对时间操作（倒放、变速）。使用基于 FiLM 的生成器 + 专家混合检测器。在标准攻击上与 AudioSeal 相当；能处理时间编辑。

### 对手利用的差距

来自 AudioMarkBench："在音高偏移下，所有水印的比特恢复准确率均低于 0.6，表明几乎被完全移除。"**音高偏移是通用攻击。** 2026 年没有水印能完全抵抗激进的音高修改。这就是为什么需要检测（AASIST）配合水印。

### C2PA / 内容真实性倡议

不是机器学习技术 — 而是一种清单格式。音频文件携带关于创建工具、作者、日期的密码学签名元数据。Audobox / Seamless 使用它。有利于溯源；但如果恶意行为者重新编码并剥离元数据则无效。

## 动手构建

### 第 1 步：简单的频谱特征检测器（玩具级）

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

合成语音通常有异常平坦的高频能量。生产检测器用 AASIST，不是这个。但直觉是对的。

### 第 2 步：AudioSeal 嵌入 + 检测

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: [0, 1] 区间浮点数 — 水印存在的概率
# decoded_payload: 16 比特；与嵌入的载荷比对
```

### 第 3 步：评估 — EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### 第 4 步：生产环境集成

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

每一次生成都输出：(1) 水印，(2) 签名清单，(3) 符合保留策略的审计日志。

## 使用场景

| 使用场景 | 防御手段 |
|----------|----------|
| 上线 TTS / 语音克隆 | 每个输出都嵌入 AudioSeal（非协商项） |
| 生物识别语音解锁 | AASIST + ECAPA 集成；活性检测 |
| 呼叫中心欺诈检测 | 对 20% 来电样本进行 AASIST 检测 |
| 播客真实性 | 上传时 C2PA 签名，AI 生成则加 AudioSeal |
| 研究 / 训练检测器 | ASVspoof 5 train/dev/eval 集合 |

## 陷阱

- **水印嵌入后从未运行检测器。** 毫无意义。在 CI 中上线检测器。
- **检测器未校准。** AASIST 在 ASVspoof LA 上训练会过拟合；现实世界准确率下降。在你的领域数据上校准。
- **音高偏移差距。** 激进的音高偏移能移除大多数水印。准备检测备选方案。
- **元数据剥离重传。** C2PA 可被重新编码轻易绕过。始终将密码学防御 + 感知防御（水印）结合使用。
- **用检测代替活性。** 让用户说一个随机短语。能防止重放攻击但不能防止实时克隆。

## 上线

保存为 `outputs/skill-spoof-defender.md`。为语音生成部署选择检测模型、水印、溯源清单和操作手册。

## 练习

1. **简单。** 运行 `code/main.py`。玩具检测器 + 玩具水印嵌入/检测，处理合成音频。
2. **中等。** 安装 `audioseal`，在 TTS 输出中嵌入 16 位载荷，重新解码。用噪声破坏音频，测量比特恢复准确率。
3. **困难。** 在 ASVspoof 2019 LA 上微调 RawNet2 或 AASIST。测量 EER。在留出的 F5-TTS 生成片段上测试 — 观察域外检测如何退化。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|-----------------------|
| ASVspoof | 基准测试 | 两年一度的挑战赛；2024 年 = ASVspoof 5。 |
| CM（countermeasure） | 检测器 | 分类器：真实语音 vs 合成/转换语音。 |
| SASV | 说话人验证 + CM | 集成的生物识别 + 欺骗检测。 |
| AudioSeal | Meta 水印 | 局部化、16 位载荷、比 WavMark 快 485 倍。 |
| Bit Recovery Accuracy | 水印存活率 | 攻击后恢复的载荷比特比例。 |
| C2PA | 溯源清单 | 关于创作/ authorship 的密码学元数据。 |
| AASIST | 检测器系列 | 基于图注意力的反欺骗当前最优模型。 |

## 延伸阅读

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — 当前基准测试。
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — 水印默认方案。
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — 用于时间攻击的 MoE 检测器。
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — 当前最优检测主干。
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — 鲁棒性评估。
- [C2PA 规范](https://c2pa.org/specifications/specifications/) — 溯源清单格式。