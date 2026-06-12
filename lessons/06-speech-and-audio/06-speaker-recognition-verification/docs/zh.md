# 说话人识别与验证

> ASR 问"他们说了什么？"说话人识别问"是谁说的？"数学看起来一样——embedding 加余弦——但每一个生产决策都取决于一个单一的 EER 数字。

**类型：** 动手构建
**语言：** Python
**前置条件：** 阶段 6 · 02（频谱图与 Mel）、阶段 5 · 22（Embedding 模型）
**时间：** 约 45 分钟

## 问题

用户说了一句暗语。你想知道：这是他们声称的那个人吗（*验证*，1:1），还是注册库中的第一个人（*识别*，1:N）？或者都不是——这是一个未知说话人（*开放集*）？

2018 年之前：GMM-UBM + i-vector。EER 尚可，但对通道偏移（手机 vs 笔记本）和情绪敏感。2018–2022：x-vector（用角距训练的 TDNN 主干）。2022+：ECAPA-TDNN 和 WavLM-large embedding。到 2026 年该领域被三个模型和一个指标主导。

这个指标是 **EER**——等错误率。设置你的决策阈值，使得误接受率 = 误拒绝率。交叉点是 EER。用于每篇论文、每个排行榜、每次采购电话。

## 概念

![注册 + 验证管道：embedding + 余弦 + EER](../assets/speaker-verification.svg)

**管道。** 注册：录制目标说话人 5–30 秒；计算固定维度 embedding（ECAPA-TDNN 为 192 维，WavLM-large 为 256 维）。验证：获取测试话语的 embedding；计算余弦相似度；与阈值比较。

**ECAPA-TDNN（2020，至今仍占主导 2026）。** 强调通道注意力、传播与聚合——时延神经网络。1D 卷积块 + 压缩激励、多头注意力池化，最后是线性层映射到 192 维。在 VoxCeleb 1+2（2700 说话人，110 万话语）上训练，使用加性角距损失（AAM-softmax）。

**WavLM-SV（2022+）。** 用 AAM 损失微调预训练 WavLM-large SSL 主干。质量更高但更慢——300+ MB vs 15 MB。

**x-vector（基线）。** TDNN + 统计池化。经典；仍在 CPU / 边缘有用。

**AAM-softmax。** 标准 softmax，在角空间增加间隔 `m`：正确类别的 `cos(θ + m)`。强制类间角距分离。典型值 `m=0.2`，缩放 `s=30`。

### 评分

- **余弦** 在注册 embedding 和测试 embedding 之间。基于阈值的决策。
- **PLDA（概率线性判别分析）。** 将 embedding 投影到潜在空间，在其中同说话人 vs 不同说话人具有封闭形式的似然比。在余弦之上加上 +10–20% EER 降低。2020 年前的标准；现在仅用于封闭集设置。
- **分数归一化。** `S-norm` 或 `AS-norm`：将每个分数相对于一组成员冒充者的均值和标准差进行归一化。跨域评估的必需品。

### 你应该知道的数字（2026）

| 模型 | VoxCeleb1-O EER | 参数 | 吞吐量（A100） |
|-------|-----------------|--------|-------------------|
| x-vector（经典） | 3.10% | 5 M | 400× 实时 |
| ECAPA-TDNN | 0.87% | 15 M | 200× 实时 |
| WavLM-SV large | 0.42% | 316 M | 20× 实时 |
| Pyannote 3.1 分割 + embedding | 0.65% | 6 M | 100× 实时 |
| ReDimNet（2024） | 0.39% | 24 M | 100× 实时 |

### 说话人分离

多说话人片段中"谁在何时说话"。管道：VAD → 分割 → 每个片段 embed → 聚类（凝聚或谱聚类）→ 平滑边界。现代栈：`pyannote.audio` 3.1，它将说话人分割 + embedding + 聚类打包在一个调用背后。2026 年 AMI 上 SOTA DER 约 15%（从 2022 年的 23% 下降）。

## 动手构建

### 第 1 步：从 MFCC 统计的玩具 embedding

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26 维
```

离 SOTA 差得远——仅用于教学。`code/main.py` 在综合说话人数据上用这个作为概念验证。

### 第 2 步：余弦相似度 + 阈值

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### 第 3 步：从相似度对计算 EER

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

返回 (eer, threshold_at_eer)。两者都要报告。

### 第 4 步：使用 SpeechBrain 生产

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# 注册：平均 3-5 个干净样本的 embedding
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# 验证
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA 典型阈值；在你的数据上调优
```

### 第 5 步：使用 pyannote 进行说话人分离

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 使用

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| 封闭集 1:1 验证、边缘 | ECAPA-TDNN + 余弦阈值 |
| 开放集验证、云端 | WavLM-SV + AS-norm |
| 说话人分离（会议、播客） | `pyannote/speaker-diarization-3.1` |
| 反欺骗（重放 / 深度伪造检测） | AASIST 或 RawNet2 |
| 微型嵌入式（KWS + 注册） | Titanet-Small（NeMo） |

## 坑

- **通道不匹配。** 在 VoxCeleb（网络视频）上训练的模型 ≠ 电话音频。始终在目标通道上评估。
- **短话语。** 测试音频低于 3 秒时 EER 急剧下降。
- **带噪声的注册。** 一个噪声注册污染锚点。使用 ≥3 个干净样本并平均。
- **跨条件固定阈值。** 始终在来自目标领域的留出开发集上调优阈值。
- **对非归一化 embedding 使用余弦。** 先做 L2 归一化；否则幅度占主导。

## 交付

保存为 `outputs/skill-speaker-verifier.md`。选择模型、注册协议、阈值调优计划和欺诈防护措施。

## 练习

1. **简单。** 运行 `code/main.py`。构建综合"说话人"（不同音调），注册，在 100 对试验列表上计算 EER。
2. **中等。** 在 30 个 VoxCeleb1 话语（5 说话人 × 6 个）上使用 SpeechBrain ECAPA。用余弦 vs PLDA 计算 EER。
3. **困难。** 构建完整的注册 → 说话人分离 → 验证管道，使用 `pyannote.audio`。在 AMI 开发集上评估 DER。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| EER | 标题指标 | 误接受 = 误拒绝的阈值。 |
| 验证 | 1:1 | "这是 Alice 吗？" |
| 识别 | 1:N | "是谁在说话？" |
| 开放集 | 可能未知 | 测试集可包含未注册说话人。 |
| 注册 | 注册 | 计算说话人的参考 embedding。 |
| AAM-softmax | 损失函数 | 带加性角距的 softmax；强制簇分离。 |
| PLDA | 经典评分 | 概率线性判别分析；在 embedding 之上的似然比评分。 |
| DER | 说话人分离指标 | 说话人分离错误率——漏检 + 误报 + 混淆。 |

## 延伸阅读

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — 经典深度 embedding 论文。
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — 2020–2026 年占主导的架构。
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) — SV 和说话人分离的 SSL 主干。
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — 生产说话人分离 + embedding + 聚类栈。
- [VoxCeleb 排行榜（2026 年更新）](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — 各模型当前 EER 排名。