# 评估 — FID、CLIP 分数与人类偏好

> 每一个人写生成模型榜单都会引用 FID、CLIP 分数以及人类偏好 arena 中的胜率。每一个数字都有其固有缺陷——一个执意"刷分"的研究者总能找到漏洞。如果你不知道这些缺陷在哪里，就无法分辨真正的改进和单纯的刷分。

**类型：** 动手构建
**语言：** Python
**前置条件：** 阶段 8 · 01（分类体系）、阶段 2 · 04（评估指标）
**时间：** 约 45 分钟

## 问题

生成模型的评判有两个维度：*样本质量*和*条件一致性*。两者都没有闭合形式的度量。你的模型需要渲染 10,000 张图像；必须有人给它们打分；你还需要在不同的模型家族、不同的分辨率、不同的架构之间相信这些数字。2014-2026 年间有三个指标经受住了考验：

- **FID（Fréchet Inception 距离）。** 在 Inception 网络特征空间中，两个分布（真实分布与生成分布）之间的距离。越低越好。
- **CLIP 分数。** 生成图像的 CLIP-image embedding 与提示词的 CLIP-text embedding 之间的余弦相似度。越高越好。衡量提示词遵循程度。
- **人类偏好。** 让两个模型在相同提示词上正面交锋，由人类（或 GPT-4 级别的模型）选出更好的那个，汇总为 Elo 分数。

你还会看到：IS（Inception Score，已基本退役）、KID、CMMD、ImageReward、PickScore、HPSv2、MJHQ-30k。每一个都是为了修正前一个的某个缺陷而出现的。

## 概念

![FID、CLIP 与偏好：三个维度，不同的缺陷](../assets/evaluation.svg)

### FID — 样本质量

Heusel 等人（2017）。步骤：

1. 对 N 张真实图像和 N 张生成图像提取 Inception-v3 特征（2048 维）。
2. 对每个池拟合一个高斯分布：计算均值 `μ_r, μ_g` 和协方差 `Σ_r, Σ_g`。
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`。

解读：特征空间中两个多元高斯分布之间的 Fréchet 距离。越低 = 分布越相似。

缺陷：
- **在小 N 上有偏。** FID 是特征分布的均方误差——小 N 会低估协方差，给出虚假偏低的 FID。务必使用 N ≥ 10,000。
- **依赖 Inception。** Inception-v3 是在 ImageNet 上训练的。与 ImageNet 差异大的领域（人脸、艺术、文字图像）会产生无意义的 FID。使用领域特定的特征提取器。
- **可被刷分。** 对 Inception 先验过拟合会给出低 FID 而没有视觉质量提升。用 CMMD 来对抗（见下文）。

### CLIP 分数 — 提示词遵循

Radford 等人（2021）。对于生成图像 + 提示词：

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

在 30k 张生成图像上取平均 → 一个可在模型间比较的标量。

缺陷：
- **CLIP 自身的盲点。** CLIP 的组合推理能力较弱（"红色立方体在蓝色球体上"经常出错）。模型可以在 CLIP 分数上排名很高，却没有真正遵循复杂提示词。
- **短提示词偏差。** 短提示词在野外有更多 CLIP-image 匹配。长提示词在机制上 CLIP 分数更低。
- **提示词刷分。** 在提示词中加入"高质量、4k、杰作"会推高 CLIP 分数而不改善图文绑定。

CMMD（Jayasumana 等人，2024）修复了其中一些问题：使用 CLIP 特征而非 Inception，最大均值差异而非 Fréchet。更擅长检测细微的质量差异。

### 人类偏好 — 地面真值

选择一个提示词池。用模型 A 和模型 B 分别生成。将配对结果展示给人类（或强 LLM 评判者）。将胜局汇总为 Elo 或 Bradley-Terry 分数。基准测试：

- **PartiPrompts（Google）**：1,600 个多样化提示词，12 个类别。
- **HPSv2**：107k 个人类标注，广泛用作自动化代理。
- **ImageReward**：137k 个提示词-图像偏好配对，MIT 许可。
- **PickScore**：在 Pick-a-Pic 2.6M 偏好上训练而来。
- **Chatbot-Arena 风格图像 arena**：https://imagearena.ai/ 等。

缺陷：
- **评判者差异。** 非专家与专家的偏好不同。两种都要用。
- **提示词分布。** 精挑细选的提示词会偏向某个家族。务必记录。
- **LLM 评判者奖励黑客。** GPT-4 评判者会被好看但错误的结果欺骗。用人类三角验证。

## 联合使用

一份生产评估报告应包含：

1. 在 10-30k 样本上相对于留出真实分布的 FID（样本质量）。
2. 在相同样本相对于其提示词的 CLIP 分数 / CMMD（遵循度）。
3. 在盲测 arena 中相对于上一模型的胜率（整体偏好）。
4. 缺陷分析：随机采样 50 个输出，标记已知问题（手部解剖、文字渲染、物体数量一致性）。

任何一个单独指标都是谎言。三个相互印证的指标 + 定性审查才是一个可信的结论。

## 动手构建

`code/main.py` 在合成"特征向量"（我们用 4 维向量作为 Inception 特征的替代品）上实现了 FID、类 CLIP 分数和 Elo 汇总。你会看到：

- 在小 N 和大 N 上计算 FID——偏差。
- "CLIP 分数"作为特征池之间的余弦相似度。
- 来自合成偏好流的 Elo 更新规则。

### 第 1 步：四行代码实现 FID

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### 第 2 步：类 CLIP 余弦相似度

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### 第 3 步：Elo 汇总

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## 陷阱

- **N=1000 时的 FID。** N<10k 时启发式方法不可靠。报告低 N FID 的论文在刷分。
- **跨分辨率比较 FID。** Inception 的 299×299 resize 会改变特征分布。只能在匹配分辨率下比较。
- **只报告一个 seed。** 至少运行 3 个 seed。报告标准差。
- **通过负提示词推高 CLIP 分数。** 一些 pipeline 通过过拟合提示词来提升 CLIP。检查视觉饱和度。
- **提示词重叠带来的 Elo 偏差。** 如果两个模型在训练期间都见过基准提示词，Elo 就毫无意义。使用留出的提示词集。
- **人类评估付费众包偏移。** Prolific、MTurk 标注者偏向更年轻/更熟悉技术的人群。与招募的艺术/设计专家混合。

## 使用方法

2026 年的生产评估协议：

| 维度 | 最低要求 | 推荐 |
|--------|---------|-------------|
| 样本质量 | 10k vs 留出真实的 FID | + 5k 上 CMMD + 每类别子集 FID |
| 提示词遵循 | 30k 上 CLIP 分数 | + HPSv2 + ImageReward + VQA 风格问答 |
| 偏好 | 200 对盲测 vs 基线 | + 2000 对配对人类 + LLM 评判 + Chatbot Arena |
| 缺陷分析 | 50 个手动标记 | 500 个手动标记 + 自动化安全分类器 |

一份报告包含全部四个维度 = 可信的主张。只用一个 = 营销。

## 交付

保存 `outputs/skill-eval-report.md`。Skill 接收新的模型 checkpoint + 基线，输出完整的评估计划：样本量、指标、缺陷探测、签字标准。

## 练习

1. **简单。** 运行 `code/main.py`。在同一合成分布上比较 N=100 和 N=1000 的 FID。报告偏差幅度。
2. **中等。** 从合成类 CLIP 特征实现 CMMD（参见 Jayasumana 等人，2024 的公式）。比较其对质量差异的敏感度与 FID 的对比。
3. **困难。** 复现 HPSv2 设置：从 Pick-a-Pic 子集中取 1000 个图像-提示词配对，在偏好上微调一个小型 CLIP 评分器，测量其与留出集的一致性。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| FID | "Fréchet Inception 距离" | 真实 vs 生成 Inception 特征的高斯拟合之间的 Fréchet 距离。 |
| CLIP 分数 | "文本-图像相似度" | CLIP 图像和文本 embedding 之间的余弦相似度。 |
| CMMD | "FID 的替代品" | CLIP 特征 MMD；偏差更小，无需高斯假设。 |
| IS | "Inception Score" | Exp KL(p(y|x) || p(y))；在现代模型上相关性差，已退役。 |
| HPSv2 / ImageReward / PickScore | "习得偏好代理" | 在人类偏好上训练的小模型；用作自动评判者。 |
| Elo | "国际象棋评级" | 成对胜局的 Bradley-Terry 汇总。 |
| PartiPrompts | "基准提示词集" | 来自 12 个类别的 1,600 个 Google 策划提示词。 |
| FD-DINO | "自监督替代品" | 使用 DINOv2 特征的 FD；更适合 ImageNet 之外的领域。 |

## 生产笔记：评估本身也是一种推理负载

在 10k 样本上运行 FID 意味着生成 10k 张图像。对于单块 L4 上的 50 步 SDXL base at 1024²，这大约是 ~11 小时的单请求推理。评估预算是真实存在的，而且场景正是离线推理场景（最大化吞吐量，忽略 TTFT）：

- **批量难，忘记延迟。** 离线评估 = 在内存允许的最大尺寸上进行静态批处理。在 80GB H100 上用 `pipe(...).images` 配合 `num_images_per_prompt=8` 运行，比单请求快 4-6 倍。
- **缓存真实特征。** Inception（FID）或 CLIP（CLIP 分数、CMMD）对真实参考集的特征提取只运行一次，存储为 `.npz`。不要每次评估都重新计算。

对于 CI / 回归门禁：每个 PR 在 500 样本子集上运行 FID + CLIP 分数（~30 分钟）；每晚运行完整的 10k FID + HPSv2 + Elo。

## 延伸阅读

- [Heusel 等人（2017）. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — FID 论文。
- [Jayasumana 等人（2024）. Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD。
- [Radford 等人（2021）. Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP。
- [Wu 等人（2023）. HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2。
- [Xu 等人（2023）. ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward。
- [Yu 等人（2023）. Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts。
- [Stein 等人（2023）. Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — 缺陷调查。
