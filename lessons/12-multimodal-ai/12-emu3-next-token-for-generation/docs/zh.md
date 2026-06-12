# Emu3：图像和视频生成的 Next-Token 预测

> BAAI 的 Emu3（Wang 等，2024 年 9 月）是 2024 年的成果，本应终结扩散与自回归之间的争论。一个单一的类 Llama 仅解码器 transformer，仅在 next-token-prediction 目标上训练，跨文本 + VQ 图像 token + 3D VQ 视频 token 的统一词汇表，在图像生成上击败了 SDXL，在感知上击败了 LLaVA-1.6。没有 CLIP loss。没有扩散调度。推理时使用无分类器引导提升质量，但核心训练目标是使用教师强制的 next-token 预测。发表于 Nature。本课解读 Emu3 论点——为什么更好的分词器加上规模就是你需要的一切——并与扩散方法对比。

**类型：** 学习型
**语言：** Python（标准库、3D 视频分词器数学 + 自回归采样器骨架）
**前置条件：** 阶段 12 · 11（Chameleon）
**时间：** 约 120 分钟

## 学习目标

- 解释为什么 Emu3 的单一 loss next-token 目标有效，尽管长期存在扩散是图像质量所必需的假设。
- 描述 3D 视频分词器：时空 VQ 码本是什么样的，为什么 patch 跨越时间。
- 比较 Emu3 与 Stable Diffusion XL（训练计算、推理成本、质量上限）。
- 说出同一 Emu3 模型的三个角色：Emu3-Gen（图像生成）、Emu3-Chat（感知）、Emu3-Stage2（视频生成）。

## 问题

2024 年前的conventional wisdom：图像生成需要扩散。论点：离散图像 token 丢失太多信息无法重构细节，且自回归采样在数千 token 上累积误差。Stable Diffusion、DALL-E 3、Imagen、Midjourney 都使用某种形式的扩散。Chameleon（第 12.11 课）在小规模上部分推翻了这个论点，但质量上未能匹配 SDXL。

Emu3 直接攻击这个论点。声称：更好的视觉分词器 + 足够规模 + next-token loss = 在同一模型中也能做感知的、击败扩散的图像生成。

发布时这个赌注有争议。两年后，开源统一生成家族（Emu3、Show-o、Janus-Pro、Transfusion）是研究的默认路径；生产前沿模型似乎使用某种变体。

## 概念

### Emu3 分词器

关键成分是视觉分词器。Emu3 以 8x8 分辨率压缩每个 token 训练自定义 IBQ 类分词器（Inverse Bottleneck Quantizer，SBER-MoVQGAN 系列）。一张 512x512 图像在码本大小 32768 下变为 64x64 = 4096 token。

这比 Chameleon 的每 512x512 图像 1024 token（K=8192）更大，但每个 token 更便宜（更小的码本查找、更简单的编解码器）。关键指标：重构 PSNR 达 30.5 dB，与 Stable Diffusion 的连续潜空间 32 dB 相当。

对于视频：3D VQ 分词器将时空 patch（4x4x4 像素）编码为一个整数。一段 4 秒 clip 在 8 FPS 有 32 帧；在 256x256 下空间和 temporal 各 4x 压缩，token 数为 (256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32,768 token。

分词器质量是上限。Emu3 的贡献部分在于"我们训练了一个非常好的分词器"。

### 单一 loss 训练

Emu3 使用一个目标：跨文本 token、2D 图像 token 和 3D 视频 token 的共享词汇表的 next-token 预测。训练期间用模态特定因子乘以权重以平衡贡献，但 loss 函数是相同的。

训练混合：
- 图像生成：`<text caption> <image> image_tokens </image>`
- 图像感知：`<image> image_tokens </image> <question> text_tokens`
- 视频生成：`<text caption> <video> video_tokens </video>`
- 视频感知：类似。
- 纯文本：标准 NTP。

模型从数据分布中学习何时发出图像 token vs 文本 token。生成从模型在 `<image>` 标签后预测图像 token 中涌现。

### 无分类器引导和温度

自回归图像生成在推理时配合无分类器引导（CFG）会好得多。Emu3 使用它：生成两次，一次用完整 caption，一次用空 caption，用引导权重（典型 3.0-7.0）混合 logit。这是扩散使用的相同 CFG 技巧，借用到自回归设置中。

温度很重要：太高，伪影；太低，模式崩塌。Emu3 推荐的温度是感知 1.0，图像生成 0.8。

### 三个角色，一个模型

Emu3 作为三个功能不同的 API 发货，但共享一个底层权重集：

- Emu3-Gen。图像生成。输入文本，输出图像 token。
- Emu3-Chat。VQA 和 captioning。输入图像（token），输出文本。
- Emu3-Stage2。视频生成和视频 VQA。输入文本或视频，输出文本或视频。

没有任务特定头部。只是不同的提示模板。相同 checkpoint。

### 基准

来自 Emu3 论文（2024 年 9 月）：

- 图像生成：在 MJHQ-30K FID 上击败 SDXL（5.4 vs 5.6），GenEval 总体持平（0.54 vs 0.55——统计上持平），Deep-Eval 复合指标相当。
- 图像感知：在 VQAv2 上击败 LLaVA-1.6（75.1 vs 72.4），在 MMMU 上大致持平。
- 视频生成：4 秒 clip 质量与 Sora 时代公开基准模型相当的 FVD。

数字并不总是领先的——Emu3 在这里让出一分在那里得一分——但"next-token 预测是你需要的一切"这一声称在跨模态上是站得住脚的。

### 计算成本

Emu3 在约 3000 亿多模态 token 上用 7B 参数模型训练。GPU 小时与 Llama-2-7B 预训练大致相当（2k-4k GPU 年，A100 类硅）。Stable Diffusion 3 等扩散模型在类似预算下训练，但需要单独的文本编码器和更复杂的 pipeline。

在推理时，Emu3 每张图像比 SDXL 慢：4096 图像 token 在 30 tok/s 下约 2 分钟生成一张 512x512 图像，而 SDXL 为 2-5 秒。投机解码和 KV 缓存优化缩小差距但不能完全关闭。自回归图像生成是计算密集型的；这是持续的权衡。

### 为什么它重要

Emu3 的深层贡献是概念性的。如果 next-token 预测能扩展到在图像生成上匹配扩散，统一模型路径（一个 loss、一个骨干网、任意模态）是可行的。未来的模型不需要单独的文本编码器、单独的扩散调度器、单独的 VAE。一个 transformer，每种模态一个分词器，规模。

Show-o、Janus-Pro 和 InternVL-U 都建立在这个论点之上或挑战它。2025 年，中国实验室（BAAI、DeepSeek）在这个方向上比美国实验室发表得更激进。

## 使用它

`code/main.py` 构建两个玩具组件：

- 一个 2D vs 3D VQ 分词器计数计算器：给定（分辨率、patch、clip 长度、FPS），计算图像 vs 视频的 token 数。
- 带无分类器引导的温度下自回归图像 token 采样器。

CFG 实现匹配 Emu3 的配方——用引导权重混合条件和非条件 logit。

## 交付它

本课产出 `outputs/skill-token-gen-cost-analyzer.md`。给定生成产品规格（图像或视频、目标分辨率、质量层级、延迟预算），它计算 token 数、推理成本，并选择 Emu3 系列 vs 扩散。

## 练习

1. Emu3 每张 512x512 图像产生 4096 token，8x8 压缩。计算 1024x1024 和 2048x2048 的等价物。推理延迟会发生什么？

2. 阅读 Emu3 第 3.3 节关于视频分词器。描述 3D VQ patch 形状以及为什么是 4x4x4 而不是 8x8x1。

3. 无分类器引导权重 5.0 vs 3.0：什么视觉效果？在 `code/main.py` 中追踪数学。

4. 计算 Emu3-7B 在 300B token 上的训练 FLOPs 并与 Stable Diffusion 3 比较。哪个训练更贵？

5. Emu3 在 FID 上击败 SDXL，但在 VQAv2 上未击败专用 VLM。解释为什么统一 loss 方法在不同基准上与专家相比显示不同优势。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|------------------------|
| Next-token prediction | "NTP" | 标准自回归 loss：在给定 token[0..i] 下预测 token[i+1]；当分词后，对每种模态都有效 |
| IBQ 分词器 | "逆瓶颈量化器" | 一类 VQ-VAE，码本更大（32768+），重构质量比 Chameleon 更好 |
| 3D VQ | "时空量化器" | 按（时间，行，列）索引的码本；一个 token 覆盖 4x4x4 像素立方体 |
| 无分类器引导 | "CFG" | 用权重 gamma 混合条件和非条件 logit；在推理时提升图像质量 |
| 统一词汇表 | "共享 token" | 文本 + 图像 + 视频都从同一个整数空间抽取；模型预测接下来出现的是哪种模态 |
| MJHQ-30K | "图像生成基准" | Midjourney 质量基准，30k 提示；Emu3 在此报告 FID |

## 延伸阅读

- [Wang 等 — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun 等 — Emu: 多模态生成预训练 (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu 等 — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu 等 — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian 等 — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)