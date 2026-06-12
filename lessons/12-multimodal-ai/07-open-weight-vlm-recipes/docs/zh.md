# 开源权重 VLM 配方：真正重要的是什么

> 2024-2026 年的开源权重 VLM 文献是一片消融实验表的森林。苹果的 MM1 测试了 13 种图像编码器、连接器和数据配比的组合。Allen AI 的 Molmo 证明了详细的人类字幕优于 GPT-4V 蒸馏。Cambrian-1 运行了 20+ 种编码器比较。Idefics2 将设计空间形式化为五个维度。Prismatic VLM 在受控基准上比较了 27 种训练配方。在所有这些噪声中，有一小部分结论在各篇论文中一致成立：图像编码器比连接器架构更重要，数据配比比两者都更重要，详细的人类字幕优于蒸馏的合成数据。本课为你解读这些表格，这样你就不必自己翻阅了。

**类型：** 学习 + 实验
**语言：** Python（标准库，消融表解析器 + 配方选择器）
**前置条件：** 阶段 12 · 05（LLaVA 基线）
**时间：** 约 180 分钟

## 学习目标

- 说出 VLM 设计空间的五个维度：图像编码器、连接器、语言模型、数据配比、分辨率调度。
- 阅读 MM1 / Idefics2 / Cambrian-1 的消融表，预测哪个旋钮会影响给定的基准。
- 给定计算预算和任务组合，为新的 VLM 选择配方（编码器、连接器、数据、分辨率）。
- 解释为什么详细的人类字幕在相同 token 数下优于 GPT-4V 蒸馏。

## 问题

目前已有数百个开源权重 VLM。"良好"与"最先进的"之间的差距大部分不在架构，而在数据、分辨率调度和编码器选择。当你的模型表现不佳时，知道首先该拧哪个旋钮，可以帮你避免一个耗费 500 万 GPU 小时的大错。

2023 年的浪潮（LLaVA-1.5、InstructBLIP、MiniGPT-4）运行在字幕对预训练 + LLaVA-Instruct-150k 上。良好的基线。在 MMMU 上约达到 35% 后触顶。

2024 年的浪潮（MM1、Idefics2、Molmo、Cambrian-1、Prismatic VLM）进行了详尽的消融实验。结果出人意料且实用。

## 概念

### 五个维度的设计空间

Idefics2（Laurençon 等人，2024）命名了以下维度：

1. 图像编码器。CLIP ViT-L/14、SigLIP SO400m/14、DINOv2 ViT-g/14、InternViT-6B。编码器在 patch 大小、分辨率和预训练目标上有所不同。
2. 连接器。MLP（2-4 层）、Q-Former（32 个查询 + 交叉注意力）、Perceiver Resampler（64 个查询）、C-Abstractor（卷积 + 双线性池化）。
3. 语言模型。Llama-3 8B / 70B、Mistral 7B、Phi-3、Gemma-2、Qwen2.5。LLM 大小是主要的参数量成本。
4. 训练数据。字幕对（CC3M、LAION）、交错数据（OBELICS、MMC4）、指令数据（LLaVA-Instruct、ShareGPT4V、PixMo、Cauldron）。
5. 分辨率调度。固定 224/336/448、AnyRes、原生动态。在训练期间逐步提升或保持恒定。

每个生产级 VLM 在每个维度上都有选择。MMMU 分数的大部分方差由维度 1、4 和 5 解释——而不是你选择了哪个连接器。

### 维度 1：编码器 > 连接器

MM1 第 3.2 节显示：从 CLIP ViT-L/14 切换到 SigLIP SO400m/14 在 MMMU 上增加了 3+ 分。从 MLP 切换到 Perceiver Resampler 增加了不到 1 分。Idefics2 复现了结果：在相同 token 数下 SigLIP > CLIP，Q-Former ≈ MLP ≈ Perceiver。

Cambrian-1 的"Cambrian 视觉编码器对决"（Tong 等人，2024）在以视觉为中心的基准（CV-Bench）上运行了 20+ 种编码器。排行榜顶端是 DINOv2 和 SigLIP 的混合；CLIP 处于中游；ImageBind 和 ViT-MAE 靠下。从 CLIP ViT-L 到 DINOv2 ViT-g/14 在 CV-Bench 上差距约 5-7 分。

2026 年开源 VLM 的默认编码器是 SigLIP 2 SO400m/14，用于语义 + 密集特征，有时与 DINOv2 ViT-g/14 特征拼接（Cambrian 的"空间视觉聚合器"就是这样做的）。

### 维度 2：连接器设计差异不大

MM1、Idefics2、Prismatic 和 MM-Interleaved 都得出相同结论：在固定的视觉 token 数下，连接器架构几乎无关紧要。一个 2 层 MLP 在 mean-pooled patches 上在相同 token 预算下与 32 查询 Q-Former 的差距在 1 分以内。

真正重要的是 token 数。更多视觉 token = 更多 LLM 计算 = 更好的性能——直到某个点，然后收益递减。每张图像 64 个 token 对 OCR 来说太少。对于大多数开源 VLM，576-1024 个 token 是最佳区间。2048+ 只对文档和图表有帮助。

Q-Former 与 MLP 是成本问题，不是质量问题：Q-Former 将 token 上限控制在 32-64，无论图像分辨率如何；MLP 发出所有 patch token。对于高分辨率输入，Q-Former 节省 LLM 上下文；对于低分辨率，差异可以忽略。

### 维度 3：LLM 大小决定了天花板

将 LLM 从 7B 加倍到 13B 可在每篇 VLM 论文中可靠地增加 MMMU 2-4 分。到了 70B，大多数基准都饱和了。VLM 的多模态推理天花板就是 LLM 的文本推理天花板——视觉编码器只能喂给它，不能替它推理。

这就是为什么 Qwen2.5-VL-72B 和 Claude Opus 4.7 在 MMMU-Pro 和 ScreenSpot-Pro 上碾压对手：语言脑容量巨大。一个 7B VLM 无法通过精巧的连接器设计来替代 70B VLM。

### 维度 4：数据——详细的人类字幕优于蒸馏

Molmo + PixMo（Deitke 等人，2024）是 2024 年每个人都应该读的结果。Allen AI 让人类注释者以 1-3 分钟的密集语音转文本方式描述图像，产生了 712K 密集标注的图像。训练数据中完全没有 GPT-4V 蒸馏。

Molmo-72B 在 11 个基准中的 11 个上击败了 Llama-3.2-90B-Vision。差距不在架构——而在字幕质量。详细的人类字幕每张图像包含的信息比短期网络字幕多 5-10 倍，并且在 GPT-4V 蒸馏会产生幻觉的地方保持事实正确。

ShareGPT4V（Chen 等人，2023）和 Cauldron（Idefics2）遵循相同的策略，使用混合人类 + GPT-4V 字幕。趋势很明确：对于 2026 年的前沿，字幕密度 > 字幕数量 > 蒸馏便利性。

### 维度 5：分辨率及其调度

Idefics2 的消融：384 -> 448 增加 1-2 分。448 -> 980 配合图像分块（AnyRes）在 OCR 基准上再增加 3-5 分。平坦分辨率训练在中等准确度时趋于平稳；分辨率逐步提升（从 224 开始，到 448 或原生分辨率结束）训练更快且最终效果更好。

Cambrian-1 运行了分辨率与 token 的权衡：在固定计算量下，你可以选择低分辨率多 token 或高分辨率少 token。高分辨率对 OCR 有利；低分辨率多 token 对通用场景理解有利。

2026 年的生产配方：第 1 阶段在 384 固定下训练，第 2 阶段动态分辨率可达 1280 用于 OCR 密集型任务。

### Prismatic 受控比较

Prismatic VLM（Karamcheti 等人，2024）是控制了所有维度的论文。相同的 13B LLM、相同的指令数据、相同的评估——每次只有一个维度变化。结果：

- 每张图像的视觉 token 数解释了约 60% 的方差。
- 编码器选择解释了约 20%。
- 连接器架构解释了约 5%。
- 其他一切（数据配比、调度器、学习率）剩余约 15%。

这是一个粗略的分解，但它是文献中"我应该首先消融哪个"的最佳答案。

### 2026 年配方选择器

根据证据，2026 年新项目的默认开源 VLM 配方：

- 编码器：SigLIP 2 SO400m/14 配合 NaFlex 原生分辨率，如果需要分割/接地则与 DINOv2 ViT-g/14 拼接获取密集特征。
- 连接器：2 层 MLP 在 patch token 上。除非你受 token 限制，否则跳过 Q-Former。
- LLM：Qwen2.5 / Llama-3.1 / Gemma 2，7B 成本导向，70B 质量导向，按目标延迟选择。
- 数据：PixMo + ShareGPT4V + Cauldron，用任务特定的指令数据补充。
- 分辨率：动态（长边最小 256，最大 1280 像素）。
- 调度：第 1 阶段对齐（仅投影仪），第 2 阶段全面微调，第 3 阶段任务特定微调。

每一个默认值都可以追溯到本课末尾引用的论文中的实测消融结果。

## 动手实现

`code/main.py` 是一个消融表解析器和配方选择器。它编码了 MM1 和 Idefics2 的消融表（精简版），让你可以查询：

- "给定预算 X 和任务 Y，哪个配方胜出？"
- "如果在 7B Llama 上将 SigLIP 换成 CLIP，预期的 MMMU 变化是多少？"
- "哪个维度应该首先消融以获得 80% 的置信度答案？"

输出是一个排名靠前的配方列表，包含预期的基准变化和"首先消融"的建议。

## 交付

本课产出 `outputs/skill-vlm-recipe-picker.md`。给定目标任务组合、计算预算和延迟目标，它输出一套完整配方（编码器、连接器、LLM、数据配比、分辨率调度），并附带消融结果的引用来说明每个选择。防止工程师在每个新 VLM 项目开始时重新发明 Idefics2 消融表。

## 练习

1. 阅读 MM1 第 3.2 节。对于固定 2B LLM 在 50M 图像预算下，哪个编码器胜出？在 13B LLM 下答案会翻转吗？为什么？

2. Cambrian-1 发现拼接 DINOv2 + SigLIP 在以视觉为中心的基准上优于单独使用任何一个，但在 MMMU 上没有增加。预测哪些基准会受益，哪些保持持平。

3. 你的目标是移动端 UI 代理，基于 2B LLM。选择编码器、连接器、分辨率和数据配比。用具体的消融表为每个选择提供理由。

4. Molmo 提供 4B 和 72B 模型。4B 与闭源 7B VLM 竞争；72B 在 11/11 基准上击败 Llama-3.2-90B-Vision。这告诉你关于 LLM 大小 plateau 假设的什么信息？

5. 设计一个消融表来隔离 7B VLM 上数据配比质量和编码器质量。需要最少多少次训练运行？提出四个维度的设置。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|------------------------|
| 消融（Ablation） | "拧一个旋钮" | 训练多次运行，仅在唯一一个设计空间维度上不同，其他一切保持不变 |
| 连接器（Connector） | "桥" / "投影仪" | 可训练模块，将视觉编码器输出映射到 LLM 的 token 空间（MLP、Q-Former、Perceiver） |
| 详细人类字幕（Detailed human caption） | "密集字幕" | 多句人类书面描述（通常 80-300 个 token），比网络 alt 文本更丰富 |
| 蒸馏（Distillation） | "GPT-4V 字幕" | 由更强的专有 VLM 生成的训练数据；方便但容易继承幻觉 |
| AnyRes / 动态分辨率（AnyRes / dynamic res） | "高分辨率路径" | 通过平铺或 M-RoPE 将大于编码器原生分辨率的图像输入的策略 |
| 分辨率逐步提升（Resolution ramp） | "课程" | 从低分辨率开始逐渐增加的训练调度，加速对齐学习 |
| 以视觉为中心的基准（Vision-centric bench） | "CV-Bench / BLINK" | 强调细粒度视觉感知而非语言密集推理的评估 |
| PixMo | "Molmo 的数据" | Allen AI 的 712K 密集标注图像数据集；人类语音转录为密集字幕 |

## 延伸阅读

- [McKinzey 等人 — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon 等人 — Idefics2 / 构建 VLM 真正重要的东西 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke 等人 — Molmo 和 PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong 等人 — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti 等人 — Prismatic VLM (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
