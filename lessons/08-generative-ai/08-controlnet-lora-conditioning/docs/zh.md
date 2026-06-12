# ControlNet、LoRA 与条件控制

> 纯文本是个笨拙的控制信号。ControlNet 让你克隆一个预训练的扩散模型，用深度图、姿态骨架、涂鸦或边缘图像来引导它。LoRA 让你通过训练 1000 万参数来微调一个 20 亿参数的模型。两者共同将 Stable Diffusion 从一个玩具变成了 2026 年每个广告公司都在使用的图像流水线。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 8 · 07（潜空间扩散）、阶段 10（从头构建 LLM——LoRA 基础）
**时间：** 约 75 分钟

## 问题

像"一位穿着红裙的女人在繁忙的街道上遛狗"这样的提示词，模型完全无法知道狗在*哪里*、女人是什么*姿态*、街道是什么*视角*。文本只能指定你需要描述的图像的大约 10%。剩下的都是视觉信息，无法用语言高效描述。

为每个信号（姿态、深度、canny、分割）从头训练一个新的条件模型代价太高。你想要保持 26 亿参数的 SDXL 主干冻结，附加一个读取条件的小型侧网络，让它轻推主干网络的中间特征。这就是 ControlNet。

你还想在不重训练完整模型的情况下教模型新概念（你的脸、你的产品、你的风格）。你想要一个 100 倍小的 delta。这就是 LoRA——低秩适配器，插入到现有的注意力权重中。

ControlNet + LoRA + 文本 = 2026 年从业者的工具包。大多数生产级图像流水线在 SDXL / SD3 / Flux 基础上叠加 2-5 个 LoRA、1-3 个 ControlNet 和一个 IP-Adapter。

## 概念

![ControlNet 克隆编码器；LoRA 添加低秩 delta](../assets/controlnet-lora.svg)

### ControlNet（Zhang 等，2023）

取一个预训练的 SD。*克隆* U-Net 的编码器一半。冻结原始模型。训练克隆模型接受额外的条件输入（边缘、深度、姿态）。用*零卷积*跳跃连接（初始化为 0 的 1×1 卷积——开始是空操作，学习一个 delta）将克隆连接回原始模型的解码器一半。

```
SD U-Net 解码器：... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

零卷积初始化意味着 ControlNet 起初是恒等映射——即使在训练前也无害。用标准的扩散损失在 100 万个（提示词、条件、图像）三元组上训练。

每种模态的 ControlNet 作为小型侧模型发布（SDXL 约 360M，SD 1.5 约 70M）。你可以在推理时组合它们：

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA（Hu 等，2021）

对于模型中任意线性层 `W ∈ R^{d×d}`，冻结 `W` 并添加一个低秩 delta：

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

其中 `r << d`。注意力层通常用 rank 4-16，重度微调用 rank 64-128。新参数数量：`2 · d · r` 而不是 `d²`。对于 d=640、r=16 的 SDXL 注意力：每个适配器 20k 参数而不是 410k——20 倍压缩。整个模型来看：LoRA 通常是 20-200MB，而基础模型是 5GB。

推理时你可以缩放 LoRA：`W' = W + α · B @ A`。`α = 0.5-1.5` 是正常范围。多个 LoRA 可加性叠加（通常的警告是它们以非线性方式相互作用）。

### IP-Adapter（Ye 等，2023）

一个小型适配器，接受一张*图像*作为条件（与文本一起）。使用 CLIP 图像编码器生成图像 token，将它们与文本 token 一起注入交叉注意力。基础模型约 20MB。让你实现"按照这个参考图的风格生成图像"而无需 LoRA。

## 可组合性矩阵

| 工具 | 控制内容 | 大小 | 使用场景 |
|------|------------------|------|-------------|
| ControlNet | 空间结构（姿态、深度、边缘） | 70-360MB | 精确布局、构图 |
| LoRA | 风格、主体、概念 | 20-200MB | 个性化、风格 |
| IP-Adapter | 参考图的风格或主体 | 20MB | 文本无法描述的外观 |
| Textual Inversion | 将单个概念作为新 token | 10KB | 已过时，大多被 LoRA 取代 |
| DreamBooth | 对主体进行完整微调 | 2-5GB | 强身份认同、高算力 |
| T2I-Adapter | 更轻量的 ControlNet 替代品 | 70MB | 边缘设备、推理预算 |

ControlNet ≈ 空间控制。LoRA ≈ 语义控制。两者都用。

## 构建它

`code/main.py` 在一维上模拟两种机制：

1. **LoRA。** 预训练的线性层 `W`。冻结它。训练一个低秩 `B @ A` 使得 `W + BA` 匹配目标线性层。展示 `r = 1` 足以完美学习一个 rank-1 校正。

2. **ControlNet-lite。** 一个"冻结基础"预测器和一个读取额外信号的"侧网络"。侧网络的输出由初始化为零的学习标量门控（我们的零卷积版本）。训练并观察门控上升。

### 第 1 步：LoRA 数学

```python
def lora(W, A, B, x, alpha=1.0):
    # W 是冻结的；A, B 是可训练的低秩因子。
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### 第 2 步：零初始化侧网络

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate 初始化为 0
h = base(x) + gated
```

在第 0 步，输出与基础模型完全相同。早期训练时 `gate` 缓慢更新——不会发生灾难性漂移。

## 陷阱

- **过度缩放 LoRA。** `α = 2` 或 `α = 3` 是常见的"让它更强"技巧，会产生过度风格化/破碎的输出。保持 `α ≤ 1.5`。
- **ControlNet 权重冲突。** 在权重 1.0 使用 Pose ControlNet 和在权重 1.0 使用 Depth ControlNet 通常会过度。权重之和 ≈ 1.0 是安全默认值。
- **LoRA 用错了基础模型。** SDXL LoRA 在 SD 1.5 上静默无效，因为注意力维度不匹配。Diffusers 0.30+ 会给出警告。
- **Textual Inversion 漂移。** 在一个检查点上训练的 token 在另一个检查点上会严重漂移。LoRA 更易移植。
- **LoRA 权重合并与存储。** 你可以将 LoRA 烘焙到基础模型权重中以加快推理（无需运行时加法），但你会失去在运行时缩放 `α` 的能力。两个版本都保留。

## 使用它

| 目标 | 2026 流水线 |
|------|---------------|
| 复现品牌艺术风格 | 在约 30 张策划图像上训练 LoRA，rank 32 |
| 将我的脸放入生成图像中 | DreamBooth 或 LoRA + IP-Adapter-FaceID |
| 特定姿态 + 提示词 | ControlNet-Openpose + SDXL + 文本 |
| 深度感构图 | ControlNet-Depth + SD3 |
| 参考图 + 提示词 | IP-Adapter + 文本 |
| 精确布局 | ControlNet-Scribble 或 ControlNet-Canny |
| 背景替换 | ControlNet-Seg + 修复（课程 09） |
| 快速 1 步风格 | LCM-LoRA on SDXL-Turbo |

## 交付它

保存 `outputs/skill-sd-toolkit-composer.md`。Skill 接受一个任务（输入资产：提示词、可选参考图、可选姿态、可选深度、可选涂鸦）并输出工具栈、权重和可复现的种子协议。

## 练习

1. **简单。** 在 `code/main.py` 中，将 LoRA rank `r` 从 1 变化到 4。在什么 rank 下 LoRA 能精确匹配一个 rank-2 的目标 delta？
2. **中等。** 在两个目标变换上训练两个独立的 LoRA。将它们一起加载并展示它们的加性相互作用。相互作用何时会破坏线性？
3. **困难。** 使用 diffusers 叠加：SDXL-base + Canny-ControlNet（权重 0.8）+ 风格 LoRA（α 0.8）+ IP-Adapter（权重 0.6）。测量堆叠权重变化时的 FID-vs-提示词遵循权衡。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|-----------------------|
| ControlNet | "空间控制" | 克隆编码器 + 零卷积跳跃；读取条件图像。 |
| 零卷积 | "起初是恒等映射" | 初始化为零的 1×1 卷积；ControlNet 起初是空操作。 |
| LoRA | "低秩适配器" | `W + B @ A`, `r << d`；比完整微调少 100 倍参数。 |
| rank r | "那个旋钮" | LoRA 压缩；典型 4-16，重度个性化用 64+。 |
| α | "LoRA 强度" | LoRA delta 的运行时缩放。 |
| IP-Adapter | "参考图像" | 通过 CLIP 图像 token 的小型图像条件适配器。 |
| DreamBooth | "完整主体微调" | 在约 30 张主体图像上训练完整模型。 |
| Textual Inversion | "新 token" | 仅学习一个新的词嵌入；已过时，大多被取代。 |

## 生产笔记：LoRA 交换、ControlNet 通道、多租户服务

一个真实的文生图 SaaS 在同一个基础检查点上服务数百个 LoRA 和十几个 ControlNet。服务问题看起来很像 LLM 多租户（生产文献在连续批处理和 LoRAX / S-LoRA 下覆盖了 LLM 案例）：

- **热交换 LoRA，不要合并。** 将 `W' = W + α·B·A` 合并到基础模型中每步推理快约 3-5%，但会冻结 `α` 和基础模型。将 LoRA 作为 rank-r delta 热保存在 VRAM 中；diffusers 暴露 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` 用于按请求激活。交换成本是 `2 · d · r · num_layers` 权重——MB 级，亚秒级。
- **ControlNet 作为第二个注意力通道。** 克隆的编码器与基础模型并行运行。两个权重各为 1.0 的 ControlNet = 每步两次额外前向传递，而不是一次合并传递。批量大小余量二次下降。预算每个活跃 ControlNet 约 1.5× 的步成本。
- **量化的 LoRA 也可以。** 如果你量化了基础模型（见课程 07，Flux 在 8GB 上），LoRA delta 也可以干净地量化到 8 位或 4 位。QLoRA 风格加载让你在 4 位 Flux 基础上叠加 5-10 个 LoRA 而不爆内存。

Flux 专用：Niels 的 Flux-on-8GB 笔记本将基础模型量化为 4 位；在那个量化基础上叠加风格 LoRA（`pipe.load_lora_weights("user/style-lora")`）使用 `weight_name="pytorch_lora_weights.safetensors"` 仍然有效。这是 2026 年大多数 SaaS 广告公司交付的方案。

## 进一步阅读

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet。
- [Hu 等 (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA（最初为 LLM 设计；移植到扩散）。
- [Ye 等 (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter。
- [Mou 等 (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — ControlNet 的轻量替代品。
- [Ruiz 等 (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth。
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter 文档](https://huggingface.co/docs/diffusers/training/controlnet) — 参考流水线。