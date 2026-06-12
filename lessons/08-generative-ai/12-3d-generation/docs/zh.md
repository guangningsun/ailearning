# 3D 生成

> 3D 是 2D 转 3D 杠杆效应最强的模态。2023 年的突破是 3D Gaussian Splatting。2024-2026 年的生成式推动在多视角扩散 + 3D 重建之上再叠加一层，以从单个提示词或照片生成物体和场景。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 4（视觉）、阶段 8 · 07（潜空间扩散）
**时间：** 约 45 分钟

## 问题

3D 内容制作很痛苦：

- **表示方法。** 网格、点云、体素网格、有符号距离场（SDF）、神经辐射场（NeRF）、3D 高斯。每个都有权衡。
- **数据稀缺。** ImageNet 有 14M 张图片。最大的干净 3D 数据集（Objaverse-XL，2023）有约 10M 个物体，大部分质量较低。
- **内存。** 512³ 的体素网格有 128M 个体素；一个有用的场景 NeRF 需要每条光线 1M 个采样。生成比重建更难。
- **监督。** 对于 2D 图像，你有像素。对于 3D，你通常只有少量 2D 视角，必须提升到 3D。

2026 年的技术栈将两个问题分开。首先，用扩散模型生成*多视角 2D 图像*。其次，用*3D 表示*（通常是 Gaussian splatting）拟合这些图像。

## 概念

![3D 生成：多视角扩散 + 3D 重建](../assets/3d-generation.svg)

### 表示方法：3D Gaussian Splatting（Kerbl 等，2023）

将场景表示为约 1M 个 3D 高斯的云。每个有 59 个参数：位置（3）、协方差（6，或四元数 4 + 缩放 3）、不透明度（1）、球谐颜色（3 度 48 个，0 度 3 个）。

渲染 = 投影 + alpha 合成。快速（4090 上 1080p 约 100 fps）。可微分。通过对 ground-truth 照片的梯度下降来拟合。场景可以在消费级 GPU 上 5-30 分钟内完成拟合。

在此之上的两个 2023-2024 年创新：
- **生成式 Gaussian splats。** 像 LGM、LRM、InstantMesh 这样的模型从一个或几张图像直接预测高斯云。
- **4D Gaussian Splatting。** 带有每帧偏移的高斯，用于动态场景。

### 多视角扩散

对一个预训练的图像扩散模型进行微调，从文本提示词或单张图像生成同一物体的多个一致视角。Zero123（Liu 等，2023）、MVDream（Shi 等，2023）、SV3D（Stability，2024）、CAT3D（Google，2024）。通常输出物体周围的 4-16 个视角，通过 Gaussian splatting 或 NeRF 提升到 3D。

### 文本转 3D 流程

| 模型 | 输入 | 输出 | 时间 |
|-------|-------|--------|------|
| DreamFusion（2022） | 文本 | 通过 SDS 的 NeRF | 每个资产约 1 小时 |
| Magic3D | 文本 | 网格 + 纹理 | 约 40 分钟 |
| Shap-E（OpenAI，2023） | 文本 | 隐式 3D | 约 1 分钟 |
| SJC / ProlificDreamer | 文本 | NeRF / 网格 | 约 30 分钟 |
| LRM（Meta，2023） | 图像 | 三平面 | 约 5 秒 |
| InstantMesh（2024） | 图像 | 网格 | 约 10 秒 |
| SV3D（Stability，2024） | 图像 | 新视角 | 约 2 分钟 |
| CAT3D（Google，2024） | 1-64 张图像 | 3D NeRF | 约 1 分钟 |
| TripoSR（2024） | 图像 | 网格 | 约 1 秒 |
| Meshy 4（2025） | 文本 + 图像 | PBR 网格 | 约 30 秒 |
| Rodin Gen-1.5（2025） | 文本 + 图像 | PBR 网格 | 约 60 秒 |
| 腾讯 Hunyuan3D 2.0（2025） | 图像 | 网格 | 约 30 秒 |

2025-2026 年方向：适用于游戏引擎的带 PBR 材质的直接文本转网格模型。多视角扩散中间步骤仍然是通用物体上表现最好的方案。

### NeRF（背景知识）

神经辐射场（Mildenhall 等，2020）。一个小型 MLP 接收 `(x, y, z, 视角方向)` 并输出 `(颜色, 密度)`。通过沿光线积分来渲染。在质量上优于基于网格的新视角合成，但渲染速度慢 100-1000 倍。对于大多数实时应用已被 Gaussian splatting 取代，但在研究中仍占主导地位。

## 动手实现

`code/main.py` 实现了一个简单的 2D"Gaussian splatting"拟合：将一个合成目标图像（平滑渐变）表示为 2D 高斯 splat 的和。通过梯度下降优化位置、颜色和协方差以匹配目标。你可以看到两个核心操作：前向渲染（splat + alpha 合成）和梯度下降拟合。

### 第 1 步：2D 高斯 splat

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### 第 2 步：通过求和 splat 进行渲染

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

真正的 3D Gaussian splatting 按深度对高斯排序，然后按顺序 alpha 合成。我们的 2D 玩具只是求和。

### 第 3 步：通过梯度下降拟合

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## 陷阱

- **视角不一致。** 如果你独立生成 4 个视角，而它们对物体结构的看法不一致，3D 拟合就会模糊。修复：带共享注意力的多视角扩散。
- **背面幻觉。** 单图像 → 3D 必须凭空想象出看不见的那一面。质量差异很大。
- **高斯 splat 爆炸。** 无约束训练会增长到 10M 个 splat 并过拟合。来自 3D-GS 原始论文的密集化 + 剪枝启发式方法是必不可少的。
- **拓扑问题。** 来自隐式场（SDF）的网格通常有洞或自交。发货前运行重网格化器（例如 blender 的体素重网格化）。
- **训练数据的许可证。** Objaverse 许可证混杂；商业使用因模型而异。

## 实际使用

| 任务 | 2026 年推荐 |
|------|-----------|
| 从照片进行场景重建 | Gaussian splatting（3DGS、Gsplat、Scaniverse） |
| 游戏用文本转 3D 物体 | Meshy 4 或 Rodin Gen-1.5（PBR 输出） |
| 图像转 3D | Hunyuan3D 2.0、TripoSR、InstantMesh |
| 从少量图像进行新视角合成 | CAT3D、SV3D |
| 动态场景重建 | 4D Gaussian Splatting |
| 化身 / 穿衣人物 | Gaussian Avatar、HUGS |
| 研究 / SOTA | 上周刚出的那个 |

对于在游戏或电商流程中交付生产的 3D：Meshy 4 或 Rodin Gen-1.5 输出可直接导入 Unity / Unreal 的 PBR 网格。

## 交付物

保存 `outputs/skill-3d-pipeline.md`。技能接受一个 3D 简报（输入：文本 / 一张图像 / 几张图像；输出：网格 / splat / NeRF；用途：渲染 / 游戏 / VR）并输出：流程（多视角扩散 + 拟合，或直接网格模型）、基础模型、迭代预算、拓扑后处理、所需的材质通道。

## 练习

1. **简单。** 用 4、16、64 个高斯运行 `code/main.py`。报告最终 MSE 与目标的对比。
2. **中等。** 扩展到彩色高斯（RGB）。确认重建与目标颜色模式匹配。
3. **困难。** 使用 gsplat 或 Nerfstudio，从 50 张照片捕获中重建一个真实物体。报告拟合时间和在保留视角上的最终 SSIM。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 3D Gaussian Splatting | "3DGS" | 将场景表示为 3D 高斯云；可微分的 alpha 合成渲染。 |
| NeRF | "神经辐射场" | 输出 3D 点颜色 + 密度的 MLP；通过光线积分渲染。 |
| 三平面 | "三个 2D 平面" | 将 3D 分解为三个 2D 轴对齐特征网格；比体素更便宜。 |
| SDS | "分数蒸馏采样" | 使用 2D 扩散分数作为伪梯度来训练 3D 模型。 |
| 多视角扩散 | "一次生成多个视角" | 输出批量一致相机视角的扩散模型。 |
| PBR | "基于物理的渲染" | 带有反照率、粗糙度、金属度、法线通道的材质。 |
| 密集化 | "增长 splat" | 3DGS 训练启发式方法：在高梯度区域分裂 / 克隆 splat。 |

## 生产注意事项：3D 还没有统一的基础设施

与图像（潜空间扩散 + DiT）和视频（时空 DiT）不同，2026 年 3D 还没有单一的主导运行时。生产决策树在表示方法上分叉：

- **NeRF / 三平面。** 推理是对每条光线进行光线行进 + 每采样一次 MLP 前向传播。512² 渲染需要数百万次 MLP 前向传播。积极地对光线采样进行批处理；SDPA/xformers 适用。
- **多视角扩散 + LRM 重建。** 两阶段流程。阶段 1（多视角 DiT）是一个扩散服务器，与第 07 课相同。阶段 2（LRM transformer）是对视角的一次性前向传递。整体延迟分布是"扩散 + 一次性"——相应地选择每阶段服务原语。
- **SDS / DreamFusion。** 每个资产优化，不是推理。构建 jobs，不是请求处理器。

对于大多数 2026 年产品，正确的答案是"在请求时运行多视角扩散模型，异步重建为 3DGS，为实时查看提供服务"。这在 GPU 推理服务器（快）和离线优化器（慢）之间干净地分割了工作负载。

## 延伸阅读

- [Mildenhall 等（2020）。NeRF：将场景表示为神经辐射场](https://arxiv.org/abs/2003.08934) — NeRF。
- [Kerbl 等（2023）。用于实时辐射场渲染的 3D 高斯泼溅](https://arxiv.org/abs/2308.04079) — 3DGS。
- [Poole 等（2022）。DreamFusion：使用 2D 扩散实现文本转 3D](https://arxiv.org/abs/2209.14988) — SDS。
- [Liu 等（2023）。Zero-1-to-3：零样本单图像转 3D 物体](https://arxiv.org/abs/2303.11328) — Zero123。
- [Shi 等（2023）。MVDream](https://arxiv.org/abs/2308.16512) — 多视角扩散。
- [Hong 等（2023）。LRM：用于单图像转 3D 的大型重建模型](https://arxiv.org/abs/2311.04400) — LRM。
- [Gao 等（2024）。CAT3D：使用多视角扩散模型在 3D 中创建任意内容](https://arxiv.org/abs/2405.10314) — CAT3D。
- [Stability AI（2024）。稳定视频 3D（SV3D）](https://stability.ai/research/sv3d) — SV3D。