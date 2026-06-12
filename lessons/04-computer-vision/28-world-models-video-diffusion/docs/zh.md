# 世界模型与视频扩散

> 一个能预测场景接下来几秒的视频模型就是一个世界模拟器。以动作 conditioning 这个预测，你就得到了一个可学习的游戏引擎。

**类型：** 学习 + 构建
**语言：** Python
**前置条件：** 阶段 4 第 10 课（扩散模型）、阶段 4 第 12 课（视频理解）、阶段 4 第 23 课（DiT + 整流流）
**时间：** 约 75 分钟

## 学习目标

- 解释纯视频生成模型（Sora 2）与动作 conditioning 世界模型（Genie 3、DreamerV3）的区别
- 描述视频 DiT：时空 patch、3D 位置编码、跨 (T, H, W) token 的联合注意力
- 追踪世界模型如何接入机器人：VLM 规划 → 视频模型模拟 → 逆动力学发出动作
- 根据具体用例（创意视频、交互仿真、自动驾驶合成）在 Sora 2、Genie 3、Runway GWM-1 Worlds、Wan-Video 和 HunyuanVideo 之间选择

## 问题

视频生成与世界建模在 2026 年合流了。一个能生成连贯一分钟视频的模型，在某种意义上已经学会了世界如何运动：物体恒常性、重力、因果关系、风格。如果你以动作（向左走、开门） conditioning 这个预测，视频模型就变成了一个可学习的模拟器，可以替代游戏引擎、驾驶模拟器或机器人环境。

其 stakes 是具体的。Genie 3 从单张图像生成可交互环境。Runway GWM-1 Worlds 合成无限可探索场景。Sora 2 生成带同步音频和建模物理的分钟级视频。NVIDIA Cosmos-Drive、Wayve Gaia-2 和 Tesla DrivingWorld 为自动驾驶训练数据生成逼真驾驶视频。世界模型范式正在悄然接管机器人领域的 sim-to-real 迁移。

本课是阶段 4 的"全局图景"课。它将图像生成、视频理解和 agentic 推理连接到主导研究正在走向的架构模式上。

## 概念

### 世界建模的三个家族

```mermaid
flowchart LR
    subgraph GEN["纯视频生成"]
        G1["文本 / 图像提示"] --> G2["视频 DiT"] --> G3["视频帧"]
    end
    subgraph ACTION["动作 conditioning 世界模型"]
        A1["过去帧 + 动作"] --> A2["隐动作视频 DiT"] --> A3["下一帧"]
        A3 --> A1
    end
    subgraph RL["用于 RL 的世界模型 (DreamerV3)"]
        R1["状态 + 动作"] --> R2["隐式转移模型"] --> R3["下一隐状态 + 奖励"]
        R3 --> R1
    end

    style GEN fill:#dbeafe,stroke:#2563eb
    style ACTION fill:#fef3c7,stroke:#d97706
    style RL fill:#dcfce7,stroke:#16a34a
```

- **Sora 2** 是纯视频生成，以提示为条件。没有动作接口。你无法在生成过程中"操控"它。
- **Genie 3**、**GWM-1 Worlds**、**Mirage / Magica** 是动作 conditioning 世界模型。从观测视频中推断隐动作，然后用推断的隐动作 conditioning未来帧预测。可交互——你按键或移动相机，场景就会响应。
- **DreamerV3** 和经典 RL 世界模型族在隐空间中预测，带显式动作 conditioning，以奖励信号训练。视觉保真度较低；但对样本效率高的 RL 更有用。

### 视频 DiT 架构

```
视频潜空间：          (C, T, H, W)
Patchify（空间）：     每帧 P_h x P_w patches 的网格
Patchify（时间）：     将 P_t 帧分组为一个时间 patch
结果 token：           (T / P_t) * (H / P_h) * (W / P_w) tokens
```

位置编码是 3D 的：每个 (t, h, w) 坐标的 rotary 或学习型 embedding。注意力可以是：

- **完全联合** —— 所有 token attends to all tokens。O(N^2)，N 为 token 数。对长视频来说过高。
- **分时** —— 交替进行时间注意力（相同空间位置，跨时间：`(H*W) * T^2`) 和空间注意力（相同时刻，跨空间：`T * (H*W)^2`)。TimeSformer 和大多数视频 DiT 使用。
- **窗口** —— (t, h, w) 中的局部窗口。Video Swin 使用。

2026 年的每个视频扩散模型都使用这三种模式之一，加上 AdaLN conditioning（第 23 课）和整流流。

### 动作 conditioning：隐动作模型

Genie 通过判别性地预测连续两帧之间的动作来学习每帧的**隐动作**。模型的解码器 then 以推断的隐动作为条件——而不是以显式的键盘按键为条件。在推理时，用户可以指定一个隐动作（或从新的先验中采样），模型生成与该动作一致的下一帧。

Sora 完全跳过动作接口。其解码器从过去的时空 token 预测下一个时空 token。提示 conditioning 起点；生成中途没有任何东西可以 steer 它。

### 物理合理性

Sora 2 的 2026 年发布明确宣传了**物理合理性**：重量、平衡、物体恒常性、因果关系。由团队通过人工评级合理性分数测量；模型在落下物体、角色碰撞和故意失败（跳跃失误）上相比 Sora 1 明显改进。

合理性仍然是主要失败模式。2024-2025 年人们吃意大利面或用杯子喝水的视频暴露了模型缺乏持久物体表示。2026 年模型（Sora 2、Runway Gen-5、HunyuanVideo）减少了但未消除这些问题。

### 自动驾驶世界模型

驾驶世界模型以轨迹、边界框或导航地图为条件生成逼真的道路场景。用途：

- **Cosmos-Drive-Dreams**（NVIDIA）—— 为 RL 训练生成分钟级驾驶视频。
- **Gaia-2**（Wayve）—— 以轨迹为条件的场景合成，用于策略评估。
- **DrivingWorld**（Tesla）—— 模拟不同天气、时间、交通状况。
- **Vista**（字节跳动）—— 反应式驾驶场景合成。

它们取代了昂贵的真实世界数据收集——用于极端情况（夜间行人乱穿马路、结冰的十字路口、特殊车辆类型），否则需要数百万英里的驾驶才能获得。

### 机器人栈：VLM + 视频模型 + 逆动力学

新兴的三组件机器人循环：

1. **VLM** 解析目标（"拿起红色杯子"），规划高层动作序列。
2. **视频生成模型** 模拟执行每个动作的样子——预测 N帧后的观测。
3. **逆动力学模型** 提取产生这些观测的具体电机命令。

这取代了奖励塑造和样本密集的 RL。世界模型负责想象；逆动力学关闭执行环。Genie Envisioner 是一个具体实现；许多研究团队正在向这个结构收敛。

### 评估

- **视觉质量** —— FVD（Fréchet 视频距离）、用户研究。
- **提示对齐** —— 每帧 CLIPScore、VQA 风格评估。
- **物理合理性** —— 在基准套件上人工评级（Sora 2 内部基准、VBench）。
- **可控性**（交互式世界模型）—— 动作 → 观测一致性；你能回到之前的状态吗？

### 2026 年模型格局

| 模型 | 用途 | 参数 | 输出 | 许可证 |
|-------|-----|------------|--------|---------|
| Sora 2 | 文本转视频、音频 | — | 1 分钟 1080p + 音频 | 仅 API |
| Runway Gen-5 | 文本/图像转视频 | — | 10 秒片段 | API |
| Runway GWM-1 Worlds | 交互式世界 | — | 无限 3D 展开 | API |
| Genie 3 | 从图像生成交互式世界 | 11B+ | 可玩帧 | 研究预览 |
| Wan-Video 2.1 | 开源文本转视频 | 14B | 高质量片段 | 非商业 |
| HunyuanVideo | 开源文本转视频 | 13B | 10 秒片段 | 宽松 |
| Cosmos / Cosmos-Drive | 自动驾驶仿真 | 7-14B | 驾驶场景 | NVIDIA 开源 |
| Magica / Mirage 2 | AI 原生游戏引擎 | — | 可修改世界 | 产品 |

## 构建

### 第 1 步：视频的 3D patchify

```python
import torch
import torch.nn as nn


class VideoPatch3D(nn.Module):
    def __init__(self, in_channels=4, dim=64, patch_t=2, patch_h=2, patch_w=2):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels, dim,
            kernel_size=(patch_t, patch_h, patch_w),
            stride=(patch_t, patch_h, patch_w),
        )
        self.patch_t = patch_t
        self.patch_h = patch_h
        self.patch_w = patch_w

    def forward(self, x):
        # x: (N, C, T, H, W)
        x = self.proj(x)
        n, c, t, h, w = x.shape
        tokens = x.reshape(n, c, t * h * w).transpose(1, 2)
        return tokens, (t, h, w)
```

步长等于核的 3D 卷积充当时空 patch化器。`(T, H, W) -> (T/2, H/2, W/2)` token 网格。

### 第 2 步：3D rotary位置编码

Rotary Position Embeddings（RoPE）分别应用于 `t`、`h`、`w` 轴：

```python
def rope_3d(tokens, t_dim, h_dim, w_dim, grid):
    """
    tokens: (N, T*H*W, D)
    grid: (T, H, W) sizes
    t_dim + h_dim + w_dim == D
    """
    T, H, W = grid
    n, seq, d = tokens.shape
    if t_dim + h_dim + w_dim != d:
        raise ValueError(f"t_dim+h_dim+w_dim ({t_dim}+{h_dim}+{w_dim}) must equal D={d}")
    assert seq == T * H * W
    t_idx = torch.arange(T, device=tokens.device).repeat_interleave(H * W)
    h_idx = torch.arange(H, device=tokens.device).repeat_interleave(W).repeat(T)
    w_idx = torch.arange(W, device=tokens.device).repeat(T * H)
    # Simplified: just scale channels by frequencies. Real RoPE rotates pairs.
    freqs_t = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(t_dim // 2, device=tokens.device) / (t_dim // 2))
    freqs_h = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(h_dim // 2, device=tokens.device) / (h_dim // 2))
    freqs_w = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(w_dim // 2, device=tokens.device) / (w_dim // 2))
    emb_t = torch.cat([torch.sin(t_idx[:, None] * freqs_t), torch.cos(t_idx[:, None] * freqs_t)], dim=-1)
    emb_h = torch.cat([torch.sin(h_idx[:, None] * freqs_h), torch.cos(h_idx[:, None] * freqs_h)], dim=-1)
    emb_w = torch.cat([torch.sin(w_idx[:, None] * freqs_w), torch.cos(w_idx[:, None] * freqs_w)], dim=-1)
    return tokens + torch.cat([emb_t, emb_h, emb_w], dim=-1)
```

简化加法形式。真正的 RoPE 以频率旋转成对通道；位置信息相同。

### 第 3 步：分时注意力块

```python
class DividedAttentionBlock(nn.Module):
    def __init__(self, dim=64, heads=2):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.space_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, grid):
        T, H, W = grid
        n, seq, d = x.shape
        # time attention: same (h, w), across t
        xt = x.view(n, T, H * W, d).permute(0, 2, 1, 3).reshape(n * H * W, T, d)
        a, _ = self.time_attn(self.ln1(xt), self.ln1(xt), self.ln1(xt), need_weights=False)
        xt = (xt + a).reshape(n, H * W, T, d).permute(0, 2, 1, 3).reshape(n, seq, d)
        # space attention: same t, across (h, w)
        xs = xt.view(n, T, H * W, d).reshape(n * T, H * W, d)
        a, _ = self.space_attn(self.ln2(xs), self.ln2(xs), self.ln2(xs), need_weights=False)
        xs = (xs + a).reshape(n, T, H * W, d).reshape(n, seq, d)
        xs = xs + self.mlp(self.ln3(xs))
        return xs
```

时间注意力在每个空间位置上跨时间 attend；空间注意力在每帧内跨位置 attend。两次 O(T^2 + (HW)^2) 操作，而非一次 O((THW)^2)。这是 TimeSformer 和每个现代视频 DiT 的核心。

### 第 4 步：组合一个微型视频 DiT

```python
class TinyVideoDiT(nn.Module):
    def __init__(self, in_channels=4, dim=64, depth=2, heads=2):
        super().__init__()
        self.patch = VideoPatch3D(in_channels=in_channels, dim=dim, patch_t=2, patch_h=2, patch_w=2)
        self.blocks = nn.ModuleList([DividedAttentionBlock(dim, heads) for _ in range(depth)])
        self.out = nn.Linear(dim, in_channels * 2 * 2 * 2)

    def forward(self, x):
        tokens, grid = self.patch(x)
        for blk in self.blocks:
            tokens = blk(tokens, grid)
        return self.out(tokens), grid
```

不是一个可用的视频生成器；是一个结构性演示，展示每个部分如何正确 shape。

### 第 5 步：检查形状

```python
vid = torch.randn(1, 4, 8, 16, 16)  # (N, C, T, H, W)
model = TinyVideoDiT()
out, grid = model(vid)
print(f"input  {tuple(vid.shape)}")
print(f"tokens grid {grid}")
print(f"output {tuple(out.shape)}")
```

期望 `grid = (4, 8, 8)`，`out = (1, 256, 32)` after patching；然后 head 将其投影到每 token 时空 patch，准备 un-patchify 回视频。

## 使用

2026 年的生产访问模式：

- **Sora 2 API**（OpenAI）—— 文本转视频，同步音频。高级定价。
- **Runway Gen-5 / GWM-1**（Runway）—— 图像转视频，交互式世界。
- **Wan-Video 2.1 / HunyuanVideo** —— 开源自托管。
- **Cosmos / Cosmos-Drive**（NVIDIA）—— 驾驶仿真开源权重。
- **Genie 3** —— 研究预览，请求访问。

对于构建交互式世界模型演示：从 Wan-Video 开始以保证质量，然后叠加隐动作适配器以实现交互性。对于自动驾驶仿真：Cosmos-Drive 是 2026 年开源参考。

对于机器人，实际工作中的栈：

1. 语言目标 -> VLM（Qwen3-VL）-> 高层计划。
2. 计划 -> 隐动作视频模型 -> 想象 rollout。
3. Rollout -> 逆动力学模型 -> 低层动作。
4. 动作执行 -> 观测反馈回步骤 1。

## 交付

本课产出：

- `outputs/prompt-video-model-picker.md` —— 根据任务、许可证和延迟在 Sora 2 / Runway / Wan / HunyuanVideo / Cosmos 之间选择。
- `outputs/skill-physical-plausibility-checks.md` —— 定义自动化检查（物体恒常性、重力、连续性）的技能，在发布前对任何生成视频运行检查。

## 练习

1. **（简单）** 计算 patch-t=2、patch-h=8、patch-w=8 时 5 秒 360p 视频的 token 数量。思考此时注意力的内存需求。
2. **（中等）** 将上述分时注意力块替换为完全联合注意力块，测量形状和参数量。解释为何分时注意力对真实视频模型是必要的。
3. **（困难）** 构建一个最小化隐动作视频模型：获取 (frame_t, action_t, frame_{t+1}) 三元组的数据集（任何简单的 2D 游戏），训练一个以动作 embedding 为条件的小型视频 DiT，并展示不同动作产生不同的下一帧。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|----------------------|
| 世界模型 | "可学习的模拟器" | 给定状态和动作预测未来观测的模型 |
| 视频 DiT | "时空 transformer" | 带3D patch 化和分时注意力的扩散 transformer |
| 隐动作 | "推断的控制" | 从帧对中推断的离散或连续动作隐变量；用于 conditioning下一帧生成 |
| 分时注意力 | "先时间后空间" | 每块两次注意力操作——先跨时间再跨空间——保持 O(N^2) 可管理 |
| 物体恒常性 | "事物保持真实" | 视频模型必须学习的场景属性；在食物、玻璃器皿上的经典失败模式 |
| FVD | "Fréchet 视频距离" | 视频版 FID；主要视觉质量指标 |
| 逆动力学模型 | "观测到动作" | 给定（状态，下一状态），输出连接它们的动作；关闭机器人循环 |
| Cosmos-Drive | "NVIDIA 驾驶仿真" | 用于 RL 和评估的开源权重自动驾驶世界模型 |

## 延伸阅读

- [Sora 技术报告 (OpenAI)](https://openai.com/index/video-generation-models-as-world-simulators/)
- [Genie: 生成式交互环境 (Bruce et al., 2024)](https://arxiv.org/abs/2402.15391) —— 隐动作世界模型
- [TimeSformer (Bertasius et al., 2021)](https://arxiv.org/abs/2102.05095) ——视频 transformer 分时注意力
- [DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104) —— 用于 RL 的世界模型
- [Cosmos-Drive-Dreams (NVIDIA, 2025)](https://research.nvidia.com/labs/toronto-ai/cosmos-drive-dreams/) —— 驾驶世界模型
- [2026 年十大视频生成模型 (DataCamp)](https://www.datacamp.com/blog/top-video-generation-models)
- [从视频生成到世界模型——综述仓库](https://github.com/ziqihuangg/Awesome-From-Video-Generation-to-World-Model/)