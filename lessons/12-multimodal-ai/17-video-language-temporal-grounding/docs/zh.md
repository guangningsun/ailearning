# 视频-语言模型：时间Token与定位

> 视频不是一堆照片的堆叠。一段5秒的片段包含因果顺序、动作动词和事件时序，这些是图像模型无法表示的。Video-LLaMA（Zhang等，2023年6月）率先推出了首个带音视频定位功能的开源视频-LLM。VideoChat和Video-LLaVA将这一模式进行了扩展。到2025年，Qwen2.5-VL的TMRoPE已弥合了与前沿闭源模型的差距。每个系统在时间Token的处理上各不相同——Q-former按片段、concat-pool按帧、TMRoPE按Token。本节课将解析这些模式，构建均匀采样与动态采样器的对比，并基于时间定位任务进行评估。

**类型：** 构建型
**语言：** Python（标准库，帧采样器 + 时间定位评估器）
**前置条件：** 阶段 12 · 08（LLaVA-OneVision）
**时间：** 约 180 分钟

## 学习目标

- 解释为什么时间位置编码会独立于视觉编码器影响视频VLM性能。
- 在Token每秒数量与定位准确率之间，对比均匀采样、动态FPS和事件驱动采样。
- 描述按片段的Q-former（Video-LLaMA）、按帧池化（Video-LLaVA）、按Token的M-RoPE（Qwen2.5-VL）三种设计。
- 说出四个视频基准测试：VideoMME、TempCompass、EgoSchema、Video-MMMU。

## 问题

一段1分钟、30 FPS的视频有1800帧。以每帧196个视觉Token（ViT-B @ 224）计算，共352k个Token——超过任何2024年代LLM的上下文窗口大小。

三种降维策略：

1. 子采样帧（根据内容不同，1-8 FPS）。
2. 对每帧的patch Token进行激进池化（3x3或4x4双线性池）。
3. 通过Q-former压缩：输入16帧片段，输出64个Token。

每种权衡各有不同。子采样丢失时间细节。池化丢失空间细节。Q-former两者都略微丢失，但节省了Token。

时间位置编码是另一条轴：模型如何知道第5帧发生在第6帧之前？方案包括简单1D时间RoPE（Video-LLaMA）、学习到的时间嵌入（Video-LLaVA）和TMRoPE（Qwen2.5-VL，完整3D）。

## 概念

### Video-LLaMA：按片段的Q-former + 音频分支

Video-LLaMA（2023）是首个开源视频-LLM。架构如下：

- 16帧片段，2 FPS（即8秒）。
- 每帧ViT特征 -> 视频Q-former跨注意力所有16帧 -> 32个学习查询 -> LLM。
- 并行音频分支：波形 -> ImageBind音频编码器 -> 音频Q-former -> 32个查询 -> LLM。

优势：音视频联合推理。劣势：固定片段长度，无法进行任意时长定位。

### VideoChat和Video-LLaVA

VideoChat保留了Video-LLaMA的思想但去掉了音频并简化了设计。Video-LLaVA（Lin等，2023）训练了一个统一的视觉编码器同时处理图像和视频帧（"先对齐再投影"），实现了统一表示。两者都是冻结的CLIP编码器 + MLP + LLM。

两者都无法处理长视频。都是8-16帧系统。

### Qwen2.5-VL与TMRoPE

Qwen2.5-VL引入了TMRoPE——时间-模态旋转位置嵌入。每个patch Token携带一个(t, h, w)位置，其中t是实际时间戳（而非帧索引）。

与简单时间嵌入的关键区别：

- 绝对时间，而非索引。模型看到的是"在4.2秒时"而非"在第15帧时"。
- 按Token旋转，而非按片段。每个视觉Token根据其时间戳独立旋转。
- 兼容动态FPS。如果在这里以2 FPS采样、在那里以4 FPS采样，TMRoPE原生处理不均匀间距。

TMRoPE使"猫在第几秒跳跃？"这类查询成为可能。模型可以输出"在4.2秒"。Video-LLaMA只能说"在片段早期"。

### 帧采样策略

均匀采样：在整个时长内均匀采样N帧。简单，但会丢失运动峰值。

动态FPS：根据运动强度自适应采样。光流或帧差分选择高运动片段进行更密集采样。Qwen2.5-VL基于此训练。

事件驱动：运行轻量级检测器，在动作发生处采样更多。VideoAgent使用此方法。

关键帧 + 上下文：在镜头边界处采样，附加少量相邻帧。用于电影内容。

### 按帧池化

以1 FPS、每帧576个Token计算，5分钟片段有172,800个Token。在Qwen2.5-VL-72B的128k上下文中可行，但代价昂贵。

3x3双线性池化将每帧压缩至64个Token -> 5分钟19,200个Token。对大多数任务而言是最佳平衡点。

对于智能体工作流（空间细节要求不高），可更激进地池化（6x6 -> 每帧16个Token）。

### 四个视频基准测试

- VideoMME：全面的视频理解，短视频 + 中视频 + 长视频。
- TempCompass：细粒度时间推理，"之前" / "之后"问题。
- EgoSchema：长时程第一人称视频。
- Video-MMMU：多学科视频多模态问题。

完整的视频VLM评估需要覆盖所有四个基准。它们侧重的维度不同——TempCompass全考排序，EgoSchema考3分钟以上推理，VideoMME跨越各种时长。

### 定位输出格式

时间定位的输出格式：

- 自由文本："猫在4秒左右跳跃。"易于解析但不精确。
- 结构化JSON：`{"event": "jump", "start": 4.1, "end": 4.3}`。Qwen2.5-VL以此格式训练。
- 基于Token：特殊的`<time>4.1</time>` Token与答案交错插入。Qwen2.5-VL的内部格式。

基于Token的格式对下游使用最精确。Qwen2.5-VL的JSON输出格式可直接解析。

### 2026年最佳实践

2026年视频VLM的最佳实践：

- 编码器：SigLIP 2 + M-RoPE或TMRoPE（Qwen2.5-VL）。
- 帧采样：动态FPS（1-4，根据运动情况）+ 最大帧数上限。
- 每帧池化：3x3双线性。
- 输出：带time和event字段的结构化JSON。
- 基准测试：通用任务用VideoMME + TempCompass；长时程用EgoSchema。

## 使用它

`code/main.py`包含：

- 均匀采样和动态FPS帧采样器。
- 一个简易时间定位评估器：给定时间T的"真实"事件和模型输出，在给定容差范围内评分。
- 在Video-LLaMA（16帧，Q-former）、Video-LLaVA（8帧，MLP）、Qwen2.5-VL（动态FPS + TMRoPE）之间进行对比。

## 交付它

本节课产出`outputs/skill-video-vlm-frame-planner.md`。给定一个视频任务（监控、动作识别、时间定位、摘要），它选择帧采样器、池化因子、输出格式和预期的准确率级别。

## 练习

1. 对于3分钟烹饪演示，选择均匀采样还是动态FPS。用Token数量来论证。

2. TMRoPE相比简单的时间嵌入表具体增加了什么能力？

3. 为时间定位编写一个JSON模式，供VLM学习生成。包含错误处理情况。

4. 阅读Video-LLaVA第3节"先对齐再投影"。为什么这比训练独立的图像和视频编码器更好？

5. 根据VideoMME排行榜，2026年顶级开源模型与顶级闭源模型之间的差距有多大？其中多少可归因于时间编码而非基础LLM规模？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| 时间定位 (Temporal grounding) | "时间局部化答案" | VLM输出事件发生的特定时间戳范围 |
| TMRoPE | "时间-多模态RoPE" | 带绝对时间戳的3D旋转位置编码，由Qwen2.5-VL使用 |
| 动态FPS (Dynamic FPS) | "运动感知采样" | 在高运动片段采样更多帧，在静止片段更少 |
| 帧池化 (Frame pooling) | "每帧空间压缩" | 在送入LLM前用双线性插值减少每帧的patch |
| 视频Q-former (Video Q-former) | "片段压缩器" | 跨注意力瓶颈，将N帧映射到K个学习查询 |
| VideoMME | "视频基准" | 全面的短视频/中视频/长视频基准，2500+样本 |

## 延伸阅读

- [Zhang等 — Video-LLaMA（arXiv:2306.02858）](https://arxiv.org/abs/2306.02858)
- [Li等 — VideoChat（arXiv:2305.06355）](https://arxiv.org/abs/2305.06355)
- [Lin等 — Video-LLaVA（arXiv:2311.10122）](https://arxiv.org/abs/2311.10122)
- [Qwen团队 — Qwen2.5-VL（arXiv:2502.13923）](https://arxiv.org/abs/2502.13923)
- [Lin等 — VILA-1.5（arXiv:2312.07533）](https://arxiv.org/abs/2312.07533)