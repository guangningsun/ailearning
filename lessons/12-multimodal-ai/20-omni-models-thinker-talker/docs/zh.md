# 全模态模型：Qwen2.5-Omni 与 Thinker-Talker 分离

> GPT-4o 在 2024 年 5 月的产品 demo 之所以颠覆性，不是因为底层模型，而是因为产品形态——一个语音界面，你说话，模型看到摄像头看到的内容，然后用不到 250ms 的延迟回应你。开源生态在 2024 和 2025 年剩下的时间里竞相追赶这一产品形态。Qwen2.5-Omni（2025 年 3 月）是最具参考价值的开源设计：一个 Thinker（大文本生成 Transformer）加一个 Talker（并行语音生成 Transformer），通过流式语音 tokens 链接。Mini-Omni 简化了它，Moshi 匹配了它的延迟，GLM-4-Voice 将其扩展到中文。本节解读 Thinker-Talker 架构和使流式实时对话成为可能的延迟预算。

**类型：** 构建型
**语言：** Python（标准库、流式流水线延迟模拟器 + VAD 循环）
**前置条件：** 阶段 12 · 19（音频-LLM）、阶段 12 · 16（任意到任意）
**时间：** 约 180 分钟

## 学习目标

- 将推理流水线拆分为 Thinker（文本推理）和 Talker（语音合成），并解释并行流式如何工作。
- 逐组件计算对话交互的首字节音频时间（TTFAB）预算。
- 描述 Thinker 内部 TMRoPE 跨视觉、音频和文本的时间对齐位置编码。
- 说出三种实时对话模式：半双工、轮次切换、全双工。

## 问题

实时语音助手需要做很多事，而且要快：

1. 听到用户。实时语音分词、语音活动检测（VAD）判断用户是否说完。
2. 可选地看到。摄像头输入以 2-4 FPS 流入 Thinker，与音频一起。
3. 思考。基于对话历史组织回复。
4. 说话。合成语音 tokens，解码为波形，流式传输到用户扬声器。

每一步都增加延迟。对话感要求总往返 < 500ms——低于这个值，用户感觉不到延迟。GPT-4o 宣称约 250ms。Moshi 约 160ms。Qwen2.5-Omni 约 350-500ms。

每个组件都需要流式。没有什么可以"批量处理再解码"。

## 概念

### Thinker 和 Talker

Qwen2.5-Omni 的分解：

- Thinker：一个 7B-80B 文本生成 Transformer。消费交错的文本 + 图像 + 音频 tokens。输出表示要说什么的文本 tokens。
- Talker：一个较小的语音生成 Transformer（200M-1B）。消费 Thinker 的文本输出 tokens 和最近的语音上下文 tokens。输出离散语音 tokens（残差 VQ 索引）。
- 语音解码器：一个流式波形解码器（SNAC、MoVQGAN 系列），将语音 tokens 实时转换为音频样本。

分离很重要。Thinker 必须大才能有好的推理能力。Talker 可以小，因为它的任务是局部的——将文本转换为语音 tokens。更大的 Talker 不会更有表现力，只会拖慢速度。

并行运行两者：

1. Thinker 发出文本 token t_i。
2. Talker 通过流式消费 t_i 并发出语音 tokens s_i, s_{i+1}, ..., s_{i+k}。
3. 语音解码器消费到来的语音 tokens 并发出音频样本。
4. 当 Thinker 到达文本 token t_{i+3} 时，Talker 已经为 t_0..t_{i+2} 流式传输了音频。

### TMRoPE — 时间对齐的多模态位置

Thinker 需要整合图像帧（假设以 4 FPS 到达）、音频帧（以 50 帧/秒到达）和对话历史中的文本。朴素的序列顺序（所有图像，然后所有音频，然后文本）会丢失时间对齐。

TMRoPE 为每个 token 分配绝对时间戳。视觉 token 在 t=2.3s。音频 token 在 t=2.32s。用户文本 token "stop" 在 t=2.35s。RoPE 按时间戳旋转注意力；模型将它们视为时间上同时发生。

这就是"他在说 hello 的同时挥手"这类场景的基础设施——模型在同一概念时刻看到视频帧和音频。

### 流式语音合成

语音 tokens 必须流式传输。Mini-Omni（Xie & Wu，2024 年）引入了"语言模型可以在流式思考的同时听和说"：Thinker 输出 tokens 和 Talker 输出 tokens 在同一序列中交错。Talker 在 Thinker 提交下一个文本 token 后立即启动。没有批处理边界。

Moshi（Défossez 等，2024 年 10 月）是目前最快的开源实现。单卡 A100 上 160ms TTFAB。架构：一个单独的 7B Transformer 在交替位置上发出文本和语音 tokens，并有一个"内心独白"将思考流和说话流分开。这实际上是 Thinker + Talker 融合为一个模型，经过精心训练。

### VAD 和轮次切换

语音活动检测在输入端运行。两种模式：

- 半双工：用户说话，模型听。模型说话，用户听。通过 VAD 静音检测实现清晰交接（约 200ms）。
- 全双工：双方可以同时说话。模型可以插话（"嗯哼"）或打断。更难。Moshi 支持此模式。

Qwen2.5-Omni 默认支持半双工，通过静音阈值进行轮次切换。全双工需要应用层处理。

### Qwen3-Omni（2025 年 11 月）

后继者。Qwen3-80B Thinker，更大的 Talker，改进的 TMRoPE-v2。延迟接近 GPT-4o 的 250ms。开权重。在 OmniBench 上与 Gemini 2.0 Live 竞争。

### 生产延迟预算

对于典型的流式交互：

- 麦克风 -> 音频 tokens：40-80ms。
- Prefill（提示词 + 历史）：100-200ms（7B），70B 则更多。
- 首个 Thinker 文本 token：40ms。
- Talker 处理首个文本 token：20ms。
- 首批语音 tokens 提交：40ms。
- 残差 VQ 解码：30ms。
- 语音波形解码：50-80ms。

总 TTFAB：7B 下 320-510ms，70B 下 600-900ms。前沿质量通常意味着 70B+；因此存在前沿延迟差距。

### Token 速率计算

16kHz 语音、50 Hz 基础语音 tokens，每秒输出需要 50 个语音 tokens。Talker 必须发出 ≥50 tok/s 才能跟上。在 H100 上典型的 LLM 吞吐量为 30-80 tok/s，一个小的（200-300M）Talker 足够快；7B Talker 会跟不上。

这就是为什么存在小的专用 Talker 模型，而不是"直接用主模型"。

## 使用它

`code/main.py`：

- 用模拟的 token 发射速率模拟 Thinker-Talker 流水线。
- 为可配置的模型大小和麦克风采样率计算 TTFAB。
- 演示带 VAD 静音阈值的半双工轮次切换。

## 交付它

本课产出 `outputs/skill-omni-streaming-budget.md`。给定实时语音产品的目标 TTFAB 和功能集（视觉输入、双语、全双工），选择 Qwen2.5-Omni、Qwen3-Omni、Moshi 或 Mini-Omni 并确定 Thinker/Talker 的规模。

## 练习

1. 你的目标 TTFAB 是 300ms。在 7B Thinker 和 300M Talker 上，列出每个组件的延迟。

2. Qwen2.5-Omni 使用 TMRoPE。描述模型在这样一个提示词中看到的内容：用户在 t=1s 开始说话，摄像头在 t=1.2s 捕捉到一个手势。

3. 全双工支持要求模型在听的同时发出音频。提出一种训练数据格式来教会这一点。

4. 阅读 Moshi 论文第 4 节。描述"内心独白"分离以及它如何避免 Thinker-Talker 分离。

5. 计算吞吐预算：Talker 必须以多快的速度发出 tokens 才能跟上 16kHz 语音、每秒 50 个基础层 tokens？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Thinker | "推理大脑" | 生成要说什么的大型文本生成 Transformer |
| Talker | "说话嘴巴" | 将 Thinker 的文本转换为离散语音 tokens 的小型 Transformer |
| TTFAB | "延迟预算" | 首字节音频时间：用户语音结束到首个音频样本输出的时间 |
| TMRoPE | "时间对齐 RoPE" | 跨视觉、音频、文本的绝对时间戳位置编码 |
| 半双工 | "轮次切换" | 用户和模型交替；VAD 静音检测用户是否说完 |
| 全双工 | "同时" | 模型可以同时说话和听；支持插话 |
| 内心独白 | "Moshi 分离" | 单模型设计，思考流和说话流在序列中交错 |

## 延伸阅读

- [Xu 等 — Qwen2.5-Omni（arXiv:2503.20215）](https://arxiv.org/abs/2503.20215)
- [Qwen 团队 — Qwen3-Omni（arXiv:2509.17765）](https://arxiv.org/html/2509.17765v1)
- [Xie & Wu — Mini-Omni（arXiv:2408.16725）](https://arxiv.org/abs/2408.16725)
- [Défossez 等 — Moshi（arXiv:2410.00037）](https://arxiv.org/abs/2410.00037)
- [Zeng 等 — GLM-4-Voice（arXiv:2412.02612）](https://arxiv.org/abs/2412.02612)