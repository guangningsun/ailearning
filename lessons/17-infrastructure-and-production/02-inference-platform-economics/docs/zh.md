# 推理平台经济学 — Fireworks、Together、Baseten、Modal、Replicate、Anyscale

> 2026 年的推理市场不再是 GPU 时间租赁。它分叉成定制硅（Groq、Cerebras、SambaNova）、GPU 平台（Baseten、Together、Fireworks、Modal）和 API 优先 marketplace（Replicate、DeepInfra）。Fireworks 于 2026 年 5 月 1 日将价格提高每小时 $1/GPU，$4B 估值和每天处理 10T+ token 的数据告诉你量驱动模式是可行的。Baseten 于 2026 年 1 月以 $5B 估值完成 $300M E 轮。竞争定位规则很简单：Fireworks 优化延迟，Together 优化目录广度，Baseten 优化企业级体验，Modal 优化 Python 原生 DX，Replicate 优化多模态覆盖，Anyscale 优化分布式 Python。本课给你一个可以递给创始人的矩阵。

**类型：** 学习型
**语言：** Python（标准库、toy 按调用经济学比较器）
**前置条件：** 阶段 17 · 01（受管理 LLM 平台）、阶段 17 · 04（vLLM 服务内部原理）
**时间：** 约 60 分钟

## 学习目标

- 说出三个市场细分（定制硅、GPU 平台、API 优先）并将每个供应商映射到一个细分。
- 解释为什么"按 token" API 定价模型压缩到 serving 引擎的成本曲线，而非硬件的成本曲线。
- 计算至少三个供应商的有效每请求成本，并解释什么时候按分钟计费（Baseten、Modal）击败按 token 计费。
- 识别哪个平台是给定工作负载的正确默认（无服务器突发、稳定高吞吐量、微调变体、多模态）。

## 问题

你评估了托管超大规模云平台。你决定需要一个更窄、更快的提供商 — 用于延迟的 Fireworks、用于广度的 Together、用于微调自定义模型的 Baseten。现在你有六个真实选择，定价页面也不一致。Fireworks 显示每百万 token 的价格；Baseten 显示每分钟；Modal 显示每秒；Replicate 显示每个预测。你无法在不建模工作负载的情况下直接比较它们。

更糟的是，每个定价页面背后的商业模式是不同的。Fireworks 在共享 GPU 上运行自己的定制引擎（FireAttention）；按 token 费率反映它们的利用率曲线。Baseten 提供 Truss + 专用 GPU；按分钟计费反映独占性。Modal 是真正的 Python 无服务器 — 按秒计费，子秒级冷启动。相同的输出（一个 LLM 响应），三种不同的成本函数。

本课对六种平台建模，告诉你每种何时胜出。

## 概念

### 三个细分

**定制硅** — Groq (LPU)、Cerebras (WSE)、SambaNova (RDU)。通常比同模型的 GPU 集群快 5-10 倍解码。按 token 价格更高（Groq 在 2025 年底约为 Llama-70B 每百万 token $0.99），但对延迟敏感用例无与伦比。Groq 是语音代理和实时翻译的生产选择。

**GPU 平台** — Baseten、Together、Fireworks、Modal、Anyscale。在 NVIDIA（H100、H200、2026 年的 B200）或有时 AMD 上运行。"原始 GPU 租赁"（RunPod、Lambda）和"超大规模云托管服务"（Bedrock）之间的经济层。

**API 优先 marketplace** — Replicate、DeepInfra、OpenRouter、Fal。广泛目录，按预测或按秒计费，强调首调用时间。

### Fireworks — 延迟优化的 GPU 平台

- FireAttention 引擎（定制）； marketed as 4x lower latency than vLLM on equivalent configs.
- 批量层约服务器无服务器费率的 50%，用于非交互式工作负载。
- 微调模型以与基础模型相同的价格提供服务 — 相对于对你的 LoRA 收取溢价的提供商，这是一个真正的差异化。
- 2026 年中期：2026 年 5 月 1 日有效提高按需 GPU 租赁 $1/小时。规模可协商批量定价。
- 财务信号：$4B 估值，每天处理 10T+ token。

### Together — 广度优化

- 200+ 模型，包括在上游发布后几天内的开源版本。
- 比 Replicate 等效 LLM 模型便宜 50-70% — "AI Native Cloud" 定位是量和目录。
- 推理 + 微调 + 训练一个 API 搞定。

### Baseten — 企业级体验优化

- Truss 框架：模型打包，包含依赖、secrets、serving 配置在一个清单中。
- GPU 范围从 T4 到 B200。按分钟计费，冷启动缓解合理。
- SOC 2 Type II，HIPAA 就绪。常见的金融科技和医疗选择。
- $5B 估值，2026 年 1 月 E 轮（$300M，来自 CapitalG、IVP、NVIDIA）。

### Modal — Python 原生优化

- 纯 Python 中的基础设施即代码。用 `@modal.function(gpu="A100")` 装饰一个函数，一行命令部署。
- 按秒计费。冷启动 2-4s 预热；小模型 <1s。
- $87M B 轮，$1.1B 估值（2025 年）。独立调查中开发者体验评分最高。

### Replicate — 多模态广度

- 按预测计费。图像、视频和音频模型的默认平台。
- 集成生态系统（Zapier、Vercel、CMS 插件）。
- 在 LLM 按 token 费率上竞争力较弱，但在多模态多样性上胜出。

### Anyscale — Ray 原生

- 构建于 Ray 之上；RayTurbo 是 Anyscale 的专有推理引擎（与 vLLM 竞争）。
- 最适合推理步骤是更大图中的一个节点的分布式 Python 工作负载。
- 托管 Ray 集群；与 Ray AIR 和 Ray Serve 紧密集成。

### 按 token vs 按分钟 — 何时各自胜出

当工作负载延迟不敏感且突发时，按 token 计费有意义；当利用率高且可预测时，按分钟计费有意义 — 一旦你饱和了 GPU，按分钟就击败按 token。

粗略规则：对于持续利用率高于约 30% 的专用 GPU，按分钟（Baseten、Modal）开始击败按 token（Fireworks、Together）。低于该值，按 token 胜出，因为你避免了为空闲付费。

### 定制引擎是真正的护城河

每个 vLLM 和 SGLang 之上的平台都声称定制引擎。FireAttention、RayTurbo、Baseten 的推理栈。定制引擎声称是营销 — 诚实的框架是 vLLM + SGLang 代表了约 80% 的生产开源推理，平台层的差异化在 DX、归属和 SLA。

### 你应该记住的数字

- Fireworks GPU 租赁：2026 年 5 月 1 日有效提高 $1/小时。
- Fireworks 声称：等效配置下比 vLLM 低 4 倍延迟。
- Together：比 Replicate 的 LLM 便宜 50-70%。
- Baseten 估值：$5B（E 轮，2026 年 1 月，$300M 轮次）。
- Modal 估值：$1.1B（B 轮，2025 年）。
- 按分钟在约 30% 持续利用率以上击败按 token。

## 使用它

`code/main.py` 在合成工作负载上比较六家供应商跨定价模型。报告每天 $ 和有效每百万 token。运行它找到按 token 和按分钟之间的盈亏平衡点。

## 交付它

本课产出 `outputs/skill-inference-platform-picker.md`。给定工作负载 profile、SLA 和预算，选择主要推理平台并命名第二名。

## 练习

1. 运行 `code/main.py`。在什么持续利用率下 Baseten（按分钟）在一部 H100 上击败 Fireworks（按 token）用于 70B 模型？自己推导交叉点并与经验法则比较。
2. 你的产品提供图像生成加聊天加语音转文本。为每种模态选择平台并命名统一它们的网关模式。
3. Fireworks 在你的主要型号上提价 $1/小时。如果 40% 的流量转向批量层（5 折），建模混合成本影响。
4. 一个受监管客户需要 SOC 2 Type II + HIPAA + 专用 GPU。哪三个平台可行，哪个在 FinOps 上胜出？
5. 比较 Llama 3.1 70B 在 Fireworks 无服务器、Together 按需、Baseten 专用和 Replicate API 上每 1000 次预测的成本。在每天 10 次预测时哪个最便宜？每天 10,000 次呢？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 定制硅 | "非 GPU 芯片" | Groq LPU、Cerebras WSE、SambaNova RDU — 针对解码优化 |
| FireAttention | "Fireworks 引擎" | 定制注意力 kernel； marketing 声称比 vLLM 低 4 倍延迟 |
| Truss | "Baseten 的格式" | 模型打包清单；依赖 + secrets + serving 配置 |
| 按 token | "API 定价" | 按消耗的 token 计费；为空闲付费 |
| 按分钟 | "专用定价" | 按 wall-clock GPU 时间计费；在高利用率时胜出 |
| 按预测 | "Replicate 定价" | 每次模型调用计费；常见于图像/视频 |
| RayTurbo | "Anyscale 引擎" | Ray 上的专有推理；在 Ray 集群上与 vLLM 竞争 |
| 批量层 | "5 折" | 非交互队列以降低费率；常见于 Fireworks、OpenAI |
| 按基础模型费率微调 | "Fireworks LoRA" | 按基础模型费率对 LoRA 服务的请求计费（差异化） |

## 延伸阅读

- [Fireworks 定价](https://fireworks.ai/pricing) — 按 token 费率、批量层、GPU 租赁。
- [Baseten 定价](https://www.baseten.co/pricing/) — 按分钟费率、承诺容量、企业层。
- [Modal 定价](https://modal.com/pricing) — 按秒 GPU 费率和无服务器层。
- [Together AI 定价](https://www.together.ai/pricing) — 模型目录和按 token 费率。
- [Anyscale 定价](https://www.anyscale.com/pricing) — RayTurbo 和托管 Ray 定价。
- [Northflank — Fireworks AI 替代品](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — 对比评估。
- [Infrabase — AI 推理 API 提供商 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) — 供应商格局。