# 顶点项目 07 — 端到端微调流水线（从数据到 SFT 到 DPO 到服务）

> 一个 8B 模型，用你自己的数据训练，用你自己的偏好做 DPO 对齐，量化，带推测解码，由 vLLM 提供服务，按 $/1M tokens 计量成本。2026 年的开源技术栈是 Axolotl v0.8、TRL 0.15、Unsloth 用于迭代、GPTQ/AWQ/GGUF 用于量化、vLLM 0.7 搭配 EAGLE-3 用于服务。这个顶点项目的目标是让整个流水线可复现运行——YAML 输入，服务端点输出——并按照 2026 Model Openness Framework 发布模型卡。

**类型：** 顶点项目
**语言：** Python（流水线）、YAML（配置）、Bash（脚本）
**前置条件：** 阶段 2（机器学习）、阶段 3（深度学习）、阶段 7（Transformer）、阶段 10（从零构建 LLM）、阶段 11（LLM 工程）、阶段 17（基础设施）、阶段 18（安全）
**涉及的阶段：** P2 · P3 · P7 · P10 · P11 · P17 · P18
**时间：** 35 小时

## 问题

2026 年，每一个正经的 AI 团队都会备有一条微调流水线。不是因为他们要发布一个前沿基础模型，而是因为下游适配——领域 SFT、基于标注偏好的 DPO、用于推测解码的蒸馏草稿、带 EAGLE-3 的服务——才是有可衡量收益的地方。Axolotl v0.8 处理多 GPU SFT 配置。TRL 0.15 处理 DPO 和 GRPO。Unsloth 让你快速进行单 GPU 迭代。vLLM 0.7 搭配 EAGLE-3 在不损失质量的前提下将解码吞吐量提升 2-3 倍。工具链是现成的；功夫在 YAML、数据卫生和评估纪律上。

你将把一个 8B 基础模型（Llama 3.3、Qwen3 或 Gemma 3）通过 SFT 然后 DPO 在任务特定数据上微调，量化以供服务，并在 lm-evaluation-harness、RewardBench-2、MT-Bench-v2 和 MMLU-Pro 上测量收益。你将按照 2026 Model Openness Framework 生成模型卡。重点在于可复现性——一条命令从头到尾重新运行整个流水线。

## 概念

流水线有五个阶段。**数据**：去重（MinHash / Datatrove）、质量过滤（Nemotron-CC 风格分类器）、PII 清除、针对公共基准污染的分裂卫生检查。**SFT**：Axolotl YAML，8xH100 上 ZeRO-3，余弦调度，打包序列，2-3 个 epoch。**DPO 或 GRPO**：TRL 配置，1 个 epoch，偏好对可以是人工标注的或模型评判的，beta 调优。**量化**：GPTQ + AWQ + GGUF，灵活部署。**服务**：vLLM 0.7 搭配 EAGLE-3 推测头（或 SGLang 搭配 SpecForge），K8s 部署，基于队列等待的 HPA。

消融实验是交付物：SFT-only vs SFT+DPO vs SFT+GRPO 在三个任务特定基准上的对比。服务指标：batch 1 / 8 / 32 下的 tokens/s，EAGLE-3 接受率，$/1M tokens。安全评估：Llama Guard 4 通过率。模型卡：偏差评估、可复现性种子、数据许可。

## 架构

```
原始数据 (HF datasets + 内部数据)
    |
    v
Datatrove 去重 + Nemotron-CC 质量过滤 + PII 清除
    |
    v
分裂卫生检查 (MMLU-Pro 污染检查)
    |
    v
Axolotl SFT 配置 (YAML)  --->  8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO 配置       --->  4xH100, 1 个 epoch
    |
    v
GPTQ + AWQ + GGUF 量化
    |
    v
vLLM 0.7 + EAGLE-3 推测解码
    |
    v
K8s 部署，基于队列等待的 HPA
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
模型卡 (2026 MOF) + 安全评估 (Llama Guard 4)
```

## 技术栈

- 数据：Datatrove 用于去重，Nemotron-CC 分类器用于质量，Presidio 用于 PII 清除
- 基础模型：Llama 3.3 8B、Qwen3 14B 或 Gemma 3 12B
- SFT：Axolotl v0.8 搭配 ZeRO-3、Flash Attention 3、打包序列
- 偏好调优：TRL 0.15 用于 DPO 或 GRPO；Unsloth 用于单 GPU 迭代
- 量化：GPTQ (Marlin)、AWQ、通过 llama.cpp 的 GGUF
- 服务：vLLM 0.7 搭配 EAGLE-3 推测解码（或 SGLang 0.4 + SpecForge）
- 评估：lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro
- 安全评估：Llama Guard 4、ShieldGemma-2
- 基础设施：Kubernetes + NVIDIA device plugin，基于队列等待指标的 HPA
- 可观测性：W&B 用于训练，Langfuse 用于推理

## 构建它

1. **数据流水线。** 在原始语料上运行 Datatrove 去重。应用 Nemotron-CC 风格的质量分类器。Presidio 清除 PII。用明确的种子编写训练/验证分裂。

2. **污染检查。** 对每个验证分裂，计算相对于 MMLU-Pro、MT-Bench-v2、RewardBench-2 测试集的 MinHash。拒绝任何重叠。

3. **Axolotl SFT。** YAML 配置 ZeRO-3、FA3、序列打包。8xH100 上 2-3 个 epoch。记录到 W&B。

4. **TRL DPO / GRPO。** 拿到 SFT 检查点，在偏好对上进行一个 epoch 的 DPO（或者用可验证奖励在数学/代码上进行 GRPO）。遍历 beta。

5. **量化。** 生成三种量化版本：GPTQ-INT4-Marlin、AWQ-INT4、GGUF-Q4_K_M for llama.cpp。记录大小和标称吞吐量。

6. **用推测解码服务。** vLLM 0.7 配置搭配通过 Red Hat Speculators 训练的 EAGLE-3 草稿头。在 batch 1 / 8 / 32 下测量接受率和尾延迟。报告相对于在同一评估上运行的 Anthropic / OpenAI 的 $/1M tokens。

7. **评估矩阵。** 在基础模型、SFT-only、SFT+DPO、SFT+GRPO 上运行 lm-eval-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro。生成一张表格。

8. **安全评估。** Llama Guard 4 在开发集上的通过率。ShieldGemma-2 输出过滤器。

9. **模型卡。** MOF 2026 模板：数据、训练、评估、安全、许可证，带有 YAML 和 commit SHA 的可复现性部分。

## 使用它

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k 去重后, 12k 过滤后, 280k 接受 (seed=7)
[SFT]     3 个 epoch, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 个 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 接受率 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md 按照 2026 MOF 生成
```

## 交付它

`outputs/skill-finetuning-pipeline.md` 描述了交付物。一条命令运行数据经过 SFT 经过 DPO 经过量化经过服务经过评估，并输出模型卡和服务端点。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 相对于基础模型的评估提升 | 在目标任务（MMLU-Pro、MT-Bench-v2、任务特定）上的可衡量提升 |
| 20 | 流水线可复现性 | 一条命令用相同种子从头到尾重新运行 |
| 20 | 数据卫生 | 去重率、PII 清除覆盖率、污染检查通过 |
| 20 | 服务效率 | bs=1/8/32 下的 tokens/s、EAGLE-3 接受率、$/1M tokens |
| 15 | 模型卡 + 安全评估 | 2026 MOF 完整性 + Llama Guard 4 通过率 |
| **100** | | |

## 练习

1. 在同一个任务特定基准上运行 SFT-only vs SFT+DPO vs SFT+GRPO。报告哪种偏好方法胜出以及胜出多少。

2. 将 Llama 3.3 8B 换成 Qwen3 14B。在匹配质量下测量 $/1M tokens。

3. 在领域数据 vs 通用 ShareGPT 上测量 EAGLE-3 接受率。报告差异以及对延迟预算的影响。

4. 注入 1% 的污染（将 MMLU-Pro 答案泄露到训练数据中）并重新运行评估。看着 MMLU-Pro 准确率不真实地飙升。构建一个能捕获此问题的污染检查 CI 门。

5. 添加 LoRA SFT 作为全量微调的替代方案。以 10 倍低的内存测量质量差距。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Axolotl | "SFT 训练器" | 统一的 YAML 驱动训练器，用于 SFT、DPO 和蒸馏 |
| TRL | "偏好调优器" | Hugging Face 的 DPO、GRPO、PPO 库 |
| GRPO | "组相对策略优化" | DeepSeek R1 的 RL 配方，带可验证奖励 |
| EAGLE-3 | "推测解码草稿" | 预测接下来 N 个 token 的草稿头；vLLM 用目标模型验证 |
| MOF | "模型开放框架" | 2026 年数据、代码、许可证模型发布分级标准 |
| 污染检查 | "分裂卫生" | 基于 MinHash 的测试集泄露到训练集的检测 |
| 接受率 | "EAGLE / MTP 指标" | 目标模型接受的草稿 token 比例 |

## 延伸阅读

- [Axolotl 文档](https://axolotl-ai-cloud.github.io/axolotl/) — 参考 SFT / DPO 训练器
- [TRL 文档](https://huggingface.co/docs/trl) — DPO 和 GRPO 参考实现
- [Unsloth](https://github.com/unslothai/unsloth) — 单 GPU 迭代参考
- [DeepSeek R1 论文 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — GRPO 方法论
- [vLLM + EAGLE-3 文档](https://docs.vllm.ai) — 参考服务栈
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 替代推测解码训练器
- [Model Openness Framework 2026](https://isocpp.org/) — 开放发布分级标准
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 规范评估运行器