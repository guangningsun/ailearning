# TensorRT-LLM on Blackwell with FP8 and NVFP4

> TensorRT-LLM 是 NVIDIA 独占的，但在 Blackwell 上表现最优。在 GB200 NVL72 上配合 Dynamo 编排，SemiAnalysis InferenceX 在 2026 年 Q1-Q2 测得 120B 模型每百万 token 成本 $0.012，而 H100 + vLLM 为 $0.09/M —— 7 倍的经济差距。该技术栈由三种浮点格式叠加构成：FP8 因动态范围需求，仍是 KV cache 和注意力 kernel 的关键；NVFP4（4 位微缩放）处理权重和激活；多 token 预测（MTP）以及分解式 prefill/decode 在此基础上再提升 2-3 倍。第 0 天即可直接加载 FP4 权重，无需后训练转换。2026 年工程团队的痛点：TRT-LLM 是一个封闭的 NVIDIA 技术栈，采用它意味着以可移植性换取吞吐量。在决定之前，先根据你的模型和硬件组合算一笔账。

**类型：** 学习型
**语言：** Python（标准库、toy FP8/NVFP4 内存与成本计算器）
**前置条件：** 阶段 17 · 04（vLLM 服务内幕）、阶段 10 · 13（量化）
**时间：** 约 75 分钟

## 学习目标

- 解释为什么即使权重处于 NVFP4，FP8 仍是 KV cache 和注意力的底线。
- 计算前沿模型在 BF16、FP8 和 NVFP4 下的 HBM 占用，并分析节省的空间从何而来。
- 列举 TRT-LLM 所利用的 Blackwell 特有功能（day-0 FP4、MTP、分解式服务、all-to-all 原语）。
- 决定 TRT-LLM 的 NVIDIA 锁定在面对 Hopper 上 vLLM 的 7 倍成本差距时是否值得。

## 问题

2026 年推理经济的核心问题是"每美元能买多少 token"。答案取决于四个叠加的选择：硬件代际（Hopper H100/H200 vs Blackwell B200/GB200）、精度（BF16 → FP8 → NVFP4）、服务引擎（vLLM vs SGLang vs TRT-LLM）以及编排方式（普通 vs 分解式 vs Dynamo）。

在 Hopper 上用 vLLM，120B MoE 每百万 token 成本约 $0.09。在 Blackwell 上用 TRT-LLM + Dynamo，同样的模型成本约 $0.012 —— 便宜 7 倍。差距一部分来自硬件（Blackwell 每 GPU LLM 吞吐量是 Hopper 的 11-15 倍），一部分来自技术栈：FP4 权重、MTP 草稿、分解式 prefill/decode，以及用于 MoE 专家通信的 NVLink 5 all-to-all。

在 NVIDIA 技术栈之外无法复现这一点。这就是权衡 —— 以可移植性换经济性。理解每个技术栈选择对整体差距的贡献，正是本课的目的。

## 概念

### 为什么 FP8 仍是 KV cache 的底线

2026 年一个常见错误：以为 NVFP4 可以处处适用。并非如此。KV cache 需要 FP8（8 位浮点），因为它存储的注意力 keys 和 values 跨越很宽的动态范围。量化 KV 到 FP4 会导致灾难性的精度损失 —— 分布的尾部被截断，注意力分数崩塌。FP8 的指数位赋予 KV cache 所需的范围。

NVFP4（2025-2026）适用于权重和激活。微缩放：每块权重有自己的缩放因子，使小块可以跨越不同的动态范围，而不会因逐张量缩放而损失精度。对于激活，FP4 表现良好，因为激活在层内是小范围的。

典型的 Blackwell 配置：

- 权重：NVFP4（4 位微缩放）。
- 激活：NVFP4。
- KV cache：FP8。
- 注意力累加器：FP32（softmax 稳定性）。

### TRT-LLM 使用的 Blackwell 特有原语

- **Day-0 FP4 权重**：模型提供商直接发布 FP4 权重；TRT-LLM 无需后训练转换即可加载。FP4 无需 AWQ / GPTQ 步骤。
- **多 token 预测（MTP）**：与 EAGLE（阶段 17 · 05）思路相同，但集成在 TRT-LLM 构建中。
- **分解式服务**：prefill 和 decode 在独立的 GPU 池上运行，KV cache 通过 NVLink 或 InfiniBand 传输。与 Dynamo 思路相同（阶段 17 · 20）。
- **All-to-all 通信原语**：NVLink 5 将 MoE 专家通信延迟削减至 Hopper 的 1/3。TRT-LLM 的 MoE kernel 针对此做了优化。
- **NVFP4 + MXFP8 微缩放**：Blackwell Tensor Core 上硬件加速的缩放因子处理。

### 你应该记住的数字

- HGX B200 通过 TRT-LLM 运行 GPT-OSS-120B，成本 $0.02/M token。
- GB200 NVL72 通过 Dynamo（编排 TRT-LLM）运行，成本 $0.012/M token。
- H100 + vLLM 在类似负载下 ≈ $0.09/M token。
- TRT-LLM 三个月更新实现 2.8 倍吞吐量提升（2026 年）。
- Blackwell vs Hopper 每 GPU LLM 吞吐量：11-15 倍。
- MLPerf Inference v6.0（2026 年 4 月）：Blackwell 在所有提报任务中占据主导。

### FP4 的实际质量代价

NVFP4 很激进。在重推理负载（思维链、数学、长上下文代码生成）上，FP4 权重会导致明显降质。逐块校准可以缓解但无法消除。发布推理模型的团队通常采用 FP8 权重 + FP4 激活作为折中，或坚持使用全 FP8 的 H200。

原则：在承诺 NVFP4 权重之前，先在评估集上验证任务质量。

### 为什么这是一个 NVIDIA 锁定的决策

TRT-LLM 是 C++ + CUDA + 闭源 kernel。模型需要为特定 GPU SKU 编译。不支持 AMD、Intel 或 ARM。如果你的基础设施战略是多供应商，TRT-LLM 对于 TRT-LLM 服务层是不可行的 —— 你仍然可以在混合硬件上从 vLLM 服务。如果你是纯 NVIDIA，7 倍差距足以覆盖锁定成本。

### 2026 实用配方

对于年度推理账单超过 $100M 的情况，在 Hopper + vLLM 上运行等于把 7-10 倍的收益留在桌上。将成本主导的工作负载迁移到 Blackwell + TRT-LLM + Dynamo。将实验层保留在 H100 + vLLM 上以加快模型迭代速度。在生产前对每个 NVFP4 转换的模型验证质量。

### 分解式服务的额外收益

TRT-LLM 的分解式服务（独立的 prefill 和 decode 池）在阶段 17 · 20 中有深入讨论。在 Blackwell 上，收益叠加：FP4 权重 × MTP 加速 × 分解式放置 × 缓存感知路由。7 倍的数字假设使用了完整技术栈。

## 实际使用

`code/main.py` 计算 HBM 占用、解码吞吐量（内存带宽受限区间）以及三种技术栈下模型的每百万 token 成本：H100 + BF16 + vLLM、H100 + FP8 + vLLM、B200 + NVFP4/FP8 + TRT-LLM。运行它可以看到叠加效应以及每个变化对差距的贡献。

## 交付物

本课产出 `outputs/skill-trtllm-blackwell-advisor.md`。给定工作负载、模型大小和年度 token 量，它决定 Blackwell + TRT-LLM 技术栈是否值得 NVIDIA 锁定。

## 练习

1. 运行 `code/main.py`。对于 30% 活跃参数的 120B MoE，计算 H100 BF16、H100 FP8 和 B200 NVFP4/FP8 下受内存带宽限制的解码吞吐量。最大的跳跃来自哪里？
2. 客户每年在 H100 + vLLM 上花费 $2M。给定 7 倍的经济差距，他们需要购买多少 Blackwell GPU 才能在 12 个月内摊销迁移到 TRT-LLM 的成本？
3. 你看到 NVFP4 权重转换后 MATH 精度下降 3 个点。说出两条恢复路径：一条质量优先（保持 FP8 权重），一条成本优先（用域内数据校准）。
4. 阅读 MLPerf v6.0 推理结果。哪个任务的 Blackwell-over-Hopper 差距最小，为什么？
5. 计算 405B 模型在 NVFP4 权重 + FP8 KV cache + 128k 上下文下所需的 HBM。它能装进单个 GB200 NVL72 节点吗？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| FP8 | "八位浮点" | 8 位浮点；因动态范围需求用于 KV cache 和注意力 |
| NVFP4 | "四位微" | NVIDIA 的 4 位微缩放 FP 格式；Blackwell 上的权重和激活 |
| MXFP8 | "MX 八" | 微缩放 FP8 变体；在 Blackwell Tensor Core 上硬件加速 |
| Day-0 FP4 | "直接发布 FP4 权重" | 模型提供商直接发布 FP4 格式的权重；无需后训练转换步骤 |
| MTP | "多 token 预测" | TRT-LLM 集成的推测解码草稿（阶段 17 · 05） |
| 分解式服务 | "分离 prefill/decode" | Prefill 和 decode 在独立 GPU 池上运行；KV 通过 NVLink/IB 传输 |
| All-to-all | "MoE 专家通信" | 将 token 路由到专家 GPU 的通信模式；NVLink 5 将延迟削减至 1/3 |
| InferenceX | "SemiAnalysis 推理基准" | 2026 年业界公认的每 token 成本基准 |

## 延伸阅读

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — 2026 年 4 月 MLPerf 结果。
- [NVIDIA — MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 all-to-all 和 MoE kernel。
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html) — 官方引擎文档。
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — TRT-LLM 之上的分解式编排。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 发布 Blackwell 数据的基准套件。