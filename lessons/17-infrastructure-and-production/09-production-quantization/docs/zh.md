# 生产量化 — AWQ、GPTQ、GGUF K-quants、FP8、MXFP4/NVFP4

> 量化格式不是通用选择 —— 它是硬件、服务引擎和工作负载的函数。GGUF Q4_K_M 或 Q5_K_M 主宰 CPU 和边缘，通过 llama.cpp 和 Ollama 交付。GPTQ 在 vLLM 内部当你需要在同一 base 上多 LoRA 时胜出。带 Marlin-AWQ kernel 的 AWQ 在 7B 级模型上提供约 741 tok/s，INT4 格式中 Pass@1 最佳 —— 2026 年数据中心生产默认选择。FP8 在 Hopper、Ada 和 Blackwell 上是中间地带 —— 近无损且广泛支持。NVFP4 和 MXFP4（Blackwell 微缩放）很激进，需要逐块验证。两个坑会咬到团队：校准数据集必须匹配部署域，以及 KV cache 与权重量化是分开的 —— AWQ  lesson "我的模型现在是 4 GB" 忘了生产 batch 大小下还有 10-30 GB 的 KV cache。

**类型：** 学习型
**语言：** Python（标准库、toy 跨格式内存与吞吐量比较）
**前置条件：** 阶段 10 · 13（量化基础）、阶段 17 · 04（vLLM 服务内幕）
**时间：** 约 75 分钟

## 学习目标

- 列举 2026 年六种生产量化格式及其最佳场景。
- 给定硬件（CPU vs GPU、Hopper vs Blackwell）、引擎（vLLM、TRT-LLM、llama.cpp）和工作负载（常规聊天、推理、多 LoRA）选择格式。
- 计算所选格式下节省的权重内存以及 KV cache 未被动过的大小。
- 说出在域内流量上降质量化模型的校准数据集陷阱。

## 问题

量化减少内存和 HBM 带宽，而这正是解码所需要的。FP16 70B 模型是 140 GB 的权重。量化权重到 INT4（AWQ 或 GPTQ）后模型是 35 GB —— 装得下一块 H100 还有空间放 KV cache，这很重要，因为在 128 并发序列 2k 上下文下，KV cache 本身就有 20-30 GB。

但量化不是免费的。激进量化会降质，尤其在重推理任务上。不同格式与不同引擎配合。不同硬件原生支持不同精度。2026 年的格式动物园是真实存在的，你不能照搬别人的选择 —— 你必须根据你的技术栈来选取。

## 概念

### 六种格式

| 格式 | 位宽 | 最佳场景 | 引擎 |
|--------|------|-----------|---------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU、边缘、笔记本 | llama.cpp、Ollama |
| GPTQ | 4-8 | vLLM 上的多 LoRA | vLLM、TGI |
| AWQ | 4 | 数据中心 GPU 生产 | vLLM（Marlin-AWQ）、TGI |
| FP8 | 8 | Hopper/Ada/Blackwell 数据中心 | vLLM、TRT-LLM、SGLang |
| MXFP4 | 4 | Blackwell 多用户 | TRT-LLM |
| NVFP4 | 4 | Blackwell 多用户 | TRT-LLM |

### GGUF — CPU/边缘默认

GGUF 是一个文件格式，而非严格的量化方案 —— 它将 K-quant 变体（Q2_K、Q3_K_M、Q4_K_M、Q5_K_M、Q6_K、Q8_0）打包在一个容器里。Q4_K_M 和 Q5_K_M 是生产默认 —— 4-5 位下接近 BF16 的质量。对于 CPU 或边缘服务来说是最佳选择，因为 llama.cpp 是迄今为止最快的 CPU 推理引擎。

vLLM 中吞吐量损失：7B 上约 93 tok/s —— 该格式未针对 GPU kernel 优化。需要 CPU/边缘部署时才用 GGUF。

### GPTQ — vLLM 上的多 LoRA

GPTQ 是带校准过程的后训练量化算法。Marlin kernel 使其在 GPU 上很快（比非 Marlin GPTQ 快 2.6 倍）。7B 上约 712 tok/s。

独特优势：GPTQ-Int4 在 vLLM 中支持 LoRA 适配器。如果你服务的是一个 base 模型加上 10-50 个微调变体（每个作为 LoRA），GPTQ 是你的路径。NVFP4截至 2026 年初还不支持 LoRA。

### AWQ — 数据中心 GPU 默认

Activation-aware Weight Quantization（激活感知权重量化）。在量化过程中保护约 1% 最显著的权重。Marlin-AWQ kernel：比 naive 实现快 10.9 倍。7B 上约 741 tok/s，INT4 格式中 Pass@1 最佳。

除非你需要多 LoRA（GPTQ）或激进的 Blackwell FP4（NVFP4），否则新 GPU 服务选 AWQ。

### FP8 — 可靠的中间地带

8 位浮点。近无损。广泛支持。Hopper Tensor Core 原生加速 FP8。Blackwell 继承支持。当质量不可妥协时（推理、医疗、代码生成），FP8 是 2026 年安全默认。内存节省是 INT4 的一半，但质量风险也低得多。

### MXFP4 / NVFP4 — Blackwell 激进方案

微缩放 FP4。每块权重有自己的缩放因子。激进但在 Blackwell Tensor Core 上有硬件加速。与 FP8 相比每 token 字节数减半 —— 阶段 17 · 07 中的经济优势。

注意事项：
- 截至 2026 年初尚无 LoRA 支持。
- 在重推理工作负载上可见质量下降。
- 每个模型在评估集上验证。

### 校准陷阱

AWQ 和 GPTQ 需要校准数据集 —— 通常是 C4 或 WikiText。对于域模型（代码、医疗、法律），在通用 web 文本上校准会让算法在哪些权重需要保护上做出错误决策。HumanEval 上 Pass@1 可能下降好几个点。

修复：用域内数据校准。通常几百个域样本就够了。发版前在评估集上测试。

### KV cache 陷阱

AWQ 将权重压缩到 4 位。KV cache 是独立的，保持 FP16/FP8。对于带 AWQ 的 70B 模型：

- 权重：约 35 GB（从 140 GB INT4）。
- KV cache（128 并发 × 2k 上下文）：约 20 GB。
- 激活：约 5 GB。
- 总计：约 60 GB —— 装得下 H100 80GB。

天真地说"我的模型量化到 4 GB"忘了还有另外 30-50 GB。要全局考虑 HBM。

另外，KV cache 量化（FP8 KV 或 INT8 KV）是另一个独立的选择，有自己的权衡 —— 它直接影响注意力精度，不是白捡的便宜。

### AWQ INT4 对推理有危险

思维链、数学、长上下文代码生成 —— 这些在激进量化下会明显受影响。AWQ INT4 在 MATH 上损失约 3-5 个点。对于重推理工作负载，上 FP8 或 BF16；接受内存成本。

### 2026 选型指南

- CPU/边缘服务：GGUF Q4_K_M。就这样。
- GPU 服务，常规聊天，无 LoRA：AWQ。
- GPU 服务，多 LoRA：带 Marlin 的 GPTQ。
- 推理工作负载：FP8。
- Blackwell 数据中心，已验证质量：NVFP4 + FP8 KV。
- 模糊不清：每个候选格式跑 1,000 样本评估。

## 实际使用

`code/main.py` 计算六种格式下各种模型大小的内存占用（权重 + KV + 激活）和相对吞吐量。展示 KV cache 在哪里占主导、权重压缩在哪里值得，以及 FP8 在哪里是安全选择。

## 交付物

本课产出 `outputs/skill-quantization-picker.md`。给定硬件、模型大小、工作负载类型和质量容忍度，选择一种格式并产生校准/验证计划。

## 练习

1. 运行 `code/main.py`。对于 128 并发 2k 上下文的 70B 模型，计算每种格式的总 HBM。哪种格式能装进一块 H100 80GB？
2. 你有一个 7B 编码模型。选择一种格式并说明理由。如果你对质量容忍度的判断错了，恢复路径是什么？
3. 计算为医学域模型校准 AWQ 所需的校准数据集大小。为什么更多数据并不总是更好？
4. 阅读 Marlin-AWQ kernel 论文或发布说明。用三句话解释为什么 AWQ 在 7B 上达到 741 tok/s，而原始 GPTQ 只有约 712。
5. 在什么情况下将 AWQ 权重与 FP8 KV cache 结合使用 vs 保持 KV 在 BF16 有意义？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| GGUF | "llama.cpp 格式" | 打包 K-quant 变体的文件格式；CPU/边缘默认 |
| Q4_K_M | "Q4 K M" | 4 位 K-quant 中等；生产 GGUF 默认 |
| GPTQ | "gee pee tee q" | 带校准的后训练 INT4；vLLM 中支持 LoRA |
| AWQ | "a w q" | 激活感知 INT4；Marlin kernel；INT4 中 Pass@1 最佳 |
| Marlin kernels | "快速 INT4 kernel" | Hopper 上 INT4 的自定义 CUDA kernel；10 倍加速 |
| FP8 | "八位浮点" | Hopper/Ada/Blackwell 上安全的精度默认 |
| MXFP4 / NVFP4 | "微缩放四" | 带逐块缩放因子的 Blackwell 4 位 FP |
| 校准数据集 | "校准数据" | 用于选取量化参数的输入文本；必须匹配域 |
| KV cache 量化 | "KV INT8" | 与权重独立的选择；直接影响注意力精度 |

## 延伸阅读

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) —  comparative benchmarks。
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — 各格式吞吐量数据。
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — 逐格式选型指南。
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) — 支持的格式和标志。
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — 原始 AWQ 公式。
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — 原始 GPTQ 公式。