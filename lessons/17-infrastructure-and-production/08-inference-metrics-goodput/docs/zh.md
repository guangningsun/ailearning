# 推理指标 — TTFT、TPOT、ITL、Goodput、P99

> 四个指标决定推理部署是否正常运转。TTFT 是 prefill 加上排队加上网络。TPOT（即 ITL）是每个 token 的内存受限解码成本。端到端延迟是 TTFT 加上 TPOT 乘以输出长度。吞吐量是跨集群聚合的每秒 token 数。但对产品而言真正重要的是 goodput —— 同时满足所有 SLO 的请求比例。 高吞吐量低 goodput 意味着你处理的 token 从未及时到达用户。2026 年 TRT-LLM 上 Llama-3.1-8B-Instruct 的参考数值：平均 TTFT 162 ms，平均 TPOT 7.33 ms，平均 E2E 1,093 ms。永远报告 P50、P90、P99 —— 永远不要只报均值。还要警惕测量陷阱：GenAI-Perf 在 ITL 计算中不含 TTFT，LLMPerf 包含它；两个工具对同一轮运行的 TPOT 结论不一致。

**类型：** 学习型
**语言：** Python（标准库、toy 百分位计算器和 goodput 报告器）
**前置条件：** 阶段 17 · 04（vLLM 服务内幕）
**时间：** 约 60 分钟

## 学习目标

- 精确定义 TTFT、TPOT、ITL、E2E、throughput 和 goodput，并说出每个指标衡量的是什么。
- 解释为什么均值是 LLM 服务的错误统计量，以及如何阅读 P50/P90/P99。
- 构建一个 SLO 多约束（例如 TTFT<500 ms 且 TPOT<15 ms 且 E2E<2 s），并针对它计算 goodput。
- 说出两个对同一轮运行 TPOT 结论不一致的基准工具，并解释原因。

## 问题

"我们的吞吐量是每秒 15,000 token。"那又怎样？如果 40% 的请求端到端超过 2 秒，用户就会放弃会话。单纯看吞吐量无法告诉你产品是否正常运转。

推理有多个维度的延迟，每个维度以不同方式失效。Prefill 是计算密集的，随 prompt 长度缩放。Decode 是内存密集的，随 batch size 缩放。排队延迟是运营问题。网络是物理距离问题。你需要对每个维度使用不同的指标，还需要百分位，更需要一个复合指标来说"用户是否得到了他期望的东西" —— 这就是 goodput。

## 概念

### TTFT — 首个 token 的时间

`TTFT = queue_time + network_request + prefill_time`

当 prompt 很长时，Prefill 占主导。在 H100 上运行 Llama-3.3-70B FP8，32k 的 prompt 纯 prefill 约需 800 ms。Queue time 是负载下调度器的行为。Network request 是包括 TLS 在内的 wire time。TTFT 是用户看到任何内容流回之前的延迟。

### TPOT / ITL — token 间延迟

同一个量有多个名称。`TPOT`（每个输出 token 的时间）、`ITL`（token 间延迟）、`每个 token 的解码延迟` —— 都是一回事。它是首个 token 之后连续流式 token 之间的时间。

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

在同一 Llama-3.3-70B H100 技术栈上启用 chunked prefill，TPOT 均值约 7 ms。若没有 chunked prefill，在相邻序列的长 prefill 期间，TPOT 会飙升到 50 ms。关注 P99，而非均值。

### E2E 延迟

`E2E = TTFT + TPOT * output_tokens + network_response`

对于长输出（>500 token），E2E 由 TPOT 主导。对于短输出加长 prompt，E2E 由 TTFT 主导。应按输出长度条件报告 E2E。

### 吞吐量

`throughput = total_output_tokens / elapsed_time`

聚合指标，反映集群效率。不反映单个请求的健康状况。

### Goodput —— 你真正关心的指标

`goodput = 满足 (TTFT <= a) 且 (TPOT <= b) 且 (E2E <= c) 的请求比例`

SLO 是一个多约束。只有每个约束都满足的请求才算"好"。Goodput 就是它的比例。高吞吐量低 goodput 是失败。低吞吐量高 goodput 才是目标。

2026 年，goodput 是 MLPerf Inference v6.0 提交以及 AI 平台提供商内部 SLA 跟踪使用的指标。

### 为什么均值是错误的统计量

LLM 延迟分布是右偏的。一个有相邻序列长 prefill 的解码 batch 可以用 TPOT ~7 ms 发出 500 token，但用 TPOT ~60 ms 发出 20 token。TPOT 均值是 9 ms。TPOT P99 是 65 ms。用户经常遇到 P99 —— 那就是他们离开的原因。

永远报告三连（P50、P90、P99）。对于用户体验，P99 才是你要优化的。

### 参考数值 — 2026 年 TRT-LLM 上的 Llama-3.1-8B-Instruct

- 平均 TTFT：162 ms
- 平均 TPOT：7.33 ms
- 平均 E2E：1,093 ms
- P99 TPOT：因 chunked prefill 配置不同，波动于 10-25 ms。

这些是已发布的 NVIDIA 参考点。它们随模型大小（70B 会显示 3-5 倍）、硬件（H100 vs B200 约 3 倍）和负载变化。

### 测量陷阱

2026 年最常用的两个基准工具对同一轮运行的 TPOT 结论不一致：

- **NVIDIA GenAI-Perf**：在 ITL 计算中不含 TTFT。ITL 从 token 2 开始计算。
- **LLMPerf**：包含 TTFT。ITL 从 token 1 开始计算。

对于 TTFT 500 ms、100 个输出 token、总解码时间 700 ms 的请求，GenAI-Perf 报告 `ITL = 700/99 = 7.07 ms`，LLMPerf 报告 `ITL = 1200/100 = 12.00 ms`。工具选择不同，数字就不同。

永远要说明使用哪个工具。永远要公布其定义。

### 构建 SLO

2026 年 70B 聊天模型面向消费者的合理 SLO：

- TTFT P99 <= 800 ms。
- TPOT P99 <= 25 ms。
- E2E P99 <= 3 s（对于 <300 token 的输出）。
- Goodput 目标 >= 99%。

企业 SLO 收紧 TTFT（200-400 ms）并放宽 E2E。关键是把它写下来，测量全部三个，并将 goodput 作为单一复合指标来跟踪。

### 如何测量

- 运行真实流量或真实感 synthetic（LLMPerf 用 `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`）。
- 基准运行目标为峰值并发量的 2 倍。
- 运行 30-50 轮，取合并样本的百分位。
- 发布时注明工具名、工具版本、模型、硬件、并发量、prompt 分布。

## 实际使用

`code/main.py` 是一个 toy goodput 计算器。生成一个 synthetic 延迟分布，应用 SLO，然后计算 goodput。也展示同一 trace 上 GenAI-Perf vs LLMPerf 的 TPOT 差异。

## 交付物

本课产出 `outputs/skill-slo-goodput-gate.md`。给定工作负载和 SLO，它产生一个 CI/CD 可用的基准配方，用 goodput 而非吞吐量来拦截部署。

## 练习

1. 运行 `code/main.py`。生成一个有 1% 尾部尖峰的分布。当你把 P99 TPOT 从 30 ms 收紧到 15 ms 时，goodput 如何变化？
2. 供应商报价"Llama 3.3 70B H100 上 15,000 tok/s"。在信任它之前要问哪三个问题？
3. 为什么 chunked prefill 保护 P99 TPOT 但不保护均值 TPOT？
4. 为语音助手构建一个消费者 SLO（首个 token 是被听到的，而非被读到的）。哪个指标用户感知最强？
5. 阅读 LLMPerf README 和 GenAI-Perf 文档。找出两个工具在其他三个指标上不一致的地方。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| TTFT | "首个 token 时间" | 排队 + 网络 + prefill；在长 prompt 下由 prefill 主导 |
| TPOT | "每个输出 token 的时间" | 首个之后的每个 token 的内存受限解码成本 |
| ITL | "token 间延迟" | 在大多数工具中与 TPOT 相同（并非全部 —— 见 GenAI-Perf） |
| E2E | "端到端" | TTFT + TPOT * output_len；再加上响应侧网络 |
| Throughput | "tok/s" | 集群效率；没有延迟百分位就毫无意义 |
| Goodput | "SLO 满足率" | 同时满足所有 SLO 约束的请求比例 |
| P99 | "尾部" | 1/100 最差情况延迟；用户体验指标 |
| SLO 多约束 | "联合约束" | 所有三个延迟上界的 AND；任一违规即导致请求失败 |
| GenAI-Perf vs LLMPerf | "工具陷阱" | 工具在 ITL 是否包含 TTFT 上不一致 |

## 延伸阅读

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — TTFT、ITL、TPOT 的规范定义。
- [Anyscale — LLM Serving Benchmarking Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — 替代定义和测量方法。
- [BentoML — LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — 真实部署上的应用测量。
- [LLMPerf](https://github.com/ray-project/llmperf) — 基于 Ray 的开源基准工具。
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) — NVIDIA 的基准工具。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 业界公认的基于 goodput 的基准。