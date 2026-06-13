# LLM API 负载测试 — 为什么 k6 和 Locust 会撒谎

> 传统负载测试工具并非为流式响应、可变输出长度、按 Token 粒度的指标或 GPU 饱和而设计。两个陷阱会让大多数团队踩坑。GIL 陷阱：Locust 的按 Token 粒度测量会在 Python GIL 下运行分词，而 GIL 与重并发下的请求生成存在竞争；分词积压会使报告的 Token 间延迟膨胀——客户端才是瓶颈，而非服务端。Prompt 均匀性陷阱：循环测试中相同的 Prompt 只测试了 Token 分布上的一个点；真实流量有可变长度和多样的前缀匹配。LLMPerf 用 `--mean-input-tokens` + `--stddev-input-tokens` 解决了这个问题。2026 年工具版图：LLM 专用工具（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）用于按 Token 粒度的精确测量；**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**——支持流式感知、Kubernetes 原生、通过 TestRun/PrivateLoadZone CRD 分布式扩展，最适合 CI/CD 门禁；Vegeta 用于 Go 恒定速率饱和测试；Locust 2.43.3 仅在与 LLM-Locust 扩展配合时可用于 LLM。负载模式：稳态、爬坡、脉冲（自动扩缩容测试）、浸泡（内存泄漏）。

**类型：** 构建
**语言：** Python（标准库 + toy 真实感 Prompt 生成器 + 延迟收集器）
**前置条件：** 阶段 17 · 08（推理指标）、阶段 17 · 03（GPU 自动扩缩容）
**时间：** 约 75 分钟

## 学习目标

- 解释两个导致通用负载测试工具对 LLM API 失效的反模式（GIL 陷阱、Prompt 均匀性陷阱）。
- 根据不同用途选择工具：LLMPerf（基准测试运行）、k6 + 流式扩展（CI 门禁）、guidellm（大规模合成）、GenAI-Perf（NVIDIA 参考）。
- 设计四种负载模式（稳态、爬坡、脉冲、浸泡）并说出每个模式能捕获的故障模式。
- 使用输入 Token 的均值 + 标准差构建真实感 Prompt 分布，而非固定长度。

## 问题

你用 k6 对 LLM 端点做了 500 并发用户的压测。扛住了。你上线了。在生产环境中 200 个真实用户服务就崩了——P99 TTFT 暴涨，GPU 打满。

发生了两件事。第一，k6 发送了 500 个完全相同的 Prompt——你的请求合并和前缀缓存让它看起来像是在处理 500 个并发解码，而实际上只处理了一个。第二，k6 不会按人眼体验的方式跟踪流式响应的 Token 间延迟；它看到的是一个 HTTP 连接，而非 500 个以不同间隔到达的 Token。

LLM 的负载测试是一个独立的学科。

## 概念

### GIL 陷阱（Locust）

Locust 使用 Python 并在 GIL 下运行客户端分词。在高并发下，分词器排在请求生成之后。报告的 Token 间延迟包含了客户端分词积压。你以为服务端慢；其实是测试工具本身的问题。

修复：LLM-Locust 扩展将分词移至独立进程，或使用编译型语言工具（k6、使用 tokenizers.rs 的 LLMPerf）。

### Prompt 均匀性陷阱

所有已知的负载测试工具都只允许配置一个 Prompt。在 10,000 次迭代的循环测试中，每次都发送完全相同的 Prompt。服务端每次看到相同的前缀——前缀缓存命中率接近 100%，吞吐量看起来很漂亮。

修复：从 Prompt 分布中采样。LLMPerf 使用 `--mean-input-tokens 500 --stddev-input-tokens 150`——多样化的长度、多样的内容。

### 四种负载模式

1. **稳态** — 30-60 分钟恒定 RPS。捕获：基线性能回归。
2. **爬坡** — 15 分钟内从 0 线性增加到目标 RPS。捕获：容量断点、预热异常。
3. **脉冲** — 突然 3-10 倍 RPS 持续 2 分钟然后恢复。捕获：自动扩缩容延迟、队列饱和、冷启动影响。
4. **浸泡** — 稳态持续 4-8 小时。捕获：内存泄漏、连接池漂移、可观测性溢出。

### 2026 年工具版图

**LLMPerf**（Anyscale）——Python 但后端用 Rust 分词。均值/标准差 Prompt。支持流式感知。最适合作为性能运行的默认选择。

**NVIDIA GenAI-Perf**——NVIDIA 的参考工具。使用 Triton client；指标覆盖全面。注意其 ITL 不含 TTFT；LLMPerf 的包含。两个工具对同一服务端产生不同的 TPOT。

**LLM-Locust**（TrueFoundry）——修复了 GIL 陷阱的 Locust 扩展。熟悉的 Locust DSL + 流式指标。

**guidellm**——大规模合成基准测试。

**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**：
- k6 本身（Go，编译，无 GIL）增加了流式感知指标。
- k6 Operator 使用 TestRun / PrivateLoadZone CRD 实现 Kubernetes 原生分布式测试。
- 最适合 CI/CD 门禁和 SLA 测试。

**Vegeta**——Go，比 k6 更简单。恒定速率 HTTP 饱和。不感知 LLM 但适合网关/限流测试。

**Locust 2.43.3 原版**——对 LLM 有 GIL 陷阱。仅在与 LLM-Locust 扩展配合时可用。

### CI 中的 SLA 门禁

在 PR 上运行 k6：

- 基线 RPS 下各跑 30-50 次迭代。
- 门禁：P50/P95 TTFT、5xx < 5%、TPOT 低于阈值。
- 超阈值时打断构建。

### 真实感 Prompt 分布

从真实流量样本构建（如果有），或从已发布的分布构建（例如 ShareGPT Prompt 用于聊天、HumanEval 用于代码）。将均值 + 标准差喂给 LLMPerf。绝对避免单 Prompt 循环。

### 需要记住的数字

- k6 Operator 1.0 GA：2025 年 9 月。
- k6 v2026.1.0：支持流式感知指标。
- 典型 LLMPerf 运行：100-1000 请求，并发度 X。
- 典型 CI 门禁：每个 PR 30-50 次迭代。
- 四种模式：稳态、爬坡、脉冲、浸泡。

## 使用它

`code/main.py` 用真实感 Prompt 分布模拟负载测试，测量有效 TPOT，并演示均匀 Prompt 陷阱。

## 交付它

本课产出 `outputs/skill-load-test-plan.md`。给定工作负载和 SLA，选择工具并设计四种负载模式。

## 练习

1. 运行 `code/main.py`。比较均匀分布 vs 真实感分布——差距在哪里？
2. 为 CI 门禁编写 k6 脚本：100 并发时 P95 TTFT < 800 ms，运行时间 5 分钟。
3. 你的浸泡测试显示内存每小时增长 50 MB。说出一个原因及用于区分的 instrumentation。
4. 从 10 RPS 脉冲到 100 RPS。如果 Karpenter + vLLM 生产栈已就位（阶段 17 · 03 + 18），预期恢复时间是多少？
5. GenAI-Perf 报告 TPOT=6ms；LLMPerf 报告同一服务端 TPOT=11ms。解释原因。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| LLMPerf | "LLM 工具" | Anyscale 基准测试工具，支持流式感知 |
| GenAI-Perf | "NVIDIA 工具" | NVIDIA 参考工具 |
| LLM-Locust | "LLM 版 Locust" | 修复 GIL 陷阱的 Locust 扩展 |
| guidellm | "合成基准" | 大规模合成工具 |
| k6 Operator | "K8s k6" | 基于 CRD 的分布式 k6 |
| GIL 陷阱 | "Python 客户端开销" | 分词积压使报告延迟膨胀 |
| Prompt 均匀性陷阱 | "单 Prompt 谎言" | 循环相同 Prompt 命中缓存，使吞吐量膨胀 |
| 稳态 | "恒定负载" | N 分钟平面 RPS |
| 爬坡 | "线性增长" | 在持续时间内从 0 到目标 |
| 脉冲 | "突发测试" | 突然倍数然后恢复 |
| 浸泡 | "长测试" | 数小时检测泄漏 |

## 延伸阅读

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)