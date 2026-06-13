# 分解式 Prefill/Decode — NVIDIA Dynamo 与 llm-d

> Prefill 是计算密集型；Decode 是内存密集型。两者跑在同一块 GPU 上会浪费一种资源。分解将它们拆分到不同的池中，通过 NIXL（RDMA/InfiniBand 或 TCP 回退）在池之间传输 KV 缓存。NVIDIA Dynamo（GTC 2025 宣布，1.0 GA）位于 vLLM/SGLang/TRT-LLM 上方——其 Planner Profiler + SLA Planner 自动匹配 prefill:decode 比例以满足 SLO。NVIDIA 发布的吞吐量提升在此范围内——developer.nvidia.com（2025-06）显示在 GB200 NVL72 + Dynamo 环境下 DeepSeek-R1 MoE 中等延迟下约提升 6 倍，Dynamo 产品页面（developer.nvidia.com，未标注日期）宣传 GB300 NVL72 + Dynamo 相比 Hopper MoE 吞吐量提升高达 50 倍。"30x" 是社区在完整 Blackwell + Dynamo + DeepSeek-R1 报道中的聚合数字；我们没有找到单一原始来源精确说明 30x，所以将其视为方向性声明。llm-d（Red Hat + AWS）是 Kubernetes 原生：prefill / decode / router 作为独立服务，有基于角色的 HPA。llm-d 0.5 新增分层 KV 卸载、缓存感知 LoRA 路由、UCCL 网络、缩容至零。经济学上：综合多个客户披露的内部分析，从共置服务切换到带 Dynamo 的分解式服务，在 SLA 不变的情况下，200 万美元级别推理支出的客户可节省 30-40%（即 $600-800K/年）；具体的 $2M→$600-800K 数字是内部综合值，不是单一公开发布的案例研究——将其作为数量级锚点，而非参考引用。短提示（<512 token，短输出）不值得承担传输成本。

**类型：** 学习型
**语言：** Python（标准库 + 玩具分解式 vs 共置模拟器）
**前置条件：** 第 17 阶段 · 04（vLLM 服务内部原理）、第 17 阶段 · 08（推理指标）
**时间：** 约 75 分钟

## 学习目标

- 解释为什么 prefill 和 decode 有不同的最优 GPU 分配，并量化共置下的浪费。
- 画出分解式架构图：prefill 池、decode 池、通过 NIXL 的 KV 传输、路由器。
- 说出在什么条件下分解式**不值得**（短提示、短输出）。
- 区分 NVIDIA Dynamo（栈上层）和 llm-d（Kubernetes 原生），并匹配到各自适用的运维场景。

## 问题

你在 8 张 H100 上跑 Llama 3.3 70B。混合工作负载下（长提示 + 短输出），GPU 在 decode 期间空闲，因为大部分计算已经花在 prefill 上了。另一种工作负载下（短提示 + 长输出），情况相反。共置 prefill + decode 意味着你两个都过度配置了。

预算影响：20-40% 的 GPU 时间浪费在错误的资源上。你买 H100 计算力来跑内存密集型的 decode，或者买 H100 HBM 带宽来跑计算密集型的 prefill。都是昂贵的浪费。

分解将 prefill 和 decode 拆分到独立池中，按各自瓶颈来调整规模。KV 缓存通过高带宽互联从 prefill 池传输到 decode 池。

## 概念

### 为什么瓶颈不同

**Prefill**——在一个前向中运行完整输入提示的 transformer。矩阵乘法占主导；计算密集型。H100 FP8 提供约 2000 TFLOPS 的有效吞吐。批效率好——一个前向处理很多 token。

**Decode**——每次生成一个 token，每次迭代读取完整权重。内存带宽密集型。HBM3 提供约 3 TB/s。只有在高并发时批效率才好——权重读取在批次间摊销。

共置它们：你买的 GPU 要同时优化两者。H100 两者都能做但成本一样。规模化时，你希望 prefill 池用 H100 / 计算密集型；decode 池用 H200 / 内存密集型，或者用激进的量化。

### 架构

```
            ┌──────────────┐
   请求 →   │    路由器    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ （仅提示）                      │
            ┌──────────────┐    KV 缓存    ┌───────▼──────┐
            │ Prefill 池   │ ─── NIXL ───► │  Decode 池   │
            │  （计算）     │               │  （内存）    │
            └──────────────┘               └──────┬───────┘
                                                   │ token
                                                   ▼
                                                 客户端
```

NIXL 是 NVIDIA 的节点间传输协议。有 RDMA/InfiniBand 时用 RDMA，否则用 TCP 回退。传输延迟是真实存在的——70B FP8 上 4K token 提示的 KV 缓存通常需要 20-80 ms。这就是为什么短提示不值得分解：传输税超过节省。

### Dynamo vs llm-d

**NVIDIA Dynamo**（GTC 2025 宣布，1.0 GA）：
- 位于 vLLM、SGLang、TRT-LLM 上方作为编排器。
- Planner Profiler 测量工作负载，SLA Planner 自动配置 prefill:decode 比例。
- Rust 核心，Python 可扩展性。
- 吞吐量提升：NVIDIA 报告 GB200 NVL72 + Dynamo 环境下 DeepSeek-R1 MoE 中等延迟下提升 6 倍（developer.nvidia.com，2025-06）；完整 Blackwell + Dynamo + DeepSeek-R1 栈的社区"高达 30x"报告缺乏单一原始来源，应视为方向性参考。
- GB300 NVL72 + Dynamo：据 Dynamo 产品页面宣传，相比 Hopper MoE 吞吐量提升高达 50 倍（developer.nvidia.com，未标注日期）。

**llm-d**（Red Hat + AWS，Kubernetes 原生）：
- Prefill / decode / router 作为独立的 Kubernetes 服务。
- 基于队列深度（prefill）/ KV 利用率（decode）信号的角色级 HPA。
- `topologyConstraint packDomain: rack` 将 prefill+decode  clique 打包在同一机架上以实现高带宽 KV 传输。
- llm-d 0.5（2026）：分层 KV 卸载、缓存感知 LoRA 路由、UCCL 网络、缩容至零。

如果想要托管的栈上层编排器，用 Dynamo。如果想要 Kubernetes 原生原语并深耕 CNCF 生态，用 llm-d。

### 经济学

内部综合（不是单一公开发布的案例研究——数量级锚点）：

- 共置服务上每年 $2M 推理支出。
- 切换到带 Dynamo 的分解式服务。
- 请求量不变，P99 延迟 SLA 不变。
- 报告的节省：$600K–$800K/年（减少 30–40%）。
- 无新硬件。

我们从多个客户披露综合出这个数字，而不是单一可引用的案例研究；最接近的已发布数据点是 Baseten 的 Dynamo KV 路由 TTFT 快 2 倍 / 吞吐量提升 61%（baseten.co，2025-10），以及 VAST + CoreWeave 的 40–60% KV 命中率下每美元 token 增加 60–130% 的预测（vastdata.com，2025-12）。节省来自每个池的正确规模；prefill 密集型工作负载（RAG 带 8K+ 前缀）比均衡工作负载受益更多。

### 什么时候**不要**分解

- 提示 < 512 token 且输出 < 200 token：传输税占主导。
- 小集群（< 4 GPU）：池多样性不足。
- 团队无法运维两个 GPU 池并各自独立扩缩：Dynamo 有帮助但不是零摩擦。
- 无 RDMA fabric：TCP 传输税更重。

### 路由器与第 17 阶段 · 11 集成

分解式路由器是 KV 缓存感知的（第 17 阶段 · 11）。请求落在持有其前缀的 decode 池上——如果没有匹配，则流向 prefill → decode。命中率和分解是叠加的——缓存感知路由器决定是否甚至需要新的 prefill。

### Blackwell 上的 MoE 才是真正出数字的地方

GB300 NVL72 + Dynamo 相比 Hopper 基线显示 MoE 吞吐量提升 50 倍。MoE 专家路由在 prefill 上计算密集，但在 decode 上（专家缓存）内存密集，所以分解是双重收益。2026 年前沿模型服务以 MoE 为主（DeepSeek-V3、未来 GPT-5 变体）。

### 你应该记住的数字

基准数字会漂移——NVIDIA 和推理栈每季度发布更新结果。引用前请重新核实。

- DeepSeek-R1 在 GB200 NVL72 + Dynamo 上：中等延迟 regime 下相比基线约 6 倍吞吐量（developer.nvidia.com，2025-06）；完整 Blackwell + Dynamo 栈上社区"高达 30x"是方向性聚合，无单一原始来源。
- GB300 NVL72 + Dynamo：相比 Hopper MoE 吞吐量提升高达 50 倍（developer.nvidia.com，未标注日期）。
- 节省锚点（内部综合，非单一案例研究）：SLA 不变的情况下，每年 $2M 支出节省 $600-800K。
- 分解阈值：提示 > 512 token + 输出 > 200 token。
- 通过 NIXL 的 KV 传输：70B FP8 上 4K 提示的 KV 约 20-80 ms。

## 使用

`code/main.py` 模拟共置 vs 分解式服务。报告吞吐量、每请求成本和提示长度交叉点。

## 交付

本课产出 `outputs/skill-disaggregation-decider.md`。给定工作负载和集群，决定是否要分解。

## 练习

1. 运行 `code/main.py`。在什么提示长度下分解胜过共置？
2. 为 P99 前缀长度 8K、输出 300 的 RAG 服务设计 prefill 池和 decode 池。
3. Dynamo vs llm-d：为没有 Python 运行时偏好的纯 Kubernetes 商店选一个。
4. 计算 KV 传输成本：70B FP8 上 4K prefill = 约 500 MB KV。RDMA 100 GB/s 下传输 = 5 ms。TCP 10 GB/s = 50 ms。哪个对你的 SLA 有影响？
5. MoE 专家路由改变 KV 访问模式。带 MoE（每个 token 激活不同专家）的分解式行为如何？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| 分解式服务 | "拆分 prefill/decode" | 每个阶段用独立的 GPU 池 |
| NIXL | "NVIDIA 传输" | Dynamo 的节点间 KV 传输（RDMA/TCP） |
| NVIDIA Dynamo | "编排器" | vLLM/SGLang/TRT-LLM 上方的栈上层协调器 |
| llm-d | "Kubernetes 原生" | Red Hat + AWS K8s 分解式栈 |
| Planner Profiler | "Dynamo 自动配置" | 测量工作负载，配置池比例 |
| SLA Planner | "Dynamo 策略" | 自动匹配 prefill:decode 以满足 SLO |
| `packDomain: rack` | "llm-d 拓扑" | 将 prefill+decode 打包在同一机架以实现快速 KV |
| UCCL | "统一集合" | llm-d 0.5 网络层，支持缩容至零 |
| MoE 专家路由 | "每个 token 专家" | DeepSeek-V3 模式；分解式对此有帮助 |

## 扩展阅读

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)