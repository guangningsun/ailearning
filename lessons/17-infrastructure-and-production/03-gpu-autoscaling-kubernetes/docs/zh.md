# Kubernetes 上的 GPU 自动扩展 — Karpenter、KAI Scheduler、Gang Scheduling

> 三层，不是一层。Karpenter 动态供应节点（不到一分钟，比 Cluster Autoscaler 快 40%）。KAI Scheduler 处理 gang scheduling、拓扑感知和分层队列 — 它防止了 7-of-8 部分分配陷阱，即七个节点等待并在一个缺失 GPU 上燃烧。应用层自动扩展器（NVIDIA Dynamo Planner、llm-d Workload Variant Autoscaler）根据推理特定信号（队列深度、KV cache 利用率）扩展，而不是 CPU/DCGM duty cycle。经典的 HPA 陷阱是 `DCGM_FI_DEV_GPU_UTIL` 是一个 duty cycle 测量：100% 可能是 10 个请求也可能是 100 个请求。vLLM 预分配 KV cache 内存，所以内存永远不会触发缩容。本课教你组合三层并避免默认的 Karpenter `WhenEmptyOrUnderutilized` 策略，该策略会在推理中途终止运行中的 GPU 作业。

**类型：** 学习型
**语言：** Python（标准库、toy 队列深度自动扩展模拟器）
**前置条件：** 阶段 17 · 02（推理平台经济学）、阶段 17 · 04（vLLM 服务内部原理）
**时间：** 约 75 分钟

## 学习目标

- 画出三层自动扩展（节点供应、gang scheduling、应用层）并在每层命名使用的工具。
- 解释为什么 `DCGM_FI_DEV_GPU_UTIL` 是 vLLM 错误的 HPA 信号，并命名两个替代方案（队列深度、KV cache 利用率）。
- 描述 gang scheduling 以及 KAI Scheduler 防止的部分分配失败模式（8 个中的 7 个 GPU 空闲）。
- 说出 Karpenter 的合并策略（`WhenEmptyOrUnderutilized`）会终止运行中的 GPU 作业，并说明 2026 年的安全替代方案。

## 问题

你的团队在 Kubernetes 上发布了一个 LLM serving 服务。你用 `DCGM_FI_DEV_GPU_UTIL` 作为信号设置 HPA。服务在营业时间内固定 100% 利用率。HPA 从不扩展 — 它认为你已经满了。你手动添加一个副本；TTFT 下降。HPA 仍然不扩展。信号在骗你。

另外，你使用 Cluster Autoscaler 管理节点。一个 100 万 token 的 prompt 在凌晨 2 点到达；集群花费 3 分钟供应一个节点，请求超时。

再另外，你部署了一个需要跨 2 个节点 8 个 GPU 的 70B 模型。集群有 7 个 GPU 空闲，1 个分散在 3 个节点上。Cluster Autoscaler 为那 1 个缺失的 GPU 供应一个节点。七个节点等待 4 分钟燃烧金钱，而 Kubernetes 让最后一个 GPU 上线。

三层，三种不同的失败模式。2026 年的 GPU 感知自动扩展不是"打开 HPA"。它是节点供应、gang scheduling 和应用信号自动扩展的组合。

## 概念

### 第 1 层 — 节点供应（Karpenter）

Karpenter 监视待处理 pod 并在约 45-60 秒内供应节点（Cluster Autoscaler 对 GPU 节点通常需要 90-120 秒）。它根据 `NodePool` 约束动态选择实例类型 — 如果你的 pod 需要 8 个 H100 而集群中没有匹配节点，Karpenter 直接供应一个，而不是扩展现有组。

**合并陷阱**：Karpenter 的默认 `consolidationPolicy: WhenEmptyOrUnderutilized` 对 GPU 池很危险。它会终止一个运行中的 GPU 节点以将 pods 迁移到更便宜的适当规模实例。对于推理工作负载，这意味着驱逐运行中的请求并在 새 节点上重新加载 70B 模型。损失是几分钟的容量加上请求失败。

GPU 池的安全设置：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

让 Karpenter 在一小时后合并真正空闲的节点，但永远不会驱逐运行中的作业。

### 第 2 层 — gang scheduling（KAI Scheduler）

KAI Scheduler（项目"Karp"后更名）处理默认 kube-scheduler 不做的事情：

**Gang scheduling** — 全有或全无调度。一个需要 8 个 GPU 的分布式推理 pod 要么 8 个一起启动，要么都不启动。没有这个，你会得到部分分配陷阱：8 个 pod 中的 7 个启动，无限期等待，燃烧金钱。

**拓扑感知** — 知道哪些 GPU 共享 NVLink，哪些在同一 rack 上，哪些之间有 InfiniBand。相应地放置 pods。一个 DeepSeek-V3 67B 张量并行工作负载必须保持在同一 NVLink 域上；KAI Scheduler 尊重这一点。

**分层队列** — 多个团队以优先级和配额竞争同一 GPU 池。团队 A 的生产 pinch 只有在优先级规则允许时才会被团队 B 的训练作业抢占。

KAI 与 kube-scheduler 并行部署作为辅助调度器；你用注解标记工作负载使用它。Ray 和 vLLM production-stack 都集成。

### 第 3 层 — 应用层信号

**HPA 陷阱**：`DCGM_FI_DEV_GPU_UTIL` 是一个 duty cycle 指标 — 它测量 GPU 在每个采样间隔是否在工作。100% 利用率可能意味着 10 个并发请求或 100 个；GPU 两种情况都很忙。按 duty cycle 扩展是盲目扩展。

更糟的是，vLLM 和类似引擎预分配 KV cache 内存（最高 `--gpu-memory-utilization`）。即使在一个请求时，内存使用率也保持在 90% 左右。基于内存的 HPA 永远不会缩容。

**2026 年替代信号**：

- 队列深度（等待 prefill 的请求数）。
- KV cache 利用率（分配给活动序列的 block 比例）。
- 每副本 P99 TTFT（你的 SLA 信号）。
- Goodput（每秒满足所有 SLO 的请求）。

NVIDIA Dynamo Planner 和 llm-d Workload Variant Autoscaler 消费这些信号并扩展副本。它们完全取代 LLM serving 的 HPA。

### 何时使用什么

| 扩展决策 | 工具 |
|----------------|------|
| 添加/移除节点 | Karpenter |
| 调度多 GPU 作业 | KAI Scheduler |
| 添加/移除副本 | Dynamo Planner / llm-d WVA（或基于队列深度的自定义 HPA） |
| 选择 GPU 类型 | Karpenter NodePool |
| 抢占低优先级 | KAI Scheduler 队列 |

### 分解的 prefill/decode 使一切复杂化

如果你运行分解的 prefill/decode（阶段 17 · 17），你有两个具有不同缩放触发器的 pod 类：prefill pod 基于队列深度扩展，decode pod 基于 KV cache 压力扩展。llm-d 将这些作为具有 per-role HPA 的独立 `Services` 公开。不要试图在两者前面放一个 HPA。

### 冷启动在这里也很重要

冷启动缓解（阶段 17 · 10）是节点供应时间变得对用户可见的地方。Karpenter 的 45-60 秒预热加上 20GB 模型加载加上引擎初始化意味着从零开始的请求需要 2-5 分钟。为 SLO 关键路径保持一个预热池（`min_workers=1`），或在应用层使用 Modal 风格的 checkpointing。

### 你应该记住的数字

- Karpenter 节点供应：约 45-60s vs Cluster Autoscaler 约 90-120s（GPU 节点）。
- KAI Scheduler 防止部分分配浪费 — 8 中 7 的陷阱。
- `DCGM_FI_DEV_GPU_UTIL` 作为 HPA 信号：坏了；使用队列深度或 KV 利用率。
- Karpenter `WhenEmptyOrUnderutilized`：终止运行中的 GPU 作业。推理用 `WhenEmpty + consolidateAfter: 1h`。

## 使用它

`code/main.py` 在突发 GPU 工作负载上模拟三层自动扩展器。比较朴素 HPA（duty cycle）、队列深度 HPA 和 KAI-gang-scheduled 扩展。报告未满足的请求、空闲 GPU 分钟数和综合评分。

## 交付它

本课产出 `outputs/skill-gpu-autoscaler-plan.md`。给定集群拓扑、工作负载形状和 SLO，设计一个三层自动扩展计划。

## 练习

1. 运行 `code/main.py`。在突发工作负载下，朴素 duty-cycle HPA 丢弃了多少请求，而队列深度 HPA 捕获了？差异来自哪里？
2. 为在 H100 SXM5 上服务 Llama 3.3 70B FP8 的集群设计一个 Karpenter NodePool。指定 `capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter`，以及一个 taint 以保持非 GPU 工作负载远离这些节点。
3. 你的团队报告部署卡在 Pending，因为"GPU 可用但 pod 无法调度"。诊断 — 这是 Karpenter、kube-scheduler 还是 KAI Scheduler？哪些指标确认？
4. 选择一个信号来自动扩展分解的 prefill pod，为 decode pod 选择一个不同的信号。为两者辩护。
5. 计算 24x7 生产服务上 `WhenEmptyOrUnderutilized` 合并陷阱的成本，该服务平均每天 60 个 P99 TTFT > 10s 的请求丢弃事件。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Karpenter | "节点供应器" | Kubernetes 节点自动扩展器；亚分钟供应 |
| Cluster Autoscaler | "旧的扩展器" | Kubernetes 节点自动扩展器前身；较慢，基于组 |
| KAI Scheduler | "GPU 调度器" | 用于 gang + 拓扑 + 队列的辅助调度器 |
| Gang scheduling | "全有或全无" | 原子调度 N 个 pods 或推迟所有 |
| 拓扑感知 | "rack 感知" | 基于 NVLink/IB/rack 放置放置 pods |
| `DCGM_FI_DEV_GPU_UTIL` | "GPU 利用率" | Duty cycle 指标；不是 LLM 的扩展信号 |
| 队列深度 | "等待的请求" | prefill 绑定扩展的正确 HPA 信号 |
| KV cache 利用率 | "内存压力" | decode 绑定扩展的正确 HPA 信号 |
| 合并 | "Karpenter 合并" | 节点终止到更便宜的实例类型 |
| `WhenEmpty + 1h` | "安全合并" | 不会驱逐运行中 GPU 作业的策略 |

## 延伸阅读

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — 设计文档和配置示例。
- [Karpenter 中断控制](https://karpenter.sh/docs/concepts/disruption/) — 合并策略语义和 GPU 安全默认值。
- [NVIDIA — Kubernetes 上的分解式 LLM 推理](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner 扩展信号。
- [Ray 文档 — RayClusters 的 KAI Scheduler](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray 集成模式。
- [AWS EKS 计算和自动扩展最佳实践](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — 托管 Kubernetes 特定指导。
- [llm-d GitHub](https://github.com/llm-d/llm-d) — Workload Variant Autoscaler 设计。