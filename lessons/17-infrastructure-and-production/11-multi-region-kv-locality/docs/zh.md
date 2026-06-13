# 多区域 LLM 服务与 KV 缓存本地性

> 轮询负载均衡对缓存的 LLM 推理主动有害。请求如果没有落在持有其前缀的节点上，将支付完整 prefill 成本——在长提示词上 P50 约 800 ms，而缓存命中约 80 ms。到 2026 年，生产模式是缓存感知路由器（vLLM Router in Rust，llm-d router），它消费 KV 缓存事件并按前缀哈希匹配路由。最近的研究（GORGO）将跨区域网络延迟作为路由目标中的显式项。商业"跨区域推理"产品（Bedrock 跨区域推理，GKE 多集群网关）将推理视为不透明——它们处理可用性，不处理 TTFT。摩根大通和 Mayo Clinic 在 2024 年 11 月 us-east-1 故障转移用时约 22 分钟。DR 的现实：32% 的 LLM DR 失败是因为团队备份了权重但忘记了 tokenizer 文件或量化配置。

**类型：** 学习型
**语言：** Python（标准库，含 toy 前缀缓存感知路由器模拟器）
**前置条件：** 阶段 17 · 04（vLLM 服务）、阶段 17 · 06（SGLang RadixAttention）
**时间：** 约 60 分钟

## 学习目标

- 解释为什么轮询负载均衡破坏缓存推理，并量化 TTFT 惩罚。
- 画出缓存感知路由器的架构图：输入（KV 缓存事件）、算法（前缀哈希匹配）、平局决胜（GPU 利用率）。
- 说出 32% LLM DR 失败的原因（缺少 tokenizer 文件 / 量化配置），并给出三文件 DR 检查清单。
- 区分商业跨区域产品（Bedrock CRI，GKE 多集群网关）与 KV 感知路由。

## 问题

你的服务运行在 us-east-1、us-west-2 和 eu-west-1。你在前面放了一个 ALB 做轮询。生产中前缀缓存命中率降至 8%。TTFT P50 翻了三倍。你的 vLLM 日志显示每个请求都在支付完整 prefill 成本。

轮询对无状态服务是最优的。LLM 推理本质是有状态的——KV 缓存编码了模型所见的一切。盲目路由就是路由到错误的缓存。

另外，你的团队有一个 DR 计划。你将模型权重备份到跨区域 S3。区域中断发生；你尝试故障转移；副本拒绝启动。你忘了 tokenizer.json、量化配置和 RoPE 缩放配置在另一个你没有同步的桶里。

多区域 LLM 服务是一个缓存问题、路由问题和 DR 卫生问题——不是负载均衡器问题。

## 概念

### 缓存感知路由

请求带着提示词到达。路由器对前缀取哈希（比如前 512 个 token）；询问每个副本"你有这个前缀的缓存吗？"。副本在分配和驱逐块时通过发布-订阅通道发布 KV 缓存事件。路由器选择匹配的副本；如果没有人匹配则回退到基于 GPU 利用率的平局决胜。

**vLLM Router**（Rust，2026 生产栈）：订阅 `kv.cache.block_added` 事件，维护前缀哈希→副本索引，以 O(1) 查询路由。没有匹配时回退到最少队列深度。

**llm-d router**：相同模式，Kubernetes 原生。通过 ControlPlane API 发布事件。

**SGLang RadixAttention**（阶段 17 · 06）是副本内的等价物。跨副本路由严格来说是上游。

### 数字

在 2K token 提示词、Llama 3.3 70B FP8、H100 上 TTFT P50：
- 缓存命中（同副本，前缀驻留）：约 80 ms。
- 缓存未命中（冷 prefill）：约 800 ms。

10 倍差距。如果你的路由器在副本间达到 60-80% 的前缀缓存命中率，你以 N 副本容量逼近单副本性能。如果只有 10%，你逼近朴素扩展。

### 跨区域有一个新约束——网络延迟

区域间 RTT：
- us-east-1 ↔ us-west-2：约 65 ms。
- us-east-1 ↔ eu-west-1：约 75 ms。
- us-east-1 ↔ ap-southeast-1：约 220 ms。

如果路由器将请求从 us-east-1 路由到 ap-southeast-1 的热前缀，节省的 prefill（800 → 80 ms）被 440 ms 往返淹没。GORGO（2026 研究）将其显式化——联合最小化 `prefill_time + network_latency`，而不是只看 prefill。通常的答案是保持区域路由，除非是巨大的多 MB 前缀，此时 prefill 占主导。

### 商业"跨区域推理"在这里没有帮助

AWS Bedrock 跨区域推理在容量压力下自动将请求路由到其他区域。它优化可用性，不优化 TTFT，并将推理视为不透明。GKE 多集群网关也一样——服务级故障转移，对 KV 缓存没有感知。

使用这些时你仍然需要一个应用层缓存感知路由器。它们处理"us-east-1 着火"的情况。缓存感知路由处理 TTFT 情况。

### DR 卫生——32% 缺失文件问题

广泛引用的 2026 年统计：32% 的 LLM DR 失败发生是因为团队备份了权重但忘了：

- `tokenizer.json` 或 `tokenizer.model`
- 量化配置（`quantize_config.json`、AWQ 缩放因子、GPTQ 零点）
- 模型特定配置（RoPE 缩放、注意力掩码、聊天模板）
- 引擎配置（`vllm_config.yaml`、采样默认值、LoRA 适配器清单）

修复方法是三文件最低 DR 清单：

1. HF 模型仓库下的所有文件（权重 + 配置 + tokenizer）。
2. 引擎特定的 serving 配置。
3. 部署清单（K8s YAML、Dockerfile、依赖锁）。

加上：每季度做一次 DR 演练。摩根大通 2024 年 11 月 us-east-1 演练之所以能在 22 分钟恢复（30 分钟 SLA），正是因为 playbook 已经演练过。

### 数据驻留是独立的

欧盟客户 PHI 不能离开欧盟。如果你的缓存感知路由器将来自巴黎的请求路由到 us-east-1 寻求前缀匹配，你就违反了 GDPR，不管 TTFT 收益如何。在优化缓存之前先按驻留边界分割路由器。

### 应记住的数字

- 缓存命中 vs 未命中 TTFT 差距：约 10 倍（2K 提示词上 80 ms vs 800 ms）。
- 区域间 RTT 美欧：约 75 ms。
- DR 失败：32% 缺少 tokenizer/量化配置。
- 摩根大通 2024 年 11 月 us-east-1 故障转移：22 分钟（30 分钟 SLA）。

## 使用方法

`code/main.py` 在多区域工作负载上模拟三种路由策略（轮询、缓存感知区域、缓存感知全局）。报告缓存命中率、TTFT P50/P99 和跨区域账单。

## 交付

本课产出 `outputs/skill-multi-region-router.md`。给定区域、驻留约束和 SLA，设计路由计划。

## 练习

1. 运行 `code/main.py`。在什么提示词长度下跨区域路由优于仅本地路由（给定 75 ms RTT）？
2. 你的缓存命中率从 70% 降至 12%。诊断三种可能原因，以及确认每种原因的可观测指标。
3. 为一个 70B AWQ 量化模型在 vLLM 中服务、配 5 个 LoRA 适配器设计 DR 清单。列出每个文件和配置。
4. 论证 Bedrock 跨区域推理对有严格 TTFT SLO 的金融科技公司是否"足够"。引用具体行为。
5. 一个来自巴黎的请求匹配 us-east-1 中的前缀。你路由吗？写出策略。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| 缓存感知路由 | "智能 LB" | 按前缀哈希匹配路由到持有 KV 缓存的副本 |
| KV 缓存事件 | "缓存发布-订阅" | 副本发布块添加/驱逐；路由器索引 |
| 前缀哈希 | "缓存键" | 前 N 个 token 的哈希作为路由器查询 |
| GORGO | "跨区域路由研究" | arXiv 2602.11688；网络延迟作为显式项 |
| 跨区域推理 | "Bedrock CRI" | AWS 产品；可用性故障转移，非 TTFT 感知 |
| DR 清单 | "备份列表" | 恢复所需每个文件——不只是权重 |
| 数据驻留 | "GDPR 边界" | 法律约束哪个区域可以看到用户数据 |
| RTT | "往返时间" | 网络延迟；美欧 75 ms，美亚太 220 ms |
| LLM 感知 LB | "缓存命中 LB" | 缓存感知路由器作为一个产品类别 |

## 扩展阅读

- [BentoML — 多云和跨区域推理](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) — 带网络延迟项的跨区域 KV 缓存复用。
- [TianPan — 多区域 LLM 服务缓存本地性](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock 跨区域推理](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — 可用性故障转移文档。
- [vLLM 生产栈路由器](https://github.com/vllm-project/production-stack) — 缓存感知路由器源码。
