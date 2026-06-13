# 带 LMCache KV 卸载的 vLLM 生产栈

> vLLM 的生产栈是参考 Kubernetes 部署——路由器、引擎、可观测性连接在一起。LMCache 是 KV 卸载层，将 KV 缓存从 GPU 内存中提取出来并在查询和引擎间复用（CPU DRAM，然后是磁盘/Ceph）。vLLM 0.11.0 KV Offloading Connector（2026 年 1 月）通过 Connector API（v0.9.0+）使其异步化和可插拔。卸载延迟不面向用户。即使没有共享前缀，LMCache 也有价值——当 GPU 的 KV 槽用尽时，被抢占的请求可以从 CPU 恢复，而不是重新计算 prefill。16x H100（80GB HBM）跨 4 个 a3-highgpu-4g 的已发布基准测试：当 KV 缓存超过 HBM 时，原生 CPU 卸载和 LMCache 都显著提升吞吐量；在低 KV 占用情况下，所有配置与基线相当，只有少量开销。

**类型：** 学习型
**语言：** Python（标准库 + 玩具 KV 溢出模拟器）
**前置条件：** 第 17 阶段 · 04（vLLM 服务内部原理）、第 17 阶段 · 06（SGLang/RadixAttention）
**时间：** 约 60 分钟

## 学习目标

- 画出 vLLM 生产栈的层次：路由器、引擎、KV 卸载、可观测性。
- 解释 KV Offloading Connector API（v0.9.0+）以及 0.11.0 异步路径如何隐藏卸载延迟。
- 量化 LMCache CPU-DRAM 何时有帮助（KV > HBM）vs 增加开销（KV 小到可以放入 HBM）。
- 根据部署约束在原生 vLLM CPU 卸载和 LMCache 连接器之间选择。

## 问题

你的 vLLM 服务显示 GPU 的 HBM 一直 100%，并发量上升时就会出现抢占事件。请求被驱逐、重新排队，你在一分钟内对同一个 2K token 提示重新 prefill 四次。GPU 计算力花在了冗余的 prefill 上；goodput 远低于原始吞吐量。

加 GPU 要线性花钱。加 HBM 不可能。但 CPU DRAM 便宜——一个插槽有 512 GB+ 之多，延迟比 HBM 差几个数量级，但对于"暂时温热"的 KV 缓存来说完全够用。

LMCache 将 KV 缓存提取到 CPU DRAM，这样被抢占的请求可以快速恢复，跨引擎的重复前缀共享缓存而不必每个引擎重新 prefill。

## 概念

### vLLM 生产栈

`github.com/vllm-project/production-stack` 是参考 Kubernetes 部署：

- **路由器**——缓存感知型（第 17 阶段 · 11）。消费 KV 事件。
- **引擎**——vLLM worker。每个 GPU 或每个 TP/PP 组一个。
- **KV 缓存卸载**——LMCache 部署或原生连接器。
- **可观测性**——Prometheus 抓取、Grafana 仪表板、OTel 追踪。
- **控制平面**——服务发现、配置、滚动更新。

以 Helm chart + operator 形式交付。

### KV Offloading Connector API（v0.9.0+）

vLLM 0.9.0 引入了可插拔 KV 缓存后端的 Connector API。你的引擎将块卸载到连接器；连接器存储它们（RAM、磁盘、对象存储、LMCache）。请求需要一个块时，连接器加载回来。

vLLM 0.11.0（2026 年 1 月）新增异步卸载路径——卸载可以在后台发生，这样引擎在常见情况下不会被它阻塞。端到端延迟和吞吐量仍取决于工作负载形状、KV 缓存命中率和系统压力；vLLM 自己的说明指出，自定义内核卸载在低命中率下可能降低吞吐量，且异步调度与投机解码有已知的交互问题。

### 原生 CPU 卸载 vs LMCache

**原生 vLLM CPU 卸载**：引擎本地的。将 KV 块存储在主机 RAM 中。实现快速，零网络跳。不跨引擎。

**LMCache 连接器**：集群规模的。将块存储在共享 LMCache 服务器（CPU DRAM + Ceph/S3 层）中。块可被任何引擎访问。有 16x H100 基准测试发布。

单个引擎 HBM 压力时选原生。多个引擎共享前缀（RAG 带通用系统提示、多租户共享模板）时选 LMCache。

### 基准测试行为

16x H100（80 GB HBM）跨 4 个 a3-highgpu-4g 测试：

- 低 KV 占用（短提示、低并发）：所有配置与基线相当，LMCache 增加约 3-5% 开销。
- 中等占用：LMCache 开始在跨引擎前缀复用上发挥作用。
- KV 超过 HBM：原生 CPU 卸载和 LMCache 都显著提升吞吐量；LMCache 增益更大，因为跨引擎共享。

### LMCache 起决定性作用的场景

- 系统提示跨租户共享的多租户服务。
- 文档块在查询间重复的 RAG。
- 同一基础模型上的微调变体（LoRA），基础模型 KV 复用减少冗余工作。
- 抢占密集型工作负载：从 CPU 恢复比重新 prefill 便宜。

### 什么时候**不要**启用

- HBM 压力小——你要承担开销而没有收益。
- 短上下文（<1K token）——传输时间 > 重新 prefill。
- 单租户单提示工作负载——没有可捕获的复用。

### 与分解式服务的集成

第 17 阶段 · 17 分解式服务 + LMCache 叠加：KV 从 prefill 池传输到 decode 池，如果未使用则落入 LMCache；后续查询从 LMCache 拉取。第 17 阶段 · 11 缓存感知路由器可以路由到其本地或 LMCache 共享缓存匹配的引擎。

### 你应该记住的数字

- vLLM 0.9.0：Connector API 发布。
- vLLM 0.11.0（2026 年 1 月）：异步卸载路径；端到端延迟影响取决于工作负载、KV 命中率和系统压力（不是绝对保证）。
- 16x H100 基准测试：KV 占用超过 HBM 时 LMCache 有帮助。
- HBM 压力小：3-5% 开销而无收益。

## 使用

`code/main.py` 模拟有和没有 LMCache 的抢占密集型工作负载。报告避免的重新 prefill 数、吞吐量增益和盈亏平衡 HBM 利用率。

## 交付

本课产出 `outputs/skill-vllm-stack-decider.md`。给定工作负载形状和 vLLM 部署，决定用原生 vs LMCache vs 都不用。

## 练习

1. 运行 `code/main.py`。在什么 HBM 利用率下 LMCache 开始划算？
2. 一个租户在 200 次查询/小时中共享 6K token 的系统提示。计算每个租户的预期 LMCache 节省。
3. LMCache 服务器是单点故障。设计 HA 策略（副本、fallback 到原生）。
4. LMCache 存储到 Ceph 旋转磁盘。对于 70B FP8 上 4K token KV（500 MB），读取时间 vs 重新 prefill 时间是多少？
5. 论证 vLLM 0.11.0 异步路径是否"免费"——开销藏在哪里？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| 生产栈 | "参考部署" | vLLM 的 Kubernetes Helm chart + operator |
| Connector API | "KV 后端接口" | vLLM 0.9.0+ 可插拔 KV 存储接口 |
| 原生 CPU 卸载 | "引擎本地溢出" | 将 KV 存储在同一引擎的主机 RAM 中 |
| LMCache | "集群 KV 缓存" | 跨引擎 KV 缓存服务器，位于 CPU DRAM + 磁盘上 |
| 0.11.0 异步 | "非阻塞卸载" | 卸载隐藏在引擎流后面 |
| 抢占 | "驱逐以腾出空间" | HBM 满时 KV 缓存 shuffle |
| 前缀复用 | "相同的系统提示" | 多个查询共享开头；缓存命中 |
| Ceph 层 | "磁盘层" | 缓存层级中位于 DRAM 以下的持久存储 |

## 扩展阅读

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator。
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — Connector 实现。
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases) — 异步路径细节。