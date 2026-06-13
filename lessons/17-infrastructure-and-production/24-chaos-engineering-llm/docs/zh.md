# LLM 生产混沌工程

> 到 2026 年，LLM 的混沌工程已是一个独立学科。在生产环境运行实验的前置条件：已定义 SLI/SLO、trace+metric+log 可观测性、自动回滚、运行手册、值班。在生产环境运行实验的前置条件：已定义 SLI/SLO、trace+metric+log 可观测性、自动回滚、运行手册、值班。架构有四个平面：控制（实验调度器）、目标（服务、基础设施、数据存储）、安全（护栏 + 中止 + 流量过滤）、可观测性（指标 + trace + 日志）、反馈（进入 SLO 调整）。护栏是强制的：燃烧率告警会在每日错误预算消耗超过预期 2 倍时暂停实验；抑制窗口 + trace-ID 关联去重告警噪音。节奏：每周小规模金丝雀 + SLO 回顾；每月游戏日 + 复盘；每季度跨团队韧性审计 + 依赖映射。LLM 特定实验：内存过载、网络故障、提供商中断、畸形 Prompt、KV 缓存驱逐风暴。工具：Harness Chaos Engineering（LLM 衍生建议、爆炸半径缩减、MCP 工具集成）；LitmusChaos（CNFC）；Chaos Mesh（CNCF Kubernetes 原生）。

**类型：** 学习
**语言：** Python（标准库 + toy 混沌实验运行器）
**前置条件：** 阶段 17 · 23（AI SRE）、阶段 17 · 13（可观测性）
**时间：** 约 60 分钟

## 学习目标

- 说出五个混沌工程前置条件（SLI/SLO、可观测性、回滚、运行手册、值班）并解释跳过任何一个会如何破坏实践。
- 绘制四个平面（控制、目标、安全、可观测性）及进入 SLO 的反馈环。
- 列举五个 LLM 特定实验（内存过载、网络故障、提供商中断、畸形 Prompt、KV 驱逐风暴）。
- 根据技术栈选择工具——Harness、LitmusChaos、Chaos Mesh。

## 问题

传统堆栈的混沌测试已经成熟。LLM 堆栈增加了新的故障模式。带有恶意字符的 4K Token Prompt 会使分词器停滞 12 秒。上游提供商 429；你的网关重试；你的服务因重试放大并发而 OOM。突发负载下的 KV 缓存驱逐风暴导致重新 prefill 级联使计算饱和。

这些都不会在单元测试中出现。混沌工程是在用户发现之前发现它们的方法。

## 概念

### 前置条件

不要在没有以下条件的情况下在生产环境运行混沌：

1. **SLI/SLO**——已定义的服务级指标和目标。
2. **可观测性**——trace、指标、日志，连接到仪表盘。
3. **自动回滚**——阶段 17 · 20 策略标志回滚。
4. **运行手册**——结构化，阶段 17 · 23。
5. **值班**——有人响应。

缺少任何一个都会让混沌变成真实事件。

### 四个平面 + 反馈

**控制平面**——实验调度器（Litmus workflow、Chaos Mesh schedule、Harness UI）。

**目标平面**——服务、Pod、节点、负载均衡器、数据存储。

**安全平面**——kill switch、抑制窗口、爆炸半径限制、错误预算门禁。

**可观测性平面**——正常指标 + trace-ID 关联以区分混沌引起的故障和自然故障。

**反馈环**——发现反馈到 SLO 调整、运行手册更新、代码修复。

### 护栏是强制的

- **燃烧率告警**：如果每日错误预算消耗超过预期的 2 倍，暂停实验。
- **抑制窗口**：在实验期间对爆炸半径内的非实验告警静音。
- **Trace-ID 关联**：所有实验引起的错误都带有标签，以便值班人员去重。

### 五个 LLM 特定实验

1. **内存过载**——通过发送长上下文请求 + 高并发，强制 KV 缓存抢占风暴。观察：服务是优雅降级还是崩溃？

2. **网络故障**——切断推理网关和提供商之间的连接。观察：故障转移是否在 SLA 内启动？（阶段 17 · 19）

3. **提供商中断模拟**——100% 429 来自 OpenAI。观察：路由是否故障转移到 Anthropic？（阶段 17 · 16、19）

4. **畸形 Prompt**——注入会使分词器停滞的有效载荷（例如，深层嵌套 unicode、超大 UTF-8 码点）。观察：单个请求是否会锁住一个 worker？

5. **KV 驱逐风暴**——通过饱和 vLLM 块预算强制驱逐。观察：LMCache 是恢复还是服务降级？

### 节奏

- **每周**——在预发环境中进行小规模金丝雀实验，可能 5% 生产流量。
- **每月**——针对特定场景的预定游戏日；跨团队参与；复盘。
- **每季度**——跨团队韧性审计；依赖图更新。

### 工具

- **Harness Chaos Engineering**——商业版；AI 衍生的实验建议；爆炸半径缩减；MCP 工具集成。
- **LitmusChaos**——CNCF 毕业；基于 Kubernetes workflow。
- **Chaos Mesh**——CNCF 沙盒；Kubernetes 原生 CRD 风格。
- **Gremlin**——商业版；广泛支持。
- **AWS FIS** / **Azure Chaos Studio**——托管云产品。

### 小处着手

第一个实验：在稳定流量下杀死一个 decode 副本。观察重路由和恢复。如果可行且安全，升级到网络混沌。

第一个 LLM 特定实验：注入 5 分钟的一个提供商 429。观察故障转移。大多数团队发现他们的故障转移并未完全测试过。

### 需要记住的数字

- 四个平面：控制、目标、安全、可观测性。
- 燃烧率暂停：预期每日预算消耗的 2 倍。
- 节奏：每周金丝雀、每月游戏日、每季度审计。
- 五个 LLM 实验：内存、网络、提供商、畸形 Prompt、KV 风暴。

## 使用它

`code/main.py` 模拟三个带有安全平面门禁的混沌实验。报告哪些实验会触发燃烧率中止。

## 交付它

本课产出 `outputs/skill-chaos-plan.md`。给定技术栈和成熟度，选择前三个实验和工具。

## 练习

1. 运行 `code/main.py`。哪个实验触发了燃烧率门禁，为什么？
2. 为基于 vLLM 的 RAG 服务设计前五个混沌实验。包括成功标准。
3. 你的燃烧率告警暂停了一个实验。你如何确定根因——混沌还是自然的？
4. 论证混沌应该运行在生产环境还是仅在预发。什么时候生产是正确答案？
5. 说出三个通用网络混沌无法复现的 LLM 特定故障模式。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| SLI / SLO | "服务目标" | 指标 + 目标；必需的前置条件 |
| 爆炸半径 | "范围" | 实验影响的服务/用户集合 |
| 燃烧率告警 | "预算门禁" | 当错误预算消耗率 > 预期 2 倍时触发 |
| 游戏日 | "每月演练" | 预定的跨团队混沌演练 |
| LitmusChaos | "CNCF workflow" | 毕业的 CNCF Kubernetes 混沌工具 |
| Chaos Mesh | "CNCF CRD" | CNCF 沙盒 Kubernetes 原生混沌 |
| Harness CE | "商业 AI 辅助" | 带 AI 建议的 Harness 混沌 |
| 畸形 Prompt | "分词器炸弹" | 使分词停滞的输入 |
| KV 驱逐风暴 | "抢占级联" | 大规模驱逐触发重新 prefill |

## 延伸阅读

- [DevSecOps School — Chaos Engineering 2026 Guide](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — Observability for LLMs (book)](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)