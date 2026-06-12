# 编排模式：Supervisor、Swarm、分层

> 2026 年框架中反复出现四种编排模式：supervisor-worker、swarm / 点对点、分层、辩论。Anthropic 的指导是："关键在于为你的需求构建合适的系统。"从简单开始；只有当单个代理加五种工作流模式不够用时才添加拓扑。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置条件：** 阶段 14 · 12（工作流模式）、阶段 14 · 25（多代理辩论）
**时间：** 约 60 分钟

## 学习目标

- 说出四种反复出现的编排模式及其各自适用场景。
- 描述 2026 年 LangChain 的建议：基于工具调用的监督与 supervisor 库。
- 解释 Anthropic 的"构建合适的系统"规则及其如何控制拓扑选择。
- 针对通用脚本化大语言模型用标准库实现所有四种模式。

## 问题

团队在不需要多代理时就伸手要用"多代理"。四种模式在各个框架中反复出现；一旦能叫出名字，就能选对——或者完全跳过拓扑。

## 概念

### Supervisor-worker

- 一个中央路由大语言模型分派给专业代理。
- 决策：循环回自己、移交给专业代理、终止。
- 专业代理之间不相互通信；所有路由都通过 supervisor。

框架：LangGraph `create_supervisor`、Anthropic orchestrator-workers、CrewAI 分层流程。

**2026 年 LangChain 建议：** 通过直接工具调用而非 `create_supervisor` 来做监督。提供更精细的上下文工程控制——你决定每个专业代理能看到什么。

### Swarm / 点对点

- 代理通过共享工具层面直接交接。
- 无中央路由。
- 比 supervisor 延迟更低（跳数更少）。
- 更难推理（无单一控制点）。

框架：LangGraph swarm 拓扑、OpenAI Agents SDK 交接（当所有代理都可以相互交接时）。

### 分层

- 管理子-supervisor 的 supervisor 管理 worker。
- 在 LangGraph 中实现为嵌套子图；在 CrewAI 中为嵌套 crew。
- 在运营复杂性的代价下扩展到大型代理群体。

适用场景：当单个 supervisor 的上下文预算无法容纳所有专业代理的描述时。

### 辩论

- 并行提议者 + 迭代交叉批判（第 25 课）。
- 严格来说不算编排——更像是验证——但在框架中作为拓扑选择出现。

### CrewAI Crew 与 Flow

CrewAI 正式定义了两种部署模式：

- **Flow** 用于确定性事件驱动自动化（生产推荐起点）。
- **Crew** 用于自主基于角色的协作。

这与上面的四种模式正交，但映射到拓扑：Flow 通常是 supervisor 或分层；Crew 通常是带大语言模型路由器的 supervisor。

### Anthropic 的指导

"在大语言模型领域的成功不在于构建最复杂的系统，而在于为你的需求构建合适的系统。"

决策顺序：

1. 单代理 + 工作流模式（第 12 课）——从这里开始。
2. Supervisor-worker——当你有 2-4 个专业代理时。
3. Swarm——当延迟比推理清晰度更重要时。
4. 分层——只有在 supervisor 上下文预算失败时。
5. 辩论——当准确性比成本更重要时。

### 这种模式出错的地方

- **拓扑优先思维。** 在明确多代理能解决什么问题之前就"我们需要多代理"。
- **Swarm 中的弹跳交接。** A -> B -> A -> B。使用跳数计数器。
- **虚假分层。** 三层因为"企业级"；实际只有两个团队。压缩。

## 构建

`code/main.py` 针对脚本化大语言模型用标准库实现所有四种模式：

- `Supervisor`——中央路由器。
- `Swarm`——带直接交接的点对点。
- `Hierarchical`——supervisor 的 supervisor。
- `Debate`——并行提议者 + 批判。

每种模式处理相同的三个意图任务（退款 / bug / 销售）。跟踪形状不同。

运行：

```bash
python3 code/main.py
```

输出：每种模式的跟踪 + 操作计数。Supervisor 最清晰；swarm 最短；分层最深；辩论最贵。

## 使用

- **LangGraph** 用于 supervisor 和分层（嵌套子图）。
- **OpenAI Agents SDK** 用于作为工具的交接（supervisor 形态）。
- **CrewAI Flow** 用于生产确定性场景。
- **自定义** 用于辩论或当你需要精确控制时。

## 交付

`outputs/skill-orchestration-picker.md` 选择一种拓扑并实现它。

## 练习

1. 通过移除路由器将 supervisor-worker 转换为 swarm。什么坏了？什么改善了？
2. 为 swarm 添加跳数计数器：3 次交接后拒绝。它能捕获 A->B->A 弹跳吗？
3. 为 12 个专业代理领域构建两级分层系统。在不嵌套的情况下，上下文预算在哪里失败？
4. 在生产形状的工作负载上分析四种模式。在哪个指标上谁赢（延迟、成本、准确性、可调试性）？
5. 阅读 Anthropic 的"Building Effective Agents"帖子。将你的每个生产流程映射到四种之一。有哪个不能干净映射的吗？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| Supervisor-worker | "路由器 + 专业代理" | 中央大语言模型分派给专业代理；他们之间不相互通信 |
| Swarm | "点对点" | 通过共享工具直接交接；无中央路由器 |
| Hierarchical | "Supervisor 的 supervisor" | 用于大型群体的嵌套子图 |
| Debate | "提议者 + 批判" | 并行提议者、交叉批判（第 25 课） |
| 基于工具调用的监督 | "无库的 Supervisor" | 通过直接工具调用实现 supervisor 以控制上下文 |
| Crew | "自主团队" | CrewAI 的基于角色协作模式 |
| Flow | "确定性工作流" | CrewAI 的事件驱动生产模式 |

## 扩展阅读

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) —— 五种模式 + 代理 vs 工作流
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) —— supervisor、swarm、分层
- [CrewAI 文档](https://docs.crewai.com/en/introduction) —— Crew vs Flow
- [Du 等人，Society of Minds（arXiv:2305.14325）](https://arxiv.org/abs/2305.14325) —— 辩论模式