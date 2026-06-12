# 生产运行时：队列、事件、Cron

> 生产代理运行在六种运行时形态上：请求-响应、流式、持久执行、基于队列的后台、事件驱动、定时执行。先选形态再选框架。在每种形态上，可观测性都是承载负载的。

**类型：** 学习
**语言：** Python（标准库）
**前置条件：** 阶段 14 · 13（LangGraph）、阶段 14 · 22（语音）
**时间：** 约 60 分钟

## 学习目标

- 说出六种生产运行时形态，并将每种与框架/产品模式匹配。
- 解释持久执行（LangGraph）对长时域任务的重要性。
- 描述事件驱动运行时以及 Claude Managed Agents 何时适用。
- 解释可观测性作为承载负载的主张对多步代理的意义。

## 问题

生产代理以 Jupyter notebook 无法暴露的方式失败：第 37 步网络超时、用户中途挂断语音电话、cron 作业在机器重启时死亡、后台 worker 内存耗尽。运行时形态决定了哪些失败是可以幸存的。

## 概念

### 请求-响应

- 同步 HTTP。用户等待完成。
- 仅适用于短任务（<30 秒）。
- 技术栈：Agno（Python + FastAPI）、Mastra（TypeScript + Express/Hono/Fastify/Koa）。
- 可观测性：标准 HTTP 访问日志 + OTel 跨度。

### 流式

- SSE 或 WebSocket 用于渐进式输出。
- LiveKit 将此扩展到 WebRTC 用于语音/视频（第 22 课）。
- 技术栈：任何支持流式的框架 + 处理 SSE/WS 的前端。
- 可观测性：每块时间、首个令牌延迟、尾延迟。

### 持久执行

- 每步后检查点状态；失败时自动恢复。
- AutoGen v0.4 参与者模型将失败隔离到单个代理（第 14 课）。
- LangGraph 的核心差异化特性（第 13 课）。
- 当步数未知且恢复成本高时必不可少。

### 基于队列/后台

- 作业进入队列，worker 拾取，结果通过 webhook 或 pub/sub 返回。
- 对于长时域代理必不可少（每个任务数十到数百步，按 Anthropic 的计算机使用公告）。
- 技术栈：Celery（Python）、BullMQ（Node）、SQS + Lambda（AWS）、自定义。
- 可观测性：队列深度、每作业延迟分布、DLQ 大小。

### 事件驱动

- 代理订阅触发器：新邮件、PR 打开、cron 触发。
- Claude Managed Agents 开箱即用覆盖此场景（第 17 课）。
- CrewAI Flows（第 15 课）构建事件驱动的确定性工作流。
- 可观测性：触发源、事件到启动延迟、代理延迟。

### 定时执行

- 定时形的代理周期性运行。
- 结合持久执行，这样失败的夜间运行会在下一个 tick 恢复。
- 技术栈：Kubernetes CronJob + 持久框架；托管服务（Render cron、Vercel cron）。

### 2026 年部署模式

- **CrewAI Flows** 用于事件驱动的生产环境。
- **Agno** 无状态 FastAPI 用于 Python 微服务。
- **Mastra** 服务器适配器（Express、Hono、Fastify、Koa）用于嵌入。
- **Pipecat Cloud / LiveKit Cloud** 用于托管语音（第 22 课）。
- **Claude Managed Agents** 用于托管长时域异步场景。

### 可观测性是承载负载的

没有 OpenTelemetry GenAI 跨度（第 23 课）加上 Langfuse/Phoenix/Opik 后端（第 24 课），你无法调试在第 40 步失败的多步代理。这对生产环境不是可选项。这是"我们快速调试"和"我们用更多日志从头重放"之间的区别。

### 生产运行时失效的地方

- **形态选择错误。** 为 5 分钟任务选择请求-响应。用户挂断；worker 堆积；重试复合。
- **无 DLQ。** 队列 worker 无死信。失败的作业消失。
- **后台工作不透明。** 后台代理运行无跟踪导出。失败不可见，直到用户报告。
- **跳过持久状态。** 任何超过 30 秒且无法承受重启的运行都需要持久执行。

## 构建

`code/main.py` 是一个标准库多形态演示：

- 请求-响应端点（普通函数）。
- 流式处理器（生成器）。
- 带 DLQ 的基于队列的 worker。
- 事件触发注册表。
- Cron 形调度器。

运行：

```bash
python3 code/main.py
```

输出：每种形态处理相同任务的五个跟踪。相同的代理逻辑，不同的外壳。持久执行（第六种形态）故意在第 13 课配合 LangGraph 检查点讲解。

## 使用

- **请求-响应** 用于聊天风格 UX。
- **流式** 用于渐进式响应。
- **持久** 用于长时域任务。
- **队列** 用于批处理 / 异步 / 长时运行。
- **事件** 用于代理响应性。
- **定时** 用于内务处理（记忆整合、评估、成本报告）。

## 交付

`outputs/skill-runtime-shape.md` 为任务选择运行时形态并连接可观测性需求。

## 练习

1. 将你的第 1 课 ReAct 循环移植到你技术栈中的所有六种形态。哪种形态适合哪种产品表面？
2. 为基于队列的演示添加 DLQ。模拟 10% 作业失败；展示 DLQ 大小。
3. 编写一个 cron 触发的评估代理，每晚针对当天的前 20 条跟踪运行。
4. 实现带背压的流式：如果客户端慢，暂停代理。这如何与轮次预算互动？
5. 阅读 Claude Managed Agents 文档。什么时候会将自托管长时域代理迁移到托管服务？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| 请求-响应 | "同步" | 用户等待；仅限短任务 |
| 流式 | "SSE / WS" | 渐进式输出；更好的 UX；每块延迟可观测 |
| 持久执行 | "从失败恢复" | 检查点状态；从最后一步重启 |
| 基于队列 | "后台作业" | 生产者 / worker 池 / DLQ |
| 事件驱动 | "触发式" | 代理响应外部事件 |
| DLQ | "死信队列" | 失败作业的停车场 |
| Claude Managed Agents | "托管工具" | Anthropic 托管的长时域异步，带缓存 + 压缩 |

## 扩展阅读

- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) —— 持久执行详情
- [Claude Managed Agents 概述](https://platform.claude.com/docs/en/managed-agents/overview) —— 托管长时域异步
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) —— "每个任务数十到数百步"
- [AutoGen v0.4（微软研究院）](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) —— 参与者模型故障隔离