# Agno 与 Mastra：生产级运行时

> Agno（Python）和 Mastra（TypeScript）是 2026 年生产级运行时的标配组合。Agno 追求微秒级智能体初始化和无状态 FastAPI 后端。Mastra 在 Vercel AI SDK 基础上打包了智能体、工具、工作流、统一模型路由和复合存储。

**类型：** 学习型
**语言：** Python、TypeScript
**前置条件：** 阶段 14 · 01（智能体循环）、阶段 14 · 13（LangGraph）
**时间：** 约 45 分钟

## 学习目标

- 了解 Agno 的性能目标及其适用场景。
- 说出 Mastra 的三大原语——智能体（Agents）、工具（Tools）、工作流（Workflows）——以及支持的服务器适配器。
- 解释为什么无状态会话级 FastAPI 后端是 Agno 推荐的生产路径。
- 根据技术栈选择 Agno 或 Mastra（Python 优先 vs TypeScript 优先）。

## 问题

LangGraph、AutoGen、CrewAI 都偏重框架化。那些只想"要一个智能体循环，跑得快，嵌入我的运行时"的团队，会转向 Agno（Python）或 Mastra（TypeScript）。两者都用部分框架所有的高层原语换取了原始速度和与周围技术栈更紧密的契合。

## 概念

### Agno

- Python 运行时，前身为 Phi-data。
- "无图、无链、无复杂模式——只有纯 Python。"
- 官方性能目标：智能体初始化约 2μs，每个智能体内存约 3.75 KiB，支持约 23 个模型提供商。
- 生产路径：无状态会话级 FastAPI 后端。每个请求启动一个全新的智能体；会话状态保存在数据库中。
- 原生多模态支持（文本、图像、音频、视频、文件）和智能体 RAG。

当你有每秒数千个短生命周期智能体时（聊天聚合、评估流水线），这些速度目标很重要。当单个智能体要跑 10 分钟时，意义就小得多。

### Mastra

- TypeScript，运行在 Vercel AI SDK 之上。
- 三大原语：**智能体（Agents）**、**工具（Tools，Zod 类型化）**、**工作流（Workflows）**。
- 统一模型路由器——横跨 94 个提供商、3300+ 模型（2026 年 3 月）。
- 复合存储：记忆、工作流、可观测性分别对接到不同后端；规模化可观测性推荐使用 ClickHouse。
- Apache 2.0 协议，`ee/` 目录采用源码可用（source-available）企业许可证。
- 支持 Express、Hono、Fastify、Koa 的服务器适配器；与 Next.js 和 Astro 一级集成。
- 附带 Mastra Studio（localhost:4111）用于调试。
- GitHub 22k+ 星，1.0 版本每周 npm 下载 30 万+（2026 年 1 月）。

### 定位

两者都不是要成为 LangGraph。它们的竞争维度是：

- **语言契合度。** Python 优先团队选 Agno；TypeScript 优先团队选 Mastra。
- **运行时 ergonomics。** Agno = 近乎零开销；Mastra = 与 Vercel 生态深度集成。
- **可观测性。** 两者都能对接 Langfuse/Phoenix/Opik（第 24 课），但 Mastra Studio 是一方的。

### 何时选哪个

- **Agno** —— Python 后端、需要高速、短生命周期智能体多、FastAPI 技术栈。
- **Mastra** —— TypeScript 后端、Next.js / Vercel 部署、统一多提供商模型路由、Zod 类型化工具。
- **LangGraph**（第 13 课）—— 当持久状态和显式图推理比原始速度更重要时。
- **OpenAI / Claude Agent SDK** —— 当你想要提供商的产品化形态时（第 16–17 课）。

### 该模式会出错的地方

- **为性能而性能。** 工作负载是每个请求一个慢速智能体调用，却因为"2μs"听起来不错就选了 Agno。开销不是瓶颈。
- **生态锁定。** Mastra 的 Vercel 风格集成在 Vercel 上是加分项，在别处是减分项。
- **企业许可证混淆。** Mastra 的 `ee/` 目录是源码可用，不是 Apache 2.0。如果你要 fork，请先阅读许可证。

## 构建它

本课主要是对比性质的——没有哪个单一代码工件能同时公正地展示两个框架。参见 `code/main.py`：一个并排的玩具实现：用最简方式"运行一个智能体、流式输出结果、持久化会话"，分别按 Agno 风格和 Mastra 风格实现两次。

运行：

```
python3 code/main.py
```

两条结构不同但功能等价的执行轨迹。

## 使用它

- **Agno** —— 需要速度且形态接近 FastAPI 的 Python 后端。
- **Mastra** —— 需要多提供商和工作流原语的 TypeScript 后端。
- 两者都附带一方的可观测性钩子。都能对接 Langfuse。

## 交付它

`outputs/skill-runtime-picker.md` 根据技术栈、延迟预算和运维形态选择 Agno、Mastra、LangGraph 或提供商 SDK。

## 练习

1. 阅读 Agno 文档。将 stdlib ReAct 循环（第 01 课）移植到 Agno。哪些东西消失了？哪些保留了下来？
2. 阅读 Mastra 文档。将同一个循环移植到 Mastra。工具类型化发生了什么变化（Zod vs 无）？
3. 基准测试：在你的技术栈上测量智能体初始化延迟。Agno 的 2μs 对你的工作负载重要吗？
4. 设计一次迁移：如果你的 Python 技术栈一直在跑 CrewAI，迁移到 Agna 会破坏什么？
5. 阅读 Mastra 的 `ee/` 许可证条款。哪些限制会影响一个开源 fork？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| Agno | "快速 Python 智能体" | 无状态会话级智能体运行时 |
| Mastra | "Vercel AI SDK 上的 TypeScript 智能体" | 智能体 + 工具 + 工作流 + 模型路由器 |
| 统一模型路由器（Unified Model Router） | "多提供商访问" | 单个客户端访问 94 个提供商下的 3300+ 模型 |
| 复合存储（Composite storage） | "多后端" | 记忆/工作流/可观测性各自对接不同存储 |
| Mastra Studio | "本地调试器" | localhost:4111 UI，用于内省智能体 |
| 源码可用（Source-available） | "不是开源" | 许可证允许阅读源码但限制商业使用 |

## 延伸阅读

- [Agno Agent Framework 文档](https://www.agno.com/agent-framework) —— 性能目标、FastAPI 集成
- [Mastra 文档](https://mastra.ai/docs) —— 原语、服务器适配器、模型路由器
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview) —— 有状态图的替代方案
- [Comet Opik](https://www.comet.com/site/products/opik/) —— Mastra 集成中引用的可观测性对比