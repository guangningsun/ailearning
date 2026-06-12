# Claude Agent SDK：子智能体与会话存储

> Claude Agent SDK 是 Claude Code 工具的形式化库。内建工具、用于上下文隔离的子智能体、钩子、W3C 跟踪传播、会话存储 parity。Claude Managed Agents 是用于长时间运行异步工作的托管替代方案。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置条件：** 阶段 14 · 01（智能体循环）、阶段 14 · 10（技能库）
**时间：** 约 75 分钟

## 学习目标

- 解释 Anthropic Client SDK（原始 API）与 Claude Agent SDK（工具形态）之间的区别。
- 描述子智能体——并行化和上下文隔离——以及何时应该使用它们。
- 说出 Python SDK 的会话存储表面（`append`、`load`、`list_sessions`、`delete`、`list_subkeys`）和 `--session-mirror` 的作用。
- 实现一个带有内建工具、子智能体生成（隔离上下文）、生命周期钩子和会话存储的标准库工具。

## 问题

原始 LLM API 给你一次往返。生产智能体需要工具执行、MCP 服务器、生命周期钩子、子智能体生成、会话持久化、跟踪传播。 Claude Agent SDK 将这种形态作为库来提供——Claude Code 使用的相同工具，为自定义智能体暴露。

## 概念

### Client SDK vs Agent SDK

- **Client SDK (`anthropic`)。** 原始 Messages API。你拥有循环、工具、状态。
- **Agent SDK (`claude-agent-sdk`)。** 内建工具执行、MCP 连接、钩子、子智能体生成、会话存储。 Claude Code 循环作为库。

### 内建工具

SDK 开箱即用提供 10+ 种工具：文件读写、shell、grep、glob、网络获取等。自定义工具通过标准工具模式接口注册。

### 子智能体

Anthropic 记录了两个目的：

1. **并行化。** 并发运行独立工作。"为这 20 个模块中的每一个找到测试文件"是 20 个并行子智能体任务。
2. **上下文隔离。** 子智能体使用自己的上下文窗口；只有结果返回给编排器。编排器的预算得以保留。

Python SDK 最新添加：`list_subagents()`、`get_subagent_messages()` 用于读取子智能体 transcripts。

### 会话存储

与 TypeScript 的协议 parity：

- `append(session_id, message)` — 添加一轮。
- `load(session_id)` — 恢复对话。
- `list_sessions()` — 枚举。
- `delete(session_id)` — 级联到子智能体会话。
- `list_subkeys(session_id)` — 列出子智能体键。

`--session-mirror`（CLI 标志）在流式传输时将 transcript 镜像到外部文件，用于调试。

### 钩子

你可以注册的生命周期钩子：

- `PreToolUse`、`PostToolUse` — 门控或审计工具调用。
- `SessionStart`、`SessionEnd` — 设置和拆卸。
- `UserPromptSubmit` — 在模型看到用户输入之前对其采取行动。
- `PreCompact` — 在上下文压缩之前运行。
- `Stop` — 在智能体退出时清理。
- `Notification` — 侧通道警报。

钩子是如何将跨领域行为添加到 pro-workflow（阶段 14 课程参考）和类似系统的方式。

### W3C 跟踪上下文

调用者上活动的 OTel span 通过 W3C 跟踪上下文头传播到 CLI 子进程。整个多进程跟踪在你的后端显示为一个跟踪。

### Claude Managed Agents

托管替代方案（beta 头 `managed-agents-2026-04-01`）。长时间运行的异步工作、内建提示缓存、内建压缩。用托管基础设施换取控制。

### 这个模式哪里出了问题

- **子智能体过度生成。** 为 100 个小任务生成 100 个子智能体。开销占主导地位。改为批处理。
- **钩子蔓延。** 每个团队添加钩子；启动时间膨胀。每季度审查钩子。
- **会话膨胀。** 会话积累；大小增长。使用 `list_sessions` + 过期策略。

## 构建它

`code/main.py` 以标准库形式实现 SDK 形态：

- `Tool`、`ToolRegistry` 带内建 `read_file`、`write_file`、`list_dir`。
- `Subagent` — 私有上下文、隔离运行、返回结果。
- `SessionStore` — append、load、list、delete、list_subkeys。
- `Hooks` — `pre_tool_use`、`post_tool_use`、`session_start`、`session_end`。
- 一个演示：主智能体并行生成 3 个子智能体（每个隔离），聚合结果，持久化会话。

运行它：

```
python3 code/main.py
```

跟踪显示子智能体上下文隔离（编排器上下文大小保持有界）、钩子执行和会话持久化。

## 使用它

- **Claude Agent SDK** 用于想要 Claude Code 工具形态的 Claude 优先产品。
- **Claude Managed Agents** 用于托管长时间运行的异步工作。
- **OpenAI Agents SDK**（第 16 课）用于 OpenAI 优先对应物。
- **LangGraph + 自定义工具** 如果你想要图形形状的状态机。

## 发货

`outputs/skill-claude-agent-scaffold.md` 搭建一个带有子智能体、钩子、会话存储、MCP 服务器附加和 W3C 跟踪传播的 Claude Agent SDK 应用。

## 练习

1. 添加一个子智能体生成器，将 20 个任务批处理为 5 个并行子智能体一组。测量编排器上下文大小 vs 每个任务一个。
2. 实现一个 `PreToolUse` 钩子，对 `write_file` 调用进行速率限制（每会话每分钟 5 次）。跟踪行为。
3. 将 `list_subkeys` 连接以渲染子智能体树。深度嵌套是什么样子的？
4. 将玩具移植到真正的 `claude-agent-sdk` Python 包。工具注册发生了什么变化？
5. 阅读 Claude Managed Agents 文档。什么时候你会从自托管切换到托管？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| Agent SDK | "Claude Code 作为库" | 工具形态：工具、MCP、钩子、子智能体、会话存储 |
| Subagent | "子智能体" | 独立上下文、自己的预算；结果向上冒泡 |
| Session store | "对话数据库" | 持久化、加载、列出、删除带有子智能体级联的轮次 |
| Hook | "生命周期回调" | 工具前后、会话、提示提交、压缩、停止 |
| W3C trace context | "跨进程跟踪" | 父 span 传播到 CLI 子进程 |
| Managed Agents | "托管工具" | Anthropic 托管的长时运行异步工作 |
| `--session-mirror` | "Transcript 镜像" | 在流式传输时将会话轮次写入外部文件 |
| MCP server | "工具表面" | 附加到智能体的外部工具/资源源 |

## 进一步阅读

- [Claude Agent SDK 概述](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Code 的库形式
- [Anthropic，使用 Claude Agent SDK 构建智能体](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — 生产模式
- [Claude Managed Agents 概述](https://platform.claude.com/docs/en/managed-agents/overview) — 托管替代方案
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 对应物