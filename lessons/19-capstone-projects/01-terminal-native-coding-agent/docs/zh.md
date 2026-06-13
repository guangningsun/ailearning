# Capstone 01 — 终端原生编码智能体

> 到 2026 年，编码智能体的形态已经定型。一个 TUI  Harness、一个带状态的计划、一个沙箱化的工具层、一个"计划—执行—观察—恢复"的循环。Claude Code、Cursor 3 和 OpenCode 从远处看都是一个样子。本 capstone 让你从头到尾构建一个——CLI 输入，PR 输出——并在 SWE-bench Pro 上用它与 mini-swe-agent 和 Live-SWE-agent 同台竞技。你会明白难点不在模型调用，而在工具循环、沙箱，以及 50 轮运行时的成本天花板。

**类型：** Capstone
**语言：** TypeScript / Bun（harness），Python（评估脚本）
**前置条件：** 阶段 11（LLM 工程）、阶段 13（工具与协议）、阶段 14（智能体）、阶段 15（自主系统）、阶段 17（基础设施）
**涉及阶段：** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**时间：** 35 小时

## 问题

编码智能体在 2026 年成为主要的 AI 应用类别。Claude Code（Anthropic）、Cursor 3 的 Composer 2 和 Agent Tabs（Cursor）、Amp（Sourcegraph）、OpenCode（112k 星）、Factory Droids 和 Google Jules 都基于同样的架构发货：一套终端 harness、一个授权的工具层、一个沙箱，以及围绕前沿模型构建的"计划—执行—观察"循环。前沿很窄——Live-SWE-agent 在 SWE-bench Verified 上达到了 79.2%——但工程技艺很广。大多数失败模式不是模型犯错，而是工具循环不稳定、上下文中毒、令牌成本失控，以及破坏性的文件系统操作。

你无法从外部理解这些智能体。你必须自己构建一个，看着它在第 47 轮因为 ripgrep 返回 8MB 匹配结果而崩溃，然后重建截断层。这正是本 capstone 的意义。

## 概念

Harness 有四个层面。**计划**维护一个 TodoWrite 风格的状态对象，模型每轮都会重写它。**执行**分发工具调用（read、edit、run、search、git）。**观察**捕获 stdout / stderr / 退出码，截断后把摘要喂回去。**恢复**处理工具错误，不让它撑爆上下文窗口或无限循环。2026 年的形态还多了一样东西：**钩子**。`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop` 和 `PreCompact`——这些是可配置扩展点，运维者在此注入策略遥测和安全护栏。

沙箱是 E2B 或 Daytona。每个任务在一个全新的 devcontainer 中运行，git worktree 挂载为读写。Harness 绝不触碰主机文件系统。工作树在成功或失败时都会被清理。成本控制在三个层面强制执行：每轮令牌上限、会话美元预算，以及硬性轮次上限（通常为 50）。可观测层是带有 GenAI 语义约定的 OpenTelemetry span，发送到自托管的 Langfuse。

## 架构

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## 技术栈

- Harness 运行时： Bun 1.2 + Ink 5（React-in-terminal）
- 模型访问： OpenRouter 统一 API，支持 Claude Sonnet 4.7、GPT-5.4-Codex、Gemini 3 Pro、Opus 4.5（处理最难的任务）
- 工具传输： Model Context Protocol StreamableHTTP（MCP 2026 修订版）
- 沙箱： E2B sandboxes（JS SDK）或 Daytona devcontainers
- 代码搜索： ripgrep 子进程，17 种语言的 tree-sitter 解析器（预编译）
- 隔离： 每个任务 `git worktree add`，成功/失败时清理
- 评估 harness： SWE-bench Pro（已验证子集）+ Terminal-Bench 2.0 + 你自己的 30 题 holdout
- 可观测性： OpenTelemetry SDK，`gen_ai.*` semconv → 自托管 Langfuse
- PR 推送： GitHub App，精细化 token，作用域限制在目标仓库

## 构建步骤

1. **TUI 和命令循环。** 用 Ink 脚手架一个 Bun 项目。接受 `agent run <repo> "<task>"`。打印分栏视图：计划窗格（顶部）、工具调用流（中部）、令牌预算（底部）。Ctrl-C 取消时在退出前触发 `SessionEnd` 钩子。

2. **计划状态。** 定义一个类型化的 TodoWrite schema（pending / in_progress / done 项目，带 notes）。模型每轮通过工具调用重写整个状态——不要让它增量变更。将计划持久化到 `.agent/state.json`，以便崩溃后恢复。

3. **工具层。** 定义六个工具：`read_file`、`edit_file`（带 diff 预览）、`ripgrep`、`tree_sitter_symbols`、`run_shell`（带超时）、`git`（status / diff / commit / push）。通过 MCP StreamableHTTP 暴露，使 harness 与传输层解耦。每个工具返回截断后的输出（每次调用上限 4k 令牌）。

4. **沙箱封装。** 每个任务启动一个 E2B 沙箱。`git worktree add -b agent/$TASK_ID` 一个新分支。所有工具调用都在沙箱内执行。主机文件系统不可达。

5. **钩子。** 实现全部八种 2026 钩子类型。至少接入四个用户编写的钩子：(a) `PreToolUse` 破坏性命令守卫，阻止 worktree 之外的 `rm -rf`；(b) `PostToolUse` 令牌统计；(c) `SessionStart` 预算初始化；(d) `Stop` 写入最终追踪包。

6. **评估循环。** 克隆 SWE-bench Pro Python 的 30 题子集。用你的 harness 运行每个题目。在 pass@1、每任务轮次和每任务 $ 上与 mini-swe-agent（最小基线）比较。将结果写入 `eval/results.jsonl`。

7. **成本控制。** 硬性截断：50 轮、200k 上下文、每任务 $5。`PreCompact` 钩子在 150k 标记处将早期轮次汇总为一个 prior-state 块，释放空间给新的观察而不丢失计划。

8. **PR 推送。** 成功后，最后一步是 `git push` + 一个 GitHub API 调用，打开一个 PR，正文中包含计划和 diff 摘要。

## 使用方法

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## 交付

可交付技能位于 `outputs/skill-terminal-coding-agent.md`。给定一个仓库路径和任务描述，它在沙箱中运行完整的计划—执行—观察循环，返回一个 PR URL 和一个追踪包。本 capstone 的评分标准：

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 vs 基线 | 你的 harness vs mini-swe-agent 在 30 道匹配的 Python 题目上 |
| 20 | 架构清晰度 | Plan/act/observe 分离、钩子层、工具 schema——对照 Live-SWE-agent 布局审查 |
| 20 | 安全性 | 沙箱逃逸测试、权限提示、破坏性命令守卫通过红队测试 |
| 20 | 可观测性 | 追踪完整性（100% 工具调用被 span）、每轮令牌统计 |
| 15 | 开发者体验 | 冷启动 < 2s、崩溃恢复恢复计划、Ctrl-C 中途取消工具干净利落 |
| **100** | | |

## 练习

1. 将后端模型从 Claude Sonnet 4.7 换成在 vLLM 上服务的 Qwen3-Coder-30B。比较 pass@1 和每任务 $。报告开放模型在哪些地方表现不佳。

2. 添加一个 `reviewer` 子智能体，在 PR 推送前读取 diff 并可以请求修订循环。测量误报审查是否将 SWE-bench pass 率拉低到单智能体基线以下（提示：通常会）。

3. 压力测试沙箱：写一个尝试 `curl` 外部 URL 的任务和一个尝试在 worktree 之外写入的任务。确认两者都被 PreToolUse 钩子阻止。记录这些尝试。

4. 用更小的模型（Haiku 4.5）实现 `PreCompact` 摘要。测量在 3 倍压缩下计划保真度损失多少。

5. 将 MCP StreamableHTTP 传输换成 stdio。评测冷启动和每次调用的延迟。为本地专用场景选出胜者。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Harness | "智能体循环" | 围绕模型、负责分发工具、维护计划状态并执行预算的代码 |
| Hook | "智能体事件监听器" | 由 harness 在八个生命周期事件之一运行的用户编写脚本 |
| Worktree | "Git 沙箱" | 在独立路径上的 linked git checkout；可丢弃而不触碰主克隆 |
| TodoWrite | "计划状态" | 类型化的 pending/in-progress/done 项目列表，模型每轮重写 |
| StreamableHTTP | "MCP 传输" | 2026 MCP 修订版：持久的 HTTP 连接，双向流；取代 SSE |
| Token ceiling | "上下文预算" | 每轮或每会话的输入+输出令牌上限；触发压缩或终止 |
| pass@1 | "单次尝试通过率" | SWE-bench 题目在首次运行无重试或测试集窥视的情况下被解决的比例 |

## 延伸阅读

- [Claude Code 文档](https://docs.anthropic.com/en/docs/claude-code) — Anthropic 的参考 harness
- [Cursor 3 更新日志](https://cursor.com/changelog) — Agent Tabs 和 Composer 2 产品说明
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — SWE-bench harness 对比的最小基线
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — 用 Opus 4.5 达到 79.2% SWE-bench Verified
- [OpenCode](https://opencode.ai) — 开放 harness，112k 星
- [SWE-bench Pro 排行榜](https://www.swebench.com) — 本 capstone 瞄准的评估
- [Model Context Protocol 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、能力元数据
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 工具调用和令牌使用的 span schema
