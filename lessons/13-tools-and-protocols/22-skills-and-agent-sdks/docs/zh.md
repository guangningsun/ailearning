# Skills 与 Agent SDK — Anthropic Skills、AGENTS.md、OpenAI Apps SDK

> MCP 定义"存在哪些工具"。Skills 定义"如何做任务"。2026 年的技术栈将两者分层。Anthropic 的 Agent Skills（2025 年 12 月发布的开放标准）以 SKILL.md 形式发布，采用渐进式披露。OpenAI 的 Apps SDK 构建在 MCP 之上并带有 Widget 元数据。AGENTS.md（目前已在 60,000+ 个仓库中使用）位于仓库根目录，作为项目级 Agent 上下文。本课命名每个层覆盖的内容，并构建一个可在 Agent 间迁移的最小化 SKILL.md + AGENTS.md 组合包。

**类型：** 学习型
**语言：** Python（标准库、SKILL.md 解析器和加载器）
**前置条件：** 阶段 13 · 07（MCP 服务器）
**时间：** 约 45 分钟

## 学习目标

- 区分三层：AGENTS.md（项目上下文）、SKILL.md（可复用know-how）、MCP（工具）。
- 编写带有 YAML frontmatter 和渐进式披露的 SKILL.md。
- 将技能以文件系统风格加载到 Agent 运行时中。
- 将技能与 MCP 服务器和 AGENTS.md 组合，使一个包在 Claude Code、Cursor 和 Codex 中都能工作。

## 问题

一位工程师将写发布说明的工作流提炼为一个多步提示词："读取最新合并的 PR。按领域分组。总结每个。按团队风格写一条变更日志。发布到 Slack 草稿。"他们把它放在团队的 Notion 文档里。

现在他们想从 Claude Code、Cursor 和 Codex CLI 中使用这个工作流。每个 Agent 加载指令的方式不同：Claude Code 斜杠命令、Cursor rules、Codex `.codex.md`。工程师将工作流复制三份，维护三份副本。

AGENTS.md 和 SKILL.md 一起解决这个问题：

- **AGENTS.md** 位于仓库根目录。每个兼容的 Agent 在会话开始时读取它。"这个项目怎么运作？有哪些约定？哪些命令用来运行测试？"
- **SKILL.md** 是一个可移植的包：YAML frontmatter（名称、描述）+ markdown 正文 + 可选资源。支持技能的 Agent 按名称按需加载。
- **MCP**（阶段 13 · 06-14）处理技能需要调用的工具。

三层，一个可移植产物。

## 概念

### AGENTS.md（agents.md）

2025 年末推出，截至 2026 年 4 月已被 60,000+ 个仓库采用。仓库根目录的一个文件。格式：

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Use Pydantic for models on the Python side.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

Agent 在会话开始时读取这些内容，并用它来调整自己对该项目的行为。2026 年每个编码 Agent 都支持 AGENTS.md：Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf、Zed。

### SKILL.md 格式

Anthropic 的 Agent Skills（2025 年 12 月作为开放标准发布）：

```markdown
---
name: release-notes-writer
description: Write a changelog entry for the latest merged PRs following this project's style.
---

# Release notes writer

When invoked, run these steps:

1. List PRs merged since the last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR in each group, write one line: `- <title> (#<num>)`.
4. Draft the release notes and stage them in CHANGELOG.md.

If the user says "ship", run `git tag vX.Y.Z` and `gh release create`.

## Notes

- Never include commits without a PR.
- Skip "chore" entries from the public changelog.
```

Frontmatter 声明技能的标识。正文是技能加载时展示给模型的提示词。

### 渐进式披露

技能可以引用子资源，Agent 仅在需要时才获取。例如：

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md 写"样式规则见 style-guide.md"。Agent 仅在技能活跃运行时才拉取 style-guide.md。这避免了用模型可能不需要的细节撑爆提示词。

### 文件系统发现

Agent 运行时扫描已知目录寻找 SKILL.md 文件：

- `~/.anthropic/skills/*/SKILL.md`
- 项目 `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

按文件夹名称和 frontmatter 的 `name` 加载。Claude Code、Anthropic Claude Agent SDK 和 SkillKit（跨 Agent）都遵循此模式。

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk`（TypeScript）和 `claude-agent-sdk`（Python）在会话开始时加载技能，在运行时内作为可调用的"Agent"暴露。Agent 循环在用户调用技能时分派到该技能。

### OpenAI Apps SDK

2025 年 10 月推出；直接构建在 MCP 之上。将 OpenAI 之前的 Connectors 和自定义 GPT Actions 统一在单一开发者表面下。Apps SDK 应用是：

- 一个 MCP 服务器（工具、资源、提示词）。
- 加上 ChatGPT UI 的 Widget 元数据。
- 加上可选的 MCP Apps `ui://` 资源，用于交互式界面。

同一协议，更丰富的用户体验。

### 通过 SkillKit 实现跨 Agent 可移植性

像 SkillKit 和类似的跨 Agent 分发层将单一 SKILL.md 翻译为 32+ 个 AI Agent 的原生格式（Claude Code、Cursor、Codex、Gemini CLI、OpenCode 等）。一个真相来源；众多消费者。

### 三层技术栈

| 层 | 文件 | 加载时机 | 用途 |
|-------|------|-------------|---------|
| AGENTS.md | 仓库根目录 | 会话开始 | 项目级约定 |
| SKILL.md | skills 目录 | 技能被调用 | 可复用工作流 |
| MCP 服务器 | 外部进程 | 需要工具时 | 可调用动作 |

三者组合：Agent 在会话开始时读取 AGENTS.md，用户调用一个技能，技能的指令包含 MCP 工具调用，Agent 通过 MCP 客户端分派。

## 使用

`code/main.py` 带有一个纯标准库的 SKILL.md 解析器和加载器。它在 `./skills/` 下发现技能，解析 YAML frontmatter 和 markdown 正文，产生按技能名称键控的字典。然后模拟一个 Agent 循环，按名称调用 `release-notes-writer` 技能。

需要关注的地方：

- 用最小化标准库解析器解析 YAML frontmatter（无 `pyyaml` 依赖）。
- 技能正文原样存储；Agent 在调用时将它前置到系统提示词。
- 通过 `read_subresource` 函数演示渐进式披露，该函数按需拉取引用的文件。

## 交付

本课产出 `outputs/skill-agent-bundle.md`。给定一个工作流，该技能生成组合的 SKILL.md + AGENTS.md + MCP 服务器蓝图包，可在 Agent 间迁移。

## 练习

1. 运行 `code/main.py`。在 `skills/` 下添加第二个技能，确认加载器能发现它。

2. 为本课程仓库写一个 AGENTS.md。包含测试命令、样式约定和阶段 13 思维模型。

3. 将团队内部文档中的多步工作流移植到 SKILL.md。在 Claude Code 中验证它能加载。

4. 手动将该技能翻译为 Cursor 和 Codex 的原生规则格式。计算格式间的差异——这就是 SkillKit 自动化的翻译工作量。

5. 阅读 Anthropic Agent Skills 博客文章。找出 Claude Agent SDK 中本课加载器未覆盖的一个功能。（提示：Agent 子调用。）

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| SKILL.md | "技能文件" | YAML frontmatter 加 markdown 正文，由 Agent 运行时加载 |
| AGENTS.md | "仓库根目录 Agent 上下文" | 项目级约定文件，在会话开始时读取 |
| 渐进式披露 | "惰性加载子资源" | 技能正文引用文件，仅在需要时拉取 |
| Frontmatter | "顶部的 YAML 块" | 元数据（名称、描述），用 `---` 分隔符包裹 |
| Claude Agent SDK | "Anthropic 的技能运行时" | `@anthropic-ai/claude-agent-sdk`，加载技能并路由 |
| OpenAI Apps SDK | "MCP + Widget 元数据" | 构建在 MCP 上加上 ChatGPT UI 钩子的 OpenAI 开发者表面 |
| 技能发现 | "文件系统扫描" | 遍历已知目录找 SKILL.md，按名称键控 |
| 跨 Agent 可移植性 | "一个技能，多个 Agent" | 通过 SkillKit 风格工具将一个 SKILL.md 翻译为 32+ 个 Agent |
| Agent Skill | "可移植的 know-how" | MCP 工具概念之外的可复用任务模板 |
| Apps SDK | "MCP 加 ChatGPT UI" | 统一在 MCP 上的 Connectors 和自定义 GPT |

## 进一步阅读

- [Anthropic — Agent Skills 公告](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 2025 年 12 月发布
- [Anthropic — Agent Skills 文档](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — SKILL.md 格式参考
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — 基于 MCP 的 ChatGPT 开发者平台
- [agents.md](https://agents.md/) — AGENTS.md 格式和采用列表
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) — 官方技能示例