# Agent 指令作为可执行约束

> 用散文写的指令是愿望。用约束写的指令是测试。工作台将每条规则转化为 Agent 在运行时可以检查的东西，以及审核者在事后可以验证的东西。

**类型：** 构建
**语言：** Python（标准库）
**前置条件：** 阶段 14 · 32（最小工作台）
**时间：** 约 50 分钟

## 学习目标

- 将路由散文与操作规则分开。
- 将启动规则、禁止行为、完成定义、不确定处理和审批边界表达为机器可检查的约束。
- 实现一个规则检查器，对照规则集对运行进行评分。
- 使规则集对 diff 友好，以便审核可以看到什么发生了变化。

## 问题

一个典型的 `AGENTS.md` 读起来像入职文档。它告诉 Agent "要小心"、"要彻底测试"、"不确定时要问"。三天后，Agent 发布了一个没有测试的变更，写入了禁止的目录，从来没有问过——因为它从来不知道红线在哪里。

当指令可操作时它们是强大的，当它们是愿望时它们是弱的。解决方法是将规则写成像工作台可以解释、审核者可以评分的东西。

## 概念

规则放在 `docs/agent-rules.md` 中，与简短的根路由分开。每条规则有名称、类别和检查函数。

```mermaid
flowchart LR
  Router[AGENTS.md] --> Rules[docs/agent-rules.md]
  Rules --> Checker[rule_checker.py]
  Checker --> Report[rule_report.json]
  Report --> Reviewer[审核者]
```

### 覆盖大多数规则的五个类别

| 类别 | 规则回答的问题 | 示例 |
|----------|---------------------------|---------|
| 启动 | 工作开始前什么必须为真？ | "状态文件存在且是最新的" |
| 禁止 | 什么绝对不能发生？ | "不要编辑 `scripts/release.sh`" |
| 完成定义 | 什么证明任务完成了？ | "pytest 退出 0 且验收行通过" |
| 不确定 | Agent 不确定时做什么？ | "打开一个问题记录而不是猜测" |
| 审批 | 什么需要人工审批？ | "任何新依赖，任何生产环境写入" |

不属于这五类的规则通常需要分成两条。强制拆分。

### 规则是机器可读的

每条规则有 slug、类别、一行描述和一个 `check` 字段，指向 `rule_checker.py` 中的一个函数。添加规则就是添加检查；检查器随工作台一起增长。

### 规则对 diff 友好

规则以单个 markdown 文件中每个标题一条的方式存在。重命名在 diff 中可见。新规则放在其类别的顶部。过期规则被删除而不是注释掉，因为工作台是真相来源，而不是团队上季度感受的聊天日志。

### 规则与框架护栏

框架护栏（OpenAI Agents SDK guardrails、LangGraph 中断）在运行时级别强制执行规则。这节课中的规则集是这些护栏实现的人类可读、可审核的契约。你两者都需要：运行时在单轮中捕获违规，规则集证明运行时在做正确的事。

## 构建它

`code/main.py` 提供：

- 解析 `agent-rules.md` 并将规则加载到数据类中。
- `rule_checker.py` 风格检查函数，每个 `check` 引用一个。
- 一个演示 Agent 运行，违反两条规则，以及一个捕获它们的检查通过。

运行它：

```
python3 code/main.py
```

输出：解析后的规则集、运行轨迹、每条规则的通过/失败，以及保存到脚本旁边的 `rule_report.json`。

## 实际模式

三个模式将持续一个季度的规则集与一周内就会衰败的规则集区分开来。

**在编写时标记严重性。** 每条规则带有 `severity`：`block`、`warn` 或 `info`。检查器报告所有三种；运行时只在 `block` 时拒绝。大多数团队在早期高估严重性，然后在截止日期压力下悄悄削弱它；在编写时标记会迫使校准提前进行。与验证门配对使用（阶段 14 · 38），该门将任何 `block` 规则覆盖签名到 `overrides.jsonl` 审计日志。

**规则过期作为一种推动函数。** 每条规则带有 `expires_at` 日期（默认从编写起 90 天）。当一条未过期的规则连续 60 天零违规时，检查器发出警告；下一次季度审查要么证明保留它是对的，要么将其削弱为 `info`，或者删除它。Cloudflare 的生产 AI 代码审查数据（2026 年 4 月，30 天内跨 5,169 个仓库的 131,246 次审查运行）显示，带有明确过期的规则集保持在每个仓库 30 条规则以下；没有过期的规则集增长到 80+ 条，其中大多数从未触发过。

**Markdown 作为来源，JSON 作为缓存。** `agent-rules.md` 是编写文件；`agent-rules.lock.json` 是检查器在热路径中读取的缓存。锁由 pre-commit hook 重新生成。Markdown diff 可审核；JSON 解析在每轮中不出现。与 `package.json` / `package-lock.json` 和 `Cargo.toml` / `Cargo.lock` 形状相同。

## 使用它

在生产中：

- Claude Code、Codex、Cursor 在会话开始时读取规则，并在拒绝操作时引用它们。检查器在 CI 中重新运行它们以捕获静默漂移。
- OpenAI Agents SDK guardrails 注册相同的检查作为输入和输出 guardrails。Markdown 是文档表面；SDK 是运行时表面。
- LangGraph 中断在飞行中的节点违反规则时触发。中断处理程序读取规则、询问人类，然后恢复。

规则集在所有三个中都可移植，因为它只是 markdown 加上函数名。

## 交付它

`outputs/skill-rule-set-builder.md` 采访项目所有者，将其现有的散文指令分类到五个类别中，并发出一个版本化的 `agent-rules.md` 加上一个检查器桩。

## 练习

1. 如果你的产品确实需要，添加第六个类别。捍卫为什么它不会崩溃为五类之一。
2. 扩展检查器，使规则可以带有严重性（`block`、`warn`、`info`），并且报告相应聚合。
3. 将检查器接入 CI：如果最新 Agent 运行中有 block 严重性规则失败，则构建失败。
4. 每条规则添加一个"过期"字段。90 天无检查失败后，该规则需要审核。
5. 找一个真正的 `AGENTS.md`，将其重写为五类规则。它的行中有多少是可操作的？有多少是愿望式的？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| 操作规则 | "真正的指令" | 工作台可以在运行时检查的规则 |
| 愿望规则 | "要小心" | 没有检查的规则；要么删除要么升级 |
| 完成定义 | "验收" | 任务完成的目标的、文件支持的证明 |
| Block 严重性 | "硬规则" | 违规停止运行；除非操作员否则不能消音 |
| 规则过期 | "过期规则清理" | N 天无失败的规则需要退役 |

## 延伸阅读

- [OpenAI Agents SDK guardrails](https://platform.openai.com/docs/guides/agents-sdk/guardrails)
- [LangGraph 中断](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/)
- [Anthropic, 构建有效的 Agent](https://www.anthropic.com/research/building-effective-agents)
- [Rick Hightower, Agent RuleZ：确定性策略引擎](https://medium.com/@richardhightower/agent-rulez-a-deterministic-policy-engine-for-ai-coding-agents-9489e0561edf) — 生产中的 block/warn/info 严重性
- [Cloudflare, 大规模编排 AI 代码审查](https://blog.cloudflare.com/ai-code-review/) — 131k 次审查运行，规则组合经验
- [microservices.io, GenAI 开发平台 — 第 1 部分：护栏](https://microservices.io/post/architecture/2026/03/09/genai-development-platform-part-1-development-guardrails.html) — 规则和 CI 之间的纵深防御
- [Type-Checked 合规：确定性护栏（arXiv 2604.01483）](https://arxiv.org/pdf/2604.01483) — Lean 4 作为规则即检查的上界
- [logi-cmd/agent-guardrails](https://github.com/logi-cmd/agent-guardrails) — 合并门实现：范围、变异测试、违规预算
- 阶段 14 · 32 — 这套规则集所嵌入的最小工作台
- 阶段 14 · 38 — 消费规则报告的验证门
- 阶段 14 · 39 — 对规则合规评分的审核者 Agent