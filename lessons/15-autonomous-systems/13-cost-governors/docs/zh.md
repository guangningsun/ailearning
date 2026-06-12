# 动作预算、迭代上限与成本Governor

> 一个中型电商Agent的月度 LLM 费用在团队启用"订单追踪"技能后，从 $1,200 跃升至 $4,800。这不是定价 bug。这是 Agent 发现了一个新循环并持续在其中消耗。微软的 Agent Governance Toolkit（2026年4月2日）将针对此类问题的防御标准化：每请求 `max_tokens`、每任务 token 和美元预算、每日/每月上限、迭代上限、分层模型路由、提示缓存、上下文窗口化、高成本动作的 HITL 检查点、预算超限时的终止开关。Anthropic 的 Claude Code Agent SDK 以不同名称提供了相同的原语。财务速率限制——例如 10 分钟内超过 $50 就切断访问——比月度上限能更快地捕获循环。

**类型：** 学习型
**语言：** Python（标准库，分层 cost-governor 模拟器）
**前置条件：** 阶段 15 · 10（权限模式）、阶段 15 · 12（持久执行）
**时间：** 约 60 分钟

## 问题

自主 Agent 在每一轮都会花费真金白银。聊天机器人的糟糕输出是一条糟糕的回复；Agent 的糟糕循环是一笔账单。行业记录在案的失败模式术语是"钱包拒绝"（Denial of Wallet）——Agent 持续推理、持续调用工具、持续计费，没有任何东西阻止它，因为根本没有设计这样的机制。

修复方案不是一个数字。而是一个在不同时间尺度和粒度上的分层限制栈：每请求、每任务、每小时、每日、每月。一个设计良好的限制栈能在几分钟内捕获失控循环，在几小时内捕获缓慢泄漏，在一天内捕获糟糕的发布版本。当 Agent 是长期且自主的时候，这套限制栈能始终保住预算。

这是一堂工程课：数学很简单，纪律才是团队失败的地方。下面的限制清单在微软 Agent Governance Toolkit 或 Anthropic Claude Code Agent SDK 文档中都有命名。

## 概念

### 成本Governor栈

1. **`max_tokens` 每请求。** 简单。防止任何一次调用产生无限长的补全。
2. **每任务 token 预算。** 在整个运行过程中，不超过 N 个 token。达到上限时硬性停止。
3. **每任务美元预算。** 与 token 相同但以货币计价。Claude Code 中的 `max_budget_usd`。
4. **每工具调用上限。** 不超过 N 次 `WebFetch` 调用、N 次 `shell_exec` 调用等。
5. **迭代上限（`max_turns`）。** 总的 Agent 循环迭代次数；防止无限推理循环。
6. **每分钟 / 每小时 / 每日 / 每月上限。** 滚动窗口。在不同时间尺度上捕获泄漏。
7. **财务速率限制。** 例如，"如果 10 分钟内支出超过 $50，切断访问。" 在月度上限触发之前捕获基于循环的消耗。
8. **分层模型路由。** 默认使用较小的模型；只有当分类器判断任务值得时才升级到较大的模型。
9. **提示缓存。** 系统提示和稳定上下文存储在提供商缓存中；重新发送 token 的成本接近零。
10. **上下文窗口化。** 压缩 / 摘要以保持活动上下文低于阈值；直接降低 token 成本。
11. **高成本动作的 HITL 检查点。** 在执行已知成本高昂的动作（长时间工具调用、大型下载、代价高昂的模型升级）之前，要求人工点击。
12. **预算超限时的终止开关。** 任何上限触发时会话中止。记录上限；需要单独的重新启用路径。

### 为什么需要栈，而不是一个上限

单一月度上限只在钱包掏空后才能捕获失控的 Agent。单一每请求上限在会话级别捕获不了任何东西。不同的失败模式需要不同的时间尺度：

- **失控循环**（Agent 卡在 5 秒重试中）：由速率限制捕获。
- **缓慢泄漏**（Agent 每个任务消耗约 2 倍预期工作）：由每日上限捕获。
- **糟糕发布**（新版本消耗 5 倍 token）：由每周 / 每月上限捕获。
- **合法激增**（真实需求，不是 bug）：由每小时 / 每日上限捕获，有清晰的日志。

### Claude Code 的预算表面

Claude Code Agent SDK 暴露的（公开文档）：

- `max_turns` — 迭代上限。
- `max_budget_usd` — 美元上限；超限时会话中止。
- `allowed_tools` / `disallowed_tools` — 工具白名单和黑名单。
- 工具使用前的钩子点，用于自定义成本核算。

与权限模式阶梯结合使用（课程 10）。没有 `max_budget_usd` 的 `autoMode` 会话是无治理的自主性。Anthropic 明确将 Auto Mode 定位为需要预算控制；分类器与成本是正交的。

### EU AI 法案、OWASP Agentic Top 10

微软的 Agent Governance Toolkit 涵盖了 OWASP Agentic Top 10 和 EU AI 法案第 14 条（人类监督）要求。对于在欧盟的生产环境，日志记录和上限执行不是可选项。

### 观察到的 $1,200 → $4,800 案例

微软文档中的真实案例：一个电商 Agent 在添加新工具后月度成本增加了两倍。该工具允许 Agent 在每个会话期间轮询订单状态。没有循环检测。没有每工具上限。没有周环比增长警报。修复方案是每工具上限加每日增长警报。这是一个模板：每个新工具表面都是一个新的潜在循环；每个新工具都需要自己的上限和自己的警报。

## 使用它

`code/main.py` 模拟了有和没有分层成本Governor栈的 Agent 运行。模拟的 Agent 在若干轮之后漂移到轮询循环；分层栈在速度窗口内捕获它，而单一月度上限要等到几天后才触发。

## 发布它

`outputs/skill-agent-budget-audit.md` 审计拟议的 Agent 部署的成本Governor栈，并标记缺失的层。

## 练习

1. 运行 `code/main.py`。确认在轮询循环轨迹上速率限制比迭代上限先触发。现在禁用速率限制，测量 Agent 在迭代上限捕获它之前"消耗"了多少。

2. 为浏览器 Agent（课程 11）设计一套每工具上限。哪个工具需要最严格的上限？哪个工具可以无限制运行而没有风险？

3. 阅读微软 Agent Governance Toolkit 文档。列出工具包命名的每种上限类型。将每种映射到一种失败模式（失控循环、缓慢泄漏、糟糕发布、激增）。

4. 为一个真实任务（例如"对仓库中的 50 个 issue 进行分类"）的夜间无人值守运行定价。将 `max_budget_usd` 设置为你的点估计的 2 倍。为 2 倍提供理由。

5. Claude Code 的 `max_budget_usd` 在会话聚合成本上触发。设计一个你会在外部强制的互补速率限制。什么触发了切断，重启需要什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| Denial of Wallet | "失控账单" | Agent 循环产生支出，但没有上限来停止它 |
| max_tokens | "每请求上限" | 单次补全大小的上限 |
| max_turns | "迭代上限" | 会话中 Agent 循环迭代的上限 |
| max_budget_usd | "美元终止开关" | 会话成本上限；超限时中止 |
| Velocity limit | "速率上限" | 短时间窗口内的支出限制（例如 $50 / 10 分钟） |
| Tiered routing | "小模型优先" | 默认使用廉价模型；只有当分类器认为值得时才升级 |
| Prompt caching | "缓存的系统提示" | 提供商侧缓存将重新发送 token 成本降至接近零 |
| HITL checkpoint | "人工批准门" | 高成本动作前需要人工点击 |

## 延伸阅读

- [Anthropic Claude Code Agent SDK — agent loop and budgets](https://code.claude.com/docs/en/agent-sdk/agent-loop) — `max_turns`、`max_budget_usd`、工具白名单。
- [Microsoft Agent Framework — human-in-the-loop and governance](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — cost-governor 检查点。
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 提供商侧成本控制。
- [Anthropic — Prompt caching (Claude API docs)](https://platform.claude.com/docs/en/prompt-caching) — 缓存机制。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 长期 Agent 的成本概况。