# Human-in-the-Loop：先提议后提交

> 2026 年关于 HITL 的共识是明确的。它不是"智能体询问，用户点击批准"，而是先提议后提交：拟议的动作被持久化到一个持久化存储中，并附带一个幂等性密钥；向审查者展示时包含意图、数据谱系、所涉及的权限、爆炸半径和回滚计划；只有在收到肯定确认后才提交执行；执行后进行验证，确认副作用确实发生了。LangGraph 的 `interrupt()` 加上 PostgreSQL 检查点机制、Microsoft Agent Framework 的 `RequestInfoEvent` 以及 Cloudflare 的 `waitForApproval()` 都实现了相同的模式。典型的失败模式是橡皮图章式批准：不经审查就点击"批准"。有记录的缓解措施是挑战-响应式检查清单。

**类型：** 学习型
**语言：** Python（标准库，带幂等性的先提议后提交状态机）
**前置条件：** 阶段 15 · 12（持久化执行）、阶段 15 · 14（绊线）
**时间：** 约 60 分钟

## 问题

智能体执行了一个动作。用户需要决定：批准还是不批准。如果决策是即时的，那它可能就不是真正的审查。如果决策是结构化的，它很慢但值得信赖。工程学的问题是：如何让结构化审查成为阻力最小的路径。

2023 年代的 HITL 模式是同步提示："智能体想向 X 发送主题为 Y 的邮件——批准吗？"用户点击批准。每个人都觉得系统是安全的。实际上这个界面被大量橡皮图章式地使用：用户快速批准，批准的可预测性很低，而且当智能体出问题的时候，审计追踪显示的是一长串用户无法回忆起的批准记录。

2026 年的模式——先提议后提交——将 HITL 移到了一个持久化底层之上，附加了结构化元数据，并要求肯定的提交。每个托管智能体 SDK 都发布了一个版本：LangGraph `interrupt()`、Microsoft Agent Framework `RequestInfoEvent`、Cloudflare `waitForApproval()`。API 名称不同，但模式相同。

## 概念

### 先提议后提交状态机

1. **提议。** 智能体产生一个拟议的动作。持久化到一个持久化存储（PostgreSQL、Redis、Durable Object）中。包括：
   - 意图（智能体为什么要这样做）
   - 数据谱系（什么来源导致了这个提议）
   - 所涉及的权限（哪些作用域/文件/端点）
   - 爆炸半径（最坏情况是什么）
   - 回滚计划（如果提交了，我们如何撤销）
   - 幂等性密钥（每个提议唯一；重新提交返回同一条记录）
2. **展示。** 审查者看到带有所有元数据的提议。审查者是一个人（不是智能体自己审查自己）。
3. **提交。** 肯定确认。动作执行。
4. **验证。** 执行后，重新读取目标资源并确认副作用确实发生了。如果验证步骤失败，系统处于已知的不良状态，告警机制启动。

### 幂等性密钥

没有幂等性密钥，一次临时故障后的重试可能导致一个已批准的动作被重复执行。具体例子：用户批准"从 A 向 B 转账 100 美元"。网络抖动。工作流重试。用户只批准了一次，但转账执行了两次。幂等性密钥将批准与单一、唯一的副作用绑定在一起；第二次执行是一个空操作。

这与 Stripe 和 AWS API 使用的幂等性模式相同。在 Microsoft Agent Framework 文档中明确指出将其重用于智能体批准。

### 持久化：为什么批准比进程更长寿

批准的候诊室是智能体不拥有的状态。工作流暂停（第 12 课）。当批准到达时，工作流从那个确切的点恢复。这就是为什么 LangGraph 将 `interrupt()` 与 PostgreSQL 检查点配对，而不是仅用内存状态——两天后的批准仍然能找到完整的工作流。

### 橡皮图章式批准及挑战-响应缓解措施

HITL 的默认 UI（"批准"/"拒绝"按钮）产生快速批准，但没有真正的审查。有记录的缓解措施：挑战-响应式检查清单，要求在启用"批准"按钮之前对具体问题给出肯定的答案。具体形式：

- "你了解这个操作涉及什么资源吗？[ ]"
- "你验证过爆炸半径是可接受的吗？[ ]"
- "如果这个操作失败了，你有不有回滚计划？[ ]"

这不是为了官僚主义而官僚主义——而是一个强制功能。无法勾选这些框的审查者要么要求澄清（升级），要么拒绝（安全默认值）。Anthropic 的智能体安全研究明确引用了检查清单驱动的 HITL 作为橡皮图章批准模式的缓解措施。

### 什么算作重大影响

并非每个动作都需要先提议后提交。2026 年的指导：

- **重大影响动作**（始终需要 HITL）：不可逆写入、金融交易、对外通信、生产数据库变更、破坏性文件系统操作。
- **可逆动作**（有时需要 HITL）：本地文件编辑、预发环境变更、有明确回滚的可逆写入。
- **读取和检查**（永远不需要 HITL）：读取文件、列出资源、调用只读 API。

### 执行后验证

"提交已运行"不等于"副作用已发生"。网络分区和竞态条件可能导致工作流认为它成功了，而后端实际上没有持久化。验证步骤在提交后重新读取目标资源以确认。这与带 `RETURNING` 子句的数据库事务或在 `PutObject` 之后执行 `GetObject` 的模式相同。

### EU AI 法案第 14 条

第 14 条要求对高风险 AI 系统进行有效的人工监督。"有效"不是装饰性的。监管语言明确排除了橡皮图章模式。先提议后提交加上挑战-响应是符合 Microsoft Agent Governance Toolkit 合规文档中第 14 条审查的模式。

## 使用它

`code/main.py` 用标准库 Python 实现了先提议后提交状态机。持久化存储是一个 JSON 文件。幂等性密钥是（thread_id，action_signature）的哈希值。驱动程序模拟三种情况：干净的批准流程、临时故障后的重试（不能重复执行），以及橡皮图章式默认值与挑战-响应式流程的对比。

## 交付它

`outputs/skill-hitl-design.md` 审查一个提议的 HITL 工作流是否符合先提议后提交模式，并标记缺失的元数据、幂等性、验证或挑战-响应层。

## 练习

1. 运行 `code/main.py`。确认对已批准提议的重试使用了持久化记录，而不会重新执行。现在将幂等性密钥改为包含时间戳，并展示重试会导致重复执行。

2. 将 `rollback` 字段扩展到提议记录中。模拟一个验证步骤失败的执行。展示回滚自动触发。

3. 阅读 Microsoft Agent Framework 的 `RequestInfoEvent` 文档。识别 API 包含的一个元数据字段，而玩具引擎缺少这个字段。添加它并解释它防止什么。

4. 为一个特定动作（如"发布到公开的 Twitter 账号"）设计一个挑战-响应检查清单。审查者必须回答哪三个问题？为什么是这三个？

5. 选一个同步"批准？"提示就足够的场景（不需要持久化存储）。解释为什么，并说明你正在接受哪类风险。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|---|---|---|
| 先提议后提交 (Propose-then-commit) | "两阶段批准" | 持久化的提议 + 肯定提交 + 验证 |
| 幂等性密钥 (Idempotency key) | "重试安全令牌" | 每个提议唯一；第二次执行是空操作 |
| 数据谱系 (Data lineage) | "它从哪里来" | 导致该提议的特定来源内容 |
| 爆炸半径 (Blast radius) | "最坏情况" | 如果动作出错，影响的范围 |
| 橡皮图章 (Rubber-stamp) | "快速批准" | 未经真正审查就点击"批准" |
| 挑战-响应 (Challenge-and-response) | "强制检查清单" | 审查者必须肯定地回答具体问题 |
| RequestInfoEvent | "MS Agent Framework 原语" | 带结构化元数据的持久化 HITL 请求 |
| `interrupt()` / `waitForApproval()` | "框架原语" | LangGraph / Cloudflare 的等价实现，模式相同 |

## 延伸阅读

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`，持久化批准。
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` 和 Durable Objects。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — HITL 作为长周期风险的缓解措施。
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — 高风险系统的监管基线。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 围绕监督的宪法框架。