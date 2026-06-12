# 群组聊天与发言者选择

> AutoGen GroupChat 与 AG2 GroupChat 在 N 个 Agent 之间共享一次对话；一个选择器函数（LLM、轮询或自定义）决定谁下一个发言。这是新兴多 Agent 对话的原型 —— Agent 不在静态图中知道自己扮演的角色，只是对共享消息池做出反应。AutoGen v0.2 的 GroupChat 语义在 AG2 分支中得以保留；AutoGen v0.4 将其重写为事件驱动的 Actor 模型。微软于 2026 年 2 月将 AutoGen 置于维护模式，并与 Semantic Kernel 合并为 Microsoft Agent Framework（2026 年 2 月 RC）。GroupChat 原语在 AG2 和 Microsoft Agent Framework 中都得到了保留 —— 学会一次，到处使用。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置条件：** 阶段 16 · 04（原始模型）
**时间：** 约 60 分钟

## 问题

静态图（LangGraph）在工作流已知时很出色。但真正的对话不是静态的：有时编码者会问评审者，有时问研究者，有时问写作者。将每个可能的交接硬编码进去会产生边爆炸。你想要的是 *Agent 对共享池做出反应*，由某个函数决定谁下一个发言。

这正是 AutoGen GroupChat 所做的事情。

## 概念

### 形态

```
              ┌─── 共享池 ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ （所有人都读取全部消息）
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "下一个发言者 = C"
```

每个 Agent 都能看到每条消息。选择器函数在每一轮被调用，以选出下一个发言者。

### 三种选择器类型

**轮询（Round-robin）。** 固定循环。确定性。与 N 成线性比例扩展，但忽略上下文 —— 当话题是法律审查时，编码者仍然会获得发言权。

**LLM 选择。** 调用 LLM，读取最近的池并返回最佳下一个发言者。上下文感知但较慢：每一轮都增加一次 LLM 调用。AutoGen 的默认选项。

**自定义。** 一个具有任意逻辑的 Python 函数。典型用法：LLM 选择 + 回退规则（例如，"编码者之后总是把发言权交给验证者"）。

### ConversableAgent API

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` 持有选择器。当一个 Agent 完成一轮时，管理器调用选择器，返回下一个 Agent。循环继续直到满足终止条件。

### 终止

三种常见模式：

- **最大轮数。** 总轮次的硬上限。
- **"TERMINATE" 标记。** Agent 可以发送一个哨兵消息；当出现该消息时管理器停止。
- **目标达成检查。** 每轮运行一个轻量级验证器，并在完成时停止聊天。

### AutoGen → AG2 分裂与 Microsoft Agent Framework 合并

2025 年初，微软开始围绕事件驱动的 Actor 模型对 AutoGen（v0.4）进行重大重写。社区将 AutoGen v0.2 的 GroupChat 语义分叉为 AG2，保留早期采用者已集成的 API。

2026 年 2 月，微软宣布 AutoGen 将进入维护模式，事件驱动的 Actor 模型合并到 **Microsoft Agent Framework**（2026 年 2 月 RC，现与 Semantic Kernel 合并）。GroupChat 概念在两条路线中都得到了保留；实现细节有所不同。AG2 是 v0.2 兼容代码的首选上游。

### 何时适合使用 GroupChat

- **新兴对话。** 你不想预先连接每个可能的下一个发言者。
- **角色混合任务。** 编码者问研究者，研究者问档案管理员，档案管理员回头问编码者。流程不是 DAG。
- **探索性问题解决。** 想的是"头脑风暴会议"，不是"装配线"。

### 何时会失败

- **严格确定性。** LLM 选择器可能不一致。同样的提示，不同的运行，不同的下一个发言者。
- **谄媚级联。** Agent 倾向于服从说话最有信心的人。需要在提示中明确反提示。
- **上下文膨胀。** 每个 Agent 读取每条消息；10 轮之后上下文巨大。使用投影（第 15 课）来限制视图。
- **热门发言者。** 一个 Agent 在对话中占主导地位，因为选择器偏向其专长。将发言者平衡作为选择器特性引入。

### 群聊 vs 监督者

相同的原语，不同的默认值：

- 监督者：一个 Agent 计划，其他人执行。选择器是"问规划者该做什么"。
- 群聊：所有 Agent 都是对等的；选择器是对共享池的一个函数。

两者都使用第 04 课的四个原语。群聊默认使用 LLM 选择的编排和完整池共享状态。

## 构建

`code/main.py` 用标准库从头实现了一个 GroupChat。三个 Agent（编码者、评审者、管理者）、轮询和 LLM 选择变体，以及基于 `TERMINATE` 标记的终止。

演示打印对话记录以及两种变体的选择器决策轨迹。

运行：

```
python3 code/main.py
```

## 使用

`outputs/skill-groupchat-selector.md` 为给定任务配置 GroupChat 选择器 —— 轮询 vs LLM 选择 vs 自定义，以及使用哪些选择器输入（最近消息、Agent 专长、轮次计数）。

## 交付

检查清单：

- **最大轮数上限。** 始终设置。典型任务 10-20 轮。
- **发言者平衡指标。** 追踪每个 Agent 的发言次数；当不平衡超过阈值时发出警告。
- **终止标记。** `TERMINATE` 或专用验证器 Agent。
- **投影或作用域内存。** 约 10 条消息后，考虑给每个 Agent 仅提供一个作用域视图以防止上下文膨胀。
- **选择器日志记录。** 对于 LLM 选择的变体，记录选择器的输入和选择。否则调试是不可能的。

## 练习

1. 运行 `code/main.py`。比较轮询与 LLM 选择下的对话。在每种情况下哪个 Agent 占主导？
2. 在选择器中添加"每个 Agent 最大发言次数"规则。它如何影响记录？
3. 实现目标达成的终止：当评审者返回"已批准"时停止。在轮次上限之前它触发多少次？
4. 阅读 AutoGen 稳定版文档中关于 GroupChat 的内容（https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html）。识别 `GroupChatManager` 使用的默认选择器。
5. 阅读 AG2 仓库（https://github.com/ag2ai/ag2）并将其 v0.2 GroupChat 与 v0.4 事件驱动版本进行比较。v0.4 添加了什么具体属性（吞吐量、容错性、可组合性）？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| GroupChat | "Agent 在一个聊天室里" | 共享消息池 + 选择器函数。AutoGen / AG2 原语。 |
| 发言者选择 | "谁下一个发言" | 选择下一个 Agent 的函数。轮询、LLM 选择或自定义。 |
| GroupChatManager | "会议主持人" | AutoGen 组件，拥有选择器并循环执行轮次。 |
| ConversableAgent | "基础 Agent" | AutoGen 基类；可以发送和接收消息的 Agent。 |
| 终止标记 | "停止词" | 结束对话的哨兵字符串（通常为 `TERMINATE`）。 |
| 热门发言者 | "一个 Agent 占主导" | 选择器持续选择同一个 Agent 的失败模式。 |
| 上下文膨胀 | "池无限增长" | 每个 Agent 读取每条先前消息；上下文随轮次增长。 |
| 投影 | "作用域视图" | 共享池的角色特定视图，用于防止上下文膨胀。 |

## 延伸阅读

- [AutoGen 群聊文档](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — 参考实现
- [AG2 仓库](https://github.com/ag2ai/ag2) — 社区 AutoGen v0.2 延续
- [Microsoft Agent Framework 文档](https://microsoft.github.io/agent-framework/) — 合并后的继承者，2026 年 2 月 RC
- [AutoGen v0.4 发布说明](https://microsoft.github.io/autogen/stable/) — 事件驱动 Actor 模型重写详情