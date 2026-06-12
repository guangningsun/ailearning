# MCP 采样 — 服务器端请求的 LLM 补全与 Agent 循环

> 大多数 MCP 服务器都是哑巴执行器：接收参数、运行代码、返回内容。采样让服务器反转方向：它请求客户端的 LLM 做出决策。这使得服务器托管的 Agent 循环无需服务器持有任何模型凭证。SEP-1577 于 2025-11-25 合并，新增了采样请求内部的工具，使得循环能够包含更深入的推理。漂移风险提示：SEP-1577 工具内采样形态在 2026 年 Q1 仍处于实验阶段，SDK API 仍在调整中。

**类型：** 构建型
**语言：** Python（标准库、采样工具）
**前置条件：** 阶段 13 · 07（MCP 服务器）、阶段 13 · 10（资源与提示词）
**时间：** 约 75 分钟

## 学习目标

- 解释 `sampling/createMessage` 解决了什么问题（无需服务器端 API 密钥的服务器托管循环）。
- 实现一个服务器，通过多轮提示词请求客户端采样并返回补全结果。
- 使用 `modelPreferences`（成本/速度/智能优先级）引导客户端的模型选择。
- 构建一个 `summarize_repo` 工具，其内部通过采样迭代而非硬编码行为。

## 问题

一个用于代码摘要工作流的实用 MCP 服务器需要：遍历文件树、选择要读取的文件、综合摘要并返回。LLM 推理发生在哪里？

方案 A：服务器调用自己的 LLM。需要 API 密钥、按服务器计费、每个用户成本高昂。

方案 B：服务器返回原始内容；客户端的 Agent 完成推理。可行，但会把服务器逻辑混入客户端提示词，容易脆弱。

方案 C：服务器通过 `sampling/createMessage` 请求客户端的 LLM。服务器保留算法（读取哪些文件、做几轮遍历），客户端保留计费权和模型选择权。服务器完全不持有凭证。

采样就是方案 C。它是一种让可信服务器托管 Agent 循环而不必成为完整 LLM 主机的机制。

## 概念

### `sampling/createMessage` 请求

服务器发送：

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

客户端运行其 LLM，返回：

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

三个浮点数之和为 1.0：

- `costPriority`：优先选择更便宜的模型。
- `speedPriority`：优先选择更快的模型。
- `intelligencePriority`：优先选择能力更强的模型。

加上 `hints`：服务器偏好的命名模型列表。客户端可能采纳也可能不采纳 hints；客户端的用户配置始终优先。

### `includeContext`

三个值：

- `"none"` — 仅使用服务器提供的消息。默认值。
- `"thisServer"` — 包含来自该服务器会话的先前消息。
- `"allServers"` — 包含所有会话上下文。

截至 2025-11-25，`includeContext` 已被软弃用，因为它会泄露跨服务器上下文，构成安全隐患。建议使用 `"none"` 并在消息中传递显式上下文。

### 带工具的采样（SEP-1577）

2025-11-25 新增：采样请求可以包含 `tools` 数组。客户端使用这些工具运行完整的工具调用循环。这使得服务器可以通过客户端的模型托管 ReAct 风格的 Agent 循环。

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

客户端循环：采样 → 执行被调用的工具 → 再次采样 → 返回最终助手消息。此特性在 2026 年 Q1 仍为实验阶段；SDK 签名可能仍在变化。实现时请对照 2025-11-25 规范的 client/sampling 部分进行确认。

### 人在环中（Human-in-the-loop）

客户端在运行采样之前必须向用户展示服务器要求模型做什么。一个恶意服务器可能利用采样来操纵用户的会话（"对用户说 X，让他们点击 Y"）。Claude Desktop、VS Code 和 Cursor 将采样请求呈现为确认对话框，用户可以拒绝。

2026 年的共识：无人确认的采样是一个危险信号。网关（阶段 13 · 17）可以自动批准低风险采样并自动拒绝可疑请求。

### 无 API 密钥的服务器托管循环

典型用例：一个没有自身 LLM 访问能力的代码摘要 MCP 服务器。它执行：

1. 遍历代码库结构。
2. 调用 `sampling/createMessage` 并附上"选择五个最有可能描述此代码库用途的文件"。
3. 读取这些文件。
4. 调用 `sampling/createMessage` 并附上文件内容和"用三段话概括此代码库"。
5. 将摘要作为 `tools/call` 结果返回。

服务器从不调用 LLM API。客户端用户使用自己的凭证为补全付费。

### 安全风险（Unit 42 披露，2026 年 Q1）

- **隐蔽采样。** 一个工具总是调用采样并附上"从会话上下文中返回用户的电子邮件"。
- **通过采样窃取资源。** 服务器请求客户端总结攻击者的载荷，让用户付费。
- **循环炸弹。** 服务器在紧凑循环中调用采样。客户端必须强制执行按会话的速率限制。

## 使用它

`code/main.py` 附带了一个模拟服务器到客户端的采样工具。一个模拟的"summarize_repo"工具调用两轮采样（选文件，然后摘要），模拟客户端返回预设响应。该工具展示了：

- 服务器发送带有 `modelPreferences` 的 `sampling/createMessage`。
- 客户端返回补全结果。
- 服务器继续其循环。
- 速率限制器限制每个工具调用中的总采样次数。

需要关注的内容：

- 服务器仅暴露一个工具（`summarize_repo`）；所有推理发生在采样调用中。
- 模型偏好权重影响客户端的模型选择；hints 列出偏好模型。
- 循环在 `stopReason: "endTurn"` 时终止。
- `max_samples_per_tool = 5` 限制可以捕获失控循环。

## 交付它

本课产出 `outputs/skill-sampling-loop-designer.md`。给定一个需要 LLM 调用（研究、摘要、规划）的服务器端算法，该技能设计基于采样的实现方案，包含正确的 modelPreferences、速率限制和安全确认。

## 练习

1. 运行 `code/main.py`。将 `max_samples_per_tool` 改为 2 并观察速率限制截断。

2. 实现 SEP-1577 工具内采样变体：采样请求携带 `tools` 数组。验证客户端循环在返回最终补全前执行了这些工具。注意漂移风险：SDK 签名在 2026 年 H1 前可能仍会变化。

3. 添加人在环中确认：在服务器第一次 `sampling/createMessage` 之前暂停并等待用户批准。被拒绝的调用返回类型化的拒绝。

4. 添加按用户的速率限制器，以客户端会话为 key。同一用户的同服务器循环应共享一个配额。

5. 设计一个 `summarize_pdf` 工具，使用采样来选择要包含的块。勾勒发送的消息。当 `modelPreferences.intelligencePriority` 为 0.1 对比 0.9 时，行为如何变化？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| 采样 (Sampling) | "服务器到客户端的 LLM 调用" | 服务器请求客户端模型返回一个补全 |
| `sampling/createMessage` | "该方法" | 采样请求的 JSON-RPC 方法 |
| `modelPreferences` | "模型优先级" | 成本/速度/智能权重加上命名 hints |
| `includeContext` | "跨会话泄露" | 软弃用的上下文包含模式 |
| SEP-1577 | "采样中的工具" | 在采样中允许工具，以实现服务器托管的 ReAct |
| 人在环中 (Human-in-the-loop) | "用户确认" | 客户端在运行前向用户展示采样请求 |
| 循环炸弹 (Loop bomb) | "失控采样" | 服务器端无限采样循环；客户端必须速率限制 |
| 隐蔽采样 (Covert sampling) | "隐藏推理" | 恶意服务器在采样提示词中隐藏意图 |
| 资源窃取 (Resource theft) | "使用用户的 LLM 配额" | 服务器强迫客户端在被拒绝的采样上花费 |
| `stopReason` | "生成停止的原因" | `endTurn`、`stopSequence` 或 `maxTokens` |

## 进一步阅读

- [MCP — 概念：采样](https://modelcontextprotocol.io/docs/concepts/sampling) — 采样的高级概述
- [MCP — 客户端采样规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) — 规范的 `sampling/createMessage` 形态
- [MCP — GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) — 采样中工具的规范演进提案（实验性）
- [Unit 42 — MCP 攻击向量](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) — 隐蔽采样和资源窃取模式
- [Speakeasy — MCP 采样核心概念](https://www.speakeasy.com/mcp/core-concepts/sampling) — 带客户端代码示例的演练