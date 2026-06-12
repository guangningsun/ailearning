# A2A — 代理到代理协议

> MCP 是代理到工具。A2A（Agent2Agent）是代理到代理 —— 一个让不透明的代理跨不同框架协作的开放协议。Google 于 2025 年 4 月发布，2025 年 6 月捐赠给 Linux 基金会，2026 年 4 月达到 v1.0，获得 150 多家支持者包括 AWS、Cisco、Microsoft、Salesforce、SAP 和 ServiceNow。它吸收了 IBM 的 ACP 并添加了 AP2 支付扩展。本课介绍 Agent Card、任务生命周期以及两种传输绑定。

**类型：** 构建型
**语言：** Python（标准库，Agent Card + 任务工具）
**前置条件：** 阶段 13 · 06（MCP 基础）、阶段 13 · 08（MCP 客户端）
**时间：** 约 75 分钟

## 学习目标

- 区分代理到工具（MCP）和代理到代理（A2A）的用例。
- 在 `/.well-known/agent.json` 发布 Agent Card，包含技能和端点元数据。
- 遍历任务生命周期（submitted → working → input-required → completed / failed / canceled / rejected）。
- 使用带 Parts 的 Messages（text、file、data）和 Artifacts 作为输出。

## 问题

客服代理需要将报告撰写委托给专门的写作代理。A2A 之前的选项：

- 自定义 REST API。可以工作，但每对组合都是一次性的。
- 共享代码库。需要两个代理运行相同的框架。
- MCP。不适用：MCP 用于调用工具，而非两个代理在保留各自不透明内部推理的情况下协作。

A2A 填补了这个空白。它将交互建模为一个代理向另一个代理发送任务，包含生命周期、消息和产物。被调用代理的内部状态保持不透明 —— 调用者只看到任务状态转换和最终输出。

A2A 是"让跨框架的代理相互通信"的协议。它不会取代 MCP；两者是互补的。

## 概念

### Agent Card

每个符合 A2A 规范的代理在 `/.well-known/agent.json` 发布一张卡片：

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

发现是基于 URL 的：获取卡片，了解 A2A 端点的 URL，枚举技能。

### 带签名的 Agent Card（AP2）

AP2 扩展（2025 年 9 月）为 Agent Card 添加了加密签名。发布者用 JWT 签署自己的卡片；消费者验证。防止冒名顶替。

### 任务生命周期

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (通过消息循环)
```

客户端通过 `tasks/send` 发起。被调用代理转换状态；客户端通过 SSE 订阅状态更新或轮询。

### 消息和 Parts

一条消息携带一个或多个 Parts：

- `text` — 纯文本内容。
- `file` — 带 mimeType 的 base64 blob。
- `data` — 类型化的 JSON 载荷（被调用代理的结构化输入）。

示例：

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### 产物

输出是产物，而非原始字符串。产物是有名称和类型的输出：

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

产物可以作为块流式传输。调用者累积它们。

### 两种传输绑定

1. **JSON-RPC over HTTP。** `/a2a` 端点，POST 用于请求，可选的 SSE 用于流式传输。默认绑定。
2. **gRPC。** 用于 gRPC 原生的企业环境。

两种绑定承载相同的逻辑消息结构。

### 不透明保持

一个关键设计原则：被调用代理的内部状态是不透明的。调用者看到任务状态和产物。被调用代理的思维链、它的工具调用、它的子代理委托 —— 全部不可见。这与 MCP 不同，MCP 中工具调用是透明的。

理由：A2A 使竞争对手能够协作而不暴露内部。A2A 可以是"调用这个客服代理"，而调用者无需了解该代理如何实现服务。

### 时间线

- **2025-04-09。** Google 宣布 A2A。
- **2025-06-23。** 捐赠给 Linux 基金会。
- **2025-08。** 吸收 IBM 的 ACP。
- **2025-09。** AP2 扩展（代理支付）发布。
- **2026-04。** v1.0 发布，获得 150 多家支持组织。

### 与 MCP 的关系

| 维度 | MCP | A2A |
|-----------|-----|-----|
| 用例 | 代理到工具 | 代理到代理 |
| 不透明度 | 透明工具调用 | 不透明内部推理 |
| 典型调用者 | 代理运行时 | 另一个代理 |
| 状态 | 工具调用结果 | 带生命周期的任务 |
| 授权 | OAuth 2.1（阶段 13 · 16） | JWT 签名 Agent Card（AP2） |
| 传输 | Stdio / Streamable HTTP | JSON-RPC over HTTP / gRPC |

当你想调用特定工具时使用 MCP。当你想将整个任务委托给另一个代理时使用 A2A。许多生产系统同时使用两者：一个代理在其工具层使用 MCP，在其协作层使用 A2A。

## 使用它

`code/main.py` 实现了一个最小 A2A 工具：一个研究代理发布其卡片，一个写作代理接收带有 PDF 和文本指令的 `tasks/send`，经历 working → input-required → working → completed 的转换，并返回一个文本产物。全部使用标准库；使用内存传输来专注于消息结构。

需要关注的重点：

- Agent Card JSON 结构。
- 任务 ID 分配和状态转换。
- 混合类型的 parts 消息。
- 任务中期的 input-required 分支。
- 完成时的产物返回。

## 交付它

本课产出 `outputs/skill-a2a-agent-spec.md`。给定一个应该可被其他代理调用的新代理，该技能生成 Agent Card JSON、技能模式和端点蓝图。

## 练习

1. 运行 `code/main.py`。追踪完整的任务生命周期，包括被调用代理请求澄清的 input-required 暂停。

2. 添加带签名的 Agent Card。用卡片规范 JSON 上的 HMAC 签名。编写验证器并确认它在卡片被篡改时失败。

3. 实现任务流式传输：写作代理通过 SSE 发出三个增量产物块，调用者累积它们。

4. 设计一个包装 MCP 服务器的 A2A 代理。将每个 MCP 工具映射到 A2A 技能。注意权衡 —— 失去了什么不透明度？

5. 阅读 A2A v1.0 公告并识别截至 2026 年 4 月尚无任何框架实现的一个功能。（提示：它与多跳任务委托有关。）

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| A2A | "代理到代理协议" | 用于不透明代理协作的开放协议 |
| Agent Card | "`.well-known/agent.json`" | 发布的元数据，描述代理的技能和端点 |
| 技能（Skill） | "可调用单元" | 代理支持的有名称的操作（MCP 工具的类比） |
| 任务（Task） | "委托单元" | 带生命周期和最终产物的工作项 |
| 消息（Message） | "任务输入" | 携带 Parts（text、file、data） |
| Part | "类型化块" | 消息的 `text` / `file` / `data` 元素 |
| 产物（Artifact） | "任务输出" | 完成时返回的命名类型化输出 |
| AP2 | "代理支付协议" | 用于信任和支付的签名 Agent Card 扩展 |
| 不透明度（Opacity） | "黑盒协作" | 被调用代理的内部对调用者隐藏 |
| Input-required | "任务暂停" | 代理需要更多信息时的生命周期状态 |

## 延伸阅读

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — 规范 A2A 规范
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — 参考实现和 SDK
- [Linux 基金会 — A2A 启动新闻稿](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — 2025 年 6 月治理转移
- [Google Cloud — A2A 协议升级](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — 路线图和合作伙伴势头
- [Google Dev — A2A 1.0 里程碑](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — v1.0 发布说明和向后兼容性指南