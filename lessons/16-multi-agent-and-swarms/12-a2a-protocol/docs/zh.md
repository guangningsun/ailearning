# A2A —— Agent 间协议

> Google 于 2025 年 4 月宣布 A2A；到 2026 年 4 月规范位于 https://a2a-protocol.org/latest/specification/，150+ 个组织为其背书。A2A 是 MCP（第 13 课）的横向补充：MCP 是纵向的（Agent ↔ 工具），A2A 是对等的（Agent ↔ Agent）。它定义了 Agent Card（发现）、带产物（文本、结构化数据、视频）的任务、不透明任务生命周期和认证。生产级多 Agent 系统越来越多地将 MCP 与 A2A 配对使用。Google Cloud 在 2025-2026 年间将 A2A 支持集成到 Vertex AI Agent Builder 中。

**类型：** 学习 + 构建
**语言：** Python（标准库、`http.server`、`json`）
**前置条件：** 阶段 16 · 04（原始模型）
**时间：** 约 75 分钟

## 问题

你的 Agent 需要调用另一个系统上的另一个 Agent。怎么做？你可以暴露一个 HTTP 端点，定义一个定制的 JSON schema，希望对方能理解它。每个 Agent 对都成为一个定制集成。

A2A 就是那个调用的通用有线协议。标准发现、标准任务模型、标准传输、标准产物。像 HTTP+REST 一样，但 Agent 是一等公民。

## 概念

### 四个要素

**Agent Card。** 位于 `/.well-known/agent.json` 的 JSON 文档，描述 Agent：名称、技能、端点、支持的形式、认证要求。发现通过读取卡片来实现。

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**任务。** 工作单元。具有生命周期的异步有状态对象：`submitted → working → completed / failed / canceled`。客户端发送任务，轮询或订阅更新。

**产物。** 任务产生的结果类型。文本、结构化 JSON、图像、视频、音频。产物是类型化的，因此不同形式是一等公民。

**不透明生命周期。** A2A 不规定远程 Agent *如何* 解决任务。客户端看到状态转换和产物；实现可以自由使用任何框架。

### MCP/A2A 分工

- **MCP**（第 13 课）：Agent ↔ 工具。Agent 通过 JSON-RPC 读取/写入工具服务器。默认无状态。
- **A2A**：Agent ↔ Agent。对等协议；双方都是具有自己推理的 Agent。

生产级多 Agent 系统两者都使用。A2A 对等方在其端调用 MCP 工具。这种分离保持了两个关注点的清晰。

### 发现流程

```
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

或者使用流式传输：通过 SSE 订阅 `/tasks/{id}/events` 进行推送更新。

### 认证

A2A 支持三种常见模式：

- **Bearer 令牌** —— OAuth2 或不透明令牌。
- **mTLS** —— 双向 TLS；组织之间相互证明身份。
- **签名请求** —— 有效负载上的 HMAC。

认证在 Agent Card 中声明；客户端发现并遵守。

### 2026 年 4 月已有 150+ 个组织

企业采用推动了 A2A 的规模化。头条是：A2A 成为了企业 Agent 系统跨越信任边界的方式。Google Cloud 交付了 Vertex AI Agent Builder A2A 支持；Microsoft Agent Framework 支持它；大多数主要框架（LangGraph、CrewAI、AutoGen）都附带 A2A 适配器。

### A2A 擅长的领域

- **跨组织调用。** A 公司 Agent 调用 B 公司 Agent。没有 A2A，每一对都是定制契约。
- **异构框架。** LangGraph Agent 调用 CrewAI Agent 调用自定义 Python Agent。A2A 使其标准化。
- **类型化产物。** 视频结果、结构化 JSON、音频 —— 都是一等公民。
- **长期运行任务。** 不透明生命周期 + 轮询使数小时长的任务变得简单。

### A2A 挣扎的地方

- **延迟敏感的微调用。** A2A 的生命周期是异步的。亚毫秒级 Agent 间调用不适合；使用直接 RPC。
- **紧耦合的进程内 Agent。** 如果两个 Agent 在同一个 Python 进程中运行，A2A 的 HTTP 往返是杀鸡用牛刀。
- **小团队。** 规范开销是真实存在的；仅内部使用的 Agent 可能不需要这种正式性。

### A2A vs ACP、ANP、NLIP

2024-2026 年间出现了几个相关规范：

- **ACP**（IBM/Linux Foundation）—— A2A 的前身，范围更窄。
- **ANP**（Agent Network Protocol）—— 以对等发现为中心，去中心化优先。
- **NLIP**（Ecma 自然语言交互协议，2025 年 12 月标准化）—— 自然语言内容类型。

截至 2026 年 4 月，A2A 是采用最广泛的对等协议。参见 arXiv:2505.02279（刘等人，《Agent 互操作性协议调查》）进行比较。

## 构建

`code/main.py` 使用 `http.server` 和 JSON 实现了一个最小 A2A 服务器和客户端。服务器：

- 暴露 `/.well-known/agent.json`，
- 接受 `POST /tasks`，
- 管理任务状态，
- 在 `GET /tasks/{id}` 上返回产物。

客户端：

- 获取 Agent Card，
- 提交任务，
- 轮询直到完成，
- 读取产物。

运行：

```
python3 code/main.py
```

该脚本在后台线程中启动服务器，然后针对它运行客户端。你可以看到完整流程：发现、提交、轮询、产物。

## 使用

`outputs/skill-a2a-integrator.md` 设计 A2A 集成：Agent Card 内容、任务 schema、认证选择、流式传输 vs 轮询。

## 交付

检查清单：

- **固定规范版本。** A2A 仍在发展；Agent Card 应声明协议版本。
- **幂等任务创建。** 重复提交（网络重试）应产生一个任务。
- **产物 schema。** 声明 Agent 返回的形状；消费者应进行验证。
- **速率限制 + 认证。** A2A 是面向公众的；应用标准网络安全。
- **失败任务死信。** 检查模式随时间的趋势以发现重复失败类型。

## 练习

1. 运行 `code/main.py`。确认客户端发现服务器并收到正确的产物。
2. 向服务器添加第二个技能（例如"summarize"）。更新 Agent Card。编写一个基于任务类型选择技能的客户端。
3. 实现 SSE 流式端点：`/tasks/{id}/events` 发出状态变化。客户端需要做什么不同的事情？
4. 阅读 A2A 规范（https://a2a-protocol.org/latest/specification/）。识别规范强制要求的三个此演示未实现的东西。
5. 比较 A2A（Agent Card 发现）与 MCP（服务器端通过 `listTools` 列出能力）。自我描述 Agent 与能力探测之间的权衡是什么？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| A2A | "Agent 到 Agent" | Agent 调用跨系统的其他 Agent 的对等协议。Google 2025。 |
| Agent Card | "Agent 的名片" | `/.well-known/agent.json` 上的 JSON，描述技能、端点、认证。 |
| 任务 | "工作单元" | 具有生命周期的异步有状态对象；完成时产生产物。 |
| 产物 | "结果" | 类型化输出：文本、结构化 JSON、图像、视频、音频。一等媒体。 |
| 不透明生命周期 | "如何解决是 Agent 的事" | 客户端看到状态转换；服务器自由选择框架/工具。 |
| 发现 | "找到 Agent" | `GET /.well-known/agent.json` 返回卡片。 |
| MCP vs A2A | "工具 vs 对等方" | MCP：纵向 Agent ↔ 工具。A2A：横向 Agent ↔ Agent。 |
| ACP / ANP / NLIP | "兄弟协议" | 相邻规范；A2A 是 2026 年采用最多的。 |

## 延伸阅读

- [A2A 规范](https://a2a-protocol.org/latest/specification/) — 规范文档
- [Google Developers Blog —— A2A 公告](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — 2025 年 4 月发布帖子
- [A2A GitHub 仓库](https://github.com/a2aproject/A2A) — 参考实现和 SDK
- [刘等人 —— Agent 互操作性协议调查](https://arxiv.org/html/2505.02279v1) — MCP、ACP、A2A、ANP 比较