# FIPA-ACL 与言语行为遗产

> 在 MCP 之前，在 A2A 之前，就已经有了 FIPA-ACL。2000 年，IEEE 智能物理代理基金会（Foundation for Intelligent Physical Agents）正式批准了一种代理通信语言，包含二十个言语行为、两种内容语言，以及一组交互协议——合同网、订阅/通知、请求-何时。它从行业中消失是因为本体论开销对于 Web 来说过于沉重，但 LLM 多代理系统的复兴正在以更轻量的方式重新实现同样的想法：JSON 契约取代言语行为，自然语言取代本体论。本节认真解读 FIPA-ACL，这样你就能看清 2026 年的哪些协议决策是重新发明，哪些是真正的新生事物，以及当前这波浪潮将会在哪里重新发现 2000 年代已经解决过的问题。

**类型：** 学习型
**语言：** Python（标准库）
**前置条件：** 阶段 16 · 01（为什么需要多代理）
**时间：** 约 60 分钟

## 问题

2026 年的代理协议格局非常热闹：MCP 用于工具，A2A 用于代理，ACP 用于企业审计，ANP 用于去中心化信任，NLIP 用于自然语言内容，再加上 CA-MCP 和数十个研究提案。每个规范都宣称自己是基础性的。

客观地说，它们大多数都在重新发现一个非常具体的、二十年前的决策树。Austin（1962）和 Searle（1969）的言语行为理论给了我们"话语即行为"。KQML（1993）将其转化为有线协议。FIPA-ACL（2000 年正式批准）产生了参考标准化：二十个言语行为、内容语言 SL0/SL1、用于合同网和订阅-通知的交互协议。JADE 和 JACK 是 Java 参考平台。这项工作在 2010 年左右衰落，因为本体论开销过于沉重，而 Web 正在胜出。

当你审视 MCP 的 `tools/call`、A2A 的任务生命周期或 CA-MCP 的共享上下文存储时，你看到的是一个更柔和的、JSON 原生的 FIPA 决策重组。了解这段历史告诉你两件事：哪些新的"创新"实际上是重新发明，以及哪些旧的失败模式新规范将重新发现。

## 概念

### 言语行为，一段话讲清楚

Austin 注意到有些句子不是描述世界的——它们改变世界。"我承诺。""我请求。""我宣布。"他称这些为述行话语（performative utterances）。Searle 将其形式化为五类：断言型（assertive）、指令型（directive）、承诺型（commissive）、表达型（expressive）、宣告型（declarative）。KQML（Finin 等，1993 年）使其对软件代理可操作：一条消息是一个言语行为（动作）加上内容（动作是关于什么的）。FIPA-ACL 清理了 KQML 的漏洞，并围绕二十个言语行为进行了标准化。

### 二十个 FIPA 言语行为（部分列表）

| 言语行为 | 意图 |
|---|---|
| `inform` | "我告诉你 P 为真" |
| `request` | "我请你做 X" |
| `query-if` | "P 为真吗？" |
| `query-ref` | "X 的值是什么？" |
| `propose` | "我提议我们做 X" |
| `accept-proposal` | "我接受该提案" |
| `reject-proposal` | "我拒绝该提案" |
| `agree` | "我同意做 X" |
| `refuse` | "我拒绝做 X" |
| `confirm` | "我确认 P 为真" |
| `disconfirm` | "我否认 P" |
| `not-understood` | "你的消息无法解析" |
| `cfp` | "就 X 征集提案" |
| `subscribe` | "当 X 变化时通知我" |
| `cancel` | "取消正在进行的 X" |
| `failure` | "我尝试 X 但失败了" |

完整列表在 `fipa00037.pdf`（FIPA ACL 消息结构）中。重点不是记住它——重点是这每一个都对应着一个 LLM 协议最终会重新添加的原始操作。

### 规范的 FIPA-ACL 消息

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

七个字段携带协议信封；一个字段（`content`）携带有效载荷。其余字段正是你每次将重试、线程化和本体论附加到 JSON 协议时重新发明的东西。

### 两个遗留平台

**JADE**（Java Agent Development framework，1999–2020 年代）是最常用的 FIPA 兼容运行时。代理扩展一个基类，交换 ACL 消息，在容器内运行，并使用"行为"进行协调。交互协议库随附了合同网、订阅-通知、请求-何时和提案-接受。

**JACK**（Agent Oriented Software，商业软件）在 FIPA 消息之上强调 BDI（信念-欲望-意图）推理。更形式化，采用较少。

两者都在 Web 栈吞噬多代理用例后衰落。MCP 和 A2A 是 2026 年的运行时"容器"。

### FIPA 为何衰落

- **本体论开销。** FIPA 需要一个共享本体论来解析 `content`。就本体论达成一致是一个长达数年的标准制定过程。Web 只是使用 HTTP + JSON。
- **无人使用的形式语义。** SL（语义语言）给出了严格的真值条件，但大多数生产系统使用自由格式内容并忽略形式化。
- **工具锁定。** JADE 仅为 Java；JACK 是商业软件。多语言团队绕过了两者。
- **互联网赢得了技术栈。** REST，然后是 JSON-RPC，然后是 gRPC 取代了 ACL 的传输层。

### LLM 复兴是 FIPA-Lite

将 FIPA 的 `request` 与 MCP 的 `tools/call` 进行比较：

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

相同的信封，不同的语法。两者都携带：谁、向谁、意图、有效载荷、关联 ID。两者都不是另一者的革命——它们是同一设计的不同权衡。

刘等人（2025 年）的调查（"A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP"，arXiv:2505.02279）明确阐述了这一谱系：MCP 对应工具使用言语行为，A2A 对应代理对等言语行为，ACP 对应审计追踪言语行为，ANP 对应去中心化身份扩展。新规范是具有 JSON 语法和更宽松语义的 ACL 后代。

### 权衡，直说

**FIPA 给你而现代规范放弃的：**

- 形式语义——你可以证明 `inform` 意味着发送者相信内容。
- 言语行为的规范目录——你不必重新争论"我们应该有一个 `cancel` 吗？"
- 数十年的交互协议模式——合同网、订阅-通知、提案-接受——具有已知的正确性属性。

**现代规范给你而 FIPA 没有的：**

- 与每个现代工具兼容的 JSON 原生有效载荷。
- LLMs 无需手写本体论就能解释的自然语言内容。
- Web 栈传输（HTTP、SSE、WebSocket）。
- 通过自描述文档进行能力发现（MCP `listTools`、A2A Agent Card）。

更宽松的意图语义以简化实现。这就是确切的权衡。

### 值得移植的交互协议

FIPA 附带了约 15 个交互协议。三个值得带入 LLM 多代理系统：

1. **合同网协议（CNP）。** 管理者发布 `cfp`（征集提案）；投标者用 `propose` 回应；管理者接受/拒绝。这是典型的任务市场模式（阶段 16 · 16 谈判）。
2. **订阅/通知。** 订阅者发送 `subscribe`；发布者每当主题变化时发送 `inform`。这是 2026 年的每个事件总线。
3. **请求-何时。** "当条件 Y 成立时做 X。" 带前置条件的延迟操作。2026 年的类比是持久化工作流引擎中的延迟任务（阶段 16 · 22 生产扩展）。

每一个都能干净地映射到现代消息队列、HTTP + 轮询或 SSE 流。

### 当你放弃本体论时什么会出问题

没有共享本体论，代理从自然语言内容中推断含义。2026 年记录在案的失败模式是**语义漂移**：两个代理用同一个词（`"customer"`）表示略有不同的概念，接收方代理根据错误的解释行动，没有模式验证器捕获它。FIPA 的本体论要求会在解析时拒绝该消息。

在不走向完整本体论的情况下缓解：

- JSON Schema 对 `content`——在传输层拒绝结构错误。
- 类型化产物（A2A）——拒绝错误的模态。
- 信封中明确的言语行为——即使内容是自然语言，意图也明确无误。

### 2026 年规范，映射到言语行为遗产

| 现代规范 | FIPA 类比 | 保留什么 | 放弃什么 |
|---|---|---|---|
| MCP `tools/call` | `request` | 明确意图、关联 ID | 形式语义、本体论 |
| MCP `resources/read` | `query-ref` | 明确意图、关联 ID | 形式语义 |
| A2A 任务生命周期 | 合同网 + 请求-何时 | 异步生命周期、状态转换 | 形式完备性保证 |
| A2A 流式事件 | 订阅/通知 | 异步推送 | 类型化谓词订阅 |
| CA-MCP 共享上下文 | 黑板（Hayes-Roth 1985） | 多写者共享内存 | 逻辑一致性模型 |
| NLIP | 自然语言内容 | LLM 原生 | 模式 |

从表头往下读，模式是：保留结构原语，放弃形式化，让 LLMs 用模糊性来掩盖。

## 动手实现

`code/main.py` 实现了一个纯标准库的 FIPA-ACL 转换器。它编码和解码规范的 ACL 信封，并展示每个 MCP / A2A 消息形状如何归约为相同的七个字段。演示：

- 将五个 MCP 风格和 A2A 风格的消息编码为 FIPA-ACL。
- 将 FIPA-ACL 解码回现代等价形式。
- 使用 `cfp`、`propose`、`accept-proposal`、`reject-proposal` 在一个管理器和三个投标者之间运行一个玩具合同网协商。

运行：

```
python3 code/main.py
```

输出是一个并排跟踪，显示每个现代消息以其 2026 JSON 形式和 FIPA-ACL 形式，然后是合同网投标的往返。相同的协议原语在往返中存活；只有语法不同。

## 实际使用

`outputs/skill-fipa-mapper.md` 是一个技能，读取任何代理协议规范并生成 FIPA-ACL 映射。在采用新协议之前使用它来回答："这真的是新东西，还是带 JSON 语法的 `inform`？"

## 交付物

不要把 FIPA-ACL 带回来。把它的检查清单带回来：

- 每条消息的意图原语（言语行为）是什么？
- 是否有用于请求-响应和取消的关联 ID？
- 是否有明确的内容语言（JSON-RPC、纯文本、结构化类型化产物）？
- 交互协议是否是一等的，还是你从零开始重新实现合同网？
- 当两个代理对内容含义产生分歧时会发生什么（语义漂移）？

在任何新协议投入生产之前，为它记录这五个问题。

## 练习

1. 运行 `code/main.py`。观察往返编码。识别哪个 FIPA 言语行为对应于 `tools/call`、`resources/read` 和 A2A 任务创建。
2. 用 `cancel` 言语行为扩展合同网演示，让管理者在投标中途撤回任务。`cancel` 解决了重试本身无法解决的什么失败案例？
3. 阅读 FIPA ACL 消息结构（http://www.fipa.org/specs/fipa00037/）第 4.1–4.3 节。选择一个本节未涵盖的言语行为，描述其现代 JSON-RPC 类比。
4. 阅读刘等人，arXiv:2505.02279。对于 MCP、A2A、ACP、ANP，列出它们保留和放弃的 FIPA 言语行为家族。
5. 为你系统中 `request` 言语行为的 `content` 字段设计一个最小的 JSON-Schema。这个 Schema 给了你纯自然语言没有的什么，它又付出了什么代价？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| 言语行为 (Speech act) | "做某事的话语" | Austin/Searle：话语即行为。ACL 的理论父级。 |
| FIPA | "那个旧的 XML 东西" | IEEE 智能物理代理基金会。2000 年标准化了 ACL。 |
| ACL | "代理通信语言" | FIPA 的信封格式：言语行为 + 内容 + 元数据。 |
| 言语行为 (Performative) | "动词" | 消息的意图类别：`inform`、`request`、`propose`、`cfp` 等。 |
| KQML | "FIPA 的前身" | 知识查询与操作语言（1993）。更简单，范围更窄。 |
| 本体论 (Ontology) | "共享词汇" | 内容语言所谈论概念的正式定义。 |
| SL0 / SL1 | "FIPA 内容语言" | 语义语言 0 级和 1 级——形式内容语言家族。 |
| 合同网 (Contract Net) | "任务市场" | 管理者发布 cfp；投标者提案；管理者接受。典型的交互协议。 |
| 交互协议 (Interaction protocol) | "消息模式" | 具有已知正确性的言语行为序列：请求-何时、订阅-通知等。 |

## 延伸阅读

- [刘等人——代理互操作性协议调查：MCP、ACP、A2A、ANP](https://arxiv.org/html/2505.02279v1)——将现代规范与 FIPA 遗产联系起来的 2025 年权威调查
- [FIPA ACL 消息结构规范（fipa00037）](http://www.fipa.org/specs/fipa00037/)——2000 年批准的信封格式
- [FIPA 通信行为库规范（fipa00037）](http://www.fipa.org/specs/fipa00037/)——完整的言语行为目录
- [MCP 规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25)——现代工具使用equivalent of `request`/`query-ref`
- [A2A 规范](https://a2a-protocol.org/latest/specification/)——合同网和订阅-通知的现代代理对等类比