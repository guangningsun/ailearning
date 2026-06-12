# 并行工具调用与工具流式传输

> 三个独立的天气查询串行化就是三次往返。并行运行它们，总时间缩减到最慢的一次调用。如今每个前沿提供商都能在一个回合内发出多个工具调用。收益是真实的；但管道是微妙的。本课走完两半：并行展开和流式参数重组，重点是 ID 关联陷阱。

**类型：** 构建型
**语言：** Python（标准库、线程池 + 流式传输工具）
**前置条件：** 阶段 13 · 02（函数调用深入解析）
**时间：** 约 75 分钟

## 学习目标

- 解释 `parallel_tool_calls: true` 存在的原因以及何时禁用它。
- 在并行展开期间将流式参数块关联到正确的工具调用 ID。
- 将部分 `arguments` 字符串重组为完整 JSON，不提前解析。
- 运行一个三城市天气基准测试，展示串行与并行的延迟差异。

## 问题

没有并行调用，回答"班加罗尔、东京和苏黎世的天气如何"的智能体会这样做：

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

三次 LLM 往返，每次还要支付执行器延迟。大约是理想挂钟时间的 4 倍。

使用并行调用：

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

一次 LLM 往返。执行器时间是三个中的最大值，而不是总和。OpenAI、Anthropic 和 Gemini 的生产基准测试显示，在展开工作负载上挂钟时间减少 60% 到 70%。

代价是关联复杂性。当三个调用乱序完成时，你的结果必须携带匹配的 `tool_call_id`，以便模型能将它们对应起来。当结果流式传输时，你必须将部分参数片段组装成完整 JSON 后再执行。Gemini 3 添加了唯一 ID，以解决同一工具的两个并行调用无法区分的真实问题。

## 概念

### 启用并行

- **OpenAI。** `parallel_tool_calls: true` 默认开启。设为 `false` 强制串行。
- **Anthropic。** 通过 `disable_parallel_tool_use: false` 并行（Claude 3.5 及以上默认开启）。设为 `true` 串行。
- **Gemini。** 始终支持并行；`tool_config.function_calling_config.mode = "AUTO"` 让模型决定。

当工具具有排序依赖（`create_file` 然后 `write_file`）、一个调用的输出影响另一个的输入、或速率限制器无法处理展开时，禁用并行。

### ID 关联

模型发出的每个调用都有一个 `id`。主机返回的每个结果必须包含相同的 ID。没有这个，结果就是模糊的。

- **OpenAI。** 每个 tool-role 消息上的 `tool_call_id`。
- **Anthropic。** 每个 `tool_result` 块上的 `tool_use_id`。
- **Gemini。** 每个 `functionResponse` 上的 `id`（Gemini 3 及以上；Gemini 2 通过名称匹配，这对同名并行调用会出问题）。

### 并发运行调用

主机在各自的线程、协程或远程工作器上运行每个调用的执行器。最简单的工具使用线程池；生产环境使用 asyncio 的 `asyncio.gather` 或结构化并发。完成顺序不可预测——ID 是标识符。

一个常见 bug：按调用列表顺序而不是完成顺序回复。这通常有效，因为模型只关心 `tool_call_id`，但如果结果被丢弃或重复，乱序提交会使调试更困难。优先使用带有明确 ID 的完成顺序回复。

### 流式工具调用

当模型流式传输时，`arguments` 分块到达。三个并行调用的三个独立流在传输线上交错。你需要每个 ID 一个累加器。

按提供商的形状：

- **OpenAI。** 每个块是 `choices[0].delta.tool_calls[i].function.arguments`（部分字符串）。块携带 `index`（在调用列表中的位置）。你按 index 累加，在首次出现时读取 `id`，并在 `finish_reason = "tool_calls"` 时解析 JSON。
- **Anthropic。** 流事件是 `message_start`，然后每个块一个 `content_block_start`，类型为 `tool_use`（包含 id、name、空 input）。`content_block_delta` 事件携带 `input_json_delta` 块。`content_block_stop` 关闭每个块。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 及以上）发出带有 `functionCallId` 的块，以便调用干净地交错。在 Gemini 3 之前，流式传输一次返回一个完整调用。

### 部分 JSON 和提前解析陷阱

在 `arguments` 完成之前你无法解析它。`{"city": "Beng` 这样的部分 JSON 是无效的，会抛出异常。正确的门控是提供商的调用结束信号：OpenAI 的 `finish_reason = "tool_calls"`、Anthropic 的 `content_block_stop` 或 Gemini 的流结束事件。只有在那时才尝试 `json.loads`。更稳健的方法是使用增量 JSON 解析器，在结构完成时生成事件；OpenAI 的流式传输指南推荐这种方法，用于显示实时"思考"指示器的 UX。大括号计数作为完整性测试是不可靠的（引号内的括号或转义内容会导致误报），只应作为非正式的调试启发式使用。

### 乱序完成

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

主机回复仍然必须引用 ID：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

回复中的顺序对 OpenAI 或 Anthropic 的正确性不重要。Gemini 接受任何顺序，只要 ID 匹配。

### 基准测试：串行 vs 并行

`code/main.py` 中的工具模拟三个执行器，延迟分别为 400、600 和 800 毫秒。串行运行总时间为 1800 毫秒。并行运行时间为 max(400, 600, 800) = 800 毫秒。差异是常数而不是比例，所以节省量随工具数量增长。

现实世界的警告：并行调用会压垮下游 API。向速率受限服务发出 10 路展开会失败。阶段 13 · 17 涵盖网关级背压；重试语义计划在未来的阶段中处理。

### 流式展开挂钟时间

如果模型本身流式传输，你可以在一个调用的参数完成后立即开始执行，而不是等待所有调用最终确定。这是一个 OpenAI 记录但并非所有 SDK 都公开的优化。本课中的工具就是这样做的：当模拟流产生完整的参数对象时，主机立即启动该调用。

## 使用它

`code/main.py` 有两半。第一半使用 `concurrent.futures.ThreadPoolExecutor` 串行和并行运行三个模拟天气调用，并打印挂钟时间。第二半重放一个假的流式响应——三个并行调用的 `arguments` 块在一个流上交错——并通过 `StreamAccumulator` 按 ID 重新组装它们。无需 LLM，无需网络，只有重组逻辑。

需要注意的地方：

- 串行计时器命中 1.8 秒。并行计时器在相同的假延迟下命中 0.8 秒。
- 累加器通过按 ID 缓冲来处理乱序到达的块，只在每个调用的 JSON 完成后才解析。
- 执行器在一个 ID 的参数确定后立即启动，而不是在所有流结束后。

## 发货

本课产出 `outputs/skill-parallel-call-safety-check.md`。给定一个工具注册表，该技能审计哪些工具可以安全并行化、哪些具有排序依赖、哪些会压垮下游速率限制——返回一个带有每个工具 `parallel_safe` 标志的修订注册表。

## 练习

1. 运行 `code/main.py` 并改变模拟延迟。确认并行与串行的比例大约是 `max/sum`（真实运行会因线程调度、序列化和工具开销而略有偏差）。在什么延迟分布下并行不再重要？

2. 扩展累加器以处理"调用在流中间被取消"的情况：丢弃其缓冲区并发出 `cancelled` 事件。哪个提供商明确记录了这种情况？检查 Anthropic 的 `content_block_stop` 语义和 OpenAI 的 `finish_reason: "length"` 行为。

3. 用 `asyncio.gather` 替换线程池。对两者进行基准测试。你应该会在异步上看到小的改进，因为上下文切换成本较低，但前提是执行器做真实的 I/O。

4. 选择两个不应该并行化的工具（例如 `create_file` 然后 `write_file`）。向注册表添加 `ordering_dependency` 图，并对该图进行并行展开的门控。这是对依赖感知调度的最低限度机制，未来的智能体工程阶段会将其形式化。

5. 阅读 OpenAI 的并行函数调用部分和 Anthropic 的 `disable_parallel_tool_use` 文档。找出 Anthropic 建议禁用并行性的一个真实世界工具类型。（提示：对同一资源的重大变更。）

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| 并行工具调用 | "一次展开" | 模型在一个助手消息中发出多个工具调用 |
| `parallel_tool_calls` | "OpenAI 的标志" | 启用或禁用多调用发出 |
| `disable_parallel_tool_use` | "Anthropic 的反向" | 选择退出标志；默认启用并行 |
| 工具调用 ID | "关联句柄" | 结果消息必须回显的每个调用标识符 |
| 累加器 | "流缓冲区" | 用于部分 `arguments` 块的每个 ID 字符串缓冲区 |
| 乱序完成 | "最快的先完成" | 并行调用以不可预测的顺序完成；ID 是粘合剂 |
| 依赖图 | "排序约束" | 输出进入其他工具输入的工具；不能并行化 |
| 提前解析陷阱 | "JSON.parse 爆炸" | 尝试解析不完整的 `arguments` 字符串 |
| `streamFunctionCallArguments` | "Gemini 3 功能" | 带有每个调用唯一 ID 的流式参数块 |
| 完成顺序回复 | "不要等待所有" | 结果到达时立即回复，按 ID 键控 |

## 进一步阅读

- [OpenAI — 并行函数调用](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — 默认行为和选择退出标志
- [Anthropic — 工具使用：实现工具使用](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use` 和结果批处理
- [Google — Gemini 函数调用并行部分](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 3 的 ID 关联并行调用
- [OpenAI — 带工具的流式响应](https://platform.openai.com/docs/api-reference/responses-streaming) — OpenAI 流的块参数重组
- [Anthropic — 流式消息](https://docs.anthropic.com/en/api/messages-streaming) — 带 `input_json_delta` 的 `content_block_delta`