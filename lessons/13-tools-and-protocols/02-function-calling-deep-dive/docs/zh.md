# 函数调用深入解析 — OpenAI、Anthropic、Gemini

> 2024 年，三大前沿提供商在同一个工具调用循环上收敛，然后在其他所有方面分道扬镳。OpenAI 使用 `tools` 和 `tool_calls`。Anthropic 使用 `tool_use` 和 `tool_result` 块。Gemini 使用 `functionDeclarations` 和唯一 ID 关联。本课将三者并排对比，这样在一套提供商上编写的代码在移植到另一套时不会出错。

**类型：** 构建型
**语言：** Python（标准库、schema 转换器）
**前置条件：** 阶段 13 · 01（工具接口）
**时间：** 约 75 分钟

## 学习目标

- 说出 OpenAI、Anthropic 和 Gemini 函数调用载荷之间的三个形状差异（声明、调用、结果）。
- 将一个工具声明翻译为三种提供商格式，并预测严格模式约束的不同之处。
- 在每个提供商中使用 `tool_choice` 来强制、禁止或自动选择工具调用。
- 了解每个提供商的硬性限制（工具数量、schema 深度、参数长度），以及超出限制时各自的错误签名。

## 问题

函数调用请求的形状因提供商而异。以下是 2026 年生产堆栈中的三个具体例子：

**OpenAI 聊天补全 / Responses API。** 你传入 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型的响应包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是一个需要解析的 JSON 字符串。严格模式（`strict: true`）通过约束解码来强制执行 schema 合规。

**Anthropic Messages API。** 你传入 `tools: [{name, description, input_schema}]`。响应以 `content: [{type: "text"}, {type: "tool_use", id, name, input}]` 的形式返回。`input` 已经是解析后的对象（不是字符串）。你通过包含 `{type: "tool_result", tool_use_id, content}` 块的新 `user` 消息来回复。

**Google Gemini API。** 你传入 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在 `functionDeclarations` 下）。响应以 `candidates[0].content.parts: [{functionCall: {name, args, id}}]` 的形式到达，其中 `id` 在 Gemini 3 及以上版本中是唯一的，用于并行调用关联。你通过 `{functionResponse: {name, id, response}}` 来回复。

相同的循环。不同的字段名、不同的嵌套、不同的字符串与对象约定、不同的关联机制。一个在 OpenAI 上编写天气智能体的团队需要两天时间移植到 Anthropic，再花一天移植到 Gemini——纯粹是管道工作。

本课构建一个将三种格式统一为一种规范工具声明的转换器，并在边缘路由。阶段 13 · 17 将同一模式泛化为 LLM 网关。

## 概念

### 通用结构

每个提供商都需要五样东西：

1. **工具列表。** 每个工具的名称、描述和输入 schema。
2. **工具选择。** 强制使用特定工具、禁止工具，或让模型决定。
3. **调用发出。** 命名工具和参数的結構化输出。
4. **调用 ID。** 将响应关联到正确的调用（对并行很重要）。
5. **结果注入。** 将结果绑定回调用的消息或块。

### 逐字段的形状差异

| 方面 | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| 声明信封 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema 字段 | `parameters` | `input_schema` | `parameters` |
| 响应容器 | 助手消息上的 `tool_calls[]` | 类型为 `tool_use` 的 `content[]` 块 | 类型为 `functionCall` 的 `parts[]` |
| 参数类型 | 字符串化的 JSON | 已解析的对象 | 已解析的对象 |
| ID 格式 | `call_...`（OpenAI 生成） | `toolu_...`（Anthropic 生成） | UUID（Gemini 3+） |
| 结果块 | 角色 `tool`，`tool_call_id` | 带 `tool_result` 的 `user`，`tool_use_id` | 带匹配 `id` 的 `functionResponse` |
| 强制工具 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁止工具 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 严格 schema | `strict: true` | schema 即 schema（始终强制执行） | 请求级别的 `responseSchema` |

### 你会实际遇到的限制

- **OpenAI。** 每次请求 128 个工具。Schema 深度 5。参数字符串 <= 8192 字节。严格模式要求无 `$ref`、无重叠的 `oneOf`/`anyOf`/`allOf`、每个属性都在 `required` 中列出。
- **Anthropic。** 每次请求 64 个工具。Schema 深度实际无限制，但实际限制为 10。无严格模式标志；schema 是一份契约，模型往往会遵守。
- **Gemini。** 每次请求 64 个函数。Schema 类型是 OpenAPI 3.0 子集（与 JSON Schema 2020-12 略有差异）。自 Gemini 3 起并行调用使用唯一 ID。

### `tool_choice` 行为

三种模式，每个提供商都支持，但名称不同。

- **Auto。** 模型选择工具或文本。默认值。
- **Required / Any。** 模型必须至少调用一个工具。
- **None。** 模型不得调用工具。

再加上每个提供商独有的一种模式：

- **OpenAI。** 按名称强制使用特定工具。
- **Anthropic。** 按名称强制使用特定工具；`disable_parallel_tool_use` 标志区分单调用与多调用。
- **Gemini。** `mode: "VALIDATED"` 将每个响应路由通过 schema 验证器，无论模型意图如何。

### 并行调用

OpenAI 的 `parallel_tool_calls: true`（默认）在一个助手消息中发出多个调用。你全部运行它们，然后用包含每个 `tool_call_id` 对应条目的批量 tool-role 消息回复。Anthropic 历史上是单调用；`disable_parallel_tool_use: false`（自 Claude 3.5 起默认）启用多调用。Gemini 2 允许并行调用但没有稳定的 ID；Gemini 3 添加了 UUID，因此乱序响应可以干净地关联。

### 流式传输

三者都支持流式工具调用。线路格式不同：

- **OpenAI。** `tool_calls[i].function.arguments` 的增量增量块。你累积直到 `finish_reason: "tool_calls"`。
- **Anthropic。** 块开始 / 块增量 / 块停止事件。`input_json_delta` 块携带部分参数。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 新增）发出带有 `functionCallId` 的块，以便多个并行调用可以交错。

阶段 13 · 03 深入探讨并行 + 流式重组。本课专注于声明和单调用形状。

### 错误与修复

无效参数错误的表现也各不相同。

- **OpenAI（非严格）。** 模型返回 `arguments: "{bad json}"`，你的 JSON 解析失败，注入错误消息并重新调用。
- **OpenAI（严格）。** 验证在解码期间发生；无效 JSON 不可能，但可能出现 `refusal`。
- **Anthropic。** `input` 可能包含意外字段；schema 是建议性的。在服务器端验证。
- **Gemini。** OpenAPI 3.0 怪癖：对象字段上的 `enum` 被静默忽略；自己验证。

### 转换器模式

你的代码中的规范工具声明是这样的（你选择形状）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小函数将其翻译为三种提供商声明 JSON。`code/main.py` 中的工具正是这样做的，然后通过每种提供商响应形状循环往返一个假工具调用。无需网络——本课教的是形状，不是 HTTP。

生产团队将此转换器包装在 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）中。阶段 13 · 17 发货一个网关，在任何三种提供商前面暴露一个 OpenAI 形状的 API。

## 使用它

`code/main.py` 定义了一个规范的 `Tool` 数据类和三个发出 OpenAI、Anthropic 和 Gemini 声明 JSON 的转换器。然后它解析每个形状的手工制作提供商响应到相同的规范调用对象，演示了语义在底层是相同的。运行它并并排对比三种声明。

需要注意的地方：

- 三种声明块只在信封和字段名上不同。
- 三种响应块在调用所在位置不同（顶级 `tool_calls`、`content[]` 块、`parts[]` 条目）。
- 一个 `canonical_call()` 函数从所有三种响应形状中提取 `{id, name, args}`。

## 发货

本课产出 `outputs/skill-provider-portability-audit.md`。给定针对一个提供商的函数调用集成，该技能生成可移植性审计：它依赖哪些提供商限制、哪些字段需要重命名、移植到其他提供商时会破坏什么。

## 练习

1. 运行 `code/main.py` 并验证三个提供商声明 JSON 都序列化相同的底层 `Tool` 对象。修改规范工具以添加一个 enum 参数，并确认只有 Gemini 转换器需要处理 OpenAPI 怪癖。

2. 为每个提供商添加 `ListToolsResponse` 解析器，提取模型在 `list_tools` 或发现调用后返回的工具列表。OpenAI 原生没有这个；注意这个不对称性。

3. 实现 `tool_choice` 转换：将规范的 `ToolChoice(mode="force", tool_name="x")` 映射到所有三种提供商形状。然后映射 `mode="any"` 和 `mode="none"`。检查本课的差异表。

4. 选择三个提供商之一，通读其函数调用指南。找出其 schema 规范中的一个字段，而其他两个不支持。候选：OpenAI `strict`、Anthropic `disable_parallel_tool_use`、Gemini `function_calling_config.allowed_function_names`。

5. 编写一个测试向量：一个参数违反声明 schema 的工具调用。通过每个提供商的验证器运行它（阶段 01 中的标准库验证器可以作为代理），记录触发了哪些错误。记录你会在生产中用于严格性的提供商。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| Function calling | "工具使用" | 用于结构化工具调用发出的提供商级 API |
| Tool declaration | "工具规范" | 名称 + 描述 + JSON Schema 输入载荷 |
| `tool_choice` | "强制 / 禁止" | Auto / required / none / specific-name 模式 |
| Strict mode | "Schema 强制执行" | 约束解码匹配 schema 的 OpenAI 标志 |
| `tool_use` 块 | "Anthropic 的调用形状" | 带 id、name、input 的内联内容块 |
| `functionCall` 部分 | "Gemini 的调用形状" | 包含 name、args 和 id 的 `parts[]` 条目 |
| 参数即字符串 | "字符串化的 JSON" | OpenAI 将 args 作为 JSON 字符串返回，而不是对象 |
| 并行工具调用 | "一次展开" | 在一个助手消息中的多个工具调用 |
| Refusal | "模型拒绝" | 严格模式独有的拒绝块，而不是调用 |
| OpenAPI 3.0 子集 | "Gemini schema 怪癖" | Gemini 使用类似 JSON Schema 的方言，但有细微差异 |

## 进一步阅读

- [OpenAI — 函数调用指南](https://platform.openai.com/docs/guides/function-calling) — 包括严格模式和并行调用的规范参考
- [Anthropic — 工具使用概述](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` 和 `tool_result` 块语义
- [Google — Gemini 函数调用](https://ai.google.dev/gemini-api/docs/function-calling) — 并行调用、唯一 ID 和 OpenAPI 子集
- [Vertex AI — 函数调用参考](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini 的企业表面
- [OpenAI — 结构化输出](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式 schema 强制执行详情