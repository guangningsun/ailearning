# 结构化输出 — JSON Schema、Pydantic、Zod、约束解码

> "客气地请求模型返回 JSON"失败率在 5% 到 15% 之间，即便在顶级模型上也是如此。结构化输出通过约束解码弥合了这一差距：模型在字面上被阻止输出任何违反 schema 的 token。OpenAI 的严格模式、Anthropic 的 schema 类型化工具使用、Gemini 的 `responseSchema`、Pydantic AI 的 `output_type`，以及 Zod 的 `.parse`，是同一思想的五种表现形式。本节课构建 schema 验证器和严格模式合约，学习者将在每一个生产级提取管道中使用它们。

**类型：** 构建
**语言：** Python（标准库，JSON Schema 2020-12 子集）
**前置条件：** 阶段 13 · 02（函数调用深度解析）
**时间：** 约 75 分钟

## 学习目标

- 使用正确的约束条件（enum、min/max、required、pattern）为目标提取编写 JSON Schema 2020-12
- 解释为什么严格模式和约束解码与"生成后验证"提供不同的保证
- 区分三种失败模式：解析错误、schema 违规、模型拒绝
- 交付一个包含类型化修复和类型化拒绝处理的提取管道

## 问题

一个读取采购订单邮件的智能体需要将自由文本转换为 `{customer, line_items, total_usd}`。三种方法。

**方法一：提示生成 JSON。** "以 JSON 格式回复，包含 customer、line_items、total_usd 字段。" 在顶级模型上成功率 85% 到 95%。有六种失败方式：缺少大括号、尾随逗号、类型错误、幻觉字段、在 token 限制处截断、泄露散文如"以下是您的 JSON："。

**方法二：生成后验证。** 自由生成，解析，根据 schema 验证，失败时重试。可靠但昂贵——每次重试都要付费，而且截断 bug 每次发生都会多耗费一个轮次。

**方法三：约束解码。** 提供商在解码时强制执行 schema。无有效的 token 被从采样分布中屏蔽。输出保证能解析、保证能验证。失败归结为一种模式：拒绝（模型判断输入不符合 schema）。

2026 年的每个顶级提供商都提供第三种方法。

- **OpenAI。** `response_format: {type: "json_schema", strict: true}` 加上模型 decline 时的 `refusal` 响应字段。
- **Anthropic。** 对 `tool_use` 输入进行 schema 强制；`stop_reason: "refusal"` 不存在，但没有工具调用的 `end_turn` 就是信号。
- **Gemini。** 请求级别的 `responseSchema`；2026 年 Gemini 为选定的类型提供 token 级语法约束。
- **Pydantic AI。** `output_type=InvoiceModel` 发出类型化为 `InvoiceModel` 的结构化 `RunResult`。
- **Zod（TypeScript）。** 运行时解析器，根据 Zod schema 验证提供商输出；与 OpenAI 的 `beta.chat.completions.parse` 配对使用。

共同点：一次声明 schema，端到端强制执行。

## 概念

### JSON Schema 2020-12 —— 通用语言

每个提供商都接受 JSON Schema 2020-12。你最常用的结构：

- `type`：对象、数组、字符串、数字、整数、布尔值、空值之一
- `properties`：字段名到子 schema 的映射
- `required`：必须出现的字段名列表
- `enum`：允许值的闭集合
- `minimum` / `maximum`（数字）、`minLength` / `maxLength` / `pattern`（字符串）
- `items`：应用于每个数组元素的子 schema
- `additionalProperties`：`false` 禁止额外字段（默认因模式而异）

OpenAI 严格模式增加了三个要求：每个属性必须在 `required` 中列出，到处都要 `additionalProperties: false`，且没有未解决的 `$ref`。违反这些，API 在请求时返回 400。

### Pydantic，Python 绑定

Pydantic v2 通过 `model_json_schema()` 从数据类形状的模型生成 JSON Schema。Pydantic AI 将其包装，使你只需写：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

智能体框架将 schema 转换为 OpenAI 严格模式、Anthropic `input_schema` 或 Gemini `responseSchema` 在边缘。模型的输出以类型化 `Invoice` 实例返回。验证错误抛出带类型化错误路径的 `ValidationError`。

### Zod，TypeScript 绑定

Zod（`z.object({customer: z.string(), ...})`）是 TS 等价物。OpenAI 的 Node SDK 暴露 `zodResponseFormat(Invoice)`，转换为 API 的 JSON Schema payload。

### 拒绝

严格模式不能强迫模型回答。如果输入不符合 schema（"邮件是一首诗，不是发票"），模型发出包含原因的 `refusal` 字段。你的代码必须将其作为一等公民结果处理，而不是失败。拒绝也作为安全信号有用：模型被要求从受保护内容邮件中提取信用卡号码时，返回带有安全原因附加的拒绝。

### 开放权重中的约束解码

开放权重实现使用三种技术。

1. **基于语法的解码**（`outlines`、`guidance`、`lm-format-enforcer`）：从 schema 构建确定性有限自动机；在每一步，屏蔽违反 FSM 的 token 的 logits。
2. **带 JSON 解析器的 Logit 屏蔽**：与模型同步运行流式 JSON 解析器；在每一步，计算有效下一 token 集合。
3. **带验证器的投机解码**：廉价草稿模型提议 token，验证器强制执行 schema。

商业提供商在幕后选择其中之一。2026 年的最新技术比纯生成短结构化输出更快，长结构化输出速度大致相同。

### 三种失败模式

1. **解析错误。** 输出不是有效 JSON。严格模式下不会发生。非严格提供商上仍可能发生。
2. **Schema 违规。** 输出被解析但违反 schema。严格模式下不会发生。在非严格模式下很常见。
3. **拒绝。** 模型 decline。必须作为类型化结果处理。

### 重试策略

当你在严格模式之外（Anthropic 工具使用、非严格 OpenAI、旧版 Gemini）时，恢复模式是：

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

一次重试通常足够。三次重试捕获弱模型 flakes。超过三次是 schema 不良的信号：模型无法为某些输入满足它，需要修复提示或 schema。

### 小模型支持

约束解码适用于小模型。配备语法强制的 3B 参数开放模型在结构化任务上优于 70B 参数模型的原始提示。这是你应该关注生产中结构化输出的主要原因：它将可靠性与模型大小解耦。

## 使用它

`code/main.py` 提供了一个使用标准库的最简 JSON Schema 2020-12 验证器（types、required、enum、min/max、pattern、items、additionalProperties）。它包装了一个 `Invoice` schema 并通过验证器运行一个假 LLM 输出，演示解析错误、schema 违规和拒绝路径。在生产中，将假输出替换为任何提供商的真实响应。

要看的内容：

- 验证器返回带路径和消息的类型化 `[ValidationError]` 列表。这是你想要暴露给重试提示的形状。
- 拒绝分支**不重试**。它记录并返回类型化拒绝。第 14 阶段 · 09 将拒绝作为安全信号使用。
- `additionalProperties: false` 检查在对抗性测试输入上触发，展示为什么严格模式关闭了幻觉字段的大门。

## 交付它

本节课产出 `outputs/skill-structured-output-designer.md`。给定一个自由文本提取目标（发票、支持工单、简历等），该 skill 产生一个严格模式兼容的 JSON Schema 2020-12 和一个镜像它的 Pydantic 模型，并带有类型化拒绝和重试处理存根。

## 练习

1. 运行 `code/main.py`。添加第四个测试用例，其 `total_usd` 为负数。确认验证器使用 `minimum` 约束路径拒绝它。

2. 扩展验证器以支持带判别器的 `oneOf`。常见情况：`line_item` 要么是产品要么是服务，通过 `kind` 标记。严格模式在这里有微妙规则；查看 OpenAI 的结构化输出指南。

3. 将相同的 Invoice schema 写为 Pydantic BaseModel，并将 `model_json_schema()` 输出与你手写的 schema 比较。找出 Pydantic 默认设置的一个字段，而手写版本省略了它。

4. 测量拒绝率。构造十个不可提取的输入（一首歌词、数学证明、空白邮件），并将它们通过严格模式的真实提供商运行。计算拒绝与幻觉输出的比例。这是你拒绝感知重试的基础事实。

5. 从头到尾阅读 OpenAI 的结构化输出指南。找出它在严格模式中明确禁止的一个构造，而普通 JSON Schema 允许。然后设计一个非必要地使用该禁止构造的 schema，并重构为严格兼容。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| JSON Schema 2020-12 | "schema 规范" | 每个现代提供商都说的 IETF 草案 schema 方言 |
| 严格模式 | "保证的 schema" | 通过约束解码强制执行 schema 的 OpenAI 标志 |
| 约束解码 | "Logit 屏蔽" | 解码时强制执行，屏蔽无效下一 token |
| 拒绝 | "模型 decline" | 当输入不符合 schema 时的类型化结果 |
| 解析错误 | "无效 JSON" | 输出未解析为 JSON；严格模式下不可能 |
| Schema 违规 | "错误形状" | 被解析但违反类型 / required / enum / 范围 |
| `additionalProperties: false` | "不允许额外项" | 禁止未知字段；OpenAI 严格模式中必需 |
| Pydantic BaseModel | "类型化输出" | 发出并验证 JSON Schema 的 Python 类 |
| Zod schema | "TypeScript 输出类型" | 用于提供商输出验证的 TS 运行时 schema |
| 语法强制 | "开放权重约束解码" | 基于 FSM 的 logit 屏蔽，如 outlines / guidance |

## 进一步阅读

- [OpenAI — 结构化输出](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式、拒绝和 schema 要求
- [OpenAI — 引入结构化输出](https://openai.com/index/introducing-structured-outputs-in-the-api/) — 2024 年 8 月发布帖子，解释解码保证
- [Pydantic AI — 输出](https://ai.pydantic.dev/output/) — 类型化 output_type 绑定，序列化到每个提供商
- [JSON Schema — 2020-12 发布说明](https://json-schema.org/draft/2020-12/release-notes) — 规范文档
- [Microsoft — Azure OpenAI 中的结构化输出](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — 企业部署说明和严格模式注意事项