# OpenTelemetry GenAI 语义约定

> OpenTelemetry 的 GenAI SIG（2024 年 4 月启动）定义了智能体遥测的标准模式。跨度名称、属性和内容捕获规则在各厂商间趋同，使智能体追踪在 Datadog、Grafana、Jaeger 和 Honeycomb 中具有相同的含义。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置条件：** 阶段 14 · 13（LangGraph），阶段 14 · 24（可观测性平台）
**时间：** 约 60 分钟

## 学习目标

- 说出 GenAI 跨度类别：model/client、agent、tool。
- 区分 `invoke_agent` CLIENT 与 INTERNAL 跨度及其各自的适用场景。
- 列出顶级 GenAI 属性：provider name、request model、data-source ID。
- 解释内容捕获契约：opt-in、`OTEL_SEMCONV_STABILITY_OPT_IN`、外部引用建议。

## 问题

每个厂商发明自己的跨度名称。运维团队最终要为每个框架构建仪表板。OpenTelemetry 的 GenAI SIG 通过定义整个生态系统共同瞄准的一个标准来解决这个问题。

## 概念

### 跨度类别

1. **Model / client 跨度。** 覆盖原始 LLM 调用。由提供商 SDK（Anthropic、OpenAI、Bedrock）和框架模型适配器发出。
2. **Agent 跨度。** `create_agent`（当智能体被构造时）和 `invoke_agent`（当智能体运行时）。
3. **Tool 跨度。** 每次工具调用一个；通过父子关系连接到 agent 跨度。

### Agent 跨度命名

- 跨度名称：若命名则为 `invoke_agent {gen_ai.agent.name}`；否则回退到 `invoke_agent`。
- 跨度类型：
  - **CLIENT** — 用于远程智能体服务（OpenAI Assistants API、Bedrock Agents）。
  - **INTERNAL** — 用于进程内智能体框架（LangChain、CrewAI、本地 ReAct）。

### 关键属性

- `gen_ai.provider.name` — `anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model` — 模型 ID。
- `gen_ai.response.model` — 解析后的模型（可能因路由而与请求不同）。
- `gen_ai.agent.name` — 智能体标识符。
- `gen_ai.operation.name` — `chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id` — 对于 RAG：咨询了哪个语料库或存储。

存在针对 Anthropic、Azure AI Inference、AWS Bedrock、OpenAI 的技术特定约定。

### 内容捕获

默认规则：插桩**不应默认捕获**输入/输出。捕获通过以下方式选择加入：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

建议的生产模式：将内容存储在外部（S3、你的日志存储），在跨度上记录引用（指针 ID，而非正文）。这是第 27 课内容中毒防御接入可观测性的方式。

### 稳定性

截至 2026 年 3 月，大多数约定仍是实验性的。通过以下方式选择加入稳定预览：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 原生将 GenAI 属性映射到其 LLM 可观测性模式。其他后端（Grafana、Honeycomb、Jaeger）支持原始属性。

### 这个模式会出错的地方

- **在跨度中捕获完整提示。** 运维可读的追踪中有 PII、密钥、客户数据。存储在外部。
- **没有 `gen_ai.provider.name`。** 多提供商仪表板在归属缺失时崩溃。
- **没有父链接的跨度。** 孤立的工具跨度。始终传播上下文。
- **未设置稳定性 opt-in。** 后端升级时你的属性可能被重命名。

## 构建

`code/main.py` 实现了一个符合 GenAI 约定的标准库跨度发射器：

- 带 GenAI 属性模式的 `Span`。
- 带 `start_span`、嵌套上下文的 `Tracer`。
- 发出以下内容的脚本化智能体运行：`create_agent`、`invoke_agent`（INTERNAL）、per-tool 跨度、LLM 调用的 `chat` 跨度。
- 一种内容捕获模式，将提示存储在外部并在跨度上记录 ID。

运行：

```
python3 code/main.py
```

输出：一个跨度树，包含所有必需的 GenAI 属性，以及一个"外部存储"显示 opt-in 内容引用。

## 使用

- **Datadog LLM 可观测性**（v1.37+）原生映射属性。
- **Langfuse / Phoenix / Opik**（第 24 课）— 自动插桩生态系统。
- **Jaeger / Honeycomb / Grafana Tempo** — 原始 OTel 追踪；从 GenAI 属性构建仪表板。
- **自托管** — 使用 GenAI 处理器运行 OTel Collector。

## 交付

`outputs/skill-otel-genai.md` 将 OTel GenAI 跨度接入现有智能体，包含内容捕获默认值和外部引用存储。

## 练习

1. 用 `invoke_agent`（INTERNAL）+ per-tool 跨度为你的第 01 课 ReAct 循环添加插桩。发送到 Jaeger 实例。
2. 以"仅引用"模式添加内容捕获：提示写入 SQLite，跨度属性仅携带行 ID。
3. 阅读 `gen_ai.data_source.id` 的规范。将它接入你的第 09 课 Mem0 搜索。
4. 设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 并验证你的属性不会被 collector 重命名。
5. 构建仪表板：仅从 GenAI 属性回答"哪些工具错误与哪些模型相关"。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| GenAI SIG | "OpenTelemetry GenAI 组" | 定义该模式的 OTel 工作组 |
| invoke_agent | "Agent 跨度" | 代表智能体运行的跨度名称 |
| CLIENT 跨度 | "远程调用" | 调用远程智能体服务的跨度 |
| INTERNAL 跨度 | "进程内" | 进程内智能体运行的跨度 |
| gen_ai.provider.name | "Provider" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG 来源" | 检索命中的哪个语料库/存储 |
| 内容捕获 | "提示日志" | 消息的选择性捕获；在生产中存储在外部 |
| 稳定性 opt-in | "预览模式" | 固定实验性约定的环境变量 |

## 延伸阅读

- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 默认使用 GenAI 跨度
- [AutoGen v0.4（Microsoft Research）](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 内置 OTel 跨度
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C 追踪上下文传播