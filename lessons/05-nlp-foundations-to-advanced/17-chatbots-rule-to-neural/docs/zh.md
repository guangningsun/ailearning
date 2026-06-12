# 聊天机器人 — 从规则到神经再到 LLM 智能体

> ELIZA 用模式匹配回复。DialogFlow 映射意图。GPT 从权重中回答。Claude 运行工具并验证结果。每个时代都解决了前一个时代最大的失败。

**类型：** 学习型
**语言：** Python
**前置条件：** 阶段 5 · 13（问答系统）、阶段 5 · 14（信息检索）
**时间：** 约 75 分钟

## 问题

用户说"我想改航班"，系统需要弄清楚他们想要什么、缺少什么信息、如何获取信息以及如何完成操作。然后用户说"等等，如果我改取消呢？"——系统需要记住上下文、切换任务并保持状态。

对话对 ML 系统来说很难。输入是开放式的，输出需要在多轮对话中保持连贯。系统可能需要作用于现实世界（改航班、扣款）。每一步错误用户都能看到。

聊天机器人架构已经历了四种范式，每种范式的引入都是因为前一种范式的失败过于明显。本课按顺序讲解它们。2026 年的生产环境是后两种范式的混合。

## 概念

![聊天机器人演进：规则型 → 检索型 → 神经型 → 智能体型](../assets/chatbot.svg)

**规则型（ELIZA、AIML、DialogFlow）。** 手工编写的模式匹配用户输入并产生响应。意图分类器将请求路由到预定义的流程。槽位填充状态机收集所需信息。在其设计的窄范围内效果出色。在范围外立即失效。仍在安全关键领域（银行认证、航空公司预订）中使用，因为这些领域不能容忍幻觉。

**检索型。** FAQ 风格的系统。对每一对（话语、响应）进行编码。运行时，对用户消息进行编码并检索最接近的存储响应。类似于 Zendesk 经典的"相似文章"功能。比规则更好地处理改写。没有生成，因此不会产生幻觉。

**神经型（seq2seq）。** 在对话日志上训练的编码器-解码器。从头生成响应。流畅但容易产生通用输出（"我不知道"）和事实漂移。从未可靠地保持主题。这是 Google、Facebook 和 Microsoft 在 2016-2019 年都有令人失望的聊天机器人的原因。

**LLM 智能体。** 一个语言模型包裹在一个循环中，负责计划、调用工具和验证结果。不是带有长提示的聊天机器人，而是一个智能体循环：计划 → 调用工具 → 观察结果 → 决定下一步。检索优先的 grounding（RAG）防止幻觉。工具调用让它真正做事。这就是 2026 年的架构。

这四种范式不是顺序替代。2026 年的生产聊天机器人会路由通过所有四种：规则型用于认证和破坏性操作，检索型用于 FAQ，神经生成用于自然措辞，LLM 智能体用于模糊的开放式查询。

## 构建

### 第 1 步：基于规则的模式匹配

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

ELIZA 仅用 20 行。反思技巧（"I feel sad" → "Why do you feel sad"）是 Weizenbaum 1966 年的经典心理治疗师演示。至今仍有启发性。

### 第 2 步：检索型（FAQ）

这个示例代码需要 `pip install sentence-transformers`（会拉取 torch）。本课的 可运行 `code/main.py` 使用标准库的 Jaccard 相似度，因此无需外部依赖即可运行。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

基于阈值的拒绝是关键设计选择。如果最佳匹配不够接近，返回 `None` 并让系统升级处理。

### 第 3 步：神经生成（基线）

使用小型指令微调的编码器-解码器（FLAN-T5）或微调的对话模型。在 2026 年单独使用时生产上不可用（矛盾、离题漂移、事实胡说），但会在混合系统内用于自然措辞。DialoGPT 风格的纯解码器模型需要明确的轮次分隔符和 EOS 处理来产生连贯的回复；FLAN-T5 text2text 管道开箱即用，适合教学示例。

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### 第 4 步：LLM 智能体循环

2026 年的生产形态：

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

三个需要命名的概念。工具是 LLM 可调用的可调用函数。循环在 LLM 返回最终答案而不是工具调用时终止。步骤预算防止在模糊任务上无限循环。

真实生产环境还会添加：检索优先的 grounding（在每次 LLM 调用前注入相关文档）、护栏（破坏性操作未经确认不执行）、可观测性（记录每一步）和评估（自动化检查确保智能体行为保持符合规范）。

### 第 5 步：混合路由

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

模式：破坏性操作用确定性规则，FAQ 用检索，其他一切用 LLM 智能体。这就是 2026 年客户支持系统的实际部署方式。

## 使用

2026 年的技术栈：

| 使用场景 | 架构 |
|---------|---------------|
| 预订、支付、认证 | 规则型状态机 + 槽位填充 |
| 客户支持 FAQ | 检索经过策划的答案 |
| 开放式帮助聊天 | 带 RAG + 工具调用的 LLM 智能体 |
| 内部工具 / IDE 助手 | 带工具调用的 LLM 智能体（搜索、读取、写入） |
| 陪伴 / 角色聊天机器人 | 经过微调的 LLM + 人设系统提示，检索知识库 |

生产环境始终使用混合路由。没有单一架构能处理所有请求。路由层本身通常是一个小型意图分类器。

## 仍在使用中的失败模式

- **自信的捏造。** LLM 智能体声称完成了它没有完成的操作。缓解措施：验证结果、记录工具调用、绝不让 LLM 声称做了某事而没有成功的工具返回。
- **提示注入。** 用户插入覆盖系统提示的文本。在 OWASP 2025 年 LLM 应用 Top 10 中排名第一。有两种形式：直接注入（粘贴到聊天中）和间接注入（隐藏在智能体读取的文档、电子邮件或工具输出中）。

  攻击率因场景而异。在前沿模型的一般工具使用和编码基准测试中，测量到的成功率范围约为 0.5-8.5%。特定高风险设置（针对 AI 编码智能体的自适应攻击、易受攻击的编排）已达到约 84%。生产环境 CVE 包括 EchoLeak（CVE-2025-32711，CVSS 9.3）——Microsoft 365 Copilot 中的零点击数据泄露漏洞，由攻击者控制的电子邮件触发。

  缓解措施：在整个循环中将用户输入视为不可信的；在工具调用前进行清理；将工具输出与主提示隔离；使用计划-验证-执行（PVE）模式，智能体先计划，然后在该计划之前验证每个操作（这阻止工具结果注入新的未计划操作）；破坏性操作需要用户确认；对工具范围应用最小特权。

  没有任何提示工程能完全消除此风险。需要外部运行时防御层（LLM Guard、允许列表验证、语义异常检测）。
- **范围蔓延。** 智能体偏离任务，因为工具调用返回了 tangential 相关的信息。缓解措施：缩小工具契约；保持系统提示专注；添加离题率的评估。
- **无限循环。** 智能体持续调用同一工具。缓解措施：步骤预算、工具调用去重、LLM 判断"我们是否正在取得进展"。
- **上下文窗口耗尽。** 长期对话将最早的轮次推出上下文。缓解措施：总结较早的轮次、通过相似性检索相关的过去轮次，或使用长上下文模型。

## 交付

保存为 `outputs/skill-chatbot-architect.md`：

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## 练习

1. **简单。** 用 10 个模式实现上述基于规则的响应，用于咖啡店点餐机器人。测试边界情况：重复订单、修改、取消、意图不清。
2. **中等。** 构建混合 FAQ + LLM 后备。50 个针对 SaaS 产品的预制 FAQ 条目，基于文档站点的检索式 LLM 后备。在 100 个真实支持问题上测量拒绝率和准确率。
3. **困难。** 用三个工具（搜索、读取用户数据、发送电子邮件）实现上述智能体循环。用 50 个测试场景进行评估，包括提示注入尝试。报告离题率、任务失败率以及任何注入成功情况。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| Intent（意图） | 用户想要什么 | 分类标签（book_flight、reset_password）。路由到处理器。 |
| Slot（槽位） | 一条信息 | 机器人需要的参数（日期、目的地）。槽位填充是提问的序列。 |
| RAG | 检索加生成 | 检索相关文档，然后为 LLM 的响应提供 grounding。 |
| Tool call（工具调用） | 函数调用 | LLM 发出带有名称和参数的結構化调用。运行时执行并返回结果。 |
| Agent loop（智能体循环） | 计划、行动、验证 | 控制器，交织运行 LLM 调用和工具调用直到任务完成。 |
| Prompt injection（提示注入） | 用户攻击提示 | 恶意输入，试图覆盖系统提示。 |

## 延伸阅读

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 最初的基于规则的聊天机器人论文。
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — Google 晚期的神经聊天机器人论文，就在 LLM 智能体取代它之前。
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — 命名智能体循环模式的论文。
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — 2024 年的生产指导，在 2026 年仍然有效。
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — 提示注入论文。
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — 使提示注入成为首要安全关注的排名。
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — 包括计划-验证-执行和用户确认流程的实用编排层防御。
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — 来自间接提示注入的经典零点击数据泄露 CVE。为什么有写访问权限的智能体需要运行时防御的参考案例。