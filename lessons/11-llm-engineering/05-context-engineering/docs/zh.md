# 上下文工程：窗口、预算、记忆与检索

> 提示词工程只是一个子集。上下文工程才是整盘游戏。提示词是你输入的一串文字。上下文则是进入模型窗口的一切：系统指令、检索到的文档、工具定义、对话历史、少样本示例，以及提示词本身。2026 年最优秀的 AI 工程师都是上下文工程师。他们决定什么放进去、什么去掉、以及以什么顺序排列。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 10（从零构建 LLM）、阶段 11 第 01-02 课
**时间：** 约 90 分钟
**相关：** 阶段 11 · 15（提示词缓存）— 缓存友好的布局是上下文工程的延伸。阶段 5 · 28（长上下文评估）— 了解如何用 NIAH/RULER 衡量"中间迷失"问题。

## 学习目标

- 计算所有上下文窗口组件的 token 预算（系统提示词、工具、历史记录、检索文档、生成预留空间）
- 实现上下文窗口管理策略：截断、摘要、滑动窗口（对话历史）
- 对上下文组件进行优先级排序和排序，使模型将注意力集中在最相关的信息上
- 构建一个上下文组装器，根据查询类型和可用窗口空间动态分配 token

## 问题

Claude Opus 4.7 有 200K token 的窗口（beta 版 1M）。GPT-5 有 400K。Gemini 3 Pro 有 2M。Llama 4 声称 10M。这些数字看起来很庞大，直到你把它们填满。

这里有一个真实的编码助手费用分解。系统提示词：500 token。50 个工具的定义：8,000 token。检索到的文档：4,000 token。对话历史（10 轮）：6,000 token。当前用户查询：200 token。生成预算（最大输出）：4,000 token。总计：22,700 token。这只是 128K 窗口的 18%。

但注意力并不随上下文长度线性扩展。一个拥有 128K token 上下文的模型要付出二次注意力成本（标准 transformer 中为 O(n²)，尽管大多数生产模型使用高效注意力变体）。更重要的是，检索准确率会下降。"大海捞针"测试表明，模型在长上下文中很难找到放在中间位置的信息。Liu 等人（2023）的研究表明，LLM 在长上下文开头和结尾位置检索信息的准确率接近完美，但放在中间位置（上下文的 40-70%）的信息准确率下降 10-20%。这种"中间迷失"效应因模型而异，但影响所有当前架构。

实践的教训：拥有 200K 可用 token 并不意味着使用 200K token 是有效的。经过精心筛选的 10K token 上下文通常优于随意塞入的 100K token 上下文。上下文工程是在上下文窗口中最大化信噪比的学科。

窗口中的每一个 token 都会挤掉一个可能携带更相关信息的 token。每一个无关的工具定义、每一次过时的对话轮次、每一段不能回答问题的检索文本——都会让模型在任务上稍微变差。

## 概念

### 上下文窗口是一种稀缺资源

把上下文窗口想象成 RAM，而不是磁盘。它很快且可直接访问，但容量有限。你无法容纳所有内容。必须有所取舍。

```mermaid
graph TD
    subgraph Window["上下文窗口 (128K tokens)"]
        direction TB
        S["系统提示词\n~500 tokens"] --> T["工具定义\n~2K-8K tokens"]
        T --> R["检索到的上下文\n~2K-10K tokens"]
        R --> H["对话历史\n~2K-20K tokens"]
        H --> F["少样本示例\n~1K-3K tokens"]
        F --> Q["用户查询\n~100-500 tokens"]
        Q --> G["生成预算\n~2K-8K tokens"]
    end

    style S fill:#1a1a2e,stroke:#e94560,color:#fff
    style T fill:#1a1a2e,stroke:#0f3460,color:#fff
    style R fill:#1a1a2e,stroke:#ffa500,color:#fff
    style H fill:#1a1a2e,stroke:#51cf66,color:#fff
    style F fill:#1a1a2e,stroke:#9b59b6,color:#fff
    style Q fill:#1a1a2e,stroke:#e94560,color:#fff
    style G fill:#1a1a2e,stroke:#0f3460,color:#fff
```

每个组件都在争夺空间。添加更多工具定义意味着对话历史的空间减少。添加更多检索上下文意味着少样本示例的空间减少。上下文工程是分配预算以最大化任务性能的的艺术。

### 中间迷失

上下文工程中最重要的实证发现。模型对上下文开头和结尾的信息注意力更好。中间的信息获得的注意力分数较低，更容易被忽略。

Liu 等人（2023）系统地测试了这一点。他们将相关文档与 20 个无关文档混在一起，放在不同位置，然后测量答案准确率。当相关文档在第一个或最后一个位置时，准确率为 85-90%。当它在中间位置（第 20 个中的第 10 个）时，准确率下降到 60-70%。

这有直接的工程含义：

- 把最重要的信息放在开头（系统提示词、关键指令）
- 把当前查询和最相关的上下文放在最后（近因偏差有帮助）
- 把上下文的中间部分视为最低优先级区域
- 如果必须在中间包含信息，把关键点在结尾重复一份

```mermaid
graph LR
    subgraph Attention["上下文中注意力的分布"]
        direction LR
        P1["位置 0-20%\n高注意力\n（系统提示词）"]
        P2["位置 20-40%\n中等注意力"]
        P3["位置 40-70%\n低注意力\n（中间迷失）"]
        P4["位置 70-90%\n中等注意力"]
        P5["位置 90-100%\n高注意力\n（当前查询）"]
    end

    style P1 fill:#51cf66,color:#000
    style P2 fill:#ffa500,color:#000
    style P3 fill:#ff6b6b,color:#fff
    style P4 fill:#ffa500,color:#000
    style P5 fill:#51cf66,color:#000
```

### 上下文组件

**系统提示词**：设置角色、约束和行为规则。这放在最前面，跨轮次保持不变。Claude Code 的系统提示词大约有 6,000 token，包括工具定义和行为指令。保持简洁。系统提示词中的每一个词在每次 API 调用中都会重复。

**工具定义**：每个工具增加 50-200 token（名称、描述、参数模式）。50 个工具每个 150 token 就是 7,500 token，在任何对话发生之前。动态工具选择——只包含与当前查询相关的工具——可以减少 60-80%。

**检索到的上下文**：来自向量数据库的文档、搜索结果、文件内容。检索质量直接决定响应质量。差的检索比不检索更糟糕——它用噪声填满窗口并主动误导模型。

**对话历史**：每一个之前的用户消息和助手回复。随对话长度线性增长。50 轮对话每轮 200 token 就是 10,000 token 的历史。其中大部分与当前查询无关。

**少样本示例**：展示期望行为的输入/输出对。两到三个精心挑选的示例通常比数千 token 的指令更能提高输出质量。但它们占用空间。

**生成预算**：为模型响应保留的 token。如果你把窗口填满到容量，模型就没有空间回答了。至少保留 2,000-4,000 token 用于生成。

### 上下文压缩策略

**历史摘要**：不保留所有先前轮次的完整内容，而是定期对对话进行摘要。"我们讨论了 X，决定了 Y，用户想要 Z"，用 100 token 替换了占用 2,000 token 的 10 轮对话。当历史超过阈值时运行摘要（例如 5,000 token）。

**相关性过滤**：对每个检索到的文档针对当前查询打分，丢弃低于阈值的文档。如果你检索了 10 个块但只有 3 个相关，丢弃其他 7 个。3 个高度相关的块比 10 个平庸的块更好。

**工具剪枝**：对用户查询意图进行分类，只包含与该意图相关的工具。代码问题不需要日历工具。日程安排问题不需要文件系统工具。这可以把工具定义从 8,000 token 减少到 1,000。

**递归摘要**：对于非常长的文档，分阶段进行摘要。首先对每个部分进行摘要，然后对摘要进行摘要。一份 50 页的文档变成一个 500 token 的摘要，捕获关键点。

### 记忆系统

上下文工程跨越三个时间范围。

**短期记忆**：当前对话。直接存储在上下文窗口中。随每轮对话增长。通过摘要和截断管理。

**长期记忆**：跨对话持久化的事实和偏好。"用户偏好 TypeScript。""项目使用 PostgreSQL。"存储在数据库中，在会话开始时检索。Claude Code 将其存储在 CLAUDE.md 文件中。ChatGPT 将其存储在其记忆功能中。

**情景记忆**：可能相关的特定过去交互。"上周二，我们调试了 auth 模块中的类似问题。"存储为 embedding，在当前对话匹配过去情景时检索。

```mermaid
graph TD
    subgraph Memory["记忆架构"]
        direction TB
        STM["短期记忆\n（当前对话）\n直接位于上下文窗口中"]
        LTM["长期记忆\n（事实、偏好）\n数据库 → 会话开始时检索"]
        EM["情景记忆\n（过去交互）\nEmbedding → 按相似度检索"]
    end

    Q["当前查询"] --> STM
    Q --> LTM
    Q --> EM

    STM --> CW["上下文窗口"]
    LTM --> CW
    EM --> CW

    style STM fill:#1a1a2e,stroke:#51cf66,color:#fff
    style LTM fill:#1a1a2e,stroke:#0f3460,color:#fff
    style EM fill:#1a1a2e,stroke:#e94560,color:#fff
    style CW fill:#1a1a2e,stroke:#ffa500,color:#fff
```

### 动态上下文组装

关键洞察：不同的查询需要不同的上下文。静态系统提示词 + 静态工具 + 静态历史是一种浪费。最好的系统根据每个查询动态组装上下文。

1. 对查询意图进行分类
2. 选择相关工具（不是所有工具）
3. 检索相关文档（不是固定集合）
4. 包含相关的历史轮次（不是所有历史）
5. 添加与任务类型匹配的少样本示例
6. 按重要性排序：关键的在最前面，重要的在最后，可选的在中间

这就是区分优秀 AI 应用和卓越 AI 应用的关键。模型是一样的。上下文才是差异所在。

## 动手构建

### 第 1 步：Token 计数器

你无法预算你无法衡量的东西。构建一个简单的 token 计数器（使用空白分割的近似值，因为确切数量取决于分词器）。

```python
import json
import numpy as np
from collections import OrderedDict

def count_tokens(text):
    if not text:
        return 0
    return int(len(text.split()) * 1.3)

def count_tokens_json(obj):
    return count_tokens(json.dumps(obj))
```

### 第 2 步：上下文预算管理器

核心抽象。预算管理器跟踪每个组件使用多少 token 并强制执行限制。

```python
class ContextBudget:
    def __init__(self, max_tokens=128000, generation_reserve=4000):
        self.max_tokens = max_tokens
        self.generation_reserve = generation_reserve
        self.available = max_tokens - generation_reserve
        self.allocations = OrderedDict()

    def allocate(self, component, content, max_tokens=None):
        tokens = count_tokens(content)
        if max_tokens and tokens > max_tokens:
            words = content.split()
            target_words = int(max_tokens / 1.3)
            content = " ".join(words[:target_words])
            tokens = count_tokens(content)

        used = sum(self.allocations.values())
        if used + tokens > self.available:
            allowed = self.available - used
            if allowed <= 0:
                return None, 0
            words = content.split()
            target_words = int(allowed / 1.3)
            content = " ".join(words[:target_words])
            tokens = count_tokens(content)

        self.allocations[component] = tokens
        return content, tokens

    def remaining(self):
        used = sum(self.allocations.values())
        return self.available - used

    def utilization(self):
        used = sum(self.allocations.values())
        return used / self.max_tokens

    def report(self):
        total_used = sum(self.allocations.values())
        lines = []
        lines.append(f"Context Budget Report ({self.max_tokens:,} token window)")
        lines.append("-" * 50)
        for component, tokens in self.allocations.items():
            pct = tokens / self.max_tokens * 100
            bar = "#" * int(pct / 2)
            lines.append(f"  {component:<25} {tokens:>6} tokens ({pct:>5.1f}%) {bar}")
        lines.append("-" * 50)
        lines.append(f"  {'Used':<25} {total_used:>6} tokens ({total_used/self.max_tokens*100:.1f}%)")
        lines.append(f"  {'Generation reserve':<25} {self.generation_reserve:>6} tokens")
        lines.append(f"  {'Remaining':<25} {self.remaining():>6} tokens")
        return "\n".join(lines)
```

### 第 3 步：中间迷失重排序

实现重排序策略：最重要的项目放在开头和最后，最不重要的放在中间。

```python
def reorder_lost_in_middle(items, scores):
    paired = sorted(zip(scores, items), reverse=True)
    sorted_items = [item for _, item in paired]

    if len(sorted_items) <= 2:
        return sorted_items

    first_half = sorted_items[::2]
    second_half = sorted_items[1::2]
    second_half.reverse()

    return first_half + second_half

def score_relevance(query, documents):
    query_words = set(query.lower().split())
    scores = []
    for doc in documents:
        doc_words = set(doc.lower().split())
        if not query_words:
            scores.append(0.0)
            continue
        overlap = len(query_words & doc_words) / len(query_words)
        scores.append(round(overlap, 3))
    return scores
```

### 第 4 步：对话历史压缩器

对旧的对话轮次进行摘要以回收 token 预算。

```python
class ConversationManager:
    def __init__(self, max_history_tokens=5000):
        self.turns = []
        self.summaries = []
        self.max_history_tokens = max_history_tokens

    def add_turn(self, role, content):
        self.turns.append({"role": role, "content": content})
        self._compress_if_needed()

    def _compress_if_needed(self):
        total = sum(count_tokens(t["content"]) for t in self.turns)
        if total <= self.max_history_tokens:
            return

        while total > self.max_history_tokens and len(self.turns) > 4:
            old_turns = self.turns[:2]
            summary = self._summarize_turns(old_turns)
            self.summaries.append(summary)
            self.turns = self.turns[2:]
            total = sum(count_tokens(t["content"]) for t in self.turns)

    def _summarize_turns(self, turns):
        parts = []
        for t in turns:
            content = t["content"]
            if len(content) > 100:
                content = content[:100] + "..."
            parts.append(f"{t['role']}: {content}")
        return "Previous: " + " | ".join(parts)

    def get_context(self):
        parts = []
        if self.summaries:
            parts.append("[Conversation Summary]")
            for s in self.summaries:
                parts.append(s)
        parts.append("[Recent Conversation]")
        for t in self.turns:
            parts.append(f"{t['role']}: {t['content']}")
        return "\n".join(parts)

    def token_count(self):
        return count_tokens(self.get_context())
```

### 第 5 步：动态工具选择器

只包含与当前查询相关的工具。对意图进行分类，然后过滤。

```python
TOOL_REGISTRY = {
    "read_file": {
        "description": "Read contents of a file",
        "tokens": 120,
        "categories": ["code", "files"],
    },
    "write_file": {
        "description": "Write content to a file",
        "tokens": 150,
        "categories": ["code", "files"],
    },
    "search_code": {
        "description": "Search for patterns in codebase",
        "tokens": 130,
        "categories": ["code"],
    },
    "run_command": {
        "description": "Execute a shell command",
        "tokens": 140,
        "categories": ["code", "system"],
    },
    "create_calendar_event": {
        "description": "Create a new calendar event",
        "tokens": 180,
        "categories": ["calendar"],
    },
    "list_emails": {
        "description": "List recent emails",
        "tokens": 160,
        "categories": ["email"],
    },
    "send_email": {
        "description": "Send an email message",
        "tokens": 200,
        "categories": ["email"],
    },
    "web_search": {
        "description": "Search the web for information",
        "tokens": 140,
        "categories": ["research"],
    },
    "query_database": {
        "description": "Run a SQL query on the database",
        "tokens": 170,
        "categories": ["code", "data"],
    },
    "generate_chart": {
        "description": "Generate a chart from data",
        "tokens": 190,
        "categories": ["data", "visualization"],
    },
}

def classify_intent(query):
    query_lower = query.lower()

    intent_keywords = {
        "code": ["code", "function", "bug", "error", "file", "implement", "refactor", "debug", "test"],
        "calendar": ["meeting", "schedule", "calendar", "appointment", "event"],
        "email": ["email", "mail", "send", "inbox", "message"],
        "research": ["search", "find", "what is", "how does", "explain", "look up"],
        "data": ["data", "query", "database", "chart", "graph", "analytics", "sql"],
    }

    scores = {}
    for intent, keywords in intent_keywords.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[intent] = score

    if not scores:
        return ["code"]

    max_score = max(scores.values())
    return [intent for intent, score in scores.items() if score >= max_score * 0.5]

def select_tools(query, token_budget=2000):
    intents = classify_intent(query)
    relevant = {}
    total_tokens = 0

    for name, tool in TOOL_REGISTRY.items():
        if any(cat in intents for cat in tool["categories"]):
            if total_tokens + tool["tokens"] <= token_budget:
                relevant[name] = tool
                total_tokens += tool["tokens"]

    return relevant, total_tokens
```

### 第 6 步：完整上下文组装管道

将所有内容连接起来。给定一个查询，动态组装最佳上下文。

```python
class ContextEngine:
    def __init__(self, max_tokens=128000, generation_reserve=4000):
        self.budget = ContextBudget(max_tokens, generation_reserve)
        self.conversation = ConversationManager(max_history_tokens=5000)
        self.system_prompt = (
            "You are a helpful AI assistant. You have access to tools for "
            "code editing, file management, web search, and data analysis. "
            "Use the appropriate tools for each task. Be concise and accurate."
        )
        self.knowledge_base = [
            "Python 3.12 introduced type parameter syntax for generic classes using bracket notation.",
            "The project uses PostgreSQL 16 with pgvector for embedding storage.",
            "Authentication is handled by Supabase Auth with JWT tokens.",
            "The frontend is built with Next.js 15 using the App Router.",
            "API rate limits are set to 100 requests per minute per user.",
            "The deployment pipeline uses GitHub Actions with Docker multi-stage builds.",
            "Test coverage must be above 80% for all new modules.",
            "The codebase follows the repository pattern for data access.",
        ]

    def assemble(self, query):
        self.budget = ContextBudget(self.budget.max_tokens, self.budget.generation_reserve)

        system_content, _ = self.budget.allocate("system_prompt", self.system_prompt, max_tokens=1000)

        tools, tool_tokens = select_tools(query, token_budget=2000)
        tool_text = json.dumps(list(tools.keys()))
        tool_content, _ = self.budget.allocate("tools", tool_text, max_tokens=2000)

        relevance = score_relevance(query, self.knowledge_base)
        threshold = 0.1
        relevant_docs = [
            doc for doc, score in zip(self.knowledge_base, relevance)
            if score >= threshold
        ]

        if relevant_docs:
            doc_scores = [s for s in relevance if s >= threshold]
            reordered = reorder_lost_in_middle(relevant_docs, doc_scores)
            doc_text = "\n".join(reordered)
            doc_content, _ = self.budget.allocate("retrieved_context", doc_text, max_tokens=3000)

        history_text = self.conversation.get_context()
        if history_text.strip():
            history_content, _ = self.budget.allocate("conversation_history", history_text, max_tokens=5000)

        query_content, _ = self.budget.allocate("user_query", query, max_tokens=500)

        return self.budget

    def chat(self, query):
        self.conversation.add_turn("user", query)
        budget = self.assemble(query)
        response = f"[Response to: {query[:50]}...]"
        self.conversation.add_turn("assistant", response)
        return budget


def run_demo():
    print("=" * 60)
    print("  Context Engineering Pipeline Demo")
    print("=" * 60)

    engine = ContextEngine(max_tokens=128000, generation_reserve=4000)

    print("\n--- Query 1: Code task ---")
    budget = engine.chat("Fix the bug in the authentication module where JWT tokens expire too early")
    print(budget.report())

    print("\n--- Query 2: Research task ---")
    budget = engine.chat("What is the best approach for implementing vector search in PostgreSQL?")
    print(budget.report())

    print("\n--- Query 3: After conversation history builds up ---")
    for i in range(8):
        engine.conversation.add_turn("user", f"Follow-up question number {i+1} about the implementation details of the system")
        engine.conversation.add_turn("assistant", f"Here is the response to follow-up {i+1} with technical details about the architecture")

    budget = engine.chat("Now implement the changes we discussed")
    print(budget.report())

    print("\n--- Tool Selection Examples ---")
    test_queries = [
        "Fix the bug in auth.py",
        "Schedule a meeting with the team for Tuesday",
        "Show me the database query performance stats",
        "Search for best practices on error handling",
    ]

    for q in test_queries:
        tools, tokens = select_tools(q)
        intents = classify_intent(q)
        print(f"\n  Query: {q}")
        print(f"  Intents: {intents}")
        print(f"  Tools: {list(tools.keys())} ({tokens} tokens)")

    print("\n--- Lost-in-the-Middle Reordering ---")
    docs = ["Doc A (most relevant)", "Doc B (somewhat relevant)", "Doc C (least relevant)",
            "Doc D (relevant)", "Doc E (moderately relevant)"]
    scores = [0.95, 0.60, 0.20, 0.80, 0.50]
    reordered = reorder_lost_in_middle(docs, scores)
    print(f"  Original order: {docs}")
    print(f"  Scores:         {scores}")
    print(f"  Reordered:      {reordered}")
    print(f"  (Most relevant at start and end, least relevant in middle)")
```

## 实际使用

### Claude Code 的上下文策略

Claude Code 以分层方式管理上下文。系统提示词包括行为规则和工具定义（约 6K token）。当你打开文件时，其内容作为上下文注入。当你搜索时，结果被添加。旧的对话轮次被摘要。CLAUDE.md 提供跨会话持久化的长期记忆。

关键工程决策：Claude Code 不会将整个代码库转储到上下文中。它按需检索相关文件。这正是上下文工程的实践。

### Cursor 的动态上下文加载

Cursor 将整个代码库索引为 embedding。当你输入查询时，它使用向量相似性检索最相关的文件和代码块。只有那些片段进入上下文窗口。一个 500K 行的代码库被压缩成 5-10 个最相关的代码块。

这就是模式：embedding 所有内容，按需检索，只包含重要的内容。

### ChatGPT 记忆

ChatGPT 将用户偏好和事实存储为长期记忆。在每次对话开始时，相关记忆被检索并包含在系统提示词中。"用户偏好 Python"花费 5 token 但在跨会话的重复指令中节省了数百 token。

### RAG 作为上下文工程

检索增强生成是正式化的上下文工程。不是将知识塞入模型权重（训练）或系统提示词（静态上下文），而是在查询时检索相关文档并将其注入上下文窗口。整个 RAG 管道——分块、embedding、检索、重排序——存在来解决一个问题：将正确的信息放入上下文窗口。

## 交付物

本课产出 `outputs/prompt-context-optimizer.md`——一个可重用的提示词，用于审计上下文组装策略并推荐优化。向它提供你的系统提示词、工具数量、平均历史长度和检索策略，它会识别 token 浪费并提出改进建议。

还产出 `outputs/skill-context-engineering.md`——一个基于任务类型、上下文窗口大小和延迟预算设计上下文组装管道的决策框架。

## 练习

1. 在 ContextBudget 类中添加一个"token 浪费检测器"。它应该标记使用超过 30% 预算的组件，并针对每种组件类型提出特定的压缩策略（摘要历史、剪枝工具、对文档重新排序）。

2. 实现检索上下文的语义去重。如果两个检索到的文档相似度超过 80%（通过词重叠或 embedding 的余弦相似度），只保留得分较高的那个。测量这能回收多少 token 预算。

3. 构建一个"上下文回放"工具。给定对话记录，通过 ContextEngine 重放它，并可视化预算分配如何逐轮变化。绘制每个组件随时间的 token 使用情况。识别上下文开始被压缩的那一轮。

4. 实现基于优先级的工具选择器。不是二元的是/否包含，而是为每个工具分配与当前查询的相关性分数。按降序相关性顺序包含工具，直到工具预算用完。使用包含 5、10、20 和 50 个工具比较任务性能。

5. 构建一个多策略上下文压缩器。实现三种压缩策略（截断、摘要、关键句子提取）并对 20 个文档的集合进行基准测试。测量压缩率与信息保留之间的权衡（压缩版本是否仍然包含查询的答案？）。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|----------------------|
| 上下文窗口 | "模型能读多少" | 模型在单次前向传播中处理的最大 token 数（输入 + 输出）—— GPT-5 为 400K，Claude Opus 4.7 为 200K（1M beta），Gemini 3 Pro 为 2M |
| 上下文工程 | "高级提示词工程" | 决定什么进入上下文窗口、以什么顺序、以什么优先级——包含检索、压缩、工具选择和记忆管理的学科 |
| 中间迷失 | "模型忘记中间的内容" | 实证发现，LLM 对上下文开头和结尾的注意力更好，对放在中间位置的信息准确率下降 10-20% |
| Token 预算 | "你还剩多少 token" | 跨组件（系统提示词、工具、历史、检索、生成）对上下文窗口容量的明确分配，每个组件有单独的限额 |
| 动态上下文 | "动态加载内容" | 根据意图分类、相关工具选择和检索结果为每个查询不同地组装上下文窗口 |
| 历史摘要 | "压缩对话" | 用简洁的摘要替换完整的旧对话轮次，在保留关键信息的同时减少 token 成本 |
| 工具剪枝 | "只包含相关工具" | 对查询意图进行分类，只包含匹配的工具定义，将工具 token 成本减少 60-80% |
| 长期记忆 | "跨会话记忆" | 存储在数据库中并在会话开始时检索的事实和偏好——CLAUDE.md、ChatGPT 记忆和类似系统 |
| 情景记忆 | "记住特定过去事件" | 存储为 embedding 的过去交互，并在当前查询与过去对话相似时检索 |
| 生成预算 | "回答的空间" | 为模型输出保留的 token——如果上下文完全填满窗口，模型就没有空间回应 |

## 延伸阅读

- [Liu 等人，2023 — "迷失在中间：语言模型如何使用长上下文"](https://arxiv.org/abs/2307.03172) — 关于位置依赖注意力的权威研究，表明模型在长上下文中间位置的信息处理上存在困难
- [Anthropic 的上下文检索博客文章](https://www.anthropic.com/news/contextual-retrieval) — Anthropic 如何处理上下文感知块检索，将检索失败减少 49%
- [Simon Willison 的"上下文工程"](https://simonwillison.net/2025/Jun/27/context-engineering/) — 命名该学科并将其与提示词工程区分开来的博客文章
- [LangChain 的 RAG 文档](https://python.langchain.com/docs/tutorials/rag/) — 将检索增强生成作为上下文工程模式实现的实用指南
- [Greg Kamradt 的海底捞针测试](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 揭示所有主要模型位置依赖检索失败的基准
- [Pope 等人，"高效扩展 Transformer 推理"(2022)](https://arxiv.org/abs/2211.05102) — 为什么上下文长度驱动内存和延迟，以及 KV 缓存、MQA 和 GQA 如何改变预算计算
- [Agrawal 等人，"SARATHI：通过 Chunked Prefills 背负解码实现高效 LLM 推理"(2023)](https://arxiv.org/abs/2308.16369) — 两个推理阶段使长提示在 TTFT 中昂贵但在 TPOT 中便宜；上下文打包权衡的真相
- [Ainslie 等人，"GQA：从多头检查点训练广义多查询 Transformer 模型"(EMNLP 2023)](https://arxiv.org/abs/2305.13245) — 分组查询注意力论文，在生产解码器中将 KV 内存削减 8 倍而质量无损
