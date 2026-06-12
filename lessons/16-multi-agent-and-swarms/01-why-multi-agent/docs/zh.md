# 为什么是多智能体？

> 一个智能体碰壁了。聪明的做法不是更大的智能体——而是更多的智能体。

**类型：** 学习型
**语言：** TypeScript
**前置条件：** 阶段 14（智能体工程）
**时间：** 约 60 分钟

## 学习目标

- 识别单智能体的上限（上下文溢出、专业混杂、顺序瓶颈），并解释何时拆分为多个智能体是正确的选择
- 对比编排模式（流水线、并行扇出、监督器、分层），并根据给定任务结构选择合适的模式
- 设计一个多智能体系统，具备清晰的角色边界、共享状态和通信契约
- 分析多智能体复杂性的权衡（延迟、成本、调试难度）与单智能体简单性的对比

## 问题

你在阶段 14 中构建了一个单智能体。它能正常运行。它可以读取文件、运行命令、调用 API，并对结果进行推理。然后你把它指向一个真实的代码库：200 个文件、三种语言、依赖基础设施的测试，以及在编写代码之前需要研究外部 API 的需求。

智能体卡住了。不是因为 LLM 太笨，而是因为任务超出了单个智能体循环能处理的能力。上下文窗口被文件内容填满。智能体忘记了 40 次工具调用前读取的内容。它试图同时扮演研究员、程序员和审查员三种角色，结果每一样都做得很差。

这就是单智能体的上限。每当任务需要以下条件时，你就会遇到它：

- **更多上下文而不止一个窗口的容量** —— 读取 50 个文件就会超过 200k token
- **不同阶段需要不同专业知识** —— 研究需要不同于代码生成的提示方式
- **可以并行完成的工作** —— 为什么要顺序读取三个文件，而不是同时读取？

## 概念

### 单智能体的上限

单个智能体是一个循环、一个上下文窗口、一个系统提示。可以这样想象它：

```
┌─────────────────────────────────────────┐
│            单智能体                      │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         上下文窗口                  │  │
│  │                                   │  │
│  │  研究笔记                         │  │
│  │  + 代码文件                       │  │
│  │  + 测试输出                       │  │
│  │  + 审查反馈                       │  │
│  │  + API 文档                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ 已满 ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  一个系统提示试图涵盖                    │
│  研究 + 编码 + 审查 + 测试              │
│                                         │
│  结果：每件事都做得很平庸                │
└─────────────────────────────────────────┘
```

有三件事会出问题：

1. **上下文饱和** —— 工具结果堆积。到第 30 轮时，智能体已经消耗了 150k token 的文件内容、命令输出和之前的推理。第 5 轮的关键细节丢失了。

2. **角色混淆** —— 一个说"你是研究员、程序员、审查员和测试员"的系统提示会产生一个半研究、半编码、永远完成不了审查的智能体。

3. **顺序瓶颈** —— 智能体先读文件 A，再读文件 B，然后读文件 C。三次串行 LLM 调用。三次串行工具执行。没有并行。

### 多智能体解决方案

拆分工作。让每个智能体做一个任务、一个上下文窗口、一个针对该任务调优的系统提示：

```
┌──────────────────────────────────────────────────────────┐
│                    编排器                                 │
│                                                          │
│  "为用户管理构建一个 REST API"                           │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │  研究员   │ │  程序员   │ │  审查员   │ │  测试员   │  │
│   │          │ │          │ │          │ │          │  │
│   │  读取    │ │  编写    │ │  检查    │ │  运行    │  │
│   │  文档，   │ │  代码    │ │  代码    │ │  测试，   │  │
│   │  发现    │ │  基于    │ │  质量，   │ │  报告    │  │
│   │  模式    │ │  研究    │ │  找出    │ │  结果    │  │
│   │          │ │  + 规范  │ │  bug    │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     合并结果                              │
└──────────────────────────────────────────────────────────┘
```

每个智能体都有：
- 一个专注的系统提示（"你是一个代码审查员。你唯一的工作是找 bug。"）
- 自己的上下文窗口（不会被其他智能体的工作污染）
- 清晰的输入/输出契约（接收研究笔记，输出代码）

### 真正这样做的系统

**Claude Code 子智能体** —— 当 Claude Code 用 `Task` 生成一个子智能体时，它会创建一个带有作用域任务的子智能体。父智能体保持上下文干净。子智能体做专注的工作并返回摘要。

**Devin** —— 运行一个规划器智能体、一个程序员智能体和一个浏览器智能体。规划器把工作分成步骤。程序员编写代码。浏览器研究文档。每个都有独立的上下文。

**多智能体编程团队（SWE-bench）** —— SWE-bench 上表现最好的系统使用一个研究员来阅读代码库、一个规划器来设计修复方案、一个程序员来实现它。单智能体系统得分较低。

**ChatGPT 深度研究** —— 并行生成多个搜索智能体，每个从不同角度探索，然后综合结果。

### 频谱

多智能体不是二元的。它是一个频谱：

```
简单 ──────────────────────────────────────────── 复杂

 单个        子智能体       流水线         团队          群体
 智能体

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │共享   │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │状态   │
             └───┘          └───┘───┘  │  消息   │    └───────┘
                                       │  总线   │
 1 个循环     父 +        按阶段       │        │    N 个对等体，
 1 个上下文   子任务       执行         └───────┘    涌现行为
                                       明确的
                                       角色
```

**单智能体** —— 一个循环、一个提示。适合简单任务。

**子智能体** —— 父智能体生成子智能体来处理专注的子任务。父智能体维护计划。子智能体汇报工作。这就是 Claude Code 做的事。

**流水线** —— 智能体按顺序运行。智能体 A 的输出成为智能体 B 的输入。适合分阶段工作流：研究 -> 编码 -> 审查 -> 测试。

**团队** —— 智能体通过共享消息总线并行运行。每个有各自角色。编排器协调。适合需要同时使用不同技能的场景。

**群体** —— 许多相同或几乎相同的智能体共享状态。没有固定的编排器。智能体从队列中领取工作。适合高吞吐量并行任务。

### 四种多智能体模式

#### 模式 1：流水线

```
输入 ──▶ 智能体 A ──▶ 智能体 B ──▶ 智能体 C ──▶ 输出
          (研究)      (编码)       (审查)
```

每个智能体转换数据并向前传递。简单易理解。一个阶段的失败阻塞其余阶段。

#### 模式 2：扇出 / 扇入

```
                ┌──▶ 智能体 A ──┐
                │              │
输入 ──▶ 拆分 ├──▶ 智能体 B ──├──▶ 合并 ──▶ 输出
                │              │
                └──▶ 智能体 C ──┘
```

跨并行智能体拆分工作，然后合并结果。适合能分解为独立子任务的任务。

#### 模式 3：编排器-工作器

```
                    ┌──────────┐
                    │  编排器   │
                    └──┬───┬───┘
                  任务 │   │ 任务
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ 工作器 A  │   │ 工作器 B  │
           └──────────┘   └──────────┘
```

智能编排器决定做什么，委托给工作器，并综合结果。编排器本身是一个带有用于生成工作器的工具的智能体。

#### 模式 4：对等群体

```
         ┌───┐ ◄──── 消息 ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      消息 │    ┌───────────┐     │ 消息
           └───▶│  共享     │◄────┘
                │  状态     │
           ┌───▶│  / 队列   │◄────┐
           │    └───────────┘     │
      消息 │                      │ 消息
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── 消息 ────▶ │ D │
         └───┘                  └───┘
```

没有中央编排器。智能体点对点通信。决策从交互中涌现。更难调试，但能扩展到多个智能体。

### 何时不使用多智能体

多智能体增加复杂性。智能体之间的每个消息都是一个潜在的故障点。调试从"阅读一个对话"变成"跨五个智能体跟踪消息"。

**在以下情况下保持单智能体：**
- 任务适合一个上下文窗口（工作数据在 ~100k token 以内）
- 不同阶段不需要不同的系统提示
- 顺序执行已经足够快
- 任务足够简单，拆分它的开销大于价值

**复杂性代价：**
- 每个智能体边界都是一个有损压缩步骤：智能体 A 的完整上下文被总结成智能体 B 的一条消息
- 协调逻辑（谁做什么、何时做、按什么顺序）本身就是 bug 的来源
- 延迟增加：N 个智能体意味着至少 N 次串行 LLM 调用，如果它们需要来回通信则更多
- 成本成倍增加：每个智能体独立消耗 token

经验法则：如果一个任务少于 20 次工具调用且在 100k token 以内，保持单智能体。

## 构建

### 第 1 步：过载的单智能体

这里是一个试图完成所有事情的单个智能体。它有一个庞大的系统提示和一个包含研究、代码和审查的上下文窗口：

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `你是一个全栈开发者。你必须：
1. 研究需求
2. 编写代码
3. 审查代码中的 bug
4. 编写测试
在一次对话中完成所有这些。`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `研究：${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `根据这项研究：\n${contextWindow.join("\n")}\n\n现在为以下任务编写代码：${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `根据所有之前的上下文：\n${contextWindow.join("\n")}\n\n审查代码。`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

这种做法的问题：
- 上下文窗口随着每个阶段增长。到审查步骤时，它包含研究笔记 AND 代码 AND 之前的推理。
- 系统提示是通用的。它不能为每个阶段调优。
- 没有东西是并行运行的。

### 第 2 步：专业智能体

现在拆分它。每个智能体做一个任务：

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "你是一个技术研究员。阅读文档，发现模式，并总结发现。只输出实现所需的事实。"
);

const coder = createSpecialist(
  "coder",
  "你是一个高级 TypeScript 开发者。根据需求和研究笔记，编写干净、有测试的代码。不做其他事。"
);

const reviewer = createSpecialist(
  "reviewer",
  "你是一个代码审查员。找出 bug、安全问题和逻辑错误。要具体。引用行号。"
);
```

每个专业智能体有一个专注的提示。每个获得一个只有它所需输入的干净上下文窗口。

### 第 3 步：通过消息协调

用显式消息传递将专业智能体连接在一起：

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[来自 ${m.from}]：${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[来自 ${m.from}]：${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]：${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

每个智能体只接收发给它的消息。没有上下文污染。研究员的 50k token 文档阅读永远不会进入审查员的上下文。

### 第 4 步：对比

```typescript
async function compare() {
  const task = "为 Express.js API 构建一个限流中间件";

  console.log("=== 单智能体 ===");
  const single = await singleAgentApproach(task);
  console.log(`Token：${single.tokensUsed}`);
  console.log(`工具调用：${single.toolCalls}`);

  console.log("\n=== 多智能体 ===");
  const multi = await multiAgentApproach(task);
  console.log(`Token：${multi.tokensUsed}`);
  console.log(`工具调用：${multi.toolCalls}`);
}
```

多智能体版本使用更多总 token（三个智能体，三次独立的 LLM 调用），但每个智能体的上下文保持干净。由于系统提示是专门的，每个阶段的质量都会提高。

## 实际使用

本课产出一个可重用的提示词，用于决定何时使用多智能体。参见 `outputs/prompt-multi-agent-decision.md`。

## 练习

1. 添加第四个专业智能体：一个"测试员"智能体，接收来自程序员的代码和来自审查员的审查反馈，然后编写测试
2. 修改流水线，使审查员可以将反馈发回给程序员进行修订循环（最多 2 轮）
3. 将顺序流水线转换为扇出：并行运行研究员和一个"需求分析器"智能体，然后在传递给程序员之前合并它们的输出

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|----------------------|
| 群体 | "AI 智能体的蜂巢思维" | 一组具有共享状态且没有固定领导的对等智能体。行为从局部交互中涌现。 |
| 编排器 | "老板智能体" | 一个其工具包含生成和管理其他智能体的智能体。它做规划和委托，但可能不做实际工作。 |
| 协调器 | "交通警察" | 一个非智能体组件（通常是代码，不是 LLM），根据规则在智能体之间路由消息。 |
| 共识 | "智能体们达成一致" | 在继续之前多个智能体必须达成协议的协议。当冲突输出需要解决时使用。 |
| 涌现行为 | "智能体们自己想出来的" | 从智能体交互中产生的但未明确编程的系统级模式。可能有用也可能有害。 |
| 扇出 / 扇入 | "智能体的 map-reduce" | 跨并行智能体拆分任务（扇出），然后组合它们的结果（扇入）。 |
| 消息传递 | "智能体们互相交谈" | 智能体之间的通信机制：从一个智能体发送到另一个的結構化数据，取代共享上下文窗口。 |

## 延伸阅读

- [新兴 AI 智能体架构全景](https://arxiv.org/abs/2409.02977) —— 多智能体模式调查
- [AutoGen：实现下一代 LLM 应用](https://arxiv.org/abs/2308.08155) —— 微软的多智能体对话框架
- [Claude Code 子智能体文档](https://docs.anthropic.com/en/docs/claude-code) —— Claude Code 如何用 Task 委托
- [CrewAI 文档](https://docs.crewai.com/) —— 基于角色的多智能体框架