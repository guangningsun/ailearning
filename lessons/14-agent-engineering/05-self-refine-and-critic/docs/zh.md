# 自我优化与 CRITIC：迭代式输出改进

> 自我优化（Madaan 等，2023）让一个大语言模型扮演三个角色——生成、反馈、优化——构成一个循环。在 7 个任务上平均提升：+20 分。CRITIC（Gou 等，2023）通过将验证路由到外部工具来强化反馈步骤。2026 年这个模式在每个框架中都以"评估器-优化器"（Anthropic）或 guardrail 循环（OpenAI Agents SDK）的形式出现。

**类型：** 构建型
**语言：** Python（标准库）
**前置条件：** 阶段 14 · 01（智能体循环）、阶段 14 · 03（Reflexion）
**时间：** 约 60 分钟

## 学习目标

- 说出自我优化的三个提示词（生成、反馈、优化），并解释为什么历史记录对优化提示词很重要。
- 解释 CRITIC 的关键洞察：没有外部接地，大语言模型在自我验证上不可靠。
- 用标准库实现带历史记录的自我优化循环，以及可选的外部验证器。
- 将此模式映射到 Anthropic 的"评估器-优化器"工作流和 OpenAI Agents SDK 的输出 guardrail。

## 问题

智能体产生了一个几乎正确的答案。也许一行代码有语法错误。也许总结太长了。也许计划漏掉了一个边界情况。你想要的是：智能体批判自己的输出，然后修复它。

自我优化证明这可以用单个模型实现，无需训练数据，无需强化学习。但有一个问题：大语言模型在硬事实上做自我验证时表现不佳。CRITIC 给出了修复方案——将验证步骤路由到外部工具（搜索、代码解释器、计算器、测试运行器）。

这两篇论文共同定义了 2026 年迭代改进的默认模式：生成，通过外部途径验证，优化，当验证器通过时停止。

## 概念

### 自我优化（Madaan 等，NeurIPS 2023）

一个模型，三种角色：

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
当 feedback 说"没有问题"或预算耗尽时停止。
```

关键细节：`refine` 能看到完整的历史——所有之前的输出和批判——所以它不会重复犯错。论文做了消融实验：去掉历史，质量会急剧下降。

 headline：7 个任务（数学、代码、缩写词、对话）上平均提升 +20 分，包括 GPT-4。不需要训练，不需要外部工具，单个模型。

### CRITIC（Gou 等，arXiv:2305.11738，v4，2024 年 2 月）

自我优化的弱点：反馈步骤是大语言模型给自己打分。对于事实性声明，这不可靠（一个幻觉往往对它自己的始作俑者看起来很有说服力）。CRITIC 将 `feedback(task, output)` 替换为 `verify(task, output, tools)`，其中 `tools` 包括：

- 用于事实声明的搜索引擎。
- 用于代码正确性的代码解释器。
- 用于算术的计算器。
- 领域特定的验证器（单元测试、类型检查器、linter）。

验证器产生一个基于工具结果的结构化批判。优化器在此批判上做条件调整。

headline：CRITIC 在事实任务上优于自我优化，因为批判是有接地的。在没有外部验证器的任务上（创意写作、格式化），CRITIC 退化为自我优化。

### 停止条件

两种常见形式：

1. **验证器通过。** 外部测试返回成功。有外部验证器时首选（单元测试、类型检查器、guardrail 断言）。
2. **没有发出反馈。** 模型说"输出没问题"。更便宜但不可靠；配上最大迭代次数上限。

2026 年默认：结合两者。"当验证器通过或模型说没问题且迭代次数 >= 2 或迭代次数 >= max_iterations 时停止。"

### 评估器-优化器（Anthropic，2024 年）

Anthropic 2024 年 12 月的文章将其列为五种工作流模式之一。两种角色：

- 评估器：对输出评分并产生批判。
- 优化器：根据批判修改输出。

循环直到评估器通过。这是自我优化/CRITIC 在 Anthropic 语境下的说法。Anthropic 补充的关键工程细节：评估器和优化器的提示词应该显著不同，否则模型只会盖橡皮章。

### OpenAI Agents SDK 输出 guardrail

OpenAI Agents SDK 将此模式作为"输出 guardrail"提供。Guardrail 是一个在智能体产生输出后运行的验证器。如果 guardrail 触发（抛出 `OutputGuardrailTripwireTriggered`），输出被拒绝，智能体可以重试。Guardrail 可以调用工具（CRITIC 风格）或作为纯函数（自我优化风格）。

### 2026 年的陷阱

- **橡皮章循环。** 同一个模型用相同风格的提示词做生成和批判，会收敛到"看起来不错"。使用结构上不同的提示词，或用更小更便宜的模型做批判。
- **过度优化。** 每次优化 pass 都会增加延迟和 token。预算 1-3 次 pass；之后升级到人工审核。
- **CRITIC 用于trivial任务。** 如果没有外部验证器，CRITIC 退化为自我优化；不要为stub验证器付出延迟代价。

## 构建它

`code/main.py` 在一个玩具任务上实现自我优化和 CRITIC：给定一个主题，产生一个简短的要点列表。验证器检查格式（3 个要点，每个不超过 60 个字符）。CRITIC 添加了一个外部"事实验证器"，对已知的幻觉进行扣分。

组件：

- `generate`——脚本化生成器。
- `feedback`——LLM 风格的自我批评。
- `verify_external`——CRITIC 风格的有接地验证器。
- `refine`——根据历史重写输出。
- 停止条件——验证器通过或最多 4 次迭代。

运行它：

```
python3 code/main.py
```

比较自我优化与 CRITIC 的运行。CRITIC 捕获了一个自我优化遗漏的事实错误，因为外部验证器有自我批评缺乏的接地。

## 使用它

Anthropic 的评估器-优化器是这个模式的 Claude 友好语言版本。OpenAI Agents SDK 的输出 guardrail 是 CRITIC 形状的（guardrail 可以调用工具）。LangGraph 提供一个读起来像自我优化的反思节点。Google 的 Gemini 2.5 Computer Use 添加了一个每步安全评估器，是 CRITIC 的变体：每个动作在提交前都经过验证。

## 交付它

`outputs/skill-refine-loop.md` 根据任务形状、验证器可用性和迭代预算配置一个评估器-优化器循环。发出生成器、评估器/验证器和优化器的提示词，加上停止策略。

## 练习

1. 用 max_iterations=1 运行玩具。CRITIC 还有帮助吗？
2. 把外部验证器换成一个有噪声的（随机 30% 误报率）。循环会怎么做？这是大多数 2026 年 guardrail 堆栈的现实。
3. 实现一个"不同模型上的生成器-批评者"变体：大模型生成，小模型批评。会比同模型版本更好吗？
4. 阅读 CRITIC 第 3 节（arXiv:2305.11738 v4）。说出三个验证工具类别并各给出一个例子。
5. 将 OpenAI Agents SDK 的 `output_guardrails` 映射到 CRITIC 的验证器角色。SDK 哪些做对了，哪些做错了？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| Self-Refine | "自己修复自己的 LLM" | 单模型中的生成 -> 反馈 -> 优化循环，带历史 |
| CRITIC | "工具接地的验证" | 用外部验证器（搜索、代码、计算、测试）替换反馈 |
| Evaluator-Optimizer | "Anthropic 工作流模式" | 两个角色——评估器评分，优化器修改——循环至收敛 |
| Output guardrail | "事后检查" | OpenAI Agents SDK 验证器，在智能体产生输出后运行 |
| Verify step | "批判阶段" | 承重决策：有接地还是自我评分 |
| Refine history | "模型已经尝试过的" | 先前的输出 + 批判被 prepend 到优化提示词；去掉历史质量就会崩溃 |
| Rubber-stamp loop | "自我同意失败" | 相同提示词的批判返回"看起来不错"；用结构上不同的提示词修复 |
| Stop condition | "收敛测试" | 验证器通过或没有反馈且迭代次数 >= 2；永远不要单条件 |

## 延伸阅读

- [Madaan 等，Self-Refine（arXiv:2303.17651）](https://arxiv.org/abs/2303.17651)——经典论文
- [Gou 等，CRITIC（arXiv:2305.11738）](https://arxiv.org/abs/2305.11738)——工具接地验证
- [Anthropic，Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)——评估器-优化器工作流模式
- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/)——输出 guardrail 作为 CRITIC 形状的验证器