# 基准测试：SWE-bench、GAIA、AgentBench

> 三个基准测试锚定了 2026 年的智能体评估。SWE-bench 测试代码修补。GAIA 测试通用工具使用。AgentBench 测试多环境推理。了解它们的构成、污染故事，以及它们没有测量什么。

**类型：** 学习型
**语言：** Python（标准库）
**前置条件：** 阶段 14 · 06（工具使用）
**时间：** 约 60 分钟

## 学习目标

- 说出 SWE-bench 的测试框架（FAIL_TO_PASS）并解释为什么它以单元测试为门槛。
- 解释为什么存在 SWE-bench Verified（OpenAI，500 个任务）以及它去除了什么。
- 描述 GAIA 的设计理念：对人类简单，对 AI 困难；三个难度级别。
- 说出 AgentBench 的八个环境及其对开源 LLM 的主要阻碍因素。
- 总结 SWE-bench+ 污染发现及其启示。

## 问题

排行榜告诉你哪个模型在某基准上获胜。它不会告诉你：

- 该基准是否被污染（训练数据中已有解决方案，测试泄露）。
- 该基准测量的是否是你关心的东西（代码 vs 浏览 vs 通用）。
- 评估器是否健壮（AST 匹配、状态检查、人工审查）。

在引用一个数字之前，先了解三个锚定基准及其失败模式。

## 概念

### SWE-bench（Jimenez 等，ICLR 2024 oral）

- 来自 12 个流行 Python 仓库的 2,294 个真实 GitHub issue。
- 智能体获得：修复前的代码库提交 + 自然语言 issue 描述。
- 智能体产出：一个补丁。
- 评估器：应用补丁，运行仓库的测试套件。补丁必须让 FAIL_TO_PASS 测试翻转（原本失败，现在通过），且不破坏 PASS_TO_PASS 测试。

SWE-agent（Yang 等，2024）在发布时达到 12.5%，该成绩强调智能体-计算机接口（文件编辑命令、模型能理解搜索语法）。

### SWE-bench Verified

OpenAI，2024 年 8 月。人类策划的 500 个任务子集。移除了有歧义的 issue、不可靠的测试以及修复方案不清晰的任务。是"你的智能体是否能交付真实补丁"的主要基准。

### 污染

- 超过 94% 的 SWE-bench issue 先于大多数模型的截止日期。
- **SWE-bench+** 发现 32.67% 的成功补丁在 issue 文本中泄露了解决方案（模型在描述中看到了修复方案），31.08% 因测试覆盖率弱而可疑。
- Verified 更干净，但不是无污染。

实际启示：在 SWE-bench 上得 50% 的模型，在 SWE-bench+ 上可能只得 35%。如果你声称 SWE-bench 性能，请同时报告两者。

### GAIA（Mialon 等，2023 年 11 月）

- 466 个问题；300 个保留用于 huggingface.co/gaia-benchmark 上的私有排行榜。
- 设计理念："对人类概念上简单（92%），对 AI 困难（GPT-4 + 插件：15%）。"
- 测试推理、多模态、网络、工具使用。
- 三个难度级别；第 3 级需要跨模态的长时间工具链。

GAIA 是你用来衡量"通用能力"的基准。别把它与代码专用基准混淆。

### AgentBench（Liu 等，ICLR 2024）

- 跨代码（Bash、DB、KG）、游戏（Alfworld、LTP）、网络（WebShop、Mind2Web）和开放式生成的 8 个环境。
- 多轮，每个拆分约 4k–13k 轮。
- 主要发现：长期推理、决策和指令遵循是开源 LLM 赶超商业模型的主要障碍。

### 这些基准没有测量什么

- 真实运营成本（token 数、墙上时钟时间）。
- 对抗性条件下的安全行为。
- 你所在领域的性能（用你自己的评估，第 30 课）。
- 尾部失败（基准取平均值；生产运营商关心最差的 1%）。

### 基准测试会出错的地方

- **单一数字执念。** SWE-bench 50% 告诉你的信息少于 P50/P75/P95 成本 + 步数分布。
- **污染性声明。** 报告 SWE-bench 而不提 Verified 或 SWE-bench+ 是误导性的。
- **以基准为开发目标。** 为基准优化会偏离生产实用性。

## 构建它

`code/main.py` 实现了一个类 SWE-bench 的玩具框架：

- 合成 bug 修复任务（3 个任务）。
- 一个脚本化"智能体"，提出补丁。
- 一个测试运行器，检查 FAIL_TO_PASS（bug 已修复）和 PASS_TO_PASS（没有破坏任何东西）。
- 一个基于问题分解深度的 GAIA 风格难度分类器。

运行：

```
python3 code/main.py
```

输出显示每个任务的解决率 + 每个难度级别的解决率，并将评估器规则具体化。

## 使用它

- **SWE-bench Verified** 用于代码智能体。始终报告 Verified 分数。
- **GAIA** 用于通用智能体。使用私有排行榜拆分。
- **AgentBench** 用于多环境对比。
- **自定义评估**（第 30 课）用于你的产品实际形态。

## 交付它

`outputs/skill-benchmark-harness.md` 为任意代码库-任务对构建一个 SWE-bench 风格的框架，带有 FAIL_TO_PASS / PASS_TO_PASS 门槛。

## 练习

1. 将玩具框架移植到真实仓库上运行（用你自己的）。为已知 bug 写 3 个 FAIL_TO_PASS 测试。
2. 添加步数指标。在你的 3 个任务上，每个解决方案花多少智能体步数？
3. 阅读 SWE-bench+ 论文。实现一个解决方案泄露检查（用 diff 对 issue 文本做模式匹配）。
4. 从公开拆分中下载一个 GAIA 问题。追踪一个 GPT-4 级智能体会怎么做。它需要哪些工具？
5. 阅读 AgentBench 按环境细分的内容。哪个环境最接近你的产品表面？该处的"SOTA"是什么样的？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| SWE-bench | "代码智能体基准" | 2,294 个 GitHub issue；补丁必须翻转 FAIL_TO_PASS 测试 |
| SWE-bench Verified | "干净的 SWE-bench" | 500 个人类策划任务，OpenAI |
| FAIL_TO_PASS | "修复门槛" | 原本失败、补丁后必须通过的测试 |
| PASS_TO_PASS | "无回归门槛" | 原本通过且必须继续保持通过的测试 |
| GAIA | "通用智能体基准" | 466 个人类容易 / AI 困难的多工具问题 |
| AgentBench | "多环境基准" | 8 个环境；长期多轮 |
| 污染（Contamination） | "训练集泄露" | 基准任务出现在模型训练数据中 |
| SWE-bench+ | "污染审计" | 在成功的 SWE-bench 补丁中发现 32.67% 的解决方案泄露 |

## 延伸阅读

- [Jimenez 等，SWE-bench（arXiv:2310.06770）](https://arxiv.org/abs/2310.06770) —— 原始基准
- [OpenAI，SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) —— 策划子集
- [Mialon 等，GAIA（arXiv:2311.12983）](https://arxiv.org/abs/2311.12983) —— 通用智能体基准
- [Liu 等，AgentBench（arXiv:2308.03688）](https://arxiv.org/abs/2308.03688) —— 多环境套件