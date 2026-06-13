# ASCII艺术与视觉越狱

> Jiang, Xu, Niu, Xiang, Ramasubramanian, Li, Poovendran, "ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs" (ACL 2024, arXiv:2402.11753). Mask the safety-relevant tokens in a harmful request, replace them with ASCII-art renderings of the same letters, and send the cloaked prompt. GPT-3.5, GPT-4, Gemini, Claude, Llama-2 all fail to robustly recognize ASCII-art tokens. The attack bypasses PPL (perplexity filters), Paraphrase defenses, and Retokenization. Related: the ViTC benchmark measures recognition of non-semantic visual prompts; StructuralSleight generalizes to Uncommon Text-Encoded Structures (trees, graphs, nested JSON) as a family of encoding attacks.

**类型：** 构建型
**语言：** Python（标准库，ArtPrompt token-masking 工具）
**前置条件：** 阶段 18 · 12（PAIR）、阶段 18 · 13（MSJ）
**时间：** 约 60 分钟

## 学习目标

- 描述 ArtPrompt 攻击：词语识别步骤、ASCII艺术替换、最终隐藏提示。
- 解释为什么标准防御（PPL、Paraphrase、Retokenization）在 ArtPrompt 上失败。
- 定义 ViTC 并描述它测量什么。
- 描述 StructuralSleight 作为对任意非常见文本编码结构的泛化。

## 问题

通过改写和角色扮演（第 12 课）以及长上下文（第 13 课）的攻击在文本级模式上操作。ArtPrompt 在识别级操作：模型不解析被禁止的 token。它解析一个用字符渲染的图像。安全过滤器看到无害的标点。模型看到一个词。

## 概念

### ArtPrompt，两步

第一步：词语识别。给定有害请求，攻击者使用 LLM 识别安全相关词汇（例如，"how to make a bomb"中的"bomb"）。

第二步：隐藏提示生成。将每个识别的词替换为其 ASCII 艺术渲染（形成字母形状的 7x5 或 7x7 字符块）。模型接收到一个标点和空间的网格，一个足够强大的模型可以将其识别为单词；安全过滤器只看到网格。

结果：GPT-4、Gemini、Claude、Llama-2、GPT-3.5 全部失败。在他们的基准子集上攻击成功率超过 75%。

### 为什么标准防御失败

- **PPL（困惑度过滤器）。** ASCII 艺术有高困惑度——但所有新输入都有。高阈值在阻止 ArtPrompt 的同时也会阻止合法的结构化输入。
- **改写。** 改写提示会破坏 ASCII 艺术。实际上，改写 LLM 通常会保留或重建该艺术。
- **再分词（Retokenization）。** 以不同方式分割 token 不会改变模型视觉在识别字母形状。

根本问题是安全过滤器是 token 级或语义级的；ArtPrompt 在视觉识别级操作。

### ViTC 基准

非语义视觉提示的识别。测量模型读取 ASCII 艺术、wingdings 和其他非文本语义视觉内容的能力。ArtPrompt 的有效性与其 ViTC 准确率相关：模型读取视觉文本的能力越好，ArtPrompt 对其效果越好。这是一个能力-安全权衡。

### StructuralSleight

将 ArtPrompt 泛化：非常见文本编码结构（UTES）。树、图、嵌套 JSON、CSV-in-JSON、diff 风格代码块。如果一种结构在训练安全数据中很少见但模型可以解析，它就可以隐藏有害内容。

防御含义：安全必须跨模型可解析的结构化表示泛化。这个集合很大且在增长。

### 图像模态类比

视觉 LLM（GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）扩展了攻击面。使用实际图像的 ArtPrompt 风格攻击比 ASCII 艺术模拟更强，因为图像编码器产生更丰富的信号。

### 这在阶段 18 中的位置

第 12-14 课描述了三个正交攻击向量：迭代优化（PAIR）、上下文长度（MSJ）和编码（ArtPrompt/StructuralSleight）。第 15 课从以模型为中心的攻击转向系统边界攻击（间接提示注入）。第 16 课描述防御工具响应。

## 使用

`code/main.py` 构建了一个玩具 ArtPrompt。你可以用 ASCII 艺术字形隐藏有害查询中的特定词语，验证隐藏字符串通过关键词过滤器，并（可选）使用简单识别器将隐藏字符串解码回来。

## 交付

本课产出 `outputs/skill-encoding-audit.md`。给定越狱防御报告，它枚举所涵盖的编码攻击家族（ASCII 艺术、base64、leet-speak、UTF-8 同形字、UTES）和捕获每种攻击的防御层。

## 练习

1. 运行 `code/main.py`。验证隐藏字符串通过简单的关键词过滤器。报告所需的字符级更改。

2. 实现第二种编码：对相同目标词使用 base64。将过滤器绕过率与 ArtPrompt 进行比较，以及恢复难度。

3. 阅读 Jiang 等人 2024 第 4.3 节（五模型结果）。提出一个理由说明为什么 Claude 在相同基准上比 Gemini 的 ArtPrompt 抵抗力更强。

4. 设计一种生成前防御，检测提示中 ASCII 艺术形状的区域。测量对合法代码、表格和数学符号的误报率。

5. StructuralSleight 列出了 10 种编码结构。 sketch 一个处理所有 10 种的泛化防御，并估计每个受防御提示的计算成本。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| ArtPrompt | "ASCII 艺术攻击" | 两步越狱，用 ASCII 艺术渲染隐藏安全词 |
| 隐藏（Cloaking） | "隐藏词语" | 用模型能读取但过滤器不能读取的视觉表示替换被禁止的 token |
| UTES | "非常见结构" | 非常见文本编码结构——树、图、嵌套 JSON 等，用于走私内容 |
| ViTC | "视觉-文本能力" | 模型读取非语义视觉编码的能力基准 |
| 困惑度过滤器 | "PPL 防御" | 拒绝高困惑度的提示；失败因为合法结构化输入也有高得分 |
| 再分词 | "分词器转换防御" | 使用不同分词器预处理提示；失败因为识别是视觉的 |
| 同形字 | "看起来相似的字符" | 看起来与拉丁字母相同的 Unicode 字符；绕过子字符串检查 |

## 延伸阅读

- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — ASCII 艺术越狱论文
- [Li et al. — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) — UTES 泛化
- [Chao et al. — PAIR (第 12 课, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 互补的迭代攻击
- [Anil et al. — Many-shot Jailbreaking (第 13 课)](https://www.anthropic.com/research/many-shot-jailbreaking) — 互补的长度攻击