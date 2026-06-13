# 内容审核系统 — OpenAI、Perspective、Llama Guard

> 生产审核系统将课程 12-16 中定义的安全策略付诸运作。OpenAI 审核 API：`omni-moderation-latest`（2024）基于 GPT-4o，单次调用可分类文本+图像；在多语言测试集上比前版本提升 42%；响应模式返回 13 个类别布尔值 — 骚扰、骚扰/威胁、仇恨、仇恨/威胁、非法、非法/暴力、自残、自残/意图、自残/指示、性、性/未成年人、暴力、暴力/图形；对大多数开发者免费。三层模式：输入审核（生成前）、输出审核（生成后）、自定义审核（领域规则）。异步并行调用隐藏延迟；标志时使用占位响应。Llama Guard 3/4（课程 16）：14 个 MLCommons 危害类别、代码解释器滥用、8 种语言（v3）、多图像（v4）。Perspective API（Google Jigsaw）：早于 LLM 审核时代的有害性评分；主要是一维有害性，严重有害性/侮辱/亵渎变体；内容审核研究的基准线。弃用：Azure Content Moderator 于 2024 年 2 月弃用，2027 年 2 月退役，由 Azure AI Content Safety 取代。

**类型：** 构建型
**语言：** Python（标准库、三层审核工具）
**前置条件：** 阶段 18 · 16（Llama Guard / Garak / PyRIT）
**时间：** 约 60 分钟

## 学习目标

- 描述 OpenAI 审核 API 的类别分类法及其与 Llama Guard 3 的 MLCommons 集合的区别。
- 描述三层审核模式（输入、输出、自定义）并为每层命名一个失败模式。
- 描述 Perspective API 作为前 LLM 时代基准线的地位，以及为什么它仍在研究中使用。
- 说明 Azure 弃用时间线。

## 问题

课程 12-16 描述了攻击和防御工具。课程 29 涵盖部署在用户接触产品的层面实际运作防御的生产审核系统。三层模式是 2026 年的默认配置。

## 概念

### OpenAI 审核 API

`omni-moderation-latest`（2024）。基于 GPT-4o。单次调用分类文本+图像。对大多数开发者免费。

类别（响应模式中的 13 个布尔值）：
- 骚扰、骚扰/威胁
- 仇恨、仇恨/威胁
- 自残、自残/意图、自残/指示
- 性、性/未成年人
- 暴力、暴力/图形
- 非法、非法/暴力

多模态支持适用于 `暴力`、`自残` 和 `性`，但不适用于 `性/未成年人`；其余仅支持文本。

对于 `code/main.py` 中的代码工具，我们为了教学简洁将 `/威胁`、`/意图`、`/指示` 和 `/图形` 子类别折叠到其顶层父类别。生产代码应使用完整的 13 类别模式。

在多语言测试集上比上一代审核端点提升 42%。按类别评分；应用设置阈值。

### Llama Guard 3/4

在课程 16 中涵盖。14 个 MLCommons 危害类别（组织方式与 OpenAI 的 13 个响应模式布尔值不同）。支持 8 种语言（v3）。Llama Guard 4（2025 年 4 月）原生多模态，12B 参数。

OpenAI 和 Llama Guard 的分类有重叠但有分歧。OpenAI 将"非法"作为一个广泛类别；Llama Guard 将"暴力犯罪"和"非暴力犯罪"分开。部署根据其策略分类适配度选择。

### Perspective API（Google Jigsaw）

有害性评分系统，早于 LLM 审核时代（2020 年之前）。类别：TOXICITY、SEVERE_TOXICITY、INSULT、PROFANITY、THREAT、IDENTITY_ATTACK。一维主要评分（TOXICITY）及子维度变体。

广泛用作内容审核研究基准线，因为 API 稳定、有文档记录、有多年的校准数据。对于现代 LLM 相关用例，Llama Guard 或 OpenAI 审核通常是更好的选择。

### 三层模式

1. **输入审核。** 在生成前对用户提示进行分类。如果被标志则拒绝。延迟：一个分类器调用。
2. **输出审核。** 在传递前对模型输出进行分类。如果被标志则替换为拒绝。延迟：生成后一个分类器调用。
3. **自定义审核。** 领域特定规则（正则表达式、允许列表、业务策略）。在输入或输出端运行。

三层按设计顺序执行：输入审核必须在生成前完成，输出审核在生成后运行。层内适用并行性 — 在同一文本上并发运行多个分类器（例如 OpenAI 审核 + Llama Guard + Perspective）可隐藏每个分类器的延迟。作为可选优化，在输入审核完成且 token-1 流被延迟时，可显示占位响应（"稍等，正在检查..."）。标志行为可配置：拒绝、净化、升级到人工审核。

### 失败模式

- **仅输入。** 无法捕捉输出幻觉（课程 12-14 编码攻击绕过输入分类器）。
- **仅输出。** 允许任何输入到达模型；增加成本；向攻击者暴露内部推理。
- **仅自定义。** 跨类别不够稳健；正则表达式脆弱。

分层是默认配置。双重保险。

### Azure 弃用

Azure Content Moderator：2024 年 2 月弃用，2027 年 2 月退役。替换为 Azure AI Content Safety，后者基于 LLM 并与 Azure OpenAI 集成。对于 Azure 部署，迁移是一个 2024-2027 年的现场级项目。

### 在阶段 18 中的位置

课程 16 涵盖红队上下文中的审核工具。课程 29 涵盖部署审核。课程 30 以当前双重用途能力证据收尾。

## 使用它

`code/main.py` 构建一个三层审核工具：输入审核器（关键词 + 类别评分）、输出审核器（对输出运行相同分类器）、自定义审核器（领域规则）。你可以运行输入并观察哪层捕捉到什么。

## 交付它

本课产出 `outputs/skill-moderation-stack.md`。给定一个部署，它推荐审核栈配置：输入用哪个分类器，输出用哪个，自定义规则用什么，边缘情况用哪个评判器。

## 练习

1. 运行 `code/main.py`。通过所有三层运行一个良性、边界和有害输入。报告每层触发哪个。

2. 用 Perspective-API 风格的有害性评分扩展该工具，针对特定类别。比较其阈值行为与类别评分的行为。

3. 阅读 OpenAI 审核 API 文档和 Llama Guard 3 类别列表。将每个 OpenAI 类别映射到最接近的 Llama Guard 类别。识别三个不能干净映射的类别。

4. 为代码助手部署（例如 GitHub Copilot）设计一个审核栈。识别最相关和最不相关的类别，并提出自定义规则。

5. Azure Content Moderator 将于 2027 年 2 月退役。计划迁移到 Azure AI Content Safety。识别迁移中风险最高的元素。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|------------------------|
| OpenAI 审核 | "omni-moderation-latest" | 基于 GPT-4o 的 13 类别（文本）分类器，部分多模态支持 |
| Perspective API | "Google Jigsaw 有害性" | 前 LLM 时代有害性评分基准线 |
| Llama Guard | "MLCommons 14 类别" | Meta 的危害分类器（v3：8B 文本，8 语言；v4：12B 多模态） |
| 输入审核 | "生成前过滤器" | 在模型调用前对用户提示进行分类 |
| 输出审核 | "生成后过滤器" | 在传递前对模型输出进行分类 |
| 自定义审核 | "领域规则" | 部署特定规则（正则表达式、允许列表、策略） |
| 分层审核 | "三层都用" | 标准生产部署模式 |

## 进一步阅读

- [OpenAI 审核 API 文档](https://platform.openai.com/docs/api-reference/moderations) — omni-moderation 端点
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Llama Guard 仓库
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) — 有害性评分
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — Azure 替换方案
