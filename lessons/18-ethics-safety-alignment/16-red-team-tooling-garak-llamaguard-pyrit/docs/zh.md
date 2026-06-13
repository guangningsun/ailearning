# 红队工具链 — Garak、Llama Guard、PyRIT

> 三大生产工具构成了 2026 年的红队技术栈。Llama Guard（Meta）—— 基于 Llama-3.1-8B 微调的分类器，在 14 个 MLCommons 危险类别上训练；2025 年的 Llama Guard 4 是一个 12B 原生多模态分类器，从 Llama 4 Scout 剪枝而来。Garak（NVIDIA）—— 开源 LLM 漏洞扫描器，支持静态、动态和自适应探测，覆盖幻觉、数据泄露、提示注入、毒害和越狱。PyRIT（Microsoft）—— 多轮红队演练，支持 Crescendo、TAP 和自定义转换链，实现深度利用。Llama Guard 3 记录在 Meta 的 "Llama 3 Herd of Models"（arXiv:2407.21783）中；Llama Guard 3-1B-INT4 在 arXiv:2411.17713 中；Garak 的探测架构在 github.com/NVIDIA/garak。这些工具是 2026 年红队研究（课程 12-15）与部署（课程 17+）之间的生产接口。

**类型：** 构建型
**语言：** Python（标准库、工具架构模拟器和 Llama Guard 风格分类器模拟）
**前置条件：** 阶段 18 · 课程 12-15（越狱和IPI）
**时间：** 约 75 分钟

## 学习目标

- 描述 Llama Guard 3/4 在安全技术栈中的位置：输入分类器、输出分类器，或两者兼有。
- 说出 14 个 MLCommons 危险类别，并指出一个非显而易见的类别（代码解释器滥用）。
- 描述 Garak 的探测架构：探测器、检测器、测试架。
- 描述 PyRIT 的多轮演练结构及其与 Garak 探测器的组合方式。

## 问题

课程 12-15 展示了攻击面。生产部署需要可重复、可扩展的评估。2026 年有三大工具主导：Llama Guard（防御分类器）、Garak（扫描器）、PyRIT（演练编排器）。每个工具针对红队生命周期的不同层面。

## 概念

### Llama Guard（Meta）

Llama Guard 3 是一个基于 Llama-3.1-8B 微调的模型，在 MLCommons AILuminate 14 个类别上进行输入/输出分类：
- 暴力犯罪、非暴力犯罪、性相关、CSAM、诽谤
- 专业建议、隐私、知识产权、无差别武器、仇恨
- 自杀/自残、性内容、选举、代码解释器滥用

支持 8 种语言。用法：放在 LLM 之前（输入审核）、放在 LLM 之后（输出审核），或两者兼用。两种用法产生不同的训练分布——Llama Guard 3 作为单一模型处理两者。

Llama Guard 3-1B-INT4（arXiv:2411.17713，440MB，在移动 CPU 上约 30 tokens/s）是量化边缘版本。

Llama Guard 4（2025 年 4 月）是 12B，原生多模态，从 Llama 4 Scout 剪枝。它用一个能摄入文本+图像的分类器替代了之前的 8B 文本和 11B 视觉两个前辈分类器。

### Garak（NVIDIA）

开源漏洞扫描器。架构：
- **探测器。** 用于幻觉、数据泄露、提示注入、毒害和越狱的攻击生成器。静态（固定提示）、动态（生成提示）、自适应（响应目标输出）。
- **检测器。** 根据预期失败模式对输出评分——有毒、泄露、越狱。
- **测试架。** 管理探测器-检测器配对、运行演练、生成报告。

TrustyAI 将 Garak 与 Llama-Stack 护盾（Prompt-Guard-86M 输入分类器、Llama-Guard-3-8B 输出分类器）集成，用于端到端护盾-目标评估。基于严重性等级评分（TBSA）替代二元通过/失败——模型可以在同一探测器的严重性等级 3 通过但在严重性等级 5 失败。

### PyRIT（Microsoft）

Python 风险识别工具包。多轮红队演练。围绕以下构建：
- **转换器。** 转换种子提示——释义、编码、翻译、角色扮演。
- **编排器。** 运行演练：Crescendo（ escalation）、TAP（ branching）、RedTeaming（自定义循环）。
- **评分。** LLM 作为评判或分类器作为评判。

PyRIT 是 Garak 的重量级表亲。Garak 运行数千个单轮探测器；PyRIT 运行深度多轮演练，旨在打破特定失败模式。

### 技术栈

在模型两侧部署 Llama Guard。每晚运行 Garak 进行回归测试。在发布前运行 PyRIT 进行演练。这 是 2026 年大多数生产部署的默认配置。

### 评估陷阱

- **评判身份。** 三个工具都可以使用 LLM 评判；评判校准驱动报告的 ASR（课程 12）。请与工具一起指定评判。
- **探测器老化。** 模型针对探测器打补丁后，Garak 探测器会老化。自适应探测器（PAIR 风格）比静态探测器老化得更慢。
- **Llama Guard 对良性内容的误报率。** 早期 Llama Guard 版本过度标记政治和 LGBTQ+ 内容；Llama Guard 3/4 的校准有所改进，但未针对每个部署进行校准。

### 在阶段 18 中的位置

课程 12-15 是攻击家族。课程 16 是生产工具。课程 17（WMDP）是双重用途能力的评估。课程 18 是将这些工具包装在策略结构中的前沿安全框架。

## 使用它

`code/main.py` 构建了一个玩具 Llama Guard 风格分类器（14 个类别上的关键词+语义特征）、一个玩具 Garak 测试架（探测器-检测器循环）和一个 PyRIT 风格的多轮转换链。你可以对模拟目标运行这三个工具并观察不同的覆盖特征。

## 交付它

本课程产出 `outputs/skill-red-team-stack.md`。给定一个部署描述，它命名三个工具中哪些是合适的、在每个中配置什么，以及运行什么回归节奏。

## 练习

1. 运行 `code/main.py`。比较 Llama Guard 风格分类器在单轮与多轮攻击中的检测率。

2. 实现一个新的 Garak 探测器：base64 编码的有害请求。用 Llama Guard 风格分类器测量其检测情况。

3. 扩展 PyRIT 风格的转换链，添加"翻译成法语，然后释义"转换器。重新测量攻击成功率。

4. 阅读 Llama Guard 3 的危险类别列表。识别两个训练数据在合法开发者内容上可能产生高假阳性率的类别。

5. 比较 Garak 和 PyRIT 的设计原则。为每个工具选择合适的部署场景进行论证。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|------------------------|
| Llama Guard | "分类器" | 基于 Llama-3.1-8B/4-12B 微调的 14 个危险类别安全分类器 |
| Garak | "扫描器" | NVIDIA 开源漏洞扫描器；探测器、检测器、测试架 |
| PyRIT | "演练工具" | Microsoft 多轮红队编排器；转换器、编排器、评分 |
| Prompt-Guard | "小分类器" | Meta 的 86M 提示注入分类器，与 Llama Guard 配对 |
| TBSA | "基于等级评分" | Garak 的基于等级通过/失败，替代二元结果 |
| 转换链 | "释义+编码+..." | PyRIT 组合原语，用于构建多步攻击 |
| MLCommons 危险类别 | "14 个分类法" | Llama Guard 面向的行业标准分类法 |

## 延伸阅读

- [Meta — Llama Guard 3（收录于 Llama 3 Herd 论文，arXiv:2407.21783）](https://arxiv.org/abs/2407.21783) —— 8B 分类器
- [Meta — Llama Guard 3-1B-INT4（arXiv:2411.17713）](https://arxiv.org/abs/2411.17713) —— 量化移动分类器
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak) —— 扫描器仓库和文档
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT) —— 演练工具包