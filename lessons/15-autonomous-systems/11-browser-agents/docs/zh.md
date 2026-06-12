# 浏览器代理与长程 Web 任务

> ChatGPT 代理（2025 年 7 月）将 Operator 和深度研究合并为一个浏览器/终端代理，在 BrowseComp 上达到 68.9% 的 SOTA。OpenAI 于 2025 年 8 月 31 日关闭了 Operator——产品层面的整合。Anthropic 的 Vercept 收购使 Claude Sonnet 在 OSWorld 上从 15% 以下提升到 72.5%。WebArena-Verified（ServiceNow，ICLR 2026）修复了原始 WebArena 中 11.3 个百分点的假阴性率，并交付了 258 个任务的 Hard 子集。这些数字是真实的。攻击面也是真实的：OpenAI 的准备状态负责人在公开场合表示，浏览器代理的间接提示注入"不是可以完全修补的 bug"。已记录的 2025–2026 攻击：Tainted Memories（Atlas CSRF）、HashJack（Cato Networks），以及 Perplexity Comet 中的一键劫持。

**类型：** 学习型
**语言：** Python（标准库、间接提示注入攻击面模型）
**前置条件：** 阶段 15 · 10（权限模式）、阶段 15 · 01（长程代理）
**时间：** 约 45 分钟

## 问题

浏览器代理是读取不受信任内容并采取有影响操作的长程代理。代理访问的每个页面都是用户未写的输入。每个页面上每个表单都是一个潜在命令通道。2025–2026 攻击语料库表明这不是假设性的：Tainted Memories 允许攻击者通过精心制作的页面将恶意指令绑定到代理的内存；HashJack 在代理访问的 URL 片段中隐藏命令；Perplexity Comet 劫持在一键中发生。

防御图景令人不安。OpenAI 的准备状态负责人说出了不为人说的部分：间接提示注入"不是可以完全修补的 bug"。这是因为攻击存在于代理的读取与行动边界，这在架构上是模糊的——模型读取的每个 token 原则上都可以被读取为指令。

本课命名攻击面，命名基准格局（BrowseComp、OSWorld、WebArena-Verified），并建模一个最小间接提示注入场景，以便你可以在第 14 和 18 课中推理真实防御。

## 概念

### 2026 格局，每个系统一段

**ChatGPT 代理（OpenAI）。** 2025 年 7 月推出。统一了 Operator（浏览）和深度研究（多小时研究）。2025 年 8 月 31 日关闭了独立 Operator。在 BrowseComp 上达到 68.9% 的 SOTA；在 OSWorld 和 WebArena-Verified 上数据强劲。

**Claude Sonnet + Vercept（Anthropic）。** Anthropic 的 Vercept 收购专注于计算机使用能力。使 Claude Sonnet 在 OSWorld 上从 <15% 提升到 72.5%。Claude Computer Use 作为工具 API 交付。

**Gemini 3 Pro 与浏览器使用（DeepMind）。** 浏览器使用集成交付计算机使用控制；FSF v3（2026 年 4 月，第 20 课）专门在 ML 研发领域跟踪自主性。

**WebArena-Verified（ServiceNow，ICLR 2026）。** 修复了一个有充分文档化的问题：原始 WebArena 约有 11.3% 的假阴性率（标记为失败的任务实际上是已解决的）。Verified 版本使用人工策划的成功标准重新评分，并添加了 258 个任务的 Hard 子集（ICLR 2026 论文，openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp vs OSWorld vs WebArena

| 基准 | 测量内容 | 时间跨度 |
|---|---|---|
| BrowseComp | 在有时间压力下在开放网络上查找特定事实 | 分钟级 |
| OSWorld | 代理操作完整桌面（鼠标、键盘、shell） | 十分钟级 |
| WebArena-Verified | 在模拟网站中的交易性 Web 任务 | 分钟级 |
| Hard 子集 | 具有多页面状态转换的 WebArena-Verified 任务 | 十分钟级 |

不同的轴。高 BrowseComp 分数表示代理能找到事实；不表示代理能预订航班。OSWorld 分数更接近"它在我的桌面上能工作吗"。WebArena-Verified 更接近"它能完成一个流程吗"。任何生产决策都需要与任务分布相匹配的基准。

### 攻击面命名

1. **间接提示注入。** 不受信任的页面内容包含指令。代理读取它们。代理执行它们。公开例子：2024 Kai Greshake 等人，2025 Tainted Memories 论文，2026 HashJack（Cato Networks）。
2. **URL 片段/查询注入。** 爬取的 URL 的 `#fragment` 或查询字符串包含命令。从不直观渲染；仍在代理的上下文中。
3. **内存绑定攻击。** 页面指示代理写入持久内存（第 12 课涵盖持久状态）。下一会话，内存以无可见触发器触发负载。
4. **经过身份验证会话上的 CSRF 形态攻击。** Tainted Memories 类：代理在某处登录；攻击者的页面发出状态更改请求，代理使用用户的 cookie 执行。
5. **一键劫持。** 视觉上无害的按钮搭乘代理跟随的有效负载。Comet 类。
6. **代理主机表面上的内容安全策略漏洞。** 渲染和工具层本身可以是攻击向量；浏览器中的浏览器代理堆栈很宽。

### 为什么"不能完全修补"

攻击与代理的能力是同构的。代理必须读取不受信任的内容才能完成工作。代理读取的任何内容都可能包含指令。代理遵循的任何指令都可能与用户的实际请求不对齐。防御（信任边界、分类器、工具允许列表、对有影响操作 HITL）提高了攻击成本并减少了其爆炸半径。它们不会关闭此类。

这与 Lob 定理的推理模式相同（第 8 课）：代理无法证明下一个 token 是安全的；它只能建立一个使不安全 token 更可检测的系统。

### 实际交付的防御姿态

- **读/写边界。** 读取从不是有影响的。写入（提交表单、发布内容、调用有副作用的工具）如果发起的内容来自信任边界之外，则需要新的人工批准。
- **每个任务的工具允许列表。** 代理可以浏览；除非该工具明确为任务启用，否则不能发起电汇。第 13 课涵盖预算。
- **会话隔离。** 浏览器代理会话仅使用范围受限的凭据运行。无生产身份验证，无个人邮箱。保留每个 HTTP 请求的日志以供审计。
- **内容清理器。** 获取的 HTML 在连接到模型上下文之前剥离已知恶意模式。（减少简单攻击；不能阻止复杂有效负载。）
- **对有影响操作 HITL。** 提出然后提交模式（第 15 课）。
- **内存上的金丝雀令牌。** 如果内存条目被触发，用户会看到它（第 14 课）。

## 使用它

`code/main.py` 对三个合成页面建模一个小浏览器代理运行。一个页面是良性的，一个在可见文本中有直接提示注入blob，一个有 URL 片段注入（不可见但在代理上下文中）。脚本显示（a）天真代理会做什么，（b）读/写边界捕获什么，（c）清理器捕获什么，（d）两者都不捕获什么。

## 交付它

`outputs/skill-browser-agent-trust-boundary.md` 界定提议的浏览器代理部署：它触及哪些信任区域，它被授权写入什么，以及在第一次运行之前必须到位的防御。

## 练习

1. 运行 `code/main.py`。识别清理器捕获但读/写边界不捕获的攻击，以及仅读/写边界捕获的攻击。

2. 扩展清理器以检测一类 HashJack 风格 URL 片段注入。在带有合法片段的良性 URL 上测量误报率。

3. 选择一个你知道的真实浏览器代理工作流（例如"预订航班"）。列出每次读取和每次写入。标记哪些写入需要 HITL 及其原因。

4. 阅读 WebArena-Verified ICLR 2026 论文。识别原始 WebArena 评分不可靠的一类任务，并解释 Verified 子集如何解决它。

5. 为浏览器代理设置设计内存金丝雀。你会存储什么，在哪里，什么触发警报？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|---|---|---|
| 间接提示注入 | "恶意页面文本" | 代理读取的页面中不受信任的内容包含代理执行的指令 |
| Tainted Memories | "内存攻击" | 代理将攻击者提供的指令写入持久内存；下一会话触发 |
| HashJack | "URL 片段攻击" | 隐藏在 URL 片段/查询字符串中的有效负载在代理上下文中但未直观渲染 |
| 一键劫持 | "恶意按钮" | 可见功能搭乘代理执行的后续有效负载 |
| BrowseComp | "Web 搜索基准" | 在开放网络上查找特定事实；分钟级时间跨度 |
| OSWorld | "桌面基准" | 完整操作系统控制；多步 GUI 任务 |
| WebArena-Verified | "固定的 Web 任务基准" | ServiceNow 的重新评分 WebArena，带 Hard 子集 |
| 读/写边界 | "副作用门控" | 读取从不是有影响的；如果内容超出信任则写入需要新批准 |

## 延伸阅读

- [OpenAI——介绍 ChatGPT 代理](https://openai.com/index/introducing-chatgpt-agent/) — Operator 和深度研究的合并；BrowseComp SOTA。
- [OpenAI——计算机使用代理](https://openai.com/index/computer-using-agent/) — Operator 血统和成为 ChatGPT 代理的架构。
- [Zhou 等人——WebArena](https://webarena.dev/) — 原始基准。
- [WebArena-Verified（OpenReview）](https://openreview.net/forum?id=94tlGxmqkN) — ICLR 2026 修复子集论文。
- [Anthropic——在实践中衡量代理自主性](https://www.anthropic.com/research/measuring-agent-autonomy) — 包括计算机使用代理的攻击面讨论。