# 基准测试：WebArena 与 OSWorld

> WebArena 测试跨四个自托管应用的 web 智能体能力。OSWorld 测试跨 Ubuntu、Windows、macOS 的桌面智能体能力。在发布时（2023–2024），两者都显示了最佳智能体与人类之间的巨大差距。这个差距正在缩小；失败模式没有改变。

**类型：** 学习型
**语言：** Python（标准库）
**前置条件：** 阶段 14 · 19（SWE-bench、GAIA）
**时间：** 约 60 分钟

## 学习目标

- 描述 WebArena 的四个自托管应用以及为什么基于执行结果的评估很重要。
- 解释为什么 OSWorld 使用真实操作系统截图而不是无障碍 API。
- 说出 OSWorld 的两个主要失败模式：GUI 定位和操作知识。
- 总结 OSWorld-G 和 OSWorld-Human 在基准之上增加了什么。

## 问题

通用智能体可以调用工具。它们能驱动浏览器完成 20 次点击的购物结账吗？它们能仅用键盘和鼠标配置一台 Linux 机器吗？这些就是 WebArena 和 OSWorld 回答的问题。

## 概念

### WebArena（Zhou 等，ICLR 2024）

- 横跨四个自托管 web 应用的 812 个长期任务：一个购物网站、一个论坛、一个类 GitLab 的开发工具、一个业务 CMS。
- 加上工具：地图、计算器、记事本。
- 评估基于 gym API 执行——订单是否下了，issue 是否关闭了，CMS 页面是否更新了？
- 发布时：最佳 GPT-4 智能体 14.41% 成功率 vs 人类 78.24%。

自托管的意义在于——基准不会因为目标应用不稳定而出现波动，它们是固定且可复现的。

### 扩展

- **VisualWebArena** —— 视觉接地任务，成功取决于对图像（截图作为一级观察）的解读。
- **TheAgentCompany**（2024 年 12 月）—— 增加了终端 + 编程；更像是真实的远程工作环境。

### OSWorld（Xie 等，NeurIPS 2024）

- 跨 Ubuntu、Windows、macOS 的 369 个真实计算机任务。
- 对真实应用的自由形式键盘和鼠标控制。
- 1920×1080 截图作为观察。
- 发布时：最佳模型 12.24% vs 人类 72.36%。

### 主要失败模式

1. **GUI 定位。** 像素 → 元素映射。模型难以在 1920×1080 中可靠地定位 UI 元素。
2. **操作知识。** 哪个菜单有设置，哪个键盘快捷键，哪个偏好设置面板。这是人类经过多年积累才建立的知识尾部。

### 后续工作

- **OSWorld-G** —— 564 样本的定位套件 + Jedi 训练集。将定位从规划中分解出来，这样你可以分别测量它们。
- **OSWorld-Human** —— 人工策划的黄金动作轨迹。显示顶级智能体使用了必要步数的 1.4–2.7 倍（轨迹效率差距）。

### 为什么这很重要

Claude computer use、OpenAI CUA、gemini 2.5 Computer Use（第 21 课）都在 WebArena 和 OSWorld 塑造的工作负载上进行训练。基准是目标；生产模型是交付的答案。

### 基准测试会出错的地方

- **仅截图评估。** OSWorld 是截图驱动的；如果评估一个使用 DOM 或无障碍 API 的智能体在 OSWorld 上跑，就错过了定位挑战。
- **忽略轨迹长度。** 只评分成功率会遗漏 OSWorld-Human 揭示的 1.4–2.7 倍步数低效。
- **过时的自托管应用。** WebArena 的应用固定了特定版本；不重新策划就升级会破坏可比性。

## 构建它

`code/main.py` 实现了一个玩具 web 智能体框架：

- 一个最简"购物应用"状态机：list_items、add_to_cart、checkout。
- 3 个任务的黄金轨迹。
- 一个尝试每个任务的脚本化智能体。
- 基于执行的评估器（状态检查）和轨迹效率指标（步数 vs 黄金）。

运行：

```
python3 code/main.py
```

输出：每个任务的成功率和轨迹效率，镜像 OSWorld-Human 的方法论。

## 使用它

- **WebArena Verified** 在内部集群上自托管，用于持续评估。
- **OSWorld** 在 VM 集群上用于桌面智能体。
- **计算机使用智能体**（第 21 课）—— Claude、OpenAI CUA、gemini —— 都在类似这些的工作负载上训练。
- **你自己的产品流程** —— 为你的 Top 20 任务捕获黄金轨迹；每周用智能体跑它们。

## 交付它

`outputs/skill-web-desktop-harness.md` 构建一个带有基于执行的评估和轨迹效率指标的 web/桌面智能体框架。

## 练习

1. 用第二个应用（论坛）扩展玩具框架。写 3 个任务加黄金轨迹。
2. 添加每个任务的轨迹效率报告。在你的玩具上，智能体是 1x、2x 还是 3x 于黄金？
3. 实现一个"干扰项"工具——黄金轨迹从不使用的一个。脚本化智能体会被诱惑吗？
4. 阅读 OSWorld-G。在你自己的评估中，你如何将定位失败与规划失败分开？
5. 阅读 WebArena 的应用 README。升级其中一个固定应用版本会破坏什么？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| WebArena | "Web 智能体基准" | 跨 4 个自托管应用的 812 个任务；gym 风格评估 |
| VisualWebArena | "视觉 WebArena" | 视觉接地的 WebArena；截图是观察 |
| OSWorld | "桌面智能体基准" | 369 个任务，运行在真实 Ubuntu/Windows/macOS 上 |
| GUI 定位（GUI grounding） | "像素到元素映射" | 模型在 1920x1080 中定位 UI 元素 |
| 操作知识（Operational knowledge） | "操作系统知识" | 哪个菜单，哪个快捷键，哪个偏好设置面板 |
| OSWorld-G | "定位套件" | 564 个仅定位的样本 + 训练集 |
| OSWorld-Human | "黄金轨迹" | 人工策划的专家动作序列，用于测量效率 |
| 轨迹效率（Trajectory efficiency） | "步数除以黄金" | 智能体步数除以人类最小步数 |

## 延伸阅读

- [Zhou 等，WebArena（arXiv:2307.13854）](https://arxiv.org/abs/2307.13854) —— 四应用 web 基准
- [Xie 等，OSWorld（arXiv:2404.07972）](https://arxiv.org/abs/2404.07972) —— 跨 OS 桌面基准
- [Anthropic，Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) —— Claude 的基准塑造能力
- [OpenAI，Computer-Using Agent](https://openai.com/index/computer-using-agent/) —— OSWorld 和 WebArena 数字