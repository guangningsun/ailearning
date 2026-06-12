# 谈判与议价

> 智能体之间谈判资源、价格、任务分配和条款。2026 年的基准测试集已明确：NegotiationArena（arXiv:2402.05863）表明，通过人格操纵（"急迫感"），LLM 可以将收益提高约 20%；"Measuring Bargaining Abilities"（arXiv:2402.15813）表明买方比卖方更难，规模化也无济于事——其 **OG-Narrator**（确定性报价生成器 + LLM 叙述者）将成交率从 26.67% 提升至 88.88%；大规模自主谈判竞赛（arXiv:2503.06416）运行了约 18 万次谈判，发现 **思维链隐藏**（chain-of-thought-concealing）智能体通过向对手隐藏推理过程获胜；Bhattacharya 等人 2025 年基于哈佛谈判项目指标的研究中，Llama-3 最有效，Claude-3 最激进，GPT-4 最公平。本课实现 Contract Net Protocol（FIPA 的前身，见第 02 课），接入 LLM 风格的买方/卖方，运行 OG-Narrator 风格的任务分解，并测量每种结构性选择如何影响成交率。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置条件：** 阶段 16 · 02（FIPA-ACL 遗产）、阶段 16 · 09（并行蜂群网络）
**时间：** 约 75 分钟

## 问题

两个智能体需要就价格达成一致。如果仅依赖语言提示，2024-2026 年的 LLM 在紧密参数化的谈判中成交率低得惊人（arXiv:2402.15813 中约为 27%）。规模化无法解决这个问题：GPT-4 在议价能力上并没有比 GPT-3.5 结构性地更好；它在议价的*语言*上更好。

根本问题在于 LLM 混淆了两个职责——决定报价和叙述报价。OG-Narrator 将二者分离：确定性报价生成器计算数值走向；LLM 只负责叙述。成交率跃升至约 89%。

这呼应了经典多智能体研究发现：将机制与通信层解耦能带来优势。Contract Net Protocol（FIPA，1996；Smith，1980）是参考任务市场机制。将 LLM 插入叙述槽，你就得到了一个现代 LLM 驱动任务市场。

## 概念

### Contract Net，一段话讲清

Smith 的 1980 年 Contract Net Protocol：**管理者**广播**招标（cfp）**；**投标者**回复包含其报价的**提议（propose）**消息；管理者选出获胜者，发送**接受提议（accept-proposal）**给获胜者，发送**拒绝提议（reject-proposal）**给失败者。获胜者执行工作。可选消息：**拒绝（refuse）**（投标者拒绝提议）。FIPA 将此规范化为 `fipa-contract-net` 交互协议。

### 为什么 OG-Narrator 获胜

"Measuring Bargaining Abilities of Language Models"（arXiv:2402.15813）观察到：

- LLM 经常违反议价规则（在荒谬的价位报价，忽略对方的 ZOPA）。
- 它们锚定能力差（接受糟糕的首报价；以策略性而非符号性的金额反报价）。
- 仅靠规模化无法修复这些。更大的模型用类似的策略错误产生更合理的语言。

OG-Narrator 分解：

```
           ┌──────────────────┐        ┌──────────────────┐
  state  → │ offer generator  │ price → │  LLM narrator    │ → message
           │  (deterministic)  │        │  (writes the     │
           │                  │        │   human-style    │
           └──────────────────┘        │   accompaniment) │
                                       └──────────────────┘
```

报价生成器是经典谈判策略：Rubinstein 议价模型、Zeuthen 策略，或简单的价格针锋相对。LLM 负责叙述。消息包含确定性价格和自然语言框架。

成交率跃升是因为：
- 价格保持在议价区间内。
- 锚定是策略性的，而非情绪性的。
- LLM 做它擅长的事：写作。

### NegotiationArena 发现

arXiv:2402.05863 提供了权威基准。主要发现：

- LLM 通过采用人格（"我迫切需要在周五前卖掉这个"）可以将收益提高约 20%——人格操纵是真实有效的策略。
- 公平/合作的智能体被对抗性智能体利用；防御需要明确的反姿态。
- 对称配对在约 40% 的基准场景中收敛到不公平结果。

这不是"LLM 不是好谈判者"。而是"LLM 谈判得太像人类了，包括那些可被利用的部分"。

### 思维链隐藏

大规模自主谈判竞赛（arXiv:2503.06416）跨多种 LLM 策略运行了约 18 万次谈判。获胜者对对手隐藏了它们的推理：

- 如果智能体将"我只愿降到 75 美元；我的保留价是 70 美元"打印到公开可见的草稿板上，对手就会读到。
- 获胜者私下计算策略；输出通道只包含报价和最少必要的叙述。

这是 1976 年经典博弈论（Aumann 关于理性与信息）的 2026 年回响：泄露你的私人估价会损失收益。LLM 不会凭直觉理解这一点，会开心地把保留价打进推理痕迹中，而这些对对手是可见的。

工程要点：将私人草稿板上下文与公共消息上下文分离。非可选。

### Bhattacharya 等人 2025——模型排名

在哈佛谈判项目指标（原则性谈判、BATNA 尊重、利益互惠）上：

- **Llama-3** 在达成交易（成交率 + 收益）方面最有效。
- **Claude-3** 是最激进的谈判者（高锚定，晚让步）。
- **GPT-4** 是最公平的（在配对中收益方差最小）。

这是 2025 年的快照。重点不是 2026 年 4 月哪个模型获胜——而是不同基础模型有持续的谈判风格。异构集成（第 15 课）将此作为多样性来源。

### 通过 Contract Net + LLM 进行任务分配

将 Contract Net 现代复用至 LLM 多智能体：

1. 管理者智能体将任务分解为单元。
2. 向工作智能体广播带任务描述的 `cfp`。
3. 每个工作器返回报价：`(price, eta, confidence)`，其中 price 可以是 token、计算单元或美元。
4. 管理者选出获胜者（单个或多个，取决于任务）并授予任务。
5. 被拒绝的工作器可以自由竞标其他任务。

这很好地扩展到超过 100 个工作器，因为协调是广播-响应式的，而非同步聊天。生产中使用：Microsoft Agent Framework 的编排模式，部分 LangGraph 实现。

### LLM-利益相关者交互谈判

NeurIPS 2024（https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf）引入了多方可评分游戏，带有**秘密分数**和**最低接受阈值**。每个利益相关者有私人效用；LLM 必须从消息中推断这些。这是从双方谈判到 N 方联盟形成的泛化。与具有异构工作器能力的生产任务市场相关。

### 叙述 vs 机制规则

在所有 2024-2026 年谈判基准测试中，一致的工程规则是：

> 让 LLM 叙述。不要让 LLM 计算报价。

如果报价需要是一个数字（价格、ETA、数量），从谈判状态中确定性生成它，并让 LLM 产生框架。如果报价需要是一个提案结构（任务分解、角色分配），让 LLM 起草，但发送前根据 schema 验证并约束检查。

## 构建它

`code/main.py` 实现：

- `ContractNetManager`、`ContractNetTask`、`Bid`——管理者 + 投标者，广播 cfp，收集提案，授予。
- `og_narrator_bargain(state, rng)`——OG-Narrator 买方：确定性 Zeuthen 风格向中点让步。
- `seller_response(state, rng)`——确定性卖方反报价策略（两种风格的结构性 ground truth）。
- `naive_llm_bargain(state, rng)`——模拟全 LLM 议价者：以高方差选价格，经常超出 ZOPA。
- 测量：在 1000 次试验中成交率，每次试验采样新的保留价。

运行：

```
python3 code/main.py
```

预期输出：naive-LLM 成交率约 65-75%；OG-Narrator 成交率约 85-95%；15-25 个百分点的差距是报价生成与叙述解耦的结构性优势。外加一个三投标者、一任务的 Contract Net 任务市场分配示例。

## 使用它

`outputs/skill-bargainer-designer.md` 设计一个议价协议：谁生成报价（确定性还是 LLM）、谁叙述、私人草稿板如何与公共消息分离、如何监控成交率。

## 交付它

生产议价清单：

- **分离草稿板。** 私人状态永远不进入对手的上下文。这是不可妥协的。
- **确定性报价生成。** 价格、数量、ETA：计算，不要提示。
- **验证所有传入报价。** 根据 schema 验证。拒绝超出 ZOPA 的报价在协议边界。
- **限制轮次。** 最多 3-5 轮；死锁时升级到调解人。
- **持续测量成交率和收益方差。** 成交率下降是一个症状——通常是提示漂移或对手侧攻击。
- **记录所有被拒绝的提案。** 带上确定性理由。对于 Contract Net 管理者，落败的投标者需要理解为什么。

## 练习

1. 运行 `code/main.py`。确认 OG-Narrator 在成交率上击败 naive-LLM。差距多大？
2. 实现**基于人格的收益改善**（arXiv:2402.05863）——买方仅在叙述中采用"本周迫切想买"的人格，报价生成器不变。成交率或收益会变化吗？
3. 实现思维链**隐藏**：维护一个不传递给对手的私人草稿板字符串。如果你不小心泄露了（通过交换通道模拟），会发生什么？
4. 将 Contract Net 扩展为带保留价的 N 投标者拍卖。当所有出价都超过保留价时，管理者如何在最低价和最高质量之间决定？你选择哪种授予规则，为什么？
5. 阅读 Bhattacharya 等人 2025 关于哈佛谈判项目指标的论文。实现两种不同风格的议价者（激进 vs 公平）。在对称和非对称配对下测量收益方差。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| Contract Net | "任务市场" | Smith 1980，FIPA 1996。cfp + propose + accept/reject。经典任务市场。 |
| ZOPA | "可能达成协议的区域" | 买方最高价与卖方最低价之间的重叠。区间外的报价无法成交。 |
| BATNA | "谈判协议的最佳替代方案" | 如果本次交易失败你的退路。设定你的保留价。 |
| OG-Narrator | "报价生成器 + 叙述者" | 解耦：确定性报价 + LLM 叙述。 |
| Zeuthen 策略 | "风险最小化让步" | 经典报价生成器，基于风险限制让步。 |
| Rubinstein 议价 | "交替报价均衡" | 无限期折扣谈判的博弈论模型。 |
| CoT 隐藏 | "隐藏你的推理" | arXiv:2503.06416 的获胜者保持私人草稿板；公共通道只显示报价。 |
| 人格操纵 | "情绪姿态" | arXiv:2402.05863：急迫/紧迫人格带来约 20% 的收益增长。 |

## 延伸阅读

- [NegotiationArena](https://arxiv.org/abs/2402.05863)——基准测试；人格操纵和利用发现
- [Measuring Bargaining Abilities of Language Models](https://arxiv.org/abs/2402.15813)——OG-Narrator 和买方难于卖方的结果
- [Large-Scale Autonomous Negotiation Competition](https://arxiv.org/abs/2503.06416)——约 18 万次谈判；思维链隐藏获胜
- [LLM-Stakeholders Interactive Negotiation (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf)——带秘密效用的多党可评分游戏
- [Smith 1980 — The Contract Net Protocol](https://ieeexplore.ieee.org/document/1675516)——经典机制，IEEE Transactions on Computers