# 直接偏好优化家族

> Rafailov et al.（2023）证明了 RLHF 的最优解在偏好数据意义上有一个封闭形式，所以你可以跳过显式奖励模型直接优化策略。这个洞见证了一系列方法——IPO、KTO、SimPO、ORPO、BPO——每一个都修复了 DPO 的一个失败模式。在 2026 年，直接对齐算法比 PPO 部署了更多前沿后训练运行。但课程 2 的过度优化曲线仍然适用：DAAs 不能逃离古德hart，它只是移动了它咬人的位置。

**类型：** 学习型
**语言：** Python（标准库，六变体偏好损失比较器）
**前置条件：** 阶段 18 · 01（InstructGPT）、阶段 18 · 02（奖励黑客）、阶段 10 · 08（DPO 基础）
**时间：** 约 75 分钟

## 学习目标

- 从带 KL 的 RLHF 最优解推导 DPO 封闭形式。
- 陈述 IPO、KTO、SimPO、ORPO、BPO 各自修复了 DPO 的哪个失败模式。
- 区分"隐式奖励差距"与"偏好强度"，并解释为什么 IPO 的恒等映射很重要。
- 解释为什么 Rafailov et al.（NeurIPS 2024）证明 DAAs 即使没有显式 RM 也会过度优化。

## 问题

RLHF 目标（课程 1）：

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

有一个已知最优解：

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

所以奖励被最优策略与参考的比值隐式定义：

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

将其代入 Bradley-Terry 偏好似然，partition function `Z(x)` 消掉，因为它只依赖 `x`。剩下的是一个只包含策略参数的损失——不需要奖励模型。这就是 DPO。

麻烦的是：推导假设最优解可达、偏好数据在分布内、参考策略是真值模态锚点。这些没有一个完全成立。每个家族成员都修复了一个不同的违反假设。

## 概念

### DPO（Rafailov et al., 2023）

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

可能出问题的地方：

- 隐式奖励差距 `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` 是无界的。一个微小的偏好可以产生任意大的差距。
- 损失驱动选中和拒绝的对数概率在相反方向。它可以将选中的绝对对数概率往下推，只要拒绝降得更快。这就是退化选中回复现象。
- 分布外的偏好（罕见-罕见对 vs 罕见-罕见对）产生任意的隐式奖励。

### IPO（Azar et al., 2024）

身份偏好优化将 log-sigmoid 替换为偏好概率上的恒等映射。损失变成有界目标上的平方误差：

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

边缘由 `1/(2 beta)` 有界。偏好强度和隐式奖励差距成正比。不会爆炸。

### KTO（Ethayarajh et al., 2024）

Kahneman-Tversky 优化完全抛弃成对结构。给定单个标记输出和二元的"可取"或"不可取"信号，它映射到前景理论效用：

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

对收益和损失有不同的权重（损失厌恶）。好处：你可以使用非成对数据，这要多得多。

### SimPO（Meng et al., 2024）

简单偏好优化使训练信号与生成对齐。完全移除参考策略，用长度归一化对数似然：

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

用边缘 `gamma` 稳定。长度归一化移除了利用 DPO 长度偏见过失模式的动机（更长的 `y_w` 根据构造给出更大的对数概率差距）。

### ORPO（Hong et al., 2024）

赔率比偏好优化将偏好项添加到标准 SFT 负对数似然：

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

没有参考策略——SFT 项就是正则化器。从基础模型到对齐模型单阶段训练。不需要单独的 SFT 检查点。

### BPO（ICLR 2026 提交，OpenReview id=b97EwMUWu7）

识别退化选中回复问题：DPO 保持排序 `y_w > y_l`，但 `y_w` 的绝对对数概率可以下降。BPO 添加了一行校正，惩罚选中回复上的向下移动。在 Llama-3.1-8B-Instruct 数学推理上比 DPO 报告 +10.1% 准确率。

### 普遍结果：DAAs 仍然过度优化

Rafailov et al. "Direct Alignment Algorithms 中奖励模型过度优化的缩放定律"（NeurIPS 2024）在多个数据集上用 DPO、IPO、SLiC 训练策略，跨 KL 预算。金奖励-vs-KL 曲线具有相同的 Gao et al. 峰和崩溃形状。隐式奖励查询在训练期间是分布外的样本；KL 正则化不能稳定这一点。

DAAs 不能逃离古德hart。它们改变了它咬人的表面，从"奖励模型过度优化"变为"参考策略比过度优化"。通用修复——更好的数据、集成、早停——对两者都适用。

### 在它们之间选择（2026）

- 如果你有大的成对偏好数据：保守 beta 的 DPO，如果长度偏差明显用 SimPO。
- 如果你有非成对二元反馈：KTO。
- 如果你想要从基础模型开始的单阶段流水线：ORPO。
- 如果在 DPO 日志中看到退化选中对数概率：BPO。
- 如果偏好强度变化很大且 DPO 饱和：IPO。

每个实验室都在一套电池上运行全部五种方法，按任务选赢者。没有理由数学推理和安全的最优解是相同的。

## 动手实现

`code/main.py` 在一个 toy 偏好数据集上比较六种损失（DPO、IPO、KTO、SimPO、ORPO、BPO），其中真实偏好强度因对而异。每种损失在相同的 500 对样本上用小 softmax 策略优化。绘制每种方法的最终胜率、选中-对数概率漂移和隐式奖励散布。

## 交付物

本课产出 `outputs/skill-preference-loss-selector.md`。给定数据集统计（成对 vs 非成对、可变 vs 均匀偏好强度、长度分布）和目标（单阶段或 SFT-然后-偏好），推荐一种偏好损失并报告它防护的失败模式。

## 练习

1. 运行 `code/main.py`。报告 DPO 和 BPO 的最终选中-对数概率下降。BPO 应该保持更高的选中绝对概率——验证这一点。

2. 修改偏好数据，使所有对具有相等强度。六种方法中哪种最鲁棒？哪种退化？解释这里 IPO 的优势。

3. 使被拒绝的回复平均比选中长 2 倍。在不改变其他任何东西的情况下，数值展示 DPO 的长度利用和 SimPO 的修复。

4. Rafailov et al.（NeurIPS 2024）声称 DAAs 过度优化。重现一个单点版本：绘制选中减拒绝的 KL 散度并观察 DPO 在大 beta 下的过度优化。

5. 阅读 BPO 论文摘要（OpenReview b97EwMUWu7）。写下 BPO 添加到 DPO 的一行校正。确认 against `code/main.py` 中的实现。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| DPO | "无奖励模型的 RLHF" | 从封闭形式 RLHF 最优解导出的损失；只有策略参数 |
| 隐式奖励 | "对数比" | `beta * log(pi(y|x) / pi_ref(y|x))` —— DPO 隐含的奖励 |
| IPO | "有界 DPO" | 用恒等映射替换 log-sigmoid；隐式奖励差距被 `1/(2 beta)` 上限 |
| KTO | "非成对 DPO" | 在带损失厌恶的单个标签上的前景理论效用 |
| SimPO | "无参考 DPO" | 长度归一化对数似然 + 边缘；没有参考策略 |
| ORPO | "单阶段 DPO" | NLL + 赔率比偏好项；从基础模型一次通过训练 |
| BPO | "保选 DPO" | DPO 加上惩罚选中回复绝对对数概率下降的项 |
| 退化选中 | "选中的下降" | DPO 降低选中对数概率，只要拒绝降得更快 |
| DAA | "直接对齐算法" | 任何跳过显式 RM 的偏好损失方法 |

## 延伸阅读

- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar et al. — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) —— IPO
- [Ethayarajh et al. — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov et al. — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)