# 对话状态追踪

> "我要北部便宜餐厅...其实改成中档...再加个意大利菜。"三个轮次，三次状态更新。DST 保持槽值字典同步以确保预订成功。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 5 · 17（聊天机器人）、阶段 5 · 20（结构化输出）
**时间：** 约 75 分钟

## 问题

在任务导向对话系统中，用户的目标被编码为一组槽值对：`{cuisine: italian, area: north, price: moderate}`。每个用户轮次都可能添加、更改或删除一个槽。系统必须读取整个对话并正确输出当前状态。

一个槽值出错，系统就会预订错误的餐厅、安排错误的航班或扣错款项。DST 是用户所说的和后端执行之间的关键环节。

为什么在 2026 年仍然重要，尽管有了 LLMs：

- 合规敏感领域（银行、医疗、航空预订）需要确定性的槽值，而非自由形式生成。
- 工具使用代理在调用 API 之前仍然需要槽解析。
- 多轮纠正确实比看起来难："其实不，改成周四。"

现代管道：经典 DST 概念 + LLM 提取器 + 结构化输出 guardrails。

## 概念

![DST：对话历史 → 槽值状态](../assets/dst.svg)

**任务结构。** 模式定义域（餐厅、酒店、出租车）及其槽（cuisine、area、price、people）。每个槽可以为空、用封闭集合中的值填充（price: {cheap, moderate, expensive}），或自由形式值（name: "The Copper Kettle"）。

**两种 DST 形式。**

- **分类。** 对于每个（槽，候选值）对，预测是/否。适用于封闭词汇槽。2020 年前的标准。
- **生成。** 给定对话，自由文本生成槽值。适用于开放词汇槽。现代默认。

**指标。** 联合目标准确率（JGA）—— 每个槽都正确的轮次比例。全有或全无。MultiWOZ 2.4 排行榜在 2026 年最高约为 83%。

**架构。**

1. **基于规则（槽正则 + 关键词）。** 窄领域的强基线。可调试。
2. **TripPy / BERT-DST。** 基于 BERT 编码的复制式生成。预 LLM 标准。
3. **LDST（LLaMA + LoRA）。** 指令微调的 LLM，带领域-槽提示。在 MultiWOZ 2.4 上达到 ChatGPT 级质量。
4. **无本体论（2024–26）。** 跳过模式；直接生成槽名和值。处理开放领域。
5. **提示 + 结构化输出（2024–26）。** 带 Pydantic 模式 + 约束解码的 LLM。5 行代码，生产就绪。

### 经典失败模式

- **跨轮次的共指。** "继续用第一个选项。"需要解析是哪个选项。
- **覆盖 vs 追加。** 用户说"加个意大利菜"。是替换 cuisine 还是追加？
- **隐式确认。** "好的 cool" —— 那是否接受了提供的预订？
- **纠正。** "其实改成 7 点。"必须更新 time 而不清除其他槽。
- **对先前系统话语的共指。** "是的，那个。"哪个"那个"？

## 构建

### 第 1 步：基于规则的槽提取器

参见 `code/main.py`。正则 + 同义词词典覆盖窄领域 70% 的规范表述：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

在规范词汇之外脆弱。适用于确定性槽确认。

### 第 2 步：状态更新循环

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

三个不变量：

- 永远不要重置用户未触及的槽。
- 显式否定（"不要 cuisine 了"）必须清除。
- 用户纠正（"其实..."）必须覆盖，而非追加。

### 第 3 步：LLM 驱动的 DST 与结构化输出

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic 保证一个有效的状态对象。没有正则，没有模式不匹配，没有幻觉的槽。

### 第 4 步：JGA 评估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校准：系统把所有槽都做对的轮次比例是多少？对于 MultiWOZ 2.4，2026 年顶级系统：80-83%。你的领域内系统在窄词汇上应该超过该水平，否则 LLM 基线就赢了你。

### 第 5 步：处理纠正

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

在检测到纠正时，覆盖最后更新的槽而非追加。没有 LLM 帮助很难做对。现代模式：始终让 LLM 从历史中重新生成整个状态，而非增量更新 —— 这自然处理纠正。

## 陷阱

- **全历史重新生成成本。** 让 LLM 每轮重新生成状态需要 O(n²) 总 token。限制历史或汇总旧轮次。
- **模式漂移。** 事后添加新槽会破坏旧训练数据。对模式进行版本控制。
- **大小写敏感。** "Italian" vs "italian" vs "ITALIAN" —— 处处规范化。
- **隐式继承。** 如果用户之前指定了"4 人"，新的不同时间请求不应清除 people。始终传递完整历史。
- **自由形式 vs 封闭集。** 名称、时间和地址需要自由形式槽； cuisine 和 area 是封闭的。在模式中混合两者。

## 使用

2026 技术栈：

| 场景 | 方法 |
|-----------|----------|
| 窄领域（一个或两个意图） | 基于规则 + 正则 |
| 宽领域，有标注数据 | LDST（LLaMA + LoRA 在 MultiWOZ 风格数据上） |
| 宽领域，无标注，生产就绪 | LLM + Instructor + Pydantic 模式 |
| 口语 / 语音 | ASR + 规范化器 + LLM-DST |
| 多领域预订流程 | 模式引导的 LLM，每个领域有独立的 Pydantic 模型 |
| 合规敏感 | 基于规则为主，LLM 备选带确认流程 |

## 交付

保存为 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## 练习

1. **简单。** 在 `code/main.py` 中为 3 个槽（cuisine、area、price）构建基于规则的状态追踪器。在 10 个手工制作的对话上测试。测量 JGA。
2. **中等。** 用 Instructor + Pydantic + 小型 LLM 处理相同数据集。比较 JGA。检查最难的轮次。
3. **困难。** 实现两者并路由：基于规则为主，当基于规则发出 <2 个高置信度槽时 LLM 备选。测量组合 JGA 和每轮推理成本。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| DST | 对话状态追踪 | 在对话轮次间维护槽值字典。 |
| 槽 | 用户意图的单位 | 后端需要的有名参数（cuisine、date）。 |
| 领域 | 任务区域 | 餐厅、酒店、出租车 —— 槽的集合。 |
| JGA | 联合目标准确率 | 每个槽都正确的轮次比例。全有或全无。 |
| MultiWOZ | 基准 | 多领域 WOZ 数据集；标准 DST 评估。 |
| 无本体论 DST | 无模式 | 直接生成槽名和值，没有固定列表。 |
| 纠正 | "其实..." | 覆盖先前填充的槽的轮次。 |

## 延伸阅读

- [Budzianowski 等 (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — 规范基准。
- [Feng 等 (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — LLaMA + LoRA DST 指令微调。
- [Heck 等 (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — 复制式 DST 主力。
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — 基于 EM 的无监督 TOD。
- [MultiWOZ 排行榜](https://github.com/budzianowski/multiwoz) — 规范 DST 结果。