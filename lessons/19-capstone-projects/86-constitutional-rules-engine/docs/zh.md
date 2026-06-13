# Capstone 86 — 宪政规则引擎

> 规则由名称、谓词和解释三部分组成。缺少任何一个都只是感觉，不是规则。

**类型：** 构建
**语言：** Python、YAML
**前置条件：** 阶段 18 安全课程、阶段 19 Track A 课程 25-29
**时间：** 约 90 分钟

## 问题

分类器覆盖可识别的失败。规则引擎覆盖合同性的。一个编写代码助手的团队想要一个约束，比如"每个包含代码的响应必须以可运行块或声明的假设结尾"。运行客户支持机器人的团队想要"每个拒绝必须提供下一步"。这些约束不是自然的分类器目标。它们是关于响应、对话和系统策略的谓词，需要非工程师也能阅读。

诚实的表示是一份声明式文件。宪政文件与代码一起存在于 YAML 中，进入版本控制，有独立的审查流程。每条规则有一个 `name`、`predicate`、`severity` 和 `explanation` 模板。引擎加载文件，对候选输出评估每条规则，并返回每个触发的规则的结构化 `Violation`。本 capstone 的规则引擎通过 `all_of`、`any_of` 和 `not_` 组合谓词，因此一条规则可以表达"如果响应包含代码，它必须以可运行块结尾 AND 不引用仅内部库"。

这节课的另一半是修订。只阻止的规则引擎只完成了一半。只提出修复的规则引擎在操作上有用：助手起草响应，引擎标记违规，修复器产生修订后的响应，引擎确认修订满足规则。这节课交付一个最小的修复器（每条规则的正则替换）和一份结构化 diff（逐行添加、移除、编辑），在草稿和修订之间。

## 概念

```mermaid
flowchart LR
  D[草稿响应] --> RE[规则引擎]
  RE -->|违规| F[修复器]
  F --> R[修订后的响应]
  R --> RE2[规则引擎 第二轮]
  RE2 -->|裁决| OUT[接受或升级]
  D -.->|diff| R
```

规则的形式如下：

```yaml
- name: end-with-runnable-or-assumption
  severity: medium
  applies_when:
    contains_regex: '```python'
  must:
    any_of:
      - ends_with_regex: '```\s*$'
      - contains_regex: 'assumption:'
  explanation: "代码响应必须以闭合围栏或明确假设结尾。"
  fix:
    append_if_missing: "\n\nAssumption: example inputs are valid."
```

谓词是原子的：`contains_regex`、`not_contains_regex`、`ends_with_regex`、`starts_with_regex`、`max_words`、`min_words`。组合是 `all_of`、`any_of`、`not_`。引擎首先评估 `applies_when`；如果规则不适用，则违规记录为 `not_applicable`。否则引擎评估 `must` 并产生 `pass` 或 `violation`。

严重级别是 `low`、`medium`、`high`，与第 85 课一致。下游门控（第 87 课）将 `high` 规则违规与 `high` 分类器裁决同等对待：block。

修复器是声明式操作列表：`append_if_missing`、`prepend_if_missing`、`replace_regex`。每个操作通过名称将规则映射到变换。修复器故意限于局部编辑；结构重写属于单独的拒绝和帮助层，不在此覆盖。

diff 是相对于原始和修订计算的。它是一个带有 `op`（add、remove、edit）和相关文本的 `Change` 记录列表。下游门控可以记录 diff，以便人类审查员随时间审计修复器的行为。

## 构建

`code/rules.yml` 持有宪政文件。`code/main.py` 中的加载器接受 YAML 文件（当 PyYAML 可用时）或 JSON 文件（内置）。这节课发布一个 `rules.yml`，课程测试通过两种代码路径解析它。`code/main.py` 定义了 `Engine` 和 `Fixer` 类以及一个 `diff` 函数。组合通过 `any_of` 的短路求值进行递归求值。

发货的宪政文件：

- `no-empty-refusal`（medium）—— 拒绝必须包含建议或重定向之一
- `end-with-runnable-or-assumption`（medium）—— 代码响应必须干净关闭
- `no-pii-in-examples`（high）—— 示例数据不得包含邮箱或电话形状
- `cite-when-asserting-fact`（low）—— 以"According to"开头的行必须包含括号引用
- `no-internal-library-leak`（high）—— 词语 `internal-only` 和 `policybot-internal` 不得出现在输出中
- `bounded-length`（low）—— 响应不得超过 800 字

## 使用

`python3 main.py`。演示通过引擎运行三个草稿响应，打印违规，运行修复器，打印 diff，并写出 `outputs/rules_report.json`。一个 fixture 有一个不适用的规则（草稿中没有代码块），报告显示该规则的 `not_applicable`，以便团队看到引擎明确评估了它。

## 交付

`outputs/skill-constitutional-rules-engine.md` 记录了规则语法和修复器操作。

## 练习

1. 添加一条规则，要求每当提示词提到安全时，响应必须包含短语"If this is urgent"。使用组合。
2. 用模板修复器替换正则修复器，该修复器接受命名槽。演示在新设计下重写的一条规则。
3. 添加一个指标端点，给定一个草稿语料库，返回每条规则的违规率，以便团队看到哪条规则过度触发。

## 关键术语

| 术语 | 常见说法 | 精确含义 |
|---|---|---|
| 宪政文件 | 一个模糊的策略文档 | 一份包含谓词、严重级别和解释的规则 YAML 文件 |
| 谓词 | 一个检查 | 从文本到布尔值或组合（通过 all_of/any_of/not_）的可调用对象 |
| 违规 | 一个失败 | 带有规则名称、严重级别、解释和匹配 span 的结构化记录 |
| 修复器 | 一个模型微调 | 一个确定性每规则变换，将草稿映射到修订 |
| diff | 一个字符串比较 | 草稿和修订之间的 add、remove、edit 操作的结构化列表 |

## 延伸阅读

第 87 课将此引擎与输入侧检测器和输出侧分类器组合成单一安全门控。