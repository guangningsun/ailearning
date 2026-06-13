# 顶点项目 09 — 代码迁移智能体（仓库级语言/运行时升级）

> Amazon 的 MigrationBench（Java 8 到 17）和 Google 的 App Engine Py2 到 Py3 迁移器设定了 2026 年的标杆。Moderne 的 OpenRewrite 做大规模确定性 AST 重写。Grit 用 codemod 风格 DSL 瞄准同样的问题。生产形态结合了两者：一个用于安全重写的确定性底层，加上一个用于模糊情况 的智能体层，一个用于每个分支构建的沙箱，以及一个在 PR 打开前变绿的测试工具。这个顶点项目是迁移 50 个真实仓库并发布通过率及失败分类法。

**类型：** 顶点项目
**语言：** Python（智能体）、Java / Python（目标）、TypeScript（仪表板）
**前置条件：** 阶段 5（NLP）、阶段 7（Transformer）、阶段 11（LLM 工程）、阶段 13（工具）、阶段 14（智能体）、阶段 15（自主）、阶段 17（基础设施）
**涉及的阶段：** P5 · P7 · P11 · P13 · P14 · P15 · P17
**时间：** 30 小时

## 问题

大规模代码迁移是 2026 年编码智能体最清晰的生产应用之一。Ground truth 是明显的（迁移后测试套件通过了吗？），奖励是真实的（Java-8  fleet 迁移是一个人员规模的项目），基准是公开的（MigrationBench 50 仓库子集）。Moderne 的 OpenRewrite 处理确定性方面。智能体层处理 OpenRewrite 配方无法处理的一切：模糊重写、构建系统漂移、长尾语法、可传递依赖破坏。

你将构建一个智能体，接收一个 Java 8 仓库（或 Python 2 仓库）并产生一个 CI 变绿的迁移分支。你将测量通过率、测试覆盖率保持、成本每个仓库，并构建一个失败分类法。与纯确定性基线的对比告诉你智能体的价值实际在哪里。

## 概念

流水线有两层。**确定性底层**（Java 用 OpenRewrite，Python 用 libcst）安全地运行大部分机械重写：导入、方法签名、空安全编辑、try-with-resources、弃用 API 替换。它快速且产生可审计的 diff。**智能体层**（OpenAI Agents SDK 或 LangGraph over Claude Opus 4.7 和 GPT-5.4-Codex）处理配方无法处理的情况：构建文件升级（Maven/Gradle/pyproject）、可传递依赖冲突、测试 flake、自定义注解。

每个仓库获得一个预装目标运行时的 Daytona 沙箱。智能体迭代：运行构建、分类失败、应用修复、重新运行。硬限制：每个仓库 30 分钟，每个仓库 8 美元，20 个智能体轮次。如果所有测试通过且覆盖率 delta 不为负，分支打开 PR。如果没有，仓库被归入一个失败类并附上证据。

失败分类法是交付物。跨 50 个仓库，什么坏了？可传递依赖？自定义注解？构建工具版本？与迁移无关的测试 flake？每个类获得一个计数和一个示例 diff。未来的配方作者可以针对前三类。

## 架构

```
目标仓库
      |
      v
OpenRewrite / libcst 确定性配方
   (安全、快速、可审计, ~70-80% 的修复)
      |
      v
每个分支的 Daytona 沙箱
      |
      v
智能体循环 (Claude Opus 4.7 / GPT-5.4-Codex):
   - 运行构建 -> 捕获失败
   - 分类失败 (构建、测试、lint)
   - 应用修复 (patch 或重试配方)
   - 重新运行
   - 预算: 30 分钟, 8 美元, 20 轮
      |
      v
测试 + 覆盖率 delta 门
      |
      v (通过)
打开 PR
      |
      v (失败)
归入失败类 + 附加复现
```

## 技术栈

- 确定性底层：OpenRewrite (Java) 或 libcst (Python)
- 智能体：OpenAI Agents SDK 或 LangGraph over Claude Opus 4.7 + GPT-5.4-Codex
- 沙箱：每个分支的 Daytona devcontainers，预装目标运行时（Java 17 / Python 3.12）
- 构建系统：Maven、Gradle、uv (Python)
- 基准：Amazon MigrationBench 50 仓库子集（Java 8 到 17），Google App Engine Py2 到 Py3 仓库
- 测试工具：并行运行器，通过 Jacoco (Java) 或 coverage.py (Python) 测量覆盖率
- 可观测性：Langfuse + 每个仓库的 trace bundle，带每个 diff chunk
- 仪表板：失败分类法仪表板，带每个类的计数和示例 diff

## 构建它

1. **配方通过。** 首先运行 OpenRewrite (Java) 或 libcst (Python) 配方。捕获 70-80% 的机械迁移。作为"配方"commit 提交。

2. **构建试用。** Daytona 沙箱：安装目标运行时，运行构建。如果变绿，跳到测试。如果变红，交给智能体。

3. **智能体循环。** LangGraph 配合工具：`run_build`、`read_file`、`edit_file`、`run_test`、`git_diff`。智能体分类失败（dep、syntax、test、build-tool）并应用针对性修复。重新运行。

4. **预算上限。** 每个仓库 30 分钟墙钟时间、8 美元成本、20 个智能体轮次。任何突破都停止并归入"budget_exhausted"，附上当前 diff。

5. **测试 + 覆盖率门。** 构建变绿后，运行测试套件。将覆盖率与基础仓库比较。如果覆盖率下降超过 2%，归入"coverage_regression"。

6. **打开 PR。** 成功后，推送分支，用应用的配方和智能体编写的 commit 的摘要打开 PR。

7. **失败分类法。** 对于每个失败的仓库，用一个类标记：`dep_upgrade_required`、`build_tool_drift`、`custom_annotation`、`test_flake`、`syntax_edge_case`、`budget_exhausted`。构建仪表板。

8. **50 仓库运行。** 在 MigrationBench 子集上执行。报告每个类的通过率、每仓库成本、覆盖率保持，以及与纯确定性基线的对比。

## 使用它

```
$ migrate legacy-java-service --target java17
[recipe]   应用了 27 个重写 (JUnit 4->5, HashMap 初始化器, try-with-resources)
[build]    失败: cannot find symbol sun.misc.BASE64Encoder
[agent]    轮次 1 分类: removed_jdk_api
[agent]    轮次 2 应用: sun.misc.BASE64Encoder -> java.util.Base64
[build]    通过
[tests]    412/412 通过; 覆盖率 84.1% -> 84.3%
[pr]       打开了 #1841  cost=$3.20  turns=4
```

## 交付它

`outputs/skill-migration-agent.md` 是交付物。给定一个仓库，它执行确定性配方然后是一个智能体循环，以产生一个变绿的迁移分支，或者将仓库归入一个分类法类。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | MigrationBench 通过率 | 50 仓库子集 pass@1 |
| 20 | 测试覆盖率保持 | 相对于基线的平均覆盖率 delta |
| 20 | 每个迁移仓库的成本 | 通过运行的 $/仓库 |
| 20 | 智能体/确定性工具集成 | OpenRewrite 处理 vs 智能体编写的修复比例 |
| 15 | 失败分析报告 | 带示例的分类法完整性 |
| **100** | | |

## 练习

1. 用纯 OpenRewrite（无智能体）运行迁移流水线。将通过率与完整流水线比较。识别智能体单独成为差异的情况。

2. 实现"lint-clean"检查：迁移后，运行风格 linter（Java 用 spotless，Python 用 ruff）。如果出现新的 lint 错误则 PR 失败。测量覆盖率保持但风格退化率。

3. 添加"最小 diff"优化器：在智能体的分支通过测试后，用第二轮修修剪不必要的变化。报告 diff 大小减少。

4. 扩展到第三个迁移：Node 18 到 Node 22。重用沙箱包装；为自定义 codemod 交换配方层。

5. 将首次变绿构建时间（TTFGB）作为 UX 指标来测量。目标：p50 在 10 分钟以下。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 确定性底层 | "配方引擎" | OpenRewrite / libcst：带安全保证的声明式 AST 重写 |
| Codemod | "代码修改程序" | 机械改变源代码的重写规则 |
| 构建漂移 | "工具版本偏斜" | 主要版本之间 Maven / Gradle / uv 行为的细微变化 |
| 失败类 | "分类桶" | 仓库未迁移的标注原因：dep、syntax、test、build-tool、budget |
| 覆盖率 delta | "覆盖率保持" | 从基线到迁移分支的测试覆盖率变化 % |
| 智能体轮次 | "工具调用轮次" | 智能体循环中的一个 plan -> act -> observe 周期 |
| 预算耗尽 | "触及上限" | 仓库在 30 分钟/8 美元/20 轮限制内未通过 |

## 延伸阅读

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — 2026 年规范基准
- [Moderne.io OpenRewrite 平台](https://www.moderne.io) — 确定性底层参考
- [OpenRewrite 文档](https://docs.openrewrite.org) — 配方编写
- [Grit.io](https://www.grit.io) — 替代 codemod DSL
- [OpenAI 沙箱迁移 cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — Agents SDK 参考
- [Google App Engine Py2 到 Py3 迁移器](https://cloud.google.com/appengine) — 替代迁移基准
- [libcst](https://github.com/Instagram/LibCST) — Python 确定性底层
- [Daytona 沙箱](https://daytona.io) — 每个分支沙箱参考