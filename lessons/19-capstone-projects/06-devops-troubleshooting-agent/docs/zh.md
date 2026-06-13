# 毕业项目 06 — Kubernetes DevOps 故障排查智能体

> AWS 的 DevOps Agent 正式发布，Resolve AI 发布了 K8s  playbook，NeuBird 演示了语义监控，Metoro 将 AI SRE 与逐服务 SLO 挂钩。生产形态已定型：告警 webhook 触发，智能体读取遥测数据，走查 K8s 对象图，对根因假设排序，在 Slack 上发布简报并带审批按钮。默认只读。所有修复操作须经人工审批。本毕业项目就是构建这个智能体，在 20 个合成故障场景上评估，并在三个共享案例上与 AWS Agent 对比。

**类型：** 毕业项目
**语言：** Python（智能体）、TypeScript（Slack 集成）
**前置条件：** 阶段 11（LLM 工程）、阶段 13（工具和 MCP）、阶段 14（智能体）、阶段 15（自主系统）、阶段 17（基础设施）、阶段 18（安全）
**涉及的阶段：** P11 · P13 · P14 · P15 · P17 · P18
**时间：** 约 30 小时

## 问题

2025-2026 年的 SRE 叙事变成了："AI 智能体分诊故障，人类审批修复方案。" AWS DevOps Agent、Resolve AI、NeuBird、Metoro、PagerDuty AIOps 都在生产中交付这种形态。智能体读取 Prometheus 指标、Loki 日志、Tempo 追踪、kube-state-metrics 和 K8s 对象知识图。它在 5 分钟内生成带遥测引用的排序根因假设。除非通过 Slack 获得明确人工审批，否则绝不执行破坏性命令。

大部分硬活是范围界定和安全，而非推理。智能体需要一个默认只读的 RBAC 表面、一个加固的 MCP 工具服务器，以及每条命令的审查日志（已考虑 vs 已执行）。它需要知道自己何时超出能力范围并升级。而且它必须运行得足够便宜，以至于 OOM-kill 级联不会产生 5000 美元的智能体账单。

## 概念

智能体在知识图谱上运作。节点是 K8s 对象（Pod、Deployment、Service、Node、HPA、PVC）以及遥测数据源（Prometheus series、Loki 流、Tempo 追踪）。边编码所有权（Pod -> ReplicaSet -> Deployment）、调度（Pod -> Node）和观测（Pod -> Prometheus series）。图谱通过 kube-state-metrics 同步保持新鲜，并在每次告警时重新采样。

当告警触发时，智能体从受影响对象出发做根因分析。它走查边，获取相关遥测切片（最近 15 分钟），并起草假设。假设按证据排序：有多少遥测引用支持它，多近，多具体。前 3 名假设发送到 Slack，带图路径可视化和修复操作审批按钮。

修复操作有门控。默认允许的操作是只读的。破坏性操作（缩容、回滚、删除 Pod）需要 Slack 审批；ArgoCD 回滚钩子需要智能体从不持有的 auth token。审查日志记录智能体*已考虑*的每条命令——而非仅仅已执行的——以便审查过程捕获"差一点就出事"的情况。

## 架构

```
PagerDuty / Alertmanager webhook
           |
           v
      FastAPI 接收器
           |
           v
    LangGraph 根因分析智能体
           |
           +---- 只读 MCP 工具 ----+
           |                             |
           v                             v
    K8s 知识图谱                  遥测切片
      （Neo4j / kuzu）          Prometheus, Loki, Tempo
    所有权 + 调度               最近 15 分钟，已限定范围
           |
           v
    假设排序（证据权重）
           |
           v
    Slack 简报 + 审批按钮
           |
           v（已审批）
    ArgoCD 回滚钩子 / PagerDuty 升级
           |
           v
    审查日志：已考虑 vs 已执行，每条命令
```

## 技术栈

- 可观测性数据源：Prometheus, Loki, Tempo, kube-state-metrics
- 知识图谱：Neo4j（托管）或 kuzu（嵌入式），K8s 对象 + 遥测边
- 智能体：LangGraph，带每个工具的白名单，默认只读
- 工具传输：FastMCP over StreamableHTTP；破坏性工具在审批门控后置于独立服务器
- 模型：Claude Sonnet 4.7 用于根因推理，Gemini 2.5 Flash 用于日志摘要
- 修复：ArgoCD 回滚 webhook，PagerDuty 升级，Slack 审批卡片
- 审查：只追加的结构化日志（已考虑、已执行、已审批、结果）
- 部署：K8s deployment，带自身窄 RBAC 角色；独立 namespace

## 构建步骤

1. **图谱摄取。** 每 30 秒将 kube-state-metrics 同步到 Neo4j/kuzu。节点：Pod、Deployment、Node、Service、PVC、HPA。边：OWNED_BY、SCHEDULED_ON、EXPOSES、MOUNTS、SCALES。遥测叠加边：OBSERVED_BY（Pod 被 Prometheus series 观测）。

2. **告警接收器。** FastAPI 端点接收 PagerDuty 或 Alertmanager webhook。提取受影响对象和 SLO 违规。

3. **只读工具表面。** 通过 FastMCP 包装 kubectl、Prometheus 查询、Loki logql、Tempo traceql。每条工具都有窄 RBAC 动词（"get"、"list"、"describe"）。默认服务器无"delete"、"exec"、"scale"。

4. **根因分析智能体。** LangGraph 含三个节点：`sample` 拉取最近 15 分钟遥测切片，`walk` 查询图的邻接对象，`hypothesize` 起草带遥测引用的排序根因候选。

5. **证据评分。** 每个假设得分 = 时效性 × 特异性 × 图路径长度倒数 × 引用数量。返回前 3 名。

6. **Slack 简报。** 发布一条带附件的消息，包含假设、图路径可视化（服务端渲染的子图图像），以及最多一个修复操作的审批按钮。

7. **修复门控。** 破坏性工具（缩容、回滚、删除）位于第二个 MCP 服务器上，由审批 token 门控。智能体只有在 Slack 卡片被人类审批后才能调用这些工具。

8. **审查日志。** 只追加的 JSONL：对每条候选命令，记录它是否被考虑、是否被执行、由谁审批。每天发送到 S3。

9. **合成故障套件。** 构建 20 个场景：OOMKill 级联、DNS 抖动、HPA 抖动、PVC 占满、吵闹邻居、有缺陷的 sidecar、错误的 ConfigMap 推出、证书轮换、镜像拉取退避等。在根因准确率和时间-到-假设上给智能体打分。

## 使用示例

```
webhook: alert.pagerduty.com -> checkout-api SLO 违规，错误率 14%
[图谱]   受影响：Deployment checkout-api（3 个 Pod，Node ip-10-2-3-4）
[走查]   邻接：ReplicaSet checkout-api-abc, Service checkout-api,
         最近一次推出在 14 分钟前
[采样]   prometheus error_rate 14%，上升趋势；loki 在 /api/v2/pay 有 500 错误
[假设]   #1 糟糕的推出：最新镜像 checkout-api:v2.41 的 /healthz 失败
          引用：deploy.yaml（第42版）、prometheus errorRate、loki 500 错误栈
[Slack]  [回滚到 v2.40]  [升级]  [忽略]
          （需要审批；智能体不会单方面回滚）
```

## 交付

`outputs/skill-devops-agent.md` 是交付物。给定一个 K8s 集群和告警源，智能体生成排序的根因假设和 Slack 门控的修复流程。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 场景套件上的 RCA 准确率 | 20 个合成故障中 ≥80% 根因正确 |
| 20 | 安全性 | 破坏性操作门控在审查日志中始终有 Slack 审批 |
| 20 | 时间-到-假设 | 从告警到 Slack 简报 p50 在 5 分钟以内 |
| 20 | 可解释性 | 每个假设都有图路径和遥测引用 |
| 15 | 集成完整性 | PagerDuty、Slack、ArgoCD、Prometheus 端到端运行 |
| **100** | | |

## 练习

1. 在 AWS DevOps Agent 演示的相同三个故障场景上运行你的智能体。发布并排对比。报告智能体出现分歧的地方。

2. 添加"险些出事"审查，标记智能体*已考虑*但未经审批就会具有破坏性的任何命令。在一周内测量险些出事率。

3. 将假设模型从 Claude Sonnet 4.7 换成自托管 Llama 3.3 70B。测量 RCA 准确率变化和每次故障的美元成本。

4. 构建因果过滤器：区分相关遥测尖峰和真实根因。在 20 个场景标签上训练小型分类器。

5. 添加回滚预演：使用相同清单对 staging 集群执行 ArgoCD 回滚。在 Slack 审批按钮之前，在实际集群中验证回滚计划。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|------------------------|
| K8s 知识图谱 (K8s knowledge graph) | "集群图" | 节点 = K8s 对象 + 遥测 series；边 = 所有权、调度、观测 |
| 默认只读 (Read-only-by-default) | "窄 RBAC" | 智能体的服务账号只有 get/list/describe 动词；破坏性动词在独立服务器上，由审批门控 |
| 审查日志 (Audit log) | "已考虑 vs 已执行" | 每条候选命令的只追加记录，记录它是否运行、由谁审批 |
| 假设排序 (Hypothesis ranking) | "证据分数" | 时效性 × 特异性 × 图路径长度倒数 × 引用数量 |
| Slack 审批卡片 (Slack approval card) | "HITL 门控" | 带修复按钮的交互式 Slack 消息；人类点击之前智能体无法继续 |
| 遥测引用 (Telemetry citation) | "证据指针" | 支持某个声明的 Prometheus 查询、Loki 选择器或 Tempo 追踪 URL |
| MTTR | "恢复时间" | 从告警触发到 SLO 恢复的 wall-clock 时间 |

## 延伸阅读

- [AWS DevOps Agent 正式发布](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 2026 年规范参考
- [Resolve AI K8s 故障排查](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — 竞品参考
- [NeuBird 语义监控](https://www.neubird.ai) — 语义图方法
- [Metoro AI SRE](https://metoro.io) — SLO 优先的生产视角
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — 集群状态来源
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 参考智能体编排器
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP 服务器框架
- [ArgoCD 回滚](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — 门控修复目标