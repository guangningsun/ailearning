# 自托管服务引擎选型——llama.cpp、Ollama、TGI、vLLM、SGLang

> 2026 年，四大引擎主导自托管推理。按硬件、规模、生态来选。**llama.cpp** 是 CPU 上最快的——模型支持最广，量化和线程控制完全自主。**Ollama** 是开发者笔记本上一键安装的选择，比 llama.cpp 慢 15-30%（Go + CGo + HTTP 序列化），生产负载下吞吐量差距达 3 倍。**TGI 于 2025 年 12 月 11 日进入维护模式**——仅修复 bug，原始吞吐量比 vLLM 慢约 10%，但历史上可观测性最强、HF 生态集成最好。维护状态使其成为高风险的长期选择——SGLang 或 vLLM 是新项目的更安全默认。**vLLM** 是通用生产默认——v0.15.1（2026 年 2 月）新增 PyTorch 2.10、RTX Blackwell SM120、H200 优化。**SGLang** 是 Agent 多轮 / 前缀密集型专用——生产部署超过 400,000 块 GPU（xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS）。硬件限制：纯 CPU → 只能 llama.cpp。AMD / 非 NVIDIA → 只能 vLLM（TRT-LLM 仅支持 NVIDIA）。2026 年流程模式：开发 = Ollama，预发布 = llama.cpp，生产 = vLLM 或 SGLang。全程使用相同的 GGUF/HF 权重。

**类型：** 学习型
**语言：** Python（标准库、引擎决策树遍历器）
**前置条件：** 所有阶段 17 中覆盖引擎的课程（04、06、07、09、18）
**时间：** 约 45 分钟

## 学习目标

- 根据硬件（CPU / AMD / NVIDIA Hopper / Blackwell）、规模（1 用户 / 100 / 10,000）和工作负载（通用聊天 / Agent / 长上下文）选择引擎。
- 说出 2026 年 TGI 维护模式的状态（2025 年 12 月 11 日）及其为何使新项目倾向于 vLLM 或 SGLang。
- 描述使用相同 GGUF 或 HF 权重的开发/预发布/生产流程。
- 解释为什么"纯 CPU"必须用 llama.cpp，以及为什么"AMD"要排除 TRT-LLM。

## 问题

你的团队启动了一个新的自托管 LLM 项目。一位工程师说用 Ollama，另一位说用 vLLM，第三位说"TGI 不是开箱即用吗？"三种说法在各自场景下都对。但没有一个适合所有场景。

2026 年，决策树很重要：硬件优先，规模其次，工作负载第三。还有一个 2025 年的特殊事件——TGI 于 12 月 11 日进入维护模式——改变了新项目的默认选择。

## 概念

### 五大引擎

| 引擎 | 最适合 | 备注 |
|--------|----------|-------|
| **llama.cpp** | CPU / 边缘 / 依赖最少 / 模型支持最广 | CPU 上最快，完全自主控制 |
| **Ollama** | 开发笔记本、单用户、一键安装 | 比 llama.cpp 慢 15-30%；生产吞吐量差距 3 倍 |
| **TGI** | HF 生态、受监管行业 | **2025 年 12 月 11 日进入维护模式** |
| **vLLM** | 通用生产、100+ 用户 | 广泛的生产默认；v0.15.1（2026 年 2 月） |
| **SGLang** | Agent 多轮、前缀密集型工作负载 | 生产部署超过 400,000 块 GPU |

### 硬件优先决策

**纯 CPU** → llama.cpp。Ollama 也可以但更慢。其他引擎在 CPU 上没有竞争力。

**AMD GPU** → vLLM（AMD ROCm 支持）。SGLang 也可以。TRT-LLM 仅支持 NVIDIA，所以排除。

**NVIDIA Hopper（H100 / H200）** → vLLM 或 SGLang 或 TRT-LLM。三者都是顶级。

**NVIDIA Blackwell（B200 / GB200）** → TRT-LLM 是吞吐量领导者（阶段 17 · 07）。vLLM 和 SGLang 紧随其后。

**Apple Silicon（M 系列）** → llama.cpp（Metal）。Ollama 在此基础上封装。

### 规模其次决策

**1 用户 / 本地开发** → Ollama。一条命令，首 token 几秒内到达。

**10-100 用户 / 小团队** → vLLM 单卡。

**100-10k 用户 / 生产** → vLLM 生产栈（阶段 17 · 18）或 SGLang。

**10k+ 用户 / 企业** → vLLM 生产栈 + 分离式架构（阶段 17 · 17）+ LMCache（阶段 17 · 18）。

### 工作负载第三决策

**通用聊天 / 问答** → vLLM 以广泛默认胜出。

**Agent 多轮（工具、规划、记忆）** → SGLang 的 RadixAttention（阶段 17 · 06）占主导。

**重型前缀复用的 RAG** → SGLang。

**代码生成** → vLLM 不错；SGLang 在缓存上略优。

**长上下文（128K+）** → vLLM + 分块预填充；SGLang + 分层 KV。

### TGI 维护陷阱

Hugging Face TGI 于 2025 年 12 月 11 日进入维护模式——仅修复 bug。历史上：顶级可观测性、最佳 HF 生态集成（模型卡片、安全工具），原始吞吐量略逊于 vLLM。

2026 年新项目：默认不选 TGI。现有 TGI 部署可继续但应最终迁移。SGLang 和 vLLM 是更安全的默认。

### 流程模式

开发（Ollama）→ 预发布（llama.cpp）→ 生产（vLLM）。全程使用相同的 GGUF 或 HF 权重。工程师在笔记本上快速迭代；预发布镜像生产量化；生产是服务目标。

### Ollama 注意事项

Ollama 很适合开发。它不适合共享生产：Go HTTP 序列化增加开销，并发管理比 vLLM 简单，OpenTelemetry 支持落后。在 Ollama 擅长之处使用——单用户、一条命令——并在共享场景切换到 vLLM。

### 自托管 vs 托管是另一个决策

阶段 17 · 01（托管超大规模）、· 02（推理平台）覆盖托管。本课假设你已经决定自托管。自托管的理由：数据主权、自定义微调、规模化总体拥有成本、托管上没有的领域模型。

### 应记住的数字

- TGI 维护模式：2025 年 12 月 11 日。
- vLLM v0.15.1：2026 年 2 月；PyTorch 2.10；Blackwell SM120 支持。
- SGLang 生产部署：超过 400,000 块 GPU。
- Ollama 吞吐量差距 vs llama.cpp：慢 15-30%；生产负载下差距 3 倍。

## 实际使用

`code/main.py` 是一个决策树遍历器：给定硬件 + 规模 + 工作负载，选择一个引擎并解释原因。

## 交付物

本课产出 `outputs/skill-engine-picker.md`。给定约束条件，选择一个引擎并写出迁移计划。

## 练习

1. 用你的硬件 / 规模 / 工作负载运行 `code/main.py`。输出与你的直觉一致吗？
2. 你的基础设施是 12 块 H100 和 8 块 MI300X AMD。用哪个引擎？为什么 TRT-LLM 排除？
3. 一个团队想在 2026 年使用 TGI，因为"这是我们熟悉的"。论证迁移的必要性。
4. Ollama 开发到 vLLM 生产：量化、配置和可观测性有什么变化？
5. RAG 产品 P99 前缀长度 8K，跨租户高复用。选一个引擎，并用阶段 17 · 11 + 18 组合堆栈。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| llama.cpp | "那个 CPU 的" | 模型支持最广，CPU 上最快 |
| Ollama | "那个笔记本的" | 一键安装，开发级吞吐量 |
| TGI | "HF 的服务" | 2025 年 12 月起维护模式 |
| vLLM | "那个默认的" | 2026 年通用生产基准 |
| SGLang | "那个 Agent 的" | 前缀密集型、RadixAttention |
| TRT-LLM | "那个 NVIDIA 专用的" | Blackwell 吞吐量领先，仅 NVIDIA |
| GGUF | "llama.cpp 格式" | 打包的 K 量化和变体 |
| 生产栈 | "vLLM K8s" | 阶段 17 · 18 参考部署 |
| 流程模式 | "开发→预发布→生产" | Ollama → llama.cpp → vLLM，使用相同权重 |

## 延伸阅读

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — Comprehensive LLM Inference Engine Comparison](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI maintenance announcement](https://github.com/huggingface/text-generation-inference) — release notes.
- [vLLM v0.15.1 release notes](https://github.com/vllm-project/vllm/releases)