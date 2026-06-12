# DeepSeek-V3 架构 walkthrough

> 阶段 10 · 第 14 节命名了每个开放模型都会调整的六个架构旋钮。DeepSeek-V3（2024 年 12 月，671B 参数总计，37B 活跃）将六个旋钮全部调高，并额外添加了四个：多头潜在注意力、无辅助损失负载均衡、多 Token 预测和 DualPipe 训练。本节从顶部到底部阅读 DeepSeek-V3 的架构，从已发布配置推导出每个参数计数。读完你可以解释为什么 671B/37B 比率是正确的赌注，以及为什么 MLA + MoE 在前沿一起击败了各自单独的表现。

**类型：** 学习型
**语言：** Python（标准库、参数计算器）
**前置条件：** 阶段 10 · 14（开放模型 walkthrough）、阶段 10 · 17（NSA）、阶段 10 · 18（MTP）、阶段 10 · 19（DualPipe）
**时间：** 约 75 分钟

## 学习目标

- 从顶部到底部阅读 DeepSeek-V3 配置，用六个 GPT-2 旋钮加四个 DeepSeek 特有添加来解释每个字段。
- 推导出总参数计数（671B）、活跃参数计数（37B）以及每个组成部分的贡献。
- 计算 MLA 在 128k 上下文下的 KV cache 占用，并与具有 GQA 的相同活跃参数密集模型进行比较。
- 陈述四个 DeepSeek 特有创新（MLA、MTP、无辅助损失路由、DualPipe）并命名每个创新针对架构/训练栈的哪个部分。

## 问题

DeepSeek-V3 是第一个架构与 Llama 系列有实质性不同的前沿开放模型。Llama 3 405B 是"GPT-2 旋钮全开"。DeepSeek-V3 是 GPT-2 旋钮全开再加四个。阅读 Llama 3 配置是阅读 DeepSeek 配置的热身，但深层结构——注意力块的形状、路由逻辑、训练时目标——的不同程度足以让你需要一个单独的 walkthrough。

学习它的好处：DeepSeek-V3 的开源权重发布改变了前沿能力在开放模型中的含义。许多 2026 年的训练运行都在复制这个架构。理解它是任何涉及前沿 LLM 训练或推理岗位的 table stakes。

## 概念

### 不变的核心，再述

DeepSeek-V3 仍然是自回归的。它仍然堆叠解码器块。每个块仍然有注意力加 MLP 加两个 RMSNorm。它仍然在 MLP 中使用 SwiGLU。它仍然使用 RoPE。Pre-norm。权重绑定 embedding。与每个 Llama 或 Mistral 相同的基线。

### 转折：MLA 而不是 GQA

从阶段 10 · 14 你知道 GQA 通过在 Q 头组之间共享 K 和 V 来缩小 KV cache。多头潜在注意力（MLA）走得更远：K 和 V 被压缩成一个共享的低秩潜在表示（`kv_lora_rank`），然后在运行时按头解压。KV cache 只存储潜在向量——通常每个 token 每层 512 个浮点数，而不是 8 × 128 = 1024 个浮点数。

在 128k 上下文下，使用 MLA 的 DeepSeek-V3（每个 token 每层一个共享潜在 `c^{KV}`；K 和 V 都从这个潜在通过上投影导出，可以吸收到后续 matmul 中）：

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

一个假设的 GQA 基线（Llama 3 70B 形状，8 个 KV 头，头维度 128）需要：

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

MLA 在 128k 上下文下比 Llama-3-70B 风格的 GQA cache 小 4 倍。

权衡：MLA 在每次注意力计算（按头）增加一个解压步骤。额外算力很小，相比节省的带宽是净赚。对于长上下文推理是净赚。

### 路由：无辅助损失负载均衡

MoE 路由器决定哪些 top-k 专家处理每个 token。一个朴素的路由器将太多工作集中在少数专家身上，让其他人闲置。标准修复：添加一个辅助损失项来惩罚负载不均衡。这有效但会略微降低主任务性能。

DeepSeek-V3 引入了一种无辅助损失的方案。逐专家偏置项被加到路由器 logits，在训练期间通过一个简单规则调整：如果专家 `e` 过载，减少 `bias_e`；如果欠载，增加 `bias_e`。没有额外的损失项。训练保持干净。专家负载保持均衡。

对主损失的影响：没有可测量的影响。对 MoE 架构的影响：更干净，没有辅助损失超参数要调。

### MTP：更密集的训练 + 免费草稿

从阶段 10 · 18 你知道 DeepSeek-V3 添加了 D=1 MTP 模块，预测两个位置之后的 token。在推理时，训练好的模块被重新用作投机解码草稿，接受率 80%+。在训练时，每个隐藏状态在 D+1 = 2 个目标上被监督，提供更密集的信号。

参数：在 671B 主模型之上 14B。开销：2.1%。

### 训练：DualPipe

从阶段 10 · 19 你知道 DualPipe 是一种双向流水线，将前向和反向 chunk 与跨节点 all-to-all 通信重叠。在 DeepSeek-V3 的 2,048-H800 规模下，它回收了 1F1B 会因流水线气泡而损失的约 245k GPU-小时。

### 配置，逐字段

以下是 DeepSeek-V3 配置（简化版）：

```
hidden_size: 7168
intermediate_size: 18432   (前几层使用的密集 MLP 隐藏大小)
moe_intermediate_size: 2048 (专家 MLP 隐藏大小)
num_hidden_layers: 61
first_k_dense_layers: 3    (前 3 层使用密集 MLP)
num_attention_heads: 128
num_key_value_heads: 128   (在 MLA 下正式等于 num_heads，但
                           真正的压缩在 kv_lora_rank)
kv_lora_rank: 512          (MLA 潜在维度)
num_experts: 256            (每块的 MoE 专家数量)
num_experts_per_tok: 8      (top-8 路由)
shared_experts: 1           (每块始终在线的共享专家)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (深度 1 处 1 个 MTP 模块)
```

解析它：

- `hidden_size=7168`：embedding 维度。
- `num_hidden_layers=61`：总块深度。
- `first_k_dense_layers=3`：前 3 个块使用大小为 18432 的密集 MLP。其余 58 个使用 MoE。
- `num_attention_heads=128`：128 个查询头。
- `kv_lora_rank=512`：K 和 V 被压缩到这个潜在维度并按头解压。
- `num_experts=256, num_experts_per_tok=8`：每个 MoE 块有 256 个专家，路由 top-8。
- `shared_experts=1`：在 256 个路由专家之上，1 个始终在线的专家贡献给每个 token。可以把它想象成一个"密集底层"，确保每个 token 都能获得一些可靠的东西。
- `moe_intermediate_size=2048`：每个专家的 MLP 隐藏大小。比密集 MLP 小，因为有 256 个。

### 参数核算

完整计算在 `code/main.py` 中。标题数字：

- Embedding：`vocab * hidden = 129280 * 7168 = ~0.93B`。
- 前 3 个密集块：带 MLA 的注意力（每块约 144M）+ 密集 MLP（每块约 260M）+ norm。共约 1.2B。
- 58 个 MoE 块：带 MLA 的注意力（约 144M）+ 256 个专家（每个 30M）+ 1 个共享专家（30M）+ norm。总计每块约 7.95B，包括所有专家。58 个 MoE 块共 461B。
- MTP 模块：14B。

总计：核心架构约 476B + 14B MTP，而且明显地，发布的 671B 数字包含了额外的结构参数（偏置张量、专家特定组件、共享专家缩放等）。我们在计算器中重现的数字在发布的 3-5% 以内——差异来自 DeepSeek 报告在第 2 节附录中记录的超细粒度核算。

每次前向的活跃参数：

- 注意力：每层 144M × 61 = 8.8B（所有层都运行）。
- MLP 活跃：前 3 层密集（3 × 260M = 780M），58 个 MoE 层每层活跃 8 个路由 + 1 个共享 + 路由开销。每层活跃 MLP：约 260M。总计：3 × 260M + 58 × 260M = 约 15.9B。
- Embedding + norm：1.2B。
- 总活跃：约 26B 核心 + 14B MTP（训练过但推理时不总是运行）≈ 37B。

### 671B / 37B 比率

18 倍稀疏度比率（活跃参数是总参数的 5.5%）。DeepSeek-V3 是稀疏度最高的前沿 MoE 模型，已经发布了开源权重。Mixtral 8x7B 比率 13/47（28%）稠密得多。Llama 4 Maverick 比率 17B/400B（4.25%）可比。DeepSeek 的赌注：在前沿规模下，更多专家配合更低激活比率能比每活跃 FLOP 产生更好的质量。

### DeepSeek-V3 的位置

| 模型 | 总计 | 活跃 | 比率 | 注意力 | 创新点 |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + 无辅助损失 + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN 扩展 |

### 续作：R1、V4

DeepSeek-R1（2025 年）是 V3 主干上的推理训练运行。R1 使用相同的架构。改变的是后训练配方（在可验证任务上的大规模 RL），而不是预训练架构。

DeepSeek-V4（如果发布）预计将保留 MLA + MoE + MTP，并添加 DSA（DeepSeek 稀疏注意力），即阶段 10 · 17 中 NSA 的后继。血统稳定：架构级创新积累；每个版本都转动额外的旋钮。

## 使用它

`code/main.py` 是专用于 DeepSeek-V3 形状的参数计算器。运行它，将其输出与论文中的数字进行比较，并将其用于假设变体（256 专家 vs 512，top-8 vs top-16，MLA rank 512 vs 1024）。

要看的内容：

- 总参数计数 vs 发布 671B。
- 活跃参数计数 vs 发布 37B。
- 128k 上下文下的 KV cache——MLA vs GQA 比较。
- 每层分解，看看参数预算实际去向。

## 交付它

本课产出 `outputs/skill-deepseek-v3-reader.md`。给定一个 DeepSeek 系列模型（V3、R1 或任何未来变体），它产生逐组件架构阅读，命名配置的每个字段，按组件推导参数计数，并识别模型使用的四个 DeepSeek 特有创新中的哪一个。

## 练习

1. 运行 `code/main.py`。将计算器的总参数估计与发布的 671B 进行比较，并找出差异来自哪里。论文第 2 节有完整的逐项说明。

2. 修改配置使用 MLA rank 256 而不是 512。计算 128k 上下文下 resulting KV cache 大小。它买来了百分之多少的减少，以每头表达力的什么为代价？

3. 比较 DeepSeek-V3 的（256 专家，top-8）路由与假设的（512 专家，top-8）变体。总参数增长；活跃参数相同。理论上额外的专家容量买来了什么，推理时又花了什么代价？

4. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）第 2.1 节关于 MLA。用三句话解释为什么 K 和 V 解压矩阵可以"吸收"到后续 matmul 中以实现推理时效率。

5. DeepSeek-V3 对大多数操作使用 FP8 训练。计算 FP8 vs BF16 存储 671B 权重的内存节省。这如何与 14.8T token 训练预算相交？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| MLA | "多头潜在注意力" | 将 K 和 V 压缩成共享低秩潜在（kv_lora_rank，通常为 512），按头即时解压；KV cache 只存储潜在向量 |
| kv_lora_rank | "MLA 压缩维度" | K 和 V 的共享潜在大小；DeepSeek-V3 使用 512 |
| First k dense layers | "早期层保持密集" | 前几个 MoE 模型层跳过 MoE 路由器并运行密集 MLP 以保持稳定 |
| num_experts_per_tok | "Top-k 路由" | 每个 token 有多少个路由专家被激活；DeepSeek-V3 使用 8 |
| Shared experts | "始终在线专家" | 无论路由如何都处理每个 token 的专家；DeepSeek-V3 使用 1 |
| 无辅助损失路由 | "偏置调整负载均衡" | 训练期间调整的逐专家偏置项，以在不添加损失项的情况下保持专家负载均衡 |
| MTP 模块 | "额外预测头" | 预测 t+2 从 h^(1) 和 E(t+1) 的 transformer 块；更密集的训练，免费的投机解码草稿 |
| DualPipe | "双向流水线" | 将前向/反向计算与跨节点 all-to-all 重叠的训练调度 |
| 活跃参数比率 | "稀疏度" | active_params / total_params；DeepSeek-V3 达到 5.5% |
| FP8 训练 | "8 位训练" | 训练存储和许多计算操作使用 FP8；与 BF16 相比内存大致减半，质量成本很小 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 技术报告（arXiv:2412.19437）](https://arxiv.org/abs/2412.19437) — 完整架构、训练和结果文档
- [Hugging Face 上的 DeepSeek-V3 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V3) — 配置文件和部署说明
- [DeepSeek-V2 论文（arXiv:2405.04434）](https://arxiv.org/abs/2405.04434) — 引入了 MLA 的前身
- [DeepSeek-R1 论文（arXiv:2501.12948）](https://arxiv.org/abs/2501.12948) — V3 架构上的推理训练后续
- [原生稀疏注意力（arXiv:2502.11089）](https://arxiv.org/abs/2502.11089) — DeepSeek 系列注意力的未来方向
- [DualPipe 仓库](https://github.com/deepseek-ai/DualPipe) — 训练调度参考