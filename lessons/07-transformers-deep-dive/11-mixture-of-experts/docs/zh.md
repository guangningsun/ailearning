# 专家混合（MoE）

> 一个密集 70B transformer 对每个 token 激活所有参数。一个 671B MoE 对每个 token 只激活 37B，却在所有基准上胜出。稀疏性是这十年最重要的扩展思想。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 7 · 05（全量 Transformer）、阶段 7 · 07（GPT）
**时间：** 约 45 分钟

## 问题

密集 transformer 的推理 FLOP 数等于其参数量（前向传播乘以 2）。扩大密集模型，每个 token 都要付全部账单。到 2024 年，前沿模型遇到了计算墙：要变得更聪明，每个 token 需要指数级更多的 FLOP。

专家混合打破了这个链接。将每个 FFN 替换为 `E` 个独立专家 + 一个路由器，每个 token 选择 `k` 个专家。总参数 = `E × FFN_size`。每个 token 激活的参数 = `k × FFN_size`。2026 年典型配置：`E=256`，`k=8`。存储随 `E` 缩放，计算随 `k` 缩放。

2026 年的前沿几乎全是 MoE：DeepSeek-V3（671B 总计 / 37B 激活）、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss。在 Artificial Analysis 的独立排行榜上，开源模型前 10 名全是 MoE。

## 概念

![MoE 层：路由器为每个 token 从 E 个专家中选择 k 个](../assets/moe.svg)

### FFN 替换

密集 transformer 块：

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE 块：

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # 每个 token 从 E 中选 k 个
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

每个专家都是一个独立的 FFN（通常是 SwiGLU）。路由器是一个线性层。每个 token 选择自己的 `k` 个专家，得到它们输出的门控混合。

### 负载均衡问题

如果路由器把 90% 的 token 都发给专家 3，其他专家就饿死了。已尝试三种解决方案：

1. **辅助负载均衡损失**（Switch Transformer、Mixtral）。根据专家使用率的方差添加惩罚。有效，但引入了一个超参数和第二个梯度信号。
2. **专家容量 + token 丢弃**（早期 Switch）。每个专家最多处理 `C × N/E` 个 token；溢出的 token 跳过该层。损害质量。
3. **无辅助损失的自由均衡**（DeepSeek-V3）。添加一个可学习的每专家偏置，移动路由器的 top-k 选择。偏置在训练损失之外更新。不对主目标加惩罚。2024 年的重大突破。

DeepSeek-V3 的方法：每步训练后，检查每个专家的使用率是否高于或低于目标。用 `±γ` 调整偏置。选择使用 `scores + bias`。用于门控的专家概率是原始 `scores`，不变。将路由与表达解耦。

### 共享专家

DeepSeek-V2/V3 还把专家分成*共享*和*路由*两类。每个 token 都经过所有共享专家。路由专家通过 top-k 选择。共享专家捕获通用知识；路由专家专精。V3 运行 1 个共享专家 + 256 个路由专家中的 top-8。

### 细粒度专家

经典 MoE（GShard、Switch）：每个专家和一个完整 FFN 一样宽。`E` 小（8–64），`k` 小（1–2）。

现代细粒度 MoE（DeepSeek-V3、Qwen-MoE）：每个专家更窄（1/8 FFN 大小）。`E` 大（256+），`k` 更大（8+）。总参数相同，但组合扩展快得多。每个 token 有 `C(256, 8) = 40 万亿` 种可能的"专家"。质量上升，延迟持平。

### 成本画像

每个 token，每层：

| 配置 | 激活参数 / token | 总参数 |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B（密集） | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2（MoE） | ~32B | 1T |

DeepSeek-V3 在几乎所有基准上胜出 Llama 3 70B（密集），同时**每个 token 激活的 FLOP 更少**。更多参数 = 更多知识。更多激活 FLOP = 每个 token 更多计算。MoE 将它们解耦。

### 陷阱：内存

所有专家都在 GPU 上，不管哪些被激活。一个 671B 模型在 fp16 权重下需要约 1.3 TB VRAM。前沿 MoE 部署需要专家并行——将专家分片到不同 GPU，跨网络路由 token。延迟由 all-to-all 通信主导，而不是矩阵乘法。

## 构建它

见 `code/main.py`。一个纯标准库的紧凑 MoE 层，包含：

- `n_experts=8` 个 SwiGLU-ish 专家（每个有一个线性层，用于说明）
- top-k=2 路由
- softmax 归一化的门控权重
- 通过每专家偏置实现无辅助损失的自由均衡

### 步骤 1：路由器

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # 在所选专家的原始 scores 上做 softmax
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

偏置影响选择，不影响门控权重。这就是 DeepSeek-V3 的技巧——偏置纠正负载不平衡，而不 steering 模型的预测。

### 步骤 2：用 100 个 token 过路由器

跟踪哪些专家被激活了多少次。没有偏置时，使用率会偏斜。有了偏置更新循环（超负荷专家 `-γ`，低负荷专家 `+γ`），使用率在几次迭代后收敛到均匀分布。

### 步骤 3：参数量对比

打印 MoE 配置的"密集等价"。DeepSeek-V3 形状：256 个路由 + 1 个共享，8 个激活，d_model=7168。总参数量惊人。激活量是密集 Llama 3 70B 的七分之一。

## 使用它

HuggingFace 加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026 年生产推理：vLLM 原生支持 MoE 路由。SGLang 有最快的专家并行路径。两者都自动处理 top-k 选择和专家并行。

**何时选 MoE：**
- 你想在更低的每 token 推理成本下获得前沿质量。
- 你有 VRAM / 专家并行基础设施。
- 你的工作负载是 token 密集型（聊天、代码）而非上下文密集型（长文档）。

**何时不选 MoE：**
- 边缘部署——你为任何激活 FLOP 支付全部存储。
- 延迟敏感的单用户服务——专家路由增加开销。
- 小模型（<7B）—— MoE 的质量优势只在计算阈值以上才出现（约 6B 激活参数）。

## 交付它

见 `outputs/skill-moe-configurator.md`。这个 skill 在给定参数预算、训练 token 数和部署目标的情况下，为新的 MoE 选择 E、k 和共享专家布局。

## 练习

1. **简单。** 运行 `code/main.py`。观察无辅助损失的自由偏置更新如何在 50 次迭代中平衡专家使用。
2. **中等。** 将学习型路由器替换为基于哈希的路由器（确定性，无学习）。比较质量和均衡。为什么学习型路由器更好？
3. **困难。** 实现 GRPO 风格的" rollout 匹配路由"（DeepSeek-V3.2 技巧）：在推理时记录哪些专家被激活，在梯度计算时强制相同的路由。在玩具策略梯度设置上测量效果。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| 专家 | "众多 FFN 之一" | 一个独立的前馈网络；参数专用于 FFN 计算的一个稀疏切片。 |
| 路由器 | "门" | 一个微小的线性层，为每个 token 给每个专家打分；top-k 选择。 |
| Top-k 路由 | "每个 token 激活 k 个专家" | 每个 token 的 FFN 计算正好经过 k 个专家，按门控权重加权。 |
| 辅助损失 | "负载均衡惩罚" | 惩罚倾斜专家使用率的额外损失项。 |
| 无辅助损失 | "DeepSeek-V3 的技巧" | 通过路由器选择上的每专家偏置来均衡；没有额外梯度。 |
| 共享专家 | "总是激活" | 每个 token 都经过的额外专家；捕获通用知识。 |
| 专家并行 | "按专家分片" | 将不同专家分配到不同 GPU；跨网络路由 token。 |
| 稀疏性 | "激活参数 < 总参数" | 比率 `k × expert_size / (E × expert_size)`；DeepSeek-V3 约为 37/671 ≈ 5.5%。 |

## 延伸阅读

- [Shazeer 等人（2017）。极其庞大的神经网络：稀疏门控专家混合层](https://arxiv.org/abs/1701.06538) —— 这个想法的起源。
- [Fedus，Zoph，Shazeer（2022）。Switch Transformer：用简单高效稀疏性扩展到万亿参数模型](https://arxiv.org/abs/2101.03961) —— Switch，经典 MoE。
- [Jiang 等人（2024）。Mixtral of Experts](https://arxiv.org/abs/2401.04088) —— Mixtral 8×7B。
- [DeepSeek-AI（2024）。DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437) —— MLA + 无辅助损失 MoE + MTP。
- [Wang 等人（2024）。专家混合无辅助损失负载均衡策略](https://arxiv.org/abs/2408.15664) —— 基于偏置的均衡论文。
- [Dai 等人（2024）。DeepSeekMoE：迈向专家混合语言模型的终极专家专精](https://arxiv.org/abs/2401.06066) —— 本课路由器使用的细粒度 + 共享专家分离。
- [Kim 等人（2022）。DeepSpeed-MoE：推进专家混合推理与训练](https://arxiv.org/abs/2201.05596) —— 原始共享专家论文。