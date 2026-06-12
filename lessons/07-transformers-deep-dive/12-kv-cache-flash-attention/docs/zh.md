# KV 缓存、Flash Attention 与推理优化

> 训练是并行的，受 FLOP 限制。推理是串行的，受内存限制。不同的瓶颈，不同的技巧。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 7 · 02（自注意力）、阶段 7 · 05（全量 Transformer）、阶段 7 · 07（GPT）
**时间：** 约 75 分钟

## 问题

一个朴素的的自回归解码器生成 `N` 个 token 需要 `O(N²)` 的工作：每一步都要在整个前缀上重新计算注意力。对于 4K token 的响应，那是 16M 次注意力操作，其中大部分是冗余的。前缀 token 的每个隐藏状态一旦计算出来就是确定的——你只需要用新 token 的 query 去查询之前所有缓存的 key 和 value。

在此之上，注意力本身移动了大量数据。标准注意力物化了一个 N×N 的分数矩阵、N×d 的 softmax 输出、N×d 的最终输出——对 HBM 有太多读写。对于 N≥2K，注意力在成为 FLOP 受限之前就已经是内存受限了。经典注意力核在现代 GPU 上利用率只有 4–10×。

Dao 等人的两个优化，将前沿推理从"慢"推向"快"：

1. **KV 缓存。** 存储每个前缀 token 的 K 和 V 向量。每个新 token 的注意力就是一次 query 对缓存的 key。推理从每生成步的 `O(N²)` 降至 `O(N)`。
2. **Flash Attention。** 对注意力计算进行分块，使完整的 N×N 矩阵永不进入 HBM。所有的 softmax + 矩阵乘法都在 SRAM 中完成。在 A100 上快 2–4 倍；在 H100 上用 FP8 快 5–10 倍。

到 2026 年，两者都已普及。每个生产推理堆栈（vLLM、TensorRT-LLM、SGLang、llama.cpp）都假设它们存在。每个前沿模型都自带 Flash Attention 启用。

## 概念

![KV 缓存增长和 Flash Attention 分块](../assets/kv-cache-flash-attn.svg)

### KV 缓存数学

每个解码器层，每个 token，每个头：

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K 和 V
```

对于 7B 模型，32 层，32 头，d_head=128，fp16：

```
每 token 每层 = 2 * 128 * 2 = 512 字节
每 token（32 层）= 16 KB
32K 上下文 = 512 MB
```

对于 Llama 3 70B（80 层，d_head=128，GQA 共 8 个 KV 头）：

```
每 token 每层 = 2 * 8 * 128 * 2 = 4096 字节（4 KB）
32K 上下文 = 10.4 GB
```

这 10 GB 就是为什么 Llama 3 70B 在 128K 上下文时，批大小为 1 时需要整整 40 GB A100 来存放 KV 缓存。

**GQA 是 KV 缓存的赢。** MHA 有 64 个头会是 32 GB。MLA 压缩得更厉害。

拖动维度看缓存大小怎么变化。增加序列长度或批大小，看它多快超过单个 GPU：

```figure
kv-cache-sizer
```

### Flash Attention —— 分块技巧

标准注意力：

```
S = Q @ K^T          （HBM 读，N×N，HBM 写）
P = softmax(S)       （HBM 读，HBM 写）
O = P @ V            （HBM 读，HBM 写）
```

三次 HBM 往返。在 H100 上，HBM 带宽 3 TB/s；SRAM 是 30 TB/s。每次 HBM 往返是将所有数据保持在芯片上的 10 倍减速。

Flash Attention：

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     （SRAM）
        running softmax aggregation             （SRAM）
        accumulate into O_tile                  （SRAM）
    write O_tile to HBM
```

每个 tile 一次 HBM 往返。总内存占用从 `O(N²)` 降至 `O(N)`。反向传播从正向传播重新计算一些值而不是存储它们——又是一次内存赢。

**数值技巧。** 运行时 softmax 维护跨 tile 的 `（max, sum）`，因此最终归一化是精确的。不是近似——Flash Attention 计算的输出与标准注意力逐位相同（除去 fp16 非结合性）。

**版本演进：**

| 版本 | 年份 | 关键变化 | 参考硬件上的加速 |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | 分块 SRAM 核 | A100 上 2× |
| Flash 2 | 2023 | 更好的并行性，causal-first 排序 | A100 上 3× |
| Flash 3 | 2024 | Hopper 异步，FP8 | H100 上 1.5–2×（约 740 TFLOPs FP16） |
| Flash 4 | 2026 | Blackwell 5 级流水线，software exp2 | 推理优先（最初只支持前向） |

Flash 4 发布时只支持前向传播。训练仍用 Flash 3。Flash 4 的 GQA 和 varlen 支持仍在路上（2026 年中）。

### 投机解码 —— 另一个延迟赢

廉价模型提出 N 个 token。大模型并行验证所有 N 个。如果验证接受了 k 个 token，你用 1 次大模型前向传播换了 k 个生成。代码和散文上典型 k=3–5。

2026 年默认：
- **EAGLE 2 / Medusa。** 集成 draft head，共享验证者的隐藏状态。2–3× 加速，质量不损失。
- **带 draft 模型的投机解码。** 在消费级硬件上 2–4× 加速。
- **Lookahead 解码。** Jacobi 迭代；不需要 draft 模型。小众但免费。

### 连续批处理

经典批处理推理：等最慢的序列完成，才开始新批次。短响应提前完成时浪费 GPU。

连续批处理（最初在 Orca 推出，现已在 vLLM、TensorRT-LLM、SGLang 中实现）：旧序列一完成就将新请求换入批次。典型聊天工作负载吞吐量提升 5–10×。

### PagedAttention —— 作为虚拟内存的 KV 缓存

vLLM 的招牌功能。KV 缓存在 16 token 块中分配；页表将逻辑位置映射到物理块。支持跨并行样本（束搜索、并行采样）共享 KV，热插拔前缀用于 prompt 缓存，内存碎片整理。相比朴素连续分配，吞吐量提升 4×。

## 构建它

见 `code/main.py`。我们实现：

1. 一个朴素的 `O(N²)` 增量解码器。
2. 一个 `O(N)` 的 KV 缓存解码器。
3. 一个模拟 Flash Attention 运行时 max 算法的分块 softmax。

### 步骤 1：KV 缓存

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

简单：在每层、每头的列表中持续增长每个 token 的 K、V 向量。

### 步骤 2：分块 softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

与 `softmax(qK) V` 一次完成的输出逐位相同，但任何时候工作集都是 `tile × d_head` 块，而不是完整的 `N × d_head`。

### 步骤 3：比较 100 token 生成上的朴素解码 vs 缓存解码

统计注意力操作数。朴素：`O(N²)` = 5050。缓存：`O(N)` = 100。代码打印两者。

## 使用它

```python
# HuggingFace transformers 在 decoder-only generate() 上自动启用 KV 缓存
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # Hopper 用 FA3
    torch_dtype="bfloat16",
)
# generate() 自动使用 KV 缓存
```

vLLM 生产：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

跨请求的前缀缓存是 2026 年的一大赢——相同的系统 prompt、少样本示例或长上下文文档在调用之间复用 KV。对于有重复工具 prompt 的 agent 工作负载，前缀缓存通常是 5× 的吞吐量提升。

## 交付它

见 `outputs/skill-inference-optimizer.md`。这个 skill 为新的推理部署选择注意力实现、KV 缓存策略、量化和投机解码。

## 练习

1. **简单。** 运行 `code/main.py`。确认朴素和缓存解码器产生相同输出；注意操作数差异。
2. **中等。** 实现前缀缓存：给定 prompt P 和多个补全，运行一次前向传播过 P 来填充 KV 缓存，然后每个补全分支。测量 vs 每个补全重新编码 P 的加速比。
3. **困难。** 实现玩具 PagedAttention：在固定 16 token 块中用 KV 缓存加一个空闲列表。当一个序列完成时，将其块归还池中。模拟 1,000 个不同长度的聊天补全。比较内存碎片 vs 连续分配。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|-----------------|-----------------------|
| KV 缓存 | "让解码变快的技巧" | 存储每个前缀 token 的 K 和 V；新 query attend 到它们而不是重新计算。 |
| HBM | "GPU 主内存" | 高带宽内存；H100 上 80 GB，B200 上 192 GB。约 3 TB/s 带宽。 |
| SRAM | "片上内存" | 每 SM 的快速内存，H100 上每 SM 约 256 KB。约 30 TB/s 带宽。 |
| Flash Attention | "分块注意力核" | 计算注意力时不将 N×N 物化到 HBM 中。 |
| 连续批处理 | "不等批处理" | 将完成的序列换出，新序列换入，不排空批次。 |
| PagedAttention | "vLLM 的招牌" | KV 缓存在固定块中分配，有页表；消除碎片。 |
| 前缀缓存 | "复用长 prompt" | 缓存跨请求共享前缀的 KV；agent 成本大幅降低。 |
| 投机解码 | "draft + 验证" | 廉价 draft 模型提出 token；大模型一次通过验证 k 个。 |

## 延伸阅读

- [Dao 等人（2022）。FlashAttention：快速和内存高效的可感知 IO 的精确注意力](https://arxiv.org/abs/2205.14135) —— Flash 1。
- [Dao（2023）。FlashAttention-2：更好的并行性和工作划分的更快注意力](https://arxiv.org/abs/2307.08691) —— Flash 2。
- [Shah 等人（2024）。FlashAttention-3：利用异步和低精度实现快速准确注意力](https://arxiv.org/abs/2407.08608) —— Flash 3。
- [FlashAttention-4 发布说明（Dao-AILab，2026）](https://github.com/Dao-AILab/flash-attention) —— Blackwell 5 级流水线和 software-exp2 技巧；读 repo README 了解本课提到的前向优先发布注意事项。
- [Kwon 等人（2023）。利用 PagedAttention 进行大语言模型服务的高效内存管理](https://arxiv.org/abs/2309.06180) —— vLLM 论文。
- [Leviathan 等人（2023）。通过投机解码实现 Transformer 快速推理](https://arxiv.org/abs/2211.17192) —— 投机解码。
- [Li 等人（2024）。EAGLE：投机采样需要重新思考特征不确定性](https://arxiv.org/abs/2401.15077) —— 本课引用的集成 draft 方法的 EAGLE-1/2 论文。
- [Cai 等人（2024）。Medusa：带有多个解码头的简单 LLM 推理加速框架](https://arxiv.org/abs/2401.10774) —— 与 EAGLE 并列引用的 Medusa 方法。
- [vLLM 文档 —— PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) —— 16 token 块和页表设计的权威深入解析。