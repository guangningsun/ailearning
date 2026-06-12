# 梯度检查点与激活重计算

> 反向传播保留每个中间激活。在 70B 参数和 128K 上下文下，这是每 rank 3TB 的激活。检查点以 FLOPs 换内存：不保存，重新计算。问题是丢弃哪些段落，答案不是"全部丢弃"。

**类型：** 构建型
**语言：** Python（带 numpy，可选 torch）
**前置条件：** 阶段 10 · 课 04（预训练 Mini-GPT），阶段 10 · 课 05（扩展与分布式）
**时间：** 约 70 分钟

## 问题

训练 transformer 会为每一层存储每个被反向求导的操作的输入：注意力输入、Q/K/V 投影、softmax 输出、FFN 输入、norm 输出和残差流。对于 hidden 大小 `d`、序列长度 `L`、batch `B`，这大约是每层 `12 * B * L * d` 个浮点数。

对于 `d=8192, L=8192, B=1`，每层 800 MB（BF16）。64 层模型是 51 GB 激活 — 还没乘以 microbatch 大小，没加上注意力 softmax 中间量（每头 `L^2`），没考虑 tensor 并行部分拷贝。

双面账单：BF16 权重加优化器状态可能塞进 80GB，但激活把你推出去了。梯度检查点（又称激活重计算）是标准修复。丢弃大部分激活；在反向时重做前向来恢复它们。代价：额外 FLOPs。收益：按检查点段落数比上总层数的比例减少内存。

做得不好，检查点每步多花约 33% 前向 FLOPs。做得好 — 按 Korthikanti et al. 的"智能选择"做选择性检查点 — 你在 5% FLOPs 开销下节省 5 倍内存。加上 FP8 矩阵乘法、FSDP 卸载和专家并行 MoE，这就真的很要紧了：你既负担不起内存，也负担不起浪费的计算。

## 概念

### 反向传播实际需要什么

`output = layer(input)`。反向想要 `grad_input` 和 `grad_params`。要计算它们，它需要：

- `input`（用于计算线性层的 `grad_params = input.T @ grad_output`）
- 一些激活导数中间量（ReLU/GELU/softmax 的导数依赖于激活值）

前向传播自动在 autograd 图中存储这些。每个 `tensor.retain_grad()` 和每个需要其输入的操作都保留一个引用。

### 朴素的全检查点

将网络分成 `N` 段。前向期间，只存储每段的*输入*。当反向需要中间量时，重新运行该段的前向传播来实例化它们，然后求导。

例子：32 层 transformer 分成 32 段，每段 1 层。

- 内存：32 个层输入（小）vs 32 * 每层激活量（巨大）。
- 额外计算：每段 1 次额外前向，即总共约 33% 更多前向 FLOPs（因为反向是前向的 2 倍，完整步骤变成 1 + 1 + 2 = 4 个单位而非 1 + 2 = 3）。

这是原始 Chen et al. 2016 配方：每 `sqrt(L)` 层一个检查点，以平衡内存和计算。对于 L=64，那是 8 个检查点。

### 选择性检查点（Korthikanti 2022）

不是所有激活都花同样的代价。注意力 softmax 输出是 `B*L*L*heads`，随序列长度*二次方*增长。FFN 隐藏激活是 `B*L*4d`，线性增长。对于长序列，softmax 占主导。

选择性检查点保留便宜存储的激活（线性投影、残差），只重计算贵的（注意力）。你花最少 FLOPs 重计算，但节省了 O(L^2) 内存。

Megatron-Core 将此实现为"选择性"激活重计算。用于大多数 2024+ 前沿训练运行。

### 卸载

重计算的替代方案：在前向和反向之间将激活发送到 CPU RAM。需要 PCIe 带宽；当空闲带宽超过重实例化成本时有益。混合策略很常见：检查点一些层，卸载其他层。

FSDP2 将卸载作为一等公民选项。当 GPU 受内存瓶颈但 CPU-GPU 传输有 headroom 时，卸载很有效。

### 重计算成本模型

每步 FLOPs，朴素检查点每 `k` 层（共 `L` 层）：

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # 段内每层一次额外前向
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

使用选择性检查点，你只重计算注意力内核，而非整层：

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 内存节省模型

每层激活量：`A`。对于 `L` 层，总激活内存：`L * A`。

完全检查点（段大小 1）：只存储 `L * input_volume`（约 `L * 1/10 A` 对于标准 transformer）。节省约 `9 * L * A * 1/10`。

每 `k` 层检查点：存储 `L/k * A` 加上段内 `k-1` 层当量。

在 `k = sqrt(L)` 时，内存和重计算成本都随 `sqrt(L)` 缩放 — 对于均匀成本层，这是最优权衡。

### 何时不检查点

- 流水线阶段中已经 in-flight 的最内层。它们反正得完成。
- 如果阶段的首层和末层主导该阶段的计算（Transformer 中很少见）。
- 已经使用 FlashAttention 的注意力内核 — Flash 已经快速重计算 softmax，所以在层之上额外加检查点收效甚微。

### 实现模式

1. **函数包装器：** 将一段用 `torch.utils.checkpoint.checkpoint(fn, input)` 包装。PyTorch 只存储 `input`，在反向时重计算其他一切。

2. **基于装饰器：** 标记层为可检查点的；训练器在配置时决定哪些段被包装。

3. **手动显式重计算：** 自己写反向传播，调用一个自定义 `recompute_forward`，用存储的输入重复前向。

三者给出相同的函数结果。包装器是标准习惯用法。

### 与 TP / PP / FP8 的交互

- **Tensor 并行：** 检查点输入必须在重计算时聚集或重新散布；处理通信成本。
- **Pipeline 并行：** 典型模式是检查点每个流水线阶段的前向，这样反向顺序的 microbatch 可以复用激活内存。
- **FP8 重计算：** 重计算期间更新的 amax 历史必须匹配原始前向的，否则 FP8 缩放会漂移。大多数框架会快照缩放。

## 构建它

### 第 1 步：带段落的玩具模型

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 第 2 步：朴素反向需要所有激活

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 第 3 步：每 k 层检查点的内存

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 第 4 步：成本模型

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 第 5 步：内存估算器

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 第 6 步：最优段大小

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 第 7 步：选择性检查点决策

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 使用它

- **torch.utils.checkpoint**：`from torch.utils.checkpoint import checkpoint` — PyTorch 中的规范包装器。包装一个函数；只存储输入，在反向时重计算。
- **Megatron-Core 激活重计算**：支持 `selective`、`full` 和 `block` 模式。2024+ 前沿训练的标准。
- **FSDP2 卸载**：`module.to_empty(device="cpu")` 配合 FSDP2 中的 `offload_policy` 将激活分片到 CPU 而非重计算。
- **DeepSpeed ZeRO-Offload**：优化器状态和激活的 CPU 卸载，补充检查点。

## 交付它

本课产出 `outputs/prompt-activation-recompute-policy.md` — 一个接受模型配置（层数、hidden、seq、batch）和可用 GPU 内存，并发出每层重计算策略（无 / 选择性 / 完全 / 卸载）的提示。

## 练习

1. 验证正确性。运行 `model_forward` + `model_backward`（完整激活）vs `model_forward_checkpointed` + `model_backward_checkpointed`（分段）。参数梯度必须精确到机器精度相同。

2. 扫描段大小 `k` 从 1 到 `L`。绘制 FLOPs 开销和内存。找到曲线拐点。

3. 实现选择性检查点：存储注意力模块输入，但不存储其中间量。在 seq=8192 的 32 层模型上测量 FLOPs 开销 vs 全层检查点。

4. 添加卸载。将段输入保存到模拟"CPU 缓冲区"（一个单独的列表）。将"PCIe 带宽"测量为 bytes/time，找出卸载和重计算之间的平衡点。

5. 用 `torch.utils.checkpoint` 对真实 PyTorch transformer 进行基准测试。有和没有它的情况下测量内存（通过 `torch.cuda.max_memory_allocated`）和步时间。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|----------------------|
| 梯度检查点 | "通过重做前向节省内存" | 只存储段输入；在反向时重计算中间量以获得梯度支持张量 |
| 激活重计算 | "与检查点相同" | 同一技术的高性能计算命名 |
| 段大小（k） | "每检查点多少层" | 其中间量被丢弃并一起重新实例化的层数 |
| 选择性检查点 | "Korthikanti 的技巧" | 只重计算贵存储的激活（注意力 softmax）；保留便宜的 |
| 完全检查点 | "朴素版本" | 重计算每段每层的中间量 |
| 块检查点 | "粗粒度" | 检查点整个 transformer 块；最大粒度 |
| FLOPs 开销 | "计算税" | 每步额外 FLOPs = 重计算 FLOPs / (fwd + bwd FLOPs)；朴素 33%，选择性 5% |
| 激活卸载 | "转移到 CPU" | 在前向→反向之间将激活移到 CPU RAM；重计算的替代方案 |
| sqrt-L 规则 | "经典最优" | 对于均匀成本层，最优检查点间隔是 sqrt(L) 层 |
| 注意力 softmax 量 | "O(L^2) 问题" | L^2 * heads * batch 浮点数；在长上下文下主导激活内存 |

## 延伸阅读

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- 形式化梯度检查点的原始论文
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- 选择性激活重计算及正式成本分析
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- 通过反向模式重实例化实现常数内存的替代方法
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) --规模化激活卸载
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) -- 标准 API
- [Megatron-Core activation recomputation documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) -- 选择性、完全和块模式