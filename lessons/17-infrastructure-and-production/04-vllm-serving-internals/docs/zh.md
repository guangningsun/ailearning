# vLLM 服务内部原理：PagedAttention、连续批处理与分块预填充

> vLLM 在 2026 年的主导地位建立在三个叠加的默认特性上，而非某个单一技巧。PagedAttention 始终开启。连续批处理在每个解码迭代之间将新请求注入活跃批次。分块预填充将长提示切分为 ~512 token 的片段，使解码 token 永不饥饿。三个特性全部开启时，Llama 3.3 70B FP8 在单卡 H100 SXM5 上以 128 并发可达到 2,200-2,400 tok/s——比 vLLM 自身的默认配置高约 25%，是朴素 PyTorch 循环的 3-4 倍。本课从你能画出架构图的角度解读调度器和注意力内核，并在 `code/main.py` 中用一个玩具连续批处理器来实现 vLLM 的预填充与解码调度方式。

**类型：** 学习型
**语言：** Python（标准库，玩具连续批处理调度器）
**前置条件：** 阶段 17 · 01（模型服务）、阶段 11（LLM 工程）
**时间：** 约 75 分钟

## 学习目标

- 将 PagedAttention 解释为 KV 缓存分配器：块、块表，以及为何在生产负载下碎片化保持在 4% 以下。
- 从迭代层面绘制连续批处理的架构图：完成的序列如何离开批次、新序列如何加入而不必排空。
- 用一句话描述分块预填充，并指出它保护的是哪个延迟指标（提示：是 TTFT 尾延迟，而非平均吞吐）。
- 指出 2026 年 vLLM v0.18.0 中让所有优化同时开启的团队踩坑的那个坑。

## 问题

朴素的 PyTorch 服务循环一次只运行一个请求：分词、预填充、解码直到 EOS、返回。一个用户时运行良好。一百个用户时，就是一队耐心等待的人。显而易见的修复——静态批处理——将每个请求填充到窗口中最长提示的长度，将每个解码填充到最长预期输出的长度，并让整个批次在最慢的序列上停滞。你为从未使用过的填充付费，而快请求要等待慢请求。

vLLM 一次性解决三个问题。PagedAttention 阻止经典连续分配方式下 KV 缓存碎片化吞噬 60-80% GPU 内存的问题。连续批处理允许请求在每个解码迭代之间加入和离开批次，因此批次始终充满真实工作。分块预填充将 32k token 的提示切分为 ~512 token 的片段，与解码交错执行，因此长提示不会冻结 GPU 上每个解码 token。

2026 年的生产默认配置是三个特性全部开启。你需要理解每个特性做什么，因为故障模式都在调度器上，不在模型上。

## 概念

### PagedAttention 作为虚拟内存系统

KV 缓存是 `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element`，每个序列约 1.25 GB（BF16）。如果为每个请求预分配 8192 个槽位，但平均请求只使用 1500 token，你就浪费了约 82% 的 HBM。经典批处理为此付出了浪费代价。

PagedAttention 从 OS 虚拟内存借用了这个思路。KV 缓存并非每个序列连续存储，而是以固定大小的块（默认 16 token）分配。每个序列有一个块表，将逻辑 token 位置映射到物理块 ID。当序列增长超过已分配块时，添加一个块。当序列完成时，其块归还给池。

碎片化从 60-80%（经典）降至 4% 以下（PagedAttention）。你不通过某个标志启用 PagedAttention——它是 vLLM 唯一的分配器。旋钮是 `--gpu-memory-utilization`（默认 0.9），它告诉 vLLM 在加载权重和激活后为 KV 块预留多少 HBM。

### 迭代层面的连续批处理

旧的"动态批处理"等待一个窗口（比如 10 ms）填满一个批次，然后运行预填充 + 解码 + 解码 + 解码，直到每个序列完成。快序列提前离开并闲置，而 GPU 还在处理慢序列。

连续批处理在每个解码步骤之间操作。将运行中的序列集合称为 `RUNNING` 列表。在每次迭代中：

1. 任何刚达到 EOS 或 max_tokens 的 `RUNNING` 中的序列被移除。
2. 调度器查看等待队列。如果有空闲 KV 块，则接纳新序列（预填充或恢复）。
3. 前向传递在 `RUNNING` 中的任何内容上运行，为每个序列发出一个新 token。

批次大小从不填充到固定数字。处于输出不同位置的序列共享一次融合前向。在 2026 年 vLLM 中，这被称为 `V1 调度器`。关键不变量：调度器每个解码迭代运行一次，而非每个请求运行一次。

### 分块预填充保护 TTFT 尾延迟

预填充是计算密集型的。Llama 3.3 70B 在单卡 H100 上，32k token 提示的纯预填充约需 800 ms。预填充运行时，批次中每个其他序列的解码 token 都在等待。在服务循环中，一个长提示的首 token 延迟（TTFT）成为其他数十个用户的 token 间延迟（ITL）波动。

分块预填充将预填充分割成固定大小的块（默认 512 token），并将每个块作为一个单元进行调度。在块之间，调度器可以推进解码序列一个 token。你用小的绝对预填充延迟损失（每块几 ms）换取低得多的解码时抖动。在已发布的基准测试中，混合负载下 P99 ITL 从约 50 ms 降至约 15 ms。

### 三个默认特性的交互

三个特性都相互依赖。PagedAttention 给调度器提供了可交易的细粒度 KV 资源。连续批处理需要这种细粒度资源，以便接纳新序列不会强制全局重排。分块预填充是调度器在同一个 `RUNNING` 列表上做出的决策——它是另一个调度器策略，而非独立系统。

你不需要知道每个标志。你需要知道调度器优化的是什么：在分块预填充切片的前提下，在 KV 块预算下的 goodput。

### 2026 年 v0.18.0 的坑

在 vLLM v0.18.0 中，你不能将 `--enable-chunked-prefill` 与草稿模型推测解码（`--speculative-model`）结合使用。文档化的例外是 V1 调度器中的 N-gram GPU 推测解码。未阅读发布说明就开启每个标志的团队会在启动时收到运行时错误，而非软回归。如果你的推测收益值得启用分块预填充，请重新考虑——2026 年的正确答案通常是 EAGLE-3 不带分块预填充，而非一个根本无法编译的草稿模型加分块预填充组合。

### 你应该记住的数字

- Llama 3.3 70B FP8，H100 SXM5，128 并发，三个特性全开：2,200-2,400 tok/s。
- 同模型，默认 vLLM（无分块预填充）：约 1,800 tok/s。
- 同模型，朴素 PyTorch 前向循环：约 600 tok/s。
- 生产负载下 PagedAttention 的 KV 碎片化浪费：<4%。
- 混合负载下 P99 ITL：分块预填充约 15 ms，无分块预填充约 50 ms。

### 调度器的样子

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # 在一个批次中调度预填充块 + 解码
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # 例如 512 token
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # 一次融合 GPU 调用
```

`code/main.py` 正是这个循环，使用标准库 Python，带有伪 token 计数和伪前向延迟。运行它可以看到分块预填充如何在长预填充期间保持解码序列存活。

## 使用它

`code/main.py` 模拟一个具有可切换特性的 vLLM 风格调度器。运行它可以看到：

- `NAIVE` 模式：一次一个请求，无批处理。
- `STATIC` 模式：填充并等待，经典批处理。
- `CONTINUOUS` 模式：迭代级接纳和释放。
- `CONTINUOUS + CHUNKED` 模式：预填充切片与解码交错。

输出显示总吞吐量（虚拟秒内的 token 数）、TTFT 平均值和 P99 ITL。`CONTINUOUS + CHUNKED` 行应该在混合流量下表现最佳。

## 交付它

本课产出 `outputs/skill-vllm-scheduler-reader.md`。给定一个服务配置（批次大小、KV 内存利用率、分块预填充大小、推测配置），它生成一个调度器诊断报告，指出三个默认特性中哪个是瓶颈以及应调整什么。

## 练习

1. 运行 `code/main.py`。在混合短请求和长请求的工作负载上比较 `STATIC` 和 `CONTINUOUS`。吞吐量差距来自哪里——预填充效率、解码效率，还是尾延迟？
2. 修改玩具调度器以添加 `--max-num-batched-tokens`。对于运行 Llama 3.3 70B FP8 的 H100，什么是正确值？（提示：它是 KV 块大小和空闲块数量的函数，而非原始 HBM。）
3. 重新阅读 vLLM v0.18.0 发布说明。哪些标志组合是互斥的？列出它们。
4. 计算在 8192 最大值下，针对 1,000 个请求的跟踪（平均 1,500 输出 token，标准差 600 token）在 (a) 连续按请求分配和 (b) 16-token 块的 PagedAttention 下的 KV 缓存碎片化浪费。
5. 用一段话解释为何分块预填充有助于 P99 ITL 但单独对吞吐量无帮助。在实践中吞吐量收益来自哪里？

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|------------------------|
| PagedAttention | "KV 小技巧" | KV 缓存的固定大小块分配器；碎片化 <4% |
| 块表 | "页表" | 每个序列从逻辑 token 位置到物理 KV 块的映射 |
| 连续批处理 | "动态批处理，但正确" | 在每个解码迭代做出接纳/释放决策 |
| 分块预填充 | "预填充分割" | 将长预填充分割为 512-token 片段，与解码交错 |
| TTFT | "首 token 时间" | 预填充 + 队列 + 网络；在长提示下由预填充主导 |
| ITL | "token 间延迟" | 连续解码 token 之间的时间；在批次大小下主导 |
| Goodput | "满足 SLO 的吞吐量" | 每个请求仍命中 TTFT 和 ITL 目标下的 tok/s |
| V1 调度器 | "新调度器" | vLLM 的 2026 年调度器；N-gram 推测解码是与分块预填充兼容的路径 |
| `--gpu-memory-utilization` | "内存旋钮" | 权重和激活后为 KV 块预留的 HBM 比例 |

## 延伸阅读

- [vLLM 文档 — 推测解码](https://docs.vllm.ai/en/latest/features/spec_decode/) — 关于分块预填充和推测解码兼容性的官方来源。
- [vLLM 发布说明 (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) — 2026 年发布节奏和版本特定行为。
- [vLLM 博客 — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — 原创文章，至今仍定义着如何理解这个分配器。
- [PagedAttention 论文 (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) — 碎片化分析和调度器设计。
- [Aleksa Gordic — 深入 vLLM](https://www.aleksagordic.com/blog/vllm) — 带火焰图的详细 V1 调度器演练。