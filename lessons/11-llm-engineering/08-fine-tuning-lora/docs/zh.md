# LoRA 与 QLoRA 微调

> 全量微调一个 7B 模型需要 56GB 显存。你没有这么多。绝大多数公司也没有。LoRA 让你只需训练不到 1% 的参数，就能在 6GB 显存内微调同一个模型。这不是妥协——在大多数任务上它能达到全量微调的质量。整个开源微调生态都建立在这一招上。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 10，第六课（指令微调 / SFT）
**时间：** 约 75 分钟
**相关：** 阶段 10 从零讲解 SFT/DPO 循环。本课将把这些循环接入 2026 年的 PEFT 工具链（PEFT、TRL、Unsloth、Axolotl、LLaMA-Factory）。

## 学习目标

- 通过向预训练模型的注意力层注入低秩适配矩阵（A 和 B）来实现 LoRA
- 计算 LoRA 与全量微调的参数量节省：维度为 d_model、秩为 r 的情况下，只需训练 2*r*d 个参数，而非 d² 个
- 使用 QLoRA（4 位量化基座 + LoRA 适配器）在消费级 GPU 显存内微调模型
- 将 LoRA 权重合并回基座模型用于部署，并对比带适配器与不带适配器的推理速度

## 问题

你有一个基座模型。Llama 3 8B。你想让它用你公司的风格回答客户支持工单。SFT 是答案。但 SFT 有成本问题。

全量微调会更新模型中的每一个参数。Llama 3 8B 有 80 亿个参数。在 fp16 下，每个参数占 2 字节。仅加载权重就需要 16GB。训练过程中，你还需要梯度（16GB）、Adam 优化器状态（动量 + 方差共 32GB）以及激活值。总计：一个 8B 模型大约需要 56GB 显存。

一张 A100 80GB 勉强能装下。两张 A100 在云服务商那里每小时费用 3-4 美元。在 50,000 个样本上训练 3 个 epoch 需要 6-10 小时，费用为 30-40 美元/次实验。跑 10 个实验调好超参数，在你实际部署之前就已经花了 400 美元。

把这个数字放大到 Llama 3 70B，情况就变得荒谬了。仅权重就需要 140GB。你需要一个集群。每次实验费用 100 美元以上。

还有一个更深层的问题。全量微调会修改模型中的每一个权重。如果你在客户支持数据上微调，可能会损害模型的通用能力。这叫做灾难性遗忘。模型在你的任务上变强了，却在其他所有事情上变弱了。

你需要一种参数量更少、显存占用更低、且不会破坏模型现有知识的方法。

## 概念

### LoRA：低秩适配

2021 年 6 月，微软的 Edward Hu 和同事发表了 LoRA。论文的洞察：微调过程中的权重更新具有低内在秩。你不需要更新一个 4096×4096 权重矩阵中的全部 1670 万个参数。更新中有用的信息可以被秩为 16 或 32 的矩阵捕获。

数学如下。标准线性层计算：

```
y = Wx
```

其中 W 是一个 d_out × d_in 矩阵。对于一个 4096×4096 的注意力投影，这对应 16,777,216 个参数。

LoRA 冻结 W 并添加一个低秩分解：

```
y = Wx + BAx
```

其中 B 是 (d_out × r)，A 是 (r × d_in)。秩 r 远小于 d——通常是 8、16 或 32。

在 4096×4096 层上取 r=16：
- 原始参数：4096 × 4096 = 16,777,216
- LoRA 参数：(4096 × 16) + (16 × 4096) = 65,536 + 65,536 = 131,072
- 缩减比例：131,072 / 16,777,216 = 0.78%

你只需训练 0.78% 的参数，却获得了 95-100% 的质量。

```mermaid
graph LR
    X["输入 x"] --> W["冻结的 W (d × d)"]
    X --> A["A (r × d)"]
    A --> B["B (d × r)"]
    W --> Plus["+ (合并)"]
    B --> Plus
    Plus --> Y["输出 y"]

    style W fill:#1a1a2e,stroke:#e94560,color:#fff
    style A fill:#0f3460,stroke:#16213e,color:#fff
    style B fill:#0f3460,stroke:#16213e,color:#fff
```

A 用随机高斯初始化。B 初始化为零。这意味着 LoRA 的贡献从零开始——模型从原始行为开始训练，然后逐渐学习适应。

### 缩放因子：Alpha

LoRA 引入了一个缩放因子 alpha，用于控制低秩更新对输出的影响程度：

```
y = Wx + (alpha / r) * BAx
```

当 alpha = r 时，缩放为 1x。当 alpha = 2r（常见的默认值）时，缩放为 2x。这个超参数独立于基础学习率，控制 LoRA 路径的学习率。

实用指导：
- alpha = 2 * rank 是社区的常见约定（原始论文在大多数实验中使用的 alpha = rank）
- alpha = rank 给出 1x 缩放，保守但稳定
- alpha 越高意味着每步更新越大，可以加速收敛或导致不稳定

### LoRA 应用在哪些层

Transformer 有很多线性层。你不需要在所有层都加 LoRA。原始论文测试了不同的组合：

| 目标层 | 可训练参数 (7B) | 质量 |
|--------------|----------------------|---------|
| 仅 q_proj | 4.7M | 良好 |
| q_proj + v_proj | 9.4M | 更好 |
| q_proj + k_proj + v_proj + o_proj | 18.9M | 注意力方面最佳 |
| 所有线性层（注意力 + MLP） | 37.7M | 收益边际，参数翻倍 |

大多数任务的最佳选择：q_proj + v_proj。这针对的是自注意力中的 query 和 value 投影，它们控制模型关注什么以及提取什么信息。为复杂任务（如代码生成）添加 MLP 层有帮助，但会使参数数量翻倍，而在简单任务上收益递减。

### 秩的选择

秩 r 控制适应的表达力：

| 秩 | 每层可训练参数 | 最佳场景 |
|------|---------------------------|----------|
| 4 | 32,768 | 简单分类、情感分析 |
| 8 | 65,536 | 单领域问答、摘要 |
| 16 | 131,072 | 多领域任务、指令跟随 |
| 32 | 262,144 | 复杂推理、代码生成 |
| 64 | 524,288 | 大多数任务收益递减 |
| 128 | 1,048,576 | 很少有充分理由使用 |

Hu 等人表明，对于简单任务，r=4 已经能捕获大部分适应。r=8 和 r=16 是实际中最常见的选择。超过 r=64 很少能提升质量，反而开始失去 LoRA 的显存优势。

### QLoRA：4 位量化 + LoRA

2023 年 5 月，华盛顿大学的 Tim Dettmers 和同事发表了 QLoRA。思路：将冻结的基座模型量化到 4 位精度，然后在上面以 fp16 附加 LoRA 适配器。

这从根本上改变了显存方程：

| 方法 | 权重显存 (7B) | 训练显存 (7B) | 所需 GPU |
|--------|-------------------|---------------------|-------------|
| 全量微调 (fp16) | 14GB | ~56GB | 1× A100 80GB |
| LoRA (fp16 基座) | 14GB | ~18GB | 1× A100 40GB |
| QLoRA (4 位基座) | 3.5GB | ~6GB | 1× RTX 3090 24GB |

QLoRA 做出了三个技术贡献：

**NF4（4 位标准浮点）**：专为神经网络权重设计的新数据类型。神经网络权重大致服从正态分布。NF4 将 16 个量化级别放置在标准正态分布的分位数上。这在信息论上对正态分布数据是最优的。与均匀 4 位量化（INT4）或标准 Float4 相比，它损失的信息更少。

**双重量化**：量化常数本身也占用显存。每 64 个权重需要一个 fp32 缩放因子（4 字节）。对于一个 7B 模型，这是额外的 0.4GB。双重量化将这些常数进一步量化到 fp8，将开销减少到 0.1GB。很小，但累加起来很可观。

**分页优化器**：训练期间，优化器状态（Adam 的动量和方差）在长序列上可能超出 GPU 显存。分页优化器使用 NVIDIA 的统一内存，当 GPU 显存耗尽时自动将优化器状态分页到 CPU RAM，需要时再分页回来。这以一定的吞吐量为代价防止 OOM 崩溃。

### 质量问题

减少参数量或量化基座会损害质量吗？多篇论文的结果：

| 方法 | MMLU (5-shot) | MT-Bench | HumanEval |
|--------|--------------|----------|-----------|
| 全量微调 (Llama 2 7B) | 48.3 | 6.72 | 14.6 |
| LoRA r=16 | 47.9 | 6.68 | 14.0 |
| QLoRA r=16 (NF4) | 47.5 | 6.61 | 13.4 |
| QLoRA r=64 (NF4) | 48.1 | 6.70 | 14.2 |

LoRA 在 r=16 时在大多数基准测试上与全量微调的差距在 1% 以内。QLoRA 在 r=16 时又损失了一小部分。QLoRA 在 r=64 时基本追平全量微调，同时节省了 90% 的显存。

### 现实成本

在 50,000 个样本上微调 Llama 3 8B（3 个 epoch）：

| 方法 | GPU | 时间 | 费用 |
|--------|-----|------|------|
| 全量微调 | 2× A100 80GB | 8 小时 | ~$32 |
| LoRA r=16 | 1× A100 40GB | 4 小时 | ~$8 |
| QLoRA r=16 | 1× RTX 4090 24GB | 6 小时 | ~$5 |
| QLoRA r=16 (Unsloth) | 1× RTX 4090 24GB | 2.5 小时 | ~$2 |
| QLoRA r=16 | 1× T4 16GB | 12 小时 | ~$4 |

在单张消费级 GPU 上用 QLoRA 的费用比一顿午餐还便宜。这就是为什么开源权重微调社区在 2023 年爆发，以及为什么在 2026 年每个训练框架默认都附带 QLoRA。

### 2026 年 PEFT 技术栈

| 框架 | 是什么 | 选择场景 |
|-----------|-----------|-----------|
| **Hugging Face PEFT** | 标准的 LoRA/QLoRA/DoRA/IA3 库 | 你需要底层控制，且你的训练循环已经在用 `transformers.Trainer` |
| **TRL** | HF 的强化反馈训练器（SFT、DPO、GRPO、PPO、ORPO） | 你在 SFT 之后需要 DPO/GRPO；构建在 PEFT 之上 |
| **Unsloth** | 前向/反向传播的 Triton 内核重写 | 你想获得 2-5 倍加速 + 一半显存，且无精度损失；适用于 Llama/Mistral/Qwen 系列 |
| **Axolotl** | 基于 PEFT + TRL + DeepSpeed + Unsloth 的 YAML 配置包装器 | 你想要可复现的、版本控制的训练运行 |
| **LLaMA-Factory** | 基于 PEFT + TRL 的 GUI/CLI/API | 你想零代码微调；支持 100+ 模型家族 |
| **torchtune** | 原生 PyTorch 配方，不依赖 `transformers` | 你想要最小依赖，且你的团队已经在标准化使用 PyTorch |

经验法则：研究用途或一次性实验 → PEFT。可复现的生产流水线 → 启用了 Unsloth 内核的 Axolotl。临时原型 → LLaMA-Factory。

### 合并适配器

训练结束后，你有两样东西：冻结的基座模型和一个小型 LoRA 适配器（通常 10-100MB）。你有两个选择：

1. **保持分离**：加载基座模型，在上面加载适配器。为不同任务切换适配器。这就是如何从一个基座模型提供多个微调变体。

2. **永久合并**：计算 W' = W + (alpha/r) * BA，并将结果保存为一个新的完整模型。合并后的模型与原始模型大小相同。没有推理开销。无需管理适配器。

对于提供多个任务（客户支持适配器、代码适配器、翻译适配器），保持分离。对于部署单个专用模型，合并。

组合多个适配器的高级合并技术：

- **TIES-Merging**（Yadav 等人，2023）：裁剪小幅度参数，解决符号冲突，然后合并。减少适配器之间的干扰。
- **DARE**（Yu 等人，2023）：在合并前随机丢弃适配器参数，然后重新缩放其余参数。在组合能力方面出奇地有效。
- **任务算术**：简单地将适配器权重相加或相减。添加一个"代码"适配器和一个"数学"适配器通常会产生一个两者都擅长的模型。

### 何时不微调

微调是第三选择，不是第一选择。

**第一：提示工程。** 写一个更好的系统提示。添加少样本示例。使用思维链。这不花钱，几分钟就能完成。如果提示能让你达到 80% 的效果，你可能不需要微调。

**第二：RAG。** 如果模型需要了解你的特定数据（文档、知识库、产品目录），检索比将其嵌入权重更便宜、更易于维护。参见第六课。

**第三：微调。** 当你需要模型采用一种通过提示无法实现的特定风格、格式或推理模式时使用。当你需要一致的结构化输出时。当你需要将大模型蒸馏到小模型时。当延迟很重要且你无法承受少样本提示带来的额外 token 时。

```mermaid
graph TD
    Start["需要更好的模型行为？"] --> PE["尝试提示工程"]
    PE -->|"有效"| Done["上线"]
    PE -->|"不够"| RAG["需要外部知识？"]
    RAG -->|"是"| RAGBuild["构建 RAG 流水线"]
    RAG -->|"否，需要风格/格式改变"| FT["用 LoRA/QLoRA 微调"]
    RAGBuild -->|"有效"| Done
    RAGBuild -->|"也需要风格改变"| FT
    FT --> Done

    style Start fill:#1a1a2e,stroke:#e94560,color:#fff
    style Done fill:#0f3460,stroke:#16213e,color:#fff
```

## 动手实现

我们从零在纯 PyTorch 中实现 LoRA。不需要库。没有魔法。你将构建 LoRA 层、将其注入模型、训练它，然后将权重合并回去。

### 第一步：LoRA 层

```python
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.A = nn.Parameter(torch.randn(in_features, rank) * (1 / math.sqrt(rank)))
        self.B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        return (x @ self.A @ self.B) * self.scaling
```

A 用缩放的随机值初始化。B 初始化为零。乘积 BA 从零开始，所以模型从其原始行为开始。

### 第二步：带 LoRA 的线性层包装

```python
class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.lora(x)
```

原始线性层被冻结。只有 LoRA 参数（A 和 B）是可训练的。

### 第三步：向模型注入 LoRA

```python
def inject_lora(model, target_modules, rank=8, alpha=16):
    for param in model.parameters():
        param.requires_grad = False

    lora_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(t in name for t in target_modules):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = dict(model.named_modules())[parent_name]
                lora_linear = LinearWithLoRA(module, rank, alpha)
                setattr(parent, child_name, lora_linear)
                lora_layers[name] = lora_linear
    return lora_layers
```

首先，冻结模型中的每个参数。然后遍历模型树，找到与目标名称匹配的线性层，并用 LoRA 包装版本替换它们。LoRA 的 A 和 B 矩阵是整个模型中唯一可训练的参数。

### 第四步：统计参数

```python
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": 100 * trainable / total if total > 0 else 0
    }
```

### 第五步：将权重合并回去

```python
def merge_lora_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            with torch.no_grad():
                merged = (
                    module.lora.A @ module.lora.B
                ) * module.lora.scaling
                module.linear.weight.data += merged.T
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
            setattr(parent, child_name, module.linear)
```

合并后，LoRA 层消失了。模型与原始大小相同，适应已融入权重中。没有推理开销。

### 第六步：模拟 QLoRA 量化

```python
def quantize_to_nf4(tensor, block_size=64):
    blocks = tensor.reshape(-1, block_size)
    scales = blocks.abs().max(dim=1, keepdim=True).values / 7.0
    scales = torch.clamp(scales, min=1e-8)
    quantized = torch.round(blocks / scales).clamp(-8, 7).to(torch.int8)
    return quantized, scales

def dequantize_from_nf4(quantized, scales, original_shape):
    dequantized = quantized.float() * scales
    return dequantized.reshape(original_shape)
```

这通过将权重映射到 64 个块的 16 个离散级别来模拟 4 位量化。生产级 QLoRA 使用 bitsandbytes 库在 GPU 上进行真正的 NF4 量化。

### 第七步：训练循环

```python
def train_lora(model, data, epochs=5, lr=1e-3, batch_size=4):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        indices = torch.randperm(len(data["inputs"]))

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            x = data["inputs"][batch_idx]
            y = data["targets"][batch_idx]

            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

    return losses
```

### 第八步：完整演示

```python
def demo():
    torch.manual_seed(42)
    d_model = 256
    n_classes = 10

    model = nn.Sequential(
        nn.Linear(d_model, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, n_classes),
    )

    n_samples = 500
    x = torch.randn(n_samples, d_model)
    y = torch.randint(0, n_classes, (n_samples,))
    y_onehot = torch.zeros(n_samples, n_classes).scatter_(1, y.unsqueeze(1), 1.0)

    data = {"inputs": x, "targets": y_onehot}

    params_before = count_parameters(model)

    lora_layers = inject_lora(
        model, target_modules=["0", "2"], rank=8, alpha=16
    )

    params_after = count_parameters(model)

    losses = train_lora(model, data, epochs=20, lr=1e-3)

    merge_lora_weights(model)
    params_merged = count_parameters(model)

    return {
        "params_before": params_before,
        "params_after": params_after,
        "params_merged": params_merged,
        "losses": losses,
    }
```

演示创建了一个小模型，向其中两层注入 LoRA，训练它，然后将权重合并回去。参数数量从全量可训练下降到 LoRA 训练期间约 1% 可训练，然后在合并后恢复到原始架构。

## 使用

使用 Hugging Face 生态，在真实模型上使用 LoRA 大约需要 20 行代码：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

对于 QLoRA，添加 bitsandbytes 量化：

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

model = get_peft_model(model, lora_config)
```

就这样。相同的训练循环。相同的数据流水线。基座模型现在以 4 位存储，LoRA 适配器以 fp16 训练，整个系统可以装进 6GB 显存。

使用 Hugging Face Trainer 进行训练：

```python
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca", split="train[:5000]")

training_args = TrainingArguments(
    output_dir="./lora-llama",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("./lora-adapter")
```

保存的适配器为 10-100MB。基座模型保持不变。你可以在 Hugging Face Hub 上分享适配器，而无需重新分发完整模型。

## 上线

本课产出：
- `outputs/prompt-lora-advisor.md` —— 一个帮助你决定 LoRA 秩、目标模块和超参数的提示词
- `outputs/skill-fine-tuning-guide.md` —— 一个教 AI 智能体何时及如何微调的技能

## 练习

1. **秩消融研究。** 用秩 2、4、8、16、32 和 64 运行演示。绘制最终损失与秩的关系曲线。找到收益递减点——在该点加倍秩不再使损失减半。对于 256 维特征的简单分类任务，这应该在 r=8-16 左右。

2. **目标模块比较。** 修改 inject_lora 分别只针对层 "0"、只针对层 "2"、只针对层 "4" 和全部三层。为每个变体训练 20 个 epoch。对比收敛速度和最终损失。这反映了真实场景中针对 q_proj 与 v_proj 与所有线性层的选择。

3. **量化误差分析。** 取训练后模型在 quantize_to_nf4 / dequantize_from_nf4 前后的权重矩阵。计算均方误差、最大绝对误差，以及原始权重与重构权重之间的相关性。尝试不同的 block_size 值：32、64、128 和 256。

4. **多适配器服务。** 在数据的不同子集（偶数索引 vs 奇数索引）上训练两个 LoRA 适配器。保存两个适配器。加载基座模型一次，然后切换适配器，验证每个适配器在同一输入上产生不同的输出。这就是生产系统如何从一个基座模型提供多个微调模型。

5. **合并 vs 未合并推理。** 在 merge_lora_weights 前后的同一 100 个输入上对比 LoRA 模型的输出。验证输出是相同的（在 1e-5 的浮点容差内）。然后对两者进行推理速度基准测试——合并版应该略快，因为它是一次矩阵乘法而非两次。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|----------------------|
| LoRA | "高效微调" | 低秩适配：冻结基座权重，训练两个小矩阵 A 和 B，其乘积逼近完整权重更新 |
| QLoRA | "在笔记本上微调" | 量化 LoRA：以 4 位 NF4 加载基座模型，在其上以 fp16 训练 LoRA 适配器，使 7B 模型可以在 6GB 显存内微调 |
| 秩 (r) | "模型能学多少" | A 和 B 矩阵的内部维度；控制表达力与参数量之间的权衡 |
| Alpha | "LoRA 学习率" | 应用于 LoRA 输出的缩放因子；alpha/r 缩放适应对最终输出的贡献 |
| NF4 | "4 位量化" | 标准浮点 4：一种 4 位数据类型，量化级别位于正态分布分位数上，对神经网络权重最优 |
| 适配器 | "训练的小部分" | LoRA 的 A 和 B 矩阵保存为单独文件（10-100MB），可加载到基座模型的任何副本之上 |
| 目标模块 | "哪些层用 LoRA" | 注入 LoRA 适配器的特定线性层（q_proj、v_proj 等） |
| 合并 | "将其嵌入" | 计算 W + (alpha/r) * BA 并替换原始权重，消除推理时的适配器开销 |
| 分页优化器 | "训练时不 OOM" | 当 GPU 显存耗尽时将优化器状态（Adam 动量、方差）卸载到 CPU |
| 灾难性遗忘 | "微调破坏了其他一切" | 更新所有权重导致模型丧失先前学习到的能力 |

## 延伸阅读

- Hu 等人，"LoRA: Low-Rank Adaptation of Large Language Models"（2021）—— 引入低秩分解方法的原始论文，在 GPT-3 175B 上以低至 4 的秩测试
- Dettmers 等人，"QLoRA: Efficient Finetuning of Quantized Language Models"（2023）—— 引入了 NF4、双重量化和分页优化器，使 65B 模型能在单张 48GB GPU 上微调
- PEFT 库文档（huggingface.co/docs/peft）—— Hugging Face 生态中 LoRA、QLoRA 和其他参数高效方法的标准库
- Yadav 等人，"TIES-Merging: Resolving Interference When Merging Models"（2023）—— 在不降低质量的情况下组合多个 LoRA 适配器的技术
- [Rafailov 等人，"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"（NeurIPS 2023）](https://arxiv.org/abs/2305.18290) —— DPO 推导；SFT 之后的偏好调优阶段，无需奖励模型。
- [TRL 文档](https://huggingface.co/docs/trl/) —— `SFTTrainer`、`DPOTrainer`、`KTOTrainer` 的官方参考，以及与 PEFT/bitsandbytes/Unsloth 的集成接口。
- [Unsloth 文档](https://docs.unsloth.ai/) —— 融合内核使微调吞吐量翻倍、显存减半；TRL 的性能层。
- [Axolotl 文档](https://axolotl-ai-cloud.github.io/axolotl/) —— YAML 配置的多 GPU SFT/DPO/QLoRA 训练器；配置即代码的替代方案，替代手工编写脚本。