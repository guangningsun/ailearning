# SAM 3 与开放词汇分割

> 给模型一个文本提示和一张图像，得到所有匹配对象的 mask。SAM 3 让这成为一次前向传播。

**类型：** 使用 + 构建
**语言：** Python
**前置条件：** 阶段 4 第 7 课（U-Net）、阶段 4 第 8 课（Mask R-CNN）、阶段 4 第 18 课（CLIP）
**时间：** 约 60 分钟

## 学习目标

- 区分 SAM（仅视觉提示）、Grounded SAM / SAM 2（检测器 + SAM）和 SAM 3（通过 Promptable Concept Segmentation 实现原生文本提示）
- 解释 SAM 3 架构：共享 backbone + 图像检测器 + 基于记忆的视频跟踪器 + 存在性头 + 解耦检测器-跟踪器设计
- 使用 Hugging Face `transformers` 的 SAM 3 集成来进行文本提示检测、分割和视频跟踪
- 根据延迟、概念复杂度和部署目标在 SAM 3、Grounded SAM 2、YOLO-World 和 SAM-MI 之间选择

## 问题

2023 年的 SAM 是一个仅支持视觉提示的模型：你点击一个点或画一个框，它返回一个 mask。对于"把这张照片里所有的橘子都给我"这样的需求，你需要用一个检测器（Grounding DINO）来生成框，然后用 SAM 来分割每个框。Grounded SAM 把这变成了一条流水线，但它是两个冻结模型的级联，误差不可避免地累积。

SAM 3（Meta，2025 年 11 月，ICLR 2026）坍缩了这个级联。它接受一个简短的名词短语或一张图像示例作为提示，在一次前向传播中返回所有匹配的 mask 和实例 ID。这就是**Promptable Concept Segmentation（PCS）**。结合 2026 年 3 月的 Object Multiplex 更新（SAM 3.1），它能高效地跟踪视频中同一概念的多个实例。

本课讲的是这代表的结构性转变。2D 分割、检测和文本-图像定位已经合并成一个模型。生产问题不再是"我该串联哪条流水线"而是"哪个提示式模型能端到端处理我的用例"。

## 概念

### 三代演进

```mermaid
flowchart LR
    subgraph SAM1["SAM (2023)"]
        A1["Image + point/box prompt"] --> A2["ViT encoder"] --> A3["Mask decoder"]
        A3 --> A4["Mask for that prompt"]
    end
    subgraph GSAM2["Grounded SAM 2 (2024)"]
        B1["Text"] --> B2["Grounding DINO"] --> B3["Boxes"] --> B4["SAM 2"] --> B5["Masks + tracking"]
        B6["Image"] --> B2
        B6 --> B4
    end
    subgraph SAM3["SAM 3 (2025)"]
        C1["Text OR image exemplar"] --> C2["Shared backbone"]
        C3["Image"] --> C2
        C2 --> C4["Image detector + memory tracker<br/>+ presence head"]
        C4 --> C5["All matching masks<br/>+ instance IDs"]
    end

    style SAM1 fill:#e5e7eb,stroke:#6b7280
    style GSAM2 fill:#fef3c7,stroke:#d97706
    style SAM3 fill:#dcfce7,stroke:#16a34a
```

### Promptable Concept Segmentation

一个"概念提示"是一个简短的名词短语（`"yellow school bus"`、`"striped red umbrella"`、`"hand holding a mug"`）或一张图像示例。模型返回图像中匹配该概念的每个实例的分割 mask，外加每个匹配的唯一实例 ID。

这与经典视觉提示 SAM 有三个关键区别：

1. 无需逐实例提示 —— 一个文本提示返回所有匹配。
2. 开放词汇 —— 概念可以是任何能用自然语言描述的东西。
3. 一次返回多个实例，而非每个提示返回一个 mask。

### 关键架构组件

- **共享 backbone** —— 单个 ViT 处理图像。检测器头和基于记忆的跟踪器都从它读取。
- **存在性头** —— 预测概念是否在图像中存在。把"这个东西在吗？"和"它在哪里？"解耦。减少对不存在概念的误报。
- **解耦检测器-跟踪器** —— 图像级检测和视频级跟踪有独立的头，互不干扰。
- **记忆库** —— 存储跨帧的每实例特征，用于视频跟踪（与 SAM 2 使用的机制相同）。

### 大规模训练

SAM 3 在**400 万个独特概念**上训练，这些概念由一个数据引擎生成，该引擎通过 AI + 人工审查迭代标注和修正。新的 **SA-CO 基准**包含 27 万个独特概念，比之前的基准大 50 倍。SAM 3 在 SA-CO 上达到人类性能的 75-80%，在图像和视频 PCS 上比现有系统提升一倍。

### SAM 3.1 Object Multiplex

2026 年 3 月更新：**Object Multiplex** 引入了一个共享记忆机制，用于同时跟踪同一概念的多个实例。之前，跟踪 N 个实例意味着 N 个独立的记忆库。Multiplex 将其压缩为一个共享记忆加每实例查询。结果：在不牺牲准确率的情况下，多目标跟踪大幅加速。

### 2026 年 Grounded SAM 仍有价值的地方

- 当你需要换入一个特定的开放词汇检测器时（DINO-X、Florence-2）。
- 当 SAM 3 的许可证（Hugging Face 上受限）是障碍时。
- 当你需要比 SAM 3 提供的更强的检测器阈值控制时。
- 用于检测器组件上的研究 / 消融实验。

模块化流水线仍有价值。对于大多数生产工作，SAM 3 是更简单的答案。

### YOLO-World vs SAM 3

- **YOLO-World** —— 仅开放词汇检测器（无 mask）。实时。适合需要高 fps 框的应用。
- **SAM 3** —— 完整分割 + 跟踪。更慢但输出更丰富。

生产分工：YOLO-World 用于快速仅检测流水线（机器人导航、快速仪表盘），SAM 3 用于任何需要 mask 或跟踪的场景。

### SAM-MI 效率

SAM-MI（2025-2026）解决了 SAM 的解码器瓶颈。关键思路：

- **稀疏点提示** —— 使用少量精心选择的点而非密集提示；将解码器调用减少 96%。
- **浅层 mask 聚合** —— 将粗糙的 mask 预测合并为一个更清晰的 mask。
- **解耦 mask 注入** —— 解码器接收预计算的 mask 特征，而非重新运行。

结果：在开放词汇基准上比 Grounded-SAM 快约 1.6 倍。

### 三个模型的输出格式

所有模型返回相同的总体结构（boxes + labels + scores + masks + IDs），这很方便 —— 下游流水线不需要根据哪个模型运行来分支。

## 构建

### 第 1 步：提示词构建

构建一个把用户句子转换成 SAM 3 概念提示列表的辅助函数。这是"用户输入"和"模型输入"之间的边界。

```python
def split_concepts(sentence):
    """
    Heuristic splitter for multi-concept prompts.
    Returns list of short noun phrases.
    """
    for sep in [",", ";", "and", "or", "&"]:
        if sep in sentence:
            parts = [p.strip() for p in sentence.replace("and ", ",").split(",")]
            return [p for p in parts if p]
    return [sentence.strip()]

print(split_concepts("cats, dogs and balloons"))
```

SAM 3 每个前向传播接受一个概念；对于多概念查询，循环或批量处理。

### 第 2 步：后处理辅助函数

把 SAM 3 的原始输出转换成符合阶段 4 第 16 课流水线契约的干净检测列表。

```python
from dataclasses import dataclass
from typing import List

@dataclass
class ConceptDetection:
    concept: str
    instance_id: int
    box: tuple          # (x1, y1, x2, y2)
    score: float
    mask_rle: str       # run-length encoded


def rle_encode(binary_mask):
    flat = binary_mask.flatten().astype("uint8")
    runs = []
    prev, count = flat[0], 0
    for v in flat:
        if v == prev:
            count += 1
        else:
            runs.append((int(prev), count))
            prev, count = v, 1
    runs.append((int(prev), count))
    return ";".join(f"{v}x{c}" for v, c in runs)
```

RLE 保持响应 payload 小巧，即使对于许多高分辨率 mask 也如此。相同格式适用于 SAM 2、SAM 3、Grounded SAM 2。

### 第 3 步：统一开放词汇分割接口

把你有的任何后端（SAM 3、Grounded SAM 2、YOLO-World + SAM 2）包装在单一方法后面。你的下游代码不会因后端改变而改变。

```python
from abc import ABC, abstractmethod
import numpy as np

class OpenVocabSeg(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray, concept: str) -> List[ConceptDetection]:
        ...


class StubOpenVocabSeg(OpenVocabSeg):
    """
    Deterministic stub used for pipeline testing when real models are not loaded.
    """
    def detect(self, image, concept):
        h, w = image.shape[:2]
        return [
            ConceptDetection(
                concept=concept,
                instance_id=0,
                box=(w * 0.2, h * 0.3, w * 0.5, h * 0.8),
                score=0.89,
                mask_rle="0x100;1x50;0x200",
            ),
            ConceptDetection(
                concept=concept,
                instance_id=1,
                box=(w * 0.55, h * 0.25, w * 0.85, h * 0.75),
                score=0.74,
                mask_rle="0x80;1x40;0x220",
            ),
        ]
```

真正的 `SAM3OpenVocabSeg` 子类将包装 `transformers.Sam3Model` 和 `Sam3Processor`。

### 第 4 步：Hugging Face SAM 3 使用（参考）

对于实际模型，`transformers` 集成：

```python
from transformers import Sam3Processor, Sam3Model
import torch

processor = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3").eval()

inputs = processor(images=pil_image, return_tensors="pt")
inputs = processor.set_text_prompt(inputs, "yellow school bus")

with torch.no_grad():
    outputs = model(**inputs)

masks = processor.post_process_masks(
    outputs.masks, inputs.original_sizes, inputs.reshaped_input_sizes
)
boxes = outputs.boxes
scores = outputs.scores
```

一个提示，一次调用，所有匹配返回。

### 第 5 步：衡量 Grounded SAM 2 免费给你的东西

一个诚实的基准：当你在真实流水线中把 Grounded SAM 2 替换成 SAM 3 时会发生什么？

- 延迟：SAM 3 节省一次前向传播（无独立检测器），但模型本身更重；通常净中性或略有加速。
- 准确率：SAM 3 在稀有或组合概念（"striped red umbrella"）上明显更好；在常见单词概念上相似。
- 灵活性：Grounded SAM 2 让你换检测器（DINO-X、Florence-2、Grounding DINO 1.5）；SAM 3 是整体式的。

结论：SAM 3 是 2026 年开放词汇分割的默认选择。当你需要检测器灵活性或不同许可证条款时，Grounded SAM 2 仍然是正确答案。

## 使用

生产部署模式：

- **实时标注** —— SAM 3 + CVAT 的 label-as-text-prompt 功能。标注员选择标签名称；SAM 3 预标注每个匹配的实例。审核并修正。
- **视频分析** —— SAM 3.1 Object Multiplex 用于多目标跟踪；将帧馈送给基于记忆的跟踪器。
- **机器人** —— SAM 3 用于开放词汇操作（"拿起红色杯子"）；作为规划原语运行。
- **医学影像** —— 在医学概念上微调的 SAM 3；需要在 HF 上申请访问。

Ultralytics 在其 Python 包中包装了 SAM 3：

```python
from ultralytics import SAM

model = SAM("sam3.pt")
results = model(image_path, prompts="yellow school bus")
```

与 YOLO 和 SAM 2 相同的接口。

## 交付

本课产出：

- `outputs/prompt-open-vocab-stack-picker.md` —— 一个提示词，根据延迟、概念复杂度和许可证在 SAM 3 / Grounded SAM 2 / YOLO-World / SAM-MI 之间选择。
- `outputs/skill-concept-prompt-designer.md` —— 一个技能，把用户话语转换成格式良好的 SAM 3 概念提示（拆分、消歧、回退）。

## 练习

1. **（简单）** 在 10 张图像上用你选择的概念提示运行 SAM 3。与 SAM 2 + Grounding DINO 1.5 在相同图像上比较。报告每个模型漏掉了哪些概念。
2. **（中等）** 在 SAM 3 之上构建一个"点击包含 / 点击排除"UI：文本提示返回候选实例；用户点击保留哪些作为正向。输出最终概念集为 JSON。
3. **（困难）** 在自定义概念集（例如 5 种电子元件）上微调 SAM 3，每种 20 张标注图像。与同测试集上的零样本 SAM 3 比较；测量 mask IoU 改进。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|----------------------|
| 开放词汇分割 | "按文本分割" | 用自然语言描述对象产生 mask，而非固定标签集 |
| PCS | "Promptable Concept Segmentation" | SAM 3 的核心任务 —— 给定一个名词短语或图像示例，分割所有匹配的实例 |
| 概念提示 | "文本输入" | 简短的名词短语或图像示例；不是完整句子 |
| 存在性头 | "它在这里吗？" | SAM 3 模块，在定位前判断概念是否存在于图像中 |
| SA-CO | "SAM 3 基准" | 27 万概念的开放词汇分割基准；比之前的开放词汇基准大 50 倍 |
| Object Multiplex | "SAM 3.1 更新" | 共享记忆多目标跟踪；快速联合跟踪多个实例 |
| Grounded SAM 2 | "模块化流水线" | 检测器 + SAM 2 级联；当需要换检测器时仍然相关 |
| SAM-MI | "高效 SAM 变体" | Mask Injection 比 Grounded-SAM 快 1.6 倍 |

## 延伸阅读

- [SAM 3: Segment Anything with Concepts（arXiv 2511.16719）](https://arxiv.org/abs/2511.16719)
- [SAM 3.1 Object Multiplex（Meta AI，2026 年 3 月）](https://ai.meta.com/blog/segment-anything-model-3/)
- [SAM 3 model page on Hugging Face](https://huggingface.co/facebook/sam3)
- [Grounded SAM 2 tutorial（PyImageSearch）](https://pyimagesearch.com/2026/01/19/grounded-sam-2-from-open-set-detection-to-segmentation-and-tracking/)
- [Ultralytics SAM 3 docs](https://docs.ultralytics.com/models/sam-3/)
- [SAM3-I: Instruction-aware SAM（arXiv 2512.04585）](https://arxiv.org/abs/2512.04585)