# OCR 与文档理解

> OCR 是一个三阶段流水线 —— 检测文本框、识别字符、还原排版。现代 OCR 系统都在重新排序或合并这些阶段。

**类型：** 学习型 + 使用
**语言：** Python
**前置条件：** 阶段 4 第 06课（目标检测），阶段 7 第 02 课（自注意力）
**时间：** 约 45 分钟

## 学习目标

- 梳理经典 OCR 流水线（检测 -> 识别 -> 排版）以及现代端到端替代方案（Donut、Qwen-VL-OCR）
- 从零实现 CTC（连接时序分类）损失函数，用于序列到序列的 OCR 训练
- 使用 PaddleOCR 或 EasyOCR 实现生产级文档解析，无需额外训练
- 区分 OCR、布局解析与文档理解，并能为不同任务选择合适的工具

## 问题

充满文字的图像无处不在：收据、发票、身份证、扫描书籍、表单、白板、指示牌、截图。从这些图像中提取结构化数据 —— 不仅要字符，还要"这是总金额"—— 是最高价值的应用视觉问题之一。

这个领域分为三个技能层次：

1. **狭义 OCR**：将像素转换为文字。
2. **布局解析**：将 OCR 输出按区域分组（标题、正文、表格、页眉）。
3. **文档理解**：从布局中提取结构化字段（"invoice_total = $42.50"）。

每一层都有经典方法和现代方法，而从"我想从图像中获取文字"到"我需要从这张收据中提取总金额"之间的差距，比大多数团队意识到的要大得多。

## 概念

### 经典流水线

```mermaid
flowchart LR
    IMG["图像"] --> DET["文本检测<br/>(DB, EAST, CRAFT)"]
    DET --> BOX["词/行<br/>边界框"]
    BOX --> CROP["裁剪每个区域"]
    CROP --> REC["识别<br/>(CRNN + CTC)"]
    REC --> TXT["文本字符串"]
    TXT --> LAY["布局<br/>排序"]
    LAY --> OUT["阅读顺序文本"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

- **文本检测**产生每行或每个词的 quadrilaterals（四边形）。
- **识别** 将每个区域裁剪到固定高度，运行 CNN + BiLSTM + CTC 产生字符序列。
- **布局** 重建阅读顺序（拉丁文从左到右、从上到下；阿拉伯文、日文不同）。

### CTC 一段话解释

OCR 识别从固定长度的特征图产生可变长度的序列。CTC（Graves 等，2006）让你无需字符级对齐即可训练。模型在每个时间步输出（词表 + 空格）的分布；CTC 损失对所有合并重复字符并去除空格后能还原目标文本的对齐方式进行边缘化。

```
原始输出： "h h h _ _ e e l l _ l l o _ _"
合并重复并去除空格后： "hello"
```

这就是 CRNN 在 2015 年奏效的原因，2026 年大多数生产级 OCR 模型仍在用 CTC 训练。

### 现代端到端模型

- **Donut**（Kim 等，2022）—— ViT 编码器 + 文本解码器；读取图像直接输出 JSON。无需文本检测器，无需布局模块。
- **TrOCR** —— ViT + transformer 解码器，用于行级 OCR。
- **Qwen-VL-OCR / InternVL** —— 完整视觉-语言模型，针对 OCR 任务微调；2026 年复杂文档上精度最高。
- **PaddleOCR** —— 经典 DB + CRNN 流水线，成熟的生产级包；仍是开源主力。

端到端模型需要更多数据和算力，但避免了多阶段流水线的误差累积。

### 布局解析

对于结构化文档，运行一个布局检测器（LayoutLMv3、DocLayNet）来标注每个区域：标题、段落、图像、表格、脚注。阅读顺序由此变为"按布局顺序遍历各区域，然后拼接"。

对于表单，使用**键值提取**模型（Donut 用于视觉丰富的文档，LayoutLMv3 用于纯扫描件）。它们接收图像 + 检测到的文字 + 位置，并预测结构化键值对。

### 评估指标

- **字符错误率（CER）** —— Levenshtein 距离 / 参考文本长度。越低越好。生产目标：干净扫描件 < 2%。
- **词错误率（WER）** —— 在词级别的相同指标。
- **结构化字段 F1** —— 用于键值任务；衡量 `{invoice_total: 42.50}` 是否正确出现。
- **JSON 编辑距离** —— 用于端到端文档解析；Donut 论文引入了归一化树编辑距离。

## 动手实现

### 第 1 步：CTC 损失 + 贪心解码器

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C) 在词表（含空格，空格在索引0）上的 log-softmax
    targets:        (N, S) int 目标（无空格）
    input_lengths:  (N,) 每个样本使用的时间步数
    target_lengths: (N,) 每个样本的目标长度
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) log-softmax
    返回：索引序列列表（已去除空格，已合并重复）
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out
```

`F.ctc_loss` 在可用时使用高效的 CuDNN 实现。贪心解码器比束搜索简单，通常与之相差不到 1% CER。

### 第 2 步：微型 CRNN 识别器

用于行级 OCR 的最小 CNN + BiLSTM。

```python
class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)
```

固定高度输入（CNN 将高度最大池化到 1）。宽度是 CTC 的时间维度。

### 第 3 步：合成 OCR 数据

生成黑底白字的数字字符串，用于端到端冒烟测试。

```python
import numpy as np

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        shade = 0.0 if c.isalnum() else 0.5
        img[6:height - 6, x + 2:x + char_width - 2] = shade
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"images: {imgs.shape}   targets: {targets.shape}   lengths: {lengths.tolist()}")
```

真实 OCR 数据集需要添加字体、噪声、旋转、模糊和颜色。上述流水线完全相同。

### 第 4 步：训练草图

```python
model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()
```

在这个简单的合成数据上，损失应从约 3 降到约 0.2（200 步）。

## 使用它

三条生产路径：

- **PaddleOCR** —— 成熟、快速、多语言。一行用法：`paddleocr.PaddleOCR(lang="en").ocr(image_path)`。
- **EasyOCR** —— Python 原生、多语言、PyTorch 主干网络。
- **Tesseract** —— 经典方法；当模型难以处理时，对旧扫描文档仍然有用。

对于端到端文档解析，使用 Donut 或 VLM：

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

对于具有重复结构的收据、发票和表单，微调 Donut。对于任意文档或需要推理的 OCR，像 Qwen-VL-OCR 这样的 VLM 是2026 年的当前默认选择。

## 交付物

本课产出：

- `outputs/prompt-ocr-stack-picker.md` —— 一个提示词，根据文档类型、语言和结构选择 Tesseract / PaddleOCR / Donut / VLM-OCR。
- `outputs/skill-ctc-decoder.md` —— 一个技能，从零编写贪心和束搜索 CTC 解码器，包括长度归一化。

## 练习

1. **（简单）** 在 5 位随机数字串上训练 TinyCRNN 500 步。报告在留出集上的 CER。
2. **（中等）** 将贪心解码替换为束搜索（beam_width=5）。报告 CER 差值。束搜索在哪些输入上胜出？
3. **（困难）** 在 20 张收据上使用 PaddleOCR，提取行项目，并根据手标真值计算 {item_name, price} 对的 F1。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|----------------------|
| OCR | "从像素读出文字" | 将图像区域转换为字符序列 |
| CTC | "无对齐损失" | 无需每时间步标签即可训练序列模型的损失；边缘化所有对齐方式 |
| CRNN | "经典 OCR 模型" | 卷积特征提取器 + BiLSTM + CTC；2015 年的基线，至今仍在生产中使用 |
| Donut | "端到端 OCR" | ViT 编码器 + 文本解码器；直接从图像输出 JSON |
| 布局解析 | "找区域" | 在文档中检测并标注标题/表格/图像/段落区域 |
| 阅读顺序 | "文本序列" | 将识别出的区域排序成句子；对拉丁文简单，对混合布局不简单 |
| CER / WER | "错误率" | Levenshtein 距离 / 参考长度，字符或词级别 |
| VLM-OCR | "能阅读的 LLM" | 针对 OCR 任务训练或提示的视觉-语言模型；在复杂文档上当前最优 |

## 延伸阅读

- [CRNN（Shi 等，2015）](https://arxiv.org/abs/1507.05717) —— 原始 CNN+RNN+CTC 架构
- [CTC（Graves 等，2006）](https://www.cs.toronto.edu/~graves/icml_2006.pdf) —— 原始 CTC 论文；算法思想密度很高
- [Donut（Kim 等，2022）](https://arxiv.org/abs/2111.15664) —— 无 OCR 的文档理解 transformer
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) —— 开源生产级 OCR 技术栈