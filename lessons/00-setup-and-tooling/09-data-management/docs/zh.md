# 数据管理

> 数据是燃料。管理方式决定了你的速度。

**类型：** 构建
**语言：** Python
**前置要求：** 了解 Python 基础
**预计时间：** 约 45 分钟

## 学习目标

- 使用 Hugging Face datasets 库加载数据
- 流式处理大型数据集（避免内存爆炸）
- 理解常用数据集格式
- 正确切分训练/验证/测试集

## 安装 datasets 库

```bash
uv pip install datasets
```

## 加载数据集

```python
from datasets import load_dataset

# 从 Hugging Face 加载
dataset = load_dataset("rotten_tomatoes")
print(dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['text', 'label'],
#         num_rows: 8530
#     })
#     validation: Dataset({...})
#     test: Dataset({...})
# })

# 访问数据
print(dataset["train"][0])
```

## 流式处理大数据集

```python
# 不下载全部数据，用流式模式
dataset = load_dataset(
    "allenai/c4",
    name="en",
    split="train",
    streaming=True
)

for i, example in enumerate(dataset):
    print(example["text"][:100])
    if i >= 9:  # 只处理前 10 条
        break
```

## 数据集格式

- **Arrow/Parquet**：Hugging Face 默认格式，支持懒加载
- **JSONL**：每行一个 JSON 对象，适合流式读取
- **CSV**：通用但不支持嵌套结构
- **HDF5**：科学计算常用

## 数据切分

```python
from sklearn.model_selection import train_test_split

train, val = train_test_split(dataset["train"], test_size=0.1)
```

## 练习

1. 从 Hugging Face 加载一个数据集
2. 用流式模式遍历数据（不下载全部）
3. 将数据导出为不同格式（JSONL、Parquet）
