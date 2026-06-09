# 调试与性能分析

> 最糟糕的 AI bug 不会崩溃。它们悄悄在垃圾数据上训练，然后报告一条漂亮的损失曲线。

**类型：** 构建
**语言：** Python
**前置要求：** Python 基础
**预计时间：** 约 60 分钟

## 学习目标

- 用打印调试、pdb 和日志定位问题
- 用 cProfile 和 line_profiler 分析性能瓶颈
- 用 memory_profiler 查找显存泄漏
- 理解常见 AI bug 的特征

## 打印调试

```python
# 最简单但有效
print(f"Step {step}, loss: {loss.item():.4f}")

# 调试列表和形状
print(f"x.shape: {x.shape}, y.shape: {y.shape}")
assert x.shape[0] == y.shape[0], "批次大小不匹配"

# 检查梯度
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"WARNING: {name} 没有梯度")
```

## Python 调试器（pdb）

```python
import pdb
pdb.set_trace()  # 在可疑位置插入断点

# 或用 breakpoint()（Python 3.7+，等同于 pdb.set_trace()）
```

常用 pdb 命令：`n`（下一行）、`s`（进入函数）、`p 变量`（打印）、`c`（继续）、`l`（查看上下文）

## Python 日志

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info(f"开始训练，batch_size={batch_size}")
logger.warning("验证集为空，跳过验证")
logger.error(f"训练失败: {e}")
```

## 性能分析

```python
# 逐行分析
from line_profiler import LineProfiler

lp = LineProfiler()
lp.add_function(my_function)
lp_wrapper = lp(my_function)
# 运行...
lp.print_stats()

# 内存分析
from memory_profiler import profile

@profile
def train_step(batch):
    ...
```

## 常见 AI Bug

1. **梯度消失/爆炸**：打印每层梯度范数
2. **数据泄漏**：训练集和验证集有重叠
3. **形状不匹配**：打印 tensor.shape 逐层核对
4. **学习率过大**：损失变成 NaN
5. **batch size=1 导致 batch norm 失效**

## 练习

1. 在代码中用 `breakpoint()` 调试一个函数
2. 用 `%%timeit` 分析一个矩阵运算的速度
3. 添加日志到训练循环，观察训练过程
