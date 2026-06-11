# 数值稳定性

> 浮点数是一个会漏水的抽象。它在训练中咬你一口的时候，你连它什么时候来都看不到。

**类型：** 构建型
**语言：** Python
**前置条件：** 阶段 1，第 01-04 课
**时间：** 约 120 分钟

## 学习目标

- 用最大值减法技巧实现数值稳定的 softmax 和 log-sum-exp
- 识别浮点运算中的上溢、下溢和灾难性抵消
- 用中心有限差分法将解析梯度与数值梯度进行对比验证
- 解释为什么训练优先用 bfloat16 而非 float16，以及损失缩放如何防止梯度下溢

## 问题

你的模型训练了三个小时，然后损失变成了 NaN。你加了一行 print。步骤 9,000 时 logits 还是正常的。步骤 9,001 就变成了 `inf`。到了步骤 9,002，每个梯度都是 `nan`，训练已经死了。

或者：模型训练完成了，但准确率比论文报告的低了 2%。你检查了所有东西。架构一致。超参数一致。数据一致。问题在于，论文用的是 float32，而你用了 float16 却没有做正确的缩放。32 位的累积舍入误差悄悄吃掉了你的精度。

或者：你从零实现了交叉熵损失。在小 logits 上工作正常。当 logits 超过 100，它返回了 `inf`。softmax 上溢了，因为 `exp(100)` 超出了 float32 的表示范围。每个 ML 框架都用两行代码的技巧处理这个问题。但你不知道这个技巧存在。

数值稳定性不是理论问题。它决定一次训练是成功还是悄然失败。你将来要排查的每一个严重 ML bug，最终都会落到浮点数头上。

## 概念

### IEEE 754：计算机如何存储实数

计算机按照 IEEE 754 标准把实数存为浮点值。一个浮点数有三个部分：符号位、指数位和尾数位（有效数字）。

```
Float32 布局（共 32 位）：
[1 符号位] [8 指数位] [23 尾数位]

值 = (-1)^符号位 * 2^(指数位 - 127) * 1.尾数位
```

尾数决定精度（有多少有效数字）。指数决定范围（一个数能有多大或多小）。

```
格式       位数   指数位  尾数位   十进制有效位数   范围（约）
float64    64     11      52       ~15-16           ± 1.8e308
float32    32     8       23       ~7-8             ± 3.4e38
float16    16     5       10       ~3-4             ± 65,504
bfloat16   16     8       7        ~2-3             ± 3.4e38
```

float32 大约有 7 个十进制有效数字。这意味着它能区分 1.0000001 和 1.0000002，但分不清 1.00000001 和 1.00000002。7 位之后，全是舍入噪声。

float16 大约有 3 个有效数字。它能表示的最大数是 65,504。这对 ML 来说小得令人不安——logits、梯度和激活值经常会超过这个数。

bfloat16 是 Google 为解决 float16 范围问题给出的答案。它和 float32 一样有 8 位指数（相同范围，最大到 3.4e38），但只有 7 位尾数（精度比 float16 还低）。对神经网络训练来说，范围比精度更重要，所以 bfloat16 通常是更好的选择。

### 为什么 0.1 + 0.2 != 0.3

数字 0.1 在二进制浮点数中无法精确表示。在基数为 2 的表示中，它是一个循环小数：

```
0.1 的二进制表示 = 0.0001100110011001100110011...（无限循环）
```

float32 把它截断为 23 位尾数。存储的值大约是 0.100000001490116。同理，0.2 被存为约 0.200000002980232。它们的和是 0.300000004470348，而不是 0.3。

```
在 Python 中：
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这对 ML 很重要，因为：

1. 损失比较（如 `if loss < threshold`）可能给出错误答案
2. 累加很多小值（几千步的梯度更新）会偏离真实总和
3. 如果你用 `==` 比较浮点数，校验和与可复现性测试就会失败

修复方法：永远不要用 `==` 比较浮点数。用 `abs(a - b) < epsilon` 或 `math.isclose()`。

### 灾难性抵消

当你把两个非常接近的浮点数相减，有效数字互相抵消，留下的结果被舍入噪声所主导。

```
a = 1.0000001    （float32 中存为 1.00000011920929）
b = 1.0000000    （float32 中存为 1.00000000000000）

真实差：        0.0000001
计算值：        0.00000011920929

相对误差： 19.2%
```

一次减法就产生了 19% 的相对误差。在 ML 中，这发生在以下场景：

- 计算均值很大时数据的方差：`E[x^2] - E[x]^2`，当 E[x] 很大时
- 减去两个几乎相等的对数概率
- 用太小的 epsilon 计算有限差分梯度

修复方法：重新排列公式，避免减去两个大而接近的数。对方差，用 Welford 算法或先把数据中心化。对对数概率，全程在对数空间中工作。

### 上溢和下溢

上溢发生在结果太大无法表示时。下溢发生在结果太小（比最小的可表示正数更接近零）时。

```
float32 的边界：
  最大值：                 3.4028235e+38
  最小正数（规格化）：      1.175e-38
  最小正数（非规格化）：    1.401e-45
  上溢：  任何 > 3.4e38 变成 inf
  下溢：  任何 < 1.4e-45 变成 0.0
```

`exp()` 函数是 ML 中上溢的主要来源：

```
exp(88.7)  = 3.40e+38   （勉强不溢出 float32）
exp(89.0)  = inf        （上溢）
exp(-87.3) = 1.18e-38   （勉强高于下溢边界）
exp(-104)  = 0.0        （下溢到零）
```

`log()` 函数在另一头出问题：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      （没问题）
log(1e-46) = -inf        （输入已经下溢为 0，然后 log(0) = -inf）
```

在 ML 中，`exp()` 出现在 softmax、sigmoid 和概率计算中。`log()` 出现在交叉熵、对数似然和 KL 散度中。`log(exp(x))` 这种组合，没有正确的技巧就是一个雷区。

### Log-Sum-Exp 技巧

直接计算 `log(sum(exp(x_i)))` 在数值上是危险的。如果任何一个 `x_i` 很大，`exp(x_i)` 就上溢。如果所有 `x_i` 都很负，每个 `exp(x_i)` 都下溢到零，然后 `log(0)` 就是 `-inf`。

技巧：在指数化之前减去最大值。

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

为什么有效：减去 `max(x)` 后，最大的指数是 `exp(0) = 1`。不会上溢。求和里至少有一项是 1，所以和至少是 1，`log(1) = 0`。不会下溢到 `-inf`。

证明：

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                  （加减 c）
= log(sum(exp(x_i - c) * exp(c)))              （exp(a+b) = exp(a)*exp(b)）
= log(exp(c) * sum(exp(x_i - c)))              （提出 exp(c)）
= c + log(sum(exp(x_i - c)))                   （log(a*b) = log(a) + log(b)）
```

设 `c = max(x)`，上溢就被消除了。

这个技巧在 ML 中无处不在：
- Softmax 归一化
- 交叉熵损失计算
- 序列模型中的对数概率求和
- 高斯混合模型
- 变分推断

### 为什么 Softmax 需要最大值减法技巧

Softmax 把 logits 转换为概率：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

没有技巧的话，logits [100, 101, 102] 会导致上溢：

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

这些在 float32 中会上溢（最大值约 3.4e38）吗？不会，2.69e43 < 3.4e38？实际上：
exp(88.7) 就已经到了 float32 的极限。
exp(100) 在 float32 中 = inf。
```

有了技巧，减去 max(x) = 102：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

概率结果完全相同。计算是安全的。这不是优化，而是正确性的必要条件。

### NaN 和 Inf：检测与预防

`nan`（Not a Number）和 `inf`（无穷大）会在计算中像病毒一样传播。梯度更新中的一个 `nan` 会让权重变成 `nan`，进而让后续所有输出都变成 `nan`。训练在一两步之内就彻底死亡。

`inf` 如何产生：
- 对很大的正数取 `exp()`
- 除以零：`1.0 / 0.0`
- `float32` 累加中上溢

`nan` 如何产生：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 对负数取 `sqrt()`
- 对负数取 `log()`
- 任何包含已有 `nan` 的运算

检测：

```python
import math

math.isnan(x)       # 如果 x 是 nan 返回 True
math.isinf(x)       # 如果 x 是 +inf 或 -inf 返回 True
math.isfinite(x)    # 如果 x 既不是 nan 也不是 inf 返回 True
```

预防策略：

1. 钳制 `exp()` 的输入：`exp(clamp(x, -80, 80))`
2. 在分母加 epsilon：`x / (y + 1e-8)`
3. 在 `log()` 里面加 epsilon：`log(x + 1e-8)`
4. 使用稳定的实现（log-sum-exp、稳定 softmax）
5. 梯度裁剪防止权重爆炸
6. 调试期间在每次前向传播后检查 `nan`/`inf`

### 数值梯度检查

解析梯度（来自反向传播）可能存在 bug。数值梯度检查用有限差分计算梯度来验证它们。

中心差分公式：

```
df/dx ≈ (f(x + h) - f(x - h)) / (2h)
```

这是 O(h^2) 精度，远比前向差分 `(f(x+h) - f(x)) / h`（仅 O(h)）要好。

h 的选择：太大则近似不准确。太小则灾难性抵消会毁掉结果。通常取 `h = 1e-5` 到 `1e-7`。

验证方法：计算解析梯度和数值梯度之间的相对差异。

```
相对误差 = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验规则：
- 相对误差 < 1e-7：完美，梯度正确
- 相对误差 < 1e-5：可接受，大概率正确
- 相对误差 > 1e-3：有地方不对
- 相对误差 > 1：梯度完全错误

实现新层或新损失函数时，务必检查梯度。PyTorch 提供了 `torch.autograd.gradcheck()` 来做这件事。

### 混合精度训练

现代 GPU 有专门的硬件（Tensor Core），能以比 float32 快 2-8 倍的速度计算 float16 矩阵乘法。混合精度训练利用这一点：

```
1. 维护 float32 的主权重副本
2. 前向传播用 float16（快）
3. 损失用 float32 计算（防止上溢）
4. 反向传播用 float16（快）
5. 梯度放大到 float32
6. 用 float32 更新主权重副本
```

纯 float16 训练的问题：梯度通常非常小（1e-8 或更小）。Float16 约 6e-8 以下的所有东西都下溢为零。你的模型停止学习，因为所有梯度更新都是零。

修复方法是损失缩放：

```
1. 损失乘以一个大缩放因子（如 1024）
2. 反向传播计算 (loss * 1024) 的梯度
3. 所有梯度都放大了 1024 倍（推到了 float16 下溢边界以上）
4. 更新权重前先把梯度除以 1024
5. 最终效果：相同的更新，但没有下溢
```

动态损失缩放自动调整缩放因子。从一个大的值（65536）开始。如果梯度上溢到 `inf`，减半。如果连续 N 步没有上溢，翻倍。

### bfloat16 vs float16：为什么 bfloat16 更适合训练

```
float16:   [1 符号位] [5 指数位]  [10 尾数位]
bfloat16:  [1 符号位] [8 指数位]  [7 尾数位]
```

float16 精度更高（10 尾数位 vs 7 位），但范围受限（最大值约 65,504）。bfloat16 精度较低，但范围与 float32 相同（最大值约 3.4e38）。

对神经网络训练来说：

- 激活值和 logits 在训练波动时经常超过 65,504。float16 会上溢，bfloat16 能处理。
- float16 必须用损失缩放，bfloat16 通常不需要，因为它的范围覆盖了整个梯度量级。
- bfloat16 就是 float32 的简单截断：丢掉尾数的最低 16 位。转换简单，指数部分无损。

float16 更适合推理，因为推理中值是有界的、精度更重要。bfloat16 更适合训练，因为训练中范围更重要。这就是为什么 TPU 和现代 NVIDIA GPU（A100、H100）都原生支持 bfloat16。

### 梯度裁剪

梯度爆炸发生在梯度沿深层网络指数增长时（常见于 RNN、深层网络和 Transformer）。单次大梯度就能一步毁掉所有权重。

两种裁剪方式：

**按值裁剪：** 独立钳制每个梯度元素。

```
grad = clamp(grad, -max_val, max_val)
```

简单，但可能改变梯度向量的方向。

**按范数裁剪：** 缩放整个梯度向量，使它的范数不超过阈值。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

保持梯度方向不变。这就是 `torch.nn.utils.clip_grad_norm_()` 的做法。是标准选择。

典型值：Transformer 用 `max_norm=1.0`，强化学习用 `max_norm=0.5`，简单网络用 `max_norm=5.0`。

梯度裁剪不是奇技淫巧，而是一种安全机制。没有它，一个异常批次产生的梯度就足以毁掉数周的训练。

### 归一化层作为数值稳定器

批归一化、层归一化和 RMS 归一化通常被当作帮助训练收敛的正则化手段。它们同时也是数值稳定器。

没有归一化，激活值通过各层时会指数增长或缩小：

```
第 1 层：值在 [0, 1]
第 5 层：值在 [0, 100]
第 10 层：值在 [0, 10,000]
第 50 层：值在 [0, inf]
```

归一化在每一层重新对中并重新缩放激活值：

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`（通常 1e-5）在所有激活值都相同时防止除以零。可学习参数 `gamma` 和 `beta` 让网络可以恢复它需要的任意尺度。

这使得整个网络的值都处于数值安全范围内，同时防止前向传播的上溢和反向传播的梯度爆炸。

### 常见 ML 数值 Bug

**Bug：损失在几个 epoch 后变成 NaN。**
原因：logits 变得太大，softmax 上溢了。或者学习率太高，权重发散。
修复：使用稳定 softmax（最大值减法）、降低学习率、加梯度裁剪。

**Bug：损失卡在 log(num_classes)。**
原因：模型输出接近均匀概率。通常意味着梯度消失或模型根本没在学习。
修复：检查数据标签是否正确、验证损失函数、检查 ReLU 是否坏死。

**Bug：验证准确率比预期低 1-3%。**
原因：混合精度没有用正确的损失缩放。梯度下溢悄悄把小更新清零了。
修复：启用动态损失缩放，或切换到 bfloat16。

**Bug：某些层的梯度范数为 0.0。**
原因：ReLU 神经元坏死了（所有输入都为负），或 float16 下溢。
修复：改用 LeakyReLU 或 GELU、使用梯度缩放、检查权重初始化。

**Bug：模型在一块 GPU 上正常，换另一块结果不同。**
原因：浮点累加次序不确定。GPU 并行规约在不同硬件上以不同顺序求和，而浮点加法不满足结合律。
修复：接受微小差异（1e-6），或设置 `torch.use_deterministic_algorithms(True)` 并接受速度损失。

**Bug：损失计算中 `exp()` 返回 `inf`。**
原因：未使用最大值减法技巧，原始 logits 直接传入 `exp()`。
修复：使用 `torch.nn.functional.log_softmax()`，它内部实现了 log-sum-exp。

**Bug：从 float32 切到 float16 后训练发散。**
原因：float16 无法表示低于 6e-8 的梯度量级或高于 65,504 的激活值。
修复：使用带损失缩放的混合精度（AMP），或改用 bfloat16。

## 动手实现

### 第 1 步：演示浮点精度限制

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### 第 2 步：实现朴素 vs 稳定的 softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) 会返回 [nan, nan, nan]
```

### 第 3 步：实现稳定的 log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) 返回 inf
```

### 第 4 步：实现稳定的交叉熵

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### 第 5 步：梯度检查

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 实际使用

### 混合精度模拟

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 梯度裁剪

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf 检测

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

完整实现见 `code/numerical.py`，包含所有边界情况的演示。

## 交付物

本课产出：
- `code/numerical.py`，包含稳定 softmax、log-sum-exp、交叉熵、梯度检查和混合精度模拟
- `outputs/prompt-numerical-debugger.md`，用于诊断训练中的 NaN/Inf 和数值问题

这些稳定实现会在阶段 3 构建训练循环和阶段 4 实现注意力机制时再次出现。

## 联系

本课的所有概念都与现代 AI 的具体部件相连接：

| 概念 | 出现在哪里 |
|---------|------------------|
| Log-sum-exp 技巧 | 每个 ML 框架的 softmax / 交叉熵实现（PyTorch 的 `log_softmax`、`cross_entropy`）|
| 灾难性抵消 | 方差计算、有限差分梯度验证、对数概率相减 |
| float16 / bfloat16 | 现代 GPU 训练中决定精度的格式选择，直接影响速度与模型质量 |
| 损失缩放 | PyTorch AMP (`torch.cuda.amp`)、NVIDIA APEX —— 用 float16 训练时防止梯度消失 |
| 梯度裁剪 | Transformer 训练标配（`clip_grad_norm_`），防止单批次异常毁掉权重 |
| 数值梯度检查 | 自定义层和损失函数的测试环节，PyTorch 的 `torch.autograd.gradcheck()` |
| 归一化层 | BatchNorm/LayerNorm/RMSNorm 同时兼做数值稳定器，防止前向和反向传播中的值爆炸 |
| NaN/Inf 检测 | 训练循环调试必备——前向传播后检查可以第一时间发现梯度爆炸或数值错误 |

最值得记住的一条联系：PyTorch 的 `F.cross_entropy` 内部没有先算 softmax 再取 log。它直接用 log-sum-exp 一步到位。如果你手动拼接 softmax 和 nll_loss 两个步骤，你就丢掉了数值稳定性。框架把这些技巧藏在 API 后面是有原因的。

## 练习

1. **灾难性抵消。** 在 float32 中用朴素公式 `E[x^2] - E[x]^2` 计算 [1000000.0, 1000001.0, 1000002.0] 的方差。然后用 Welford 在线算法计算。将两种结果与真实方差（0.6667）对比误差。

2. **精度猎人。** 在 Python 中找到满足 `1.0 + x == 1.0` 的最小正 float32 值 `x`。这就是机器 epsilon。验证它与 `numpy.finfo(numpy.float32).eps` 是否一致。

3. **Log-sum-exp 边界情况。** 用以下输入测试你的 `logsumexp_stable` 函数：(a) 所有值相等，(b) 一个值远大于其余，(c) 所有值都非常负（-1000）。验证它在朴素版本失败的场景下给出正确结果。

4. **对神经网络层做梯度检查。** 实现一个单层线性层 `y = Wx + b` 及其解析反向传播。用 `numerical_gradient` 在一个 3x2 权重矩阵上验证正确性。

5. **损失缩放实验。** 模拟 float16 训练：生成 [1e-9, 1e-3] 范围内的随机梯度，转换为 float16，测量有多少比例变成了零。然后应用损失缩放（乘以 1024），转为 float16，再缩放回来，重新测量零比例。

## 关键术语

| 术语 | 大家怎么说的 | 实际含义 |
|------|----------------|----------------------|
| IEEE 754 | "浮点数标准" | 定义二进制浮点格式、舍入规则和特殊值（inf、nan）的国际标准。所有现代 CPU 和 GPU 都遵循它。 |
| 机器 epsilon | "精度极限" | 在给定浮点格式中满足 1.0 + e != 1.0 的最小值 e。float32 约 1.19e-7。 |
| 灾难性抵消 (Catastrophic cancellation) | "减法丢精度" | 两个相近浮点数相减时有效数字抵消，舍入噪声主导结果。 |
| 上溢 (Overflow) | "数字太大了" | 结果超出最大可表示值变为 inf。exp(89) 在 float32 中上溢。 |
| 下溢 (Underflow) | "数字太小了" | 结果比最小可表示正数更接近零，变为 0.0。exp(-104) 在 float32 中下溢。 |
| Log-sum-exp 技巧 | "先减最大值" | 通过提取 exp(max(x)) 计算 log(sum(exp(x)))，防止上溢和下溢。用于 softmax、交叉熵和对数概率计算。 |
| 稳定 softmax (Stable softmax) | "不会爆炸的 softmax" | 指数化前先减去 max(logits)。数值结果相同，但没有上溢可能。 |
| 梯度检查 (Gradient checking) | "验证你的反向传播" | 将反向传播的解析梯度与有限差分数值梯度进行对比，发现实现 bug。 |
| 混合精度 (Mixed precision) | "float16 前向，float32 反向" | 对速度关键的操作使用低精度浮点，对数值敏感的操作使用高精度。典型加速 2-3 倍。 |
| 损失缩放 (Loss scaling) | "防止梯度下溢" | 反向传播前将损失乘以一个大常数，使梯度留在 float16 的可表示范围内，更新权重前再除以该常数。 |
| bfloat16 | "Brain 浮点数" | Google 的 16 位格式，8 位指数（与 float32 相同范围），7 位尾数（低于 float16 精度）。更适用训练。 |
| 梯度裁剪 (Gradient clipping) | "给梯度范数设上限" | 缩放梯度向量使其范数不超过阈值。防止梯度爆炸毁掉权重。 |
| NaN | "非数字" | 由未定义操作（0/0、inf-inf、sqrt(-1)）产生的特殊浮点值。会在后续所有运算中传播。 |
| Inf | "无穷大" | 由上溢或除以零产生的特殊浮点值。可以组合产生 NaN（inf - inf、inf * 0）。 |
| 数值梯度 (Numerical gradient) | "暴力求导" | 通过计算 f(x+h) 和 f(x-h) 并除以 2h 来近似导数。慢但验证可靠。 |

## 进一步阅读

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) —— 权威参考，厚重但完整
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) —— NVIDIA 提出 float16 训练中损失缩放的论文
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) —— PyTorch 中混合精度的实操指南
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) —— Google 为何为 TPU 选择这一格式
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) —— 减少浮点求和舍入误差的算法
