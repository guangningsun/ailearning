# GPU 配置与云端

> 用 CPU 学习完全没问题。要真正训练就需要 GPU 了。

**类型：** 构建
**语言：** Python
**前置要求：** 无
**预计时间：** 约 30 分钟

## 学习目标

- 检查本地 NVIDIA GPU 是否可用
- 在 Google Colab 上配置 GPU
- 选择并配置云端 GPU 服务

## GPU 选项

### 方案 1：本机 NVIDIA GPU

```bash
nvidia-smi
```

确认 CUDA 版本，然后安装对应版本的 PyTorch：

```python
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"GPU 数量: {torch.cuda.device_count()}")
print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
```

### 方案 2：Google Colab

免费且无需配置：
1. 打开 [colab.research.google.com](https://colab.research.google.com)
2. 新建笔记本 → 运行时 → 更改运行时类型 → T4 GPU
3. 验证：`!nvidia-smi`

### 方案 3：云端 GPU

| 服务 | GPU 选项 | 价格 |
|------|---------|------|
| Vast.ai | RTX 4090, A100 | 按小时计费 |
| Lambda Labs | A100, H100 | 按小时计费 |
| RunPod | 多卡选项 | 按秒计费 |
| AWS/EC2 | V100, A100 | 按需付费 |

## 验证 PyTorch CUDA

```python
import torch

# 基本检查
assert torch.cuda.is_available(), "CUDA 不可用"

# 张量在 GPU 上
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = x @ y  # 矩阵乘法在 GPU 上运行

# 计时对比
import time
device = 'cuda'

# GPU
t0 = time.time()
for _ in range(100):
    _ = torch.randn(1000, 1000, device=device) @ torch.randn(1000, 1000, device=device)
gpu_time = (time.time() - t0) / 100

# CPU
t0 = time.time()
for _ in range(100):
    _ = torch.randn(1000, 1000) @ torch.randn(1000, 1000)
cpu_time = (time.time() - t0) / 100

print(f"GPU: {gpu_time*1000:.2f}ms, CPU: {cpu_time*1000:.2f}ms")
print(f"GPU 加速比: {cpu_time/gpu_time:.1f}x")
```

## 练习

1. 运行 `nvidia-smi`，记录 GPU 型号和显存
2. 在 PyTorch 中验证 CUDA 可用性
3. 如果有 GPU，运行矩阵乘法计时对比
