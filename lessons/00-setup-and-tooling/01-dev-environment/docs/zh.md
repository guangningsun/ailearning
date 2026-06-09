# 开发环境

> 你的工具塑造你的思维。一次性配置好，配置正确。

**类型：** 构建
**语言：** Python、Node.js、Rust
**前置要求：** 无
**预计时间：** 约 45 分钟

## 学习目标

- 从零配置 Python 3.11+、Node.js 20+ 和 Rust 工具链
- 配置虚拟环境和包管理器，确保构建可复现
- 验证 GPU 访问（CUDA/MPS）并运行测试张量运算
- 理解四层技术栈：系统层、包管理器层、运行时层、AI 库层

## 问题所在

你将用 Python、TypeScript、Rust 和 Julia 学习 200+ 节课的 AI 工程。如果你的环境有问题，每一节课都会变成和工具搏斗，而不是学习。

大多数人会跳过环境配置。然后花大量时间调试 import 错误、版本冲突和缺失的 CUDA 驱动。我们要把这件事一次性做好。

## 核心概念

AI 工程环境有四层：

<div class="layer-stack">
  <div class="layer-card layer-4">
    <span class="layer-num">4</span>
    <div class="layer-body">
      <div class="layer-title">AI/ML 库</div>
      <div class="layer-tools">PyTorch · JAX · transformers · vLLM 等</div>
    </div>
  </div>
  <div class="layer-link"><span>依赖</span></div>
  <div class="layer-card layer-3">
    <span class="layer-num">3</span>
    <div class="layer-body">
      <div class="layer-title">语言运行时</div>
      <div class="layer-tools">Python 3.11+ · Node 20+ · Rust · Julia</div>
    </div>
  </div>
  <div class="layer-link"><span>依赖</span></div>
  <div class="layer-card layer-2">
    <span class="layer-num">2</span>
    <div class="layer-body">
      <div class="layer-title">包管理器</div>
      <div class="layer-tools">uv · pnpm · cargo · juliaup</div>
    </div>
  </div>
  <div class="layer-link"><span>依赖</span></div>
  <div class="layer-card layer-1">
    <span class="layer-num">1</span>
    <div class="layer-body">
      <div class="layer-title">系统基础</div>
      <div class="layer-tools">操作系统 · shell · git · 编辑器 · GPU 驱动</div>
    </div>
  </div>
</div>

我们从下往上安装。每一层依赖于它下面的一层。

## 先手写

### 第 1 步：系统基础

检查你的系统并安装基础工具。

```bash
# macOS
xcode-select --install
brew install git curl wget

# Ubuntu/Debian
sudo apt update && sudo apt install -y build-essential git curl wget

# Windows（使用 WSL2）
wsl --install -d Ubuntu-24.04
```

### 第 2 步：Python（用 uv）

我们用 `uv`——它比 pip 快 10-100 倍，且自动处理虚拟环境。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.12

uv venv
source .venv/bin/activate  # Windows 上用 .venv\Scripts\activate

uv pip install numpy matplotlib jupyter
```

验证：

```python
import sys
print(f"Python {sys.version}")

import numpy as np
print(f"NumPy {np.__version__}")
a = np.array([1, 2, 3])
print(f"向量: {a}, 与自身的点积: {np.dot(a, a)}")
```

### 第 3 步：Node.js（用 pnpm）

用于 TypeScript 课程（agent、MCP 服务器、Web 应用）。

```bash
curl -fsSL https://fnm.vercel.app/install | bash
fnm install 22
fnm use 22

npm install -g pnpm

node -e "console.log('Node', process.version)"
```

### 第 4 步：Rust

用于性能关键的课程（推理、系统）。

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

rustc --version
cargo --version
```

### 第 5 步：Julia（可选）

用于数学密集的课程，Julia 在这些场景下表现出色。

```bash
curl -fsSL https://install.julialang.org | sh

julia -e 'println("Julia ", VERSION)'
```

### 第 6 步：GPU 配置（如有）

```bash
# NVIDIA
nvidia-smi

# 安装 PyTorch（CUDA 版）
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

```python
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

没有 GPU？没问题。大多数课程在 CPU 上都能跑。对于训练密集的课程，可以用 Google Colab 或云端 GPU。

### 第 7 步：全面验证

运行验证脚本：

```bash
python phases/00-setup-and-tooling/01-dev-environment/code/verify.py
```

## 再用框架

你的环境现在已为课程中每一节课准备好了。语言使用场景如下：

| 语言 | 应用场景 | 包管理器 |
|------|---------|---------|
| Python | 第 1-12 阶段（机器学习、深度学习、NLP、视觉、音频、LLM） | uv |
| TypeScript | 第 13-17 阶段（工具、Agent、蜂群、基础设施） | pnpm |
| Rust | 第 12、15-17 阶段（性能关键系统） | cargo |
| Julia | 第 1 阶段（数学基础） | Pkg |

## 产出

本课产出一个验证脚本，任何人都可以运行来检查自己的配置。

参见 `outputs/prompt-env-check.md`，这是一个帮助 AI 助手诊断环境问题的提示词模板。

## 练习

1. 运行验证脚本，修复任何失败项
2. 为本课程创建一个 Python 虚拟环境并安装 PyTorch
3. 用四种语言各写一个 "hello world" 并分别运行
