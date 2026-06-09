# Python 环境

> 依赖地狱是真实存在的。虚拟环境是解药。

**类型：** 构建
**语言：** Shell
**前置要求：** 无
**预计时间：** 约 40 分钟

## 学习目标

- 用 uv 创建和管理虚拟环境
- 理解何时用 conda
- 掌握避免常见错误的策略

## 方案 1：uv（推荐）

`uv` 比 pip 快 10-100 倍，自动处理环境：

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境
uv venv

# 激活
source .venv/bin/activate

# 安装包
uv pip install numpy pandas torch

# 导出依赖
uv pip freeze > requirements.txt

# 从依赖文件安装
uv pip install -r requirements.txt
```

## 方案 2：venv（内置）

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas
```

## 方案 3：conda

当需要与系统级库交互或需要 Python 2 时使用：

```bash
conda create -n ai python=3.12
conda activate ai
conda install numpy pytorch -c pytorch
```

## 本课程的环境策略

每个阶段用独立环境：

```bash
# Phase 1（数学基础）
uv venv .venv-phase1
source .venv-phase1/bin/activate
uv pip install numpy scipy matplotlib jupyter

# Phase 8（生成式 AI）
uv venv .venv-phase8
source .venv-phase8/bin/activate
uv pip install diffusers transformers accelerate
```

## 常见错误

1. **忘记激活环境**：装完包发现 import 失败——先 `source .venv/bin/activate`
2. **pip 和 conda 混用**：导致依赖冲突，坚定选其一
3. **.venv 提交到 git**：一定要在 `.gitignore` 里加 `.venv/`
4. **CUDA 版本不匹配**：安装 PyTorch 前先确认 CUDA 版本：`nvidia-smi`

## 练习

1. 用 uv 创建两个独立环境，各安装不同的包
2. 验证激活/退出环境的行为
3. 将依赖导出到 requirements.txt 并在新环境中安装
