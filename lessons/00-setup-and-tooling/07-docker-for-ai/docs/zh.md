# Docker for AI

> 容器让「在我机器上能跑」成为历史。

**类型：** 构建
**语言：** Docker
**前置要求：** 了解终端基本操作
**预计时间：** 约 60 分钟

## 学习目标

- 安装 Docker 和 NVIDIA Container Toolkit
- 理解 AI 项目中常见的容器模式
- 为 AI 开发编写 Dockerfile
- 用 Docker Compose 管理多服务应用

## 为什么 AI 项目更需要 Docker

- **环境可复现**：训练环境和生产环境完全一致
- **GPU 隔离**：避免不同项目的 CUDA 版本冲突
- **团队协作**：新成员一行命令即可搭建完整环境

## 安装 Docker

```bash
# macOS/Windows：安装 Docker Desktop
# Linux：
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```

## NVIDIA Container Toolkit（Linux + NVIDIA GPU）

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## AI 开发 Dockerfile

```dockerfile
FROM nvidia/cuda:12.4.0-cudnn9-runtime-ubuntu22.04

WORKDIR /app

# 安装 Python 和基础工具
RUN apt-get update && apt-get install -y python3.12 python3-pip curl git && rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# 复制依赖文件
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# 复制代码
COPY . .

# 默认命令
CMD ["python3", "train.py"]
```

## Docker Compose（多服务）

```yaml
version: '3.8'
services:
  training:
    build: .
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./data:/data
      - ./models:/models

  tensorboard:
    image: tensorflow/tensorflow:latest
    ports:
      - "6006:6006"
    volumes:
      - ./logs:/logs
    command: tensorboard --logdir=/logs
```

## 练习

1. 安装 Docker Desktop（或 Linux 版 Docker）
2. 运行官方测试镜像验证安装：`docker run hello-world`
3. 写一个 Dockerfile 来运行 PyTorch 训练脚本
