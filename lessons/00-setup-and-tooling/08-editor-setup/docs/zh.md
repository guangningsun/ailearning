# 编辑器配置

> 你的编辑器是你的副驾驶。一次性配置好，让它开始发挥作用而不是碍事。

**类型：** 构建
**前置要求：** 无
**预计时间：** 约 30 分钟

## 学习目标

- 配置 VS Code 作为 AI 开发主力编辑器
- 安装关键扩展
- 设置远程开发（SSH 到 GPU 机器）

## VS Code 基础配置

### 必须安装的扩展

- **Python** — Python 语言支持、linting、格式化
- **Jupyter** — Notebook 支持
- **GitLens** — 增强 git 可视化
- **GitHub Copilot** / **Continue** — AI 代码补全
- **Remote - SSH** — 远程服务器开发
- **Docker** — 容器管理

### 设置.json

```json
{
  "editor.formatOnSave": true,
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting ruffEnabled": true,
  "jupyter.askForKernelRestart": false,
  "files.exclude": {
    "**/.git": true,
    "**/__pycache__": true,
    "**/.venv": true
  }
}
```

## 远程开发（SSH 到 GPU 机器）

1. 安装 Remote - SSH 扩展
2. `Cmd+Shift+P` → `Remote-SSH: Connect to Host`
3. 输入 `user@gpu-server.com`
4. 在远程机器上安装 Python 扩展
5. 打开文件夹，像本地一样工作

## 其他编辑器

- **Cursor**：基于 VS Code，内置 Copilot 深度集成
- **Windsurf**：AI 优先的编辑器
- **Vim/Neovim**：键盘流高效编辑器，适合远程工作

## 练习

1. 安装 VS Code 和 Python、Jupyter 扩展
2. 配置 formatOnSave
3. 如果有远程 GPU 服务器，配置 SSH 远程开发
