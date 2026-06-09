# Git 与协作

> 版本控制不是可选项。每一次实验、每一个模型、每一节课都会被追踪。

**类型：** 学习
**前置要求：** 无
**预计时间：** 约 30 分钟

## 学习目标

- 配置 git 用户信息和 SSH 密钥
- 掌握日常使用的 git 工作流
- 用分支管理实验
- 学会与本课程仓库协作

## 为什么版本控制不可或缺

没有版本控制的 AI 工程，就像没有保存功能的文档编辑。你改了模型超参数发现效果更差了，却不记得改了什么。Git 让一切可追溯、可撤销、可协作。

## 核心工作流

```bash
# 克隆课程仓库
git clone https://github.com/rohitg00/ai-engineering-from-scratch.git
cd ai-engineering-from-scratch

# 每天开始时：拉取最新
git pull

# 创建实验分支
git checkout -b experiment/lr-sweep

# 做修改
# ... 运行实验 ...

# 提交
git add -A
git commit -m "feat: 尝试 0.001 的学习率"

# 推送
git push -u origin experiment/lr-sweep
```

## 分支策略

- `main`：稳定版，始终可运行
- `feat/xxx`：新功能开发
- `experiment/xxx`：探索性实验
- `fix/xxx`：问题修复

## 与本课程仓库协作

1. Fork 仓库到你自己的 GitHub 账户
2. 在你的 fork 中做练习
3. 如有勘误，在原仓库提交 Issue

## 练习

1. 配置你的 git 用户信息（`git config --global user.name` 和 `user.email`）
2. 生成 SSH 密钥并添加到 GitHub
3. Fork 本课程仓库，创建分支，在上面做修改并提交
