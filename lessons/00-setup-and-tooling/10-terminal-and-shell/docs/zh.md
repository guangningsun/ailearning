# 终端与 Shell

> 终端是 AI 工程师的居所。在这里要游刃有余。

**类型：** 学习
**前置要求：** 无
**预计时间：** 约 45 分钟

## 学习目标

- 掌握 shell 基本操作
- 用管道和重定向连接命令
- 管理后台进程
- 使用 tmux 保持会话

## Shell 基础

```bash
# 导航
cd ~/projects/ai-course
ls -la
pwd

# 搜索
find . -name "*.py"
grep -r "def train" .
rg "def train" .   # 更快的替代（ripgrep）

# 文件操作
cp src/*.py backup/
mv old_name.py new_name.py
mkdir -p phases/01  # -p 自动创建父目录
```

## 管道和重定向

```bash
# 管道：将一个命令的输出传给另一个
ps aux | grep python        # 找到 Python 进程
df -h | grep sda            # 找到磁盘使用

# 重定向
python train.py > output.log 2>&1   # 输出和错误都写入文件
python train.py >> output.log      # 追加到文件末尾

# 管道组合
cat logs/*.log | grep ERROR | wc -l  # 统计 ERROR 数量
```

## 后台进程和 tmux

```bash
# 后台运行
python train.py &
jobs            # 查看后台任务
fg %1          # 把任务 1 拉回前台

# tmux（推荐）
tmux new -s training
# 运行训练...
# 按 Ctrl+B 然后 D（分离）
tmux attach -t training   # 重新连接
```

## 监控

```bash
# 资源监控
htop          # 交互式进程/内存监控
nvtop          # GPU 监控（需要 NVIDIA）
df -h          # 磁盘空间

# 查看 NVIDIA GPU
watch -n 1 nvidia-smi   # 每秒刷新
```

## SSH 到 GPU 机器

```bash
ssh -L 8888:localhost:8888 user@gpu-server.com
# -L 将远程 8888 端口映射到本地
```

## 练习

1. 用 `grep` 在课程代码中找到所有 `import torch` 的文件
2. 用 tmux 启动一个后台会话
3. 用 `nvidia-smi` 查看 GPU 使用情况
