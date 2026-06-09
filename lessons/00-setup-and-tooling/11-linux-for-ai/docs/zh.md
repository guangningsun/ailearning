# AI 必备 Linux

> 大多数 AI 运行在 Linux 上。你需要足够了解它才不会卡住。

**类型：** 学习
**前置要求：** 了解终端基础
**预计时间：** 约 40 分钟

## 学习目标

- 在 Linux 上进行日常文件操作
- 搜索文件和内容
- 理解 Linux 文件权限

## 移动和导航

```bash
cd /home/user/projects    # 进入目录
ls -la                    # 列出所有文件（含隐藏文件）
pwd                       # 显示当前路径
realpath file.py          # 显示绝对路径
```

## 文件操作

```bash
# 创建
touch README.md
mkdir -p phases/01/lessons

# 复制和移动
cp -r src/ backup/
mv old.py new.py

# 删除（谨慎！）
rm -rf temp/     # -r 递归 -f 不询问
```

## 搜索

```bash
# 搜索文件
find . -name "*.py"
find /home -size +100M  # 找大于 100MB 的文件

# 搜索内容
grep -rn "TODO" .
rg "TODO" .              # ripgrep，更快
```

## 权限

```bash
ls -l file.py
# -rw-r--r--  user  group  4096  Jun 8  10:00  file.py
#  ^^^^^  ^^^^
#   所有者   组

# 修改权限
chmod +x script.sh    # 添加执行权限
chmod 644 file.py     # rw-r--r--
chown user:group file  # 改变所有者
```

## 磁盘和内存

```bash
df -h          # 磁盘使用
du -sh *       # 各目录大小
free -h        # 内存使用
```

## 练习

1. 用 `find` 找到课程中所有包含 `torch` 的 Python 文件
2. 检查你的 home 目录，找出最大的 5 个文件
3. 用 `chmod` 给一个脚本添加执行权限并运行
