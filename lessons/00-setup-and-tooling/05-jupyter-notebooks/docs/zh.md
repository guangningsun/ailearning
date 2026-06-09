# Jupyter Notebook

> Notebook 是 AI 工程的实验台。先在这里原型验证，再将可行的部分迁移到生产环境。

**类型：** 构建
**语言：** Python
**前置要求：** 无
**预计时间：** 约 45 分钟

## 学习目标

- 选择并配置 Jupyter 环境
- 掌握关键快捷键
- 理解 Cell 类型和使用场景
- 使用 Magic 命令

## 选择界面

| 界面 | 优点 | 适合场景 |
|------|------|---------|
| Jupyter Notebook | 经典，浏览器内运行 | 简单探索 |
| JupyterLab | 现代化，多标签 | 复杂项目 |
| VS Code | 代码编辑 + Notebook | 统一工作流 |
| Google Colab | 免费 GPU，云端 | 快速实验 |

## 关键快捷键

| 快捷键 | 作用 |
|--------|------|
| `Shift + Enter` | 运行当前 Cell，跳到下一个 |
| `Ctrl + Enter` | 运行当前 Cell，保持位置 |
| `Esc` | 进入命令模式 |
| `A` / `B` | 在上方/下方插入 Cell |
| `DD` | 删除当前 Cell（按两次 D） |
| `M` | 将 Cell 转为 Markdown |
| `Y` | 将 Cell 转为代码 |

## Cell 类型

- **Code**：可执行代码，结果显示在下方
- **Markdown**：格式文本，用于文档和解释
- **Raw**：纯文本，不执行

## Magic 命令

```python
# 计时单次执行
%timeit sum(range(1000000))

# 计时多次执行
%%time
result = sum(range(1000000))

# 显示 Matplotlib 图形内联
%matplotlib inline

# 加载外部脚本
%load_ext my_script.py

# 显示所有魔法命令
%lsmagic
```

## 常见陷阱

1. **在旧数据上重新运行**：Cell 的执行顺序和视觉顺序可能不一致，用 `Kernel → Restart & Run All` 验证
2. **显存泄漏**：每次运行后重启内核释放 GPU 显存
3. **Notebook 版本混乱**：用 `!pip freeze > requirements.txt` 记录依赖

## 练习

1. 在本地启动 Jupyter，创建一个包含代码和 Markdown 的 Notebook
2. 用 `%%time` 测量一个循环的执行时间
3. 写一个 Markdown 单元格解释你的代码在做什么
