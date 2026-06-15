# 从零学 AI 工程

> 一个免费开源的 AI 工程课程，从线性代数到自主智能体，每一步都先自己手写。503 节课、20 个阶段、四种语言：Python、TypeScript、Rust、Julia。

[![Site](https://img.shields.io/badge/网站-aiengineeringfromscratch.com-blue)](https://aiengineeringfromscratch.com)
[![Lessons](https://img.shields.io/badge/lessons-503-blue)](#课程结构)
[![Phases](https://img.shields.io/badge/phases-20-blue)](#课程结构)
[![License](https://img.shields.io/badge/license-ISC-green)](LICENSE)

<!-- STATS:START (generated from site/stats.json by build.js — do not edit by hand) -->
<p align="center"><sub><b>150,639</b> readers &nbsp;·&nbsp; <b>241,669</b> page views in the last 30 days &nbsp;·&nbsp; as of 2026-06-07</sub></p>
<!-- STATS:END -->

## 这是什么

「从零学 AI 工程」是一门动手写代码的 AI 课程。每一节课都遵循同一个原则：

1. **先手写**：用最朴素的 numpy/标准库把算法实现一遍，把数学跑通。
2. **再用框架**：切到 PyTorch / transformers / vLLM，看生产实现怎么写。

课程不要求你预先懂机器学习，但要求你能跑 Python、写 TypeScript、按 README 配环境。

## 课程结构

20 个阶段、503 节课、四种语言。中文版 100% 翻译完成。

| # | 阶段 | 代表课节 | 语言 |
|---|------|----------|------|
| 0  | 环境搭建              | 开发环境、Git、Linux、API 密钥 | Python · Node · Rust |
| 1  | 数学基础              | 线性代数、概率、信息论 | Python · Julia |
| 2  | 机器学习基础          | 回归、决策树、评估指标 | Python |
| 3  | 深度学习核心          | 张量、反向传播、CNN、训练循环 | Python |
| 4  | 计算机视觉            | 图像分类、目标检测、分割 | Python |
| 5  | NLP 基础到进阶        | Tokenization、词向量、RNN、注意力 | Python |
| 6  | 语音与音频            | 频谱、ASR、TTS | Python |
| 7  | Transformer 深入       | 多头注意力、位置编码、KV 缓存 | Python |
| 8  | 生成式 AI             | VAE、GAN、Diffusion | Python |
| 9  | 强化学习              | Q-learning、Policy Gradient | Python |
| 10 | 从零构建 LLM          | 预训练、SFT、RLHF | Python |
| 11 | LLM 工程              | 推理优化、量化、提示词工程 | Python |
| 12 | 多模态 AI             | CLIP、LLaVA、图像生成 | Python |
| 13 | 工具与协议            | MCP、Function Calling | Python · TypeScript |
| 14 | 智能体工程            | ReAct、Reflexion、Planning | Python · TypeScript |
| 15 | 自主系统              | Computer Use、Browser Agent | TypeScript · Python |
| 16 | 多智能体与群体        | 角色协作、群智涌现 | TypeScript · Python |
| 17 | 基础设施与生产        | vLLM、向量数据库、可观测性 | TypeScript · Rust |
| 18 | 伦理、安全与对齐      | Jailbreak、Red Team、Constitutional AI | Python |
| 19 | 毕业项目              | 端到端 LLM 应用、Agent 产品 | Python · TypeScript |

每节课的目录结构统一为：

```
lessons/<phase>/<lesson>/
├── code/          # 可运行的代码
├── docs/
│   └── zh.md      # 中文课节正文（唯一源文件）
└── outputs/       # 可复用的提示词 / skill / agent 模板
```

## 目录结构

```
.
├── README.md               # 你正在看
├── package.json            # 依赖：marked2
├── glossary/               # 术语表（terms.md / myths.md）
├── lessons/                # 课程内容（20 个阶段 × 503 节课）
└── site/                   # 静态站点源文件
    ├── index.html          # 首页
    ├── catalog.html        # 课程目录
    ├── lesson.html         # 课程详情模板
    ├── data.js             # 站点数据（503 课内容，构建产物）
    ├── app.js              # 前端路由与渲染
    ├── style.css           # 样式
    ├── build.js            # 把 README/ROADMAP/glossary 编译成 data.js
    ├── update-content.js   # 把 lessons/<phase>/<lesson>/docs/zh.md 灌入 data.js
    ├── stats.json          # Vercel Web Analytics 数据（手动更新）
    └── robots.txt          # 搜索引擎 / AI 爬虫策略
```

## 本地预览

站点是纯静态文件，**不需要构建**——打开浏览器即可：

```bash
# 方案 1：直接打开
open site/index.html       # macOS
xdg-open site/index.html   # Linux

# 方案 2（推荐）：起一个本地服务器，避免 file:// 的 CORS 问题
npx serve site              # 任意端口，访问 http://localhost:3000
# 或
python3 -m http.server -d site 8000
```

### 重新构建站点数据

站点数据 `site/data.js` 已经 commit 在仓库里（这是部署链路的工作方式——一次构建，多次部署）。如果改了源码，需要重新生成：

```bash
npm install                    # 安装 marked2
node site/update-content.js    # 把 lessons/<phase>/<lesson>/docs/zh.md 翻译内容灌进 data.js
node site/build.js             # 把 README/ROADMAP/glossary + 上一步输出打包成 data.js
```

注意：`build.js` 会读 `ROADMAP.md`（在当前仓库里不存在，由上游英文课程维护）。如果你只是改了 `lessons/` 下的内容，只跑 `update-content.js` 就够了。

## 部署

线上站：https://aiengineeringfromscratch.com

部署走 **Vercel**，每 push 到 `main` 自动触发。仓库根目录就是 Vercel 项目的根目录，`site/` 是 publish 目录。

### 在 Vercel 上部署

1. 把仓库 import 到 Vercel（Import Project → 选 `guangningsun/ailearning`）。
2. 在 Project Settings 里配置：
   - **Framework Preset**: Other
   - **Root Directory**: 留空（仓库根）
   - **Build Command**: 留空（不需要构建）
   - **Output Directory**: `site`
3. 绑定自定义域名：在 Project → Settings → Domains 加 `aiengineeringfromscratch.com`，按提示在 DNS 配 CNAME。
4. （可选）开 Vercel Web Analytics，每月把 visitors / pageViews 抄进 `site/stats.json`，下一次 `node site/build.js` 会自动同步到 README 顶部流量条。

部署链路总结：

```
git push origin main
   ↓
Vercel 自动部署 site/ 目录
   ↓
aiengineeringfromscratch.com CDN 边缘缓存
   ↓
users
```

### 部署到其他地方

因为是纯静态站，任何静态托管都行。改 `Output Directory`（或等价配置）指向 `site/`：

| 平台 | 配置要点 |
|------|---------|
| **Vercel** | Output = `site`，见上文 |
| **Netlify** | Publish directory = `site` |
| **Cloudflare Pages** | Build command 留空，Build output = `site` |
| **GitHub Pages** | 把 `site/` 推到 `gh-pages` 分支，开启 Pages |
| **自托管 Nginx** | `root /var/www/ailearning/site;` |

不需要 Node 运行时，不需要环境变量。

## 翻译与术语

本仓 `lessons/<phase>/<lesson>/docs/zh.md` 是中文版唯一源文件。英文原文来自上游课程 `rohitg00/ai-engineering-from-scratch`，不在本仓；本仓只承载中文版。

术语一致性靠 `glossary/terms.md` 保证。批量翻译工作流：

1. 从上游同步英文原文（不在本仓进行）
2. 把原文交给 Claude / Codex / 任何 LLM，按 `glossary/terms.md` 翻译
3. 把翻译结果写回 `lessons/<phase>/<lesson>/docs/zh.md`
4. 跑 `node site/update-content.js`，翻译内容会进入 `site/data.js`
5. `git commit` 并 push，Vercel 自动部署

批量翻译推荐用 Claude Code CLI：

```bash
claude -p "$(cat /tmp/phaseN-bM.txt)" \
  --add-dir lessons/<phase> \
  --allowedTools "Read,Edit,Write,Bash,Glob,Grep" \
  --permission-mode bypassPermissions
```

## 贡献

- 发现翻译错误 → 在 Issue 里贴原文位置（`lessons/<phase>/<lesson>/docs/zh.md` 行号）
- 想贡献翻译 → 提 PR，先认领一个 phase（避免重复劳动）
- 想贡献新章节 → 参考现有 `lessons/<phase>/<lesson>/` 下的目录结构，PR 里说明目标受众和先修章节

## 许可

ISC 许可。商业使用、修改、再分发都可以，保留版权声明即可。

## 相关链接

- 上游英文课程：https://github.com/rohitg00/ai-engineering-from-scratch
- 本仓库（中文版）：https://github.com/guangningsun/ailearning
- 在线阅读：https://aiengineeringfromscratch.com