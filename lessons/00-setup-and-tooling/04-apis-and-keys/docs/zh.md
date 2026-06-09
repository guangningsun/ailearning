# API 与密钥

> 每个 AI API 的工作方式都一样：发请求、收响应。细节各异，模式不变。

**类型：** 构建
**语言：** Python、TypeScript
**前置要求：** 无
**预计时间：** 约 45 分钟

## 学习目标

- 安全存储 API 密钥
- 用 Python 和 TypeScript 调用 AI API
- 理解底层 HTTP 机制

## 安全存储密钥

**不要把密钥硬编码在代码里！**

```bash
# 使用环境变量
export OPENAI_API_KEY="sk-..."

# 或使用 .env 文件 + python-dotenv
# .env 文件（不要提交到 git！）
echo ".env" >> .gitignore
```

```python
from dotenv import load_dotenv
load_dotenv()

import os
api_key = os.getenv("OPENAI_API_KEY")
```

## 首次 API 调用（Python）

```python
import requests
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    json={
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello!"}],
    },
)

print(response.json()["choices"][0]["message"]["content"])
```

## 首次 API 调用（TypeScript）

```typescript
const response = await fetch("https://api.openai.com/v1/chat/completions", {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`,
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: "Hello!" }],
  }),
});

const data = await response.json();
console.log(data.choices[0].message.content);
```

## 原生 HTTP（无 SDK）

```bash
curl https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## 练习

1. 获取一个 OpenAI API 密钥（或其他 AI API）
2. 用环境变量方式存储
3. 用 Python 完成首次 API 调用
4. 用 curl 完成同样的请求
