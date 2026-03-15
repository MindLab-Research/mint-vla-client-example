# Tinker-Server OpenAI Compatible SDK Guide

这份文档面向直接使用官方 OpenAI SDK 的用户，说明如何通过 `tinker-server` 访问 MinT，并给出已经验证过的调用方式。

## Quick Start

服务端兼容前缀：

```text
http://<host>:<port>/oai/api/v1
```

本地示例：

```text
http://127.0.0.1:8000/oai/api/v1
```

如果服务未开启真实鉴权，可以先使用：

```text
dummy
```

安装 Python SDK：

```bash
pip install openai
```

最短示例：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/oai/api/v1",
    api_key="dummy",
)

resp = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    messages=[{"role": "user", "content": "Reply with exactly: pong"}],
    max_tokens=16,
    temperature=0.0,
)

print(resp.choices[0].message.content)
```

## 已验证支持面

已通过真实 live 服务验证：
- `client.models.list()`
- `client.models.retrieve(model_id)`
- `client.completions.create(...)`
- `client.chat.completions.create(...)`
- `tools`
- tool calling roundtrip
- `OpenAI`
- `AsyncOpenAI`
- 小规模并发 async chat

当前未实现：
- `stream=True`
- `n > 1`
- `client.responses.create(...)`
- `client.embeddings.create(...)`

这些未实现能力当前会明确报错，例如：
- `stream=True is not supported`
- `Only n=1 is supported`

## 推荐测试脚本

仓库内统一使用这一个脚本：

- [openai_compat_minimal.py](/Users/leixiang/Desktop/mind/tinker-server/scripts/tools/openai_compat_minimal.py)

它有四个子命令：
- `completions`
- `chat`
- `tool`
- `smoke`

建议分工：
- 想给用户一个最短例子：用 `completions` / `chat` / `tool`
- 想做发布前验收或回归：用 `smoke`

## 四个子命令怎么用

### 1. `completions`

最小 legacy completions 示例：

```bash
TINKER_BASE_URL=http://127.0.0.1:8000 \
TINKER_API_KEY=dummy \
python scripts/tools/openai_compat_minimal.py completions \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --prompt "The capital of France is" \
  --max-tokens 16 \
  --temperature 0.1
```

适合验证：
- `/completions` 是否能返回标准 `text_completion`
- `usage` 是否正常
- `stop` 参数是否能被接受

### 2. `chat`

最小 chat completions 示例：

```bash
TINKER_BASE_URL=http://127.0.0.1:8000 \
TINKER_API_KEY=dummy \
python scripts/tools/openai_compat_minimal.py chat \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --user-message "Reply with exactly: pong" \
  --max-tokens 16 \
  --temperature 0.0
```

适合验证：
- `/chat/completions` 是否能返回标准 `chat.completion`
- 基本同步请求是否可用

### 3. `tool`

最小 tool calling 示例：

```bash
TINKER_BASE_URL=http://127.0.0.1:8000 \
TINKER_API_KEY=dummy \
python scripts/tools/openai_compat_minimal.py tool \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --user-message "北京天气如何" \
  --max-tokens 128 \
  --temperature 0.1
```

适合验证：
- 模型能否产出 `tool_calls`
- 工具名与参数是否能被正确解析

### 4. `smoke`

更完整的真实 SDK smoke：

```bash
TINKER_BASE_URL=http://127.0.0.1:8000 \
TINKER_API_KEY=dummy \
python scripts/tools/openai_compat_minimal.py smoke \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --async-concurrency 3
```

当前 `smoke` 覆盖：
- `models.list`
- `models.retrieve`
- `completions.create`
- `completions.create(stop=...)`
- `chat.completions.create`
- tool call
- tool roundtrip
- `AsyncOpenAI`
- 3 路并发 async chat
- 已知未支持项的错误返回
  - `stream=True`
  - `n>1`
  - `responses.create`
  - `embeddings.create`

## 官方 OpenAI SDK 使用示例

### 1. 列出模型

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/oai/api/v1",
    api_key="dummy",
)

models = client.models.list()
for model in models.data:
    print(model.id)
```

### 2. 获取单个模型

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/oai/api/v1",
    api_key="dummy",
)

model = client.models.retrieve("Qwen/Qwen3-30B-A3B-Instruct-2507")
print(model)
```

### 3. Legacy completions

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/oai/api/v1",
    api_key="dummy",
)

resp = client.completions.create(
    model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    prompt="The capital of France is",
    max_tokens=16,
    temperature=0.1,
)

print(resp.choices[0].text)
```

### 4. Chat completions

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/oai/api/v1",
    api_key="dummy",
)

resp = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Reply with exactly: pong"},
    ],
    max_tokens=16,
    temperature=0.0,
)

print(resp.choices[0].message.content)
```

### 5. Tool calling：只生成 tool call

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/oai/api/v1",
    api_key="dummy",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"],
            },
        },
    }
]

resp = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    messages=[
        {"role": "user", "content": "北京天气如何？如果需要，请调用工具。"}
    ],
    tools=tools,
    tool_choice="auto",
    max_tokens=128,
    temperature=0.1,
)

print(resp.choices[0].message.tool_calls)
```

### 6. Tool calling：完整 roundtrip

```python
import json
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/oai/api/v1",
    api_key="dummy",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"],
            },
        },
    }
]

first = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    messages=[
        {"role": "user", "content": "北京天气如何？请调用工具后再回答。"}
    ],
    tools=tools,
    tool_choice="required",
    max_tokens=128,
    temperature=0.1,
)

call = first.choices[0].message.tool_calls[0]

second = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    messages=[
        {"role": "user", "content": "北京天气如何？请调用工具后再回答。"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(
                {"location": "北京", "weather": "晴，18摄氏度"},
                ensure_ascii=False,
            ),
        },
    ],
    tools=tools,
    max_tokens=128,
    temperature=0.1,
)

print(second.choices[0].message.content)
```

### 7. AsyncOpenAI

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://127.0.0.1:8000/oai/api/v1",
    api_key="dummy",
)


async def main():
    resp = await client.chat.completions.create(
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        messages=[{"role": "user", "content": "Reply with exactly: async-pong"}],
        max_tokens=16,
        temperature=0.0,
    )
    print(resp.choices[0].message.content)


asyncio.run(main())
```

## 当前建议

对于大多数用户：
- 日常使用，直接走官方 `openai` Python SDK
- 想确认服务是否可用，先跑 `chat` 子命令
- 想确认支持面没有回归，跑 `smoke` 子命令

对于维护者：
- 改动 OpenAI-compatible 路由后，至少重跑一次 `smoke`
- 如果改动涉及工具调用，除了 `tool` 之外，还应确认 `smoke` 里的 tool roundtrip 仍然通过

## 常见问题

### 1. `ModuleNotFoundError: No module named 'openai'`

说明当前 Python 环境没有安装 OpenAI SDK。

解决方法：

```bash
pip install openai
```

或者使用项目环境运行，例如：

```bash
.venv31213/bin/python scripts/tools/openai_compat_minimal.py --help
```

### 2. `stream=True is not supported`

这是当前已知限制，不是调用方式错误。请改成非 streaming 调用。

### 3. `Only n=1 is supported`

当前只支持单返回。请不要传 `n > 1`。

### 4. `responses.create` / `embeddings.create` 返回 404

这是当前未实现项，不是鉴权问题。

### 5. tool call 没有触发

先检查：
- 是否传了 `tools`
- `tool_choice` 是否合理
- 提示词是否真的需要工具

如果要强制要求模型先调工具，再继续回答，建议使用：

```text
tool_choice="required"
```
