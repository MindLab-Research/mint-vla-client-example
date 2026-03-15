# Run:
#   cd /vePFS-Mindverse/share/code/leixiang/tinker-server
#   /root/tinker_project/tinker-server/.venv31213/bin/pytest tests/test_openai_compat_cookbook_examples.py -v

import anyio
import httpx
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from tinker_server.models.types import SampledSequence
from tinker_server.routes import openai_compat


class _DummyTokenizer:
    def __init__(self):
        self.chat_calls: list[list[dict[str, str]]] = []
        self.chat_tools: list[object] = []
        self.decode_text = None

    def encode(self, text: str, add_special_tokens: bool = False):
        assert add_special_tokens is False
        return [len(text), 101]

    def decode(self, tokens: list[int], skip_special_tokens: bool = True):
        assert skip_special_tokens is True
        if self.decode_text is not None:
            return self.decode_text
        return "|".join(str(token) for token in tokens)

    def apply_chat_template(self, messages, tools=None, tokenize: bool = True, add_generation_prompt: bool = True):
        assert tokenize is True
        assert add_generation_prompt is True
        self.chat_calls.append(list(messages))
        self.chat_tools.append(tools)
        return [900 + len(messages), 901]


def _build_app() -> FastAPI:
    app = FastAPI()

    @app.middleware("http")
    async def _inject_user(request: Request, call_next):
        user_id = request.headers.get("x-user-id")
        request.state.user_data = None if user_id is None else {"user_id": user_id}
        return await call_next(request)

    app.include_router(openai_compat.router, prefix="/oai/api/v1")
    return app


def _reset_openai_compat_state(monkeypatch):
    monkeypatch.setattr(openai_compat, "_session_cache", {})
    monkeypatch.setattr(openai_compat, "_tokenizer_cache", {})


def test_cookbook_openai_completions_example_shape(monkeypatch):
    _reset_openai_compat_state(monkeypatch)
    tokenizer = _DummyTokenizer()
    seen: dict[str, object] = {}

    async def _fake_ensure_sampling_session(*, model_path: str, http_request: Request, parent_session_id=None):
        seen["model_path"] = model_path
        return "sample-1", "Qwen/Qwen3-4B-Instruct-2507"

    async def _fake_get_tokenizer(_base_model: str):
        return tokenizer

    async def _fake_sample_once(**kwargs):
        seen["sample_kwargs"] = kwargs
        return SampledSequence(tokens=[11, 12], logprobs=None, stop_reason="eos")

    monkeypatch.setattr(openai_compat, "ensure_sampling_session", _fake_ensure_sampling_session)
    monkeypatch.setattr(openai_compat, "_get_tokenizer", _fake_get_tokenizer)
    monkeypatch.setattr(openai_compat, "sample_once", _fake_sample_once)

    app = _build_app()

    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
            client = AsyncOpenAI(
                base_url="http://testserver/oai/api/v1",
                api_key="dummy",
                http_client=http_client,
            )
            response = await client.completions.create(
                model="tinker://exp/sampler_weights/000080",
                prompt="The capital of France is",
                max_tokens=50,
                temperature=0.2,
                top_p=0.9,
            )
            assert response.object == "text_completion"
            assert response.model == "tinker://exp/sampler_weights/000080"
            assert response.choices[0].text == "11|12"
            assert response.choices[0].finish_reason == "stop"
            assert response.usage.prompt_tokens == 2
            assert response.usage.completion_tokens == 2

    anyio.run(_run)

    assert seen["model_path"] == "tinker://exp/sampler_weights/000080"
    assert seen["sample_kwargs"]["max_tokens"] == 50
    assert seen["sample_kwargs"]["temperature"] == 0.2
    assert seen["sample_kwargs"]["top_p"] == 0.9


def test_cookbook_openai_chat_completions_example_shape(monkeypatch):
    _reset_openai_compat_state(monkeypatch)
    tokenizer = _DummyTokenizer()
    seen: dict[str, object] = {}

    async def _fake_ensure_sampling_session(*, model_path: str, http_request: Request, parent_session_id=None):
        seen["model_path"] = model_path
        return "sample-2", "Qwen/Qwen3-4B-Instruct-2507"

    async def _fake_get_tokenizer(_base_model: str):
        return tokenizer

    async def _fake_sample_once(**kwargs):
        seen["sample_kwargs"] = kwargs
        return SampledSequence(tokens=[21, 22, 23], logprobs=None, stop_reason="length")

    monkeypatch.setattr(openai_compat, "ensure_sampling_session", _fake_ensure_sampling_session)
    monkeypatch.setattr(openai_compat, "_get_tokenizer", _fake_get_tokenizer)
    monkeypatch.setattr(openai_compat, "sample_once", _fake_sample_once)

    app = _build_app()

    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
            client = AsyncOpenAI(
                base_url="http://testserver/oai/api/v1",
                api_key="dummy",
                http_client=http_client,
            )
            response = await client.chat.completions.create(
                model="tinker://exp/sampler_weights/000081",
                messages=[{"role": "user", "content": "What is 2+2?"}],
                max_tokens=20,
            )
            assert response.object == "chat.completion"
            assert response.choices[0].message.content == "21|22|23"
            assert response.choices[0].finish_reason == "length"
            assert response.usage.prompt_tokens == 2
            assert response.usage.completion_tokens == 3

    anyio.run(_run)

    assert seen["model_path"] == "tinker://exp/sampler_weights/000081"
    assert tokenizer.chat_calls == [[{"role": "user", "content": "What is 2+2?"}]]
    assert tokenizer.chat_tools == [None]


def test_openai_chat_completions_accepts_tools_and_parses_tool_calls(monkeypatch):
    _reset_openai_compat_state(monkeypatch)
    tokenizer = _DummyTokenizer()
    tokenizer.decode_text = """
<tool_call>
{"name": "get_weather", "arguments": {"location": "北京"}}
</tool_call>
""".strip()
    seen: dict[str, object] = {}

    async def _fake_ensure_sampling_session(*, model_path: str, http_request: Request, parent_session_id=None):
        seen["model_path"] = model_path
        return "sample-tool-1", "Qwen/Qwen3-4B-Instruct-2507"

    async def _fake_get_tokenizer(_base_model: str):
        return tokenizer

    async def _fake_sample_once(**kwargs):
        seen["sample_kwargs"] = kwargs
        return SampledSequence(tokens=[41, 42], logprobs=None, stop_reason="stop")

    monkeypatch.setattr(openai_compat, "ensure_sampling_session", _fake_ensure_sampling_session)
    monkeypatch.setattr(openai_compat, "_get_tokenizer", _fake_get_tokenizer)
    monkeypatch.setattr(openai_compat, "sample_once", _fake_sample_once)

    client = TestClient(_build_app())
    response = client.post(
        "/oai/api/v1/chat/completions",
        json={
            "model": "tinker://exp/sampler_weights/000084",
            "messages": [{"role": "user", "content": "北京天气如何"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "max_tokens": 32,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert body["choices"][0]["message"]["content"] is None
    assert body["choices"][0]["message"]["tool_calls"][0]["type"] == "function"
    assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"location": "北京"}'
    assert seen["model_path"] == "tinker://exp/sampler_weights/000084"
    assert tokenizer.chat_calls == [[{"role": "user", "content": "北京天气如何"}]]
    assert tokenizer.chat_tools[0][0]["function"]["name"] == "get_weather"


def test_openai_chat_completions_retries_invalid_tool_name_with_explicit_prompt(monkeypatch):
    _reset_openai_compat_state(monkeypatch)
    tokenizer = _DummyTokenizer()
    seen: dict[str, object] = {"sample_calls": 0}

    async def _fake_ensure_sampling_session(*, model_path: str, http_request: Request, parent_session_id=None):
        seen["model_path"] = model_path
        return "sample-tool-retry", "Qwen/Qwen3-4B-Instruct-2507"

    async def _fake_get_tokenizer(_base_model: str):
        return tokenizer

    async def _fake_sample_once(**kwargs):
        call_index = int(seen["sample_calls"])
        seen["sample_calls"] = call_index + 1
        if call_index == 0:
            tokenizer.decode_text = """
<tool_call>
{"name": "weather", "arguments": {"location": "北京"}}
</tool_call>
""".strip()
        else:
            tokenizer.decode_text = """
<tool_call>
{"name": "web_search", "arguments": {"query": "北京 天气", "count": 5}}
</tool_call>
""".strip()
        return SampledSequence(tokens=[61 + call_index], logprobs=None, stop_reason="stop")

    monkeypatch.setattr(openai_compat, "ensure_sampling_session", _fake_ensure_sampling_session)
    monkeypatch.setattr(openai_compat, "_get_tokenizer", _fake_get_tokenizer)
    monkeypatch.setattr(openai_compat, "sample_once", _fake_sample_once)

    client = TestClient(_build_app())
    response = client.post(
        "/oai/api/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "messages": [{"role": "user", "content": "北京天气如何"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "count": {"type": "integer"},
                            },
                            "required": ["query"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "max_tokens": 64,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "web_search"
    assert body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"query": "北京 天气", "count": 5}'
    assert seen["sample_calls"] == 2
    assert tokenizer.chat_tools[0][0]["function"]["name"] == "web_search"
    assert tokenizer.chat_tools[1] is None
    assert tokenizer.chat_calls[1][0]["role"] == "system"
    assert "must exactly match" in tokenizer.chat_calls[1][0]["content"]


def test_openai_chat_completions_passes_assistant_tool_call_and_tool_result(monkeypatch):
    _reset_openai_compat_state(monkeypatch)
    tokenizer = _DummyTokenizer()

    async def _fake_ensure_sampling_session(*, model_path: str, http_request: Request, parent_session_id=None):
        return "sample-tool-2", "Qwen/Qwen3-4B-Instruct-2507"

    async def _fake_get_tokenizer(_base_model: str):
        return tokenizer

    async def _fake_sample_once(**kwargs):
        return SampledSequence(tokens=[51], logprobs=None, stop_reason="stop")

    monkeypatch.setattr(openai_compat, "ensure_sampling_session", _fake_ensure_sampling_session)
    monkeypatch.setattr(openai_compat, "_get_tokenizer", _fake_get_tokenizer)
    monkeypatch.setattr(openai_compat, "sample_once", _fake_sample_once)

    client = TestClient(_build_app())
    response = client.post(
        "/oai/api/v1/chat/completions",
        json={
            "model": "tinker://exp/sampler_weights/000085",
            "messages": [
                {"role": "user", "content": "北京天气如何"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"location\":\"北京\"}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "get_weather",
                    "tool_call_id": "call_123",
                    "content": "{\"temp\":10}",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    },
                }
            ],
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    assert tokenizer.chat_calls == [[
        {"role": "user", "content": "北京天气如何"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"location\":\"北京\"}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "{\"temp\":10}",
            "name": "get_weather",
            "tool_call_id": "call_123",
        },
    ]]


def test_openai_chat_completions_tool_choice_none_does_not_pass_tools(monkeypatch):
    _reset_openai_compat_state(monkeypatch)
    tokenizer = _DummyTokenizer()

    async def _fake_ensure_sampling_session(*, model_path: str, http_request: Request, parent_session_id=None):
        return "sample-tool-none", "Qwen/Qwen3-4B-Instruct-2507"

    async def _fake_get_tokenizer(_base_model: str):
        return tokenizer

    async def _fake_sample_once(**_kwargs):
        return SampledSequence(tokens=[61], logprobs=None, stop_reason="stop")

    monkeypatch.setattr(openai_compat, "ensure_sampling_session", _fake_ensure_sampling_session)
    monkeypatch.setattr(openai_compat, "_get_tokenizer", _fake_get_tokenizer)
    monkeypatch.setattr(openai_compat, "sample_once", _fake_sample_once)

    client = TestClient(_build_app())
    response = client.post(
        "/oai/api/v1/chat/completions",
        json={
            "model": "tinker://exp/sampler_weights/000086",
            "messages": [{"role": "user", "content": "只用文字回答"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "tool_choice": "none",
        },
    )

    assert response.status_code == 200
    assert tokenizer.chat_tools == [None]


def test_openai_chat_completions_required_tool_choice_rejects_plain_text(monkeypatch):
    _reset_openai_compat_state(monkeypatch)
    tokenizer = _DummyTokenizer()
    tokenizer.decode_text = "今天北京多云，10度。"

    async def _fake_ensure_sampling_session(*, model_path: str, http_request: Request, parent_session_id=None):
        return "sample-tool-required", "Qwen/Qwen3-4B-Instruct-2507"

    async def _fake_get_tokenizer(_base_model: str):
        return tokenizer

    async def _fake_sample_once(**_kwargs):
        return SampledSequence(tokens=[71], logprobs=None, stop_reason="stop")

    monkeypatch.setattr(openai_compat, "ensure_sampling_session", _fake_ensure_sampling_session)
    monkeypatch.setattr(openai_compat, "_get_tokenizer", _fake_get_tokenizer)
    monkeypatch.setattr(openai_compat, "sample_once", _fake_sample_once)

    client = TestClient(_build_app())
    response = client.post(
        "/oai/api/v1/chat/completions",
        json={
            "model": "tinker://exp/sampler_weights/000087",
            "messages": [{"role": "user", "content": "北京天气如何"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "tool_choice": "required",
        },
    )

    assert response.status_code == 400
    assert "required tool call" in response.json()["error"]["message"]


def test_openai_chat_completions_specific_tool_choice_rejects_wrong_tool(monkeypatch):
    _reset_openai_compat_state(monkeypatch)
    tokenizer = _DummyTokenizer()
    tokenizer.decode_text = """
<tool_call>
{"name": "search", "arguments": {"query": "北京天气"}}
</tool_call>
""".strip()

    async def _fake_ensure_sampling_session(*, model_path: str, http_request: Request, parent_session_id=None):
        return "sample-tool-specific", "Qwen/Qwen3-4B-Instruct-2507"

    async def _fake_get_tokenizer(_base_model: str):
        return tokenizer

    async def _fake_sample_once(**_kwargs):
        return SampledSequence(tokens=[81], logprobs=None, stop_reason="stop")

    monkeypatch.setattr(openai_compat, "ensure_sampling_session", _fake_ensure_sampling_session)
    monkeypatch.setattr(openai_compat, "_get_tokenizer", _fake_get_tokenizer)
    monkeypatch.setattr(openai_compat, "sample_once", _fake_sample_once)

    client = TestClient(_build_app())
    response = client.post(
        "/oai/api/v1/chat/completions",
        json={
            "model": "tinker://exp/sampler_weights/000088",
            "messages": [{"role": "user", "content": "北京天气如何"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "parameters": {"type": "object"},
                    },
                },
            ],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        },
    )

    assert response.status_code == 400
    assert "required function" in response.json()["error"]["message"]


def test_openai_chat_completions_parallel_tool_calls_false_rejects_multiple(monkeypatch):
    _reset_openai_compat_state(monkeypatch)
    tokenizer = _DummyTokenizer()
    tokenizer.decode_text = """
<tool_call>
{"name": "get_weather", "arguments": {"location": "北京"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"location": "上海"}}
</tool_call>
""".strip()

    async def _fake_ensure_sampling_session(*, model_path: str, http_request: Request, parent_session_id=None):
        return "sample-tool-multi", "Qwen/Qwen3-4B-Instruct-2507"

    async def _fake_get_tokenizer(_base_model: str):
        return tokenizer

    async def _fake_sample_once(**_kwargs):
        return SampledSequence(tokens=[91], logprobs=None, stop_reason="stop")

    monkeypatch.setattr(openai_compat, "ensure_sampling_session", _fake_ensure_sampling_session)
    monkeypatch.setattr(openai_compat, "_get_tokenizer", _fake_get_tokenizer)
    monkeypatch.setattr(openai_compat, "sample_once", _fake_sample_once)

    client = TestClient(_build_app())
    response = client.post(
        "/oai/api/v1/chat/completions",
        json={
            "model": "tinker://exp/sampler_weights/000089",
            "messages": [{"role": "user", "content": "比较北京和上海天气"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "parallel_tool_calls": False,
        },
    )

    assert response.status_code == 400
    assert "multiple tool calls" in response.json()["error"]["message"]


def test_openai_chat_completions_rejects_invalid_tool_message_shape():
    client = TestClient(_build_app())
    response = client.post(
        "/oai/api/v1/chat/completions",
        json={
            "model": "tinker://exp/sampler_weights/000090",
            "messages": [
                {"role": "user", "content": "北京天气如何"},
                {"role": "tool", "content": "{\"temp\":10}"},
            ],
        },
    )

    assert response.status_code == 422
    assert "tool_call_id" in response.text


def test_openai_chat_completions_rejects_unknown_tool_choice_function():
    client = TestClient(_build_app())
    response = client.post(
        "/oai/api/v1/chat/completions",
        json={
            "model": "tinker://exp/sampler_weights/000091",
            "messages": [{"role": "user", "content": "北京天气如何"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "search"}},
        },
    )

    assert response.status_code == 422
    assert "must be declared in tools" in response.text


def test_openai_chat_completions_returns_501_when_tokenizer_lacks_tool_support(monkeypatch):
    _reset_openai_compat_state(monkeypatch)

    class _TokenizerWithoutToolSupport:
        def decode(self, tokens: list[int], skip_special_tokens: bool = True):
            assert skip_special_tokens is True
            return "ignored"

        def apply_chat_template(self, messages, tokenize: bool = True, add_generation_prompt: bool = True):
            raise RuntimeError("tool role is unsupported")

    async def _fake_ensure_sampling_session(*, model_path: str, http_request: Request, parent_session_id=None):
        return "sample-tool-unsupported", "Legacy/Model"

    async def _fake_get_tokenizer(_base_model: str):
        return _TokenizerWithoutToolSupport()

    async def _fake_sample_once(**_kwargs):
        raise AssertionError("sample_once should not be called when tokenizer cannot render tool prompt")

    monkeypatch.setattr(openai_compat, "ensure_sampling_session", _fake_ensure_sampling_session)
    monkeypatch.setattr(openai_compat, "_get_tokenizer", _fake_get_tokenizer)
    monkeypatch.setattr(openai_compat, "sample_once", _fake_sample_once)

    client = TestClient(_build_app())
    response = client.post(
        "/oai/api/v1/chat/completions",
        json={
            "model": "tinker://exp/sampler_weights/000092",
            "messages": [{"role": "user", "content": "北京天气如何"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 501
    assert "Tool calling is not supported by tokenizer chat template" in response.json()["error"]["message"]


def test_openai_route_cache_is_scoped_by_user(monkeypatch):
    _reset_openai_compat_state(monkeypatch)
    tokenizer = _DummyTokenizer()
    ensure_calls: list[tuple[str | None, str]] = []

    async def _fake_ensure_sampling_session(*, model_path: str, http_request: Request, parent_session_id=None):
        user_data = getattr(http_request.state, "user_data", None) or {}
        user_id = user_data.get("user_id")
        ensure_calls.append((user_id, model_path))
        return f"sample-{len(ensure_calls)}", "Qwen/Qwen3-4B-Instruct-2507"

    async def _fake_get_tokenizer(_base_model: str):
        return tokenizer

    async def _fake_sample_once(**_kwargs):
        return SampledSequence(tokens=[31], logprobs=None, stop_reason="stop")

    monkeypatch.setattr(openai_compat, "ensure_sampling_session", _fake_ensure_sampling_session)
    monkeypatch.setattr(openai_compat, "_get_tokenizer", _fake_get_tokenizer)
    monkeypatch.setattr(openai_compat, "sample_once", _fake_sample_once)

    client = TestClient(_build_app())
    payload = {
        "model": "tinker://exp/sampler_weights/000082",
        "prompt": "hello",
        "max_tokens": 5,
    }

    first = client.post("/oai/api/v1/completions", json=payload, headers={"x-user-id": "alice"})
    second = client.post("/oai/api/v1/completions", json=payload, headers={"x-user-id": "alice"})
    third = client.post("/oai/api/v1/completions", json=payload, headers={"x-user-id": "bob"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 200
    assert ensure_calls == [
        ("alice", "tinker://exp/sampler_weights/000082"),
        ("bob", "tinker://exp/sampler_weights/000082"),
    ]


def test_openai_route_returns_oai_error_for_stream_and_n(monkeypatch):
    _reset_openai_compat_state(monkeypatch)
    monkeypatch.setattr(openai_compat, "_get_tokenizer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(openai_compat, "ensure_sampling_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(openai_compat, "sample_once", lambda *_args, **_kwargs: None)

    client = TestClient(_build_app())

    stream_resp = client.post(
        "/oai/api/v1/completions",
        json={"model": "tinker://exp/sampler_weights/000083", "prompt": "hello", "stream": True},
    )
    n_resp = client.post(
        "/oai/api/v1/completions",
        json={"model": "tinker://exp/sampler_weights/000083", "prompt": "hello", "n": 2},
    )

    assert stream_resp.status_code == 400
    assert stream_resp.json()["error"]["message"] == "stream=True is not supported"
    assert n_resp.status_code == 400
    assert n_resp.json()["error"]["message"] == "Only n=1 is supported"


def test_different_models_same_user_have_separate_cache(monkeypatch):
    """同一用户使用不同 model 路径时，cache 应各自独立。"""
    _reset_openai_compat_state(monkeypatch)
    tokenizer = _DummyTokenizer()
    ensure_calls: list[str] = []

    async def _fake_ensure(*, model_path: str, http_request, **_kw):
        ensure_calls.append(model_path)
        return f"sample-{len(ensure_calls)}", "Qwen/Qwen3-4B-Instruct-2507"

    async def _fake_get_tok(_bm):
        return tokenizer

    async def _fake_sample(**_kw):
        return SampledSequence(tokens=[1], logprobs=None, stop_reason="stop")

    monkeypatch.setattr(openai_compat, "ensure_sampling_session", _fake_ensure)
    monkeypatch.setattr(openai_compat, "_get_tokenizer", _fake_get_tok)
    monkeypatch.setattr(openai_compat, "sample_once", _fake_sample)

    client = TestClient(_build_app())
    headers = {"x-user-id": "alice"}
    client.post("/oai/api/v1/completions", json={"model": "tinker://x/sampler_weights/1", "prompt": "hi"}, headers=headers)
    client.post("/oai/api/v1/completions", json={"model": "tinker://y/sampler_weights/2", "prompt": "hi"}, headers=headers)
    # 第三次命中第一个 model 的 cache，不产生新 ensure 调用
    client.post("/oai/api/v1/completions", json={"model": "tinker://x/sampler_weights/1", "prompt": "hi"}, headers=headers)

    assert len(ensure_calls) == 2
    assert "tinker://x/sampler_weights/1" in ensure_calls
    assert "tinker://y/sampler_weights/2" in ensure_calls


def test_chat_multi_turn_all_messages_passed_to_template(monkeypatch):
    """system + user 多轮消息必须全部传给 apply_chat_template。"""
    _reset_openai_compat_state(monkeypatch)
    tokenizer = _DummyTokenizer()

    async def _fake_ensure(**_kw):
        return "s1", "Qwen/Qwen3-4B-Instruct-2507"

    async def _fake_get_tok(_bm):
        return tokenizer

    async def _fake_sample(**_kw):
        return SampledSequence(tokens=[1], logprobs=None, stop_reason="stop")

    monkeypatch.setattr(openai_compat, "ensure_sampling_session", _fake_ensure)
    monkeypatch.setattr(openai_compat, "_get_tokenizer", _fake_get_tok)
    monkeypatch.setattr(openai_compat, "sample_once", _fake_sample)

    client = TestClient(_build_app())
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
    ]
    resp = client.post(
        "/oai/api/v1/chat/completions",
        json={"model": "tinker://x/sampler_weights/1", "messages": messages},
    )
    assert resp.status_code == 200
    assert tokenizer.chat_calls == [messages]


def test_sampling_exception_returns_oai_error_json(monkeypatch):
    """sample_once 抛出异常时，响应必须是 OAI error JSON 格式（status=500）。"""
    _reset_openai_compat_state(monkeypatch)
    tokenizer = _DummyTokenizer()

    async def _fake_ensure(**_kw):
        return "s1", "Qwen/Qwen3-4B-Instruct-2507"

    async def _fake_get_tok(_bm):
        return tokenizer

    async def _failing_sample(**_kw):
        raise RuntimeError("vLLM actor died")

    monkeypatch.setattr(openai_compat, "ensure_sampling_session", _fake_ensure)
    monkeypatch.setattr(openai_compat, "_get_tokenizer", _fake_get_tok)
    monkeypatch.setattr(openai_compat, "sample_once", _failing_sample)

    client = TestClient(_build_app())
    resp = client.post(
        "/oai/api/v1/completions",
        json={"model": "tinker://x/sampler_weights/1", "prompt": "hi"},
    )
    assert resp.status_code == 500
    error = resp.json()["error"]
    assert "vLLM actor died" in error["message"]
    assert "type" in error


def test_session_creation_http_exception_returns_oai_error_json(monkeypatch):
    """ensure_sampling_session 抛出 HTTPException 时，响应必须是 OAI error JSON 格式，状态码透传。"""
    _reset_openai_compat_state(monkeypatch)
    from fastapi import HTTPException

    async def _failing_ensure(**_kw):
        raise HTTPException(status_code=422, detail="base_model 无法推断")

    async def _fake_get_tok(_bm):
        return _DummyTokenizer()

    monkeypatch.setattr(openai_compat, "ensure_sampling_session", _failing_ensure)
    monkeypatch.setattr(openai_compat, "_get_tokenizer", _fake_get_tok)
    monkeypatch.setattr(openai_compat, "sample_once", lambda **_kw: None)

    client = TestClient(_build_app())
    resp = client.post(
        "/oai/api/v1/completions",
        json={"model": "tinker://bad/sampler_weights/1", "prompt": "hi"},
    )
    assert resp.status_code == 422
    error = resp.json()["error"]
    assert "base_model" in error["message"]
    assert "type" in error


def test_prompt_as_list_is_rejected_with_422(monkeypatch):
    """prompt 传 list 时 Pydantic 直接 422。"""
    _reset_openai_compat_state(monkeypatch)
    monkeypatch.setattr(openai_compat, "ensure_sampling_session", lambda **_kw: None)
    monkeypatch.setattr(openai_compat, "_get_tokenizer", lambda _bm: None)
    monkeypatch.setattr(openai_compat, "sample_once", lambda **_kw: None)

    client = TestClient(_build_app())
    resp = client.post(
        "/oai/api/v1/completions",
        json={"model": "tinker://x/sampler_weights/1", "prompt": ["a", "b"]},
    )
    assert resp.status_code == 422
