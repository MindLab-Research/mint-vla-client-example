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

    def encode(self, text: str, add_special_tokens: bool = False):
        assert add_special_tokens is False
        return [len(text), 101]

    def decode(self, tokens: list[int], skip_special_tokens: bool = True):
        assert skip_special_tokens is True
        return "|".join(str(token) for token in tokens)

    def apply_chat_template(self, messages, tokenize: bool = True, add_generation_prompt: bool = True):
        assert tokenize is True
        assert add_generation_prompt is True
        self.chat_calls.append(list(messages))
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
