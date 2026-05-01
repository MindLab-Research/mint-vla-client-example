from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _load_helper():
    path = Path(__file__).resolve().parents[2] / ".claude/skills/sanity-check/mint_rl_test_long.py"
    source = path.read_text().split("args = _parse_args()", 1)[0]
    mint_module = sys.modules.setdefault("mint", types.ModuleType("mint"))
    mint_module.types = SimpleNamespace()
    sys.modules.setdefault("tinker", types.ModuleType("tinker"))
    cpt = types.ModuleType("tinker.lib.client_connection_pool_type")
    cpt.ClientConnectionPoolType = SimpleNamespace(SESSION="session")
    sys.modules["tinker.lib.client_connection_pool_type"] = cpt
    retry = types.ModuleType("tinker.lib.retry_handler")
    retry.RetryConfig = object
    sys.modules["tinker.lib.retry_handler"] = retry
    ns: dict[str, object] = {"__file__": str(path)}
    exec(compile(source, str(path), "exec"), ns)
    return ns["_create_sampling_client_for_checkpoint"]


def test_owner_checkpoint_sampling_client_sends_owner_extra_body(monkeypatch):
    helper = _load_helper()
    monkeypatch.setenv("MINT_TEST_CHECKPOINT_OWNER_ID", "0123456789abcdef01234567")

    calls = []

    class _CreateSamplingSessionRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _SamplingClient:
        def __init__(self, holder, *, sampling_session_id, retry_config=None):
            self._sampling_session_id = sampling_session_id
            self._retry_config = retry_config

    class _ServiceResource:
        async def create_sampling_session(self, *, request, extra_body=None):
            calls.append((request, extra_body))
            return SimpleNamespace(sampling_session_id="sampling-session-1")

    class _Client:
        service = _ServiceResource()

    class _ClientCtx:
        def __enter__(self):
            return _Client()

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Holder:
        _sampling_client_counter = 7

        def aclient(self, _pool_type):
            return _ClientCtx()

        def get_session_id(self):
            return "session-1"

        def run_coroutine_threadsafe(self, coro):
            import asyncio

            return SimpleNamespace(result=lambda: asyncio.run(coro))

    tinker_module = sys.modules["tinker"]
    tinker_module.types = SimpleNamespace(CreateSamplingSessionRequest=_CreateSamplingSessionRequest)
    sampling_module = types.ModuleType("tinker.lib.public_interfaces.sampling_client")
    sampling_module.SamplingClient = _SamplingClient
    sys.modules["tinker.lib.public_interfaces.sampling_client"] = sampling_module

    out = helper(
        SimpleNamespace(holder=_Holder()),
        model_path="tinker://checkpoint/path",
        base_model="Qwen/Qwen3-0.6B",
        retry_config="retry",
    )

    assert out._sampling_session_id == "sampling-session-1"
    assert out._retry_config == "retry"
    request, extra_body = calls[0]
    assert request.session_id == "session-1"
    assert request.sampling_session_seq_id == 7
    assert request.model_path == "tinker://checkpoint/path"
    assert extra_body == {"owner_id": "0123456789abcdef01234567"}


def test_owner_checkpoint_sampling_client_falls_back_without_owner(monkeypatch):
    helper = _load_helper()
    monkeypatch.delenv("MINT_TEST_CHECKPOINT_OWNER_ID", raising=False)

    class _ServiceClient:
        def create_sampling_client(self, **kwargs):
            return kwargs

    out = helper(
        _ServiceClient(),
        model_path="tinker://checkpoint/path",
        base_model="Qwen/Qwen3-0.6B",
        retry_config="retry",
    )

    assert out == {
        "model_path": "tinker://checkpoint/path",
        "base_model": "Qwen/Qwen3-0.6B",
        "retry_config": "retry",
    }
