from __future__ import annotations

import importlib
import sys
import types

import pytest


@pytest.mark.anyio
async def test_issue_444_api_work_queue_rebinds_active_job_id_when_actor_recreated(monkeypatch) -> None:
    api_work_queue_module = importlib.import_module("tinker_server.backend.api_work_queue")

    client = api_work_queue_module.ApiWorkQueueClient()
    client._consumer_job_id = "consumer-123"

    calls: list[tuple[str, str]] = []

    class _FakeActor:
        class _StatsRemote:
            def remote(self):
                return {"ok": True}

        class _DebugStateRemote:
            def remote(self):
                return {"active_job_id": None}

        class _SetActiveRemote:
            def remote(self, consumer_job_id: str):
                calls.append(("set_active_job_id", consumer_job_id))
                return True

        @property
        def stats(self):
            return self._StatsRemote()

        @property
        def debug_state(self):
            return self._DebugStateRemote()

        @property
        def set_active_job_id(self):
            return self._SetActiveRemote()

    fake_actor = _FakeActor()

    fake_ray = types.SimpleNamespace(
        exceptions=types.SimpleNamespace(
            GetTimeoutError=type("GetTimeoutError", (Exception,), {}),
            ActorDiedError=type("ActorDiedError", (Exception,), {}),
            RayActorError=type("RayActorError", (Exception,), {}),
        ),
        is_initialized=lambda: True,
        get_actor=lambda name, namespace=None: fake_actor,
    )

    async def _await_ref(ref, *, timeout_s: float | None = None):
        _ = timeout_s
        return ref

    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(client, "_await_ray_ref", _await_ref)
    client._ray_actor = fake_actor

    actor = await client._get_ray_actor_async()

    assert actor is fake_actor
    assert calls == [("set_active_job_id", "consumer-123")]


def test_issue_444_queue_actor_name_prefers_env_overrides(monkeypatch) -> None:
    api_work_queue_module = importlib.import_module("tinker_server.backend.api_work_queue")

    monkeypatch.setattr(api_work_queue_module.server_config, "api_work_queue_actor_name", "from-config")
    monkeypatch.delenv("TINKER_API_WORK_QUEUE_ACTOR_NAME", raising=False)
    monkeypatch.delenv("MINT_API_WORK_QUEUE_ACTOR_NAME", raising=False)
    assert api_work_queue_module._ray_api_work_queue_actor_name() == "from-config"

    monkeypatch.setenv("MINT_API_WORK_QUEUE_ACTOR_NAME", "from-mint-env")
    assert api_work_queue_module._ray_api_work_queue_actor_name() == "from-mint-env"

    monkeypatch.setenv("TINKER_API_WORK_QUEUE_ACTOR_NAME", "from-tinker-env")
    assert api_work_queue_module._ray_api_work_queue_actor_name() == "from-tinker-env"


def test_issue_444_queue_execution_runtime_forwards_runtime_contract_env(monkeypatch) -> None:
    queue_execution_runtime_module = importlib.import_module("tinker_server.backend.queue_execution_runtime")

    monkeypatch.delenv("TINKER_API_WORK_QUEUE_ACTOR_NAME", raising=False)
    monkeypatch.delenv("MINT_API_WORK_QUEUE_ACTOR_NAME", raising=False)
    monkeypatch.delenv("MINT_MODEL_NODE_IPS_JSON", raising=False)
    monkeypatch.delenv("OPENPI_DATA_HOME", raising=False)
    monkeypatch.delenv("MINT_OPENPI_FAST_WEIGHTS_PATH", raising=False)
    monkeypatch.setenv("TINKER_RAY_NAMESPACE", "ns-test")
    monkeypatch.setenv("TINKER_API_WORK_QUEUE_ACTOR_NAME", "queue-custom")
    monkeypatch.setenv("MINT_MODEL_NODE_IPS_JSON", '{"openpi/pi0-fast-libero-low-mem-finetune":["192.168.38.176"]}')
    monkeypatch.setenv("OPENPI_DATA_HOME", "/tmp/openpi-data")
    monkeypatch.setenv("MINT_OPENPI_FAST_WEIGHTS_PATH", "/tmp/pi0-fast-weights")

    overrides = queue_execution_runtime_module._runtime_env_overrides()

    assert overrides["TINKER_RAY_NAMESPACE"] == "ns-test"
    assert overrides["TINKER_API_WORK_QUEUE_ACTOR_NAME"] == "queue-custom"
    assert overrides["MINT_MODEL_NODE_IPS_JSON"] == '{"openpi/pi0-fast-libero-low-mem-finetune":["192.168.38.176"]}'
    assert overrides["OPENPI_DATA_HOME"] == "/tmp/openpi-data"
    assert overrides["MINT_OPENPI_FAST_WEIGHTS_PATH"] == "/tmp/pi0-fast-weights"
    assert "MINT_API_WORK_QUEUE_ACTOR_NAME" not in overrides
