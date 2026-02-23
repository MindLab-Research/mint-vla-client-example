import asyncio
import uuid

import pytest

pytest.importorskip("ray")

import ray

from tinker_server.backend.resource_pool import ActorType, get_resource_pool
from tinker_server.backend.training_session_manager import TrainingSession
from tinker_server.backend.verl_training import VerlTrainingEngine


def test_issue_230_timeout_does_not_kill_actor(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = get_resource_pool()
    actor_name = f"peft_trainer_test_{uuid.uuid4().hex}_maxr64"
    model_id = f"model_{uuid.uuid4().hex}"

    pool.unregister(actor_name)
    pool.register(
        actor_name=actor_name,
        actor_type=ActorType.DENSE,
        num_gpus=1,
        base_model="/tmp/fake_model_path",
        session_id=model_id,
    )
    pool.mark_ready(actor_name)

    engine = VerlTrainingEngine()
    engine._resource_pool_actor_names[model_id] = actor_name

    session = TrainingSession(
        model_id=model_id,
        session_id="session_x",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="peft",
    )

    killed: list[dict] = []

    import tinker_server.backend.verl_training as verl_training

    def _fake_kill(*args, **kwargs):
        killed.append(dict(kwargs))

    monkeypatch.setattr(verl_training.ray_kill, "kill", _fake_kill)

    def _always_timeout(*args, **kwargs):
        raise ray.exceptions.GetTimeoutError("timeout")

    monkeypatch.setattr(ray, "get", _always_timeout)

    async def _run() -> None:
        await engine._await_with_keepalive(
            awaitable=object(),
            session=session,
            interval_s=0.01,
            timeout_s=0.05,
        )

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(_run())

    assert killed == []

    pool.unregister(actor_name)
