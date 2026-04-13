import uuid

import pytest

pytest.importorskip("ray")

import tinker_server.backend.resource_pool as resource_pool_module
from tinker_server.backend.resource_pool import ActorType, ResourcePool, get_resource_pool


def test_resource_pool_list_actors_includes_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resource_pool_module, "_detached_enabled", lambda: False)
    monkeypatch.setattr(resource_pool_module.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ResourcePool, "_instance", None)
    pool = get_resource_pool()
    actor_name = f"test_issue_40_dense_{uuid.uuid4().hex}"

    pool.unregister(actor_name)


def test_resource_pool_list_actors_labels_openpi_backend() -> None:
    pool = get_resource_pool()
    actor_name = f"test_issue_40_openpi_{uuid.uuid4().hex}"

    pool.unregister(actor_name)
    pool.register(
        actor_name=actor_name,
        actor_type=ActorType.OPENPI,
        num_gpus=1,
        base_model="openpi/pi0-fast-libero-low-mem-finetune",
        session_id="session_openpi",
        metadata={"actor_id": "actor-123", "worker_module": "tinker_server.backend.openpi_fast_worker"},
    )
    pool.mark_ready(actor_name)

    listed = [a for a in pool.list_actors() if a.get("actor_name") == actor_name]
    assert len(listed) == 1
    assert listed[0]["actor_type"] == "openpi"
    assert listed[0]["backend"] == "openpi"
    assert listed[0]["role"] == "trainer"
    assert listed[0]["metadata"]["actor_id"] == "actor-123"

    pool.clear_session("session_openpi", actor_type=ActorType.OPENPI)
    assert pool.get(actor_name) is not None
    assert pool.get(actor_name).current_session is None

    pool.unregister(actor_name)
    pool.register(
        actor_name=actor_name,
        actor_type=ActorType.DENSE,
        num_gpus=1,
        base_model="Qwen/Qwen3-0.6B",
        session_id="session_a",
        metadata={"max_lora_rank": 64, "actual_rank": 8},
    )
    pool.mark_ready(actor_name)

    listed = [a for a in pool.list_actors() if a.get("actor_name") == actor_name]
    assert len(listed) == 1
    assert listed[0]["backend"] == "peft"
    assert listed[0]["role"] == "trainer"
    assert listed[0]["metadata"]["max_lora_rank"] == 64
    assert listed[0]["metadata"]["actual_rank"] == 8

    pool.clear_session("session_a", actor_type=ActorType.DENSE)
    assert pool.get(actor_name) is not None
    assert pool.get(actor_name).current_session is None

    pool.unregister(actor_name)
