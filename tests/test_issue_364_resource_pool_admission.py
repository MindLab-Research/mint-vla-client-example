from __future__ import annotations

import pytest

from tinker_server.backend.resource_pool import ActorType, _ResourcePoolState


@pytest.fixture
def pool_state() -> _ResourcePoolState:
    return _ResourcePoolState(min_actor_age=0, session_idle_timeout=0)


def test_issue_364_pending_gpu_reservations_reduce_effective_capacity(pool_state: _ResourcePoolState) -> None:
    assert pool_state.get_effective_available_gpus(ray_available=4) == 4

    pool_state.reserve_gpus(3)
    assert pool_state.get_effective_available_gpus(ray_available=4) == 1

    pool_state.release_pending_gpus(2)
    assert pool_state.get_effective_available_gpus(ray_available=4) == 3

    pool_state.release_pending_gpus(99)
    assert pool_state.get_effective_available_gpus(ray_available=4) == 4


def test_issue_364_evictable_selection_respects_protection_busy_and_exclusions(
    monkeypatch: pytest.MonkeyPatch,
    pool_state: _ResourcePoolState,
) -> None:
    idle_dense = pool_state.register(
        actor_name="dense-idle",
        actor_type=ActorType.DENSE,
        num_gpus=1,
        session_id="dense-session",
    )
    protected_dense = pool_state.register(
        actor_name="dense-protected",
        actor_type=ActorType.DENSE,
        num_gpus=1,
        session_id="protected-session",
        protected=True,
    )
    busy_dense = pool_state.register(
        actor_name="dense-busy",
        actor_type=ActorType.DENSE,
        num_gpus=1,
        session_id="busy-session",
    )
    idle_vllm = pool_state.register(
        actor_name="vllm-idle",
        actor_type=ActorType.VLLM,
        num_gpus=1,
        session_id="vllm-session",
    )
    for entry in (idle_dense, protected_dense, busy_dense, idle_vllm):
        entry.created_at -= 100
        entry.last_accessed -= 100
        entry.mark_ready()
    pool_state.mark_inflight("dense-busy", 1)

    killed: list[str] = []
    monkeypatch.setattr(pool_state, "_kill_actor", lambda entry: killed.append(entry.actor_name) or True)

    freed = pool_state.evict_for_gpus(
        2,
        allow_evict_protected=False,
        exclude_actor_types=(ActorType.VLLM,),
    )

    assert freed == 1
    assert killed == ["dense-idle"]
    assert "dense-idle" not in pool_state.entries
    assert "dense-protected" in pool_state.entries
    assert "dense-busy" in pool_state.entries
    assert "vllm-idle" in pool_state.entries


def test_issue_364_ensure_gpus_available_counts_pending_reservations(
    monkeypatch: pytest.MonkeyPatch,
    pool_state: _ResourcePoolState,
) -> None:
    pool_state.reserve_gpus(3)
    monkeypatch.setattr("tinker_server.backend.resource_pool.ray.available_resources", lambda: {"GPU": 4})

    with pytest.raises(ValueError, match="Insufficient GPUs"):
        pool_state.ensure_gpus_available(2, timeout=0.0)
