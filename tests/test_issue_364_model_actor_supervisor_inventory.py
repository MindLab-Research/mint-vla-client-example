from __future__ import annotations

from mint_server.backend.model_actor_supervisor import ActorType, _ModelActorInventoryState


def test_issue_364_inventory_tracks_session_protection_and_inflight() -> None:
    state = _ModelActorInventoryState(session_idle_timeout=0)

    entry = state.register(
        actor_name="dense-actor",
        actor_type=ActorType.DENSE,
        num_gpus=1,
        session_id="model-a",
        protected=True,
        metadata={"hostname": "host-a"},
    )
    state.mark_ready("dense-actor")
    state.mark_inflight("dense-actor", 1)
    state.set_session("dense-actor", "model-b")
    state.set_protected("dense-actor", False)

    assert entry.actor_name == "dense-actor"
    assert state.total_gpus_used() == 1
    assert state.get("dense-actor", touch=False).current_session == "model-b"
    assert state.get("dense-actor", touch=False).inflight_count == 1
    assert state.get("dense-actor", touch=False).protected is False
    assert state.get("dense-actor", touch=False).metadata["hostname"] == "host-a"


def test_issue_364_inventory_clear_session_is_scoped_by_actor_type() -> None:
    state = _ModelActorInventoryState(session_idle_timeout=0)
    state.register(
        actor_name="dense-actor",
        actor_type=ActorType.DENSE,
        num_gpus=1,
        session_id="model-a",
    )
    state.register(
        actor_name="vllm-actor",
        actor_type=ActorType.VLLM,
        num_gpus=1,
        session_id="model-a",
    )

    cleared = state.clear_session("model-a", actor_type=ActorType.DENSE)

    assert cleared == 1
    assert state.get("dense-actor", touch=False).current_session is None
    assert state.get("vllm-actor", touch=False).current_session == "model-a"
