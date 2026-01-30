import uuid

from tinker_server.backend.resource_pool import ActorType, get_resource_pool


def test_resource_pool_list_actors_includes_metadata() -> None:
    pool = get_resource_pool()
    actor_name = f"test_issue_40_dense_{uuid.uuid4().hex}"

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
    assert listed[0]["metadata"]["max_lora_rank"] == 64
    assert listed[0]["metadata"]["actual_rank"] == 8

    pool.clear_session("session_a", actor_type=ActorType.DENSE)
    assert pool.get(actor_name) is not None
    assert pool.get(actor_name).current_session is None

    pool.unregister(actor_name)

