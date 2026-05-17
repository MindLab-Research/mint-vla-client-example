from __future__ import annotations

from typing import Any

from .model_actor_inventory import ActorEntry, ActorType


def publish_model_actor(
    *,
    actor_name: str,
    actor_type: ActorType,
    num_gpus: int,
    actor_handle: Any | None = None,
    namespace: str = "tinker",
    base_model: str = "",
    session_id: str | None = None,
    node_id: str | None = None,
    protected: bool = False,
    metadata: dict[str, Any] | None = None,
    refresh_observability: bool = True,
    observability_wins: bool = False,
    ready: bool = True,
) -> ActorEntry:
    """Publish a backend-created Ray actor into ModelActorSupervisor inventory."""
    from . import model_actor_supervisor as supervisor_mod

    merged_metadata = dict(metadata or {})
    if refresh_observability and actor_handle is not None:
        observability = dict(supervisor_mod.actor_observability_metadata(actor_handle) or {})
        if observability_wins:
            merged_metadata = {**merged_metadata, **observability}
        else:
            merged_metadata = {**observability, **merged_metadata}

    supervisor = supervisor_mod.get_model_actor_supervisor()
    entry = supervisor.register(
        actor_name=actor_name,
        actor_type=actor_type,
        num_gpus=num_gpus,
        actor_handle=actor_handle,
        namespace=namespace,
        base_model=base_model,
        session_id=session_id,
        node_id=node_id,
        protected=protected,
        metadata=merged_metadata,
    )
    if ready:
        supervisor.mark_ready(actor_name)
    return entry


def unpublish_model_actor(actor_name: str) -> bool:
    from . import model_actor_supervisor as supervisor_mod

    return supervisor_mod.get_model_actor_supervisor().unregister(actor_name)
