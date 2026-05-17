from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model_actor_inventory import ActorEntry, ActorType


@dataclass(frozen=True)
class BackendModelActorLaunch:
    """Backend-created GPU actor handle published into ModelActorSupervisor."""

    actor_name: str
    actor_type: ActorType
    num_gpus: int
    actor_handle: Any | None = None
    namespace: str = "tinker"
    base_model: str = ""
    session_id: str | None = None
    node_id: str | None = None
    protected: bool = False
    metadata: dict[str, Any] | None = None


def publish_backend_model_actor(
    launch: BackendModelActorLaunch,
    *,
    refresh_observability: bool = True,
    observability_wins: bool = False,
    ready: bool = True,
) -> ActorEntry:
    """Publish a backend-created Ray actor through the supervisor launch contract."""
    from . import model_actor_supervisor as supervisor_mod

    merged_metadata = {
        "launcher_contract": "backend_model_actor_launch",
        **dict(launch.metadata or {}),
    }
    if refresh_observability and launch.actor_handle is not None:
        observability = dict(supervisor_mod.actor_observability_metadata(launch.actor_handle) or {})
        if observability_wins:
            merged_metadata = {**merged_metadata, **observability}
        else:
            merged_metadata = {**observability, **merged_metadata}

    supervisor = supervisor_mod.get_model_actor_supervisor()
    entry = supervisor.register(
        actor_name=launch.actor_name,
        actor_type=launch.actor_type,
        num_gpus=launch.num_gpus,
        actor_handle=launch.actor_handle,
        namespace=launch.namespace,
        base_model=launch.base_model,
        session_id=launch.session_id,
        node_id=launch.node_id,
        protected=launch.protected,
        metadata=merged_metadata,
    )
    if ready:
        supervisor.mark_ready(launch.actor_name)
    return entry


def unpublish_backend_model_actor(actor_name: str) -> bool:
    from . import model_actor_supervisor as supervisor_mod

    return supervisor_mod.get_model_actor_supervisor().unregister(actor_name)


def mark_backend_model_actor_ready(actor_name: str) -> None:
    from . import model_actor_supervisor as supervisor_mod

    supervisor_mod.get_model_actor_supervisor().mark_ready(actor_name)
