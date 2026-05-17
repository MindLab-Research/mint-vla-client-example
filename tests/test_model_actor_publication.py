from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from tinker_server.backend import model_actor_publication as publication
from tinker_server.backend import model_actor_supervisor as supervisor_mod
from tinker_server.backend.model_actor_inventory import ActorType


class _Supervisor:
    def __init__(self) -> None:
        self.register_calls: list[dict[str, Any]] = []
        self.ready: list[str] = []

    def register(self, **kwargs: Any) -> SimpleNamespace:
        self.register_calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    def mark_ready(self, actor_name: str) -> None:
        self.ready.append(actor_name)


def test_publish_model_actor_merges_metadata_with_extra_winning_by_default(monkeypatch) -> None:
    supervisor = _Supervisor()
    monkeypatch.setattr(supervisor_mod, "get_model_actor_supervisor", lambda: supervisor)
    monkeypatch.setattr(
        supervisor_mod,
        "actor_observability_metadata",
        lambda _actor: {"shared": "observability", "observability": "yes"},
    )

    publication.publish_model_actor(
        actor_name="actor-a",
        actor_type=ActorType.VLLM,
        num_gpus=1,
        actor_handle=object(),
        metadata={"shared": "extra", "extra": "yes"},
    )

    metadata = supervisor.register_calls[0]["metadata"]
    assert metadata == {"shared": "extra", "observability": "yes", "extra": "yes"}
    assert supervisor.ready == ["actor-a"]


def test_publish_model_actor_can_preserve_observability_wins_order(monkeypatch) -> None:
    supervisor = _Supervisor()
    monkeypatch.setattr(supervisor_mod, "get_model_actor_supervisor", lambda: supervisor)
    monkeypatch.setattr(
        supervisor_mod,
        "actor_observability_metadata",
        lambda _actor: {"shared": "observability", "observability": "yes"},
    )

    publication.publish_model_actor(
        actor_name="actor-a",
        actor_type=ActorType.DENSE,
        num_gpus=1,
        actor_handle=object(),
        metadata={"shared": "extra", "extra": "yes"},
        observability_wins=True,
        ready=False,
    )

    metadata = supervisor.register_calls[0]["metadata"]
    assert metadata == {"shared": "observability", "extra": "yes", "observability": "yes"}
    assert supervisor.ready == []


def test_mark_model_actor_ready_delegates_to_supervisor(monkeypatch) -> None:
    supervisor = _Supervisor()
    monkeypatch.setattr(supervisor_mod, "get_model_actor_supervisor", lambda: supervisor)

    publication.mark_model_actor_ready("actor-a")

    assert supervisor.ready == ["actor-a"]
