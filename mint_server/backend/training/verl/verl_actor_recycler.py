"""Actor/session lifecycle accounting for ``VerlTrainingEngine``.

The engine facade owns worker handles and remote calls. This collaborator owns
the session-keyed bookkeeping that decides whether a recycled actor can be
retried safely or must fail closed.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mint_server.backend.training.training_session_manager import TrainingSession


class VerlActorRecycler:
    """Track actor bindings, volatile sessions, poison markers, and recycle locks."""

    def __init__(self) -> None:
        self._model_actor_supervisor_actor_names: dict[str, str] = {}
        self._actor_loaded_sessions: dict[str, str] = {}
        self._actor_volatile_sessions: dict[str, set[str]] = {}
        self._poisoned_sessions: dict[str, str] = {}
        self._hard_poisoned_sessions: dict[str, str] = {}
        self._recycle_locks: dict[str, asyncio.Lock] = {}
        self._recycle_locks_guard = asyncio.Lock()

    def actor_name_for_model(self, model_id: str) -> str | None:
        return self._model_actor_supervisor_actor_names.get(model_id)

    def actor_name_for_session(self, session: "TrainingSession") -> str | None:
        return self.actor_name_for_model(session.model_id) or str(getattr(session, "actor_name", "") or "") or None

    def bind_session_actor(self, model_id: str, actor_name: str) -> None:
        self._model_actor_supervisor_actor_names[model_id] = actor_name

    def unbind_session_actor(self, model_id: str) -> str | None:
        return self._model_actor_supervisor_actor_names.pop(model_id, None)

    def bound_model_ids_for_actor(self, actor_name: str, *, exclude_model_id: str | None = None) -> list[str]:
        return [
            model_id
            for model_id, bound_actor in self._model_actor_supervisor_actor_names.items()
            if bound_actor == actor_name and model_id != exclude_model_id
        ]

    def raise_if_session_poisoned(self, session: "TrainingSession", *, op: str) -> None:
        hard_error = self._hard_poisoned_sessions.get(session.model_id)
        if hard_error is not None:
            raise RuntimeError(hard_error)
        if op == "load_weights":
            return
        error = self._poisoned_sessions.get(session.model_id)
        if error is not None:
            raise RuntimeError(error)

    def mark_poisoned(self, model_id: str, error: str) -> None:
        self._poisoned_sessions[model_id] = error

    def mark_hard_poisoned(self, model_id: str, error: str) -> None:
        self._poisoned_sessions[model_id] = error
        self._hard_poisoned_sessions[model_id] = error

    def clear_poisoned(self, model_id: str) -> None:
        self._poisoned_sessions.pop(model_id, None)
        self._hard_poisoned_sessions.pop(model_id, None)

    def note_successful_worker_call(self, session: "TrainingSession", *, op: str) -> None:
        actor_name = self.actor_name_for_session(session)
        if not actor_name:
            return
        self._actor_loaded_sessions[actor_name] = session.model_id
        if op in {"forward_backward", "optim_step", "train_step"}:
            self.mark_session_volatile(actor_name, session.model_id)
        if op == "load_weights":
            self.clear_poisoned(session.model_id)

    def mark_session_volatile(self, actor_name: str, model_id: str) -> None:
        self._actor_volatile_sessions.setdefault(actor_name, set()).add(model_id)

    def discard_session_volatile(self, actor_name: str, model_id: str) -> None:
        volatile_sessions = self._actor_volatile_sessions.get(actor_name)
        if not volatile_sessions:
            return
        volatile_sessions.discard(model_id)
        if not volatile_sessions:
            self._actor_volatile_sessions.pop(actor_name, None)

    def volatile_sessions_for_actor(self, actor_name: str | None) -> list[str]:
        if actor_name is None:
            return []
        return sorted(self._actor_volatile_sessions.get(actor_name, set()))

    def clear_actor_runtime_state(self, actor_name: str | None) -> None:
        if actor_name is None:
            return
        self._actor_loaded_sessions.pop(actor_name, None)
        self._actor_volatile_sessions.pop(actor_name, None)

    def clear_session_runtime_state(self, model_id: str, actor_name: str | None) -> None:
        if actor_name is not None and self._actor_loaded_sessions.get(actor_name) == model_id:
            self._actor_loaded_sessions.pop(actor_name, None)
        if actor_name is not None:
            self.discard_session_volatile(actor_name, model_id)
        self._model_actor_supervisor_actor_names.pop(model_id, None)

    async def recycle_lock_for_actor(self, actor_name: str) -> asyncio.Lock:
        async with self._recycle_locks_guard:
            lock = self._recycle_locks.get(actor_name)
            if lock is None:
                lock = asyncio.Lock()
                self._recycle_locks[actor_name] = lock
            return lock

    def snapshot(self) -> dict[str, object]:
        """Return a read-only diagnostics snapshot without exposing mutable maps."""
        return {
            "bound_actor_names": dict(self._model_actor_supervisor_actor_names),
            "actor_loaded_sessions": dict(self._actor_loaded_sessions),
            "actor_volatile_sessions": {
                actor_name: sorted(model_ids)
                for actor_name, model_ids in self._actor_volatile_sessions.items()
            },
            "poisoned_sessions": dict(self._poisoned_sessions),
            "hard_poisoned_sessions": dict(self._hard_poisoned_sessions),
        }
