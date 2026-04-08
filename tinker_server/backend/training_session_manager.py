"""Training session manager for per-model training state.

Each training model gets its own session with LoRA weights and optimizer.
Sessions are automatically cleaned up after inactivity.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models.types import LoRAConfig

logger = logging.getLogger(__name__)

# Default training inactivity timeout: 1 hour.
# Training clients may have long intervals between steps (sampling, reward
# computation, etc.), so use a longer timeout than inference (30 min).
DEFAULT_TRAINING_INACTIVITY_TIMEOUT = 3600


@dataclass
class TrainingSession:
    """A single training session with LoRA fine-tuning state."""

    model_id: str
    session_id: str
    model_seq_id: int
    base_model: str
    user_id: str | None = None
    lora_config: LoRAConfig | None = None
    rollout_correction_config: dict[str, Any] | None = None
    user_metadata: dict = field(default_factory=dict)
    learning_rate: float = 1e-4

    # Training state
    current_step: int = 0
    total_samples_processed: int = 0
    accumulated_gradients: int = 0
    is_active: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_activity: float = field(default_factory=time.time)
    inflight_ops: int = 0  # Prevent cleanup while requests are queued or running
    backend: str = "peft"  # "peft" for dense models, "megatron" for MoE
    metadata_version: int = 1  # Monotonic metadata version for cache coherence
    pending_persist: bool = True  # Local create path before detached state is visible

    # Per-session inference engine for isolated concurrent access
    # Lazily initialized on first save_weights_for_sampler call
    inference_engine: Any = None

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "model_id": self.model_id,
            "session_id": self.session_id,
            "model_seq_id": self.model_seq_id,
            "base_model": self.base_model,
            "user_id": self.user_id,
            "lora_config": self.lora_config.model_dump() if self.lora_config else None,
            "rollout_correction_config": self.rollout_correction_config,
            "user_metadata": self.user_metadata,
            "current_step": self.current_step,
            "total_samples_processed": self.total_samples_processed,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "learning_rate": self.learning_rate,
            "backend": self.backend,
        }


@dataclass(frozen=True)
class TrainingSessionSnapshot:
    """Request-scope immutable view of training metadata."""

    model_id: str
    session_id: str
    model_seq_id: int
    base_model: str
    backend: str
    current_step: int
    lora_config: dict[str, Any] | None
    rollout_correction_config: dict[str, Any] | None
    user_metadata: dict[str, Any]
    learning_rate: float
    metadata_version: int


class TrainingSessionManager:
    """Manages multiple training sessions.

    Maps model_id → TrainingSession.
    Includes background cleanup of idle sessions (mirrors SessionManager pattern).
    """

    def __init__(self, inactivity_timeout: float = DEFAULT_TRAINING_INACTIVITY_TIMEOUT):
        self._sessions: dict[str, TrainingSession] = {}
        self._inactivity_timeout = inactivity_timeout
        self._cleanup_task: asyncio.Task | None = None
        self._engine = None  # Set via start_cleanup_task; used by cleanup loop
        self._obs_training_totals: dict[str, int] = {
            "training_sessions_total": 0,
            "training_sessions_active": 0,
            "training_sessions_inflight": 0,
        }
        self._obs_training_by_model: dict[tuple[str, str], dict[str, int | str]] = {}
        self._obs_training_state_by_model_id: dict[str, dict[str, int | str]] = {}
        logger.info(
            "TrainingSessionManager initialized "
            f"(inactivity_timeout={self._inactivity_timeout}s)"
        )

    def create_session(
        self,
        model_id: str,
        session_id: str,
        model_seq_id: int,
        base_model: str,
        lora_config: LoRAConfig | None = None,
        rollout_correction_config: dict[str, Any] | None = None,
        user_metadata: dict | None = None,
        user_id: str | None = None,
        learning_rate: float = 1e-4,
        metadata_version: int | None = None,
    ) -> TrainingSession:
        """Create a new training session.

        Args:
            model_id: Unique model identifier (usually session_id_model_seq_id).
            session_id: Session identifier from tinker client.
            model_seq_id: Model sequence number within session.
            base_model: Base model name.
            lora_config: Optional LoRA configuration.
            rollout_correction_config: Optional session-level rollout correction policy.
            user_metadata: Optional user metadata.
            learning_rate: Learning rate for optimizer.

        Returns:
            TrainingSession instance.

        Raises:
            ValueError: If model_id already exists.
        """
        if model_id in self._sessions:
            raise ValueError(f"Model ID '{model_id}' already exists")

        session = TrainingSession(
            model_id=model_id,
            session_id=session_id,
            model_seq_id=model_seq_id,
            base_model=base_model,
            user_id=user_id,
            lora_config=lora_config,
            rollout_correction_config=rollout_correction_config,
            user_metadata=user_metadata or {},
            learning_rate=learning_rate,
            metadata_version=max(1, int(metadata_version) if metadata_version is not None else 1),
        )

        session.pending_persist = True
        self._sessions[model_id] = session
        logger.info(
            f"Created training session: {model_id} "
            f"(session={session_id}, seq={model_seq_id}, base={base_model})"
        )
        self.refresh_observability_session(model_id)

        return session

    def get_training_session_snapshot(self, model_id: str) -> TrainingSessionSnapshot | None:
        session = self._sessions.get(model_id)
        if session is None:
            return None
        return TrainingSessionSnapshot(
            model_id=session.model_id,
            session_id=session.session_id,
            model_seq_id=int(session.model_seq_id),
            base_model=session.base_model,
            backend=session.backend,
            current_step=int(session.current_step),
            lora_config=session.lora_config.model_dump() if session.lora_config else None,
            rollout_correction_config=session.rollout_correction_config,
            user_metadata=dict(session.user_metadata or {}),
            learning_rate=float(session.learning_rate),
            metadata_version=max(1, int(session.metadata_version)),
        )

    def get_session_metadata_version(self, model_id: str) -> int | None:
        session = self._sessions.get(model_id)
        if session is None:
            return None
        return max(1, int(session.metadata_version))

    def restore_training_session_info(self, info: dict[str, Any]) -> TrainingSession | None:
        model_id = str(info.get("model_id") or "")
        session_id = str(info.get("session_id") or "")
        base_model = str(info.get("base_model") or "")
        if not model_id or not session_id or not base_model:
            return None

        incoming_version = max(1, int(info.get("metadata_version") or 1))
        session = self._sessions.get(model_id)

        lora_cfg = None
        if info.get("lora_config"):
            try:
                from ..models.types import LoRAConfig

                lora_cfg = LoRAConfig(**info["lora_config"])
            except Exception:
                lora_cfg = None

        if session is None:
            session = self.create_session(
                model_id=model_id,
                session_id=session_id,
                model_seq_id=int(info.get("model_seq_id", 0)),
                base_model=base_model,
                lora_config=lora_cfg,
                rollout_correction_config=info.get("rollout_correction_config"),
                user_metadata=info.get("user_metadata") or {},
                user_id=info.get("user_id"),
                learning_rate=float(info.get("learning_rate", 1e-4)),
                metadata_version=incoming_version,
            )
            before = TrainingSession(**vars(session))
            session.backend = str(info.get("backend", session.backend))
            session.pending_persist = False
            try:
                session.current_step = int(info.get("current_step", session.current_step))
            except Exception:
                pass
            self.refresh_observability_session(model_id, before=before)
        elif incoming_version <= max(1, int(session.metadata_version)):
            # Allow monotonic activity/step updates without overwriting newer metadata.
            try:
                session.current_step = max(session.current_step, int(info.get("current_step", session.current_step)))
            except Exception:
                pass
            try:
                raw_last_activity = info.get("last_activity")
                if raw_last_activity is not None:
                    session.last_activity = max(session.last_activity, float(raw_last_activity))
            except Exception:
                pass
            return session
        else:
            before = TrainingSession(**vars(session))
            session.session_id = session_id
            session.model_seq_id = int(info.get("model_seq_id", session.model_seq_id))
            session.base_model = base_model
            session.lora_config = lora_cfg
            session.rollout_correction_config = info.get("rollout_correction_config")
            session.user_metadata = info.get("user_metadata") or {}
            session.user_id = info.get("user_id")
            try:
                session.learning_rate = float(info.get("learning_rate", session.learning_rate))
            except Exception:
                pass
            session.backend = str(info.get("backend", session.backend))
            try:
                session.current_step = max(session.current_step, int(info.get("current_step", session.current_step)))
            except Exception:
                pass
            session.metadata_version = incoming_version
            session.pending_persist = False
            self.refresh_observability_session(model_id, before=before)

        try:
            raw_last_activity = info.get("last_activity")
            if raw_last_activity is not None:
                session.last_activity = float(raw_last_activity)
        except Exception:
            pass
        created_at = info.get("created_at")
        if isinstance(created_at, str) and created_at:
            session.created_at = created_at
        return session

    def mark_persisted(self, model_id: str) -> None:
        session = self._sessions.get(model_id)
        if session is not None:
            session.pending_persist = False

    def get_session(self, model_id: str) -> TrainingSession | None:
        """Get training session by model_id.

        Detached training_session_store is the authoritative state source.
        The local map is only a request-path cache plus create-time scratch state.
        """
        session = self._sessions.get(model_id)
        if session is not None and bool(getattr(session, "pending_persist", False)):
            return session

        try:
            from .training_session_store import get_training_session_info

            info = get_training_session_info(model_id)
        except Exception:
            return session

        if not isinstance(info, dict):
            if session is not None and not bool(getattr(session, "pending_persist", False)):
                self._sessions.pop(model_id, None)
            return session if session is not None and bool(getattr(session, "pending_persist", False)) else None

        restored = self.restore_training_session_info(info)
        if restored is not None:
            restored.pending_persist = False
        return restored

    def list_sessions(self) -> list[TrainingSession]:
        """List training sessions from detached authority plus local pending creates."""
        out: list[TrainingSession] = []
        seen: set[str] = set()
        try:
            from .training_session_store import list_training_sessions

            infos = list_training_sessions()
        except Exception:
            return list(self._sessions.values())

        for info in infos:
            if not isinstance(info, dict):
                continue
            restored = self.restore_training_session_info(info)
            if restored is None:
                continue
            restored.pending_persist = False
            out.append(restored)
            seen.add(restored.model_id)

        for model_id, session in self._sessions.items():
            if model_id in seen:
                continue
            if bool(getattr(session, "pending_persist", False)):
                out.append(session)
        return out

    def delete_session(self, model_id: str) -> bool:
        """Delete a training session.

        Returns:
            True if deleted, False if not found.
        """
        if model_id not in self._sessions:
            logger.warning(f"Attempted to delete non-existent session: {model_id}")
            return False

        session = self._sessions.pop(model_id)
        self.refresh_observability_session(model_id)
        logger.info(
            f"Deleted training session: {model_id} "
            f"(step={session.current_step}, samples={session.total_samples_processed})"
        )
        return True

    def get_session_count(self) -> int:
        """Get total number of active sessions."""
        return len(self._sessions)

    @staticmethod
    def _training_obs_state(session: TrainingSession | None) -> dict[str, int | str] | None:
        if session is None:
            return None
        return {
            "base_model": str(session.base_model or "unknown"),
            "backend": str(session.backend or "unknown"),
            "training_sessions_total": 1,
            "training_sessions_active": 1 if bool(session.is_active) else 0,
            "training_sessions_inflight": 1 if int(session.inflight_ops) > 0 else 0,
            "total": 1,
            "active": 1 if bool(session.is_active) else 0,
            "inflight": 1 if int(session.inflight_ops) > 0 else 0,
        }

    def _apply_training_obs_delta(self, state: dict[str, int | str] | None, sign: int) -> None:
        if state is None:
            return
        for key in (
            "training_sessions_total",
            "training_sessions_active",
            "training_sessions_inflight",
        ):
            self._obs_training_totals[key] = max(0, int(self._obs_training_totals.get(key, 0)) + sign * int(state[key]))

        bucket_key = (str(state["base_model"]), str(state["backend"]))
        bucket = self._obs_training_by_model.get(bucket_key)
        if bucket is None and sign > 0:
            bucket = {
                "base_model": bucket_key[0],
                "backend": bucket_key[1],
                "total": 0,
                "active": 0,
                "inflight": 0,
            }
            self._obs_training_by_model[bucket_key] = bucket
        if bucket is None:
            return
        for key in ("total", "active", "inflight"):
            bucket[key] = max(0, int(bucket.get(key, 0)) + sign * int(state[key]))
        if int(bucket["total"]) == 0 and int(bucket["active"]) == 0 and int(bucket["inflight"]) == 0:
            self._obs_training_by_model.pop(bucket_key, None)

    def refresh_observability_session(self, model_id: str, *, before: TrainingSession | None = None) -> None:
        after = self._sessions.get(model_id)
        before_state = self._training_obs_state(before)
        if before_state is None:
            before_state = self._obs_training_state_by_model_id.get(model_id)
        after_state = self._training_obs_state(after)
        self._apply_training_obs_delta(before_state, -1)
        self._apply_training_obs_delta(after_state, 1)
        if after_state is None:
            self._obs_training_state_by_model_id.pop(model_id, None)
        else:
            self._obs_training_state_by_model_id[model_id] = dict(after_state)

    def observability_snapshot(self) -> dict[str, int | list[dict[str, int | str]]]:
        return {
            **self._obs_training_totals,
            "training_sessions_by_model": [
                dict(self._obs_training_by_model[key]) for key in sorted(self._obs_training_by_model)
            ],
        }

    def _persist_last_activity(self, model_id: str, last_activity: float) -> None:
        try:
            from .training_session_store import set_training_session_last_activity

            set_training_session_last_activity(model_id, last_activity)
        except Exception:
            pass

    def touch_session(self, model_id: str) -> None:
        """Update last_activity timestamp for a session.

        Called from training HTTP handlers when accepting a request (before
        enqueue) to prevent idle cleanup during queue delay.
        """
        session = self._sessions.get(model_id)
        if session is not None:
            session.last_activity = time.time()
            self._persist_last_activity(model_id, session.last_activity)

    def mark_inflight(self, model_id: str, delta: int) -> None:
        """Mark a session as having in-flight work to prevent cleanup.

        Called from _do_* background workers: +1 at start, -1 in finally.
        Also refreshes last_activity so long-running operations (actor
        creation, checkpoint load) do not trip the idle timeout.
        """
        session = self._sessions.get(model_id)
        if session is not None:
            before = TrainingSession(**vars(session))
            session.last_activity = time.time()
            session.inflight_ops = max(0, session.inflight_ops + delta)
            self._persist_last_activity(model_id, session.last_activity)
            self.refresh_observability_session(model_id, before=before)

    # =========================================================================
    # Background cleanup of idle training sessions
    # =========================================================================

    async def start_cleanup_task(self, engine) -> None:
        """Start the background cleanup task.

        Args:
            engine: VerlTrainingEngine used to shutdown idle sessions.
        """
        self._engine = engine
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info(
                f"Started training session cleanup task "
                f"(timeout={self._inactivity_timeout}s)"
            )

    async def _cleanup_loop(self) -> None:
        """Periodically check for and cleanup inactive training sessions."""
        while True:
            await asyncio.sleep(60)  # Check every minute
            try:
                await self._cleanup_inactive()
            except Exception as e:
                logger.error(f"Training session cleanup error: {e}")

    async def _cleanup_inactive(self) -> None:
        """Cleanup training sessions inactive for longer than timeout."""
        for model_id, session in list(self._sessions.items()):
            try:
                from .training_session_store import async_get_training_session_info

                detached = await async_get_training_session_info(model_id)
            except Exception:
                detached = None
            if isinstance(detached, dict):
                try:
                    session.last_activity = max(
                        float(session.last_activity),
                        float(detached.get("last_activity", session.last_activity)),
                    )
                except Exception:
                    pass

        now = time.time()
        inactive = [
            model_id
            for model_id, session in self._sessions.items()
            if session.inflight_ops == 0
            if now - session.last_activity > self._inactivity_timeout
        ]
        for model_id in inactive:
            session = self._sessions.get(model_id)
            if session is None:
                continue  # Concurrently deleted (e.g. explicit DELETE)
            idle_s = now - session.last_activity
            logger.info(
                f"Auto-cleaning inactive training session {model_id} "
                f"(idle {idle_s:.0f}s > {self._inactivity_timeout}s)"
            )
            await self._cleanup_session(model_id)

    async def _cleanup_session(self, model_id: str) -> None:
        """Full cleanup of a single training session.

        Mirrors the DELETE /models/{model_id} flow:
        1. engine.delete_session (delete actor-local state, then unbind/kill actor if applicable)
        2. delete_session (remove from in-memory manager)
        3. delete_training_session (remove from detached Ray store)
        4. resource_pool.clear_session (clear stale session pins)
        """
        session = self._sessions.get(model_id)
        if session is None:
            return

        # Re-check: session may have acquired in-flight work or been touched
        # between candidate selection in _cleanup_inactive and this point.
        if session.inflight_ops > 0:
            return
        if time.time() - session.last_activity <= self._inactivity_timeout:
            return

        # 1. Delete actor-local session state, then unbind the live engine
        #    session. This mirrors explicit DELETE semantics.
        if self._engine is not None:
            try:
                await self._engine.delete_session(session)
            except Exception as e:
                logger.error(
                    f"Failed to delete training session {model_id} "
                    f"during idle cleanup: {e}"
                )

        # 2. Shutdown per-session inference engine if present
        if session.inference_engine is not None:
            try:
                await session.inference_engine.shutdown()
                logger.info(
                    f"Shutdown inference engine for idle session: {model_id}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to shutdown inference engine for {model_id}: {e}"
                )

        # 3. Remove from in-memory manager
        self.delete_session(model_id)

        # 4. Remove from detached Ray store (best-effort)
        try:
            from .training_session_store import delete_training_session

            delete_training_session(model_id)
        except Exception as e:
            logger.warning(
                f"Failed to delete training session {model_id} from store: {e}"
            )

        # 5. Clear ResourcePool session tracking (best-effort)
        try:
            from .resource_pool import get_resource_pool

            get_resource_pool().clear_session(model_id)
        except Exception:
            pass

    async def shutdown_all(self, engine) -> None:
        """Shutdown all sessions. Called on application exit.

        Args:
            engine: VerlTrainingEngine to shutdown sessions with.
        """
        # Stop cleanup task
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        session_ids = list(self._sessions.keys())
        for model_id in session_ids:
            session = self._sessions.get(model_id)
            if session:
                # Application shutdown should only release live actors; it
                # must not delete persisted session state.
                if session.is_active:
                    try:
                        await engine.unbind_session(session)
                        logger.info(f"Shutdown training session: {model_id}")
                    except Exception as e:
                        logger.error(f"Failed to shutdown session {model_id}: {e}")
                # Shutdown per-session inference engine if present
                if session.inference_engine is not None:
                    try:
                        await session.inference_engine.shutdown()
                        logger.info(f"Shutdown inference engine for session: {model_id}")
                    except Exception as e:
                        logger.error(f"Failed to shutdown inference engine {model_id}: {e}")
            self._sessions.pop(model_id, None)
        logger.info(f"Shutdown {len(session_ids)} training sessions")
