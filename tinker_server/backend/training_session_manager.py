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
        )

        self._sessions[model_id] = session
        logger.info(
            f"Created training session: {model_id} "
            f"(session={session_id}, seq={model_seq_id}, base={base_model})"
        )

        return session

    def get_session(self, model_id: str) -> TrainingSession | None:
        """Get training session by model_id.

        Does NOT update last_activity; call touch_session() explicitly
        in training operation routes to prevent idle cleanup.  Read-only
        lookups (GET /models, GET /training_runs, existence checks) must
        not extend the idle deadline.
        """
        return self._sessions.get(model_id)

    def list_sessions(self) -> list[TrainingSession]:
        """List all training sessions."""
        return list(self._sessions.values())

    def delete_session(self, model_id: str) -> bool:
        """Delete a training session.

        Returns:
            True if deleted, False if not found.
        """
        if model_id not in self._sessions:
            logger.warning(f"Attempted to delete non-existent session: {model_id}")
            return False

        session = self._sessions.pop(model_id)
        logger.info(
            f"Deleted training session: {model_id} "
            f"(step={session.current_step}, samples={session.total_samples_processed})"
        )
        return True

    def get_session_count(self) -> int:
        """Get total number of active sessions."""
        return len(self._sessions)

    def touch_session(self, model_id: str) -> None:
        """Update last_activity timestamp for a session.

        Called from training HTTP handlers when accepting a request (before
        enqueue) to prevent idle cleanup during queue delay.
        """
        session = self._sessions.get(model_id)
        if session is not None:
            session.last_activity = time.time()

    def mark_inflight(self, model_id: str, delta: int) -> None:
        """Mark a session as having in-flight work to prevent cleanup.

        Called from _do_* background workers: +1 at start, -1 in finally.
        Also refreshes last_activity so long-running operations (actor
        creation, checkpoint load) do not trip the idle timeout.
        """
        session = self._sessions.get(model_id)
        if session is not None:
            session.last_activity = time.time()
            session.inflight_ops = max(0, session.inflight_ops + delta)

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
        1. engine.shutdown_session (release GPU actor if applicable)
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

        # 1. Shutdown training worker (best-effort, unconditional —
        #    matches DELETE route which calls shutdown_session regardless
        #    of is_active; shutdown_session is a safe no-op when no
        #    worker mapping exists for this model_id).
        if self._engine is not None:
            try:
                await self._engine.shutdown_session(session)
            except Exception as e:
                logger.error(
                    f"Failed to shutdown training session {model_id} "
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
                # Shutdown training worker
                if session.is_active:
                    try:
                        await engine.shutdown_session(session)
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
