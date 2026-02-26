"""Training session manager for per-model training state.

Each training model gets its own session with LoRA weights and optimizer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models.types import LoRAConfig
    from .verl_inference import VerlInferenceEngine

logger = logging.getLogger(__name__)


@dataclass
class TrainingSession:
    """A single training session with LoRA fine-tuning state."""

    model_id: str
    session_id: str
    model_seq_id: int
    base_model: str
    user_id: str | None = None
    lora_config: LoRAConfig | None = None
    user_metadata: dict = field(default_factory=dict)
    learning_rate: float = 1e-4

    # Training state
    current_step: int = 0
    total_samples_processed: int = 0
    accumulated_gradients: int = 0
    is_active: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
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
    """

    def __init__(self):
        self._sessions: dict[str, TrainingSession] = {}
        logger.info("TrainingSessionManager initialized")

    def create_session(
        self,
        model_id: str,
        session_id: str,
        model_seq_id: int,
        base_model: str,
        lora_config: LoRAConfig | None = None,
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
        """Get training session by model_id."""
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

    async def shutdown_all(self, engine) -> None:
        """Shutdown all sessions. Called on application exit.

        Args:
            engine: VerlTrainingEngine to shutdown sessions with.
        """
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
