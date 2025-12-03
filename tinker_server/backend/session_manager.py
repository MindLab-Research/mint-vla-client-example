"""Session manager for per-session VerlInferenceEngine instances.

Each sampling session gets its own engine with dedicated LoRA weights.
Sessions are automatically cleaned up after inactivity.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .verl_inference import VerlInferenceEngine

logger = logging.getLogger(__name__)

# Default inactivity timeout: 5 minutes
DEFAULT_INACTIVITY_TIMEOUT = 300


@dataclass
class SessionInfo:
    """Tracks session state."""

    engine: VerlInferenceEngine
    last_activity: float  # time.time()
    lora_rank: int


class SessionManager:
    """Manages per-session VerlInferenceEngine instances.

    Each session has its own engine with dedicated LoRA adapter,
    enabling session isolation for different LoRA variants.
    Sessions are automatically cleaned up after inactivity.
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        inactivity_timeout: float = DEFAULT_INACTIVITY_TIMEOUT,
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.inactivity_timeout = inactivity_timeout
        self._sessions: dict[str, SessionInfo] = {}
        self._cleanup_task: asyncio.Task | None = None

    async def start_cleanup_task(self) -> None:
        """Start the background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info(
                f"Started session cleanup task (timeout={self.inactivity_timeout}s)"
            )

    async def _cleanup_loop(self) -> None:
        """Periodically check for and cleanup inactive sessions."""
        while True:
            await asyncio.sleep(60)  # Check every minute
            await self._cleanup_inactive()

    async def _cleanup_inactive(self) -> None:
        """Cleanup sessions inactive for longer than timeout."""
        now = time.time()
        inactive = [
            sid
            for sid, info in self._sessions.items()
            if now - info.last_activity > self.inactivity_timeout
        ]
        for sid in inactive:
            logger.info(f"Auto-cleaning inactive session {sid}")
            await self.end_session(sid)

    async def create_session(
        self, session_id: str, lora_rank: int = 32
    ) -> VerlInferenceEngine:
        """Create a new session with dedicated engine.

        Args:
            session_id: Unique identifier for the session.
            lora_rank: LoRA rank for the adapter (0 = no LoRA).

        Returns:
            The initialized VerlInferenceEngine for this session.

        Raises:
            ValueError: If session_id already exists.
        """
        if session_id in self._sessions:
            raise ValueError(f"Session {session_id} already exists")

        from .verl_inference import VerlInferenceEngine

        engine = VerlInferenceEngine(
            model_path=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            lora_rank=lora_rank,
        )
        await engine.initialize()

        self._sessions[session_id] = SessionInfo(
            engine=engine,
            last_activity=time.time(),
            lora_rank=lora_rank,
        )
        logger.info(f"Created session {session_id} with lora_rank={lora_rank}")
        return engine

    def get_engine(self, session_id: str) -> VerlInferenceEngine | None:
        """Get the engine for a session and update activity timestamp.

        Args:
            session_id: The session identifier.

        Returns:
            The engine if session exists, None otherwise.
        """
        info = self._sessions.get(session_id)
        if info is None:
            return None
        info.last_activity = time.time()
        return info.engine

    async def end_session(self, session_id: str) -> bool:
        """End a session and shutdown its engine.

        Args:
            session_id: The session identifier.

        Returns:
            True if session was ended, False if not found.
        """
        info = self._sessions.pop(session_id, None)
        if info is None:
            return False

        await info.engine.shutdown()
        logger.info(f"Ended session {session_id}")
        return True

    async def shutdown_all(self) -> None:
        """Shutdown all sessions and cleanup task. Called on application exit."""
        # Stop cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        # Shutdown all sessions
        session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            await self.end_session(session_id)
        logger.info(f"Shutdown {len(session_ids)} sessions")

    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())


# Global session manager (initialized in app lifespan)
session_manager: SessionManager | None = None
