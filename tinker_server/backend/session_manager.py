"""Session manager for per-session VerlInferenceEngine instances.

Each sampling session gets its own engine with dedicated LoRA weights.
Sessions are automatically cleaned up after inactivity.

Supports two modes:
1. Per-session engine: Traditional mode, spawns new engine per session (slow init)
2. Shared engine: Single engine for ephemeral weight syncs, uses hot LoRA reload (fast)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .multi_lora_engine import MultiLoRAInferenceEngine, MultiModelInferenceManager
    from .verl_inference import VerlInferenceEngine

logger = logging.getLogger(__name__)

# Default inactivity timeout: 5 minutes
DEFAULT_INACTIVITY_TIMEOUT = 300

# Reserved session ID for shared engine
SHARED_ENGINE_SESSION_ID = "__shared__"


@dataclass
class SessionInfo:
    """Tracks session state."""

    engine: VerlInferenceEngine | None  # None if using multi-LoRA mode
    last_activity: float  # time.time()
    lora_rank: int
    is_shared: bool = False  # True for sessions using the shared engine
    uses_multi_lora: bool = False  # True if using MultiLoRAInferenceEngine
    uses_base_model: bool = False  # True if multi-LoRA without any LoRA adapter
    base_model: str | None = None  # Base model name for multi-model support
    adapter_path: str | None = None  # Optional (ephemeral) adapter directory to cleanup
    inflight_requests: int = 0  # Prevent cleanup while requests are running


class SessionManager:
    """Manages per-session VerlInferenceEngine instances.

    Each session has its own engine with dedicated LoRA adapter,
    enabling session isolation for different LoRA variants.
    Sessions are automatically cleaned up after inactivity.

    For ephemeral weight syncs during training, uses a shared engine
    with hot LoRA reload for 100x+ faster weight updates.
    """

    def __init__(
        self,
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int | None = None,
        inactivity_timeout: float = DEFAULT_INACTIVITY_TIMEOUT,
        shared_engine_lora_rank: int = 32,
    ):
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = data_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.inactivity_timeout = inactivity_timeout
        self.shared_engine_lora_rank = shared_engine_lora_rank
        self._sessions: dict[str, SessionInfo] = {}
        self._cleanup_task: asyncio.Task | None = None
        self._shared_engine: VerlInferenceEngine | None = None
        self._shared_engine_lock = asyncio.Lock()

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
            if info.inflight_requests == 0
            if now - info.last_activity > self.inactivity_timeout
            # Keep shared engine sessions unless they are multi-LoRA registrations
            # (multi-LoRA uses per-session LoRA IDs and must be GC'd to avoid leaks).
            and (not info.is_shared or info.uses_multi_lora)
        ]
        for sid in inactive:
            logger.info(f"Auto-cleaning inactive session {sid}")
            await self.end_session(sid)

    def mark_session_inflight(self, session_id: str, delta: int) -> None:
        """Mark a session as having in-flight work to prevent cleanup.

        This is required for long-context requests that can exceed the inactivity
        timeout while still actively running.
        """
        info = self._sessions.get(session_id)
        if info is None:
            return
        info.last_activity = time.time()
        info.inflight_requests = max(0, info.inflight_requests + delta)

    # =========================================================================
    # Shared Engine Methods (for fast ephemeral weight sync)
    # =========================================================================

    async def _get_or_create_shared_engine(self) -> "VerlInferenceEngine":
        """Lazily initialize the shared engine for ephemeral sessions.

        Uses a lock to ensure only one engine is created even with concurrent calls.
        The shared engine is initialized with LoRA enabled but no adapter loaded.

        Returns:
            The shared VerlInferenceEngine instance.
        """
        async with self._shared_engine_lock:
            if self._shared_engine is None:
                from .verl_inference import VerlInferenceEngine

                logger.info(
                    f"Initializing shared engine (lora_rank={self.shared_engine_lora_rank})"
                )
                self._shared_engine = VerlInferenceEngine(
                    model_path=self.model_path,
                    tensor_parallel_size=self.tensor_parallel_size,
                    data_parallel_size=self.data_parallel_size,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    max_model_len=self.max_model_len,
                    lora_rank=self.shared_engine_lora_rank,
                    lora_adapter_path=None,  # No initial adapter
                )
                await self._shared_engine.initialize()
                logger.info("Shared engine initialized")

            return self._shared_engine

    async def create_ephemeral_session(
        self,
        session_id: str,
        adapter_path: str,
        lora_rank: int = 32,
    ) -> "VerlInferenceEngine":
        """Create an ephemeral session using the shared engine with hot LoRA reload.

        Instead of spawning a new vLLM engine (30-60s), hot-loads the LoRA adapter
        into the existing shared engine (100-300ms).

        Args:
            session_id: Unique identifier for the session.
            adapter_path: Filesystem path to LoRA adapter directory.
            lora_rank: LoRA rank (for validation, must match shared engine).

        Returns:
            The shared VerlInferenceEngine with the adapter loaded.

        Raises:
            ValueError: If session_id already exists.
            RuntimeError: If LoRA rank doesn't match shared engine.
        """
        if session_id in self._sessions:
            raise ValueError(f"Session {session_id} already exists")

        if lora_rank != self.shared_engine_lora_rank:
            raise RuntimeError(
                f"LoRA rank mismatch: session requests {lora_rank}, "
                f"shared engine has {self.shared_engine_lora_rank}"
            )

        # Get or create the shared engine
        engine = await self._get_or_create_shared_engine()

        # Hot-reload LoRA adapter (100-300ms vs 30-60s for new engine)
        start_time = time.time()
        await engine.load_lora_from_path(adapter_path)
        reload_time = time.time() - start_time
        logger.info(f"Hot-reloaded LoRA adapter in {reload_time:.3f}s")

        # Register session pointing to shared engine
        self._sessions[session_id] = SessionInfo(
            engine=engine,
            last_activity=time.time(),
            lora_rank=lora_rank,
            is_shared=True,
        )
        logger.info(
            f"Created ephemeral session {session_id} using shared engine "
            f"(reload took {reload_time:.3f}s)"
        )
        return engine

    async def create_session(
        self,
        session_id: str,
        lora_rank: int = 32,
        model_path: str | None = None,
    ) -> VerlInferenceEngine:
        """Create a new session with dedicated engine.

        Loads LoRA adapter at engine initialization time using vLLM's native
        file-based LoRA loading. The adapter must be on a shared filesystem
        accessible to both training and inference nodes.

        Args:
            session_id: Unique identifier for the session.
            lora_rank: LoRA rank for the adapter (0 = no LoRA).
            model_path: Optional file:// URI or path to pre-trained LoRA adapter.

        Returns:
            The initialized VerlInferenceEngine for this session.

        Raises:
            ValueError: If session_id already exists.
        """
        if session_id in self._sessions:
            raise ValueError(f"Session {session_id} already exists")

        from .verl_inference import VerlInferenceEngine

        # Resolve adapter path if provided
        adapter_path = None
        if model_path:
            adapter_path = self._resolve_model_path(model_path)

        # Initialize engine WITH adapter path - vLLM loads LoRA at init
        engine = VerlInferenceEngine(
            model_path=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            data_parallel_size=self.data_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            lora_rank=lora_rank,
            lora_adapter_path=adapter_path,  # Load at init time
        )
        await engine.initialize()

        self._sessions[session_id] = SessionInfo(
            engine=engine,
            last_activity=time.time(),
            lora_rank=lora_rank,
        )
        logger.info(
            f"Created session {session_id} with lora_rank={lora_rank}, "
            f"adapter_path={adapter_path}"
        )
        return engine

    def _resolve_model_path(self, model_path: str) -> str:
        """Resolve model_path URI to filesystem path.

        Args:
            model_path: URI like file:///path, mint://{uuid}/..., or absolute path.

        Returns:
            Absolute filesystem path to adapter directory.
        """
        from ..checkpoints import get_checkpoints_dir

        checkpoint_dir = get_checkpoints_dir()

        if model_path.startswith("file://"):
            return model_path[7:]  # Strip file:// prefix
        elif model_path.startswith("tinker://"):
            # tinker://{model_id}/{checkpoint_name}
            path_part = model_path[len("tinker://"):]
            return os.path.join(checkpoint_dir, path_part)
        elif model_path.startswith("mint://"):
            # Legacy mint://{model_id}/{checkpoint_name}
            path_part = model_path[len("mint://"):]
            return os.path.join(checkpoint_dir, path_part)
        else:
            # Assume absolute path
            return model_path

    def create_session_with_engine(
        self,
        session_id: str,
        engine: "VerlInferenceEngine",
        lora_rank: int = 32,
    ) -> None:
        """Register a session with an already-initialized engine.

        Used for per-session inference engines created by training sessions.
        The engine is already initialized and owns its own lifecycle.

        Args:
            session_id: Unique identifier for the session.
            engine: An already-initialized VerlInferenceEngine.
            lora_rank: LoRA rank for the adapter.

        Raises:
            ValueError: If session_id already exists.
        """
        if session_id in self._sessions:
            raise ValueError(f"Session {session_id} already exists")

        self._sessions[session_id] = SessionInfo(
            engine=engine,
            last_activity=time.time(),
            lora_rank=lora_rank,
            is_shared=True,  # Mark as shared to prevent SessionManager from shutting it down
        )
        logger.info(
            f"Registered session {session_id} with external engine "
            f"(lora_rank={lora_rank})"
        )

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

        For shared engine sessions, only removes the session registration
        without shutting down the engine (it's reused for other sessions).

        Args:
            session_id: The session identifier.

        Returns:
            True if session was ended, False if not found.
        """
        info = self._sessions.pop(session_id, None)
        if info is None:
            return False

        if info.uses_multi_lora:
            # Best-effort: remove LoRA from vLLM and delete ephemeral adapter dir.
            manager = self.get_multi_model_manager()
            if manager is not None and info.base_model:
                engine = manager.get_engine_if_exists(info.base_model)
                if engine is not None:
                    try:
                        await engine.remove_session(session_id)
                    except Exception as e:
                        logger.warning(f"Failed to remove multi-LoRA session {session_id} from engine: {e}")

            if info.adapter_path:
                import os
                import shutil

                adapter_path = info.adapter_path
                try:
                    if os.path.isdir(adapter_path) and os.path.basename(adapter_path).startswith("_ephemeral_"):
                        await asyncio.to_thread(shutil.rmtree, adapter_path)
                        logger.info(f"Deleted ephemeral adapter dir for session {session_id}: {adapter_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete adapter dir for session {session_id}: {adapter_path}: {e}")

            logger.info(f"Ended multi-LoRA session {session_id}")
            return True

        # Only shutdown non-shared engines (shared engine is reused)
        if not info.is_shared:
            await info.engine.shutdown()
            logger.info(f"Ended session {session_id} (engine shutdown)")
        else:
            logger.info(f"Ended ephemeral session {session_id} (shared engine kept)")

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

        # Shutdown shared engine
        if self._shared_engine is not None:
            await self._shared_engine.shutdown()
            self._shared_engine = None
            logger.info("Shutdown shared engine")

    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())

    # =========================================================================
    # Multi-LoRA Mode Methods (Multi-Model Support)
    # =========================================================================

    def set_multi_model_manager(self, manager: "MultiModelInferenceManager") -> None:
        """Set the multi-model inference manager.

        Args:
            manager: The MultiModelInferenceManager instance.
        """
        self._multi_model_manager = manager
        logger.info("Multi-model inference manager set")

    def get_multi_model_manager(self) -> "MultiModelInferenceManager | None":
        """Get the multi-model inference manager."""
        return getattr(self, "_multi_model_manager", None)

    def get_multi_lora_engine(self) -> "MultiLoRAInferenceEngine | None":
        """Get multi-LoRA engine for backward compatibility.

        DEPRECATED: Use get_engine_for_session() instead.
        Returns the first engine from multi-model manager, or None.
        """
        manager = self.get_multi_model_manager()
        if manager is None:
            return getattr(self, "_multi_lora_engine", None)
        # Return first available engine for backward compatibility
        models = manager.list_models()
        if models:
            return manager.get_engine_if_exists(models[0])
        return None

    async def get_engine_for_model(self, model_name: str) -> "MultiLoRAInferenceEngine":
        """Get or create vLLM engine for a specific model.

        Args:
            model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-7B-Instruct")

        Returns:
            MultiLoRAInferenceEngine for the model.
        """
        manager = await self.ensure_multi_model_manager()
        return await manager.get_engine(model_name)

    def get_session_base_model(self, session_id: str) -> str | None:
        """Get the base model for a session.

        Args:
            session_id: The session identifier.

        Returns:
            Base model name, or None if session not found.
        """
        info = self._sessions.get(session_id)
        if info is not None:
            info.last_activity = time.time()
        return info.base_model if info else None

    async def get_engine_for_session(self, session_id: str) -> "MultiLoRAInferenceEngine | None":
        """Get vLLM engine for a session's model.

        Args:
            session_id: The session identifier.

        Returns:
            MultiLoRAInferenceEngine for the session's model, or None if not found.
        """
        base_model = self.get_session_base_model(session_id)
        if base_model is None:
            return None
        return await self.get_engine_for_model(base_model)

    def register_multi_lora_session(
        self,
        session_id: str,
        base_model: str,
        lora_rank: int = 32,
        adapter_path: str | None = None,
    ) -> None:
        """Register a sampling session that uses the shared multi-LoRA engine.

        The session's LoRA weights are already loaded in the multi-LoRA engine.

        Args:
            session_id: Unique identifier for the sampling session.
            base_model: Base model name for this session.
            lora_rank: LoRA rank for the adapter.
            adapter_path: Optional adapter directory path to delete when the
                sampling session expires (ephemeral save_weights_for_sampler).

        Raises:
            ValueError: If session_id already exists.
        """
        if session_id in self._sessions:
            raise ValueError(f"Session {session_id} already exists")

        self._sessions[session_id] = SessionInfo(
            engine=None,  # No per-session engine
            last_activity=time.time(),
            lora_rank=lora_rank,
            is_shared=True,
            uses_multi_lora=True,
            base_model=base_model,
            adapter_path=adapter_path,
        )
        logger.info(
            f"Registered multi-LoRA session {session_id} (model={base_model}, lora_rank={lora_rank})"
        )

    def is_multi_lora_session(self, session_id: str) -> bool:
        """Check if a session uses multi-LoRA mode.

        Args:
            session_id: The session identifier.

        Returns:
            True if session uses multi-LoRA, False otherwise.
        """
        info = self._sessions.get(session_id)
        return info is not None and info.uses_multi_lora

    def is_base_model_session(self, session_id: str) -> bool:
        """Check if a session uses base model (no LoRA) on multi-LoRA engine.

        Args:
            session_id: The session identifier.

        Returns:
            True if session uses base model, False otherwise.
        """
        info = self._sessions.get(session_id)
        return info is not None and info.uses_base_model

    def register_base_model_session(self, session_id: str, base_model: str) -> None:
        """Register a sampling session that uses base model on multi-LoRA engine.

        The session will use the shared multi-LoRA engine without any LoRA adapter.

        Args:
            session_id: Unique identifier for the sampling session.
            base_model: Base model name for this session.

        Raises:
            ValueError: If session_id already exists.
        """
        if session_id in self._sessions:
            raise ValueError(f"Session {session_id} already exists")

        self._sessions[session_id] = SessionInfo(
            engine=None,  # No per-session engine
            last_activity=time.time(),
            lora_rank=0,  # No LoRA
            is_shared=True,
            uses_multi_lora=True,
            uses_base_model=True,
            base_model=base_model,
        )
        logger.info(f"Registered base model session {session_id} (model={base_model})")

    async def ensure_multi_model_manager(self) -> "MultiModelInferenceManager":
        """Initialize multi-model manager if not already done.

        Lazily creates the manager. Engines are created on-demand per model.

        Returns:
            The MultiModelInferenceManager instance.
        """
        if not hasattr(self, "_multi_model_manager") or self._multi_model_manager is None:
            from .multi_lora_engine import MultiModelInferenceManager

            logger.info("Initializing multi-model inference manager...")
            self._multi_model_manager = MultiModelInferenceManager(
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
            )
            logger.info("Multi-model inference manager initialized")

        return self._multi_model_manager

    async def ensure_multi_lora_engine(self) -> "MultiLoRAInferenceEngine":
        """Initialize multi-LoRA engine if not already done.

        DEPRECATED: Use ensure_multi_model_manager() and get_engine_for_model() instead.
        For backward compatibility, uses self.model_path.

        Returns:
            The initialized MultiLoRAInferenceEngine instance.
        """
        # For backward compatibility, get engine for the configured model_path
        # This is deprecated - callers should use get_engine_for_model() directly
        manager = await self.ensure_multi_model_manager()

        # Extract model name from path if needed
        model_name = self.model_path
        if "/" in model_name and "models--" in model_name:
            # Extract from HuggingFace cache path like /path/models--Qwen--Qwen2.5-7B/...
            parts = model_name.split("models--")
            if len(parts) > 1:
                model_part = parts[1].split("/")[0]  # "Qwen--Qwen2.5-7B"
                model_name = model_part.replace("--", "/")  # "Qwen/Qwen2.5-7B"

        return await manager.get_engine(model_name)


# Global session manager (initialized in app lifespan)
session_manager: SessionManager | None = None
