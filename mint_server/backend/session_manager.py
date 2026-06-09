"""Session manager for per-session VerlInferenceEngine instances.

Each sampling session gets its own engine with dedicated LoRA weights.
Detached maintenance owns periodic cleanup; this manager only exposes explicit
session operations for the runtime that instantiated it.

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

# Default inactivity timeout: 30 minutes
DEFAULT_INACTIVITY_TIMEOUT = 1800

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
    lora_loaded: bool = True  # For multi-LoRA sessions: whether LoRA is loaded into vLLM
    lora_int_id: int | None = None  # Persisted multi-LoRA id for restart recovery
    metadata_version: int = 1  # Monotonic metadata version for worker-local cache coherence
    inflight_requests: int = 0  # Prevent cleanup while requests are running
    pending_persist: bool = False  # Local create path before detached state is visible


@dataclass(frozen=True)
class SamplingSessionSnapshot:
    """Request-scope immutable view of sampling metadata."""

    session_id: str
    uses_multi_lora: bool
    uses_base_model: bool
    base_model: str | None
    lora_rank: int
    adapter_path: str | None
    lora_loaded: bool
    lora_int_id: int | None
    metadata_version: int


class SessionManager:
    """Manages per-session VerlInferenceEngine instances.

    Each session has its own engine with dedicated LoRA adapter,
    enabling session isolation for different LoRA variants.

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
        self._shared_engine: VerlInferenceEngine | None = None
        self._shared_engine_lock = asyncio.Lock()
        self._obs_sampling_totals: dict[str, int] = {
            "sampling_sessions_total": 0,
            "sampling_sessions_multi_lora": 0,
            "sampling_sessions_base_model": 0,
            "sampling_sessions_lora_loaded": 0,
            "sampling_sessions_inflight": 0,
        }
        self._obs_sampling_by_model: dict[str, dict[str, int | str]] = {}

    def _persist_sampling_session_info(self, session_id: str, info: SessionInfo) -> None:
        if not info.uses_multi_lora:
            return
        try:
            from .sampling_session_store import upsert_sampling_session

            upsert_sampling_session(
                {
                    "session_id": session_id,
                    "base_model": info.base_model,
                    "lora_rank": int(info.lora_rank),
                    "adapter_path": info.adapter_path,
                    "lora_loaded": bool(info.lora_loaded),
                    "lora_int_id": info.lora_int_id,
                    "uses_base_model": bool(info.uses_base_model),
                    "last_activity": float(info.last_activity),
                    "inflight_requests": int(info.inflight_requests),
                    "metadata_version": int(info.metadata_version),
                }
            )
        except Exception as e:
            logger.debug("Failed to persist sampling session %s: %s", session_id, e)

    def _touch_info(self, session_id: str, info: SessionInfo) -> None:
        info.last_activity = time.time()
        self._persist_sampling_session_info(session_id, info)

    @staticmethod
    def _sampling_obs_state(info: SessionInfo | None) -> dict[str, int | str] | None:
        if info is None:
            return None
        return {
            "base_model": str(info.base_model or "unknown"),
            "sampling_sessions_total": 1,
            "sampling_sessions_multi_lora": 1 if bool(info.uses_multi_lora) else 0,
            "sampling_sessions_base_model": 1 if bool(info.uses_base_model) else 0,
            "sampling_sessions_lora_loaded": 1 if bool(info.lora_loaded) and not bool(info.uses_base_model) else 0,
            "sampling_sessions_inflight": 1 if int(info.inflight_requests) > 0 else 0,
            "total": 1,
            "inflight": 1 if int(info.inflight_requests) > 0 else 0,
            "lora_loaded": 1 if bool(info.lora_loaded) and not bool(info.uses_base_model) else 0,
        }

    def _apply_sampling_obs_delta(self, state: dict[str, int | str] | None, sign: int) -> None:
        if state is None:
            return
        for key in (
            "sampling_sessions_total",
            "sampling_sessions_multi_lora",
            "sampling_sessions_base_model",
            "sampling_sessions_lora_loaded",
            "sampling_sessions_inflight",
        ):
            self._obs_sampling_totals[key] = max(0, int(self._obs_sampling_totals.get(key, 0)) + sign * int(state[key]))

        base_model = str(state["base_model"])
        bucket = self._obs_sampling_by_model.get(base_model)
        if bucket is None and sign > 0:
            bucket = {
                "base_model": base_model,
                "total": 0,
                "inflight": 0,
                "lora_loaded": 0,
            }
            self._obs_sampling_by_model[base_model] = bucket
        if bucket is None:
            return
        for key in ("total", "inflight", "lora_loaded"):
            bucket[key] = max(0, int(bucket.get(key, 0)) + sign * int(state[key]))
        if int(bucket["total"]) == 0 and int(bucket["inflight"]) == 0 and int(bucket["lora_loaded"]) == 0:
            self._obs_sampling_by_model.pop(base_model, None)

    def _refresh_sampling_observability(
        self,
        *,
        before: SessionInfo | None,
        after: SessionInfo | None,
    ) -> None:
        self._apply_sampling_obs_delta(self._sampling_obs_state(before), -1)
        self._apply_sampling_obs_delta(self._sampling_obs_state(after), 1)

    async def _cleanup_inactive(self) -> None:
        """Cleanup inactive sessions when invoked by a detached authority."""
        now = time.time()
        for sid, info in list(self._sessions.items()):
            if not info.uses_base_model and (not info.is_shared or info.uses_multi_lora):
                try:
                    from .sampling_session_store import async_get_sampling_session_info

                    persisted = await async_get_sampling_session_info(sid)
                except Exception:
                    persisted = None
                if isinstance(persisted, dict):
                    try:
                        info.last_activity = max(
                            float(info.last_activity),
                            float(persisted.get("last_activity", info.last_activity)),
                        )
                    except Exception:
                        pass

        now = time.time()
        inactive = [
            sid
            for sid, info in self._sessions.items()
            if info.inflight_requests == 0
            if now - info.last_activity > self.inactivity_timeout
            # Base-model sessions are purely logical routing entries (no per-session engine or
            # per-session adapter resources). Evicting them breaks clients that cache the
            # sampling_session_id across idle gaps, while providing no resource relief.
            if not info.uses_base_model
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
        info = self._get_session_info(session_id, touch=False)
        if info is None:
            return
        before = SessionInfo(**vars(info))
        info.last_activity = time.time()
        info.inflight_requests = max(0, info.inflight_requests + delta)
        self._persist_sampling_session_info(session_id, info)
        self._refresh_sampling_observability(before=before, after=info)

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
            pending_persist=True,
        )
        self._refresh_sampling_observability(before=None, after=self._sessions[session_id])
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
            pending_persist=True,
        )
        self._refresh_sampling_observability(before=None, after=self._sessions[session_id])
        logger.info(
            f"Created session {session_id} with lora_rank={lora_rank}, "
            f"adapter_path={adapter_path}"
        )
        return engine

    def _resolve_model_path(self, model_path: str) -> str:
        """Resolve model_path URI to filesystem path.

        Args:
            model_path: URI like file:///path, mint://{run_id}/{kind}/{name}, or absolute path.

        Returns:
            Absolute filesystem path to adapter directory.
        """
        if model_path.startswith(("mint://", "ckpt_")):
            raise ValueError("Checkpoint URIs must be resolved before SessionManager.create_session")
        if model_path.startswith("file://"):
            return model_path[7:]
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
            pending_persist=True,
        )
        self._refresh_sampling_observability(before=None, after=self._sessions[session_id])
        logger.info(
            f"Registered session {session_id} with external engine "
            f"(lora_rank={lora_rank})"
        )

    def _restore_from_detached_store(self, session_id: str) -> SessionInfo | None:
        try:
            from .sampling_session_store import get_sampling_session_info

            info = get_sampling_session_info(session_id)
        except Exception:
            return self._sessions.get(session_id)

        if not isinstance(info, dict):
            return self._sessions.get(session_id)
        if not self.restore_sampling_session(info):
            return self._sessions.get(session_id)
        restored = self._sessions.get(session_id)
        if restored is not None:
            restored.pending_persist = False
        return restored

    def _get_session_info(self, session_id: str, *, touch: bool = True) -> SessionInfo | None:
        info = self._sessions.get(session_id)
        if info is None:
            info = self._restore_from_detached_store(session_id)
        elif not bool(getattr(info, "pending_persist", False)) and info.uses_multi_lora:
            refreshed = self._restore_from_detached_store(session_id)
            if refreshed is not None:
                info = refreshed
        if info is not None and touch:
            self._touch_info(session_id, info)
        return info

    def get_engine(self, session_id: str) -> VerlInferenceEngine | None:
        """Get the engine for a session and update activity timestamp."""
        info = self._get_session_info(session_id)
        if info is None:
            return None
        return info.engine

    def get_sampling_session_snapshot(self, session_id: str) -> SamplingSessionSnapshot | None:
        """Build a request-scope immutable snapshot for sampling session metadata."""
        info = self._get_session_info(session_id)
        if info is None:
            return None
        return SamplingSessionSnapshot(
            session_id=session_id,
            uses_multi_lora=bool(info.uses_multi_lora),
            uses_base_model=bool(info.uses_base_model),
            base_model=info.base_model,
            lora_rank=int(info.lora_rank),
            adapter_path=info.adapter_path,
            lora_loaded=bool(info.lora_loaded),
            lora_int_id=None if info.lora_int_id is None else int(info.lora_int_id),
            metadata_version=max(1, int(info.metadata_version)),
        )

    def _cleanup_sampler_indices(self, sampler_id: str) -> None:
        try:
            from .session_index_store import (
                delete_sampler_index,
                get_sampler_index,
                remove_sampler_from_session,
            )

            sampler_info = get_sampler_index(sampler_id)
            parent_session_id = None
            if isinstance(sampler_info, dict):
                raw_session_id = sampler_info.get("session_id")
                if isinstance(raw_session_id, str) and raw_session_id:
                    parent_session_id = raw_session_id

            delete_sampler_index(sampler_id)
            if parent_session_id is not None:
                remove_sampler_from_session(parent_session_id, sampler_id)
        except Exception as e:
            logger.debug("Failed to cleanup sampler index %s: %s", sampler_id, e)

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
            info = self._restore_from_detached_store(session_id)
            if info is None:
                return False
            self._sessions.pop(session_id, None)
        self._refresh_sampling_observability(before=info, after=None)
        try:
            from .sampling_session_store import delete_sampling_session

            delete_sampling_session(session_id)
        except Exception as e:
            logger.debug("Failed to delete sampling session %s from store: %s", session_id, e)

        if info.uses_multi_lora:
            self._cleanup_sampler_indices(session_id)

            # Best-effort: remove LoRA from vLLM and delete ephemeral adapter dir.
            manager = self.get_multi_model_manager()
            if manager is not None and info.base_model:
                engine = manager.get_engine_if_exists(info.base_model)
                if engine is not None:
                    try:
                        await engine.remove_session(session_id)
                    except Exception as e:
                        logger.warning(f"Failed to remove multi-LoRA session {session_id} from engine: {e}")

            # Drop per-session sampling locks only after teardown has finished.
            from ..routes.sampling import _drop_lora_load_lock

            await _drop_lora_load_lock(session_id)

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
        """Shutdown all sessions. Called on runtime exit."""
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
        """List active session IDs from TaskStateStore-backed metadata plus local pending sessions."""
        session_ids: set[str] = set(self._sessions.keys())
        try:
            from .sampling_session_store import list_sampling_sessions

            for info in list_sampling_sessions():
                if isinstance(info, dict):
                    session_id = str(info.get("session_id") or "").strip()
                    if session_id:
                        session_ids.add(session_id)
        except Exception:
            pass
        return list(session_ids)

    def observability_snapshot(self) -> dict[str, int | list[dict[str, int | str]]]:
        return {
            **self._obs_sampling_totals,
            "sampling_sessions_by_model": [
                dict(self._obs_sampling_by_model[key]) for key in sorted(self._obs_sampling_by_model)
            ],
        }

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
        info = self._get_session_info(session_id)
        return info.base_model if info else None

    async def get_engine_for_session(self, session_id: str) -> "MultiLoRAInferenceEngine | None":
        """Get vLLM engine for a session's model.

        Args:
            session_id: The session identifier.

        Returns:
            MultiLoRAInferenceEngine for the session's model, or None if not found.
        """
        info = self._get_session_info(session_id)
        if info is None:
            return None
        base_model = info.base_model
        if base_model is None:
            return None
        engine = await self.get_engine_for_model(base_model)
        await self._restore_loaded_lora_registration(session_id, info, engine)
        return engine

    def register_multi_lora_session(
        self,
        session_id: str,
        base_model: str,
        lora_rank: int = 32,
        adapter_path: str | None = None,
        *,
        lora_loaded: bool = True,
        lora_int_id: int | None = None,
        last_activity: float | None = None,
        metadata_version: int | None = None,
        persist: bool = True,
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
            last_activity=float(last_activity) if last_activity is not None else time.time(),
            lora_rank=lora_rank,
            is_shared=True,
            uses_multi_lora=True,
            base_model=base_model,
            adapter_path=adapter_path,
            lora_loaded=bool(lora_loaded),
            lora_int_id=None if lora_int_id is None else int(lora_int_id),
            metadata_version=max(1, int(metadata_version) if metadata_version is not None else 1),
            pending_persist=bool(persist),
        )
        self._refresh_sampling_observability(before=None, after=self._sessions[session_id])
        if persist:
            self._persist_sampling_session_info(session_id, self._sessions[session_id])
            self._sessions[session_id].pending_persist = False
        logger.info(
            f"Registered multi-LoRA session {session_id} (model={base_model}, lora_rank={lora_rank})"
        )

    def get_session_lora_rank(self, session_id: str) -> int | None:
        info = self._get_session_info(session_id)
        if info is None:
            return None
        return int(info.lora_rank)

    def get_session_adapter_path(self, session_id: str) -> str | None:
        info = self._get_session_info(session_id)
        if info is None:
            return None
        return info.adapter_path

    def get_session_lora_int_id(self, session_id: str) -> int | None:
        info = self._get_session_info(session_id)
        if info is None:
            return None
        return None if info.lora_int_id is None else int(info.lora_int_id)

    def get_session_metadata_version(self, session_id: str) -> int | None:
        info = self._get_session_info(session_id)
        if info is None:
            return None
        return max(1, int(info.metadata_version))

    def is_session_lora_loaded(self, session_id: str) -> bool:
        info = self._get_session_info(session_id)
        if info is None:
            return False
        return bool(info.lora_loaded)

    def mark_session_lora_loaded(
        self,
        session_id: str,
        loaded: bool = True,
        *,
        lora_int_id: int | None = None,
    ) -> None:
        info = self._get_session_info(session_id)
        if info is None:
            return
        before = SessionInfo(**vars(info))
        new_loaded = bool(loaded)
        changed = bool(info.lora_loaded) != new_loaded
        info.lora_loaded = new_loaded
        if lora_int_id is not None:
            changed = changed or (info.lora_int_id != int(lora_int_id))
            info.lora_int_id = int(lora_int_id)
        if changed:
            info.metadata_version = max(1, int(info.metadata_version) + 1)
        self._persist_sampling_session_info(session_id, info)
        self._refresh_sampling_observability(before=before, after=info)

    def mark_model_lora_sessions_unloaded(self, base_model: str) -> int:
        """Invalidate multi-LoRA load state for sessions bound to a base model.

        When the shared vLLM actor is killed or recreated, per-session LoRA load
        state tracked in the API process is stale. Those sessions must force a
        fresh add_lora_from_path on the next request instead of silently
        sampling without their adapter.
        """
        count = 0
        now = time.time()
        for info in self._sessions.values():
            if not info.uses_multi_lora:
                continue
            if info.uses_base_model:
                continue
            if info.base_model != base_model:
                continue
            if not info.adapter_path:
                continue
            if not info.lora_loaded:
                continue
            before = SessionInfo(**vars(info))
            info.last_activity = now
            info.lora_loaded = False
            self._refresh_sampling_observability(before=before, after=info)
            count += 1
        return count

    def is_multi_lora_session(self, session_id: str) -> bool:
        """Check if a session uses multi-LoRA mode.

        Args:
            session_id: The session identifier.

        Returns:
            True if session uses multi-LoRA, False otherwise.
        """
        info = self._get_session_info(session_id, touch=False)
        return info is not None and info.uses_multi_lora

    def is_base_model_session(self, session_id: str) -> bool:
        """Check if a session uses base model (no LoRA) on multi-LoRA engine.

        Args:
            session_id: The session identifier.

        Returns:
            True if session uses base model, False otherwise.
        """
        info = self._get_session_info(session_id, touch=False)
        return info is not None and info.uses_base_model

    def register_base_model_session(
        self,
        session_id: str,
        base_model: str,
        *,
        metadata_version: int | None = None,
    ) -> None:
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
            lora_loaded=False,
            metadata_version=max(1, int(metadata_version) if metadata_version is not None else 1),
            pending_persist=False,
        )
        self._refresh_sampling_observability(before=None, after=self._sessions[session_id])
        self._persist_sampling_session_info(session_id, self._sessions[session_id])
        logger.info(f"Registered base model session {session_id} (model={base_model})")

    def restore_sampling_session(self, info: dict) -> bool:
        """Restore a multi-LoRA sampling session from detached control-plane state."""
        session_id = str(info.get("session_id") or "")
        base_model = str(info.get("base_model") or "")
        if not session_id or not base_model:
            return False
        incoming_version = max(1, int(info.get("metadata_version") or 1))
        existing = self._sessions.get(session_id)
        if existing is not None:
            if incoming_version <= max(1, int(existing.metadata_version)):
                try:
                    existing.last_activity = max(
                        float(existing.last_activity),
                        float(info.get("last_activity", existing.last_activity)),
                    )
                except Exception:
                    pass
                return True
            before = SessionInfo(**vars(existing))
            existing.uses_base_model = bool(info.get("uses_base_model"))
            existing.base_model = base_model
            existing.last_activity = float(info.get("last_activity", existing.last_activity))
            if existing.uses_base_model:
                existing.lora_rank = 0
                existing.adapter_path = None
                existing.lora_loaded = False
                existing.lora_int_id = None
            else:
                existing.lora_rank = int(info.get("lora_rank") or 0)
                existing.adapter_path = info.get("adapter_path")
                existing.lora_loaded = bool(info.get("lora_loaded"))
                existing.lora_int_id = info.get("lora_int_id")
            existing.metadata_version = incoming_version
            existing.inflight_requests = int(info.get("inflight_requests") or existing.inflight_requests)
            existing.pending_persist = False
            self._refresh_sampling_observability(before=before, after=existing)
            return True

        uses_base_model = bool(info.get("uses_base_model"))
        last_activity = float(info.get("last_activity", time.time()))
        if uses_base_model:
            self._sessions[session_id] = SessionInfo(
                engine=None,
                last_activity=last_activity,
                lora_rank=0,
                is_shared=True,
                uses_multi_lora=True,
                uses_base_model=True,
                base_model=base_model,
                lora_loaded=False,
                metadata_version=incoming_version,
                inflight_requests=int(info.get("inflight_requests") or 0),
                pending_persist=False,
            )
            self._refresh_sampling_observability(before=None, after=self._sessions[session_id])
            return True

        self.register_multi_lora_session(
            session_id=session_id,
            base_model=base_model,
            lora_rank=int(info.get("lora_rank") or 0),
            adapter_path=info.get("adapter_path"),
            lora_loaded=bool(info.get("lora_loaded")),
            lora_int_id=info.get("lora_int_id"),
            last_activity=last_activity,
            metadata_version=incoming_version,
            persist=False,
        )
        restored = self._sessions.get(session_id)
        if restored is not None:
            before = SessionInfo(**vars(restored))
            restored.inflight_requests = int(info.get("inflight_requests") or 0)
            restored.pending_persist = False
            self._refresh_sampling_observability(before=before, after=restored)
        return True

    async def _restore_loaded_lora_registration(
        self,
        session_id: str,
        info: SessionInfo,
        engine: "MultiLoRAInferenceEngine",
    ) -> None:
        if not info.uses_multi_lora or info.uses_base_model or not info.lora_loaded:
            return
        if not info.adapter_path or info.lora_int_id is None:
            return
        existing_id = await engine.registry.get_lora_id(session_id)
        if existing_id is not None:
            info.lora_int_id = int(existing_id)
            return
        restore_loaded = getattr(engine, "restore_loaded_session", None)
        if restore_loaded is None:
            return
        info.lora_int_id = int(
            await restore_loaded(
                sampling_session_id=session_id,
                adapter_path=info.adapter_path,
                lora_int_id=int(info.lora_int_id),
            )
        )

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
