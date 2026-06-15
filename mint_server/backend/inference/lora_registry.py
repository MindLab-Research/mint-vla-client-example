"""LoRA registry and eviction helpers.

This module is intentionally ray-free so local tests and repro scripts can import it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from mint_server.runtime_env import env_get

logger = logging.getLogger(__name__)


@dataclass
class LoRASlotInfo:
    """Metadata for a loaded LoRA adapter."""

    lora_int_id: int
    sampling_session_id: str
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    def idle_time(self) -> float:
        return time.time() - self.last_used


class LoRARegistry:
    """Maps sampling_session_id to lora_int_id with LRU tracking.

    Each sampling session gets a unique lora_int_id for frozen weights.
    Tracks usage for LRU eviction when GPU slots are exhausted.
    """

    def __init__(self):
        self._session_to_id: dict[str, int] = {}
        self._id_to_session: dict[int, str] = {}
        self._session_to_path: dict[str, str] = {}
        self._path_to_id: dict[str, int] = {}
        self._id_to_path: dict[int, str] = {}
        self._path_refcount: dict[str, int] = {}
        self._slot_info: dict[int, LoRASlotInfo] = {}
        self._lru_order: OrderedDict[int, None] = OrderedDict()  # LRU at front
        self._next_id: int = 1
        self._lock = asyncio.Lock()

    async def allocate(self, sampling_session_id: str) -> int:
        """Allocate a unique lora_int_id for a sampling session."""
        async with self._lock:
            if sampling_session_id in self._session_to_id:
                raise ValueError(
                    f"Session {sampling_session_id} already has lora_int_id "
                    f"{self._session_to_id[sampling_session_id]}"
                )

            lora_id = self._next_id
            self._next_id += 1

            self._session_to_id[sampling_session_id] = lora_id
            self._id_to_session[lora_id] = sampling_session_id
            self._slot_info[lora_id] = LoRASlotInfo(
                lora_int_id=lora_id,
                sampling_session_id=sampling_session_id,
            )
            self._lru_order[lora_id] = None

            logger.debug(
                f"Allocated lora_int_id={lora_id} for session {sampling_session_id}"
            )
            return lora_id

    async def allocate_for_path(self, sampling_session_id: str, adapter_path: str) -> tuple[int, bool]:
        """Allocate or reuse a lora_int_id for a sampling session by adapter path.

        Returns:
            (lora_id, is_new_load)
        """
        async with self._lock:
            if sampling_session_id in self._session_to_id:
                raise ValueError(
                    f"Session {sampling_session_id} already has lora_int_id "
                    f"{self._session_to_id[sampling_session_id]}"
                )

            existing_id = self._path_to_id.get(adapter_path)
            if existing_id is not None:
                self._session_to_id[sampling_session_id] = existing_id
                self._session_to_path[sampling_session_id] = adapter_path
                self._path_refcount[adapter_path] = self._path_refcount.get(adapter_path, 0) + 1
                self._lru_order.move_to_end(existing_id)
                if existing_id in self._slot_info:
                    self._slot_info[existing_id].last_used = time.time()
                logger.debug(
                    "Reused lora_int_id=%s for session %s path=%s",
                    existing_id,
                    sampling_session_id,
                    adapter_path,
                )
                return existing_id, False

            lora_id = self._next_id
            self._next_id += 1
            self._session_to_id[sampling_session_id] = lora_id
            self._id_to_session[lora_id] = sampling_session_id
            self._session_to_path[sampling_session_id] = adapter_path
            self._path_to_id[adapter_path] = lora_id
            self._id_to_path[lora_id] = adapter_path
            self._path_refcount[adapter_path] = 1
            self._slot_info[lora_id] = LoRASlotInfo(
                lora_int_id=lora_id,
                sampling_session_id=sampling_session_id,
            )
            self._lru_order[lora_id] = None
            logger.debug(
                "Allocated new lora_int_id=%s for session %s path=%s",
                lora_id,
                sampling_session_id,
                adapter_path,
            )
            return lora_id, True

    async def get_lora_id(self, sampling_session_id: str) -> int | None:
        """Get lora_int_id for a sampling session and update LRU."""
        async with self._lock:
            lora_id = self._session_to_id.get(sampling_session_id)
            if lora_id is not None:
                self._lru_order.move_to_end(lora_id)
                if lora_id in self._slot_info:
                    self._slot_info[lora_id].last_used = time.time()
            return lora_id

    async def restore_existing_session(
        self,
        sampling_session_id: str,
        *,
        adapter_path: str,
        lora_int_id: int,
    ) -> int:
        """Rehydrate an already-loaded LoRA mapping after API restart."""
        async with self._lock:
            existing_id = self._session_to_id.get(sampling_session_id)
            if existing_id is not None:
                return existing_id

            adapter_path = str(adapter_path)
            lora_id = int(lora_int_id)
            now = time.time()

            self._session_to_id[sampling_session_id] = lora_id
            self._session_to_path[sampling_session_id] = adapter_path

            existing_path_id = self._path_to_id.get(adapter_path)
            if existing_path_id is not None and existing_path_id != lora_id:
                raise ValueError(
                    f"Adapter path {adapter_path} already mapped to lora_int_id={existing_path_id}, "
                    f"cannot restore lora_int_id={lora_id}"
                )

            self._path_to_id[adapter_path] = lora_id
            self._id_to_path[lora_id] = adapter_path
            self._path_refcount[adapter_path] = self._path_refcount.get(adapter_path, 0) + 1

            slot = self._slot_info.get(lora_id)
            if slot is None:
                slot = LoRASlotInfo(
                    lora_int_id=lora_id,
                    sampling_session_id=sampling_session_id,
                    loaded_at=now,
                    last_used=now,
                )
                self._slot_info[lora_id] = slot
            else:
                slot.last_used = now

            self._id_to_session.setdefault(lora_id, slot.sampling_session_id)
            self._lru_order[lora_id] = None
            self._lru_order.move_to_end(lora_id)
            self._next_id = max(self._next_id, lora_id + 1)
            return lora_id

    async def get_lru_candidates(self, count: int) -> list[int]:
        """Get the least recently used lora_int_ids for eviction."""
        async with self._lock:
            try:
                min_idle_s = float(env_get(os.environ, "MINT_LORA_EVICT_MIN_IDLE_S", "5.0") or "5.0")
            except ValueError:
                min_idle_s = 5.0

            candidates = []
            for lora_id in self._lru_order:
                if len(candidates) >= count:
                    break
                slot = self._slot_info.get(lora_id)
                if slot is None:
                    continue
                if slot.idle_time() <= min_idle_s:
                    continue
                candidates.append(lora_id)
            return candidates

    async def remove(self, lora_id: int) -> str | None:
        """Remove a lora_int_id from the registry."""
        async with self._lock:
            session_id = self._id_to_session.pop(lora_id, None)
            if session_id:
                sessions_to_remove = [
                    sid for sid, existing_id in self._session_to_id.items() if existing_id == lora_id
                ]
                if session_id not in sessions_to_remove:
                    sessions_to_remove.append(session_id)
                for sid in sessions_to_remove:
                    self._session_to_id.pop(sid, None)
                    self._session_to_path.pop(sid, None)
                adapter_path = self._id_to_path.pop(lora_id, None)
                if adapter_path is not None:
                    self._path_to_id.pop(adapter_path, None)
                    self._path_refcount.pop(adapter_path, None)
                self._slot_info.pop(lora_id, None)
                self._lru_order.pop(lora_id, None)
                logger.debug(
                    "Removed lora_int_id=%s (sessions=%s)", lora_id, sessions_to_remove
                )
            return session_id

    async def remove_session(self, sampling_session_id: str) -> tuple[int | None, bool]:
        """Remove a session reference.

        Returns:
            (lora_id, should_unload)
        """
        async with self._lock:
            lora_id = self._session_to_id.pop(sampling_session_id, None)
            if lora_id is None:
                return None, False

            adapter_path = self._session_to_path.pop(sampling_session_id, None)
            if adapter_path is None:
                # Fallback to old behavior if path bookkeeping was absent.
                owner_session = self._id_to_session.pop(lora_id, None)
                if owner_session is not None and owner_session != sampling_session_id:
                    self._session_to_id[owner_session] = lora_id
                    self._id_to_session[lora_id] = owner_session
                    return lora_id, False
                self._slot_info.pop(lora_id, None)
                self._lru_order.pop(lora_id, None)
                return lora_id, True

            remaining = self._path_refcount.get(adapter_path, 0) - 1
            if remaining > 0:
                self._path_refcount[adapter_path] = remaining
                owner_session = self._id_to_session.get(lora_id)
                if owner_session == sampling_session_id:
                    replacement = next(
                        (sid for sid, path in self._session_to_path.items() if path == adapter_path),
                        None,
                    )
                    if replacement is not None:
                        self._id_to_session[lora_id] = replacement
                        if lora_id in self._slot_info:
                            self._slot_info[lora_id].sampling_session_id = replacement
                if lora_id in self._slot_info:
                    self._slot_info[lora_id].last_used = time.time()
                return lora_id, False

            self._path_refcount.pop(adapter_path, None)
            self._path_to_id.pop(adapter_path, None)
            self._id_to_path.pop(lora_id, None)
            self._id_to_session.pop(lora_id, None)
            self._slot_info.pop(lora_id, None)
            self._lru_order.pop(lora_id, None)
            logger.debug(
                "Removed final session %s for lora_int_id=%s path=%s",
                sampling_session_id,
                lora_id,
                adapter_path,
            )
            return lora_id, True

    async def count(self) -> int:
        """Get the number of registered sessions."""
        async with self._lock:
            return len(self._slot_info)

    async def list_sessions(self) -> list[str]:
        """List all registered sampling session IDs."""
        async with self._lock:
            return list(self._session_to_id.keys())
