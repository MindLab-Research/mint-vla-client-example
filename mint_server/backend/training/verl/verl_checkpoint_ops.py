from __future__ import annotations

import logging
import math
import os
from typing import Any, cast

logger = logging.getLogger(__name__)


class CheckpointOps:
    """Checkpoint metadata helpers for `VerlTrainingEngine`."""

    @staticmethod
    def strict_megatron_save_meta_enabled() -> bool:
        raw = os.environ.get("MINT_MEGATRON_STRICT_SAVE_META", "1").strip().lower()
        return raw not in ("0", "false", "no", "off")

    @staticmethod
    def apply_megatron_loaded_lora_config(session: Any, meta: dict[str, object]) -> None:
        lora_cfg = getattr(session, "lora_config", None)
        if lora_cfg is None:
            return
        updates = {
            "rank": int(cast(int, meta["actual_rank"])),
            "train_attn": bool(meta["train_attn"]),
            "train_mlp": bool(meta["train_mlp"]),
            "train_unembed": bool(meta["train_unembed"]),
        }
        if hasattr(lora_cfg, "model_copy"):
            session.lora_config = lora_cfg.model_copy(update=updates)
            return
        if hasattr(lora_cfg, "copy"):
            session.lora_config = lora_cfg.copy(update=updates)
            return
        for key, value in updates.items():
            setattr(lora_cfg, key, value)

    @staticmethod
    def validate_megatron_load_meta(meta: Any, *, op: str) -> dict[str, object]:
        if not isinstance(meta, dict):
            raise RuntimeError(
                f"Megatron load_checkpoint returned invalid metadata for {op}: "
                f"expected dict, got {type(meta).__name__}"
            )

        required_keys = {
            "current_step",
            "learning_rate",
            "actual_rank",
            "actor_only_state_dirty",
            "checkpoint_path",
            "optimizer_restored",
            "train_attn",
            "train_mlp",
            "train_unembed",
        }
        missing = sorted(required_keys - set(meta))
        if missing:
            raise RuntimeError(
                "Megatron load_checkpoint returned invalid metadata for "
                f"{op}: missing keys {missing}"
            )

        current_step = meta["current_step"]
        if not isinstance(current_step, int) or isinstance(current_step, bool) or current_step < 0:
            raise RuntimeError(
                "Megatron load_checkpoint returned invalid metadata for "
                f"{op}: current_step must be a non-negative int, got {current_step!r}"
            )

        lr_value = meta["learning_rate"]
        if not isinstance(lr_value, (int, float)) or isinstance(lr_value, bool):
            raise RuntimeError(
                "Megatron load_checkpoint returned invalid metadata for "
                f"{op}: learning_rate must be finite, got {lr_value!r}"
            )
        learning_rate = float(lr_value)
        if not math.isfinite(learning_rate):
            raise RuntimeError(
                "Megatron load_checkpoint returned invalid metadata for "
                f"{op}: learning_rate must be finite, got {lr_value!r}"
            )

        actual_rank = meta["actual_rank"]
        if not isinstance(actual_rank, int) or isinstance(actual_rank, bool) or actual_rank <= 0:
            raise RuntimeError(
                "Megatron load_checkpoint returned invalid metadata for "
                f"{op}: actual_rank must be a positive int, got {actual_rank!r}"
            )

        checkpoint_path = meta["checkpoint_path"]
        if not isinstance(checkpoint_path, str) or not checkpoint_path:
            raise RuntimeError(
                "Megatron load_checkpoint returned invalid metadata for "
                f"{op}: checkpoint_path must be a non-empty string, got {checkpoint_path!r}"
            )

        normalized: dict[str, object] = {
            "current_step": current_step,
            "learning_rate": learning_rate,
            "actual_rank": actual_rank,
            "checkpoint_path": checkpoint_path,
        }
        for key in (
            "actor_only_state_dirty",
            "optimizer_restored",
            "train_attn",
            "train_mlp",
            "train_unembed",
        ):
            value = meta[key]
            if not isinstance(value, bool):
                raise RuntimeError(
                    "Megatron load_checkpoint returned invalid metadata for "
                    f"{op}: {key} must be bool, got {value!r}"
                )
            normalized[key] = value
        return normalized

    @staticmethod
    def update_session_step_monotonic(
        session: Any,
        meta: Any,
        *,
        op: str,
        strict: bool = False,
    ) -> None:
        """Monotonic, type-safe current_step update from worker metadata."""
        model_id = session.model_id
        if not isinstance(meta, dict):
            msg = (
                f"[{model_id}] {op}: invalid meta type {type(meta).__name__}; "
                f"current_step={session.current_step}"
            )
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
            return

        if "current_step" not in meta:
            msg = (
                f"[{model_id}] {op}: meta missing current_step; "
                f"current_step={session.current_step}"
            )
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
            return

        meta_step = meta.get("current_step")
        if not isinstance(meta_step, int) or isinstance(meta_step, bool):
            msg = (
                f"[{model_id}] {op}: invalid current_step type={type(meta_step).__name__} "
                f"value={meta_step!r}; current_step={session.current_step}"
            )
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
            return

        prev_step = session.current_step
        next_step = max(prev_step, meta_step)
        if meta_step < prev_step:
            logger.warning(
                "[%s] %s: stale current_step=%s < existing=%s; keep monotonic value=%s",
                model_id,
                op,
                meta_step,
                prev_step,
                next_step,
            )
        session.current_step = next_step

    @staticmethod
    def update_session_from_load_meta(
        session: Any,
        meta: Any,
        *,
        op: str,
    ) -> None:
        """Best-effort load metadata application without polluting session state."""
        model_id = session.model_id
        if not isinstance(meta, dict):
            logger.warning(
                "[%s] %s: invalid meta type %s; preserving current_step=%s lr=%s",
                model_id,
                op,
                type(meta).__name__,
                session.current_step,
                session.learning_rate,
            )
            return

        meta_step = meta.get("current_step")
        if isinstance(meta_step, int) and not isinstance(meta_step, bool):
            session.current_step = meta_step
        elif "current_step" in meta:
            logger.warning(
                "[%s] %s: invalid current_step type=%s value=%r; preserving current_step=%s",
                model_id,
                op,
                type(meta_step).__name__,
                meta_step,
                session.current_step,
            )
        else:
            logger.warning(
                "[%s] %s: meta missing current_step; preserving current_step=%s",
                model_id,
                op,
                session.current_step,
            )

        if "learning_rate" not in meta:
            return
        try:
            session.learning_rate = float(meta["learning_rate"])
        except Exception:
            logger.warning(
                "[%s] %s: invalid learning_rate type=%s value=%r; preserving learning_rate=%s",
                model_id,
                op,
                type(meta["learning_rate"]).__name__,
                meta["learning_rate"],
                session.learning_rate,
            )
