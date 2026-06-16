"""Configuration helpers for ``VerlTrainingEngine``."""

from __future__ import annotations

import os


class VerlEngineConfig:
    """Parse environment-backed Verl engine knobs in one place."""

    def megatron_guard_preflight_enabled(self) -> bool:
        raw = os.environ.get("MINT_MEGATRON_GUARD_PREFLIGHT", "1").strip().lower()
        return raw not in ("0", "false", "no", "off")

    def megatron_guard_query_timeout_s(self) -> float:
        raw = os.environ.get("MINT_MEGATRON_GUARD_QUERY_TIMEOUT_S", "30").strip()
        try:
            timeout_s = float(raw)
        except Exception:
            timeout_s = 30.0
        return max(1.0, timeout_s)
