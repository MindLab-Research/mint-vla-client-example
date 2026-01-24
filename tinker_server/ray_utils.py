from __future__ import annotations

import os
from typing import Any


def ray_log_to_driver_enabled() -> bool:
    v = os.environ.get("MINT_RAY_LOG_TO_DRIVER", "").strip().lower()
    return v not in {"", "0", "false", "no", "off"}


def ray_log_to_driver_kwargs() -> dict[str, Any]:
    return {"log_to_driver": True} if ray_log_to_driver_enabled() else {}

