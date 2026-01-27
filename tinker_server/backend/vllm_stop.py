from __future__ import annotations

from typing import Any


def vllm_stop_kwargs(
    stop: Any | None,
    *,
    default_stop_token_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Map Tinker stop param to vLLM SamplingParams kwargs.

    Supported input shapes (from tinker_server.models.types.SamplingParams.stop):
    - None
    - str
    - list[str]
    - list[int]
    """
    if stop is None:
        if default_stop_token_ids is None:
            return {}
        return {"stop_token_ids": list(default_stop_token_ids)}

    if isinstance(stop, str):
        if not stop:
            return {}
        return {"stop": stop}

    if isinstance(stop, list):
        if not stop:
            return {}
        if all(isinstance(x, int) for x in stop):
            return {"stop_token_ids": [int(x) for x in stop]}
        if all(isinstance(x, str) for x in stop):
            return {"stop": [str(x) for x in stop if x]}
        raise ValueError(f"stop must be list[int] or list[str], got mixed: {stop!r}")

    raise TypeError(f"stop must be None, str, list[str], or list[int]; got {type(stop)}")

