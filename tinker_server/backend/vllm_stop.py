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
    def _expand_newlines(s: str) -> list[str]:
        # Issue #222: some trained outputs end with literal "\\n" (backslash+n) sequences
        # used as a delimiter. Expand newline-only stop sequences (e.g. "\n\n") to also
        # match the escaped form ("\\n\\n"), without changing semantics for more complex
        # stop strings that legitimately contain newlines.
        if len(s) < 2:
            return [s]
        if any(ch != "\n" for ch in s):
            return [s]
        return [s, s.replace("\n", "\\n")]

    if stop is None:
        if default_stop_token_ids is None:
            return {}
        return {"stop_token_ids": list(default_stop_token_ids)}

    if isinstance(stop, str):
        if not stop:
            return {}
        expanded = _expand_newlines(stop)
        if len(expanded) == 1:
            return {"stop": expanded[0]}
        return {"stop": expanded}

    if isinstance(stop, list):
        if not stop:
            return {}
        if all(isinstance(x, int) for x in stop):
            return {"stop_token_ids": [int(x) for x in stop]}
        if all(isinstance(x, str) for x in stop):
            expanded: list[str] = []
            seen: set[str] = set()
            for raw in stop:
                if not raw:
                    continue
                for s in _expand_newlines(str(raw)):
                    if s in seen:
                        continue
                    seen.add(s)
                    expanded.append(s)
            return {"stop": expanded}
        raise ValueError(f"stop must be list[int] or list[str], got mixed: {stop!r}")

    raise TypeError(f"stop must be None, str, list[str], or list[int]; got {type(stop)}")
