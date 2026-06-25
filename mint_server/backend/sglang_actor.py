from __future__ import annotations

import hashlib
import importlib
import logging
import os
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .sglang_capabilities import (
    SGLangUnsupportedFeatureError,
    canonical_peft_adapter_path,
    validate_sglang_lora_adapter_supported,
)

logger = logging.getLogger(__name__)

SGLANG_DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class SGLangBackendUnavailableError(RuntimeError):
    """SGLang cannot be imported or initialized in the worker runtime."""


@contextmanager
def _disable_config_actor_hydration_for_child_processes() -> Iterator[None]:
    previous = os.environ.get("MINT_CONFIG_ACTOR_HYDRATE")
    os.environ["MINT_CONFIG_ACTOR_HYDRATE"] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MINT_CONFIG_ACTOR_HYDRATE", None)
        else:
            os.environ["MINT_CONFIG_ACTOR_HYDRATE"] = previous


def _adapter_name_for_path(canonical_path: str) -> str:
    digest = hashlib.sha1(str(canonical_path).encode("utf-8")).hexdigest()[:16]
    return f"mint_sglang_lora_{digest}"


def _sglang_result_success(result: object) -> bool:
    if result is None:
        return True
    if isinstance(result, bool):
        return result
    if isinstance(result, dict) and "success" in result:
        return bool(result.get("success"))
    success = getattr(result, "success", None)
    if success is not None:
        return bool(success)
    return True


def _sglang_result_error(result: object) -> str | None:
    if isinstance(result, dict):
        for key in ("error", "message", "detail"):
            value = result.get(key)
            if value:
                return str(value)
    for key in ("error", "message", "detail"):
        value = getattr(result, key, None)
        if value:
            return str(value)
    return None


def _prepend_pythonpath_from_env() -> None:
    raw = os.environ.get("MINT_SGLANG_PYTHONPATH", "").strip()
    if not raw:
        return
    for item in reversed([x for x in raw.split(os.pathsep) if x.strip()]):
        if item not in sys.path:
            sys.path.insert(0, item)


_SGLANG_EXCLUDED_SOURCE_MARKERS = (
    "/src/Megatron-LM",
    "/src/Megatron-Bridge",
    "/src/verl",
    "/src/vllm",
)


def _is_training_source_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return any(marker in normalized for marker in _SGLANG_EXCLUDED_SOURCE_MARKERS)


def _drop_training_source_paths_from_import_state() -> dict[str, Any]:
    before_sys_path = list(sys.path)
    sys.path[:] = [path for path in before_sys_path if not _is_training_source_path(path)]

    before_pythonpath = os.environ.get("PYTHONPATH", "")
    if before_pythonpath:
        kept = [path for path in before_pythonpath.split(os.pathsep) if path and not _is_training_source_path(path)]
        os.environ["PYTHONPATH"] = os.pathsep.join(kept)

    after_pythonpath = os.environ.get("PYTHONPATH", "")
    return {
        "removed_sys_path": len(before_sys_path) - len(sys.path),
        "removed_pythonpath": len([p for p in before_pythonpath.split(os.pathsep) if p and _is_training_source_path(p)]),
        "pythonpath_entries": len([p for p in after_pythonpath.split(os.pathsep) if p]),
    }


def _apply_sglang_import_boundary() -> dict[str, Any]:
    _prepend_pythonpath_from_env()
    return _drop_training_source_paths_from_import_state()


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return int(default)
    return value if value > 0 else int(default)


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return float(default)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _csv_env(name: str, default: tuple[str, ...]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return list(default)
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    return values or list(default)


def _disable_sgl_kernel_module_flashinfer(module_name: str, *, import_if_missing: bool = True) -> dict[str, Any]:
    try:
        if import_if_missing:
            module = importlib.import_module(module_name)
        else:
            module = sys.modules.get(module_name)
            if module is None:
                return {
                    "available": False,
                    "skipped": "not_loaded",
                }
    except Exception as e:
        return {
            "available": False,
            "error": f"{type(e).__name__}: {e}",
        }

    before = getattr(module, "_has_flashinfer", None)
    if hasattr(module, "_has_flashinfer"):
        setattr(module, "_has_flashinfer", False)
    return {
        "available": True,
        "has_flashinfer_before": bool(before) if before is not None else None,
        "has_flashinfer_after": bool(getattr(module, "_has_flashinfer", False)),
    }


def _apply_sglang_flashinfer_kernel_compatibility(*, import_direct_kernel_modules: bool) -> dict[str, Any]:
    """Disable FlashInfer paths without importing heavy direct-kernel modules unless required."""
    enabled = _bool_env("MINT_SGLANG_DISABLE_FLASHINFER_KERNELS", True)
    result: dict[str, Any] = {
        "enabled": enabled,
        "env_var": "MINT_SGLANG_DISABLE_FLASHINFER_KERNELS",
        "import_direct_kernel_modules": import_direct_kernel_modules,
    }
    if not enabled:
        return result

    os.environ["SGLANG_IS_FLASHINFER_AVAILABLE"] = "0"
    result["SGLANG_IS_FLASHINFER_AVAILABLE"] = os.environ["SGLANG_IS_FLASHINFER_AVAILABLE"]
    common = sys.modules.get("sglang.srt.utils.common")
    if common is None and import_direct_kernel_modules:
        try:
            common = importlib.import_module("sglang.srt.utils.common")
        except Exception as e:
            result["sglang_is_flashinfer_available_error"] = f"{type(e).__name__}: {e}"
    if common is None:
        result["sglang_common"] = {"available": False, "skipped": "not_loaded"}
    else:
        try:
            is_flashinfer_available = getattr(common, "is_flashinfer_available")
            cache_clear = getattr(is_flashinfer_available, "cache_clear", None)
            if callable(cache_clear):
                cache_clear()
            result["sglang_is_flashinfer_available"] = bool(is_flashinfer_available())
        except Exception as e:
            result["sglang_is_flashinfer_available_error"] = f"{type(e).__name__}: {e}"

    result["modules"] = {
        "sgl_kernel.elementwise": _disable_sgl_kernel_module_flashinfer(
            "sgl_kernel.elementwise",
            import_if_missing=import_direct_kernel_modules,
        ),
        "sgl_kernel.sampling": _disable_sgl_kernel_module_flashinfer(
            "sgl_kernel.sampling",
            import_if_missing=import_direct_kernel_modules,
        ),
    }
    return result


def _run_sglang_scheduler_process_with_mint_compatibility(*args: Any, **kwargs: Any) -> Any:
    _prepend_pythonpath_from_env()
    _apply_sglang_flashinfer_kernel_compatibility(import_direct_kernel_modules=True)
    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)


def _engine_class_with_mint_scheduler_compatibility(engine_cls: type[Any]) -> type[Any]:
    if not _bool_env("MINT_SGLANG_DISABLE_FLASHINFER_KERNELS", True):
        return engine_cls

    class MintSGLangEngine(engine_cls):  # type: ignore[misc, valid-type]
        run_scheduler_process_func = staticmethod(_run_sglang_scheduler_process_with_mint_compatibility)

    MintSGLangEngine.__name__ = f"Mint{getattr(engine_cls, '__name__', 'SGLangEngine')}"
    return MintSGLangEngine


def _sglang_lora_engine_kwargs(
    *,
    default_enable_lora: bool = True,
    default_max_lora_rank: int = 64,
    default_max_loaded_loras: int = 8,
    default_max_loras_per_batch: int | None = None,
    default_lora_target_modules: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    if not _bool_env("MINT_SGLANG_ENABLE_LORA", default_enable_lora):
        return {"enable_lora": False}
    max_loaded_loras = _positive_int_env(
        "MINT_SGLANG_MAX_LOADED_LORAS",
        max(1, int(default_max_loaded_loras)),
    )
    default_per_batch = int(default_max_loras_per_batch or min(8, max_loaded_loras))
    max_loras_per_batch = _positive_int_env(
        "MINT_SGLANG_MAX_LORAS_PER_BATCH",
        max(1, default_per_batch),
    )
    if max_loras_per_batch > max_loaded_loras:
        max_loras_per_batch = max_loaded_loras
    lora_target_modules = tuple(default_lora_target_modules or SGLANG_DEFAULT_LORA_TARGET_MODULES)
    return {
        "enable_lora": True,
        "enable_lora_overlap_loading": _bool_env("MINT_SGLANG_ENABLE_LORA_OVERLAP_LOADING", False),
        "max_lora_rank": _positive_int_env(
            "MINT_SGLANG_MAX_LORA_RANK",
            max(1, int(default_max_lora_rank)),
        ),
        "lora_target_modules": _csv_env(
            "MINT_SGLANG_LORA_TARGET_MODULES",
            lora_target_modules,
        ),
        "max_loaded_loras": max_loaded_loras,
        "max_loras_per_batch": max_loras_per_batch,
        "lora_backend": os.environ.get("MINT_SGLANG_LORA_BACKEND", "triton").strip() or "triton",
    }


def _engine_init_log_fields(kwargs: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "dtype",
        "tp_size",
        "context_length",
        "mem_fraction_static",
        "max_running_requests",
        "max_total_tokens",
        "max_prefill_tokens",
        "chunked_prefill_size",
        "disable_cuda_graph",
        "disable_piecewise_cuda_graph",
        "disable_flashinfer_autotune",
        "disable_custom_all_reduce",
        "skip_server_warmup",
        "trust_remote_code",
        "watchdog_timeout",
        "enable_lora",
        "max_lora_rank",
        "max_loaded_loras",
        "max_loras_per_batch",
        "lora_backend",
    )
    return {key: kwargs.get(key) for key in allowed if key in kwargs}


def _init_trace(message: str, **fields: Any) -> None:
    pieces = [f"SGLang actor initialize {message}"]
    for key, value in fields.items():
        pieces.append(f"{key}={value}")
    print(" ".join(pieces), flush=True)


def _build_sampling_params(
    *,
    max_tokens: int,
    stop: object | None,
    temperature: float,
    top_k: int,
    top_p: float,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": int(max_tokens),
    }
    if int(top_k) > 0:
        params["top_k"] = int(top_k)
    if stop is None:
        return params
    if isinstance(stop, str):
        params["stop"] = stop
        return params
    if isinstance(stop, list) and all(isinstance(x, str) for x in stop):
        params["stop"] = list(stop)
        return params
    if isinstance(stop, list) and all(isinstance(x, int) for x in stop):
        params["stop_token_ids"] = [int(x) for x in stop]
        return params
    raise TypeError(f"stop must be None, str, list[str], or list[int]; got {type(stop)}")


def _normalize_stop_reason(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("type", "matched", "reason"):
            if value.get(key) is not None:
                return str(value[key])
        return None
    reason = getattr(value, "type", None)
    if reason is not None:
        return str(reason)
    return str(value)


def _coerce_sglang_logprob_value(item: Any, *, field_name: str) -> float | None:
    value: object
    if item is None:
        return None
    if isinstance(item, (list, tuple)) and item:
        value = item[0]
    elif isinstance(item, dict):
        value = item.get("logprob")
    else:
        value = item
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"Malformed SGLang {field_name} entry: {item!r}") from e


def _coerce_sglang_token_id(value: Any, *, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"Malformed SGLang {field_name} token id: {value!r}") from e


def _normalize_generated_logprobs(meta_info: dict[str, Any]) -> list[float] | None:
    raw = meta_info.get("output_token_logprobs")
    if raw is None:
        return None
    out: list[float] = []
    for item in raw:
        value = _coerce_sglang_logprob_value(item, field_name="output_token_logprobs")
        if value is None:
            raise RuntimeError(f"Malformed SGLang output_token_logprobs entry: {item!r}")
        out.append(value)
    return out


def normalize_sglang_prompt_logprobs_response(
    response: Any,
    *,
    prompt_ids: list[int],
) -> list[float | None]:
    """Normalize SGLang prompt logprobs to Mint/Tinker prompt-token shape."""
    if not isinstance(response, dict):
        raise RuntimeError(f"SGLang prompt logprobs response was {type(response).__name__}, expected dict")
    meta_info = response.get("meta_info")
    if not isinstance(meta_info, dict):
        raise RuntimeError("SGLang prompt logprobs response missing dict meta_info")
    raw = meta_info.get("input_token_logprobs")
    if raw is None:
        raise RuntimeError("SGLang prompt logprobs response missing meta_info.input_token_logprobs")
    if not isinstance(raw, list):
        raise RuntimeError(f"SGLang input_token_logprobs must be a list, got {type(raw).__name__}")
    if len(raw) != len(prompt_ids):
        raise RuntimeError(
            f"SGLang input_token_logprobs length {len(raw)} does not match prompt length {len(prompt_ids)}"
        )

    out: list[float | None] = []
    for i, item in enumerate(raw):
        value = _coerce_sglang_logprob_value(item, field_name="input_token_logprobs")
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            token_id = _coerce_sglang_token_id(item[1], field_name="input_token_logprobs")
            expected = int(prompt_ids[i])
            if token_id != expected:
                raise RuntimeError(
                    f"SGLang input_token_logprobs[{i}] token id {token_id} does not match prompt_ids[{i}] {expected}"
                )
        elif isinstance(item, dict) and item.get("token_id") is not None:
            token_id = _coerce_sglang_token_id(item.get("token_id"), field_name="input_token_logprobs")
            expected = int(prompt_ids[i])
            if token_id != expected:
                raise RuntimeError(
                    f"SGLang input_token_logprobs[{i}] token id {token_id} does not match prompt_ids[{i}] {expected}"
                )
        if i == 0:
            out.append(None)
            continue
        if item is None:
            out.append(None)
            continue
        if value is None:
            raise RuntimeError(f"SGLang input_token_logprobs[{i}] missing logprob for prompt token")
        out.append(value)
    return out


def normalize_sglang_prompt_topk_response(
    response: Any,
    *,
    prompt_ids: list[int],
    k: int,
) -> list[list[tuple[int, float]] | None]:
    """Normalize SGLang prompt top-k logprobs to Mint/Tinker prompt-token shape."""
    kk = int(k)
    if kk <= 0:
        return [None] * len(prompt_ids)
    if not isinstance(response, dict):
        raise RuntimeError(f"SGLang prompt top-k response was {type(response).__name__}, expected dict")
    meta_info = response.get("meta_info")
    if not isinstance(meta_info, dict):
        raise RuntimeError("SGLang prompt top-k response missing dict meta_info")
    raw = meta_info.get("input_top_logprobs")
    if raw is None:
        raise RuntimeError("SGLang prompt top-k response missing meta_info.input_top_logprobs")
    if not isinstance(raw, list):
        raise RuntimeError(f"SGLang input_top_logprobs must be a list, got {type(raw).__name__}")
    if len(raw) != len(prompt_ids):
        raise RuntimeError(
            f"SGLang input_top_logprobs length {len(raw)} does not match prompt length {len(prompt_ids)}"
        )

    out: list[list[tuple[int, float]] | None] = []
    for i, entry in enumerate(raw):
        if i == 0:
            out.append(None)
            continue
        if entry is None:
            out.append(None)
            continue
        if not isinstance(entry, list):
            raise RuntimeError(f"SGLang input_top_logprobs[{i}] must be a list or None, got {type(entry).__name__}")
        pairs: list[tuple[int, float]] = []
        for pair in entry:
            if isinstance(pair, dict):
                logprob = _coerce_sglang_logprob_value(pair, field_name="input_top_logprobs")
                token_value = pair.get("token_id")
            elif isinstance(pair, (list, tuple)) and len(pair) >= 2:
                logprob = _coerce_sglang_logprob_value(pair, field_name="input_top_logprobs")
                token_value = pair[1]
            else:
                raise RuntimeError(f"SGLang input_top_logprobs[{i}] has malformed pair: {pair!r}")
            if logprob is None:
                raise RuntimeError(f"SGLang input_top_logprobs[{i}] missing logprob in pair: {pair!r}")
            token_id = _coerce_sglang_token_id(token_value, field_name="input_top_logprobs")
            pairs.append((token_id, float(logprob)))
        pairs.sort(key=lambda item: item[1], reverse=True)
        out.append(pairs[:kk])
    return out


def normalize_sglang_generation_response(response: Any) -> dict[str, Any]:
    """Normalize SGLang offline Engine.generate output to Mint backend shape."""
    if not isinstance(response, dict):
        raise RuntimeError(f"SGLang generate returned {type(response).__name__}, expected dict")
    raw_ids = response.get("output_ids")
    if raw_ids is None:
        raise RuntimeError(f"SGLang generate response missing output_ids: keys={sorted(response.keys())}")
    try:
        token_ids = [int(x) for x in raw_ids]
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"Malformed SGLang output_ids: {raw_ids!r}") from e

    meta_info = response.get("meta_info") or {}
    if not isinstance(meta_info, dict):
        raise RuntimeError(f"SGLang generate meta_info must be a dict, got {type(meta_info).__name__}")

    stop_reason = _normalize_stop_reason(
        response.get("finish_reason")
        or meta_info.get("finish_reason")
        or meta_info.get("stop_reason")
    )
    logprobs = _normalize_generated_logprobs(meta_info)
    if logprobs is not None and len(logprobs) != len(token_ids):
        raise RuntimeError(
            f"SGLang output_token_logprobs length {len(logprobs)} does not match output_ids length {len(token_ids)}"
        )
    return {
        "token_ids": token_ids,
        "text": response.get("text"),
        "logprobs": logprobs,
        "stop_reason": stop_reason,
    }


class _WriterPreferringRWLock:
    """Writer-preferring read/write lock for the threaded SGLang actor.

    Mirrors the discipline of the vLLM backend's async `_AsyncRWLock`
    (mint_server/backend/inference/multinode_inference.py): many readers run
    concurrently, a writer runs exclusively, and queued writers block new
    readers so adapter mutations cannot be starved by a steady stream of
    generations.

    Generation paths take the read lock so SGLang batches concurrent requests
    internally; adapter load/unload, initialize, and shutdown take the write
    lock so they observe a quiesced engine (no generation in flight). Neither
    side is reentrant -- do not re-acquire from a context that already holds it.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0

    @contextmanager
    def read_locked(self) -> Iterator[None]:
        with self._cond:
            while self._writer or self._writers_waiting > 0:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write_locked(self) -> Iterator[None]:
        with self._cond:
            self._writers_waiting += 1
            try:
                while self._writer or self._readers > 0:
                    self._cond.wait()
                self._writer = True
            finally:
                self._writers_waiting -= 1
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()


class SGLangEngineActor:
    """Detached Ray actor wrapping a single-node SGLang offline Engine."""

    def __init__(
        self,
        *,
        model_name: str,
        model_path: str,
        actor_name: str,
        max_model_len: int,
        tp_size: int = 1,
        dtype: str = "auto",
        default_enable_lora: bool = True,
        default_max_lora_rank: int = 64,
        default_max_loaded_loras: int = 8,
        default_max_loras_per_batch: int | None = None,
        default_lora_target_modules: tuple[str, ...] | list[str] | None = None,
        default_mem_fraction_static: float = 0.4,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = str(model_name)
        self.model_path = str(model_path)
        self.actor_name = str(actor_name)
        self.max_model_len = int(max_model_len)
        self.tp_size = int(tp_size)
        self.dtype = str(dtype)
        self.default_enable_lora = bool(default_enable_lora)
        self.default_max_lora_rank = max(1, int(default_max_lora_rank))
        self.default_max_loaded_loras = max(1, int(default_max_loaded_loras))
        self.default_max_loras_per_batch = (
            max(1, int(default_max_loras_per_batch))
            if default_max_loras_per_batch is not None
            else None
        )
        self.default_lora_target_modules = tuple(default_lora_target_modules or SGLANG_DEFAULT_LORA_TARGET_MODULES)
        self.default_mem_fraction_static = float(default_mem_fraction_static)
        self.engine_kwargs = dict(engine_kwargs or {})
        self._engine: Any | None = None
        self._ready = False
        self._started_at = time.time()
        self._last_error: str | None = None
        self._session_adapters: dict[str, str] = {}
        self._adapter_paths: dict[str, str] = {}
        self._path_to_adapter_name: dict[str, str] = {}
        self._adapter_refcounts: dict[str, int] = {}
        self._engine_lock = _WriterPreferringRWLock()

    def is_ready(self) -> bool:
        return bool(self._ready)

    def health(self) -> dict[str, Any]:
        return {
            "actor_name": self.actor_name,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "max_model_len": self.max_model_len,
            "tp_size": self.tp_size,
            "ready": bool(self._ready),
            "started_at": self._started_at,
            "last_error": self._last_error,
            "loaded_adapter_count": len(self._adapter_paths),
            "session_adapter_count": len(self._session_adapters),
        }

    def get_rss_bytes(self) -> int:
        with open("/proc/self/statm", encoding="utf-8") as f:
            parts = f.read().strip().split()
        if len(parts) < 2:
            raise ValueError(f"unexpected /proc/self/statm format: {parts!r}")
        rss_pages = int(parts[1])
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return rss_pages * page_size

    def get_observability_binding(self) -> dict[str, Any]:
        import socket

        import ray

        from .ray_cluster.gpu_binding_helpers import gpu_bindings_from_ray_gpu_ids

        hostname = socket.gethostname()
        node_id = None
        node_ip = None
        try:
            node_id = str(ray.get_runtime_context().get_node_id())
        except Exception:
            node_id = None
        try:
            get_node_ip_address = getattr(getattr(ray, "util", None), "get_node_ip_address", None)
            if callable(get_node_ip_address):
                node_ip = str(get_node_ip_address())
        except Exception:
            node_ip = None
        gpu_bindings = gpu_bindings_from_ray_gpu_ids(hostname=hostname, node_id=node_id)
        return {
            "hostname": hostname,
            "node_id": node_id,
            "node_ip": node_ip,
            "gpu_indices": [binding["gpu_index"] for binding in gpu_bindings if "gpu_index" in binding],
            "gpu_bindings": gpu_bindings,
            "active_sessions": len(self._session_adapters),
            "loaded_adapter_count": len(self._adapter_paths),
            "ready_count": 1 if self._ready else 0,
        }

    def initialize(self) -> dict[str, Any]:
        with self._engine_lock.write_locked():
            if self._ready and self._engine is not None:
                return self.health()
            init_started = time.perf_counter()
            pycache_prefix = os.environ.get("PYTHONPYCACHEPREFIX") or os.environ.get("MINT_SGLANG_PYCACHE_PREFIX")
            logger.info(
                "SGLang actor initialize started backend=sglang actor=%s model=%s pycache_prefix=%s",
                self.actor_name,
                self.model_name,
                pycache_prefix,
            )
            _init_trace(
                "started",
                actor=self.actor_name,
                model=self.model_name,
                pycache_prefix=pycache_prefix,
            )
            stage_started = time.perf_counter()
            import_boundary = _apply_sglang_import_boundary()
            elapsed_s = max(0.0, time.perf_counter() - stage_started)
            total_s = max(0.0, time.perf_counter() - init_started)
            logger.info(
                "SGLang actor initialize stage complete backend=sglang actor=%s model=%s stage=import_boundary elapsed_s=%.3f total_s=%.3f removed_sys_path=%s removed_pythonpath=%s",
                self.actor_name,
                self.model_name,
                elapsed_s,
                total_s,
                import_boundary["removed_sys_path"],
                import_boundary["removed_pythonpath"],
            )
            _init_trace(
                "stage_complete",
                actor=self.actor_name,
                model=self.model_name,
                stage="import_boundary",
                elapsed_s=f"{elapsed_s:.3f}",
                total_s=f"{total_s:.3f}",
                removed_sys_path=import_boundary["removed_sys_path"],
                removed_pythonpath=import_boundary["removed_pythonpath"],
                pythonpath_entries=import_boundary["pythonpath_entries"],
            )
            stage_started = time.perf_counter()
            flashinfer_compat = _apply_sglang_flashinfer_kernel_compatibility(import_direct_kernel_modules=False)
            elapsed_s = max(0.0, time.perf_counter() - stage_started)
            total_s = max(0.0, time.perf_counter() - init_started)
            logger.info(
                "SGLang actor initialize stage complete backend=sglang actor=%s model=%s stage=flashinfer_compat elapsed_s=%.3f total_s=%.3f",
                self.actor_name,
                self.model_name,
                elapsed_s,
                total_s,
            )
            _init_trace(
                "stage_complete",
                actor=self.actor_name,
                model=self.model_name,
                stage="flashinfer_compat",
                elapsed_s=f"{elapsed_s:.3f}",
                total_s=f"{total_s:.3f}",
            )
            try:
                stage_started = time.perf_counter()
                from sglang.srt.entrypoints.engine import Engine
            except Exception as e:
                self._last_error = f"{type(e).__name__}: {e}"
                logger.warning(
                    "SGLang import failed backend=sglang actor=%s model=%s error_type=%s error=%s",
                    self.actor_name,
                    self.model_name,
                    type(e).__name__,
                    e,
                )
                raise SGLangBackendUnavailableError(
                    f"SGLang import failed in actor {self.actor_name}: {self._last_error}"
                ) from e
            elapsed_s = max(0.0, time.perf_counter() - stage_started)
            total_s = max(0.0, time.perf_counter() - init_started)
            logger.info(
                "SGLang actor initialize stage complete backend=sglang actor=%s model=%s stage=import_engine elapsed_s=%.3f total_s=%.3f",
                self.actor_name,
                self.model_name,
                elapsed_s,
                total_s,
            )
            _init_trace(
                "stage_complete",
                actor=self.actor_name,
                model=self.model_name,
                stage="import_engine",
                elapsed_s=f"{elapsed_s:.3f}",
                total_s=f"{total_s:.3f}",
            )
            stage_started = time.perf_counter()
            Engine = _engine_class_with_mint_scheduler_compatibility(Engine)
            elapsed_s = max(0.0, time.perf_counter() - stage_started)
            total_s = max(0.0, time.perf_counter() - init_started)
            logger.info(
                "SGLang actor initialize stage complete backend=sglang actor=%s model=%s stage=engine_class_patch elapsed_s=%.3f total_s=%.3f",
                self.actor_name,
                self.model_name,
                elapsed_s,
                total_s,
            )
            _init_trace(
                "stage_complete",
                actor=self.actor_name,
                model=self.model_name,
                stage="engine_class_patch",
                elapsed_s=f"{elapsed_s:.3f}",
                total_s=f"{total_s:.3f}",
            )

            stage_started = time.perf_counter()
            kwargs = {
                "model_path": self.model_path,
                "tokenizer_path": self.model_path,
                "dtype": self.dtype,
                "tp_size": int(self.tp_size),
                "context_length": int(self.max_model_len),
                "mem_fraction_static": _float_env(
                    "MINT_SGLANG_MEM_FRACTION_STATIC",
                    self.default_mem_fraction_static,
                ),
                "max_running_requests": _positive_int_env("MINT_SGLANG_MAX_RUNNING_REQUESTS", 2),
                "max_total_tokens": _positive_int_env(
                    "MINT_SGLANG_MAX_TOTAL_TOKENS",
                    max(1, int(self.max_model_len)),
                ),
                "max_prefill_tokens": _positive_int_env(
                    "MINT_SGLANG_MAX_PREFILL_TOKENS",
                    min(max(1, int(self.max_model_len)), 8192),
                ),
                "chunked_prefill_size": _positive_int_env("MINT_SGLANG_CHUNKED_PREFILL_SIZE", 8192),
                "disable_cuda_graph": _bool_env("MINT_SGLANG_DISABLE_CUDA_GRAPH", True),
                "disable_piecewise_cuda_graph": _bool_env("MINT_SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH", True),
                "disable_flashinfer_autotune": _bool_env("MINT_SGLANG_DISABLE_FLASHINFER_AUTOTUNE", True),
                "disable_custom_all_reduce": _bool_env("MINT_SGLANG_DISABLE_CUSTOM_ALL_REDUCE", True),
                "skip_server_warmup": _bool_env("MINT_SGLANG_SKIP_SERVER_WARMUP", True),
                "trust_remote_code": _bool_env("MINT_SGLANG_TRUST_REMOTE_CODE", False),
                "log_level": os.environ.get("MINT_SGLANG_LOG_LEVEL", "error"),
                "show_time_cost": _bool_env("MINT_SGLANG_SHOW_TIME_COST", False),
                "watchdog_timeout": _positive_int_env("MINT_SGLANG_WATCHDOG_TIMEOUT_S", 300),
            }
            kwargs.update(
                _sglang_lora_engine_kwargs(
                    default_enable_lora=self.default_enable_lora,
                    default_max_lora_rank=self.default_max_lora_rank,
                    default_max_loaded_loras=self.default_max_loaded_loras,
                    default_max_loras_per_batch=self.default_max_loras_per_batch,
                    default_lora_target_modules=self.default_lora_target_modules,
                )
            )
            kwargs.update(self.engine_kwargs)
            log_fields = _engine_init_log_fields(kwargs)
            elapsed_s = max(0.0, time.perf_counter() - stage_started)
            total_s = max(0.0, time.perf_counter() - init_started)
            logger.info(
                "SGLang actor initialize stage complete backend=sglang actor=%s model=%s stage=build_engine_kwargs elapsed_s=%.3f total_s=%.3f config=%s",
                self.actor_name,
                self.model_name,
                elapsed_s,
                total_s,
                log_fields,
            )
            _init_trace(
                "stage_complete",
                actor=self.actor_name,
                model=self.model_name,
                stage="build_engine_kwargs",
                elapsed_s=f"{elapsed_s:.3f}",
                total_s=f"{total_s:.3f}",
                config=log_fields,
            )
            try:
                logger.info(
                    "SGLang FlashInfer kernel compatibility backend=sglang actor=%s model=%s config=%s",
                    self.actor_name,
                    self.model_name,
                    flashinfer_compat,
                )
                stage_started = time.perf_counter()
                with _disable_config_actor_hydration_for_child_processes():
                    self._engine = Engine(**kwargs)
                engine_elapsed_s = max(0.0, time.perf_counter() - stage_started)
                total_s = max(0.0, time.perf_counter() - init_started)
                self._ready = True
                self._last_error = None
                logger.info(
                    "SGLang actor initialize complete backend=sglang actor=%s model=%s engine_elapsed_s=%.3f total_s=%.3f",
                    self.actor_name,
                    self.model_name,
                    engine_elapsed_s,
                    total_s,
                )
                _init_trace(
                    "complete",
                    actor=self.actor_name,
                    model=self.model_name,
                    engine_elapsed_s=f"{engine_elapsed_s:.3f}",
                    total_s=f"{total_s:.3f}",
                )
                return self.health()
            except Exception as e:
                self._ready = False
                self._last_error = f"{type(e).__name__}: {e}"
                total_s = max(0.0, time.perf_counter() - init_started)
                logger.warning(
                    "SGLang engine initialization failed backend=sglang actor=%s model=%s "
                    "error_type=%s error=%s total_s=%.3f config=%s",
                    self.actor_name,
                    self.model_name,
                    type(e).__name__,
                    e,
                    total_s,
                    log_fields,
                )
                _init_trace(
                    "failed",
                    actor=self.actor_name,
                    model=self.model_name,
                    error_type=type(e).__name__,
                    error=e,
                    total_s=f"{total_s:.3f}",
                )
                raise RuntimeError(f"SGLang engine initialization failed for {self.model_name}: {self._last_error}") from e

    def _load_lora_adapter(self, *, adapter_name: str, adapter_path: str) -> object:
        if self._engine is None:
            raise RuntimeError("SGLang engine is not initialized")
        load = getattr(self._engine, "load_lora_adapter", None)
        if not callable(load):
            raise SGLangUnsupportedFeatureError("SGLang Engine does not provide load_lora_adapter()")
        result = load(str(adapter_name), str(adapter_path))
        if not _sglang_result_success(result):
            detail = _sglang_result_error(result) or repr(result)
            raise RuntimeError(f"SGLang load_lora_adapter failed for {adapter_name}: {detail}")
        return result

    def _unload_lora_adapter(self, *, adapter_name: str) -> object:
        if self._engine is None:
            raise RuntimeError("SGLang engine is not initialized")
        unload = getattr(self._engine, "unload_lora_adapter", None)
        if not callable(unload):
            raise SGLangUnsupportedFeatureError("SGLang Engine does not provide unload_lora_adapter()")
        result = unload(str(adapter_name))
        if not _sglang_result_success(result):
            detail = _sglang_result_error(result) or repr(result)
            raise RuntimeError(f"SGLang unload_lora_adapter failed for {adapter_name}: {detail}")
        return result

    def add_lora_for_session_from_path(self, *, sampling_session_id: str, lora_path: str) -> dict[str, Any]:
        with self._engine_lock.write_locked():
            if not self._ready or self._engine is None:
                raise RuntimeError("SGLang engine is not initialized")
            session_id = str(sampling_session_id)
            canonical_path = canonical_peft_adapter_path(lora_path)
            validate_sglang_lora_adapter_supported(
                model_name=self.model_name,
                model_path=self.model_path,
                adapter_path=canonical_path,
            )
            adapter_name = _adapter_name_for_path(canonical_path)
            lora_int_id = 1

            existing_name = self._session_adapters.get(session_id)
            if existing_name is not None:
                existing_path = self._adapter_paths.get(existing_name)
                if existing_path == canonical_path:
                    return {
                        "sampling_session_id": session_id,
                        "adapter_name": existing_name,
                        "adapter_path": existing_path,
                        "lora_int_id": lora_int_id,
                        "loaded": False,
                        "reused": True,
                    }
                self._remove_session_locked(session_id)

            loaded_name = self._path_to_adapter_name.get(canonical_path)
            if loaded_name is not None:
                self._session_adapters[session_id] = loaded_name
                self._adapter_refcounts[loaded_name] = self._adapter_refcounts.get(loaded_name, 0) + 1
                return {
                    "sampling_session_id": session_id,
                    "adapter_name": loaded_name,
                    "adapter_path": canonical_path,
                    "lora_int_id": lora_int_id,
                    "loaded": False,
                    "reused": True,
                }

            started = time.perf_counter()
            try:
                self._load_lora_adapter(adapter_name=adapter_name, adapter_path=canonical_path)
            except Exception as e:
                self._session_adapters.pop(session_id, None)
                self._adapter_refcounts.pop(adapter_name, None)
                self._adapter_paths.pop(adapter_name, None)
                self._path_to_adapter_name.pop(canonical_path, None)
                raise RuntimeError(
                    f"Failed to load SGLang LoRA adapter for session {session_id} from {canonical_path}: "
                    f"{type(e).__name__}: {e}"
                ) from e

            self._adapter_paths[adapter_name] = canonical_path
            self._path_to_adapter_name[canonical_path] = adapter_name
            self._adapter_refcounts[adapter_name] = 1
            self._session_adapters[session_id] = adapter_name
            logger.info(
                "Loaded SGLang LoRA adapter session=%s adapter_name=%s path=%s elapsed_s=%.3f",
                session_id,
                adapter_name,
                canonical_path,
                max(0.0, time.perf_counter() - started),
            )
            return {
                "sampling_session_id": session_id,
                "adapter_name": adapter_name,
                "adapter_path": canonical_path,
                "lora_int_id": lora_int_id,
                "loaded": True,
                "reused": False,
            }

    def remove_session(self, sampling_session_id: str) -> dict[str, Any]:
        with self._engine_lock.write_locked():
            return self._remove_session_locked(str(sampling_session_id))

    def _remove_session_locked(self, sampling_session_id: str) -> dict[str, Any]:
        """Drop a session's adapter reference. Caller must hold the write lock."""
        session_id = str(sampling_session_id)
        adapter_name = self._session_adapters.get(session_id)
        if adapter_name is None:
            return {
                "sampling_session_id": session_id,
                "removed": False,
                "unloaded": False,
            }

        current_refcount = max(1, int(self._adapter_refcounts.get(adapter_name, 1)))
        refcount = current_refcount - 1
        if refcount > 0:
            self._session_adapters.pop(session_id, None)
            self._adapter_refcounts[adapter_name] = refcount
            return {
                "sampling_session_id": session_id,
                "adapter_name": adapter_name,
                "removed": True,
                "unloaded": False,
                "remaining_refcount": refcount,
            }

        adapter_path = self._adapter_paths.get(adapter_name)
        try:
            self._unload_lora_adapter(adapter_name=adapter_name)
        except Exception as e:
            logger.warning(
                "Failed to unload SGLang LoRA adapter session=%s adapter_name=%s path=%s: %s: %s",
                session_id,
                adapter_name,
                adapter_path,
                type(e).__name__,
                e,
            )
            raise

        self._session_adapters.pop(session_id, None)
        self._adapter_refcounts.pop(adapter_name, None)
        self._adapter_paths.pop(adapter_name, None)
        if adapter_path is not None:
            self._path_to_adapter_name.pop(adapter_path, None)
        logger.info(
            "Unloaded SGLang LoRA adapter session=%s adapter_name=%s path=%s",
            session_id,
            adapter_name,
            adapter_path,
        )
        return {
            "sampling_session_id": session_id,
            "adapter_name": adapter_name,
            "adapter_path": adapter_path,
            "removed": True,
            "unloaded": True,
            "remaining_refcount": 0,
        }

    def generate_base(
        self,
        *,
        prompt_ids: list[int],
        request_id: str,
        max_tokens: int,
        stop: object | None = None,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
        logprobs: bool = True,
        sampling_session_id: str | None = "__base__",
        prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> dict[str, Any]:
        with self._engine_lock.read_locked():
            if not self._ready or self._engine is None:
                raise RuntimeError("SGLang engine is not initialized")
            if len(prompt_ids) + int(max_tokens) > self.max_model_len:
                raise ValueError(
                    f"Prompt+max_tokens length {len(prompt_ids) + int(max_tokens)} exceeds "
                    f"max_model_len {self.max_model_len} for model {self.model_name}"
                )

            adapter_name: str | None = None
            if sampling_session_id not in (None, "__base__"):
                adapter_name = self._session_adapters.get(str(sampling_session_id))
                if adapter_name is None:
                    raise RuntimeError(
                        f"No SGLang LoRA adapter loaded for sampling_session_id={sampling_session_id!r}; "
                        "call add_lora_for_session_from_path() before generate()"
                    )

            generate_kwargs: dict[str, Any] = {}
            if adapter_name is not None:
                generate_kwargs["lora_path"] = adapter_name

            started = time.perf_counter()
            response = self._engine.generate(
                input_ids=[int(x) for x in prompt_ids],
                sampling_params=_build_sampling_params(
                    max_tokens=int(max_tokens),
                    stop=stop,
                    temperature=float(temperature),
                    top_k=int(top_k),
                    top_p=float(top_p),
                ),
                return_logprob=bool(logprobs or prompt_logprobs or int(topk_prompt_logprobs) > 0),
                logprob_start_len=0,
                top_logprobs_num=max(0, int(topk_prompt_logprobs)),
                rid=str(request_id),
                **generate_kwargs,
            )
            normalized = normalize_sglang_generation_response(response)
            if bool(prompt_logprobs):
                normalized["prompt_logprobs"] = normalize_sglang_prompt_logprobs_response(
                    response,
                    prompt_ids=[int(x) for x in prompt_ids],
                )
            if int(topk_prompt_logprobs) > 0:
                normalized["topk_prompt_logprobs"] = normalize_sglang_prompt_topk_response(
                    response,
                    prompt_ids=[int(x) for x in prompt_ids],
                    k=int(topk_prompt_logprobs),
                )
            normalized["_timing_total_s"] = max(0.0, time.perf_counter() - started)
            normalized["request_id"] = str(request_id)
            if adapter_name is not None:
                normalized["_sglang_lora_adapter_name"] = adapter_name
            return normalized

    def compute_prompt_logprobs(
        self,
        *,
        prompt_ids: list[int],
        request_id: str,
        sampling_session_id: str | None = "__base__",
    ) -> list[float | None]:
        if not prompt_ids:
            return []
        response = self.generate_base(
            prompt_ids=[int(x) for x in prompt_ids],
            request_id=str(request_id),
            max_tokens=1,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            logprobs=False,
            sampling_session_id=sampling_session_id,
            prompt_logprobs=True,
            topk_prompt_logprobs=0,
        )
        raw = response.get("prompt_logprobs")
        if not isinstance(raw, list):
            raise RuntimeError("SGLang prompt_logprobs normalization did not return a list")
        return list(raw)

    def compute_prompt_topk(
        self,
        *,
        prompt_ids: list[int],
        request_id: str,
        k: int,
        sampling_session_id: str | None = "__base__",
    ) -> list[list[tuple[int, float]] | None]:
        kk = int(k)
        if not prompt_ids:
            return []
        if kk <= 0:
            return [None] * len(prompt_ids)
        response = self.generate_base(
            prompt_ids=[int(x) for x in prompt_ids],
            request_id=str(request_id),
            max_tokens=1,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            logprobs=False,
            sampling_session_id=sampling_session_id,
            prompt_logprobs=False,
            topk_prompt_logprobs=kk,
        )
        raw = response.get("topk_prompt_logprobs")
        if not isinstance(raw, list):
            raise RuntimeError("SGLang topk_prompt_logprobs normalization did not return a list")
        return list(raw)

    def abort(self, request_id: str) -> bool:
        engine = self._engine
        abort = getattr(engine, "abort", None)
        if callable(abort):
            abort(str(request_id))
            return True
        return False

    def shutdown(self) -> dict[str, Any]:
        with self._engine_lock.write_locked():
            engine = self._engine
            self._engine = None
            self._ready = False
            self._session_adapters.clear()
            self._adapter_paths.clear()
            self._path_to_adapter_name.clear()
            self._adapter_refcounts.clear()
            if engine is not None:
                shutdown = getattr(engine, "shutdown", None)
                if callable(shutdown):
                    shutdown()
            return self.health()
