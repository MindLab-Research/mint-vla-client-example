"""Process-startup patches for Ray/vLLM worker subprocesses.

Python automatically imports `sitecustomize` (if present on sys.path) on
interpreter startup. We use this to patch code paths that run in vLLM
subprocesses spawned with the `spawn` method, where in-process monkey patches
from the parent process do not propagate.

This file activates only when explicitly enabled via environment variables
propagated into vLLM worker processes.
"""

from __future__ import annotations

import importlib.util
import json
import multiprocessing.spawn as _mp_spawn
import os
import sys
import sysconfig
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


_DRIVER_ONLY_RAY_RUNTIME_ENV_KEYS = (
    "MINT_RAY_TEMP_DIR",
    "MINT_RAY_NODE_IP_ADDRESS",
    "RAY_TMPDIR",
    "TMPDIR",
    "TMP",
    "TEMP",
    "RAY_ADDRESS",
    "RAY_CLIENT_ADDRESS",
    "MINT_RAY_CLIENT_ADDRESS",
)


def _is_ray_worker_bootstrap_process(argv: list[str] | tuple[str, ...] | None = None) -> bool:
    args = sys.argv if argv is None else argv
    for arg in args:
        value = str(arg)
        if value == "ray._private.workers.default_worker":
            return True
        if value.replace("\\", "/").endswith("/ray/_private/workers/default_worker.py"):
            return True
    return False


def _is_ray_actor_or_worker_process_environment(
    environ: os._Environ[str] | dict[str, str] | None = None,
) -> bool:
    target = os.environ if environ is None else environ
    for key in (
        "RAY_ACTOR_ID",
        "RAY_JOB_ID",
        "RAY_WORKER_ID",
        "RAY_WORKER_MODE",
        "RAY_RAYLET_PID",
    ):
        if str(target.get(key, "")).strip():
            return True
    return False


def _is_vllm_worker_patch_environment(
    environ: os._Environ[str] | dict[str, str] | None = None,
) -> bool:
    target = os.environ if environ is None else environ
    return _env_flag_from_mapping(target, "MINT_ENABLE_VLLM_IMPORT_PATCHES", default=False)


def _should_sanitize_ray_worker_environment(
    environ: os._Environ[str] | dict[str, str] | None = None,
    argv: list[str] | tuple[str, ...] | None = None,
) -> bool:
    return (
        _is_ray_worker_bootstrap_process(argv)
        or _is_ray_actor_or_worker_process_environment(environ)
        or _is_vllm_worker_patch_environment(environ)
    )


def _sanitize_ray_worker_bootstrap_process_environment(
    environ: os._Environ[str] | dict[str, str] | None = None,
    argv: list[str] | tuple[str, ...] | None = None,
) -> None:
    if not _should_sanitize_ray_worker_environment(environ, argv):
        return
    target = os.environ if environ is None else environ
    for key in _DRIVER_ONLY_RAY_RUNTIME_ENV_KEYS:
        target.pop(key, None)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_flag_from_mapping(
    environ: os._Environ[str] | dict[str, str],
    name: str,
    *,
    default: bool = False,
) -> bool:
    value = environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


_sanitize_ray_worker_bootstrap_process_environment()


def _is_cv2_package_dir(path: object) -> bool:
    if not isinstance(path, str) or not path:
        return False
    norm = os.path.normpath(path)
    return norm.endswith("/site-packages/cv2")


def _is_cv2_typing_file(path: object) -> bool:
    if not isinstance(path, str) or not path:
        return False
    norm = os.path.normpath(path)
    return norm.endswith("/site-packages/cv2/typing/__init__.py")


def _sanitize_paths(paths: list[str]) -> list[str]:
    out: list[str] = []
    for p in paths:
        if not p:
            continue
        if _is_cv2_package_dir(p):
            continue
        if p in out:
            continue
        out.append(p)
    return out


def _sanitize_vllm_worker_pythonpath(raw: str | None) -> str:
    if raw is None:
        return ""
    try:
        from mint_server.runtime_env import sanitize_worker_pythonpath

        return sanitize_worker_pythonpath(
            raw,
            env_root=os.environ.get("PFS_RUNTIME_ENV_ROOT"),
        )
    except Exception:
        return ":".join(p for p in str(raw).split(":") if p)


def _sanitize_vllm_ray_runtime_env_dict(payload: object) -> object:
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    env_vars = out.get("env_vars")
    if isinstance(env_vars, dict):
        cleaned = dict(env_vars)
        for key in _DRIVER_ONLY_RAY_RUNTIME_ENV_KEYS:
            cleaned[key] = ""
        out["env_vars"] = cleaned
    return out


def _patch_ray_runtime_env_to_dict_drop_driver_attach_hints() -> None:
    try:
        from ray.runtime_env import RuntimeEnv
    except Exception:
        return

    original = getattr(RuntimeEnv, "to_dict", None)
    if not callable(original) or getattr(
        original, "_mint_drops_driver_attach_hints", False
    ):
        return

    def to_dict(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return _sanitize_vllm_ray_runtime_env_dict(
            original(self, *args, **kwargs)
        )

    to_dict._mint_drops_driver_attach_hints = True  # type: ignore[attr-defined]
    RuntimeEnv.to_dict = to_dict  # type: ignore[method-assign]


def _strip_host_only_sys_path_entries(paths: list[str]) -> list[str]:
    env_root = os.environ.get("PFS_RUNTIME_ENV_ROOT")
    if not env_root:
        return paths
    try:
        from mint_server.runtime_env import host_only_pythonpath_entries

        excluded = {
            os.path.normcase(os.path.abspath(path))
            for path in host_only_pythonpath_entries(env_root)
        }
    except Exception:
        return paths
    return [
        path
        for path in paths
        if not path or os.path.normcase(os.path.abspath(path)) not in excluded
    ]


def _preferred_torch_lib_dirs() -> list[str]:
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    env_root = os.environ.get("PFS_RUNTIME_ENV_ROOT", "").strip()
    candidates: list[str] = []
    if env_root:
        candidates.extend(
            [
                os.path.join(env_root, "host-venv", "lib", pyver, "site-packages", "torch", "lib"),
                os.path.join(env_root, "site-packages", "torch", "lib"),
            ]
        )
    candidates.append(f"/usr/local/lib/{pyver}/dist-packages/torch/lib")

    out: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        norm = os.path.normcase(os.path.abspath(path))
        if norm in seen or not os.path.isdir(path):
            continue
        seen.add(norm)
        out.append(path)
    return out


def _patch_torch_ld_library_path() -> None:
    preferred = _preferred_torch_lib_dirs()
    if not preferred:
        return

    blocked = {
        os.path.normcase("/usr/local/lib/python3.10/dist-packages/torch/lib"),
        os.path.normcase("/usr/local/lib/python3.10/site-packages/torch/lib"),
    }
    current = [
        p
        for p in os.environ.get("LD_LIBRARY_PATH", "").split(":")
        if p and os.path.normcase(os.path.abspath(p)) not in blocked
    ]
    os.environ["LD_LIBRARY_PATH"] = ":".join(_sanitize_paths([*preferred, *current]))


def _patch_multiprocessing_executable() -> None:
    try:
        import multiprocessing

        executable = os.environ.get("MINT_VLLM_CHILD_PYTHON_EXECUTABLE", "").strip() or sys.executable
        if executable and os.path.exists(executable):
            multiprocessing.set_executable(executable)
    except Exception:
        pass


def _maybe_log_vllm_child_startup() -> None:
    try:
        import multiprocessing

        proc_name = multiprocessing.current_process().name
    except Exception:
        proc_name = ""
    if not proc_name.startswith(("EngineCore_", "VllmWorker", "SpawnProcess")):
        return

    payload: dict[str, object] = {
        "proc_name": proc_name,
        "python": sys.executable,
        "sys_path_head": sys.path[:10],
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    try:
        spec = importlib.util.find_spec("vllm")
        payload["vllm_spec"] = getattr(spec, "origin", None)
    except Exception as e:
        payload["vllm_spec_error"] = repr(e)
    try:
        import vllm  # type: ignore

        payload["vllm_file"] = getattr(vllm, "__file__", None)
    except Exception as e:
        payload["vllm_import_error"] = repr(e)
    try:
        import vllm._C as vllm_c  # type: ignore

        payload["vllm_c_file"] = getattr(vllm_c, "__file__", None)
    except Exception as e:
        payload["vllm_c_import_error"] = repr(e)
    print(f"[sitecustomize:vllm_child_startup] {json.dumps(payload, default=str)}", flush=True)


def _patch_cv2_typing_shadow() -> None:
    """Prevent accidental import of `cv2/typing` as top-level `typing`.

    Some worker subprocesses inherit polluted sys.path containing
    `.../site-packages/cv2`, which can shadow stdlib `typing.py` and crash with
    `ImportError: libxcb.so.1`.
    """
    if _env_flag("MINT_DISABLE_CV2_TYPING_PATCH", default=False):
        return

    # 1) Clean current process paths.
    _patch_torch_ld_library_path()
    _patch_multiprocessing_executable()
    sanitized_pythonpath = _sanitize_vllm_worker_pythonpath(os.environ.get("PYTHONPATH"))
    if sanitized_pythonpath:
        os.environ["PYTHONPATH"] = sanitized_pythonpath
    sys.path[:] = _sanitize_paths(_strip_host_only_sys_path_entries(list(sys.path)))

    raw_py = os.environ.get("PYTHONPATH", "")
    if raw_py:
        parts = [p.strip() for p in raw_py.split(":")]
        os.environ["PYTHONPATH"] = ":".join(_sanitize_paths(_strip_host_only_sys_path_entries(parts)))

    # 2) Ensure multiprocessing spawn does not reintroduce bad paths.
    orig = _mp_spawn.get_preparation_data
    if not getattr(orig, "__mint_cv2_typing_patched__", False):

        def _mint_get_preparation_data(*args, **kwargs):
            data = orig(*args, **kwargs)
            raw = data.get("sys_path")
            if isinstance(raw, list):
                data["sys_path"] = _sanitize_paths(raw)
            return data

        _mint_get_preparation_data.__mint_cv2_typing_patched__ = True  # type: ignore[attr-defined]
        _mp_spawn.get_preparation_data = _mint_get_preparation_data  # type: ignore[assignment]

    # 3) Pin top-level typing module to stdlib implementation.
    try:
        import typing as _typing  # noqa: F401
    except Exception:
        _typing = None  # type: ignore[assignment]

    if _typing is None or _is_cv2_typing_file(getattr(_typing, "__file__", None)):
        stdlib_typing = os.path.join(sysconfig.get_path("stdlib"), "typing.py")
        spec = importlib.util.spec_from_file_location("typing", stdlib_typing)
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules["typing"] = mod


def _patch_vllm_fused_moe_slice_for_fully_sharded_loras() -> None:
    import vllm.lora.layers.fused_moe as fused_moe_mod

    def _patch_cls(cls: type) -> None:
        original = getattr(cls, "_slice_w13_a", None)
        if original is None:
            raise RuntimeError(f"vLLM class {cls.__name__} has no _slice_w13_a")
        if getattr(original, "_mint_patched_fully_sharded", False):
            return

        def _slice_w13_a(self, w13_lora_a):  # type: ignore[no-untyped-def]
            if self.tp_size == 1 or not self.fully_sharded:
                return w13_lora_a

            if getattr(w13_lora_a, "ndim", None) != 3:
                raise RuntimeError(
                    "vLLM fused_moe _slice_w13_a expects 3D w13_lora_a: "
                    f"w13_lora_a.ndim={getattr(w13_lora_a, 'ndim', None)}"
                )

            # Robust rank-dimension inference for K2: the LoRA rank is the smallest
            # non-expert dimension (typically R=64). Fully-sharded LoRA shards that
            # rank across TP, so each rank sees R_local = R / tp_size.
            d1, d2 = int(w13_lora_a.shape[1]), int(w13_lora_a.shape[2])
            rank_dim = 1 if d1 <= d2 else 2
            global_rank = min(d1, d2)
            if global_rank <= 0:
                raise RuntimeError(
                    "vLLM fused_moe _slice_w13_a invalid global_rank: "
                    f"w13_lora_a.shape={tuple(w13_lora_a.shape)}"
                )
            if global_rank > 4096:
                raise RuntimeError(
                    "vLLM fused_moe _slice_w13_a suspicious global_rank (refusing to slice): "
                    f"w13_lora_a.shape={tuple(w13_lora_a.shape)} tp_size={self.tp_size}"
                )
            if global_rank % int(self.tp_size) != 0:
                # Already sharded (or unexpected rank); do not touch.
                return w13_lora_a

            local_rank = global_rank // int(self.tp_size)
            start_idx = int(self.tp_rank) * local_rank
            end_idx = (int(self.tp_rank) + 1) * local_rank
            if rank_dim == 1:
                return w13_lora_a[:, start_idx:end_idx, :]
            return w13_lora_a[:, :, start_idx:end_idx]

        _slice_w13_a._mint_patched_fully_sharded = True  # type: ignore[attr-defined]
        cls._slice_w13_a = _slice_w13_a  # type: ignore[method-assign]

    for name in ("FusedMoEWithLoRA", "FusedMoE3DWithLoRA"):
        cls = getattr(fused_moe_mod, name, None)
        if cls is None:
            raise RuntimeError(f"vLLM fused_moe is missing class {name}")
        _patch_cls(cls)


def _patch_vllm_ray_executor_sample_tokens_no_compiled_dag() -> None:
    """Bypass Ray compiled DAG for sample_tokens on PP=1 when explicitly enabled.

    Keep the Ray backend and TP>1 placement, but avoid the compiled-DAG channel
    path for the final execute_model/sample_tokens sequence. This is an
    experiment-only escape hatch for cases where EngineCore wedges in
    shared_memory_channel read/write during generation.
    """

    if not _env_flag("MINT_VLLM_RAY_EXECUTOR_NO_COMPILED_DAG_SAMPLE", default=False):
        return

    import threading
    from concurrent.futures import Future

    import vllm.v1.executor.ray_executor as ray_exec_mod

    cls = getattr(ray_exec_mod, "RayDistributedExecutor", None)
    if cls is None:
        raise RuntimeError("vLLM missing RayDistributedExecutor")

    original = getattr(cls, "sample_tokens", None)
    if original is None:
        raise RuntimeError("vLLM RayDistributedExecutor missing sample_tokens")
    if getattr(original, "_mint_patched_no_compiled_dag_sample", False):
        return

    completed_none = getattr(ray_exec_mod, "COMPLETED_NONE_FUTURE")

    def _run_plain_collective(self, grammar_output):  # type: ignore[no-untyped-def]
        scheduler_output = self.scheduler_output
        if scheduler_output is None:
            return None

        self.scheduler_output = None
        execute_out = self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            non_block=False,
        )
        # For sampler models this should be None on each worker; keep the path
        # strict instead of adding any fallback behavior.
        if self.uses_sampler and scheduler_output.total_num_scheduled_tokens:
            if any(x is not None for x in execute_out):
                raise RuntimeError(
                    "Ray executor no-compiled-dag sample expected execute_model "
                    "to return only None before sample_tokens"
                )

        sample_out = self.collective_rpc(
            "sample_tokens",
            args=(grammar_output,),
            non_block=False,
        )
        if self.has_connector:
            if self.kv_output_aggregator is None:
                raise RuntimeError(
                    "Ray executor no-compiled-dag sample requires "
                    "kv_output_aggregator when connector is enabled"
                )
            return self.kv_output_aggregator.aggregate(sample_out)
        return sample_out[0]

    def sample_tokens(self, grammar_output, non_block: bool = False):  # type: ignore[no-untyped-def]
        if self.parallel_config.pipeline_parallel_size != 1:
            return original(self, grammar_output, non_block=non_block)

        if self.scheduler_output is None:
            return completed_none if non_block else None

        if not getattr(self, "_mint_logged_no_cdag_sample", False):
            print(
                "[mint patch] RayDistributedExecutor.sample_tokens using "
                "no-compiled-dag fallback",
                file=sys.stderr,
                flush=True,
            )
            self._mint_logged_no_cdag_sample = True

        if not non_block:
            return _run_plain_collective(self, grammar_output)

        fut: Future = Future()

        def _runner() -> None:
            try:
                fut.set_result(_run_plain_collective(self, grammar_output))
            except BaseException as e:
                fut.set_exception(e)

        threading.Thread(
            target=_runner,
            name="mint-ray-no-cdag-sample",
            daemon=True,
        ).start()
        return fut

    sample_tokens._mint_patched_no_compiled_dag_sample = True  # type: ignore[attr-defined]
    cls.sample_tokens = sample_tokens  # type: ignore[method-assign]


def _patch_vllm_ray_executor_use_explicit_cluster_address() -> None:
    """Prevent vLLM Ray executor from silently starting a local Ray head.

    In our server, multinode vLLM actors already run inside a managed Ray
    cluster. If vLLM initializes its Ray executor with no address, Ray starts a
    standalone local head inside EngineCore, which hides the real failure behind
    a nested cluster. Fail closed unless Mint's explicit GCS address is
    available, with legacy `RAY_ADDRESS` accepted only as a fallback.
    """

    import vllm.v1.executor.ray_executor as ray_exec_mod
    import vllm.v1.executor.ray_utils as ray_utils_mod

    original = getattr(ray_utils_mod, "initialize_ray_cluster", None)
    if original is None:
        raise RuntimeError("vLLM missing initialize_ray_cluster")
    if getattr(original, "_mint_patched_explicit_cluster_address", False):
        return

    def initialize_ray_cluster(parallel_config, ray_address=None):  # type: ignore[no-untyped-def]
        mint_gcs_address = os.environ.get("MINT_RAY_GCS_ADDRESS", "").strip()
        _sanitize_ray_worker_bootstrap_process_environment()
        addr = ray_address
        if addr is None or (isinstance(addr, str) and addr.strip() in {"", "auto"}):
            env_addr = mint_gcs_address
            if not env_addr:
                ray_addr = os.environ.get("RAY_ADDRESS", "").strip()
                if ray_addr and not ray_addr.startswith("ray://"):
                    env_addr = ray_addr
            if not env_addr:
                raise RuntimeError(
                    "vLLM RayDistributedExecutor requires explicit MINT_RAY_GCS_ADDRESS; "
                    "refusing to start nested local Ray inside EngineCore"
                )
            addr = env_addr

        # Reuse Mint's Ray init helper so EngineCore children inherit Ray Client
        # attach hints and temp-dir/node-ip overrides instead of falling back to a
        # fresh local-driver attach inside the worker process.
        import ray
        from mint_server.ray_utils import init_ray as mint_init_ray

        if not ray.is_initialized() and not _is_ray_worker_bootstrap_process():
            mint_init_ray(
                address=addr,
                runtime_env=getattr(parallel_config, "ray_runtime_env", None),
            )
        return original(parallel_config, ray_address=addr)

    initialize_ray_cluster._mint_patched_explicit_cluster_address = True  # type: ignore[attr-defined]
    ray_utils_mod.initialize_ray_cluster = initialize_ray_cluster  # type: ignore[assignment]
    ray_exec_mod.initialize_ray_cluster = initialize_ray_cluster  # type: ignore[assignment]


def _patch_vllm_skip_dummy_lora_setup_when_inactive() -> None:
    """Avoid expensive dummy-LoRA warmup during profiling runs.

    vLLM has changed the `maybe_dummy_run_with_lora(...)` signature across releases.
    Older builds passed `activate_lora: bool`; newer ones pass
    `num_active_loras: int` plus `mapping_type`. In both cases, the inactive path
    should skip dummy-LoRA setup entirely. If we misread the arguments, vLLM ends
    up creating MoE dummy LoRAs during profiling, which is extremely expensive for
    large models (e.g. K2) and can stall engine startup after weights finish loading.

    This patch keeps the existing behavior when LoRA is actually activated, but
    skips dummy-LoRA setup/selection when the current call is inactive.
    """

    from contextlib import contextmanager
    import inspect

    import vllm.v1.worker.lora_model_runner_mixin as mixin_mod

    cls = getattr(mixin_mod, "LoRAModelRunnerMixin", None)
    if cls is None:
        raise RuntimeError("vLLM missing LoRAModelRunnerMixin")

    original = getattr(cls, "maybe_dummy_run_with_lora", None)
    if original is None:
        raise RuntimeError(
            "vLLM LoRAModelRunnerMixin missing maybe_dummy_run_with_lora"
        )
    if getattr(original, "_mint_patched_skip_dummy_inactive", False):
        return
    original_sig = inspect.signature(original)

    @contextmanager
    def maybe_dummy_run_with_lora(  # type: ignore[no-untyped-def]
        self,
        lora_config,
        num_scheduled_tokens,
        num_sampled_tokens,
        *args,
        **kwargs,
    ):
        bound = original_sig.bind_partial(
            self,
            lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            *args,
            **kwargs,
        )
        arguments = bound.arguments
        remove_lora = bool(arguments.get("remove_lora", True))
        if "activate_lora" in original_sig.parameters:
            inactive = not bool(arguments.get("activate_lora", True))
        elif "num_active_loras" in original_sig.parameters:
            inactive = int(arguments.get("num_active_loras", 0)) <= 0
        else:
            raise RuntimeError(
                "vLLM maybe_dummy_run_with_lora signature missing both "
                "`activate_lora` and `num_active_loras`"
            )

        if inactive:
            if lora_config is not None and remove_lora:
                self.maybe_remove_all_loras(lora_config)
            yield
            return

        with original(
            self,
            lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            *args,
            **kwargs,
        ):
            yield

    maybe_dummy_run_with_lora._mint_patched_skip_dummy_inactive = True  # type: ignore[attr-defined]
    cls.maybe_dummy_run_with_lora = maybe_dummy_run_with_lora  # type: ignore[method-assign]


def _patch_vllm_profile_run_disable_dummy_active_loras() -> None:
    """Disable dummy active-LoRA profiling during worker startup.

    K2 on the current vLLM build crashes during profile-time fused MoE LoRA
    startup in `fused_moe_lora_op.py` with illegal CUDA memory access. This
    happens before any real adapter is loaded, while `_dummy_run(...)` is
    capturing a synthetic `num_active_loras > 0` path. Keep real LoRA behavior
    intact, but force profile-time dummy runs down the no-active-LoRA path.
    """

    import vllm.v1.worker.gpu_model_runner as gpu_model_runner_mod

    cls = getattr(gpu_model_runner_mod, "GPUModelRunner", None)
    if cls is None:
        raise RuntimeError("vLLM missing GPUModelRunner")

    import inspect

    original = getattr(cls, "_dummy_run", None)
    if original is None:
        raise RuntimeError("vLLM GPUModelRunner missing _dummy_run")
    if getattr(original, "_mint_patched_disable_profile_dummy_loras", False):
        return
    try:
        original_sig = inspect.signature(original)
    except Exception:
        original_sig = None

    def _supports_kw(name: str) -> bool:
        if original_sig is None:
            return True
        if name in original_sig.parameters:
            return True
        return any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in original_sig.parameters.values()
        )

    def _dummy_run(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        is_profile = bool(kwargs.get("is_profile", False))
        activate_lora = bool(kwargs.get("activate_lora", False))
        hf_config = getattr(getattr(self, "model_config", None), "hf_config", None)
        num_experts = (
            getattr(hf_config, "num_experts", 0) if hf_config is not None else 0
        )
        is_moe = bool(num_experts and int(num_experts) > 1)
        if is_moe and not is_profile and activate_lora:
            return original(self, *args, **kwargs)

        if _supports_kw("num_active_loras"):
            kwargs["num_active_loras"] = 0
        else:
            kwargs.pop("num_active_loras", None)
        kwargs.pop("activate_lora", None)
        original_lora_config = getattr(self, "lora_config", None)
        old_bypass = os.environ.get("MINT_VLLM_BYPASS_FUSED_MOE_LORA_OP")
        old_bypass_dense = os.environ.get("MINT_VLLM_BYPASS_DUMMY_LORA_EMBEDDING_OP")
        try:
            if original_lora_config is not None:
                self.maybe_remove_all_loras(original_lora_config)
            self.lora_config = None
            os.environ["MINT_VLLM_BYPASS_FUSED_MOE_LORA_OP"] = "1"
            os.environ["MINT_VLLM_BYPASS_DUMMY_LORA_EMBEDDING_OP"] = "1"
            return original(self, *args, **kwargs)
        finally:
            self.lora_config = original_lora_config
            if old_bypass is None:
                os.environ.pop("MINT_VLLM_BYPASS_FUSED_MOE_LORA_OP", None)
            else:
                os.environ["MINT_VLLM_BYPASS_FUSED_MOE_LORA_OP"] = old_bypass
            if old_bypass_dense is None:
                os.environ.pop("MINT_VLLM_BYPASS_DUMMY_LORA_EMBEDDING_OP", None)
            else:
                os.environ["MINT_VLLM_BYPASS_DUMMY_LORA_EMBEDDING_OP"] = (
                    old_bypass_dense
                )

    _dummy_run._mint_patched_disable_profile_dummy_loras = True  # type: ignore[attr-defined]
    cls._dummy_run = _dummy_run  # type: ignore[method-assign]


def _patch_vllm_fused_moe_lora_profile_noop() -> None:
    """Skip fused MoE LoRA custom op during startup/profile dummy runs.

    K2 startup currently trips CUDA illegal-address faults inside
    `fused_moe_lora_op._fused_moe_lora(...)`. During profile-only dummy runs the
    op's numeric result is irrelevant; only shape flow matters. Under the scoped
    env flag set by `_patch_vllm_profile_run_disable_dummy_active_loras()`,
    convert the op into a no-op so startup can continue.
    """

    try:
        import importlib

        op = importlib.import_module("vllm.lora.ops.triton_ops.fused_moe_lora_op")
        triton_ops = importlib.import_module("vllm.lora.ops.triton_ops")
    except Exception:
        return

    def _wrap_noop(original):  # type: ignore[no-untyped-def]
        if not callable(original) or getattr(original, "_mint_profile_noop", False):
            return original

        def wrapped(*args, **kwargs):  # type: ignore[no-untyped-def]
            if not _env_flag("MINT_VLLM_BYPASS_FUSED_MOE_LORA_OP", default=False):
                return original(*args, **kwargs)
            return None

        wrapped._mint_profile_noop = True  # type: ignore[attr-defined]
        return wrapped

    for name in (
        "_fused_moe_lora",
        "_fused_moe_lora_shrink",
        "_fused_moe_lora_expand",
        "fused_moe_lora",
        "fused_moe_lora_shrink",
        "fused_moe_lora_expand",
    ):
        if hasattr(op, name):
            setattr(op, name, _wrap_noop(getattr(op, name)))
        if hasattr(triton_ops, name):
            setattr(triton_ops, name, _wrap_noop(getattr(triton_ops, name)))

    try:
        punica_gpu = importlib.import_module("vllm.lora.punica_wrapper.punica_gpu")
    except Exception:
        return
    for name in ("fused_moe_lora",):
        if hasattr(punica_gpu, name):
            setattr(punica_gpu, name, _wrap_noop(getattr(punica_gpu, name)))

    add_lora_embedding = getattr(punica_gpu, "PunicaWrapperGPU", None)
    add_lora_embedding = getattr(add_lora_embedding, "add_lora_embedding", None)
    if callable(add_lora_embedding) and not getattr(
        add_lora_embedding, "_mint_profile_noop", False
    ):

        def wrapped_add_lora_embedding(
            self, y, x, lora_b_stacked, add_inputs=True, **kwargs
        ):  # type: ignore[no-untyped-def]
            if _env_flag("MINT_VLLM_BYPASS_DUMMY_LORA_EMBEDDING_OP", default=False):
                return None
            return add_lora_embedding(
                self, y, x, lora_b_stacked, add_inputs=add_inputs, **kwargs
            )

        wrapped_add_lora_embedding._mint_profile_noop = True  # type: ignore[attr-defined]
        punica_gpu.PunicaWrapperGPU.add_lora_embedding = wrapped_add_lora_embedding  # type: ignore[method-assign]

    try:
        vocab_layer_mod = importlib.import_module(
            "vllm.lora.layers.vocal_parallel_embedding"
        )
        vocab_cls = getattr(vocab_layer_mod, "VocabParallelEmbeddingWithLoRA", None)
        vocab_forward = getattr(vocab_cls, "forward", None)
    except Exception:
        return
    if callable(vocab_forward) and not getattr(
        vocab_forward, "_mint_profile_noop", False
    ):

        def wrapped_vocab_forward(self, x):  # type: ignore[no-untyped-def]
            if _env_flag("MINT_VLLM_BYPASS_DUMMY_LORA_EMBEDDING_OP", default=False):
                return self.base_layer.forward(x)
            return vocab_forward(self, x)

        wrapped_vocab_forward._mint_profile_noop = True  # type: ignore[attr-defined]
        vocab_cls.forward = wrapped_vocab_forward  # type: ignore[method-assign]


def _patch_vllm_profile_run_scope_bypass_fused_moe_lora() -> None:
    """Scope fused-MoE-LoRA bypass to vLLM startup profiling.

    `GPUModelRunner.profile_run()` is the stable startup path that exercises the
    fused MoE LoRA kernels before any real user request is served. Set a narrow
    env guard around that call so `_fused_moe_lora(...)` can be a no-op only
    during profiling, while real inference/training retains LoRA behavior.
    """

    import vllm.v1.worker.gpu_model_runner as gpu_model_runner_mod

    cls = getattr(gpu_model_runner_mod, "GPUModelRunner", None)
    if cls is None:
        raise RuntimeError("vLLM missing GPUModelRunner")

    original = getattr(cls, "profile_run", None)
    if not callable(original) or getattr(
        original, "_mint_profile_scope_bypass", False
    ):
        return

    def profile_run(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        old = os.environ.get("MINT_VLLM_BYPASS_FUSED_MOE_LORA_OP")
        os.environ["MINT_VLLM_BYPASS_FUSED_MOE_LORA_OP"] = "1"
        try:
            return original(self, *args, **kwargs)
        finally:
            if old is None:
                os.environ.pop("MINT_VLLM_BYPASS_FUSED_MOE_LORA_OP", None)
            else:
                os.environ["MINT_VLLM_BYPASS_FUSED_MOE_LORA_OP"] = old

    profile_run._mint_profile_scope_bypass = True  # type: ignore[attr-defined]
    cls.profile_run = profile_run  # type: ignore[method-assign]


def _patch_vllm_unquantized_moe_startup_use_naive_batched_experts() -> None:
    """Route startup-only unquantized batched MoE away from Triton.

    After the fused-MoE-LoRA startup bypasses, K2 still reaches Triton-backed
    unquantized batched MoE during profile/warmup on A800 and can crash with a
    CUDA illegal-memory-access fault. Under the same startup-only env guard,
    switch just that path to the in-tree `NaiveBatchedExperts` reference kernel.
    Real serving keeps the normal Triton backend.
    """

    import vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method as method_mod
    from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
        NaiveBatchedExperts,
    )
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEActivationFormat,
    )

    cls = getattr(method_mod, "UnquantizedFusedMoEMethod", None)
    if cls is None:
        raise RuntimeError("vLLM missing UnquantizedFusedMoEMethod")

    original = getattr(cls, "select_gemm_impl", None)
    if not callable(original):
        raise RuntimeError("vLLM UnquantizedFusedMoEMethod missing select_gemm_impl")
    if getattr(original, "_mint_patched_startup_naive_batched", False):
        return

    def select_gemm_impl(self, prepare_finalize, layer):  # type: ignore[no-untyped-def]
        assert self.moe_quant_config is not None
        if (
            _env_flag("MINT_VLLM_BYPASS_FUSED_MOE_LORA_OP", default=False)
            and prepare_finalize.activation_format
            == FusedMoEActivationFormat.BatchedExperts
        ):
            method_mod.logger.info_once(
                "Using NaiveBatchedExperts for startup-only unquantized MoE",
                scope="local",
            )
            return NaiveBatchedExperts(
                max_num_tokens=self.moe.max_num_tokens,
                num_dispatchers=prepare_finalize.num_dispatchers(),
                quant_config=self.moe_quant_config,
            )
        return original(self, prepare_finalize, layer)

    select_gemm_impl._mint_patched_startup_naive_batched = True  # type: ignore[attr-defined]
    cls.select_gemm_impl = select_gemm_impl  # type: ignore[method-assign]


def _patch_vllm_fused_moe_forward_startup_fake() -> None:
    """Bypass fused MoE execution during startup-only dummy runs.

    The K2 startup crash still reaches Triton MoE kernels through
    `FusedMoE.forward_native(...)` even after lower-level LoRA and GEMM
    selection patches. Under the same startup-only env guard, return zero
    outputs with the correct shapes instead of executing any fused MoE kernel.
    """

    import torch
    import vllm.model_executor.layers.fused_moe.layer as layer_mod

    cls = getattr(layer_mod, "FusedMoE", None)
    if cls is None:
        raise RuntimeError("vLLM missing FusedMoE")

    original = getattr(cls, "forward_native", None)
    if not callable(original):
        raise RuntimeError("vLLM FusedMoE missing forward_native")
    if getattr(original, "_mint_patched_startup_fake_forward", False):
        return

    def forward_native(self, hidden_states, router_logits):  # type: ignore[no-untyped-def]
        if not _env_flag("MINT_VLLM_BYPASS_FUSED_MOE_LORA_OP", default=False):
            return original(self, hidden_states, router_logits)

        layer_mod.logger.info_once(
            "Bypassing FusedMoE.forward_native during startup-only dummy runs",
            scope="local",
        )
        fused_out = torch.zeros_like(hidden_states)
        if self.shared_experts is None:
            return fused_out
        return torch.zeros_like(hidden_states), fused_out

    forward_native._mint_patched_startup_fake_forward = True  # type: ignore[attr-defined]
    cls.forward_native = forward_native  # type: ignore[method-assign]


def _patch_vllm_invoke_fused_moe_kernel_startup_noop() -> bool:
    """Skip the Triton fused-MoE kernel during startup-only dummy runs."""

    import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe_mod

    original = getattr(fused_moe_mod, "invoke_fused_moe_kernel", None)
    if not callable(original):
        return False
    if getattr(original, "_mint_patched_startup_noop", False):
        return True

    def invoke_fused_moe_kernel(*args, **kwargs):  # type: ignore[no-untyped-def]
        if not _env_flag("MINT_VLLM_BYPASS_FUSED_MOE_LORA_OP", default=False):
            return original(*args, **kwargs)

        if len(args) < 3:
            raise RuntimeError(
                "vLLM invoke_fused_moe_kernel startup noop expected output tensor as third positional arg"
            )
        output = args[2]
        fused_moe_mod.logger.info_once(
            "Bypassing invoke_fused_moe_kernel during startup-only dummy runs",
            scope="local",
        )
        output.zero_()
        return None

    invoke_fused_moe_kernel._mint_patched_startup_noop = True  # type: ignore[attr-defined]
    fused_moe_mod.invoke_fused_moe_kernel = invoke_fused_moe_kernel  # type: ignore[assignment]
    return True


def _patch_vllm_device_memory_profiler_skip_exit_measure() -> None:
    """Avoid post-load memory measurement hanging worker startup.

    On the current K2 bringup, workers can finish loading all checkpoint shards
    and then stall inside `DeviceMemoryProfiler.__exit__()` while querying
    `current_platform.get_current_memory_usage(...)`. This metric is only used
    for logging; skipping the exit-time re-measure keeps startup moving without
    changing model execution.
    """

    import vllm.utils.mem_utils as mem_utils_mod

    cls = getattr(mem_utils_mod, "DeviceMemoryProfiler", None)
    if cls is None:
        raise RuntimeError("vLLM missing DeviceMemoryProfiler")

    original = getattr(cls, "__exit__", None)
    if original is None:
        raise RuntimeError("vLLM DeviceMemoryProfiler missing __exit__")
    if getattr(original, "_mint_patched_skip_exit_measure", False):
        return

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore[no-untyped-def]
        self.final_memory = self.initial_memory
        self.consumed_memory = 0
        import gc

        gc.collect()

    __exit__._mint_patched_skip_exit_measure = True  # type: ignore[attr-defined]
    cls.__exit__ = __exit__  # type: ignore[method-assign]


def _patch_vllm_skip_startup_memory_profile() -> None:
    """Skip vLLM startup memory profiling on the current K2 path.

    The current profile-time forward pass is not stable on A800 K2 startup.
    For eager-mode startup with an explicit gpu_memory_utilization cap, a
    conservative KV-cache budget can still be computed as:

        requested_memory - weights_memory

    where `requested_memory` already encodes the configured utilization cap and
    `weights_memory` is the measured torch memory used by loaded weights.
    """

    import gc

    import vllm.v1.worker.gpu_worker as gpu_worker_mod

    cls = getattr(gpu_worker_mod, "Worker", None)
    if cls is None:
        raise RuntimeError("vLLM missing GPU Worker")

    original = getattr(cls, "determine_available_memory", None)
    if not callable(original) or getattr(
        original, "_mint_skip_startup_profile", False
    ):
        return

    def determine_available_memory(self):  # type: ignore[no-untyped-def]
        import torch

        weights_memory = int(getattr(self.model_runner, "model_memory_usage", 0))
        requested_memory = int(getattr(self, "requested_memory", 0))
        current_free_memory = int(torch.cuda.mem_get_info()[0])
        kv_headroom_bytes = 2 << 30
        available_kv_cache_memory_bytes = min(
            requested_memory - weights_memory,
            current_free_memory - kv_headroom_bytes,
        )
        if available_kv_cache_memory_bytes <= 0:
            raise RuntimeError(
                "Skipping startup profile left no KV-cache budget: "
                f"requested_memory={requested_memory}, "
                f"weights_memory={weights_memory}, "
                f"current_free_memory={current_free_memory}, "
                f"kv_headroom_bytes={kv_headroom_bytes}"
            )
        self.non_torch_memory = 0
        self.peak_activation_memory = 0
        self.available_kv_cache_memory_bytes = available_kv_cache_memory_bytes
        gc.collect()
        return available_kv_cache_memory_bytes

    determine_available_memory._mint_skip_startup_profile = True  # type: ignore[attr-defined]
    cls.determine_available_memory = determine_available_memory  # type: ignore[method-assign]


def _patch_vllm_dummy_lora_weights_use_empty() -> None:
    """Speed up vLLM startup by avoiding huge CPU memset in dummy LoRA weights.

    vLLM's memory-profiling path can create large dummy LoRA weights via
    `torch.zeros(...)` on CPU (pin_memory), which is extremely slow for large
    MoE models. For profiling, the values are irrelevant; only shapes and
    allocations matter. Use `torch.empty(...)` to keep allocations but avoid
    the expensive initialization.
    """

    import torch
    from vllm.lora import lora_weights as lw  # type: ignore

    LoRALayerWeights = getattr(lw, "LoRALayerWeights", None)
    if LoRALayerWeights is None:
        raise RuntimeError("vLLM LoRALayerWeights missing; cannot patch dummy lora")

    original = getattr(LoRALayerWeights, "create_dummy_lora_weights", None)
    if original is None:
        raise RuntimeError("vLLM create_dummy_lora_weights missing; cannot patch")
    if getattr(original, "_mint_patched_dummy_empty", False):
        return

    is_pin_memory_available = getattr(lw, "is_pin_memory_available", None)
    if is_pin_memory_available is None:
        raise RuntimeError("vLLM is_pin_memory_available missing; cannot patch")

    @classmethod
    def create_dummy_lora_weights(  # type: ignore[no-untyped-def]
        cls,
        module_name,
        input_dim,
        output_dim,
        rank,
        dtype,
        device,
    ):
        pin_memory = str(device) == "cpu" and is_pin_memory_available()
        lora_a = torch.empty(
            (rank, input_dim),
            dtype=dtype,
            device=device,
            pin_memory=pin_memory,
        )
        lora_b = torch.empty(
            (output_dim, rank),
            dtype=dtype,
            device=device,
            pin_memory=pin_memory,
        )
        return cls(
            module_name,
            rank=rank,
            lora_alpha=1,
            lora_a=lora_a,
            lora_b=lora_b,
        )

    create_dummy_lora_weights._mint_patched_dummy_empty = True  # type: ignore[attr-defined]
    LoRALayerWeights.create_dummy_lora_weights = create_dummy_lora_weights  # type: ignore[method-assign]


def _patch_vllm_lora_from_tensors_disable_pin_memory() -> None:
    """Disable LoRA load-time pinning without replacing vLLM method signatures.

    vLLM computes the load-time pinning decision inside `LoRAModel.
    from_lora_tensors()` via the module-local `is_pin_memory_available()`
    import. Replacing the whole classmethod is brittle because newer vLLM
    builds can add kwargs such as `skip_prefixes`. Patch only the pinning
    predicate so the upstream implementation and signature stay intact.
    """

    import vllm.lora.lora_model as lora_model  # type: ignore

    current = getattr(lora_model, "is_pin_memory_available", None)
    if current is None or getattr(current, "_mint_disable_pin_memory", False):
        return

    def _disabled_pin_memory() -> bool:
        return False

    _disabled_pin_memory._mint_disable_pin_memory = True  # type: ignore[attr-defined]
    lora_model.is_pin_memory_available = _disabled_pin_memory  # type: ignore[assignment]


def _patch_vllm_worker_lora_load_to_device() -> None:
    """Load LoRA tensors directly onto the worker device.

    vLLM's default worker path loads adapter tensors onto CPU first, then
    pins/transfers them later. For large sparse MoE K2 adapters this becomes a
    long per-worker bottleneck during `add_lora()`. Loading directly to the
    target CUDA device avoids the CPU pinning path and substantially reduces
    load latency.
    """

    from vllm.lora.worker_manager import WorkerLoRAManager

    original = getattr(WorkerLoRAManager, "_load_adapter", None)
    if not callable(original) or getattr(original, "_mint_load_to_device", False):
        return

    def _load_adapter(self, lora_request):  # type: ignore[no-untyped-def]
        try:
            supported_lora_modules = self._adapter_manager.supported_lora_modules
            packed_modules_mapping = self._adapter_manager.packed_modules_mapping
            expected_lora_lst = []
            for module in supported_lora_modules:
                if module in packed_modules_mapping:
                    expected_lora_lst.extend(packed_modules_mapping[module])
                else:
                    expected_lora_lst.append(module)
                if module == "experts":
                    expected_lora_lst.append(module)
            expected_lora_modules = set(expected_lora_lst)

            from vllm.lora.peft_helper import PEFTHelper
            from vllm.lora.utils import get_adapter_absolute_path

            lora_path = get_adapter_absolute_path(lora_request.lora_path)
            peft_helper = PEFTHelper.from_local_dir(
                lora_path,
                self.max_position_embeddings,
                lora_request.tensorizer_config_dict,
            )
            peft_helper.validate_legal(self.lora_config)

            model = self._adapter_manager.model
            hf_to_vllm_mapper = getattr(model, "hf_to_vllm_mapper", None)
            device = str(self.device)
            return self._lora_model_cls.from_local_checkpoint(
                lora_path,
                expected_lora_modules,
                peft_helper=peft_helper,
                lora_model_id=lora_request.lora_int_id,
                device=device,
                dtype=self.lora_config.lora_dtype,
                model_vocab_size=self.vocab_size,
                tensorizer_config_dict=lora_request.tensorizer_config_dict,
                weights_mapper=hf_to_vllm_mapper,
            )
        except FileNotFoundError as e:
            raise ValueError(
                f"Loading lora {lora_request.lora_name} failed: No adapter found for {lora_request.lora_path}"
            ) from e

    _load_adapter._mint_load_to_device = True  # type: ignore[attr-defined]
    WorkerLoRAManager._load_adapter = _load_adapter  # type: ignore[method-assign]


class _SparseShardTensor:
    def __init__(
        self,
        shard_tensors,
        shard_starts: tuple[int, ...],
        num_experts: int,
        scale_factors=None,
    ):  # type: ignore[no-untyped-def]
        if not shard_tensors:
            raise RuntimeError("Sparse shard tensor requires representative tensors")
        self._mint_sparse_shards = tuple(shard_tensors)
        self._mint_shard_starts = tuple(int(x) for x in shard_starts)
        self._mint_num_experts = int(num_experts)
        self._mint_scale_factors = (
            None if scale_factors is None else tuple(float(x) for x in scale_factors)
        )
        first = shard_tensors[0]
        self.shape = (len(shard_tensors),) + tuple(first.shape)
        self.device = first.device

    def map(self, fn):  # type: ignore[no-untyped-def]
        return _SparseShardTensor(
            tuple(fn(t) for t in self._mint_sparse_shards),
            self._mint_shard_starts,
            self._mint_num_experts,
            self._mint_scale_factors,
        )

    def get_rep(self, rep_idx: int):  # type: ignore[no-untyped-def]
        rep_idx = int(rep_idx)
        rep = self._mint_sparse_shards[rep_idx]
        if self._mint_scale_factors is None:
            return rep
        return rep * self._mint_scale_factors[rep_idx]

    def pin_memory(self):  # type: ignore[no-untyped-def]
        return self


def _patch_vllm_pack_moe_sparse_ok() -> None:
    """Patch vLLM MoE LoRA packing to tolerate missing expert adapters.

    vLLM's PackedLoRALayerWeights.pack_moe can assume every expert has (w1,w2,w3)
    LoRA weights present. Our adapter export can be shared across experts (export
    expert 0 only). Missing experts should be treated as sharing the base expert
    weights, without materializing full per-expert tensors when possible.
    """

    from vllm.lora import lora_weights as lw  # type: ignore

    Packed = getattr(lw, "PackedLoRALayerWeights", None)
    LoRALayerWeights = getattr(lw, "LoRALayerWeights", None)
    if Packed is None or LoRALayerWeights is None:
        raise RuntimeError("vLLM lora_weights types missing; cannot patch pack_moe")

    cm = Packed.__dict__.get("pack_moe")
    orig_fn = getattr(cm, "__func__", None)
    if orig_fn is None:
        raise RuntimeError("vLLM pack_moe not found; cannot patch")
    if getattr(orig_fn, "__mint_sparse_ok__", False):
        return

    import inspect

    try:
        sig = inspect.signature(orig_fn)
    except Exception as e:
        raise RuntimeError(
            f"Unable to inspect vLLM PackedLoRALayerWeights.pack_moe signature: {type(e).__name__}: {e}"
        ) from e

    has_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if "is_non_gated_moe" not in sig.parameters and not has_kwargs:
        import vllm  # type: ignore

        raise RuntimeError(
            "vLLM PackedLoRALayerWeights.pack_moe signature mismatch for sparse-ok patch: "
            f"expected 'is_non_gated_moe' kwarg (or **kwargs), got signature={sig}. "
            f"installed_vllm_version={getattr(vllm, '__version__', 'unknown')!r}"
        )

    def pack_moe_sparse_ok(
        cls, loras, module_name: str, is_non_gated_moe: bool = False
    ):  # type: ignore[no-untyped-def]
        timing = _env_flag("MINT_VLLM_TIMING_SET_LORA", default=False)
        if not loras or (len(loras) % 3) != 0:
            raise RuntimeError(
                f"Unexpected MoE LoRA pack_moe inputs for module={module_name!r}: len(loras)={len(loras)}"
            )

        n_experts = len(loras) // 3

        base_any = next((lora for lora in loras if lora is not None), None)
        if base_any is None:
            raise RuntimeError(
                f"MoE LoRA pack_moe got all-None loras for module={module_name!r}"
            )
        rank = int(getattr(base_any, "rank"))
        lora_alpha = int(getattr(base_any, "lora_alpha"))

        base_w1 = next(
            (loras[i * 3] for i in range(n_experts) if loras[i * 3] is not None), None
        )
        base_w2 = next(
            (
                loras[i * 3 + 1]
                for i in range(n_experts)
                if loras[i * 3 + 1] is not None
            ),
            None,
        )
        base_w3 = next(
            (
                loras[i * 3 + 2]
                for i in range(n_experts)
                if loras[i * 3 + 2] is not None
            ),
            None,
        )
        if base_w1 is None or base_w2 is None or base_w3 is None:
            raise RuntimeError(
                f"MoE LoRA pack_moe missing base weight(s) for module={module_name!r}"
            )

        def _same_lora(x, y):  # type: ignore[no-untyped-def]
            def _first_value(t):  # type: ignore[no-untyped-def]
                if int(t.numel()) == 0:
                    return None
                return t.reshape(-1)[0].item()

            return (
                int(getattr(x, "rank")) == int(getattr(y, "rank"))
                and int(getattr(x, "lora_alpha")) == int(getattr(y, "lora_alpha"))
                and float(getattr(x, "scaling", lora_alpha / rank))
                == float(getattr(y, "scaling", lora_alpha / rank))
                and tuple(x.lora_a.shape) == tuple(y.lora_a.shape)
                and x.lora_a.dtype == y.lora_a.dtype
                and tuple(x.lora_b.shape) == tuple(y.lora_b.shape)
                and x.lora_b.dtype == y.lora_b.dtype
                and _first_value(x.lora_a) == _first_value(y.lora_a)
                and _first_value(x.lora_b) == _first_value(y.lora_b)
            )

        def _build_sparse_from_representatives(starts):  # type: ignore[no-untyped-def]
            present_w1 = {eid: loras[eid * 3] for eid in starts}
            present_w2 = {eid: loras[eid * 3 + 1] for eid in starts}
            present_w3 = {eid: loras[eid * 3 + 2] for eid in starts}

            def _shard_lora_a(present: dict[int, object]):  # type: ignore[no-untyped-def]
                return _SparseShardTensor(
                    tuple(present[start].lora_a for start in starts),  # type: ignore[index,attr-defined]
                    tuple(starts),
                    int(n_experts),
                )

            def _shard_lora_b(present: dict[int, object]):  # type: ignore[no-untyped-def]
                return _SparseShardTensor(
                    tuple(present[start].lora_b for start in starts),  # type: ignore[index,attr-defined]
                    tuple(starts),
                    int(n_experts),
                    tuple(
                        float(getattr(present[start], "scaling", lora_alpha / rank))  # type: ignore[index,attr-defined]
                        for start in starts
                    ),
                )

            return cls(
                module_name,
                rank,
                [lora_alpha, lora_alpha, lora_alpha],
                [
                    _shard_lora_a(present_w1),
                    _shard_lora_a(present_w2),
                    _shard_lora_a(present_w3),
                ],
                [
                    _shard_lora_b(present_w1),
                    _shard_lora_b(present_w2),
                    _shard_lora_b(present_w3),
                ],
                scaling=[1.0, 1.0, 1.0],
            )

        if all(lora is not None for lora in loras):
            # Dense checkpoints can still be block-shared across contiguous expert
            # ranges. If so, keep only one representative per contiguous block and
            # reuse the sparse-shard set_lora path to avoid materializing the full
            # [num_experts, ...] packed tensors.
            t_detect0 = time.perf_counter() if timing else 0.0
            shard_starts = [0]
            rep_eid = 0
            for eid in range(1, n_experts):
                if not (
                    _same_lora(loras[eid * 3], loras[rep_eid * 3])
                    and _same_lora(loras[eid * 3 + 1], loras[rep_eid * 3 + 1])
                    and _same_lora(loras[eid * 3 + 2], loras[rep_eid * 3 + 2])
                ):
                    shard_starts.append(eid)
                    rep_eid = eid
            t_detect1 = time.perf_counter() if timing else 0.0
            if len(shard_starts) < n_experts:
                t_build0 = time.perf_counter() if timing else 0.0
                obj = _build_sparse_from_representatives(shard_starts)
                t_build1 = time.perf_counter() if timing else 0.0
                print(
                    f"[vLLM dense pack_moe dedup] module={module_name} "
                    f"n_experts={n_experts} reps={len(shard_starts)} starts={shard_starts}",
                    flush=True,
                )
                if timing:
                    print(
                        f"[vLLM dense pack_moe dedup timing] module={module_name} "
                        f"detect_s={t_detect1 - t_detect0:.6f} build_s={t_build1 - t_build0:.6f}",
                        flush=True,
                    )
                return obj
            return orig_fn(cls, loras, module_name, is_non_gated_moe=is_non_gated_moe)

        only_expert0 = True
        for eid in range(1, n_experts):
            if (
                loras[eid * 3] is not None
                or loras[eid * 3 + 1] is not None
                or loras[eid * 3 + 2] is not None
            ):
                only_expert0 = False
                break

        if only_expert0:
            # Route single-expert exports through the sparse-shard path so
            # fused_moe.set_lora consumes shard metadata instead of falling back
            # to dense per-expert tensors.
            return _build_sparse_from_representatives([0])
        else:
            present_w1 = {
                eid: loras[eid * 3]
                for eid in range(n_experts)
                if loras[eid * 3] is not None
            }
            present_w2 = {
                eid: loras[eid * 3 + 1]
                for eid in range(n_experts)
                if loras[eid * 3 + 1] is not None
            }
            present_w3 = {
                eid: loras[eid * 3 + 2]
                for eid in range(n_experts)
                if loras[eid * 3 + 2] is not None
            }
            present_eids = sorted(set(present_w1) | set(present_w2) | set(present_w3))
            if not present_eids:
                raise RuntimeError(
                    f"MoE LoRA pack_moe got no present experts for module={module_name!r}"
                )
            if set(present_w1) != set(present_w2) or set(present_w1) != set(present_w3):
                raise RuntimeError(
                    "Sparse MoE LoRA adapter has inconsistent expert coverage across w1/w2/w3. "
                    f"module={module_name!r} w1={sorted(present_w1)} w2={sorted(present_w2)} w3={sorted(present_w3)}"
                )

            shard_starts = present_eids
            shard_ends = [*shard_starts[1:], n_experts]
            shard_spans = list(zip(shard_starts, shard_ends))
            expected_rep_count = len(shard_starts)
            if (
                shard_starts[0] != 0
                or any(start >= end for start, end in shard_spans)
                or n_experts % expected_rep_count != 0
            ):
                raise RuntimeError(
                    "Unsupported sparse MoE LoRA shard layout for vLLM pack_moe. "
                    f"module={module_name!r} n_experts={n_experts} present_expert_ids={present_eids}"
                )
            shard_size = n_experts // expected_rep_count
            expected_starts = list(range(0, n_experts, shard_size))
            if shard_starts != expected_starts:
                raise RuntimeError(
                    "Sparse MoE LoRA adapter representatives do not match EP-shard boundaries. "
                    f"module={module_name!r} n_experts={n_experts} "
                    f"present_expert_ids={present_eids} expected_present_expert_ids={expected_starts}"
                )
            return _build_sparse_from_representatives(shard_starts)

    pack_moe_sparse_ok.__mint_sparse_ok__ = True  # type: ignore[attr-defined]
    Packed.pack_moe = classmethod(pack_moe_sparse_ok)


def _patch_vllm_fused_moe_set_lora_sparse_shards() -> None:
    import vllm.lora.layers.fused_moe as fused_moe_mod  # type: ignore

    def _get_spans(
        tensor: "torch.Tensor", num_experts: int
    ) -> list[tuple[int, int]] | None:  # type: ignore[name-defined]
        starts = getattr(tensor, "_mint_shard_starts", None)
        if starts is None:
            return None
        starts = [int(x) for x in starts]
        if not starts:
            raise RuntimeError("Sparse MoE shard metadata is empty")
        if starts[0] != 0:
            raise RuntimeError(
                f"Sparse MoE shard metadata must start at expert 0: starts={starts}"
            )
        if starts != sorted(starts):
            raise RuntimeError(
                f"Sparse MoE shard metadata must be sorted: starts={starts}"
            )
        if int(getattr(tensor, "_mint_num_experts", num_experts)) != num_experts:
            raise RuntimeError(
                "Sparse MoE shard metadata num_experts mismatch: "
                f"tensor_num_experts={getattr(tensor, '_mint_num_experts', None)} expected={num_experts}"
            )
        if tensor.shape[0] != len(starts):
            raise RuntimeError(
                "Sparse MoE shard metadata length mismatch: "
                f"tensor.shape[0]={tensor.shape[0]} starts={starts}"
            )
        ends = [*starts[1:], num_experts]
        spans = list(zip(starts, ends))
        for start, end in spans:
            if start < 0 or end <= start or end > num_experts:
                raise RuntimeError(
                    "Sparse MoE shard span is invalid: "
                    f"spans={spans} num_experts={num_experts}"
                )
        return spans

    def _copy_sparse(
        dst: "torch.Tensor",
        index: int,
        src: "torch.Tensor",
        spans: list[tuple[int, int]],
    ) -> None:  # type: ignore[name-defined]
        for rep_idx, (start, end) in enumerate(spans):
            expert_count = end - start
            if hasattr(src, "_mint_sparse_shards"):
                rep = src.get_rep(rep_idx)
            else:
                rep = src[rep_idx : rep_idx + 1]
            target = dst[index, start:end]
            if rep.ndim == target.ndim - 1:
                rep = rep.unsqueeze(0)
            elif rep.ndim != target.ndim:
                raise RuntimeError(
                    "Sparse MoE representative rank mismatch: "
                    f"rep.shape={tuple(rep.shape)} target.shape={tuple(target.shape)} "
                    f"rep_idx={rep_idx} span=({start},{end})"
                )
            if rep.shape[0] == 1:
                expanded = rep.expand(expert_count, *rep.shape[1:])
            elif rep.shape[0] == expert_count:
                expanded = rep
            else:
                raise RuntimeError(
                    "Sparse MoE representative leading dimension mismatch: "
                    f"rep.shape={tuple(rep.shape)} expert_count={expert_count} "
                    f"rep_idx={rep_idx} span=({start},{end})"
                )
            target_view = target[
                (slice(None),) + tuple(slice(0, dim) for dim in expanded.shape[1:])
            ]
            if tuple(target_view.shape) != tuple(expanded.shape):
                raise RuntimeError(
                    "Sparse MoE target slice mismatch: "
                    f"target_view.shape={tuple(target_view.shape)} expanded.shape={tuple(expanded.shape)} "
                    f"rep_idx={rep_idx} span=({start},{end})"
                )
            target_view.copy_(expanded, non_blocking=True)

    def _slice_sparse(src, slicer, reshape_fn=None):  # type: ignore[no-untyped-def]
        if hasattr(src, "_mint_sparse_shards"):
            if reshape_fn is None:
                return src.map(slicer)
            return src.map(lambda t: slicer(reshape_fn(t)))
        return slicer(src)

    cls = getattr(fused_moe_mod, "FusedMoEWithLoRA", None)
    if cls is None:
        raise RuntimeError("vLLM fused_moe missing FusedMoEWithLoRA")
    original = getattr(cls, "set_lora", None)
    if not callable(original):
        raise RuntimeError("vLLM FusedMoEWithLoRA.set_lora missing")
    if not getattr(original, "_mint_sparse_shards", False):

        def set_lora(self, index: int, lora_a, lora_b):  # type: ignore[no-untyped-def]
            assert isinstance(lora_a, list)
            assert isinstance(lora_b, list)
            spans = _get_spans(lora_a[0], int(self.w13_lora_a_stacked[0].shape[1]))
            if spans is None:
                return original(self, index, lora_a, lora_b)

            module_tag = (
                getattr(self, "prefix", None)
                or getattr(getattr(self, "base_layer", None), "prefix", None)
                or type(self).__name__
            )

            timing = _env_flag("MINT_VLLM_TIMING_SET_LORA", default=False)
            t0 = time.perf_counter() if timing else 0.0
            self.reset_lora(index)
            self.adapter_enabled[index] = 1

            w1_lora_a, w2_lora_a, w3_lora_a = lora_a
            w1_lora_b, w2_lora_b, w3_lora_b = lora_b

            t1 = time.perf_counter() if timing else 0.0
            sliced_w1_lora_a = _slice_sparse(
                w1_lora_a,
                self._slice_w13_a,
                reshape_fn=lambda t: t.reshape(1, -1, t.shape[-1]),
            )
            sliced_w1_lora_b = _slice_sparse(
                w1_lora_b,
                self._slice_w13_b,
                reshape_fn=lambda t: t.reshape(1, t.shape[0], t.shape[1]),
            )
            sliced_w3_lora_a = _slice_sparse(
                w3_lora_a,
                self._slice_w13_a,
                reshape_fn=lambda t: t.reshape(1, -1, t.shape[-1]),
            )
            sliced_w3_lora_b = _slice_sparse(
                w3_lora_b,
                self._slice_w13_b,
                reshape_fn=lambda t: t.reshape(1, t.shape[0], t.shape[1]),
            )
            sliced_w2_lora_a = _slice_sparse(
                w2_lora_a,
                self._slice_w2_a,
                reshape_fn=lambda t: t.reshape(1, -1, t.shape[-1]),
            )
            sliced_w2_lora_b = _slice_sparse(
                w2_lora_b,
                self._slice_w2_b,
                reshape_fn=lambda t: t.reshape(1, t.shape[0], t.shape[1]),
            )
            t2 = time.perf_counter() if timing else 0.0
            copy_times: list[tuple[str, float]] = []

            def _timed_copy(name: str, dst, src):  # type: ignore[no-untyped-def]
                c0 = time.perf_counter() if timing else 0.0
                _copy_sparse(dst, index, src, spans)
                c1 = time.perf_counter() if timing else 0.0
                if timing:
                    copy_times.append((name, c1 - c0))
                    print(
                        f"[vLLM sparse set_lora copy] module={module_tag} name={name} "
                        f"reps={len(spans)} elapsed_s={c1 - c0:.6f}",
                        flush=True,
                    )

            _timed_copy("w13_a_0", self.w13_lora_a_stacked[0], sliced_w1_lora_a)
            _timed_copy("w13_a_1", self.w13_lora_a_stacked[1], sliced_w3_lora_a)
            _timed_copy("w13_b_0", self.w13_lora_b_stacked[0], sliced_w1_lora_b)
            _timed_copy("w13_b_1", self.w13_lora_b_stacked[1], sliced_w3_lora_b)
            _timed_copy("w2_a_0", self.w2_lora_a_stacked[0], sliced_w2_lora_a)
            _timed_copy("w2_b_0", self.w2_lora_b_stacked[0], sliced_w2_lora_b)
            t3 = time.perf_counter() if timing else 0.0
            if timing:
                copy_breakdown = ",".join(
                    f"{name}:{elapsed:.6f}" for name, elapsed in copy_times
                )
                print(
                    f"[vLLM sparse set_lora timing] module={module_tag} reps={len(spans)} "
                    f"slice_s={t2 - t1:.6f} copy_s={t3 - t2:.6f} total_s={t3 - t0:.6f} "
                    f"copy_breakdown={copy_breakdown}",
                    flush=True,
                )

        set_lora._mint_sparse_shards = True  # type: ignore[attr-defined]
        cls.set_lora = set_lora  # type: ignore[method-assign]

    cls3d = getattr(fused_moe_mod, "FusedMoE3DWithLoRA", None)
    if cls3d is None:
        raise RuntimeError("vLLM fused_moe missing FusedMoE3DWithLoRA")
    original3d = getattr(cls3d, "set_lora", None)
    if not callable(original3d):
        raise RuntimeError("vLLM FusedMoE3DWithLoRA.set_lora missing")
    if not getattr(original3d, "_mint_sparse_shards", False):

        def set_lora_3d(self, index: int, lora_a, lora_b):  # type: ignore[no-untyped-def]
            assert isinstance(lora_a, list)
            assert isinstance(lora_b, list)
            assert len(lora_a) == len(lora_b) == 2
            num_experts = int(self.w13_lora_a_stacked[0].shape[1])
            spans = _get_spans(lora_a[0], num_experts)
            if spans is None:
                return original3d(self, index, lora_a, lora_b)

            module_tag = (
                getattr(self, "prefix", None)
                or getattr(getattr(self, "base_layer", None), "prefix", None)
                or type(self).__name__
            )

            timing = _env_flag("MINT_VLLM_TIMING_SET_LORA", default=False)
            t0 = time.perf_counter() if timing else 0.0
            self.reset_lora(index)
            self.adapter_enabled[index] = 1

            w13_lora_a, w2_lora_a = lora_a
            w13_lora_b, w2_lora_b = lora_b
            src_experts = len(spans)
            if not hasattr(w13_lora_a, "_mint_sparse_shards"):
                w13_lora_a = w13_lora_a.reshape(src_experts, -1, w13_lora_a.shape[-1])
                w2_lora_a = w2_lora_a.reshape(src_experts, -1, w2_lora_a.shape[-1])
                w13_lora_b = w13_lora_b.reshape(
                    w13_lora_b.shape[0], src_experts, -1
                ).permute(1, 0, 2)
                w2_lora_b = w2_lora_b.reshape(
                    w2_lora_b.shape[0], src_experts, -1
                ).permute(1, 0, 2)

            t1 = time.perf_counter() if timing else 0.0
            sliced_w13_lora_a = _slice_sparse(
                w13_lora_a,
                self._slice_w13_a,
                reshape_fn=lambda t: t.reshape(1, -1, t.shape[-1]),
            )
            sliced_w13_lora_b = _slice_sparse(
                w13_lora_b,
                self._slice_w13_b,
                reshape_fn=lambda t: t.reshape(1, t.shape[0], t.shape[1]),
            )
            sliced_w2_lora_a = _slice_sparse(
                w2_lora_a,
                self._slice_w2_a,
                reshape_fn=lambda t: t.reshape(1, -1, t.shape[-1]),
            )
            sliced_w2_lora_b = _slice_sparse(
                w2_lora_b,
                self._slice_w2_b,
                reshape_fn=lambda t: t.reshape(1, t.shape[0], t.shape[1]),
            )
            t2 = time.perf_counter() if timing else 0.0
            copy_times: list[tuple[str, float]] = []

            def _timed_copy(name: str, dst, src):  # type: ignore[no-untyped-def]
                c0 = time.perf_counter() if timing else 0.0
                _copy_sparse(dst, index, src, spans)
                c1 = time.perf_counter() if timing else 0.0
                if timing:
                    copy_times.append((name, c1 - c0))
                    print(
                        f"[vLLM sparse set_lora copy] module={module_tag} name={name} "
                        f"reps={len(spans)} elapsed_s={c1 - c0:.6f}",
                        flush=True,
                    )

            _timed_copy("w13_a_0", self.w13_lora_a_stacked[0], sliced_w13_lora_a)
            _timed_copy("w2_a_0", self.w2_lora_a_stacked[0], sliced_w2_lora_a)
            _timed_copy("w13_b_0", self.w13_lora_b_stacked[0], sliced_w13_lora_b)
            _timed_copy("w2_b_0", self.w2_lora_b_stacked[0], sliced_w2_lora_b)
            t3 = time.perf_counter() if timing else 0.0
            if timing:
                copy_breakdown = ",".join(
                    f"{name}:{elapsed:.6f}" for name, elapsed in copy_times
                )
                print(
                    f"[vLLM sparse set_lora timing] module={module_tag} reps={len(spans)} "
                    f"slice_s={t2 - t1:.6f} copy_s={t3 - t2:.6f} total_s={t3 - t0:.6f} "
                    f"copy_breakdown={copy_breakdown}",
                    flush=True,
                )

        set_lora_3d._mint_sparse_shards = True  # type: ignore[attr-defined]
        cls3d.set_lora = set_lora_3d  # type: ignore[method-assign]


def _patch_vllm_lora_optimize_overlap_safe() -> None:
    """Avoid in-place LoRA optimize on overlapping tensors.

    vLLM merges scaling into `lora_b` via in-place ops (`*= scaling`) inside
    LoRA `optimize()`. This fails for tensors with internal overlap (commonly
    produced by `expand(...)`), raising:
      "unsupported operation: more than one element ... refers to a single
       memory location"

    When overlap is detected, use out-of-place multiplication and replace the
    tensor reference to preserve semantics without invalid writes.
    """

    try:
        import torch
        from vllm.lora import lora_weights as lw  # type: ignore
    except Exception:
        return

    LoRA = getattr(lw, "LoRALayerWeights", None)
    Packed = getattr(lw, "PackedLoRALayerWeights", None)
    if LoRA is None or Packed is None:
        return

    def _has_internal_overlap(t: "torch.Tensor") -> bool:  # type: ignore[name-defined]
        try:
            return bool(torch._C._debug_has_internal_overlap(t))  # type: ignore[attr-defined]
        except Exception:
            return False

    orig_opt = getattr(LoRA, "optimize", None)
    if callable(orig_opt) and not getattr(orig_opt, "_mint_overlap_safe", False):

        def optimize(self):  # type: ignore[no-untyped-def]
            if getattr(self, "scaling", 1) == 1:
                return self
            lb = getattr(self, "lora_b", None)
            if lb is None:
                return self
            if _has_internal_overlap(lb):
                self.lora_b = lb * float(self.scaling)
            else:
                self.lora_b *= float(self.scaling)
            self.scaling = 1
            return self

        optimize._mint_overlap_safe = True  # type: ignore[attr-defined]
        LoRA.optimize = optimize  # type: ignore[method-assign]

    orig_popt = getattr(Packed, "optimize", None)
    if callable(orig_popt) and not getattr(orig_popt, "_mint_overlap_safe", False):

        def optimize(self):  # type: ignore[no-untyped-def]
            for i in range(len(self.lora_b)):
                if self.scaling[i] == 1 or self.lora_b[i] is None:  # type: ignore
                    continue
                lb = self.lora_b[i]  # type: ignore
                if _has_internal_overlap(lb):
                    self.lora_b[i] = lb * float(self.scaling[i])  # type: ignore
                else:
                    self.lora_b[i] *= float(self.scaling[i])  # type: ignore
                self.scaling[i] = 1  # type: ignore
            return self

        optimize._mint_overlap_safe = True  # type: ignore[attr-defined]
        Packed.optimize = optimize  # type: ignore[method-assign]


def _patch_vllm_lora_pin_memory_overlap_safe() -> None:
    """Avoid pin_memory() crash on overlapping MoE LoRA tensors.

    vLLM pins LoRA tensors after packing/merging inside
    `LoRAModelManager._create_merged_loras_inplace`:
      lora.lora_b[index] = lora.lora_b[index].pin_memory()

    When MoE LoRA weights are represented as `expand(...)` views (to avoid
    materializing [num_experts, ...] tensors for shared-expert exports), calling
    `pin_memory()` can fail with:
      "unsupported operation: more than one element ... refers to a single
       memory location"

    Do not materialize the expanded tensor. Instead, leave it unpinned when
    pinning fails due to internal overlap.
    """

    try:
        import torch
        from vllm.lora.model_manager import LoRAModelManager  # type: ignore
    except Exception:
        return

    orig = getattr(LoRAModelManager, "_create_merged_loras_inplace", None)
    if not callable(orig) or getattr(orig, "_mint_pin_memory_overlap_safe", False):
        return

    def _create_merged_loras_inplace(self, lora_model):  # type: ignore[no-untyped-def]
        orig_pin = torch.Tensor.pin_memory  # type: ignore[attr-defined]

        def _safe_pin_memory(t, *args, **kwargs):  # type: ignore[no-untyped-def]
            try:
                return orig_pin(t, *args, **kwargs)
            except RuntimeError as e:
                msg = str(e)
                if (
                    "more than one element of the written-to tensor refers to a single memory location"
                    in msg
                ):
                    return t
                raise

        torch.Tensor.pin_memory = _safe_pin_memory  # type: ignore[assignment]
        try:
            return orig(self, lora_model)
        finally:
            torch.Tensor.pin_memory = orig_pin  # type: ignore[assignment]

    _create_merged_loras_inplace._mint_pin_memory_overlap_safe = True  # type: ignore[attr-defined]
    LoRAModelManager._create_merged_loras_inplace = _create_merged_loras_inplace  # type: ignore[method-assign]


def _patch_vllm_ray_env_carry_over_pythonpath() -> None:
    """Ensure vLLM Ray workers start with our PYTHONPATH.

    vLLM's Ray backend uses `vllm.ray.ray_env.get_env_vars_to_copy()` to decide
    which env vars are propagated via Ray runtime_env at actor startup.

    By default it only carries vLLM-defined env vars; it does not include
    `PYTHONPATH`, so worker processes (EngineCore, TP workers, etc.) may not
    import `sitecustomize.py`, and thus miss our vLLM monkey patches.
    """

    try:
        import vllm.ray.ray_env as ray_env  # type: ignore
    except Exception:
        return

    orig = getattr(ray_env, "get_env_vars_to_copy", None)
    if not callable(orig) or getattr(orig, "_mint_pythonpath_carryover", False):
        return

    def get_env_vars_to_copy(  # type: ignore[no-untyped-def]
        exclude_vars=None,
        additional_vars=None,
        destination=None,
    ):
        os.environ["PYTHONPATH"] = _sanitize_vllm_worker_pythonpath(os.environ.get("PYTHONPATH"))
        extra = {
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
            "MINT_ENABLE_VLLM_IMPORT_PATCHES",
            "MINT_RAY_GCS_ADDRESS",
            "MINT_VLLM_RAY_EXECUTOR_NO_COMPILED_DAG_SAMPLE",
            "VLLM_USE_V1",
            "MINT_VLLM_DISABLE_MOE_LORA_PACKING",
            "MINT_VLLM_FULLY_SHARDED_LORAS",
            "MINT_VLLM_DISABLE_TORCH_DIST_TP",
            "TVM_FFI_DISABLE_TORCH_C_DLPACK",
        }
        if additional_vars is None:
            additional_vars2 = set(extra)
        else:
            additional_vars2 = set(additional_vars) | set(extra)
        exclude_vars2 = set(exclude_vars or set()) | set(_DRIVER_ONLY_RAY_RUNTIME_ENV_KEYS)
        return orig(
            exclude_vars=exclude_vars2,
            additional_vars=additional_vars2,
            destination=destination,
        )

    get_env_vars_to_copy._mint_pythonpath_carryover = True  # type: ignore[attr-defined]
    ray_env.get_env_vars_to_copy = get_env_vars_to_copy  # type: ignore[method-assign]


def _patch_ray_placement_group_bundle_cache() -> None:
    """Handle Ray state payloads that omit `bundles` for direct placement-group lookup."""
    try:
        import importlib

        pg_mod = importlib.import_module("ray.util.placement_group")
    except Exception:
        return

    original = getattr(pg_mod, "_get_bundle_cache", None)
    if not callable(original) or getattr(original, "_mint_bundle_cache_patch", False):
        return

    def _normalize_pg_id(pg_id):  # type: ignore[no-untyped-def]
        try:
            return pg_id.hex()
        except Exception:
            return str(pg_id)

    def _get_bundle_cache(pg_id):  # type: ignore[no-untyped-def]
        worker = pg_mod.ray._private.worker.global_worker
        worker.check_connected()

        info = pg_mod.ray._private.state.state.placement_group_table(pg_id)
        bundles = info.get("bundles")
        if bundles is not None:
            return list(bundles.values())

        target = _normalize_pg_id(pg_id)
        table = pg_mod.placement_group_table()
        for key, candidate in table.items():
            candidate_id = str(candidate.get("placement_group_id") or key)
            if candidate_id == target and candidate.get("bundles") is not None:
                return list(candidate["bundles"].values())

        raise KeyError(
            f"placement group {target} missing bundles in both direct and full table lookup: "
            f"direct_keys={sorted(info.keys())}"
        )

    _get_bundle_cache._mint_bundle_cache_patch = True  # type: ignore[attr-defined]
    pg_mod._get_bundle_cache = _get_bundle_cache  # type: ignore[assignment]


def _patch_vllm_fused_moe_lora_use_torch_dist_tp_collectives() -> None:
    """Force torch.distributed collectives in fused_moe_lora TP path."""

    if _env_flag("MINT_VLLM_DISABLE_TORCH_DIST_TP", default=False):
        return

    try:
        import importlib

        import torch
        import torch.distributed as dist

        op = importlib.import_module("vllm.lora.ops.triton_ops.fused_moe_lora_op")
    except Exception:
        return

    if getattr(op, "_mint_patched_fused_moe_lora_torch_dist_tp", False):
        return

    def _get_tp_process_group():  # type: ignore[no-untyped-def]
        try:
            import vllm.distributed.parallel_state as ps

            if not ps.model_parallel_is_initialized():
                return None
            tp = ps.get_tp_group()
            for attr in ("process_group", "pg", "group", "device_group", "_group"):
                g = getattr(tp, attr, None)
                if g is not None and not isinstance(g, bool):
                    return g
            for meth in ("get_process_group", "get_group", "get_device_group"):
                m = getattr(tp, meth, None)
                if callable(m):
                    try:
                        g = m()
                    except Exception:
                        continue
                    if g is not None and not isinstance(g, bool):
                        return g
        except Exception:
            return None
        return None

    def tensor_model_parallel_all_reduce(x):  # type: ignore[no-untyped-def]
        pg = _get_tp_process_group()
        if pg is None:
            raise RuntimeError("vLLM TP process group not found for all_reduce")
        y = x.contiguous().clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM, group=pg)
        return y

    def tensor_model_parallel_all_gather(x):  # type: ignore[no-untyped-def]
        if not hasattr(dist, "all_gather_into_tensor"):
            raise RuntimeError("torch.distributed.all_gather_into_tensor unavailable")
        pg = _get_tp_process_group()
        if pg is None:
            raise RuntimeError("vLLM TP process group not found for all_gather")
        world_size = dist.get_world_size(group=pg)
        x2 = x.contiguous()
        flat = x2.view(-1, x2.shape[-1])
        out = torch.empty(
            (flat.shape[0], flat.shape[1] * world_size),
            device=flat.device,
            dtype=flat.dtype,
        )
        dist.all_gather_into_tensor(out, flat, group=pg)
        return out.view(*x2.shape[:-1], x2.shape[-1] * world_size)

    setattr(op, "tensor_model_parallel_all_reduce", tensor_model_parallel_all_reduce)
    setattr(op, "tensor_model_parallel_all_gather", tensor_model_parallel_all_gather)
    setattr(op, "_mint_patched_fused_moe_lora_torch_dist_tp", True)


def _patch_vllm_gpu_worker_kv_debug_info() -> None:
    try:
        import vllm.v1.worker.gpu_worker as gpu_worker_mod
    except Exception:
        return

    cls = getattr(gpu_worker_mod, "Worker", None) or getattr(gpu_worker_mod, "GPUWorker", None)
    if cls is None:
        return

    original = getattr(cls, "get_kv_debug_info", None)
    if callable(original) and getattr(original, "_mint_kv_debug_info", False):
        return

    def get_kv_debug_info(self):  # type: ignore[no-untyped-def]
        kv_cfg = getattr(getattr(self, "model_runner", None), "kv_cache_config", None)
        return {
            "available_kv_cache_memory_bytes": int(getattr(self, "available_kv_cache_memory_bytes", 0) or 0),
            "requested_memory_bytes": int(getattr(self, "requested_memory", 0) or 0),
            "non_torch_memory_bytes": int(getattr(self, "non_torch_memory", 0) or 0),
            "peak_activation_memory_bytes": int(getattr(self, "peak_activation_memory", 0) or 0),
            "kv_cache_num_blocks": int(getattr(kv_cfg, "num_blocks", 0) or 0) if kv_cfg is not None else 0,
            "kv_cache_groups": len(getattr(kv_cfg, "kv_cache_groups", []) or []) if kv_cfg is not None else 0,
        }

    get_kv_debug_info._mint_kv_debug_info = True  # type: ignore[attr-defined]
    cls.get_kv_debug_info = get_kv_debug_info  # type: ignore[method-assign]


def _patch_vllm_qwen35_text_only_adapter() -> None:
    try:
        from mint_server.backend.qwen35_text_vllm_adapter import (
            install_vllm_qwen35_text_only_adapter_patches,
        )
    except Exception:
        return

    install_vllm_qwen35_text_only_adapter_patches()


def _apply_vllm_worker_patches() -> None:
    if not _env_flag("MINT_ENABLE_VLLM_IMPORT_PATCHES", default=False):
        return
    if "VLLM_USE_V1" not in os.environ:
        return

    # Prevent vLLM workers from spawning repeated optional builds of Torch C DLPack
    # bindings via tvm_ffi (observed as many `_build_optional_torch_c_dlpack.py`
    # processes on Ray worker nodes).
    os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")

    _patch_ray_runtime_env_to_dict_drop_driver_attach_hints()
    _patch_vllm_ray_env_carry_over_pythonpath()
    if not _env_flag("MINT_VLLM_DISABLE_MOE_LORA_PACKING", default=False):
        _patch_vllm_pack_moe_sparse_ok()
        _patch_vllm_fused_moe_set_lora_sparse_shards()
    _patch_vllm_lora_from_tensors_disable_pin_memory()
    if _env_flag("MINT_VLLM_WORKER_LORA_LOAD_TO_DEVICE", default=False):
        _patch_vllm_worker_lora_load_to_device()
    _patch_vllm_lora_optimize_overlap_safe()
    _patch_vllm_lora_pin_memory_overlap_safe()
    _patch_vllm_ray_executor_use_explicit_cluster_address()
    _patch_vllm_ray_executor_sample_tokens_no_compiled_dag()
    _patch_vllm_gpu_worker_kv_debug_info()
    _patch_vllm_qwen35_text_only_adapter()
    # Keep worker/runtime LoRA fixes, but do not alter vLLM's native startup
    # profiling path. Knob sizing depends on upstream accounting:
    # `weights_memory + peak_activation_memory + non_torch_increase`.
    if _env_flag("MINT_VLLM_FULLY_SHARDED_LORAS", default=False):
        _patch_vllm_fused_moe_slice_for_fully_sharded_loras()
        _patch_vllm_fused_moe_lora_use_torch_dist_tp_collectives()


_patch_torch_ld_library_path()
_patch_multiprocessing_executable()
_patch_cv2_typing_shadow()
_patch_ray_placement_group_bundle_cache()
_maybe_log_vllm_child_startup()
_apply_vllm_worker_patches()
