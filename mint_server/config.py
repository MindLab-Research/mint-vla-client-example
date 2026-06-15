"""Server configuration."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config_file import MintConfigFile

from .runtime_env import (
    build_runtime_pythonpath,
    env_get as _runtime_env_get,
    env_nonempty as _runtime_env_nonempty,
    join_pythonpath,
)
from .checkpoints import DEFAULT_PERSISTENT_CHECKPOINTS_DIR, DEFAULT_RUNTIME_CHECKPOINTS_DIR
from .config_hydration import hydrate_from_config_actor

hydrate_from_config_actor()


def _env_nonempty(environ: dict[str, str], name: str) -> str | None:
    return _runtime_env_nonempty(environ, name)


def _env_get(environ: dict[str, str], name: str, default: str = "") -> str:
    value = _runtime_env_get(environ, name)
    return default if value is None else value


def _resolve_env_or_config(name: str, env_value: str | None, file_value: str | None) -> str:
    if env_value and file_value and env_value != file_value:
        raise RuntimeError(
            f"{name} mismatch between environment and config file: env={env_value!r} config={file_value!r}"
        )
    return env_value or file_value or ""


def _parse_bool(s: str) -> bool:
    return str(s).strip().lower() in ("true", "1", "yes", "y", "on")


def _default_task_state_store_db_path(*, auth_enabled: bool) -> str:
    if auth_enabled:
        return "/vePFS-Mindverse/share/mint/prod/data/task-state/task_state.sqlite3"
    return "/vePFS-Mindverse/share/mint/dev/data/task-state/task_state.sqlite3"


def _deployment_env_for_defaults(environ: dict[str, str], *, auth_enabled: bool) -> str:
    raw = _env_nonempty(environ, "MINT_DEPLOYMENT_ENV")
    if raw is not None:
        return raw
    return "prod" if auth_enabled else "dev"


def _default_supervisor_state_db_path(deployment_env: str) -> str:
    env = str(deployment_env or "").strip() or "dev"
    return f"/vePFS-Mindverse/share/mint/{env}/runtime/supervisor_state.sqlite3"


def _load_config_file_for_process(environ: dict[str, str]) -> tuple[str | None, object | None]:
    path = _env_nonempty(environ, "MINT_CONFIG_PATH")
    if not path:
        return None, None
    try:
        from .config_file import load_mint_config_file
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "MINT_CONFIG_PATH is set but config parsing dependencies are missing "
            f"(missing module: {e.name!r}). Install pydantic on this runtime or unset MINT_CONFIG_PATH."
        ) from e
    return path, load_mint_config_file(path)


_CONFIG_PATH, _CONFIG_FILE = _load_config_file_for_process(os.environ)

# Ray namespace for all server-owned actors (vLLM, Megatron, trainer pools).
# Override for concurrent dev runs on a shared Ray cluster.
#
# Compatibility aliases are accepted by the env helper, but `MINT_*` is canonical.
_env_ray_ns = _env_nonempty(os.environ, "MINT_RAY_NAMESPACE")
_file_ray_ns = _CONFIG_FILE.ray.namespace if _CONFIG_FILE is not None else None
if _env_ray_ns and _file_ray_ns and _env_ray_ns != _file_ray_ns:
    raise RuntimeError(
        "Ray namespace mismatch between environment and config file: "
        f"env={_env_ray_ns!r} config={_file_ray_ns!r}"
    )
RAY_NAMESPACE = _env_ray_ns or _file_ray_ns or "mint"

# PFS paths for Ray worker runtime_env
# NOTE: vLLM requires PyTorch 2.9.0, which requires NCCL 2.21+
# System has NCCL 2.x (older) - cannot use PFS PyTorch 2.9.0
# MoE LoRA blocked until Docker image upgraded with newer CUDA stack
#
# Default to the *current* repo root so Ray actors use the same code as the
# running API server deployment (dev/prod/aliyun).
#
_file_mint_code_root = _CONFIG_FILE.paths.mint_code_root if _CONFIG_FILE is not None else None
MINT_CODE_ROOT = _resolve_env_or_config(
    "MINT_CODE_ROOT",
    _env_nonempty(os.environ, "MINT_CODE_ROOT"),
    _file_mint_code_root,
)

# Canonical runtime env root. This contains:
# - `site-packages/` for shared pure-Python runtime deps
# - `src/Megatron-Bridge`, `src/verl`, `src/Megatron-LM` pinned source trees
# - `host-venv/bin/python` as the thin host interpreter for API-server startup
_file_pfs_runtime_env_root = _CONFIG_FILE.paths.pfs_runtime_env_root if _CONFIG_FILE is not None else None
PFS_RUNTIME_ENV_ROOT = _resolve_env_or_config(
    "PFS_RUNTIME_ENV_ROOT",
    _env_nonempty(os.environ, "PFS_RUNTIME_ENV_ROOT"),
    _file_pfs_runtime_env_root,
)

# Toggle to use Megatron-Bridge export_adapter_weights API instead of custom implementation
_file_use_lora_export = _CONFIG_FILE.megatron_bridge.use_mbridge_lora_export if _CONFIG_FILE is not None else None
_env_use_lora_export = _env_nonempty(os.environ, "MINT_USE_MBRIDGE_LORA_EXPORT")
USE_MBRIDGE_LORA_EXPORT = (
    _parse_bool(_env_use_lora_export)
    if _env_use_lora_export is not None
    else (bool(_file_use_lora_export) if _file_use_lora_export is not None else False)
)

# HuggingFace modules path for trust_remote_code models (K2, etc.)
# Custom model code is cached here when models are first loaded
_file_pfs_hf_modules_path = _CONFIG_FILE.paths.pfs_hf_modules_path if _CONFIG_FILE is not None else None
PFS_HF_MODULES_PATH = _resolve_env_or_config(
    "PFS_HF_MODULES_PATH",
    _env_nonempty(os.environ, "PFS_HF_MODULES_PATH"),
    _file_pfs_hf_modules_path,
)

MINT_ACTOR_EXTRA_PYTHONPATH = _env_nonempty(os.environ, "MINT_ACTOR_EXTRA_PYTHONPATH") or ""

def ensure_runtime_env_configured() -> str:
    if not PFS_RUNTIME_ENV_ROOT:
        raise RuntimeError("PFS_RUNTIME_ENV_ROOT must be set")
    if not MINT_CODE_ROOT:
        raise RuntimeError("MINT_CODE_ROOT must be set")
    if not PFS_HF_MODULES_PATH:
        raise RuntimeError("PFS_HF_MODULES_PATH must be set")
    return PFS_RUNTIME_ENV_ROOT


PFS_PYTHONPATH = (
    join_pythonpath(
        MINT_ACTOR_EXTRA_PYTHONPATH,
        build_runtime_pythonpath(
            env_root=PFS_RUNTIME_ENV_ROOT,
            mint_code_root=MINT_CODE_ROOT,
            pfs_hf_modules_path=PFS_HF_MODULES_PATH,
        ),
    )
    if PFS_RUNTIME_ENV_ROOT and MINT_CODE_ROOT and PFS_HF_MODULES_PATH
    else ""
)

# OTEL env vars forwarded into Ray actors so actor-side logging/tracing
# can use the same collector/auth as the API server process.
_OTEL_FORWARD_KEYS = (
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_EXPORTER_OTLP_INSECURE",
    "OTEL_SERVICE_NAME",
    "OTEL_RESOURCE_ATTRIBUTES",
    "OTEL_LOG_LEVEL",
    "MINT_APMPLUS_APP_KEY",
    "MINT_DEPLOYMENT_ENV",
    "MINT_CLUSTER_ID",
)


def otel_env_vars() -> dict[str, str]:
    """Return non-empty OTEL env vars for Ray runtime_env injection."""
    out: dict[str, str] = {}
    for k in _OTEL_FORWARD_KEYS:
        v = _env_nonempty(os.environ, k)
        if v is not None:
            out[k] = v
    # Support legacy alias used by some deployments' .env files.
    app_key = _env_nonempty(os.environ, "MINT_APMPLUS_APP_KEY") or _env_nonempty(
        os.environ, "OTEL_APMPLUS_APP_KEY"
    )
    if app_key is not None:
        out["MINT_APMPLUS_APP_KEY"] = app_key
    return out


def preferred_control_plane_resources(cluster_resources: dict[str, float] | None) -> dict[str, float] | None:
    if not cluster_resources:
        return None
    configured_node_ip = _env_nonempty(os.environ, "MINT_CONTROL_PLANE_NODE_IP")
    if configured_node_ip is not None:
        configured_node_key = f"node:{configured_node_ip}"
        if configured_node_key in cluster_resources:
            return {configured_node_key: 0.001}
        if "node:__internal_head__" in cluster_resources:
            return {"node:__internal_head__": 0.001}
        return None
    if "node:__internal_head__" in cluster_resources:
        return {"node:__internal_head__": 0.001}
    try:
        from ray.util import get_node_ip_address

        driver_node_key = f"node:{get_node_ip_address()}"
    except Exception:
        driver_node_key = ""
    if driver_node_key and driver_node_key in cluster_resources:
        return {driver_node_key: 0.001}
    return None

def _env_nonempty_any(environ: dict[str, str], *names: str) -> tuple[str | None, str | None]:
    for name in names:
        value = _env_nonempty(environ, name)
        if value is not None:
            return value, name
    return None, None


_RAY_ATTACH_RUNTIME_ENV_KEYS = frozenset(
    {
        "MINT_RAY_TEMP_DIR",
        "MINT_RAY_NODE_IP_ADDRESS",
        "RAY_TMPDIR",
        "TMPDIR",
        "TMP",
        "TEMP",
        "RAY_ADDRESS",
        "RAY_CLIENT_ADDRESS",
        "MINT_RAY_CLIENT_ADDRESS",
    }
)


def actor_runtime_env_vars(
    *,
    pythonpath: str,
    extra: dict[str, str] | None = None,
    include_config_snapshot: bool = True,
    include_ray_attach_hints: bool = True,
) -> dict[str, str]:
    if not PFS_RUNTIME_ENV_ROOT:
        raise RuntimeError("PFS_RUNTIME_ENV_ROOT is required")
    if not MINT_CODE_ROOT:
        raise RuntimeError("MINT_CODE_ROOT is required")
    if not PFS_HF_MODULES_PATH:
        raise RuntimeError("PFS_HF_MODULES_PATH is required")
    from .ray_utils import strict_ray_gcs_address

    direct_ray_address = strict_ray_gcs_address()
    if include_ray_attach_hints and direct_ray_address is None:
        raise RuntimeError("MINT_RAY_GCS_ADDRESS is required")

    out = {
        "MINT_RAY_NAMESPACE": RAY_NAMESPACE,
        "PYTHONPATH": pythonpath,
        "PFS_RUNTIME_ENV_ROOT": PFS_RUNTIME_ENV_ROOT,
        "MINT_CODE_ROOT": MINT_CODE_ROOT,
        "PFS_HF_MODULES_PATH": PFS_HF_MODULES_PATH,
    }
    if direct_ray_address is not None:
        out["MINT_RAY_GCS_ADDRESS"] = direct_ray_address
    config_actor_name = _env_nonempty(os.environ, "MINT_CONFIG_ACTOR_NAME")
    if config_actor_name is not None:
        out["MINT_CONFIG_ACTOR_NAME"] = config_actor_name
    supervisor_actor_name = _env_nonempty(os.environ, "MINT_MODEL_ACTOR_SUPERVISOR_ACTOR_NAME")
    if supervisor_actor_name is not None:
        out["MINT_MODEL_ACTOR_SUPERVISOR_ACTOR_NAME"] = supervisor_actor_name
    config_path = _env_nonempty(os.environ, "MINT_CONFIG_PATH")
    if config_path is not None:
        out["MINT_CONFIG_PATH"] = config_path
    for key in (
        "MINT_ACTOR_LD_LIBRARY_PATH",
        "MINT_RAY_HEAD_ADDRESS_PATH",
        "MINT_RAY_NODE_IP_ADDRESS",
        "MINT_CONTROL_PLANE_NODE_IP",
        "MINT_RAY_TEMP_DIR",
        "MINT_RAY_JOB_WORKING_DIR",
        "MINT_RAY_WORKING_DIR",
        "MINT_RAY_PY_MODULES_CSV",
        "MINT_ACTOR_EXTRA_PYTHONPATH",
        "MINT_MODEL_WORK_SCHEDULER_USE_TASK_STATE_STORE",
        "MINT_TASK_STATE_STORE_DB_PATH",
        "MINT_FUTURE_STATE_STORE_DB_PATH",
    ):
        if not include_ray_attach_hints and key in _RAY_ATTACH_RUNTIME_ENV_KEYS:
            continue
        if include_ray_attach_hints and key in _RAY_ATTACH_RUNTIME_ENV_KEYS:
            continue
        value = _env_nonempty(os.environ, key)
        if value is not None:
            out[key] = value
    if extra:
        out.update(extra)
    if not include_ray_attach_hints:
        for key in _RAY_ATTACH_RUNTIME_ENV_KEYS:
            # Ray runtime_env env_vars overlay the job/worker environment; they
            # do not reliably delete inherited variables.  Empty values make
            # env_nonempty()/Ray attach helpers treat these as absent inside
            # detached actor workers, preventing nested ray.init/direct attach
            # attempts from a Ray worker process.
            out[key] = ""
    if include_config_snapshot:
        out["MINT_CONFIG_ACTOR_HYDRATE"] = "1"
    return out

def _runtime_env_value_is_uri(value: str) -> bool:
    head = str(value or "").split("://", 1)
    return len(head) == 2 and bool(head[0]) and all(ch.isalnum() or ch in "+-." for ch in head[0])


def _actor_runtime_env_allows_local_paths() -> bool:
    for name in ("MINT_RAY_CLIENT_ADDRESS", "RAY_CLIENT_ADDRESS"):
        value = _env_nonempty(os.environ, name)
        if value and value.startswith("ray://"):
            return False
    return True


def actor_runtime_env(
    *,
    pythonpath: str,
    extra: dict[str, str] | None = None,
    include_config_snapshot: bool = True,
    include_ray_attach_hints: bool = True,
) -> dict[str, object]:
    runtime_env: dict[str, object] = {
        "env_vars": actor_runtime_env_vars(
            pythonpath=pythonpath,
            extra=extra,
            include_config_snapshot=include_config_snapshot,
            include_ray_attach_hints=include_ray_attach_hints,
        )
    }
    allow_local_paths = _actor_runtime_env_allows_local_paths()
    py_modules_csv = _env_nonempty(os.environ, "MINT_RAY_PY_MODULES_CSV")
    if py_modules_csv:
        py_modules = [x.strip() for x in py_modules_csv.split(",") if x.strip()]
        if allow_local_paths:
            runtime_env["py_modules"] = py_modules
        else:
            uri_modules = [x for x in py_modules if _runtime_env_value_is_uri(x)]
            if uri_modules:
                runtime_env["py_modules"] = uri_modules
    working_dir = _env_nonempty(os.environ, "MINT_RAY_WORKING_DIR")
    # Ray Client accepts local-path working_dir only at ray.init(job) level.
    if working_dir and (allow_local_paths or _runtime_env_value_is_uri(working_dir)):
        runtime_env["working_dir"] = working_dir
    if not include_ray_attach_hints:
        preferred_python = (preferred_vllm_python_executable() or "").strip()
        if preferred_python:
            runtime_env["py_executable"] = preferred_python
    return runtime_env


def detached_actor_resource_key(ray_module: Any | None = None) -> str | None:
    try:
        cluster_resources = (ray_module or __import__("ray")).cluster_resources()
    except Exception:
        return None
    preferred = preferred_control_plane_resources(cluster_resources)
    if preferred is not None:
        return next(iter(preferred))
    return None


def apply_detached_actor_resources(
    options: dict[str, object],
    ray_module: Any | None = None,
    *,
    pin_to_control_plane: bool = True,
) -> None:
    if not pin_to_control_plane:
        return
    key = detached_actor_resource_key(ray_module)
    if key is not None:
        options["resources"] = {key: 0.001}


def _worker_visible_py_executable(path: str | None) -> str | None:
    raw = str(path or "").strip()
    if not raw:
        return None

    job_working_dir = _env_nonempty(os.environ, "MINT_RAY_JOB_WORKING_DIR")
    if not job_working_dir:
        return raw

    try:
        rel = Path(raw).resolve().relative_to(Path(job_working_dir).resolve())
    except Exception:
        return raw

    rel_path = f"./{rel.as_posix()}"
    # Ray Client uploads `working_dir` as a package on the cluster. Workers cannot
    # see the driver's absolute repo path. `.py` wrappers also lose their executable
    # bit after packaging on this cluster, so invoke them through `python`.
    if rel.suffix == ".py":
        return f"python {rel_path}"
    return rel_path


def preferred_vllm_python_executable() -> str | None:
    explicit = _env_nonempty(os.environ, "MINT_VLLM_CHILD_PYTHON_EXECUTABLE")
    if explicit:
        worker_visible = _worker_visible_py_executable(explicit)
        if worker_visible != explicit:
            return worker_visible
        if not Path(explicit).exists():
            raise RuntimeError(f"MINT_VLLM_CHILD_PYTHON_EXECUTABLE does not exist: {explicit}")
        return worker_visible
    if MINT_CODE_ROOT:
        candidate = Path(MINT_CODE_ROOT) / "scripts" / "vllm_worker_python.py"
        if candidate.exists():
            return _worker_visible_py_executable(str(candidate))
    return _worker_visible_py_executable(_env_nonempty(os.environ, "MINT_VLLM_CHILD_PYTHON_EXECUTABLE"))


def preferred_torch_lib_dirs(environ: dict[str, str] | None = None) -> list[str]:
    """Return torch lib directories in priority order for this Python runtime."""
    environ = os.environ if environ is None else environ
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    env_root = _env_nonempty(environ, "PFS_RUNTIME_ENV_ROOT") or PFS_RUNTIME_ENV_ROOT
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


def actor_ld_library_path() -> str:
    """Return the library path Ray actors should use on GPU workers.

    Do not inherit the API host's LD_LIBRARY_PATH by default. The host may be a
    CPU-only bootstrap environment with incompatible torch libs.
    """
    override = _env_nonempty(os.environ, "MINT_ACTOR_LD_LIBRARY_PATH")
    if override is not None:
        return override
    return ":".join(
        [
            *preferred_torch_lib_dirs(),
            "/usr/local/cuda/compat/lib",
            "/usr/local/nvidia/lib",
            "/usr/local/nvidia/lib64",
            "/usr/local/cuda/lib64",
        ]
    )

# When false (default), reject requests for base_model not in list_supported_models().
ALLOW_UNSUPPORTED_MODELS = _parse_bool(_env_nonempty(os.environ, "MINT_ALLOW_UNSUPPORTED_MODELS") or "false")


@dataclass
class ServerConfig:
    """Configuration for mint-server."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Authentication
    api_key: str = ""  # Deprecated; retained for signed-download token compatibility.
    internal_api_token: str = ""  # Shared token for trusting platform-forwarded identity headers.

    # Usage billing
    usage_backend: str = "postgres"  # only postgres is supported
    usage_pg_dsn: str = ""
    usage_pg_host: str = ""
    usage_pg_port: int = 5432
    usage_pg_database: str = "mint_billing"
    usage_pg_user: str = "mint_user"
    usage_pg_password: str = ""
    usage_pg_pool_min: int = 10
    usage_pg_pool_max: int = 30
    usage_write_timeout_ms: int = 2000
    usage_pg_table: str = "usage_event"
    checkpoint_index_pg_dsn: str = ""
    checkpoint_index_write_timeout_ms: int = 2000
    # Model settings (no default model - clients specify per-request)
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1  # For MoE: EP = TP * DP
    gpu_memory_utilization: float = 0.85
    max_model_len: int | None = None
    session_inactivity_timeout_s: float | None = None
    router_replay_mode: str = "disabled"  # Options: "disabled", "R2", "R3"

    # Multi-LoRA settings
    enable_multi_lora: bool = True  # Enable shared multi-LoRA engine
    max_loras: int = 64  # GPU slots for concurrent LoRA adapters (~2.5GB for 64 rank-32 Qwen-7B)
    max_cpu_loras: int = 1024  # CPU cache for evicted adapters
    max_lora_rank: int = 64  # Maximum supported LoRA rank
    vllm_attention_backend: str = "DUAL_CHUNK_FLASH_ATTN"

    # Sampling settings (routes/sampling.py)
    sampling_max_inflight_sample_tasks: int = 64
    sampling_max_pending_asample_per_apikey: int = 64
    sampling_max_inflight_per_principal_domain: int = 1024
    sampling_max_inflight_per_domain: int = 10240
    sampling_max_inflight_tokens_per_principal_domain: int = 0
    sampling_max_inflight_tokens_per_domain: int = 0
    sampling_inflight_admission_mode: str = "observe"
    sampling_max_concurrent_samples_per_request: int = 8
    sampling_sample_coalesce: bool = True
    sampling_sample_coalesce_window_ms: float = 50.0
    sampling_sample_coalesce_max_batch: int = 32
    sampling_sample_coalesce_max_samples: int = 16
    sampling_require_seq_id: bool = False

    # ModelActorSupervisor inventory settings
    model_actor_inventory_session_idle_timeout_s: int = 300

    # ModelActorSupervisor state backend. Desired/runtime actor source of truth
    # stays in config/Ray/reconcile; KV stores operational hints only.
    supervisor_state_backend: str = "memory"
    supervisor_state_db_path: str = "/vePFS-Mindverse/share/mint/dev/runtime/supervisor_state.sqlite3"
    supervisor_state_owner_ttl_s: float = 30.0
    supervisor_state_event_limit: int = 1000

    # Retrieve polling/cache settings. Durable terminal state lives in TaskStateStore.
    retrieve_future_hot_ttl_s: float = 300.0
    retrieve_future_grace_s: float = 600.0
    retrieve_future_min_poll_s: float = 1.0
    retrieve_future_wait_timeout_s: float = 20.0
    task_pending_ttl_s: float = 86400.0
    task_result_ttl_s: float = 86400.0
    task_tombstone_ttl_s: float = 604800.0

    # TaskStateStore settings (backend/task_state_store.py)
    task_state_store_actor_name: str = "mint_task_state_store"
    task_state_store_db_path: str = "/vePFS-Mindverse/share/mint/dev/data/task-state/task_state.sqlite3"
    task_state_store_owner_ttl_s: float = 30.0
    task_state_store_owner_renew_s: float = 10.0

    # Future-state RocksDB component owned by TaskStateStore actor.
    future_state_store_db_path: str = "/vePFS-Mindverse/share/mint/dev/data/future-state/futures.rocksdb"

    # Training settings (backend/verl_training.py)
    training_inactivity_timeout_s: int = 3600
    training_force_grad_checkpointing: bool = True
    training_enable_sdp: bool = True
    training_megatron_create_timeout_s: float = 1800.0
    training_dense_get_or_create_timeout_s: float = 1800.0
    training_dense_session_state_root: str = os.path.join(
        DEFAULT_RUNTIME_CHECKPOINTS_DIR,
        "dense_session_state",
    )
    training_reinit_lora_timeout_s: float = 0.0
    training_actor_ready_timeout_s: float | None = None
    training_remote_call_timeout_s: float | None = None

    # Docs / internal paths
    doc_path: str | None = None  # MINT_DOC_PATH
    checkpoint_dir: str = "/tos-mindverse/mint_checkpoints"  # MINT_CHECKPOINT_DIR

    # Config file (MINT_CONFIG_PATH)
    config_path: str | None = None

    @classmethod
    def from_env(cls) -> "ServerConfig":
        return cls.from_sources(environ=os.environ, config_path=None, config_file=None)

    @classmethod
    def from_sources(
        cls,
        *,
        environ: dict[str, str],
        config_path: str | None,
        config_file: MintConfigFile | None,
    ) -> "ServerConfig":
        """Load configuration from env vars + optional config file (env wins)."""
        api_key = _env_get(environ, "MINT_API_KEY", "")
        inactivity_s = _env_nonempty(environ, "MINT_SESSION_INACTIVITY_TIMEOUT_S") or _env_nonempty(
            environ, "MINT_INACTIVITY_TIMEOUT_S"
        )
        file_server = config_file.server if config_file is not None else None
        internal_api_token = _env_nonempty(environ, "MINT_INTERNAL_API_TOKEN") or (
            file_server.internal_api_token if file_server is not None else None
        )
        # Auth is disabled in dev unless a platform internal token is configured.
        auth_enabled = bool(internal_api_token)
        file_sampling = config_file.sampling if config_file is not None else None
        file_model_actor_inventory = config_file.model_actor_inventory if config_file is not None else None
        file_supervisor_state = config_file.supervisor_state if config_file is not None else None
        file_future = config_file.future if config_file is not None else None
        file_task_state_store = config_file.task_state_store if config_file is not None else None
        file_future_state_store = config_file.future_state_store if config_file is not None else None
        file_training = config_file.training if config_file is not None else None
        file_docs = config_file.docs if config_file is not None else None
        file_internal = config_file.internal if config_file is not None else None
        deployment_env = _deployment_env_for_defaults(environ, auth_enabled=auth_enabled)
        dense_session_state_default_root = os.path.join(
            _env_nonempty(environ, "MINT_RUNTIME_CHECKPOINT_DIR") or DEFAULT_RUNTIME_CHECKPOINTS_DIR,
            "dense_session_state",
        )

        def _pick_str(name: str, file_value: str | None, default: str) -> str:
            v = _env_nonempty(environ, name)
            return v if v is not None else (file_value if file_value is not None else default)

        def _pick_str_alias(primary: str, aliases: tuple[str, ...], file_value: str | None, default: str) -> str:
            v, _source = _env_nonempty_any(environ, primary, *aliases)
            return v if v is not None else (file_value if file_value is not None else default)

        def _pick_int(name: str, file_value: int | None, default: int) -> int:
            v = _env_nonempty(environ, name)
            return int(v) if v is not None else (int(file_value) if file_value is not None else int(default))

        def _pick_float(name: str, file_value: float | None, default: float) -> float:
            v = _env_nonempty(environ, name)
            return float(v) if v is not None else (float(file_value) if file_value is not None else float(default))

        def _pick_float_alias(primary: str, aliases: tuple[str, ...], file_value: float | None, default: float) -> float:
            v, _source = _env_nonempty_any(environ, primary, *aliases)
            return float(v) if v is not None else (float(file_value) if file_value is not None else float(default))

        def _pick_bool(name: str, file_value: bool | None, default: bool) -> bool:
            v = _env_nonempty(environ, name)
            return _parse_bool(v) if v is not None else (bool(file_value) if file_value is not None else bool(default))

        max_model_len_env = _env_nonempty(environ, "MINT_MAX_MODEL_LEN")
        max_model_len = int(max_model_len_env) if max_model_len_env is not None else (file_server.max_model_len if file_server is not None else None)

        inactivity_from_file = file_server.session_inactivity_timeout_s if file_server is not None else None
        actor_ready_timeout_env = _env_nonempty(environ, "MINT_ACTOR_READY_TIMEOUT_S")
        actor_ready_timeout_s = (
            float(actor_ready_timeout_env)
            if actor_ready_timeout_env is not None
            else (float(file_training.actor_ready_timeout_s) if file_training is not None and file_training.actor_ready_timeout_s is not None else None)
        )
        remote_call_timeout_env = _env_nonempty(environ, "MINT_TRAINING_REMOTE_CALL_TIMEOUT_S")
        remote_call_timeout_s = (
            float(remote_call_timeout_env)
            if remote_call_timeout_env is not None
            else (
                float(file_training.remote_call_timeout_s)
                if file_training is not None and file_training.remote_call_timeout_s is not None
                else None
            )
        )

        cfg = cls(
            host=_pick_str("MINT_HOST", file_server.host if file_server is not None else None, "0.0.0.0"),
            port=_pick_int("MINT_PORT", file_server.port if file_server is not None else None, 8000),
            api_key=api_key,
            internal_api_token=_pick_str(
                "MINT_INTERNAL_API_TOKEN",
                file_server.internal_api_token if file_server is not None else None,
                "",
            ),
            usage_backend=_pick_str(
                "MINT_USAGE_BACKEND",
                file_server.usage_backend if file_server is not None else None,
                "postgres",
            ).lower(),
            usage_pg_dsn=(
                _pick_str(
                    "MINT_USAGE_PG_DSN",
                    file_server.usage_pg_dsn if file_server is not None else None,
                    "",
                )
                or (
                    (
                        "postgresql://"
                        f"{_pick_str('MINT_USAGE_PG_USER', file_server.usage_pg_user if file_server is not None else None, 'mint_user')}:"
                        f"{_pick_str('MINT_USAGE_PG_PASSWORD', file_server.usage_pg_password if file_server is not None else None, '')}@"
                        f"{_pick_str('MINT_USAGE_PG_HOST', file_server.usage_pg_host if file_server is not None else None, '')}:"
                        f"{_pick_int('MINT_USAGE_PG_PORT', file_server.usage_pg_port if file_server is not None else None, 5432)}/"
                        f"{_pick_str('MINT_USAGE_PG_DATABASE', file_server.usage_pg_database if file_server is not None else None, 'mint_billing')}"
                    )
                    if _pick_str("MINT_USAGE_PG_HOST", file_server.usage_pg_host if file_server is not None else None, "")
                    else ""
                )
            ),
            usage_pg_host=_pick_str(
                "MINT_USAGE_PG_HOST",
                file_server.usage_pg_host if file_server is not None else None,
                "",
            ),
            usage_pg_port=_pick_int(
                "MINT_USAGE_PG_PORT",
                file_server.usage_pg_port if file_server is not None else None,
                5432,
            ),
            usage_pg_database=_pick_str(
                "MINT_USAGE_PG_DATABASE",
                file_server.usage_pg_database if file_server is not None else None,
                "mint_billing",
            ),
            usage_pg_user=_pick_str(
                "MINT_USAGE_PG_USER",
                file_server.usage_pg_user if file_server is not None else None,
                "mint_user",
            ),
            usage_pg_password=_pick_str(
                "MINT_USAGE_PG_PASSWORD",
                file_server.usage_pg_password if file_server is not None else None,
                "",
            ),
            usage_pg_pool_min=_pick_int(
                "MINT_USAGE_PG_POOL_MIN",
                file_server.usage_pg_pool_min if file_server is not None else None,
                10,
            ),
            usage_pg_pool_max=_pick_int(
                "MINT_USAGE_PG_POOL_MAX",
                file_server.usage_pg_pool_max if file_server is not None else None,
                30,
            ),
            usage_write_timeout_ms=_pick_int(
                "MINT_USAGE_WRITE_TIMEOUT_MS",
                file_server.usage_write_timeout_ms if file_server is not None else None,
                2000,
            ),
            usage_pg_table=_pick_str(
                "MINT_USAGE_PG_TABLE",
                file_server.usage_pg_table if file_server is not None else None,
                "usage_event",
            ),
            checkpoint_index_pg_dsn=_pick_str(
                "MINT_CHECKPOINT_INDEX_PG_DSN",
                file_server.checkpoint_index_pg_dsn if file_server is not None else None,
                "",
            ),
            checkpoint_index_write_timeout_ms=_pick_int(
                "MINT_CHECKPOINT_INDEX_WRITE_TIMEOUT_MS",
                file_server.checkpoint_index_write_timeout_ms if file_server is not None else None,
                2000,
            ),
            tensor_parallel_size=_pick_int("MINT_TP_SIZE", file_server.tensor_parallel_size if file_server is not None else None, 1),
            data_parallel_size=_pick_int("MINT_DP_SIZE", file_server.data_parallel_size if file_server is not None else None, 1),
            gpu_memory_utilization=_pick_float(
                "MINT_GPU_MEM_UTIL", file_server.gpu_memory_utilization if file_server is not None else None, 0.85
            ),
            max_model_len=max_model_len,
            session_inactivity_timeout_s=float(inactivity_s)
            if inactivity_s
            else (float(inactivity_from_file) if inactivity_from_file is not None else None),
            # Multi-LoRA settings
            enable_multi_lora=_pick_bool(
                "MINT_ENABLE_MULTI_LORA", file_server.enable_multi_lora if file_server is not None else None, True
            ),
            max_loras=_pick_int("MINT_MAX_LORAS", file_server.max_loras if file_server is not None else None, 64),
            max_cpu_loras=_pick_int(
                "MINT_MAX_CPU_LORAS", file_server.max_cpu_loras if file_server is not None else None, 1024
            ),
            max_lora_rank=_pick_int("MINT_MAX_LORA_RANK", file_server.max_lora_rank if file_server is not None else None, 64),
            vllm_attention_backend=_pick_str(
                "MINT_VLLM_ATTENTION_BACKEND",
                file_server.vllm_attention_backend if file_server is not None else None,
                "DUAL_CHUNK_FLASH_ATTN",
            ),
            # Sampling settings
            sampling_max_inflight_sample_tasks=_pick_int(
                "MINT_MAX_INFLIGHT_SAMPLE_TASKS",
                file_sampling.max_inflight_sample_tasks if file_sampling is not None else None,
                64,
            ),
            sampling_max_pending_asample_per_apikey=_pick_int(
                "MINT_MAX_PENDING_ASAMPLE_PER_APIKEY",
                getattr(file_sampling, "max_pending_asample_per_apikey", None) if file_sampling is not None else None,
                64,
            ),
            sampling_max_inflight_per_principal_domain=_pick_int(
                "MINT_SAMPLING_MAX_INFLIGHT_PER_PRINCIPAL_DOMAIN",
                getattr(file_sampling, "max_inflight_per_principal_domain", None) if file_sampling is not None else None,
                1024,
            ),
            sampling_max_inflight_per_domain=_pick_int(
                "MINT_SAMPLING_MAX_INFLIGHT_PER_DOMAIN",
                getattr(file_sampling, "max_inflight_per_domain", None) if file_sampling is not None else None,
                10240,
            ),
            sampling_max_inflight_tokens_per_principal_domain=_pick_int(
                "MINT_SAMPLING_MAX_INFLIGHT_TOKENS_PER_PRINCIPAL_DOMAIN",
                getattr(file_sampling, "max_inflight_tokens_per_principal_domain", None)
                if file_sampling is not None
                else None,
                0,
            ),
            sampling_max_inflight_tokens_per_domain=_pick_int(
                "MINT_SAMPLING_MAX_INFLIGHT_TOKENS_PER_DOMAIN",
                getattr(file_sampling, "max_inflight_tokens_per_domain", None) if file_sampling is not None else None,
                0,
            ),
            sampling_inflight_admission_mode=_pick_str(
                "MINT_SAMPLING_INFLIGHT_ADMISSION_MODE",
                getattr(file_sampling, "inflight_admission_mode", None) if file_sampling is not None else None,
                "observe",
            ),
            sampling_max_concurrent_samples_per_request=_pick_int(
                "MINT_MAX_CONCURRENT_SAMPLES_PER_REQUEST",
                file_sampling.max_concurrent_samples_per_request if file_sampling is not None else None,
                8,
            ),
            sampling_sample_coalesce=_pick_bool(
                "MINT_SAMPLE_COALESCE",
                file_sampling.sample_coalesce if file_sampling is not None else None,
                True,
            ),
            sampling_sample_coalesce_window_ms=_pick_float(
                "MINT_SAMPLE_COALESCE_WINDOW_MS",
                file_sampling.sample_coalesce_window_ms if file_sampling is not None else None,
                50.0,
            ),
            sampling_sample_coalesce_max_batch=_pick_int(
                "MINT_SAMPLE_COALESCE_MAX_BATCH",
                file_sampling.sample_coalesce_max_batch if file_sampling is not None else None,
                32,
            ),
            sampling_sample_coalesce_max_samples=_pick_int(
                "MINT_SAMPLE_COALESCE_MAX_SAMPLES",
                file_sampling.sample_coalesce_max_samples if file_sampling is not None else None,
                16,
            ),
            sampling_require_seq_id=_pick_bool(
                "MINT_SAMPLE_REQUIRE_SEQ_ID",
                file_sampling.require_seq_id if file_sampling is not None else None,
                False,
            ),
            # ModelActorSupervisor inventory settings
            model_actor_inventory_session_idle_timeout_s=_pick_int(
                "MINT_MODEL_ACTOR_INVENTORY_SESSION_IDLE_TIMEOUT_S",
                file_model_actor_inventory.session_idle_timeout_s if file_model_actor_inventory is not None else None,
                300,
            ),
            supervisor_state_backend=_pick_str(
                "MINT_SUPERVISOR_STATE_BACKEND",
                file_supervisor_state.backend if file_supervisor_state is not None else None,
                "memory",
            ),
            supervisor_state_db_path=_pick_str(
                "MINT_SUPERVISOR_STATE_DB_PATH",
                file_supervisor_state.db_path if file_supervisor_state is not None else None,
                _default_supervisor_state_db_path(deployment_env),
            ),
            supervisor_state_owner_ttl_s=_pick_float(
                "MINT_SUPERVISOR_STATE_OWNER_TTL_S",
                file_supervisor_state.owner_ttl_s if file_supervisor_state is not None else None,
                30.0,
            ),
            supervisor_state_event_limit=_pick_int(
                "MINT_SUPERVISOR_STATE_EVENT_LIMIT",
                file_supervisor_state.event_limit if file_supervisor_state is not None else None,
                1000,
            ),
            # Retrieve settings
            retrieve_future_hot_ttl_s=_pick_float(
                "MINT_RETRIEVE_FUTURE_HOT_TTL_S",
                file_future.retrieve_future_hot_ttl_s if file_future is not None else None,
                300.0,
            ),
            retrieve_future_grace_s=_pick_float_alias(
                "MINT_RETRIEVE_FUTURE_GRACE_S",
                ("MINT_RETRIEVE_FUTURE_GRACE_S",),
                file_future.retrieve_future_grace_s if file_future is not None else None,
                600.0,
            ),
            retrieve_future_min_poll_s=_pick_float_alias(
                "MINT_RETRIEVE_FUTURE_MIN_POLL_S",
                ("MINT_RETRIEVE_FUTURE_MIN_POLL_S",),
                file_future.retrieve_future_min_poll_s if file_future is not None else None,
                1.0,
            ),
            retrieve_future_wait_timeout_s=_pick_float(
                "MINT_RETRIEVE_FUTURE_WAIT_TIMEOUT_S",
                file_future.retrieve_future_wait_timeout_s if file_future is not None else None,
                20.0,
            ),
            task_pending_ttl_s=_pick_float(
                "MINT_TASK_PENDING_TTL_S",
                file_future.task_pending_ttl_s if file_future is not None else None,
                86400.0,
            ),
            task_result_ttl_s=_pick_float(
                "MINT_TASK_RESULT_TTL_S",
                file_future.task_result_ttl_s if file_future is not None else None,
                86400.0,
            ),
            task_tombstone_ttl_s=_pick_float(
                "MINT_TASK_TOMBSTONE_TTL_S",
                file_future.task_tombstone_ttl_s if file_future is not None else None,
                604800.0,
            ),
            # TaskStateStore settings
            task_state_store_actor_name=_pick_str(
                "MINT_TASK_STATE_STORE_ACTOR_NAME",
                file_task_state_store.actor_name if file_task_state_store is not None else None,
                "mint_task_state_store",
            ),
            task_state_store_db_path=_pick_str(
                "MINT_TASK_STATE_STORE_DB_PATH",
                file_task_state_store.db_path if file_task_state_store is not None else None,
                _default_task_state_store_db_path(auth_enabled=auth_enabled),
            ),
            task_state_store_owner_ttl_s=_pick_float(
                "MINT_TASK_STATE_STORE_OWNER_TTL_S",
                file_task_state_store.owner_ttl_s if file_task_state_store is not None else None,
                30.0,
            ),
            task_state_store_owner_renew_s=_pick_float(
                "MINT_TASK_STATE_STORE_OWNER_RENEW_S",
                file_task_state_store.owner_renew_s if file_task_state_store is not None else None,
                10.0,
            ),
            future_state_store_db_path=_pick_str(
                "MINT_FUTURE_STATE_STORE_DB_PATH",
                file_future_state_store.db_path if file_future_state_store is not None else None,
                str(Path(_pick_str(
                    "MINT_TASK_STATE_STORE_DB_PATH",
                    file_task_state_store.db_path if file_task_state_store is not None else None,
                    _default_task_state_store_db_path(auth_enabled=auth_enabled),
                )).parent.parent / "future-state" / "futures.rocksdb"),
            ),
            # Training settings
            training_inactivity_timeout_s=_pick_int(
                "MINT_TRAINING_INACTIVITY_TIMEOUT",
                file_training.inactivity_timeout_s if file_training is not None else None,
                3600,
            ),
            training_force_grad_checkpointing=_pick_bool(
                "MINT_FORCE_GRAD_CHECKPOINTING",
                file_training.force_grad_checkpointing if file_training is not None else None,
                True,
            ),
            training_enable_sdp=_pick_bool(
                "MINT_ENABLE_SDP",
                file_training.enable_sdp if file_training is not None else None,
                True,
            ),
            training_megatron_create_timeout_s=_pick_float(
                "MINT_MEGATRON_CREATE_TIMEOUT_S",
                file_training.megatron_create_timeout_s if file_training is not None else None,
                1800.0,
            ),
            training_dense_get_or_create_timeout_s=_pick_float(
                "MINT_DENSE_GET_OR_CREATE_TIMEOUT_S",
                file_training.dense_get_or_create_timeout_s if file_training is not None else None,
                1800.0,
            ),
            training_dense_session_state_root=_pick_str(
                "MINT_DENSE_SESSION_STATE_ROOT",
                file_training.dense_session_state_root if file_training is not None else None,
                dense_session_state_default_root,
            ),
            training_reinit_lora_timeout_s=_pick_float(
                "MINT_REINIT_LORA_TIMEOUT_S",
                file_training.reinit_lora_timeout_s if file_training is not None else None,
                0.0,
            ),
            training_actor_ready_timeout_s=actor_ready_timeout_s,
            training_remote_call_timeout_s=remote_call_timeout_s,
            router_replay_mode=_pick_str(
                "MINT_ROUTER_REPLAY_MODE",
                None,
                "disabled",
            ),
            doc_path=_pick_str(
                "MINT_DOC_PATH",
                file_docs.doc_path if file_docs is not None else None,
                "",
            ) or None,
            checkpoint_dir=_pick_str(
                "MINT_CHECKPOINT_DIR",
                file_internal.checkpoint_dir if file_internal is not None else None,
                DEFAULT_PERSISTENT_CHECKPOINTS_DIR,
            ),
            config_path=config_path,
        )
        cfg.validate_deprecated_usage_config()
        return cfg

    @property
    def auth_enabled(self) -> bool:
        """Check if any authentication is configured."""
        return bool(self.internal_api_token)

    @property
    def download_token_secret(self) -> str:
        """Secret used for short-lived SDK archive download URLs."""
        return (self.internal_api_token or self.api_key or "").strip()

    def validate_deprecated_usage_config(self) -> None:
        backend = str(self.usage_backend or "").strip().lower()
        if backend and backend not in {"postgres", "disabled", "noop"}:
            raise ValueError(
                f"Unsupported usage backend {self.usage_backend!r}; expected one of 'postgres', 'disabled', or 'noop'"
            )

# Global config instance
config = ServerConfig.from_sources(environ=os.environ, config_path=_CONFIG_PATH, config_file=_CONFIG_FILE)
