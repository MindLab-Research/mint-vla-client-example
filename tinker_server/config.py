"""Server configuration."""

from __future__ import annotations

import os
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config_file import TinkerConfigFile

from .runtime_env import (
    build_runtime_pythonpath,
    env_nonempty as _runtime_env_nonempty,
)
from .checkpoints import DEFAULT_RUNTIME_CHECKPOINTS_DIR


def _env_nonempty(environ: dict[str, str], name: str) -> str | None:
    return _runtime_env_nonempty(environ, name)


def _resolve_env_or_config(name: str, env_value: str | None, file_value: str | None) -> str:
    if env_value and file_value and env_value != file_value:
        raise RuntimeError(
            f"{name} mismatch between environment and config file: env={env_value!r} config={file_value!r}"
        )
    return env_value or file_value or ""


def _parse_bool(s: str) -> bool:
    return str(s).strip().lower() in ("true", "1", "yes", "y", "on")


def _default_future_replay_root_dir(*, auth_enabled: bool) -> str:
    if auth_enabled:
        return "/vePFS-Mindverse/share/mint-prod-data/future-replay"
    return "/vePFS-Mindverse/share/mint-prod-dev/future-replay"


def _load_config_file_for_process(environ: dict[str, str]) -> tuple[str | None, object | None]:
    path = _env_nonempty(environ, "TINKER_CONFIG_PATH")
    if not path:
        return None, None
    try:
        from .config_file import load_tinker_config_file
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "TINKER_CONFIG_PATH is set but config parsing dependencies are missing "
            f"(missing module: {e.name!r}). Install pydantic on this runtime or unset TINKER_CONFIG_PATH."
        ) from e
    return path, load_tinker_config_file(path)


_CONFIG_PATH, _CONFIG_FILE = _load_config_file_for_process(os.environ)

# Ray namespace for all server-owned actors (vLLM, Megatron, trainer pools).
# Override for concurrent dev runs on a shared Ray cluster.
#
# `MINT_RAY_NAMESPACE` is a legacy alias still used in some scripts; prefer
# `TINKER_RAY_NAMESPACE` but accept the alias as a fallback.
_env_ray_ns = _env_nonempty(os.environ, "TINKER_RAY_NAMESPACE") or _env_nonempty(os.environ, "MINT_RAY_NAMESPACE")
_file_ray_ns = _CONFIG_FILE.ray.namespace if _CONFIG_FILE is not None else None
if _env_ray_ns and _file_ray_ns and _env_ray_ns != _file_ray_ns:
    raise RuntimeError(
        "Ray namespace mismatch between environment and config file: "
        f"env={_env_ray_ns!r} config={_file_ray_ns!r}"
    )
RAY_NAMESPACE = _env_ray_ns or _file_ray_ns or "tinker"

# PFS paths for Ray worker runtime_env
# NOTE: vLLM requires PyTorch 2.9.0, which requires NCCL 2.21+
# System has NCCL 2.x (older) - cannot use PFS PyTorch 2.9.0
# MoE LoRA blocked until Docker image upgraded with newer CUDA stack
#
# Default to the *current* repo root so Ray actors use the same code as the
# running API server deployment (dev/prod/aliyun).
#
# Historical default hard-coded `/vePFS-Mindverse/share/code/tinker-server-auth`, which breaks
# non-volcano deployments (e.g. `tinker-server-aliyun`) by setting worker runtime_env PYTHONPATH
# to a non-existent code directory.
_file_pfs_tinker_path = _CONFIG_FILE.paths.pfs_tinker_path if _CONFIG_FILE is not None else None
PFS_TINKER_PATH = _resolve_env_or_config(
    "PFS_TINKER_PATH",
    _env_nonempty(os.environ, "PFS_TINKER_PATH"),
    _file_pfs_tinker_path,
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
_env_use_lora_export = _env_nonempty(os.environ, "USE_MBRIDGE_LORA_EXPORT")
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

def ensure_runtime_env_configured() -> str:
    if not PFS_RUNTIME_ENV_ROOT:
        raise RuntimeError("PFS_RUNTIME_ENV_ROOT must be set")
    if not PFS_TINKER_PATH:
        raise RuntimeError("PFS_TINKER_PATH must be set")
    if not PFS_HF_MODULES_PATH:
        raise RuntimeError("PFS_HF_MODULES_PATH must be set")
    return PFS_RUNTIME_ENV_ROOT


PFS_PYTHONPATH = (
    build_runtime_pythonpath(
        env_root=PFS_RUNTIME_ENV_ROOT,
        pfs_tinker_path=PFS_TINKER_PATH,
        pfs_hf_modules_path=PFS_HF_MODULES_PATH,
    )
    if PFS_RUNTIME_ENV_ROOT and PFS_TINKER_PATH and PFS_HF_MODULES_PATH
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


def _env_nonempty_any(environ: dict[str, str], *names: str) -> tuple[str | None, str | None]:
    for name in names:
        value = _env_nonempty(environ, name)
        if value is not None:
            return value, name
    return None, None


def actor_runtime_env_vars(*, pythonpath: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    if not PFS_RUNTIME_ENV_ROOT:
        raise RuntimeError("PFS_RUNTIME_ENV_ROOT is required")
    if not PFS_TINKER_PATH:
        raise RuntimeError("PFS_TINKER_PATH is required")
    if not PFS_HF_MODULES_PATH:
        raise RuntimeError("PFS_HF_MODULES_PATH is required")
    ray_address = _env_nonempty(os.environ, "RAY_ADDRESS")
    if ray_address is None:
        raise RuntimeError("RAY_ADDRESS is required")

    out = {
        "TINKER_RAY_NAMESPACE": RAY_NAMESPACE,
        "PYTHONPATH": pythonpath,
        "PFS_RUNTIME_ENV_ROOT": PFS_RUNTIME_ENV_ROOT,
        "PFS_TINKER_PATH": PFS_TINKER_PATH,
        "PFS_HF_MODULES_PATH": PFS_HF_MODULES_PATH,
        "RAY_ADDRESS": ray_address,
    }
    config_path = _env_nonempty(os.environ, "TINKER_CONFIG_PATH")
    if config_path is not None:
        out["TINKER_CONFIG_PATH"] = config_path
    for key in (
        "MINT_VLLM_CHILD_PYTHON_EXECUTABLE",
        "TINKER_ACTOR_LD_LIBRARY_PATH",
        "MINT_SFT_DIAG_FAIL",
        "MINT_REVERSE_KL_DIAG_FAIL",
        "TINKER_DENSE_SESSION_STATE_ROOT",
        "TINKER_RUNTIME_CHECKPOINT_DIR",
        "TINKER_LEGACY_DENSE_SESSION_STATE_ROOTS",
        "MINT_DETACHED_ACTOR_NODE_IP",
        "MINT_RAY_HEAD_ADDRESS_PATH",
    ):
        value = _env_nonempty(os.environ, key)
        if value is not None:
            out[key] = value
    for primary, aliases in (
        ("MINT_API_WORK_QUEUE_ACTOR_NAME", ("TINKER_API_WORK_QUEUE_ACTOR_NAME",)),
        ("MINT_CAPACITY_MANAGER_ACTOR_NAME", ("TINKER_CAPACITY_MANAGER_ACTOR_NAME",)),
    ):
        value, _source = _env_nonempty_any(os.environ, primary, *aliases)
        if value is not None:
            out[primary] = value
    if extra:
        out.update(extra)
    return out

def actor_runtime_env(*, pythonpath: str, extra: dict[str, str] | None = None) -> dict[str, object]:
    runtime_env: dict[str, object] = {
        "env_vars": actor_runtime_env_vars(pythonpath=pythonpath, extra=extra)
    }
    py_modules_csv = _env_nonempty(os.environ, "MINT_RAY_PY_MODULES_CSV")
    if py_modules_csv:
        runtime_env["py_modules"] = [x.strip() for x in py_modules_csv.split(",") if x.strip()]
    working_dir = _env_nonempty(os.environ, "MINT_RAY_WORKING_DIR")
    if working_dir:
        runtime_env["working_dir"] = working_dir
    return runtime_env


def detached_actor_resource_key(ray_module: Any | None = None) -> str | None:
    pinned_ip = _env_nonempty(os.environ, "MINT_DETACHED_ACTOR_NODE_IP")
    if pinned_ip is not None:
        return f"node:{pinned_ip}"
    try:
        cluster_resources = (ray_module or __import__("ray")).cluster_resources()
    except Exception:
        return None
    if "node:__internal_head__" in cluster_resources:
        return "node:__internal_head__"
    return None


def apply_detached_actor_resources(options: dict[str, object], ray_module: Any | None = None) -> None:
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
        return explicit
    if PFS_TINKER_PATH:
        candidate = Path(PFS_TINKER_PATH) / "scripts" / "vllm_worker_python.py"
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
    override = _env_nonempty(os.environ, "TINKER_ACTOR_LD_LIBRARY_PATH")
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
ALLOW_UNSUPPORTED_MODELS = _parse_bool(_env_nonempty(os.environ, "ALLOW_UNSUPPORTED_MODELS") or "false")


@dataclass
class ServerConfig:
    """Configuration for tinker-server."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Authentication
    api_key: str = ""  # Hardcoded API key (legacy). If set, accepts this key directly.
    token_secret_key: str = ""  # Secret for sk- token decryption. If set, accepts encrypted tokens.
    internal_api_token: str = ""  # Shared token for trusting gateway-forwarded billing headers.

    # Usage billing
    usage_log_dir: str = "/tmp/tinker_usage"  # active JSONL sink
    usage_backend: str = "postgres"  # deprecated, ignored by the producer path
    usage_pg_dsn: str = ""  # deprecated, ignored by the producer path
    usage_pg_host: str = ""
    usage_pg_port: int = 5432
    usage_pg_database: str = "mint_billing"
    usage_pg_user: str = "mint_user"
    usage_pg_password: str = ""
    usage_pg_pool_min: int = 10
    usage_pg_pool_max: int = 30
    usage_write_timeout_ms: int = 2000
    usage_pg_table: str = "billing.usage_event"
    skip_actor_cleanup: bool = False  # MINT_SKIP_ACTOR_CLEANUP

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
    sampling_max_concurrent_samples_per_request: int = 8
    sampling_sample_coalesce: bool = True
    sampling_sample_coalesce_window_ms: float = 50.0
    sampling_sample_coalesce_max_batch: int = 32
    sampling_sample_coalesce_max_samples: int = 16
    sampling_require_seq_id: bool = False

    # ResourcePool settings (backend/resource_pool.py)
    resource_pool_min_actor_age_s: int = 300
    resource_pool_session_idle_timeout_s: int = 300

    # Future store settings (backend/future_store.py)
    future_store_actor_name: str = "tinker_future_store"
    future_store_ttl_s: float = 86400.0
    # Maximum time a request may stay QUEUED (not RUNNING) before being marked FAILED.
    # This is a safety net for worker/queue failures; it is not the execution timeout.
    future_store_queue_ttl_s: float = 7 * 86400.0
    future_store_done_ttl_s: float = 7200.0
    future_store_tombstone_ttl_s: float = 300.0
    future_replay_root_dir: str = "/vePFS-Mindverse/share/mint-prod-dev/future-replay"
    future_replay_hot_ttl_s: float = 60.0
    future_replay_disk_ttl_s: float = 86400.0
    future_replay_sweep_interval_s: float = 21600.0

    # Admission control + API work queue (issue #84)
    capacity_manager_actor_name: str = "tinker_capacity_manager"
    api_work_queue_actor_name: str = "tinker_api_work_queue"
    capacity_queue_bytes_budget: int = 512 * 1024 * 1024
    api_work_queue_num_workers: int = 128
    api_work_queue_reap_interval_s: float = 5.0

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

    # Persistent-prewarm settings (app.py)
    prewarm_persistent_models_csv: str = ""
    prewarm_train_lora_rank: int = 16
    prewarm_train_lr: float = 5e-5
    prewarm_megatron_ready_timeout_s: float = 3600.0
    prewarm_enable_training: bool = True
    prewarm_enable_inference: bool = True

    # Docs / internal paths
    doc_path: str | None = None  # MINT_DOC_PATH
    checkpoint_dir: str = "/tos-mindverse/tinker_checkpoints"  # TINKER_CHECKPOINT_DIR

    # Config file (TINKER_CONFIG_PATH)
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
        config_file: TinkerConfigFile | None,
    ) -> "ServerConfig":
        """Load configuration from env vars + optional config file (env wins)."""
        api_key = environ.get("TINKER_API_KEY", "")
        token_secret_key = environ.get("TINKER_TOKEN_SECRET_KEY", "")
        # Auth disabled (dev mode) if neither api_key nor token_secret_key is set
        auth_enabled = bool(api_key or token_secret_key)
        inactivity_s = environ.get("TINKER_SESSION_INACTIVITY_TIMEOUT_S") or environ.get("TINKER_INACTIVITY_TIMEOUT_S")
        file_server = config_file.server if config_file is not None else None
        file_sampling = config_file.sampling if config_file is not None else None
        file_resource_pool = config_file.resource_pool if config_file is not None else None
        file_future_store = config_file.future_store if config_file is not None else None
        file_training = config_file.training if config_file is not None else None
        file_prewarm = config_file.prewarm if config_file is not None else None
        file_docs = config_file.docs if config_file is not None else None
        file_internal = config_file.internal if config_file is not None else None
        dense_session_state_default_root = os.path.join(
            _env_nonempty(environ, "TINKER_RUNTIME_CHECKPOINT_DIR") or DEFAULT_RUNTIME_CHECKPOINTS_DIR,
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

        def _pick_bool(name: str, file_value: bool | None, default: bool) -> bool:
            v = _env_nonempty(environ, name)
            return _parse_bool(v) if v is not None else (bool(file_value) if file_value is not None else bool(default))

        max_model_len_env = _env_nonempty(environ, "TINKER_MAX_MODEL_LEN")
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

        return cls(
            host=_pick_str("TINKER_HOST", file_server.host if file_server is not None else None, "0.0.0.0"),
            port=_pick_int("TINKER_PORT", file_server.port if file_server is not None else None, 8000),
            api_key=api_key,
            token_secret_key=token_secret_key,
            internal_api_token=_pick_str(
                "INTERNAL_API_TOKEN",
                file_server.internal_api_token if file_server is not None else None,
                "",
            ),
            usage_backend=_pick_str(
                "TINKER_USAGE_BACKEND",
                file_server.usage_backend if file_server is not None else None,
                "postgres",
            ).lower(),
            usage_log_dir=_pick_str(
                "TINKER_USAGE_LOG_DIR",
                file_server.usage_log_dir if file_server is not None else None,
                "/tmp/tinker_usage",
            ),
            usage_pg_dsn=(
                _pick_str(
                    "TINKER_USAGE_PG_DSN",
                    file_server.usage_pg_dsn if file_server is not None else None,
                    "",
                )
                or (
                    (
                        "postgresql://"
                        f"{_pick_str('TINKER_USAGE_PG_USER', file_server.usage_pg_user if file_server is not None else None, 'mint_user')}:"
                        f"{_pick_str('TINKER_USAGE_PG_PASSWORD', file_server.usage_pg_password if file_server is not None else None, '')}@"
                        f"{_pick_str('TINKER_USAGE_PG_HOST', file_server.usage_pg_host if file_server is not None else None, '')}:"
                        f"{_pick_int('TINKER_USAGE_PG_PORT', file_server.usage_pg_port if file_server is not None else None, 5432)}/"
                        f"{_pick_str('TINKER_USAGE_PG_DATABASE', file_server.usage_pg_database if file_server is not None else None, 'mint_billing')}"
                    )
                    if _pick_str("TINKER_USAGE_PG_HOST", file_server.usage_pg_host if file_server is not None else None, "")
                    else ""
                )
            ),
            usage_pg_host=_pick_str(
                "TINKER_USAGE_PG_HOST",
                file_server.usage_pg_host if file_server is not None else None,
                "",
            ),
            usage_pg_port=_pick_int(
                "TINKER_USAGE_PG_PORT",
                file_server.usage_pg_port if file_server is not None else None,
                5432,
            ),
            usage_pg_database=_pick_str(
                "TINKER_USAGE_PG_DATABASE",
                file_server.usage_pg_database if file_server is not None else None,
                "mint_billing",
            ),
            usage_pg_user=_pick_str(
                "TINKER_USAGE_PG_USER",
                file_server.usage_pg_user if file_server is not None else None,
                "mint_user",
            ),
            usage_pg_password=_pick_str(
                "TINKER_USAGE_PG_PASSWORD",
                file_server.usage_pg_password if file_server is not None else None,
                "",
            ),
            usage_pg_pool_min=_pick_int(
                "TINKER_USAGE_PG_POOL_MIN",
                file_server.usage_pg_pool_min if file_server is not None else None,
                10,
            ),
            usage_pg_pool_max=_pick_int(
                "TINKER_USAGE_PG_POOL_MAX",
                file_server.usage_pg_pool_max if file_server is not None else None,
                30,
            ),
            usage_write_timeout_ms=_pick_int(
                "TINKER_USAGE_WRITE_TIMEOUT_MS",
                file_server.usage_write_timeout_ms if file_server is not None else None,
                2000,
            ),
            usage_pg_table=_pick_str(
                "TINKER_USAGE_PG_TABLE",
                file_server.usage_pg_table if file_server is not None else None,
                "billing.usage_event",
            ),
            skip_actor_cleanup=_pick_bool(
                "MINT_SKIP_ACTOR_CLEANUP", file_server.skip_actor_cleanup if file_server is not None else None, False
            ),
            tensor_parallel_size=_pick_int("TINKER_TP_SIZE", file_server.tensor_parallel_size if file_server is not None else None, 1),
            data_parallel_size=_pick_int("TINKER_DP_SIZE", file_server.data_parallel_size if file_server is not None else None, 1),
            gpu_memory_utilization=_pick_float(
                "TINKER_GPU_MEM_UTIL", file_server.gpu_memory_utilization if file_server is not None else None, 0.85
            ),
            max_model_len=max_model_len,
            session_inactivity_timeout_s=float(inactivity_s)
            if inactivity_s
            else (float(inactivity_from_file) if inactivity_from_file is not None else None),
            # Multi-LoRA settings
            enable_multi_lora=_pick_bool(
                "TINKER_ENABLE_MULTI_LORA", file_server.enable_multi_lora if file_server is not None else None, True
            ),
            max_loras=_pick_int("TINKER_MAX_LORAS", file_server.max_loras if file_server is not None else None, 64),
            max_cpu_loras=_pick_int(
                "TINKER_MAX_CPU_LORAS", file_server.max_cpu_loras if file_server is not None else None, 1024
            ),
            max_lora_rank=_pick_int("TINKER_MAX_LORA_RANK", file_server.max_lora_rank if file_server is not None else None, 64),
            vllm_attention_backend=_pick_str(
                "TINKER_VLLM_ATTENTION_BACKEND",
                file_server.vllm_attention_backend if file_server is not None else None,
                "DUAL_CHUNK_FLASH_ATTN",
            ),
            # Sampling settings
            sampling_max_inflight_sample_tasks=_pick_int(
                "TINKER_MAX_INFLIGHT_SAMPLE_TASKS",
                file_sampling.max_inflight_sample_tasks if file_sampling is not None else None,
                64,
            ),
            sampling_max_pending_asample_per_apikey=_pick_int(
                "TINKER_MAX_PENDING_ASAMPLE_PER_APIKEY",
                getattr(file_sampling, "max_pending_asample_per_apikey", None) if file_sampling is not None else None,
                64,
            ),
            sampling_max_concurrent_samples_per_request=_pick_int(
                "TINKER_MAX_CONCURRENT_SAMPLES_PER_REQUEST",
                file_sampling.max_concurrent_samples_per_request if file_sampling is not None else None,
                8,
            ),
            sampling_sample_coalesce=_pick_bool(
                "TINKER_SAMPLE_COALESCE",
                file_sampling.sample_coalesce if file_sampling is not None else None,
                True,
            ),
            sampling_sample_coalesce_window_ms=_pick_float(
                "TINKER_SAMPLE_COALESCE_WINDOW_MS",
                file_sampling.sample_coalesce_window_ms if file_sampling is not None else None,
                50.0,
            ),
            sampling_sample_coalesce_max_batch=_pick_int(
                "TINKER_SAMPLE_COALESCE_MAX_BATCH",
                file_sampling.sample_coalesce_max_batch if file_sampling is not None else None,
                32,
            ),
            sampling_sample_coalesce_max_samples=_pick_int(
                "TINKER_SAMPLE_COALESCE_MAX_SAMPLES",
                file_sampling.sample_coalesce_max_samples if file_sampling is not None else None,
                16,
            ),
            sampling_require_seq_id=_pick_bool(
                "TINKER_SAMPLE_REQUIRE_SEQ_ID",
                file_sampling.require_seq_id if file_sampling is not None else None,
                False,
            ),
            # ResourcePool settings
            resource_pool_min_actor_age_s=_pick_int(
                "MINT_MIN_ACTOR_AGE",
                file_resource_pool.min_actor_age_s if file_resource_pool is not None else None,
                300,
            ),
            resource_pool_session_idle_timeout_s=_pick_int(
                "MINT_SESSION_IDLE_TIMEOUT",
                file_resource_pool.session_idle_timeout_s if file_resource_pool is not None else None,
                300,
            ),
            # Future store settings
            future_store_actor_name=_pick_str(
                "MINT_FUTURE_STORE_ACTOR_NAME",
                file_future_store.actor_name if file_future_store is not None else None,
                "tinker_future_store",
            ),
            future_store_ttl_s=_pick_float(
                "MINT_FUTURE_TTL_S",
                file_future_store.ttl_s if file_future_store is not None else None,
                86400.0,
            ),
            future_store_queue_ttl_s=_pick_float(
                "MINT_FUTURE_QUEUE_TTL_S",
                file_future_store.queue_ttl_s if file_future_store is not None else None,
                7 * 86400.0,
            ),
            future_store_done_ttl_s=_pick_float(
                "MINT_FUTURE_DONE_TTL_S",
                file_future_store.done_ttl_s if file_future_store is not None else None,
                7200.0,
            ),
            future_store_tombstone_ttl_s=_pick_float(
                "MINT_FUTURE_TOMBSTONE_TTL_S",
                file_future_store.tombstone_ttl_s if file_future_store is not None else None,
                300.0,
            ),
            future_replay_root_dir=_pick_str(
                "MINT_FUTURE_REPLAY_ROOT_DIR",
                file_future_store.replay_root_dir if file_future_store is not None else None,
                _default_future_replay_root_dir(auth_enabled=auth_enabled),
            ),
            future_replay_hot_ttl_s=_pick_float(
                "MINT_FUTURE_REPLAY_HOT_TTL_S",
                file_future_store.replay_hot_ttl_s if file_future_store is not None else None,
                60.0,
            ),
            future_replay_disk_ttl_s=_pick_float(
                "MINT_FUTURE_REPLAY_DISK_TTL_S",
                file_future_store.replay_disk_ttl_s if file_future_store is not None else None,
                86400.0,
            ),
            future_replay_sweep_interval_s=_pick_float(
                "MINT_FUTURE_REPLAY_SWEEP_INTERVAL_S",
                file_future_store.replay_sweep_interval_s if file_future_store is not None else None,
                21600.0,
            ),
            # Admission control + API work queue (issue #84)
            capacity_manager_actor_name=_pick_str_alias(
                "MINT_CAPACITY_MANAGER_ACTOR_NAME",
                ("TINKER_CAPACITY_MANAGER_ACTOR_NAME",),
                None,
                "tinker_capacity_manager",
            ),
            api_work_queue_actor_name=_pick_str_alias(
                "MINT_API_WORK_QUEUE_ACTOR_NAME",
                ("TINKER_API_WORK_QUEUE_ACTOR_NAME",),
                None,
                "tinker_api_work_queue",
            ),
            capacity_queue_bytes_budget=_pick_int(
                "TINKER_CAPACITY_QUEUE_BYTES_BUDGET",
                None,
                512 * 1024 * 1024,
            ),
            api_work_queue_num_workers=_pick_int(
                "TINKER_API_WORK_QUEUE_NUM_WORKERS",
                None,
                128,
            ),
            api_work_queue_reap_interval_s=_pick_float(
                "TINKER_API_WORK_QUEUE_REAP_INTERVAL_S",
                None,
                5.0,
            ),
            # Training settings
            training_inactivity_timeout_s=_pick_int(
                "MINT_TRAINING_INACTIVITY_TIMEOUT",
                file_training.inactivity_timeout_s if file_training is not None else None,
                3600,
            ),
            training_force_grad_checkpointing=_pick_bool(
                "TINKER_FORCE_GRAD_CHECKPOINTING",
                file_training.force_grad_checkpointing if file_training is not None else None,
                True,
            ),
            training_enable_sdp=_pick_bool(
                "TINKER_ENABLE_SDP",
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
                "TINKER_DENSE_SESSION_STATE_ROOT",
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
            # Persistent prewarm settings
            prewarm_persistent_models_csv=_pick_str(
                "MINT_PERSISTENT_MODELS",
                file_prewarm.persistent_models_csv if file_prewarm is not None else None,
                "",
            ),
            prewarm_train_lora_rank=_pick_int(
                "MINT_PERSISTENT_TRAIN_LORA_RANK",
                file_prewarm.train_lora_rank if file_prewarm is not None else None,
                16,
            ),
            prewarm_train_lr=_pick_float(
                "MINT_PERSISTENT_TRAIN_LR",
                file_prewarm.train_lr if file_prewarm is not None else None,
                5e-5,
            ),
            prewarm_megatron_ready_timeout_s=_pick_float(
                "MINT_PERSISTENT_MEGATRON_READY_TIMEOUT_S",
                file_prewarm.megatron_ready_timeout_s if file_prewarm is not None else None,
                3600.0,
            ),
            prewarm_enable_training=_pick_bool(
                "MINT_PERSISTENT_PREWARM_TRAINING",
                None,
                True,
            ),
            prewarm_enable_inference=_pick_bool(
                "MINT_PERSISTENT_PREWARM_INFERENCE",
                None,
                True,
            ),
            doc_path=_pick_str(
                "MINT_DOC_PATH",
                file_docs.doc_path if file_docs is not None else None,
                "",
            ) or None,
            checkpoint_dir=_pick_str(
                "TINKER_CHECKPOINT_DIR",
                file_internal.checkpoint_dir if file_internal is not None else None,
                "/vePFS-Mindverse/share/code/tinker-server/checkpoints",
            ),
            config_path=config_path,
        )

    @property
    def auth_enabled(self) -> bool:
        """Check if any authentication is configured."""
        return bool(self.api_key or self.token_secret_key)

    def validate_api_key(self, provided_key: str) -> bool:
        """Validate hardcoded API key using constant-time comparison."""
        if not self.api_key:
            return False
        return secrets.compare_digest(self.api_key, provided_key)

# Global config instance
config = ServerConfig.from_sources(environ=os.environ, config_path=_CONFIG_PATH, config_file=_CONFIG_FILE)
