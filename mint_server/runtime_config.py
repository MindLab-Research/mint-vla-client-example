"""Runtime configuration snapshot and classification helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Mapping

from .config import ServerConfig, config as server_config
from .runtime_env import canonical_mint_env_name

CONFIG_SNAPSHOT_SCHEMA_VERSION = 1
CONFIG_ACTOR_DEFAULT_NAME = "mint_config"

CONFIG_CLASS_BOOTSTRAP_RUNTIME_ENV = "bootstrap_runtime_env"
CONFIG_CLASS_ACTOR_CREATION_INPUT = "actor_creation_input"
CONFIG_CLASS_SNAPSHOT_CONFIG = "snapshot_config"
CONFIG_CLASS_OBSERVABILITY = "observability"
CONFIG_CLASS_TASK_STATE = "task_state"
CONFIG_CLASS_UNCLASSIFIED = "unclassified"

# Values required before an actor can import code or connect back to Ray. These
# are expected to remain explicit runtime_env inputs even after ConfigActor
# migration.
BOOTSTRAP_RUNTIME_ENV_KEYS = frozenset(
    {
        "PYTHONPATH",
        "PFS_RUNTIME_ENV_ROOT",
        "MINT_CODE_ROOT",
        "PFS_HF_MODULES_PATH",
        "RAY_ADDRESS",
        "MINT_RAY_NAMESPACE",
        "MINT_CONFIG_PATH",
        "MINT_ACTOR_LD_LIBRARY_PATH",
        "MINT_RAY_CLIENT_ADDRESS",
        "RAY_CLIENT_ADDRESS",
        "MINT_RAY_HEAD_ADDRESS_PATH",
        "MINT_RAY_JOB_WORKING_DIR",
        "MINT_RAY_WORKING_DIR",
        "MINT_RAY_PY_MODULES_CSV",
        "MINT_RAY_NODE_IP_ADDRESS",
        "MINT_RAY_TEMP_DIR",
    }
)

# Values Ray must see at actor creation time. A snapshot can record them for
# auditability, but an actor cannot consume them after it has already started.
ACTOR_CREATION_INPUT_KEYS = frozenset(
    {
        "MINT_CONFIG_ACTOR_NAME",
        "MINT_MEGATRON_NODE_IPS_CSV",
        "MINT_MODEL_WORK_SCHEDULER_ACTOR_NAME",
        "MINT_MAINTENANCE_CRON_ACTOR_NAME",
    }
)

# Low-frequency deployment/runtime settings that are good candidates for the
# ConfigActor snapshot once consumers are migrated away from env reads.
SNAPSHOT_CONFIG_ENV_KEYS = frozenset(
    {
        "MINT_VLLM_CHILD_PYTHON_EXECUTABLE",
        "MINT_VLLM_SERIALIZE_ADD_LORA_UNTIL_IDLE",
        "MINT_VLLM_REQUEST_TIMING",
        "MINT_VLLM_SKIP_PEFT_SHAPE_VALIDATION",
        "MINT_VLLM_ENABLE_SLEEP_MODE",
        "MINT_MODEL_CONFIG_OVERRIDES_JSON",
        "MINT_VLLM_MAX_NUM_SEQS",
        "MINT_VLLM_MAX_NUM_BATCHED_TOKENS",
        "MINT_VLLM_MAX_LORAS",
        "MINT_VLLM_MAX_CPU_LORAS",
        "MINT_VLLM_MAX_LORA_RANK",
        "MINT_SFT_DIAG_FAIL",
        "MINT_REVERSE_KL_DIAG_FAIL",
        "MINT_DENSE_SESSION_STATE_ROOT",
        "MINT_RUNTIME_CHECKPOINT_DIR",
        "MINT_USAGE_BACKEND",
        "MINT_USAGE_PG_DSN",
        "MINT_CHECKPOINT_INDEX_PG_DSN",
        "MINT_CHECKPOINT_INDEX_WRITE_TIMEOUT_MS",
        "MINT_CHECKPOINT_INDEX_UPLOADING_STALE_S",
        "MINT_CHECKPOINT_INDEX_PUBLISH_RETRY_S",
        "MINT_MODEL_WORK_SCHEDULER_DEBUG_LOG_PATH",
        "MINT_SCHEDULER_ENABLE",
        "MINT_SCHEDULER_MAX_CONSECUTIVE",
        "MINT_SCHEDULER_FAIRNESS",
        "MINT_SCHEDULER_STARVATION_S",
        "MINT_SCHEDULER_COALESCE_MS",
        "MINT_MEGATRON_STICKY_TRAIN_MODE",
        "MINT_MEGATRON_STICKY_IDLE_TIMEOUT_S",
        "MINT_MEGATRON_STICKY_CLOSE_ON_OPTIM",
        "MINT_MEGATRON_STICKY_TIMING_DIAG",
        "MINT_MEGATRON_STACK_DUMP_TIMEOUT_S",
        "MINT_MEGATRON_STACK_DUMP_LIMIT",
        "MINT_MBRIDGE_EXPORT_GLOO_TIMEOUT_S",
        "MINT_MBRIDGE_EXPORT_GATHER_DEBUG",
        "MINT_MBRIDGE_EXPORT_GLOO_BARRIER_DEBUG",
        "MINT_MEGATRON_ENABLE_DEEPEP",
        "MINT_MEGATRON_MOE_TOKEN_DISPATCHER_TYPE",
        "MINT_MEGATRON_MOE_FLEX_DISPATCHER_BACKEND",
        "MINT_MEGATRON_MOE_ROUTER_DTYPE",
        "MINT_MEGATRON_MOE_DEEPEP_NUM_SMS",
        "MINT_NCCL_IB_DISABLE",
        "MINT_TIMING_DIAG",
        "MINT_SUPPORTED_MODELS",
        "MINT_GATEWAY_CONFIG_JSON",
        "MINT_MAINTENANCE_REAP_INTERVAL_S",
        "MINT_ACTOR_RECONCILE_INTERVAL_S",
        "MINT_TOPOLOGY_CONFIG_PATH",
        "MINT_TOPOLOGY_STATE_PATH",
        "MINT_MODEL_RUNTIME_POLL_INTERVAL_S",
        "MINT_MODEL_RUNTIME_LEASE_TTL_S",
        "MINT_TRAINING_HEARTBEAT_STALE_S",
        "MINT_SUPERVISOR_STATE_BACKEND",
        "MINT_SUPERVISOR_STATE_DB_PATH",
        "MINT_SUPERVISOR_STATE_OWNER_TTL_S",
        "MINT_SUPERVISOR_STATE_EVENT_LIMIT",
        "MINT_SESSION_HEARTBEAT_MAX_AGE_S",
        "MINT_SESSION_HEARTBEAT_PRUNE_EVERY",
        "MINT_RETRIEVE_FUTURE_HOT_TTL_S",
        "MINT_RETRIEVE_FUTURE_GRACE_S",
        "MINT_RETRIEVE_FUTURE_MIN_POLL_S",
        "MINT_RETRIEVE_FUTURE_WAIT_TIMEOUT_S",
        "MINT_TASK_PENDING_TTL_S",
        "MINT_TASK_RESULT_TTL_S",
        "MINT_TASK_TOMBSTONE_TTL_S",
        "MINT_VLLM_ACTOR_MAX_CONCURRENCY",
        "MINT_VLLM_OMP_NUM_THREADS",
        "MINT_VLLM_MKL_NUM_THREADS",
        "MINT_VLLM_OPENBLAS_NUM_THREADS",
        "MINT_VLLM_NUMEXPR_NUM_THREADS",
        "MINT_VLLM_VECLIB_MAXIMUM_THREADS",
        "MINT_VLLM_BLIS_NUM_THREADS",
        "MINT_VLLM_ENGINE_LOCK_MODE",
        "MINT_VLLM_MULTISAMPLE_MODE",
        "MINT_VLLM_GENERATE_TIMEOUT_S",
        "MINT_VLLM_POST_GENERATE_DELAY_S",
        "MINT_VLLM_IS_READY_TIMEOUT_S",
        "MINT_VLLM_DISTRIBUTED_EXECUTOR_BACKEND",
        "MINT_VLLM_LORA_DTYPE",
        "MINT_VLLM_ALL2ALL_BACKEND",
        "MINT_VLLM_SLOW_REQUEST_LOG_THRESHOLD_S",
        "MINT_VLLM_RAY_GET_TIMEOUT_S",
        "MINT_VLLM_ALLOW_INSECURE_SERIALIZATION",
        "MINT_VLLM_LORA_DEBUG",
        "MINT_VLLM_ENGINE_READY_WAIT_S",
        "MINT_DENSE_INFLIGHT_WAIT_S",
        "MINT_DENSE_ACTOR_INIT_TIMEOUT_S",
        "MINT_OPENPI_XLA_FLAGS",
        "MINT_OPENPI_RAY_ACTOR_READY_TIMEOUT_S",
        "MINT_OPENPI_FAST_TOKENIZER_PATH",
        "MINT_OPENPI_FAST_ACTION_SESSION_STATE_ROOT",
        "MINT_OPENPI_FAST_DEBUG_TOKENS",
        "MINT_OPENPI_FAST_ACTION_REQUEST_TIMEOUT_S",
        "MINT_OPENPI_FAST_ACTION_PYTHON",
        "MINT_OPENPI_FAST_ACTION_STARTUP_TIMEOUT_S",
        "MINT_OPENPI_FAST_ACTION_CWD",
        "MINT_OPENPI_FAST_WEIGHTS_PATH",
        "MINT_OPENPI_FAST_RANDOM_INIT",
        "MINT_OPENPI_PI05_WEIGHTS_PATH",
        "MINT_OPENPI_PI05_RANDOM_INIT",
        "MINT_OPENPI_FAST_PYTHONPATH",
        "MINT_OPENPI_FAST_REQUEST_TIMEOUT_S",
        "MINT_OPENPI_FAST_PYTHON",
        "MINT_OPENPI_FAST_STARTUP_TIMEOUT_S",
        "MINT_OPENPI_FAST_CREATE_SESSION_TIMEOUT_S",
        "MINT_OPENPI_FAST_SAVE_TIMEOUT_S",
        "MINT_OPENPI_FAST_LOAD_TIMEOUT_S",
        "MINT_OPENPI_FAST_CWD",
        "MINT_PERSISTENT_INFER_TIMEOUT_S",
        "MINT_MODEL_RUNTIME_EXECUTION_TIMEOUT_S",
        "MINT_MODEL_RUNTIME_EXECUTION_TIMEOUT_GRACE_S",
        "MINT_MODEL_RUNTIME_SAVE_WEIGHTS_TIMEOUT_S",
        "MINT_REVERSE_KL_VOCAB_BLOCK",
        "MINT_MEGATRON_GUARD_PREFLIGHT",
        "MINT_MEGATRON_GUARD_QUERY_TIMEOUT_S",
        "MINT_WORKER_CUDA_SUMMARY_TIMEOUT_S",
        "MINT_MEGATRON_STRICT_SAVE_META",
        "MINT_MEGATRON_SKIP_CREATE_READY_WAIT",
        "MINT_SAVE_LORA_TIMEOUT_S",
        "MINT_SAVE_CHECKPOINT_TIMEOUT_S",
        "MINT_LOAD_CHECKPOINT_TIMEOUT_S",
        "MINT_DISABLE_EXTERNAL_LABEL",
        "MINT_PPO_LOSS_DEBUG",
        "MINT_VERL_DIAGNOSTICS",
        "MINT_LORA_EVICT_MIN_IDLE_S",
        "MINT_MEGATRON_SESSIONS_BASE_PATH",
        "MINT_TORCH_DIST_TIMEOUT_S",
        "MINT_MEGATRON_ENFORCE_TRUSTED_PAIR",
        "MINT_MEGATRON_SAVE_CHECKPOINT_TIMEOUT_S",
        "MINT_MEGATRON_SAVE_LORA_TIMEOUT_S",
        "MINT_TP_SIZE",
        "MINT_DP_SIZE",
        "MINT_GPU_MEM_UTIL",
        "MINT_MAX_MODEL_LEN",
        "MINT_SESSION_INACTIVITY_TIMEOUT_S",
        "MINT_INACTIVITY_TIMEOUT_S",
        "MINT_ENABLE_MULTI_LORA",
        "MINT_MAX_LORAS",
        "MINT_MAX_CPU_LORAS",
        "MINT_MAX_LORA_RANK",
        "MINT_VLLM_ATTENTION_BACKEND",
        "MINT_MAX_INFLIGHT_SAMPLE_TASKS",
        "MINT_MAX_PENDING_ASAMPLE_PER_APIKEY",
        "MINT_MAX_CONCURRENT_SAMPLES_PER_REQUEST",
        "MINT_SAMPLE_COALESCE",
        "MINT_SAMPLE_COALESCE_WINDOW_MS",
        "MINT_SAMPLE_COALESCE_MAX_BATCH",
        "MINT_SAMPLE_COALESCE_MAX_SAMPLES",
        "MINT_SAMPLE_REQUIRE_SEQ_ID",
        "MINT_MODEL_ACTOR_INVENTORY_SESSION_IDLE_TIMEOUT_S",
        "MINT_TRAINING_INACTIVITY_TIMEOUT",
        "MINT_FORCE_GRAD_CHECKPOINTING",
        "MINT_ENABLE_SDP",
        "MINT_MEGATRON_CREATE_TIMEOUT_S",
        "MINT_DENSE_GET_OR_CREATE_TIMEOUT_S",
        "MINT_REINIT_LORA_TIMEOUT_S",
        "MINT_ACTOR_READY_TIMEOUT_S",
        "MINT_TRAINING_REMOTE_CALL_TIMEOUT_S",
        "MINT_ROUTER_REPLAY_MODE",
        "MINT_DOC_PATH",
        "MINT_CHECKPOINT_DIR",
    }
)

OBSERVABILITY_ENV_KEYS = frozenset(
    {
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_EXPORTER_OTLP_INSECURE",
        "OTEL_SERVICE_NAME",
        "OTEL_RESOURCE_ATTRIBUTES",
        "OTEL_LOG_LEVEL",
        "MINT_APMPLUS_APP_KEY",
        "OTEL_APMPLUS_APP_KEY",
        "MINT_DEPLOYMENT_ENV",
        "MINT_CLUSTER_ID",
    }
)

TASK_STATE_ENV_KEYS = frozenset(
    {
        "MINT_TASK_STATE_STORE_ACTOR_NAME",
        "MINT_TASK_STATE_STORE_DB_PATH",
        "MINT_TASK_STATE_STORE_OWNER_TTL_S",
        "MINT_TASK_STATE_STORE_OWNER_RENEW_S",
        "MINT_TASK_PAYLOAD_ROOT_DIR",
    }
)

KNOWN_CONFIG_CLASSES = (
    CONFIG_CLASS_BOOTSTRAP_RUNTIME_ENV,
    CONFIG_CLASS_ACTOR_CREATION_INPUT,
    CONFIG_CLASS_SNAPSHOT_CONFIG,
    CONFIG_CLASS_OBSERVABILITY,
    CONFIG_CLASS_TASK_STATE,
    CONFIG_CLASS_UNCLASSIFIED,
)

SECRET_MARKERS = (
    "API_KEY",
    "SECRET",
    "PASSWORD",
    "PG_DSN",
    "HEADERS",
)
SECRET_PARTS = frozenset({"TOKEN"})
REDACTED_VALUE = "<redacted>"
CONFIG_ACTOR_HYDRATION_CONTROL_KEYS = frozenset(
    {
        "MINT_CONFIG_ACTOR_HYDRATE",
        "MINT_CONFIG_ACTOR_SELF",
    }
)
CONFIG_ACTOR_ENV_EXCLUDED_KEYS = frozenset(
    {
        "MINT_API_KEY",
        "MINT_BASE_URL",
        "MINT_PROD_CONFIG_ENV",
        "MINT_PROD_SECRETS_ENV",
        "MINT_DEV_CONFIG_ENV",
        "MINT_DEV_SECRETS_ENV",
        "MINT_MODEL_PLACEMENT_JSON",
        "MINT_VLLM_MODEL_PLACEMENT_JSON",
        "MINT_DENSE_MODEL_PLACEMENT_JSON",
        "MINT_MEGATRON_MODEL_PLACEMENT_JSON",
        "MINT_MODEL_ACTOR_REPLICA",
        "MINT_MODEL_ACTOR_REPLICA_ID",
    }
)
CONFIG_ACTOR_HYDRATION_PREFIXES = ("OTEL_",)


def config_actor_name(environ: Mapping[str, str] | None = None) -> str:
    environ = os.environ if environ is None else environ
    value = str(environ.get("MINT_CONFIG_ACTOR_NAME") or "").strip()
    return value or CONFIG_ACTOR_DEFAULT_NAME


def classify_env_key(key: str) -> str:
    key = canonical_mint_env_name(key)
    if key in BOOTSTRAP_RUNTIME_ENV_KEYS:
        return CONFIG_CLASS_BOOTSTRAP_RUNTIME_ENV
    if key in ACTOR_CREATION_INPUT_KEYS:
        return CONFIG_CLASS_ACTOR_CREATION_INPUT
    if key in OBSERVABILITY_ENV_KEYS:
        return CONFIG_CLASS_OBSERVABILITY
    if key in TASK_STATE_ENV_KEYS:
        return CONFIG_CLASS_TASK_STATE
    if key in SNAPSHOT_CONFIG_ENV_KEYS:
        return CONFIG_CLASS_SNAPSHOT_CONFIG
    return CONFIG_CLASS_UNCLASSIFIED


def classify_env(environ: Mapping[str, str]) -> dict[str, dict[str, str]]:
    grouped: dict[str, dict[str, str]] = {name: {} for name in KNOWN_CONFIG_CLASSES}
    for key, value in sorted(environ.items()):
        if not value:
            continue
        canonical_key = canonical_mint_env_name(key)
        if canonical_key != key and canonical_key in environ:
            continue
        if canonical_key in CONFIG_ACTOR_ENV_EXCLUDED_KEYS:
            continue
        config_class = classify_env_key(canonical_key)
        if config_class == CONFIG_CLASS_UNCLASSIFIED:
            continue
        grouped[config_class][canonical_key] = _redact_config_value(canonical_key, str(value))
    return grouped


def _redact_config_value(key: str, value: object) -> object:
    upper = key.upper()
    parts = {part for part in re.split(r"[^A-Z0-9]+", upper) if part}
    if any(marker in upper for marker in SECRET_MARKERS) or parts.intersection(SECRET_PARTS):
        return REDACTED_VALUE
    return value


def _redact_server_config(raw: dict[str, object]) -> dict[str, object]:
    return {key: _redact_config_value(key, value) for key, value in raw.items()}


def actor_env_from_environ(environ: Mapping[str, str]) -> dict[str, str]:
    """Return non-bootstrap config that actors hydrate from ConfigActor."""
    out: dict[str, str] = {}
    for key, value in sorted(environ.items()):
        if not value:
            continue
        if key in CONFIG_ACTOR_HYDRATION_CONTROL_KEYS:
            continue
        canonical_key = canonical_mint_env_name(key)
        if canonical_key != key and canonical_key in environ:
            continue
        if canonical_key in CONFIG_ACTOR_ENV_EXCLUDED_KEYS:
            continue
        config_class = classify_env_key(canonical_key)
        if config_class in {
            CONFIG_CLASS_ACTOR_CREATION_INPUT,
            CONFIG_CLASS_SNAPSHOT_CONFIG,
            CONFIG_CLASS_OBSERVABILITY,
            CONFIG_CLASS_TASK_STATE,
        } or (config_class == CONFIG_CLASS_UNCLASSIFIED and canonical_key.startswith(CONFIG_ACTOR_HYDRATION_PREFIXES)):
            out[canonical_key] = str(value)
    return out


@dataclass(frozen=True)
class ConfigSnapshot:
    schema_version: int
    created_at: float
    ray_namespace: str
    actor_name: str
    config_path: str | None
    env: dict[str, dict[str, str]]
    actor_env: dict[str, str]
    server_config: dict[str, object]
    fingerprint: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _fingerprint_payload(snapshot: dict[str, object]) -> dict[str, object]:
    payload = dict(snapshot)
    payload.pop("created_at", None)
    payload.pop("fingerprint", None)
    return payload


def snapshot_fingerprint(snapshot: dict[str, object]) -> str:
    payload = _fingerprint_payload(snapshot)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_config_snapshot(
    *,
    environ: Mapping[str, str] | None = None,
    ray_namespace: str,
    actor_name: str | None = None,
    config: ServerConfig | None = None,
    created_at: float | None = None,
) -> ConfigSnapshot:
    environ = os.environ if environ is None else environ
    actor = actor_name or config_actor_name(environ)
    cfg = config or server_config
    snapshot_without_fingerprint: dict[str, object] = {
        "schema_version": CONFIG_SNAPSHOT_SCHEMA_VERSION,
        "created_at": float(time.time() if created_at is None else created_at),
        "ray_namespace": str(ray_namespace),
        "actor_name": str(actor),
        "config_path": cfg.config_path,
        "env": classify_env(environ),
        "actor_env": actor_env_from_environ(environ),
        "server_config": _redact_server_config(asdict(cfg)),
        "fingerprint": "",
    }
    fingerprint = snapshot_fingerprint(snapshot_without_fingerprint)
    return ConfigSnapshot(
        schema_version=CONFIG_SNAPSHOT_SCHEMA_VERSION,
        created_at=float(snapshot_without_fingerprint["created_at"]),
        ray_namespace=str(ray_namespace),
        actor_name=str(actor),
        config_path=cfg.config_path,
        env=snapshot_without_fingerprint["env"],  # type: ignore[arg-type]
        actor_env=snapshot_without_fingerprint["actor_env"],  # type: ignore[arg-type]
        server_config=snapshot_without_fingerprint["server_config"],  # type: ignore[arg-type]
        fingerprint=fingerprint,
    )
