from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError


try:
    import tomllib  # py311+
except Exception:  # pragma: no cover
    import tomli as tomllib


class _ServerSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str | None = None
    port: int | None = None

    internal_api_token: str | None = None
    usage_log_dir: str | None = None  # active JSONL sink
    usage_backend: str | None = None  # deprecated compatibility field; if set, it must remain 'postgres'
    usage_pg_dsn: str | None = None  # deprecated, ignored by the producer path
    usage_pg_host: str | None = None
    usage_pg_port: int | None = None
    usage_pg_database: str | None = None
    usage_pg_user: str | None = None
    usage_pg_password: str | None = None
    usage_pg_pool_min: int | None = None
    usage_pg_pool_max: int | None = None
    usage_write_timeout_ms: int | None = None
    usage_pg_table: str | None = None
    skip_actor_cleanup: bool | None = None

    tensor_parallel_size: int | None = None
    data_parallel_size: int | None = None
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    session_inactivity_timeout_s: float | None = None

    enable_multi_lora: bool | None = None
    max_loras: int | None = None
    max_cpu_loras: int | None = None
    max_lora_rank: int | None = None
    vllm_attention_backend: str | None = None


class _SamplingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_inflight_sample_tasks: int | None = None
    max_concurrent_samples_per_request: int | None = None

    sample_coalesce: bool | None = None
    sample_coalesce_window_ms: float | None = None
    sample_coalesce_max_batch: int | None = None
    sample_coalesce_max_samples: int | None = None
    require_seq_id: bool | None = None


class _RaySection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace: str | None = None


class _PathsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pfs_runtime_env_root: str | None = None
    pfs_tinker_path: str | None = None
    pfs_hf_modules_path: str | None = None


class _MegatronBridgeSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_mbridge_lora_export: bool | None = None


class _ResourcePoolSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_actor_age_s: int | None = None
    session_idle_timeout_s: int | None = None


class _FutureStoreSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor_name: str | None = None
    ttl_s: float | None = None
    queue_ttl_s: float | None = None
    done_ttl_s: float | None = None
    tombstone_ttl_s: float | None = None
    replay_root_dir: str | None = None
    replay_hot_ttl_s: float | None = None
    replay_disk_ttl_s: float | None = None
    replay_sweep_interval_s: float | None = None


class _TrainingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inactivity_timeout_s: int | None = None
    force_grad_checkpointing: bool | None = None
    enable_sdp: bool | None = None

    megatron_create_timeout_s: float | None = None
    dense_get_or_create_timeout_s: float | None = None
    dense_session_state_root: str | None = None
    reinit_lora_timeout_s: float | None = None
    actor_ready_timeout_s: float | None = None
    remote_call_timeout_s: float | None = None


class _PrewarmSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    persistent_models_csv: str | None = None
    train_lora_rank: int | None = None
    train_lr: float | None = None
    megatron_ready_timeout_s: float | None = None


class _DocsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_path: str | None = None


class _InternalSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint_dir: str | None = None


class TinkerConfigFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server: _ServerSection = Field(default_factory=_ServerSection)
    sampling: _SamplingSection = Field(default_factory=_SamplingSection)
    ray: _RaySection = Field(default_factory=_RaySection)
    paths: _PathsSection = Field(default_factory=_PathsSection)
    megatron_bridge: _MegatronBridgeSection = Field(default_factory=_MegatronBridgeSection)
    resource_pool: _ResourcePoolSection = Field(default_factory=_ResourcePoolSection)
    future_store: _FutureStoreSection = Field(default_factory=_FutureStoreSection)
    training: _TrainingSection = Field(default_factory=_TrainingSection)
    prewarm: _PrewarmSection = Field(default_factory=_PrewarmSection)
    docs: _DocsSection = Field(default_factory=_DocsSection)
    internal: _InternalSection = Field(default_factory=_InternalSection)


def load_tinker_config_file(path: str | Path) -> TinkerConfigFile:
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    try:
        data = tomllib.loads(raw)
    except Exception as e:
        raise ValueError(f"TOML parse failed: path={str(p)!r} err={e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"TOML root must be a table: path={str(p)!r} type={type(data)}")

    try:
        return TinkerConfigFile.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Config validation failed: path={str(p)!r} err={e}") from e
