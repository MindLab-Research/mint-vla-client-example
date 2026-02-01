# tinker-server config file (TOML)

`tinker-server` supports an optional TOML config file.

## Location and precedence

- `scripts/run_server.py --config /path/to/config.toml` sets `TINKER_CONFIG_PATH` before importing the app.
- `TINKER_CONFIG_PATH=/path/to/config.toml` loads the file at process startup.

Precedence:
1. Environment variables
2. Config file (`TINKER_CONFIG_PATH`)
3. Built-in defaults

Unknown keys and type mismatches fail fast at startup with a validation error.

## Supported keys

Secrets stay in environment variables (`TINKER_API_KEY`, `TINKER_TOKEN_SECRET_KEY`); the config file schema forbids them.

### `[server]`

- `host` (str)
- `port` (int)
- `usage_log_dir` (str)
- `skip_actor_cleanup` (bool) [env: `MINT_SKIP_ACTOR_CLEANUP`]
- `tensor_parallel_size` (int) [env: `TINKER_TP_SIZE`]
- `data_parallel_size` (int) [env: `TINKER_DP_SIZE`]
- `gpu_memory_utilization` (float) [env: `TINKER_GPU_MEM_UTIL`]
- `max_model_len` (int) [env: `TINKER_MAX_MODEL_LEN`]
- `session_inactivity_timeout_s` (float) [env: `TINKER_SESSION_INACTIVITY_TIMEOUT_S`]
- `enable_multi_lora` (bool) [env: `TINKER_ENABLE_MULTI_LORA`]
- `max_loras` (int) [env: `TINKER_MAX_LORAS`]
- `max_cpu_loras` (int) [env: `TINKER_MAX_CPU_LORAS`]
- `max_lora_rank` (int) [env: `TINKER_MAX_LORA_RANK`]

### `[sampling]`

- `max_inflight_sample_tasks` (int) [env: `TINKER_MAX_INFLIGHT_SAMPLE_TASKS`]
- `max_concurrent_samples_per_request` (int) [env: `TINKER_MAX_CONCURRENT_SAMPLES_PER_REQUEST`]
- `sample_coalesce` (bool) [env: `TINKER_SAMPLE_COALESCE`]
- `sample_coalesce_window_ms` (float) [env: `TINKER_SAMPLE_COALESCE_WINDOW_MS`]
- `sample_coalesce_max_batch` (int) [env: `TINKER_SAMPLE_COALESCE_MAX_BATCH`]
- `sample_coalesce_max_samples` (int) [env: `TINKER_SAMPLE_COALESCE_MAX_SAMPLES`]

### `[ray]`

- `namespace` (str) [env: `TINKER_RAY_NAMESPACE` / `MINT_RAY_NAMESPACE`]

### `[paths]`

- `pfs_tinker_path` (str) [env: `PFS_TINKER_PATH`]
- `pfs_verl_path` (str) [env: `PFS_VERL_PATH`]
- `pfs_vllm_path` (str) [env: `PFS_VLLM_PATH`]
- `pfs_megatron_bridge_path` (str) [env: `PFS_MEGATRON_BRIDGE_PATH`]
- `pfs_megatron_bridge_hollowman_path` (str) [env: `PFS_MEGATRON_BRIDGE_HOLLOWMAN_PATH`]
- `pfs_hf_modules_path` (str) [env: `PFS_HF_MODULES_PATH`]
- `pfs_extra_pythonpath` (str) [env: `PFS_EXTRA_PYTHONPATH`]

### `[megatron_bridge]`

- `use_hollowman_mbridge` (bool) [env: `USE_HOLLOWMAN_MBRIDGE`]
- `use_mbridge_lora_export` (bool) [env: `USE_MBRIDGE_LORA_EXPORT`]

### `[resource_pool]`

- `min_actor_age_s` (int) [env: `MINT_MIN_ACTOR_AGE`]
- `session_idle_timeout_s` (int) [env: `MINT_SESSION_IDLE_TIMEOUT`]

### `[future_store]`

- `actor_name` (str) [env: `MINT_FUTURE_STORE_ACTOR_NAME`]
- `ttl_s` (float) [env: `MINT_FUTURE_TTL_S`]
- `done_ttl_s` (float) [env: `MINT_FUTURE_DONE_TTL_S`]

### `[training]`

- `force_grad_checkpointing` (bool) [env: `TINKER_FORCE_GRAD_CHECKPOINTING`]
- `enable_sdp` (bool) [env: `TINKER_ENABLE_SDP`]
- `megatron_create_timeout_s` (float) [env: `MINT_MEGATRON_CREATE_TIMEOUT_S`]
- `dense_get_or_create_timeout_s` (float) [env: `MINT_DENSE_GET_OR_CREATE_TIMEOUT_S`]
- `reinit_lora_timeout_s` (float) [env: `MINT_REINIT_LORA_TIMEOUT_S`]
- `actor_ready_timeout_s` (float) [env: `MINT_ACTOR_READY_TIMEOUT_S`]

### `[prewarm]`

- `persistent_models_csv` (str) [env: `MINT_PERSISTENT_MODELS`]
- `train_lora_rank` (int) [env: `MINT_PERSISTENT_TRAIN_LORA_RANK`]
- `train_lr` (float) [env: `MINT_PERSISTENT_TRAIN_LR`]
- `megatron_ready_timeout_s` (float) [env: `MINT_PERSISTENT_MEGATRON_READY_TIMEOUT_S`]

### `[docs]`

- `doc_path` (str) [env: `MINT_DOC_PATH`]

### `[internal]`

- `checkpoint_dir` (str) [env: `TINKER_CHECKPOINT_DIR`]

## Example

```toml
[server]
port = 8000
max_loras = 64

[sampling]
max_inflight_sample_tasks = 64
```
