#!/bin/sh

# Authoritative non-secret runtime configuration for mint-prod-volcano.
# This file is version controlled on purpose.

export MINT_MOE_LORA_SHARED_EXPERT_EXPORT=0
export MINT_MOE_LORA_SPARSE_EXPERT_EXPORT=1

export TINKER_HOST=0.0.0.0
export TINKER_PORT=18000
export TINKER_CHECKPOINT_DIR=/vePFS-Mindverse/share/tinker_checkpoints

export MINT_SUPPORTED_MODELS="Qwen/Qwen3-0.6B,Qwen/Qwen3-4B-Instruct-2507,Qwen/Qwen3-30B-A3B-Instruct-2507,Qwen/Qwen3-235B-A22B-Instruct-2507"
export MINT_PERSISTENT_MODELS="Qwen/Qwen3-0.6B,Qwen/Qwen3-4B-Instruct-2507,Qwen/Qwen3-30B-A3B-Instruct-2507"
export MINT_PERSISTENT_PREWARM_INFERENCE=1
export MINT_PERSISTENT_PREWARM_TRAINING=1
export MINT_PERSISTENT_TRAIN_LORA_RANK=16
export MINT_PERSISTENT_TRAIN_LR=5e-5
export MINT_PERSISTENT_MEGATRON_READY_TIMEOUT_S=3600

export TINKER_GATEWAY_CONFIG_JSON=""
export MINT_SAVE_LORA_TIMEOUT_S=1800
export MINT_SCHEDULER_ENABLE=1

export MINT_VLLM_PINNED_NODE_IP_JSON='{"Qwen/Qwen3-0.6B":"192.168.37.90","Qwen/Qwen3-4B-Instruct-2507":"192.168.37.90","Qwen/Qwen3-30B-A3B-Instruct-2507":"192.168.37.88"}'
export MINT_MODEL_NODE_IPS_JSON='{"Qwen/Qwen3-0.6B":["192.168.37.90"],"Qwen/Qwen3-4B-Instruct-2507":["192.168.37.90"],"Qwen/Qwen3-30B-A3B-Instruct-2507":["192.168.37.88"],"Qwen/Qwen3-235B-A22B-Instruct-2507":["192.168.37.92","192.168.37.93","192.168.37.94","192.168.37.95"],"Qwen/Qwen3-235B-A22B-Thinking-2507":["192.168.37.92","192.168.37.93","192.168.37.94","192.168.37.95"]}'

export MINT_MODEL_CONFIG_OVERRIDES_JSON='{"Qwen/Qwen3-0.6B":{"vllm_engine":"async","vllm_distributed_executor_backend":"mp"},"Qwen/Qwen3-4B-Instruct-2507":{"vllm_engine":"async","vllm_distributed_executor_backend":"mp"}}'

export TINKER_ENABLE_MULTI_LORA=1
export MINT_VLLM_ENABLE_CHUNKED_PREFILL=1
export MINT_VLLM_ENABLE_PREFIX_CACHING=1
export MINT_VLLM_FULLY_SHARDED_LORAS=1
export MINT_VLLM_ADMISSION_CONTROL=1

export MINT_LOG_FILE=/tmp/tinker_server_auth.log
export MINT_LOG_MAX_BYTES=10485760
export MINT_LOG_BACKUP_COUNT=5
export OTEL_SERVICE_NAME=mint
export OTEL_EXPORTER_OTLP_ENDPOINT=http://apmplus-cn-beijing.ivolces.com:4317
export OTEL_EXPORTER_OTLP_HEADERS=
export OTEL_METRIC_EXPORT_INTERVAL_MS=10000
export OTEL_LOG_LEVEL=DEBUG
export MINT_HEALTHZ_RAY_TIMEOUT_S=30.0
export MINT_VLLM_REQUEST_TIMING=1
export MINT_TIMING_DIAG=1
export MINT_VERL_DIAGNOSTICS=1
export MINT_LOG_KILL_STACK=1

export HF_HOME=/vePFS-Mindverse/share/huggingface
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1

export MINT_MEGATRON_VOLC_RESOURCE_QUEUE_ID=q-20251126180002-26lwz
export MINT_VLLM_VOLC_RESOURCE_QUEUE_ID=q-20251126180002-26lwz
if [ -f /vePFS-Mindverse/share/code/tinker-server-auth/ray_head_ip.txt ]; then
  export RAY_ADDRESS="$(cat /vePFS-Mindverse/share/code/tinker-server-auth/ray_head_ip.txt):6379"
else
  echo "Missing /vePFS-Mindverse/share/code/tinker-server-auth/ray_head_ip.txt" >&2
  return 1 2>/dev/null || exit 1
fi

export TINKER_API_WORK_QUEUE_ACTOR_NAME=tinker_api_work_queue_v20260309
export MINT_API_WORK_QUEUE_ACTOR_MAX_CONCURRENCY=1024

export PFS_RUNTIME_ENV_ROOT=/vePFS-Mindverse/share/code/tinker-server-auth/tinker-runtime-py31213
export PFS_TINKER_PATH=/vePFS-Mindverse/share/code/tinker-server-auth
export PFS_HF_MODULES_PATH=/vePFS-Mindverse/share/huggingface/modules
export TINKER_USAGE_LOG_DIR=/vePFS-Mindverse/share/mint-prod-data/billing

export LD_LIBRARY_PATH=/vePFS-Mindverse/share/code/tinker-server-auth/tinker-runtime-py31213/host-venv/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
