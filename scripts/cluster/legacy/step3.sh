ssh driver 'MINT_CODE_ROOT=/vePFS-Mindverse/share/code/wenxi/dev_vla_wenxi \
  MINT_DEV_USER=wenxi \
  MINT_RAY_NAMESPACE=mint_wenxi_dev \
  MINT_TASK_STATE_STORE_DB_PATH=/vePFS-Mindverse/share/mint/dev/data/wenxi/task-state/task_state.sqlite3 \
  MINT_LOG_FILE=/vePFS-Mindverse/share/mint/dev/logs/mint-dev-server.log \
  MINT_DISABLE_MINT_ROUTE=0 \
  MINT_UVICORN_WORKERS=1 \
  MINT_SUPERVISOR_STATE_BACKEND=memory \
  MINT_SUPPORTED_MODELS="Qwen/Qwen3.6-27B,Qwen/Qwen3-0.6B,Qwen/Qwen3-4B-Instruct-2507,Qwen/Qwen3-30B-A3B-Instruct-2507" \
  MINT_DEV_RUN_ENV=/tmp/mint_dev_run.env \
  PFS_RUNTIME_ENV_ROOT=/vePFS-Mindverse/share/mint/dev/runtime \
  nohup /vePFS-Mindverse/share/code/wenxi/dev_vla_wenxi/scripts/start_dev_server.sh \
  >> /tmp/mint_dev_launch_wenxi.log 2>&1 &'