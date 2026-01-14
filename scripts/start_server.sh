#!/bin/bash
cd /root/tinker_project/tinker-server
export PYTHONPATH=/root/tinker_project/tinker-server:$PYTHONPATH
export HF_HUB_OFFLINE=1
export HF_HOME=/vePFS-Mindverse/share/huggingface
export PYTHONDONTWRITEBYTECODE=1
nohup python scripts/run_server.py > /tmp/tinker_server.log 2>&1 &
echo "Server started, PID: $!"
