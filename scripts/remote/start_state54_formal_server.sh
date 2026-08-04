#!/usr/bin/env bash
set -euo pipefail
SEED=${1:?seed42/43/44 required}
GPUS=${2:-4,5,6,7}
PORT=${3:-30540}
case "$SEED" in 42|43|44) ;; *) echo "seed must be42/43/44" >&2;exit 64;; esac
[[ "$GPUS" == "4,5,6,7" ]] || { echo "formal State54 server is pinned to physical GPUs4-7" >&2;exit 64;}
ROOT=/vePFS-Mindverse/user/intern/rongenz/pi05-finetune
SHARE=/vePFS-Mindverse/share/intern/rongenz/pi05-finetune
CLIENT=$ROOT/mint-vla-client-example-state54-replay-v1
SESSION=rongenz-state54-trainonly-seed${SEED}-server
RUNTIME_ROOT=$SHARE/runtime-checkpoints-state54-trainonly/seed${SEED}
LOG_ROOT=$SHARE/results/training/state54_replay_train_only_v1/server_logs
SERVER_LOG=$LOG_ROOT/seed${SEED}_$(date -u +%Y%m%dT%H%M%SZ).log
for gpu in 4 5 6 7; do
  IFS=, read -r memory utilization < <(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  if (( memory > 10 || utilization != 0 )); then
    echo "GPU$gpu is not idle: memory=${memory}MiB utilization=${utilization}%" >&2
    exit 2
  fi
done
python3 - "$PORT" <<'PY'
import socket,sys
port=int(sys.argv[1]);s=socket.socket();s.settimeout(.5)
try:s.connect(("127.0.0.1",port))
except OSError:pass
else:raise SystemExit(f"port{port} is occupied")
finally:s.close()
PY
! tmux has-session -t "$SESSION" 2>/dev/null || { echo "tmux $SESSION exists" >&2;exit 2;}
mkdir -p "$RUNTIME_ROOT" "$LOG_ROOT"
tmux new-session -d -s "$SESSION" -n server
tmux send-keys -t "$SESSION:server" -l -- "bash $CLIENT/scripts/remote/run_state54_formal_server.sh $SEED $GPUS $PORT $RUNTIME_ROOT $SERVER_LOG"
tmux send-keys -t "$SESSION:server" Enter
printf 'launched tmux=%s seed=%s gpus=%s port=%s runtime=%s log=%s\n' "$SESSION" "$SEED" "$GPUS" "$PORT" "$RUNTIME_ROOT" "$SERVER_LOG"
