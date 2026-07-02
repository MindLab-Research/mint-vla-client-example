#!/usr/bin/env bash
# step2_clean_restart.sh — 干净重启 mint dev server（pi0.5 冒烟前置）
#
# 为什么需要这一步：
#   mint dev server 依赖一批 *detached* Ray actor（控制面 + OpenPI worker），
#   它们不随 server 进程的 kill -TERM 退出。若只重启 server 而不清这些 actor，
#   下一次请求会复用旧 actor 进程，而旧进程可能：
#     - 没拿到 OPENPI_DATA_HOME（cache_dir 退化成 ~/.cache/openpi → 找不到
#       paligemma tokenizer → 去 gs:// 下载 → 撞缺失的 gcsfs，save 阶段炸）；
#     - 持有过期的 scheduler owner epoch（TaskStateConflictError，/act 间歇 500）。
#
#   清掉旧 actor 后，Ray 会在下次请求时按 runtime_env 重新拉起 worker，
#   新 actor 通过 _openpi_runtime_env_vars() 拿到正确的 OPENPI_DATA_HOME，
#   tokenizer 在 share/models/openpi 命中缓存，全链路才稳定通过。
#
# 用法：bash step2_clean_restart.sh   然后  bash PI05check.sh
set -uo pipefail   # 故意不加 -e：清理步骤允许“目标本就不存在”而不中断

RUNTIME_ROOT=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime
PY="${RUNTIME_ROOT}/cpu/base-python/bin/python3.13"
RAY_SITE="${RUNTIME_ROOT}/cpu/site-packages"
RAY_ADDR=127.0.0.1:6379
NAMESPACE=mint_wenxi_dev
PORT=30496

echo "=== [1/5] 清理 ~/.cache/openpi 误建的软链残留 ==="
# 这些是早前为绕 gcsfs 误建的软链；resolve() 会跳出 cache_dir 反而报错，须移除。
CACHE=/root/.cache/openpi
[ -L "${CACHE}/big_vision/paligemma_tokenizer.model" ] && rm -v "${CACHE}/big_vision/paligemma_tokenizer.model"
for name in pi05_base pi0_fast_base pi0_fast_base_official_20260428 probe; do
  [ -L "${CACHE}/${name}" ] && rm -v "${CACHE}/${name}"
done
# 若 big_vision 已是空目录（下载失败留下的残骸），一并清掉
[ -d "${CACHE}/big_vision" ] && rmdir "${CACHE}/big_vision" 2>/dev/null && echo "removed empty ${CACHE}/big_vision"
echo "    cache 清理完成"

echo "=== [2/5] 停 server 进程 ==="
SPID=$(pgrep -f "scripts/run_server.py" | head -1)
if [ -n "${SPID}" ]; then
  kill -TERM "${SPID}" && echo "    已发 TERM 给 server pid ${SPID}"
  sleep 5
else
  echo "    没有在跑的 server"
fi

echo "=== [3/5] ray.kill 控制面 detached actor + OpenPI worker actor ==="
PYTHONPATH="${RAY_SITE}" RAY_ADDR="${RAY_ADDR}" NS="${NAMESPACE}" "${PY}" - <<'PYEOF'
import os, ray
ray.init(address=os.environ["RAY_ADDR"], namespace=os.environ["NS"])
from ray.util.state import list_actors
control_plane = {
    "mint_config",
    "mint_model_actor_supervisor",
    "mint_model_work_scheduler",
    "mint_task_state_store",
    "mint_maintenance_cron",
}
killed = 0
for a in list_actors(filters=[("state", "=", "ALIVE")], limit=500):
    name = a.name or ""
    cls = a.class_name or ""
    if "OpenPI" in cls or name in control_plane:
        try:
            ray.kill(ray.get_actor(name), no_restart=True)
            print(f"    killed {name or cls}")
            killed += 1
        except Exception as e:
            print(f"    skip {name or cls}: {str(e)[:60]}")
print(f"    共清理 {killed} 个 actor")
ray.shutdown()
PYEOF

echo "=== [4/5] 重启 server（step3_start.sh）==="
bash "$(dirname "$0")/step3_start.sh"

echo "=== [5/5] 等待并健康检查 ==="
for i in $(seq 1 20); do
  sleep 3
  if curl -s -m 5 "http://localhost:${PORT}/api/v1/healthz" 2>/dev/null | grep -q '"ready"'; then
    echo "    server ready ✓ (port ${PORT})"
    echo
    echo "现在可以跑：bash PI05check.sh"
    exit 0
  fi
done
echo "    !! ${PORT} 在 60s 内未 ready，检查 /tmp/mint_dev_launch_wenxi.log"
exit 1
