#!/usr/bin/env bash
# clean2.sh — 杀掉 mint_wenxi_dev 命名空间里的几个核心 actor（本地化，无 ssh）
#
# 替代远程版：原来 ssh 到 driver、读 share 的 head-ip、用 share 的 python。单机部署下
# 直接连本机 GCS(127.0.0.1:6379)、用固定的 mint_env cpu tier 解释器。
set -euo pipefail

PFS_RUNTIME_ENV_ROOT=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime
RUNTIME_ROOT="${PFS_RUNTIME_ENV_ROOT:-/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime}"
CPU_PY="${RUNTIME_ROOT}/cpu/base-python/bin/python3.13"
CPU_SP="${RUNTIME_ROOT}/cpu/site-packages"
GCS="${MINT_RAY_GCS:-127.0.0.1:6379}"
NS="${MINT_RAY_NAMESPACE:-mint_wenxi_dev}"

PYTHONPATH="${CPU_SP}" "${CPU_PY}" - "${GCS}" "${NS}" <<'PYEOF'
import sys
import ray

gcs, ns = sys.argv[1], sys.argv[2]
ray.init(address=gcs, namespace=ns, ignore_reinit_error=True, log_to_driver=False)
for name in [
    "mint_config",
    "mint_task_state_store",
    "mint_model_work_scheduler",
    "mint_maintenance_cron",
    "mint_model_actor_supervisor",
]:
    try:
        a = ray.get_actor(name, namespace=ns)
        ray.kill(a, no_restart=True)
        print(f"killed {name}")
    except Exception:
        pass
ray.shutdown()
PYEOF
echo "done"
