#!/usr/bin/env bash
# cleanup.sh — 杀掉 mint_wenxi_dev + issue 命名空间里所有 named actor（本地化，无 ssh）
#
# 比 clean2 更彻底：遍历所有 named actor，杀掉本命名空间的全部，以及
# mint_wenxi_issue_* 这些 issue-scoped 命名空间的全部。
# 替代远程版：直接连本机 GCS(127.0.0.1:6379)、用固定的 mint_env cpu tier 解释器。
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

gcs, ns_main = sys.argv[1], sys.argv[2]
ray.init(address=gcs, namespace=ns_main, ignore_reinit_error=True, log_to_driver=False)


def _kill(name, ns):
    try:
        a = ray.get_actor(name, namespace=ns)
        ray.kill(a, no_restart=True)
        print(f"killed {name} in {ns}")
    except Exception as e:
        print(f"skip {name} in {ns}: {e}")


for actor in ray.util.list_named_actors(all_namespaces=True):
    ns = str(actor.get("namespace") or "")
    name = str(actor.get("name") or "")
    if not name:
        continue
    if ns == ns_main or ns.startswith("mint_wenxi_issue_"):
        _kill(name, ns)
ray.shutdown()
PYEOF
echo "done"
