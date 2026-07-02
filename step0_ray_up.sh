#!/usr/bin/env bash
# step0_ray_up.sh — 单机起一套 Ray 集群（head + worker 两进程，同机）
#
# 替代三机时代的「远程 Volcano head pod + worker pod」。在本台 8 卡机上：
#   - head 进程：0 GPU，提供 GCS(:6379) + dashboard(:8265)
#   - worker 进程：注册到本机 GCS，广播全部 8 张卡
#
# 关键约束（见 wenxi_dev_md/Ray_Deployment.md §8 铁律）：
#   1. head 必须 0 GPU，否则 gen_dev_placement.py 的「跳过 head」逻辑被搞混。
#   2. head 自报 127.0.0.1，worker 自报真实 IP，二者必须不同——否则 placement
#      生成器（gen_dev_placement.py:104 的 node_ip==head_ip）会把 worker 误杀。
#   3. ray start 的解释器 Ray 版本必须与 gpu_rl tier 一致（都是 2.51.1，已验证）。
#   4. 直连 GCS，绝不用 ray:// client 模式。
#
# 用固定后的 /vePFS-Mindverse/user/intern/wenxi/mint_env runtime（cpu tier 解释器）起 Ray。
set -euo pipefail
PFS_RUNTIME_ENV_ROOT=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime

RUNTIME_ROOT="${PFS_RUNTIME_ENV_ROOT:-/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime}"
CPU_PY="${RUNTIME_ROOT}/cpu/base-python/bin/python3.13"
CPU_SP="${RUNTIME_ROOT}/cpu/site-packages"

# CUDA forward-compat libcuda：主机驱动 535 跑 cu13 torch / jax(cuda13) 都需要
# /usr/local/cuda/compat 里的 580 libcuda。compat/lib 软链（指向 compat 本体，
# .so 直接在该目录下）是容器本地的，重启易失——这里固化重建，避免 actor 看不到
# GPU。见 wenxi_dev_md/Pin_Runtime.md 及 cuda13-compat 记忆。
if [ -d /usr/local/cuda/compat ] && [ ! -e /usr/local/cuda/compat/lib ]; then
  ln -sfn /usr/local/cuda/compat /usr/local/cuda/compat/lib
  echo "已重建 compat/lib 软链 -> /usr/local/cuda/compat"
fi


# standalone 解释器没有 `ray` console-script，用模块入口调用。
RAY=( env "PYTHONPATH=${CPU_SP}" "${CPU_PY}" -m ray.scripts.scripts )

# worker 自报 IP：取本机第一个 192.168.* 地址（与 placement 的 worker node_ip 对齐）。
WORKER_IP="${MINT_RAY_WORKER_IP:-$(hostname -I | tr ' ' '\n' | grep -E '^192\.168\.' | head -1)}"
if [ -z "${WORKER_IP}" ]; then
  echo "error: 无法确定 worker IP（hostname -I 没有 192.168.* 地址），请设 MINT_RAY_WORKER_IP" >&2
  exit 1
fi

echo "=== ray 版本（用于 start 的解释器） ==="
"${RAY[@]}" --version 2>/dev/null || PYTHONPATH="${CPU_SP}" "${CPU_PY}" -c "import ray;print('ray',ray.__version__)"

echo "=== 起 head：127.0.0.1, 0 GPU, GCS :6379 + dashboard :8265 ==="
"${RAY[@]}" start --head \
  --node-ip-address=127.0.0.1 \
  --num-gpus=0 \
  --port=6379 \
  --dashboard-host=0.0.0.0 --dashboard-port=8265 \
  --disable-usage-stats

sleep 3

echo "=== 起 worker：${WORKER_IP}, 8 GPU, 注册到 127.0.0.1:6379 ==="
"${RAY[@]}" start \
  --address=127.0.0.1:6379 \
  --node-ip-address="${WORKER_IP}" \
  --num-gpus=8

sleep 3

echo "=== 验证 cluster_resources（期望 GPU=8, 2 节点） ==="
PYTHONPATH="${CPU_SP}" "${CPU_PY}" - <<PYEOF
import ray
ray.init(address="127.0.0.1:6379", namespace="mint_wenxi_dev")
res = ray.cluster_resources()
nodes = [n for n in ray.nodes() if n.get("Alive")]
print("cluster_resources:", {k: v for k, v in res.items() if k in ("GPU", "CPU")})
print("alive nodes:", len(nodes))
for n in nodes:
    print("  node", n["NodeManagerAddress"], "GPU=", n.get("Resources", {}).get("GPU", 0))
assert res.get("GPU", 0) == 8, f"expected 8 GPU, got {res.get('GPU')}"
print("OK: 8 GPU 在集群里")
ray.shutdown()
PYEOF

echo
echo "worker IP = ${WORKER_IP}  （step2 的 placement node_ip 应是这个）"
echo "done: Ray 集群已就绪。下一步 step1(rsync) -> step2(placement) -> step3(server)"
