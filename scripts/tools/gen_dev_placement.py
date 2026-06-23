#!/usr/bin/env python3
"""Generate a mint_dev_run.env file for the current Ray cluster.

Reads worker IPs from the Ray dashboard and writes a placement config
file that maps models to specific worker nodes.

Usage:
    HEAD_IP=$(cat /vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt)
    python scripts/tools/gen_dev_placement.py --head-ip $HEAD_IP \
        --model Qwen/Qwen3-0.6B --gpu-count 1 \
        --output /tmp/mint_dev_run.env
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import urllib.request


KNOWN_MODEL_GPU_COUNTS = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": 4,
}

PLACEMENT_ENV_VARS = (
    "MINT_MODEL_PLACEMENT_JSON",
    "MINT_DENSE_MODEL_PLACEMENT_JSON",
    "MINT_VLLM_MODEL_PLACEMENT_JSON",
    "MINT_MEGATRON_MODEL_PLACEMENT_JSON",
)


def _read_json_url(url: str, *, timeout_s: float = 10.0) -> object:
    with urllib.request.urlopen(url, timeout=float(timeout_s)) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _dashboard_node_rows(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if isinstance(data, dict):
        result = data.get("result")
        if isinstance(result, dict):
            rows = result.get("result")
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
        cluster = data.get("clusterStatus")
        if isinstance(cluster, dict):
            lm = cluster.get("loadMetricsReport")
            if isinstance(lm, dict):
                node_types = lm.get("nodeTypes")
                rows_out: list[dict[str, object]] = []
                if isinstance(node_types, list):
                    for item in node_types:
                        if not isinstance(item, (list, tuple)) or not item:
                            continue
                        nt = item[0]
                        if not isinstance(nt, dict):
                            continue
                        gpu = int(float(nt.get("GPU", 0) or 0))
                        for key in nt:
                            if isinstance(key, str) and key.startswith("node:"):
                                rows_out.append(
                                    {
                                        "node_ip": key[5:],
                                        "state": "ALIVE",
                                        "is_head_node": key == "node:__internal_head__",
                                        "resources_total": {"GPU": gpu},
                                    }
                                )
                                break
                return rows_out
    return []


def get_worker_ips(head_ip: str) -> list[tuple[str, int]]:
    """Get (ip, gpu_count) for all alive GPU workers from Ray dashboard."""
    urls = [
        f"http://{head_ip}:8265/api/v0/nodes",
        f"http://{head_ip}:8265/api/cluster_status",
    ]
    last_error: Exception | None = None
    rows: list[dict[str, object]] = []
    for url in urls:
        try:
            payload = _read_json_url(url, timeout_s=10.0)
        except Exception as e:
            last_error = e
            continue
        rows = _dashboard_node_rows(payload)
        if rows:
            break
    if not rows and last_error is not None:
        raise RuntimeError(f"failed to query Ray dashboard for head {head_ip}: {last_error}") from last_error
    workers = []
    for row in rows:
        state = str(row.get("state") or "").strip().upper()
        alive = state == "ALIVE" if state else bool(row.get("alive", True))
        if not alive:
            continue
        node_ip = str(row.get("node_ip") or row.get("nodeName") or "").strip()
        if not node_ip or node_ip == head_ip or bool(row.get("is_head_node") or row.get("isHeadNode")):
            continue
        resources = row.get("resources_total") or row.get("resourcesTotal") or {}
        gpu = int(float(resources.get("GPU", 0) or 0)) if isinstance(resources, dict) else 0
        if gpu <= 0:
            continue
        workers.append((node_ip, gpu))
    return workers


def _csv_models(value: str | None) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def models_from_env() -> list[str]:
    return _csv_models(os.environ.get("MINT_PERSISTENT_MODELS")) or _csv_models(
        os.environ.get("MINT_SUPPORTED_MODELS")
    )


def model_gpu_counts_from_json(value: str | None) -> dict[str, int]:
    if not value:
        return {}
    data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("model GPU count JSON must be an object")
    counts: dict[str, int] = {}
    for model, count in data.items():
        if not isinstance(model, str) or not model:
            raise ValueError("model GPU count keys must be non-empty strings")
        parsed_count = int(count)
        if parsed_count <= 0:
            raise ValueError(f"GPU count for {model!r} must be positive")
        counts[model] = parsed_count
    return counts


def model_gpu_counts_from_env() -> dict[str, int]:
    return model_gpu_counts_from_json(os.environ.get("MINT_DEV_AUTO_PLACEMENT_GPU_COUNTS_JSON"))


def gpu_count_for_model(model: str, *, default_gpu_count: int, model_gpu_counts: dict[str, int]) -> int:
    return int(model_gpu_counts.get(model, KNOWN_MODEL_GPU_COUNTS.get(model, default_gpu_count)))


def pick_worker(
    workers: list[tuple[str, int]],
    remaining_gpu_by_ip: dict[str, int],
    *,
    requested_gpu_count: int,
) -> str | None:
    selected_ip: str | None = None
    selected_remaining = -1
    for worker_ip, _worker_gpu_count in workers:
        remaining = remaining_gpu_by_ip.get(worker_ip, 0)
        if remaining >= requested_gpu_count and remaining > selected_remaining:
            selected_ip = worker_ip
            selected_remaining = remaining
    if selected_ip is None:
        return None
    remaining_gpu_by_ip[selected_ip] -= requested_gpu_count
    return selected_ip


def write_env_file(
    output: str | os.PathLike[str],
    *,
    head_ip: str,
    models: list[str],
    workers: list[tuple[str, int]],
    gpu_count: int,
    max_model_len: int,
    model_gpu_counts: dict[str, int] | None = None,
    force: bool = False,
) -> dict[str, dict[str, int | str]]:
    if not workers:
        raise RuntimeError(f"no alive GPU workers found on Ray head {head_ip}")
    if not models:
        raise RuntimeError("no models provided; set --model or MINT_PERSISTENT_MODELS/MINT_SUPPORTED_MODELS")

    per_model_gpu_counts = model_gpu_counts or {}
    remaining_gpu_by_ip = {worker_ip: int(worker_gpu_count) for worker_ip, worker_gpu_count in workers}
    placement: dict[str, dict[str, int | str]] = {}
    for model in models:
        model_gpu_count = gpu_count_for_model(
            model,
            default_gpu_count=int(gpu_count),
            model_gpu_counts=per_model_gpu_counts,
        )
        worker_ip = pick_worker(
            workers,
            remaining_gpu_by_ip,
            requested_gpu_count=model_gpu_count,
        )
        if worker_ip is None:
            raise RuntimeError(
                "not enough alive GPU worker capacity for "
                f"model {model!r} requesting {model_gpu_count} GPUs"
            )
        placement[model] = {"replica": 0, "node_ip": worker_ip, "gpu_count": model_gpu_count}

    placement_json = json.dumps(placement, sort_keys=True, separators=(",", ":"))
    overrides = {m: {"max_model_len": int(max_model_len)} for m in models}
    overrides_json = json.dumps(overrides, sort_keys=True, separators=(",", ":"))

    target = Path(output)
    if target.exists() and not force:
        raise RuntimeError(
            f"output file already exists: {target} (use --force to overwrite)"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        f.write("# Auto-generated by scripts/tools/gen_dev_placement.py\n")
        f.write(f"# Cluster head: {head_ip}\n")
        for name in PLACEMENT_ENV_VARS:
            f.write(f"export {name}={placement_json!r}\n")
        f.write(f"export MINT_MODEL_CONFIG_OVERRIDES_JSON={overrides_json!r}\n")
        f.write("export MINT_CHECKPOINT_DIR=/vePFS-Mindverse/share/mint/dev/data/checkpoints\n")
    return placement


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-ip", required=True, help="Ray head node IP")
    parser.add_argument("--model", action="append", help="Model name (repeatable)")
    parser.add_argument(
        "--models-from-env",
        action="store_true",
        help="Read models from MINT_PERSISTENT_MODELS or MINT_SUPPORTED_MODELS",
    )
    parser.add_argument("--gpu-count", type=int, default=1, help="GPUs per model (default: 1)")
    parser.add_argument(
        "--model-gpu-count-json",
        default=None,
        help=(
            "JSON object mapping model names to GPU counts. Defaults to "
            "MINT_DEV_AUTO_PLACEMENT_GPU_COUNTS_JSON plus known dev model sizes."
        ),
    )
    parser.add_argument("--max-model-len", type=int, default=32768, help="Max model length (default: 32768)")
    parser.add_argument("--output", default="/tmp/mint_dev_run.env", help="Output file path")
    parser.add_argument("--force", action="store_true", help="Overwrite output file if it already exists")
    args = parser.parse_args()

    models = list(args.model or [])
    if args.models_from_env:
        models.extend(model for model in models_from_env() if model not in models)
    if not models:
        print("error: no models provided; pass --model or --models-from-env")
        return 1

    try:
        model_gpu_counts = model_gpu_counts_from_env()
        model_gpu_counts.update(model_gpu_counts_from_json(args.model_gpu_count_json))
        workers = get_worker_ips(args.head_ip)
        placement = write_env_file(
            args.output,
            head_ip=args.head_ip,
            models=models,
            workers=workers,
            gpu_count=args.gpu_count,
            max_model_len=args.max_model_len,
            model_gpu_counts=model_gpu_counts,
            force=args.force,
        )
    except Exception as e:
        print(f"error: {e}")
        return 1

    print(f"Found {len(workers)} GPU workers:")
    for ip, gpu in workers:
        print(f"  {ip}: {gpu} GPUs")

    print(f"\nWrote {args.output}")
    print(f"  Models: {', '.join(models)}")
    print(f"  Placement: {placement}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
