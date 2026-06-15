#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time
import uuid
from typing import Any

import requests


BASE_URL = os.environ.get("MINT_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("MINT_API_KEY", "dummy")
MINT_RAY_GCS_ADDRESS = os.environ.get("MINT_RAY_GCS_ADDRESS", "").strip()
RAY_NAMESPACE = (
    os.environ.get("MINT_RAY_NAMESPACE")
    or os.environ.get("MINT_RAY_NAMESPACE")
    or ""
).strip()
SSH_HOST = os.environ.get("MINT_DEV_SSH_HOST", "mint-dev").strip()

BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
HIGH_RANK = 64
LOW_RANK = 16
MODEL_SEQ_ID = 0
ACTOR_PREFIX = "megatron_qwen3_30b_a3b_instruct_2507"


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _post(path: str, payload: dict[str, Any], timeout_s: float = 60.0) -> requests.Response:
    return requests.post(f"{BASE_URL}{path}", headers=_headers(), json=payload, timeout=timeout_s)


def _post_json(path: str, payload: dict[str, Any], timeout_s: float = 60.0) -> dict[str, Any]:
    resp = _post(path, payload, timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"POST {path} -> {resp.status_code}: {resp.text[:400]!r}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"POST {path} returned non-dict JSON: {type(data).__name__}")
    return data


def _poll_future(request_id: str, timeout_s: float = 1800.0) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last: tuple[int, str] | None = None
    while time.time() < deadline:
        resp = _post("/api/v1/retrieve_future", {"request_id": request_id}, timeout_s=300.0)
        last = (resp.status_code, resp.text[:400])
        if resp.status_code == 408:
            time.sleep(2)
            continue
        if resp.status_code == 503 and "ModelWorkScheduler position lookup failed" in resp.text:
            time.sleep(2)
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"retrieve_future -> {resp.status_code}: {resp.text[:400]!r}")
        data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"retrieve_future returned non-dict JSON: {type(data).__name__}")
        return data
    raise RuntimeError(f"retrieve_future timeout request_id={request_id} last={last!r}")


def _create_session(tag: str) -> str:
    out = _post_json(
        "/api/v1/create_session",
        {
            "tags": [tag],
            "user_metadata": {},
            "sdk_version": "scripts/tools/reproduce_issue_476.py",
        },
    )
    session_id = out.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError(f"create_session missing session_id: {out!r}")
    return session_id


def _create_model(session_id: str, rank: int) -> str:
    out = _post_json(
        "/api/v1/create_model",
        {
            "session_id": session_id,
            "model_seq_id": MODEL_SEQ_ID,
            "base_model": BASE_MODEL,
            "lora_config": {
                "rank": int(rank),
                "train_attn": True,
                "train_mlp": True,
                "train_unembed": True,
            },
        },
        timeout_s=120.0,
    )
    request_id = out.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError(f"create_model missing request_id: {out!r}")
    result = _poll_future(request_id, timeout_s=2400.0)
    model_id = result.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"create_model missing model_id: {result!r}")
    return model_id


def _train_step(model_id: str, *, learning_rate: float = 2e-4) -> dict[str, Any]:
    target_tokens = [1, 2, 3]
    weights = [1.0, 1.0, 1.0]
    datum = {
        "model_input": {
            "chunks": [
                {"type": "encoded_text", "tokens": [1]},
                {"type": "encoded_text", "tokens": [2, 3]},
            ]
        },
        "weights": {
            "data": weights,
            "shape": [len(weights)],
            "dtype": "float32",
        },
        "loss_fn_inputs": {
            "target_tokens": {
                "data": target_tokens,
                "shape": [len(target_tokens)],
                "dtype": "int64",
            },
            "weights": {
                "data": weights,
                "shape": [len(weights)],
                "dtype": "float32",
            },
        },
    }
    out = _post_json(
        "/api/v1/train_step",
        {
            "model_id": model_id,
            "forward_backward_input": {
                "data": [datum],
                "loss_fn": "cross_entropy",
            },
            "adam_params": {"learning_rate": learning_rate},
        },
        timeout_s=120.0,
    )
    request_id = out.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError(f"train_step missing request_id: {out!r}")
    return _poll_future(request_id, timeout_s=2400.0)


def _run_python(code: str, timeout_s: float = 120.0) -> str:
    local_hosts = {"", "local", "localhost", "127.0.0.1"}
    if SSH_HOST in local_hosts:
        cmd = ["/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python", "-"]
        env = os.environ.copy()
        if MINT_RAY_GCS_ADDRESS:
            env["MINT_RAY_GCS_ADDRESS"] = MINT_RAY_GCS_ADDRESS
        if RAY_NAMESPACE:
            env["MINT_RAY_NAMESPACE"] = RAY_NAMESPACE
            env["MINT_RAY_NAMESPACE"] = RAY_NAMESPACE
    else:
        cmd = ["ssh", SSH_HOST]
        if MINT_RAY_GCS_ADDRESS:
            cmd.append(f"MINT_RAY_GCS_ADDRESS={MINT_RAY_GCS_ADDRESS}")
        if RAY_NAMESPACE:
            cmd.append(f"MINT_RAY_NAMESPACE={RAY_NAMESPACE}")
            cmd.append(f"MINT_RAY_NAMESPACE={RAY_NAMESPACE}")
        cmd.extend(["/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python", "-"])
        env = None
    proc = subprocess.run(
        cmd,
        input=code,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"python probe failed rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout


def _actor_infos() -> list[dict[str, Any]]:
    if not MINT_RAY_GCS_ADDRESS:
        raise RuntimeError("MINT_RAY_GCS_ADDRESS is required")
    if not RAY_NAMESPACE:
        raise RuntimeError("MINT_RAY_NAMESPACE or MINT_RAY_NAMESPACE is required")
    code = f"""
import json
import os
import ray

ray.init(address=os.environ["MINT_RAY_GCS_ADDRESS"], ignore_reinit_error=True)
ns = os.environ["MINT_RAY_NAMESPACE"]
rows = []
for entry in ray.util.list_named_actors(all_namespaces=True):
    if entry.get("namespace") != ns:
        continue
    name = entry.get("name") or ""
    if not name.startswith({ACTOR_PREFIX!r}):
        continue
    actor = ray.get_actor(name, namespace=ns)
    rows.append({{
        "name": name,
        "session_info": ray.get(actor.get_session_info.remote(), timeout=30),
    }})
print(json.dumps(rows))
"""
    out = _run_python(code, timeout_s=180.0).strip()
    data = json.loads(out)
    if not isinstance(data, list):
        raise RuntimeError(f"actor infos returned non-list JSON: {type(data).__name__}")
    return data


def _save_state(model_id: str, checkpoint_name: str) -> dict[str, Any]:
    out = _post_json(
        "/api/v1/save_state",
        {
            "model_id": model_id,
            "path": checkpoint_name,
        },
        timeout_s=120.0,
    )
    request_id = out.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError(f"save_state missing request_id: {out!r}")
    result = _poll_future(request_id, timeout_s=2400.0)
    if not isinstance(result, dict):
        raise RuntimeError(f"save_state returned non-dict JSON: {type(result).__name__}")
    return result


def _delete_model(model_id: str) -> None:
    try:
        requests.delete(f"{BASE_URL}/api/v1/models/{model_id}", headers=_headers(), timeout=60.0)
    except Exception:
        pass


def main() -> int:
    model_64: str | None = None
    model_16: str | None = None
    try:
        print(f"base_url={BASE_URL}")
        print(f"base_model={BASE_MODEL}")
        print(f"ray_namespace={RAY_NAMESPACE!r}")
        print(f"mint_ray_gcs_address={MINT_RAY_GCS_ADDRESS!r}")

        session_64 = _create_session(f"issue476-rank-{HIGH_RANK}-{uuid.uuid4().hex[:8]}")
        model_64 = _create_model(session_64, HIGH_RANK)
        print(f"rank={HIGH_RANK} create_model -> model_id={model_64}")
        train_64 = _train_step(model_64)
        print(f"rank={HIGH_RANK} train_step -> keys={sorted(train_64.keys())}")

        session_16 = _create_session(f"issue476-rank-{LOW_RANK}-{uuid.uuid4().hex[:8]}")
        model_16 = _create_model(session_16, LOW_RANK)
        print(f"rank={LOW_RANK} create_model -> model_id={model_16}")
        train_16 = _train_step(model_16)
        print(f"rank={LOW_RANK} train_step -> keys={sorted(train_16.keys())}")

        save_result = _save_state(model_16, f"issue476-r16-{uuid.uuid4().hex[:8]}")
        filesystem_path = save_result.get("filesystem_path")
        if not isinstance(filesystem_path, str) or not filesystem_path:
            return _fail(f"save_state missing filesystem_path: {save_result!r}")
        adapter_config_path = os.path.join(filesystem_path, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            return _fail(f"missing adapter_config.json at {adapter_config_path}")
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        exported_rank = adapter_cfg.get("r")

        infos = _actor_infos()
        print(f"actor_count={len(infos)}")

        rank_scoped = [row for row in infos if "_maxr" in str(row.get("name") or "")]
        if rank_scoped:
            return _fail(f"rank-scoped Megatron actor exists: {json.dumps(rank_scoped, sort_keys=True)}")

        matching = []
        for row in infos:
            session_info = row.get("session_info") or {}
            if session_info.get("current_session") == model_16:
                matching.append(row)

        if not matching:
            return _fail(f"no Megatron actor is serving low-rank session {model_16}; actor_infos={json.dumps(infos, sort_keys=True)}")

        ok = any(
            int((row.get("session_info") or {}).get("max_lora_rank") or -1) == HIGH_RANK
            and int((row.get("session_info") or {}).get("actual_rank") or -1) == LOW_RANK
            for row in matching
        )
        if not ok:
            return _fail(
                "rank-16 session is bound to the wrong trainer/export rank. "
                f"expected max_lora_rank={HIGH_RANK} and actual_rank={LOW_RANK}, actor_infos={json.dumps(matching, sort_keys=True)}"
            )

        if int(exported_rank or -1) != LOW_RANK:
            return _fail(
                f"exported adapter rank mismatch: adapter_config.json r={exported_rank!r} expected {LOW_RANK}; "
                f"save_result={json.dumps(save_result, sort_keys=True)} actor_infos={json.dumps(matching, sort_keys=True)}"
            )

        print(
            f"PASS: rank={LOW_RANK} session reuses shared max_lora_rank={HIGH_RANK} actor "
            f"and exports adapter rank={LOW_RANK}"
        )
        return 0
    except Exception as exc:
        return _fail(f"{type(exc).__name__}: {exc}")
    finally:
        if model_16:
            _delete_model(model_16)
        if model_64:
            _delete_model(model_64)


if __name__ == "__main__":
    raise SystemExit(main())
