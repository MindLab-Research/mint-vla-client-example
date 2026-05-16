#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
import time
import uuid
from typing import Any

import requests


BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")
RAY_ADDRESS = os.environ.get("RAY_ADDRESS", "").strip()
RAY_NAMESPACE = (
    os.environ.get("TINKER_RAY_NAMESPACE")
    or os.environ.get("MINT_RAY_NAMESPACE")
    or ""
).strip()
SSH_HOST = os.environ.get("TINKER_DEV_SSH_HOST", "mint-dev").strip() or "mint-dev"

BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
ALLOWED_RANK = 64
REJECTED_RANK = 128
MODEL_SEQ_ID = 0


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _normalize_model_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return normalized.strip("_")


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
            "sdk_version": "scripts/tools/reproduce_issue_416.py",
        },
    )
    session_id = out.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError(f"create_session missing session_id: {out!r}")
    return session_id


def _create_model(session_id: str, rank: int) -> dict[str, Any]:
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
    return _poll_future(request_id, timeout_s=2400.0)


def _forward_backward(model_id: str) -> dict[str, Any]:
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
        "/api/v1/forward_backward",
        {
            "model_id": model_id,
            "forward_backward_input": {
                "data": [datum],
                "loss_fn": "cross_entropy",
            },
        },
        timeout_s=120.0,
    )
    request_id = out.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError(f"forward_backward missing request_id: {out!r}")
    return _poll_future(request_id, timeout_s=2400.0)


def _delete_model(model_id: str) -> None:
    try:
        requests.delete(f"{BASE_URL}/api/v1/models/{model_id}", headers=_headers(), timeout=60.0)
    except Exception:
        pass


def _ssh_python(code: str, timeout_s: float = 120.0) -> str:
    local_hosts = {"", "local", "localhost", "127.0.0.1"}
    if SSH_HOST in local_hosts:
        cmd = ["/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python", "-"]
        env = os.environ.copy()
        if RAY_ADDRESS:
            env["RAY_ADDRESS"] = RAY_ADDRESS
        if RAY_NAMESPACE:
            env["TINKER_RAY_NAMESPACE"] = RAY_NAMESPACE
            env["MINT_RAY_NAMESPACE"] = RAY_NAMESPACE
    else:
        cmd = ["ssh", SSH_HOST]
        if RAY_ADDRESS:
            cmd.extend([f"RAY_ADDRESS={RAY_ADDRESS}"])
        if RAY_NAMESPACE:
            cmd.extend([f"TINKER_RAY_NAMESPACE={RAY_NAMESPACE}", f"MINT_RAY_NAMESPACE={RAY_NAMESPACE}"])
        cmd.extend(
            [
                "/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python",
                "-",
            ]
        )
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
            f"ssh python failed rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout


def _actor_state() -> dict[str, Any]:
    if not RAY_ADDRESS:
        raise RuntimeError("RAY_ADDRESS is required")
    if not RAY_NAMESPACE:
        raise RuntimeError("TINKER_RAY_NAMESPACE or MINT_RAY_NAMESPACE is required")
    actor_name = f"megatron_{_normalize_model_name(BASE_MODEL)}"
    code = f"""
import json
import os
import ray

ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True)
ns = os.environ["TINKER_RAY_NAMESPACE"]
actor = ray.get_actor({actor_name!r}, namespace=ns)
out = {{
    "diagnostics": ray.get(actor.get_diagnostics.remote()),
    "session_info": ray.get(actor.get_session_info.remote()),
}}
print(json.dumps(out))
"""
    out = _ssh_python(code, timeout_s=180.0).strip()
    if not out:
        raise RuntimeError("empty actor state output")
    data = json.loads(out)
    if not isinstance(data, dict):
        raise RuntimeError(f"actor state returned non-dict JSON: {type(data).__name__}")
    return data


def main() -> int:
    allowed_model_id: str | None = None
    rejected_model_id: str | None = None
    try:
        print(f"base_url={BASE_URL}")
        print(f"base_model={BASE_MODEL}")
        print(f"ray_namespace={RAY_NAMESPACE!r}")
        print(f"ray_address={RAY_ADDRESS!r}")

        allowed_session = _create_session(f"issue416-rank-{ALLOWED_RANK}-{uuid.uuid4().hex[:8]}")
        allowed_create = _create_model(allowed_session, ALLOWED_RANK)
        allowed_model_id = allowed_create.get("model_id")
        if not isinstance(allowed_model_id, str) or not allowed_model_id:
            return _fail(f"rank={ALLOWED_RANK} create_model missing model_id: {allowed_create!r}")
        print(f"rank={ALLOWED_RANK} create_model -> model_id={allowed_model_id}")

        allowed_fb = _forward_backward(allowed_model_id)
        print(f"rank={ALLOWED_RANK} forward_backward -> keys={sorted(allowed_fb.keys())}")

        rejected_session = _create_session(f"issue416-rank-{REJECTED_RANK}-{uuid.uuid4().hex[:8]}")
        try:
            rejected_create = _create_model(rejected_session, REJECTED_RANK)
        except Exception as exc:
            print(f"PASS: rank={REJECTED_RANK} create_model rejected: {type(exc).__name__}: {exc}")
            return 0

        rejected_model_id = rejected_create.get("model_id")
        print(f"rank={REJECTED_RANK} create_model UNEXPECTEDLY succeeded -> model_id={rejected_model_id}")
        rejected_fb = _forward_backward(str(rejected_model_id))
        print(f"rank={REJECTED_RANK} forward_backward UNEXPECTEDLY succeeded -> keys={sorted(rejected_fb.keys())}")

        state = _actor_state()
        return _fail(
            "rank>max_lora_rank was accepted. "
            f"create_model(rank={REJECTED_RANK}) -> {rejected_create!r}; "
            f"forward_backward(rank={REJECTED_RANK}) -> keys={sorted(rejected_fb.keys())}; "
            f"actor_state={json.dumps(state, sort_keys=True)}"
        )
    except Exception as exc:
        return _fail(f"{type(exc).__name__}: {exc}")
    finally:
        if rejected_model_id:
            _delete_model(rejected_model_id)
        if allowed_model_id:
            _delete_model(allowed_model_id)


if __name__ == "__main__":
    raise SystemExit(main())
