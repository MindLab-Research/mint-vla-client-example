from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from typing import Any

import requests


BASE_URL = (os.environ.get("MINT_BASE_URL") or f"http://localhost:{os.environ.get('MINT_PORT', '10319')}").rstrip("/")
API_KEY = os.environ.get("MINT_API_KEY", "dummy")
SSH_HOST = os.environ.get("MINT_SSH_HOST", "mint-dev")
BASE_MODEL = os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-0.6B")
POLL_TIMEOUT_S = float(os.environ.get("MINT_POLL_TIMEOUT_S", "300"))
POLL_SLEEP_S = float(os.environ.get("MINT_POLL_SLEEP_S", "1.0"))


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr, flush=True)
    return 1


def _ssh_python(code: str) -> str:
    proc = subprocess.run(
        ["ssh", SSH_HOST, "python3 -"],
        input=code,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ssh python failed rc={proc.returncode}: {proc.stderr[-400:]}")
    return proc.stdout.strip()


def _post_json(path: str, payload: dict[str, Any], *, timeout_s: float = 60.0) -> tuple[int, dict[str, Any]]:
    resp = requests.post(f"{BASE_URL}{path}", headers=_headers(), json=payload, timeout=timeout_s)
    try:
        data = resp.json()
    except Exception:
        data = {"_non_json_body": resp.text[:800]}
    if not isinstance(data, dict):
        data = {"_non_dict_json": str(type(data))}
    return resp.status_code, data


def _get_json(path: str, *, timeout_s: float = 60.0) -> tuple[int, dict[str, Any]]:
    resp = requests.get(f"{BASE_URL}{path}", headers=_headers(), timeout=timeout_s)
    try:
        data = resp.json()
    except Exception:
        data = {"_non_json_body": resp.text[:800]}
    if not isinstance(data, dict):
        data = {"_non_dict_json": str(type(data))}
    return resp.status_code, data


def _poll_future(request_id: str) -> dict[str, Any]:
    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        status, data = _post_json("/api/v1/retrieve_future", {"request_id": request_id}, timeout_s=30.0)
        if status == 200:
            return data
        if status != 408:
            raise RuntimeError(f"retrieve_future returned {status}: {data!r}")
        time.sleep(POLL_SLEEP_S)
    raise TimeoutError(f"retrieve_future timed out after {POLL_TIMEOUT_S:.1f}s (request_id={request_id})")


def _server_checkpoints_root() -> str:
    explicit = os.environ.get("MINT_CHECKPOINTS_ROOT")
    if explicit:
        return explicit
    code = r"""
import os

pid_path = "/tmp/mint_server_issue_319.pid"
with open(pid_path, "r", encoding="utf-8") as f:
    pid = f.read().strip()
env = {}
with open(f"/proc/{pid}/environ", "rb") as f:
    for raw in f.read().split(b"\0"):
        if not raw or b"=" not in raw:
            continue
        key, value = raw.split(b"=", 1)
        env[key.decode("utf-8")] = value.decode("utf-8")
print(
    env.get("MINT_PERSISTENT_CHECKPOINT_DIR")
    or env.get("MINT_CHECKPOINT_DIR")
    or "/tos-mindverse/mint_checkpoints"
)
"""
    root = _ssh_python(code).strip()
    if not root:
        raise RuntimeError("could not determine checkpoint root from issue-scoped server env")
    return root


def _write_invalid_sampler_checkpoint(*, checkpoints_root: str, model_id: str, checkpoint_name: str) -> None:
    checkpoint_dir = os.path.join(checkpoints_root, "anonymous", model_id, checkpoint_name)
    metadata = {
        "checkpoint_id": checkpoint_name,
        "owner_id": None,
        "model_id": model_id,
        "model_name": BASE_MODEL,
        "created_at": "2026-03-14T00:00:00Z",
        "step": 5,
        "checkpoint_type": "sampler",
        "optimizer_present": False,
        "backend": "dense",
        "type": "sampler",
    }
    code = f"""
import json, os
checkpoint_dir = {checkpoint_dir!r}
metadata = {json.dumps(metadata)!r}
os.makedirs(checkpoint_dir, exist_ok=True)
with open(os.path.join(checkpoint_dir, "metadata.json"), "w", encoding="utf-8") as f:
    f.write(metadata + "\\n")
print(checkpoint_dir)
"""
    _ssh_python(code)


def _delete_checkpoint_dir(*, checkpoints_root: str, model_id: str) -> None:
    checkpoint_root = os.path.join(checkpoints_root, "anonymous", model_id)
    code = f"""
import shutil
shutil.rmtree({checkpoint_root!r}, ignore_errors=True)
print("removed")
"""
    _ssh_python(code)


def _create_session() -> str:
    status, data = _post_json(
        "/api/v1/create_session",
        {
            "tags": ["scripts/tools/reproduce_issue_319.py", f"repro-319-{uuid.uuid4().hex[:8]}"],
            "user_metadata": {},
            "sdk_version": "repro-319",
        },
    )
    if status != 200:
        raise RuntimeError(f"create_session returned {status}: {data!r}")
    session_id = data.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError(f"create_session missing session_id: {data!r}")
    return session_id


def _assert_valid_base_sampling() -> None:
    session_id = _create_session()
    status, sampling = _post_json(
        "/api/v1/create_sampling_session",
        {
            "session_id": session_id,
            "sampling_session_seq_id": 0,
            "base_model": BASE_MODEL,
        },
        timeout_s=120.0,
    )
    if status != 200:
        raise RuntimeError(f"create_sampling_session(base) returned {status}: {sampling!r}")
    sampling_session_id = sampling.get("sampling_session_id")
    if not isinstance(sampling_session_id, str) or not sampling_session_id:
        raise RuntimeError(f"create_sampling_session(base) missing sampling_session_id: {sampling!r}")

    status, sample = _post_json(
        "/api/v1/asample",
        {
            "sampling_session_id": sampling_session_id,
            "seq_id": 0,
            "num_samples": 1,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 1, 1]}]},
            "sampling_params": {"max_tokens": 1, "temperature": 0.0, "top_k": 1, "top_p": 1.0},
        },
        timeout_s=120.0,
    )
    if status != 200:
        raise RuntimeError(f"asample(base) returned {status}: {sample!r}")
    request_id = sample.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError(f"asample(base) missing request_id: {sample!r}")
    result = _poll_future(request_id)
    if "error" in result:
        raise RuntimeError(f"base sampling future returned error: {result!r}")
    sequences = result.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise RuntimeError(f"base sampling future missing sequences: {result!r}")


def main() -> int:
    model_id = f"run-319-{uuid.uuid4().hex[:8]}"
    checkpoint_name = f"sampler-bad-{uuid.uuid4().hex[:8]}"
    checkpoint_id = f"sampler_weights/{checkpoint_name}"
    model_path = f"mint://{model_id}/{checkpoint_id}"
    checkpoints_root: str | None = None
    try:
        health_status, _ = _get_json("/api/v1/healthz", timeout_s=10.0)
        if health_status != 200:
            return _fail(f"healthz returned {health_status}")

        _assert_valid_base_sampling()
        checkpoints_root = _server_checkpoints_root()
        _write_invalid_sampler_checkpoint(
            checkpoints_root=checkpoints_root,
            model_id=model_id,
            checkpoint_name=checkpoint_name,
        )

        status, listed = _get_json(f"/api/v1/training_runs/{model_id}/checkpoints")
        if status not in (200, 404):
            return _fail(f"list_checkpoints returned {status}: {listed!r}")
        if status == 404:
            print(f"checkpoint_root={checkpoints_root}", flush=True)
            print("PASS", flush=True)
            return 0
        checkpoints = listed.get("checkpoints")
        if not isinstance(checkpoints, list):
            return _fail(f"list_checkpoints payload invalid: {listed!r}")

        listed_ids = [c.get("checkpoint_id") for c in checkpoints if isinstance(c, dict)]
        if checkpoint_id in listed_ids:
            session_id = _create_session()
            create_status, create_data = _post_json(
                "/api/v1/create_sampling_session",
                {
                    "session_id": session_id,
                    "sampling_session_seq_id": 0,
                    "base_model": BASE_MODEL,
                    "model_path": model_path,
                    "lora_rank": 8,
                },
                timeout_s=120.0,
            )
            if create_status != 400:
                return _fail(
                    "invalid sampler checkpoint is still listed and create_sampling_session "
                    f"did not fail with 400: status={create_status} body={create_data!r}"
                )
            detail = str(create_data.get("detail"))
            if "Adapter weights not found" not in detail:
                return _fail(
                    "invalid sampler checkpoint is still listed but create_sampling_session failed "
                    f"for an unexpected reason: {create_data!r}"
                )
            return _fail(
                "list_checkpoints still advertised an invalid sampler checkpoint that "
                "create_sampling_session rejects"
            )

        print(f"checkpoint_root={checkpoints_root}", flush=True)
        print("PASS", flush=True)
        return 0
    except Exception as e:
        return _fail(str(e))
    finally:
        if checkpoints_root is not None:
            try:
                _delete_checkpoint_dir(checkpoints_root=checkpoints_root, model_id=model_id)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
