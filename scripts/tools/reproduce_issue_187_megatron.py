import os
import shlex
import subprocess
import time
import uuid
from typing import Any

import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")
SSH_HOST = os.environ.get("TINKER_SSH_HOST", "mint-dev")

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
BASE_MODEL = os.environ.get("TINKER_MODEL", DEFAULT_MODEL)
LORA_RANK = int(os.environ.get("TINKER_LORA_RANK", "8"))


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY, "Content-Type": "application/json"}


def _post(url: str, payload: dict[str, Any], *, timeout_s: float) -> requests.Response:
    return requests.post(url, headers=_headers(), json=payload, timeout=timeout_s)


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    r = _post(url, payload, timeout_s=timeout_s)
    if r.status_code >= 400:
        raise RuntimeError(f"POST {url} -> {r.status_code}: {r.text[:400]!r}")
    return r.json()


def _poll_future(request_id: str, *, timeout_s: float) -> dict[str, Any]:
    url = f"{BASE_URL}/api/v1/retrieve_future"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = requests.post(url, headers=_headers(), json={"request_id": request_id}, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 408:
            time.sleep(2.0)
            continue
        raise RuntimeError(f"POST {url} returned {resp.status_code}: {resp.text[:400]!r}")
    raise TimeoutError(f"retrieve_future timed out after {timeout_s}s (request_id={request_id})")


def _ssh(cmd: str) -> str:
    return subprocess.check_output(["ssh", SSH_HOST, cmd], text=True).strip()


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", flush=True)
    return 1


def main() -> int:
    model_id: str | None = None
    try:
        session_id = f"repro-187-mega-{uuid.uuid4().hex[:8]}"
        created = _post_json(
            f"{BASE_URL}/api/v1/create_model",
            {
                "session_id": session_id,
                "model_seq_id": 0,
                "base_model": BASE_MODEL,
                "lora_config": {"rank": LORA_RANK},
            },
            timeout_s=60.0,
        )
        if "request_id" in created:
            created = _poll_future(str(created["request_id"]), timeout_s=3600.0)

        model_id = created.get("model_id")
        backend = created.get("backend")
        if not isinstance(model_id, str) or not model_id:
            return _fail(f"create_model missing model_id: {created!r}")
        if backend != "megatron":
            return _fail(f"expected backend=megatron; got {backend!r} (model={BASE_MODEL!r})")

        saved = _post_json(
            f"{BASE_URL}/api/v1/save_state",
            {"model_id": model_id, "path": "mega-step-0"},
            timeout_s=60.0,
        )
        if "request_id" in saved:
            saved = _poll_future(str(saved["request_id"]), timeout_s=3600.0)

        training_uri = saved.get("path")
        training_fs_path = saved.get("filesystem_path")
        if not isinstance(training_uri, str) or not training_uri:
            return _fail(f"save_state missing path: {saved!r}")
        if not isinstance(training_fs_path, str) or not training_fs_path:
            return _fail(f"save_state missing filesystem_path: {saved!r}")

        opt_files = _ssh(
            f"find {shlex.quote(training_fs_path)} -maxdepth 2 -type f -name '*_optimizer.pt' -print | head"
        )
        if not opt_files.strip():
            return _fail(f"save_state missing *_optimizer.pt shards under: {training_fs_path}")

        sampler = _post_json(
            f"{BASE_URL}/api/v1/save_weights_for_sampler",
            {"model_id": model_id, "path": "mega-sampler-0"},
            timeout_s=60.0,
        )
        if "request_id" in sampler:
            sampler = _poll_future(str(sampler["request_id"]), timeout_s=3600.0)
        sampler_uri = sampler.get("path")
        if not isinstance(sampler_uri, str) or not sampler_uri:
            return _fail(f"save_weights_for_sampler missing path: {sampler!r}")

        bad = _post(
            f"{BASE_URL}/api/v1/load_state",
            {"model_id": model_id, "path": sampler_uri, "optimizer": True},
            timeout_s=30.0,
        )
        if bad.status_code != 400:
            return _fail(
                f"expected 400 for load_state_with_optimizer(sampler); got {bad.status_code}: {bad.text!r}"
            )

        # Training checkpoint must not be rejected at request time.
        good = _post(
            f"{BASE_URL}/api/v1/load_state",
            {"model_id": model_id, "path": training_uri, "optimizer": True},
            timeout_s=30.0,
        )
        if good.status_code == 400:
            return _fail(f"training checkpoint was rejected: {good.text!r}")

        print("PASS", flush=True)
        return 0
    except Exception as e:
        return _fail(str(e))
    finally:
        if model_id:
            try:
                requests.delete(
                    f"{BASE_URL}/api/v1/models/{model_id}", headers=_headers(), timeout=60
                )
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

