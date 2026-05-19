import os
import shlex
import subprocess
import sys
import time
import uuid
from typing import Any

import requests

BASE_URL = os.environ.get("MINT_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("MINT_API_KEY", "dummy")
SSH_HOST = os.environ.get("MINT_SSH_HOST", "volcano")

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"  # MoE => Megatron backend (TP=4)
BASE_MODEL = os.environ.get("MINT_MODEL", DEFAULT_MODEL)
LORA_RANK = int(os.environ.get("MINT_LORA_RANK", "8"))

CREATE_TIMEOUT_S = float(os.environ.get("MINT_CREATE_MODEL_TIMEOUT_S", "3600"))
SAVE_TIMEOUT_S = float(os.environ.get("MINT_SAVE_STATE_TIMEOUT_S", "3600"))


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    resp = requests.post(url, headers=_headers(), json=payload, timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"POST {url} returned {resp.status_code}: {resp.text[:400]!r}")
    return resp.json()


def _poll_future(request_id: str, timeout_s: float) -> dict[str, Any]:
    url = f"{BASE_URL}/api/v1/retrieve_future"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = requests.post(url, headers=_headers(), json={"request_id": request_id}, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 408:
            time.sleep(2)
            continue
        raise RuntimeError(f"POST {url} returned {resp.status_code}: {resp.text[:400]!r}")
    raise TimeoutError(f"retrieve_future timed out after {timeout_s}s (request_id={request_id})")


def _ssh_find_optimizer_files(checkpoint_path: str) -> list[str]:
    q = shlex.quote(checkpoint_path)
    cmd = [
        "ssh",
        SSH_HOST,
        f"find {q} -maxdepth 2 -type f -name '*_optimizer.pt' -print || true",
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return [ln for ln in out.splitlines() if ln.strip()]


def main() -> int:
    model_id: str | None = None
    try:
        # Create a training model (MoE => Megatron backend) so /save_state uses MegatronRankWorker.save_checkpoint.
        session_id = f"repro-67-{uuid.uuid4().hex[:8]}"
        create_payload = {
            "session_id": session_id,
            "model_seq_id": 0,
            "base_model": BASE_MODEL,
            "lora_config": {"rank": LORA_RANK},
        }
        created = _post_json(f"{BASE_URL}/api/v1/create_model", create_payload, timeout_s=60.0)
        if "request_id" in created:
            created = _poll_future(str(created["request_id"]), timeout_s=CREATE_TIMEOUT_S)

        model_id = created.get("model_id")
        if not model_id:
            return _fail(f"create_model missing model_id: {created!r}")
        backend = created.get("backend")
        if backend != "megatron":
            return _fail(f"expected backend=megatron (MoE model). got backend={backend!r} (model={BASE_MODEL!r})")

        # Save state (should include optimizer shards on disk when issue #67 is fixed).
        saved = _post_json(
            f"{BASE_URL}/api/v1/save_state",
            {"model_id": model_id},
            timeout_s=60.0,
        )
        if "request_id" in saved:
            saved = _poll_future(str(saved["request_id"]), timeout_s=SAVE_TIMEOUT_S)

        ckpt_path = saved.get("path")
        if not isinstance(ckpt_path, str) or not ckpt_path:
            return _fail(f"save_state missing path: {saved!r}")

        opt_files = _ssh_find_optimizer_files(ckpt_path)
        if not opt_files:
            return _fail(f"no *_optimizer.pt files found under checkpoint dir: {ckpt_path}")

        print("PASS")
        return 0
    except Exception as e:
        return _fail(str(e))
    finally:
        if model_id:
            try:
                requests.delete(
                    f"{BASE_URL}/api/v1/models/{model_id}",
                    headers=_headers(),
                    timeout=60,
                )
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
