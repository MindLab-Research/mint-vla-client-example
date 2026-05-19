from __future__ import annotations

import os
import sys
import time
import uuid
from typing import Any

import requests


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


BASE_URL = (os.environ.get("MINT_BASE_URL") or os.environ.get("MINT_BASE_URL") or "http://localhost:10025").rstrip(
    "/"
)
API_KEY = os.environ.get("MINT_API_KEY") or os.environ.get("MINT_API_KEY") or ""

BASE_MODEL = os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-0.6B")
LORA_A = os.environ.get("MINT_LORA_A", "mint://ckpt_a7c51d595f34")
LORA_B = os.environ.get("MINT_LORA_B", "mint://ckpt_6722008bf425")

POLL_TIMEOUT_S = float(os.environ.get("MINT_POLL_TIMEOUT_S", "600"))
EXPECTED_MAX_LORAS = int(os.environ.get("MINT_EXPECTED_MAX_LORAS", "1"))
EXPECTED_MAX_CPU_LORAS = int(os.environ.get("MINT_EXPECTED_MAX_CPU_LORAS", "1"))


def _headers() -> dict[str, str]:
    if API_KEY:
        return {"X-API-Key": API_KEY}
    return {}


def _post_json(path: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    r = requests.post(f"{BASE_URL}{path}", headers=_headers(), json=payload, timeout=timeout_s)
    try:
        data = r.json()
    except Exception:
        data = {"_non_json_body": r.text[:400]}
    if r.status_code != 200:
        raise RuntimeError(f"{path} returned {r.status_code}: {data!r}")
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} returned non-dict JSON: {type(data)}")
    return data


def _get_json(path: str, *, timeout_s: float) -> dict[str, Any]:
    r = requests.get(f"{BASE_URL}{path}", headers=_headers(), timeout=timeout_s)
    try:
        data = r.json()
    except Exception:
        data = {"_non_json_body": r.text[:400]}
    if r.status_code != 200:
        raise RuntimeError(f"{path} returned {r.status_code}: {data!r}")
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} returned non-dict JSON: {type(data)}")
    return data


def _create_sampling_session(model_path: str) -> str:
    data = _post_json(
        "/api/v1/create_sampling_session",
        payload={
            "session_id": str(uuid.uuid4()),
            "base_model": BASE_MODEL,
            "model_path": model_path,
            "lora_rank": 32,
        },
        timeout_s=300.0,
    )
    sid = data.get("sampling_session_id")
    if not isinstance(sid, str) or not sid:
        raise RuntimeError(f"create_sampling_session missing sampling_session_id: {data!r}")
    return sid


def _asample(sampling_session_id: str, *, max_tokens: int) -> str:
    payload = {
        "sampling_session_id": sampling_session_id,
        "seq_id": 0,
        "num_samples": 1,
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1000] * 512}]},
        "sampling_params": {
            "max_tokens": int(max_tokens),
            "temperature": 1.0,
            "top_k": -1,
            "top_p": 1.0,
        },
    }
    data = _post_json("/api/v1/asample", payload=payload, timeout_s=30.0)
    rid = data.get("request_id")
    if not isinstance(rid, str) or not rid:
        raise RuntimeError(f"asample missing request_id: {data!r}")
    return rid


def _retrieve(request_id: str) -> tuple[int, dict[str, Any]]:
    r = requests.post(
        f"{BASE_URL}/api/v1/retrieve_future",
        headers=_headers(),
        json={"request_id": request_id},
        timeout=30.0,
    )
    try:
        data = r.json()
    except Exception:
        data = {"_non_json_body": r.text[:400]}
    if not isinstance(data, dict):
        data = {"_non_dict_json": str(type(data))}
    return r.status_code, data


def main() -> int:
    try:
        info = _get_json("/api/v1/server_info", timeout_s=30.0)
        cfg = info.get("config")
        if not isinstance(cfg, dict):
            raise RuntimeError(f"server_info missing config: {info!r}")
        max_loras = cfg.get("max_loras")
        max_cpu_loras = cfg.get("max_cpu_loras")
        if max_loras != EXPECTED_MAX_LORAS or max_cpu_loras != EXPECTED_MAX_CPU_LORAS:
            raise RuntimeError(
                f"server not configured for eviction stress: max_loras={max_loras!r} max_cpu_loras={max_cpu_loras!r} "
                f"(expected {EXPECTED_MAX_LORAS}/{EXPECTED_MAX_CPU_LORAS})"
            )

        sid_a = _create_sampling_session(LORA_A)
        request_id_a = _asample(sid_a, max_tokens=4096)

        t0 = time.time()
        status, data = _retrieve(request_id_a)
        if status == 200:
            raise RuntimeError(f"initial retrieve_future returned 200 unexpectedly: {data!r}")
        if status != 408:
            raise RuntimeError(f"initial retrieve_future returned {status}: {data!r}")

        sid_b = _create_sampling_session(LORA_B)
        request_id_b = _asample(sid_b, max_tokens=256)

        done_a = False
        done_b = False
        err_a: dict[str, Any] | None = None
        err_b: dict[str, Any] | None = None

        while time.time() - t0 < POLL_TIMEOUT_S:
            if not done_a:
                sa, da = _retrieve(request_id_a)
                if sa == 200:
                    done_a = True
                    if "error" in da:
                        err_a = da
                elif sa != 408:
                    raise RuntimeError(f"retrieve_future A returned {sa}: {da!r}")

            if not done_b:
                sb, db = _retrieve(request_id_b)
                if sb == 200:
                    done_b = True
                    if "error" in db:
                        err_b = db
                elif sb != 408:
                    raise RuntimeError(f"retrieve_future B returned {sb}: {db!r}")

            if done_a and done_b:
                break

            time.sleep(0.5)

        if not (done_a and done_b):
            raise RuntimeError("timed out waiting for requests to complete")
        if err_a is not None:
            raise RuntimeError(f"request A failed after concurrent multi-LoRA sampling: {err_a!r}")
        if err_b is not None:
            raise RuntimeError(f"request B failed after concurrent multi-LoRA sampling: {err_b!r}")

        print("PASS")
        return 0

    except Exception as e:
        return _fail(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
