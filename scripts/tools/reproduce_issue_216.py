import os
import sys
import time
import uuid
from typing import Any

import requests


BASE_URL = (os.environ.get("TINKER_BASE_URL") or "http://localhost:10216").rstrip("/")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

BASE_MODEL = os.environ.get("TINKER_BASE_MODEL") or "Qwen/Qwen3-30B-A3B-Instruct-2507"
ADAPTER_PATH = os.environ.get("TINKER_ADAPTER_PATH") or (
    "/vePFS-Mindverse/share/code/tinker-server/checkpoints/"
    "eb69ea65-3302-4c97-a74e-f6838f79909d_0/rl-step-0"
)

PRESSURE_MODEL = os.environ.get("TINKER_PRESSURE_MODEL") or "moonshotai/Kimi-K2-Instruct"
PRESSURE_LORA_RANK = int(os.environ.get("TINKER_PRESSURE_LORA_RANK", "16"))
PRESSURE_CREATE_MODEL_COUNT = int(os.environ.get("TINKER_PRESSURE_CREATE_MODEL_COUNT", "2"))

IDLE_WAIT_S = float(os.environ.get("TINKER_IDLE_WAIT_S", "2.0"))
EVICT_DELAY_S = float(os.environ.get("TINKER_EVICT_DELAY_S", "2.0"))
POLL_DELAY_S = float(os.environ.get("TINKER_POLL_DELAY_S", "0.25"))
POLL_TIMEOUT_S = float(os.environ.get("TINKER_POLL_TIMEOUT_S", "300.0"))


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _get_json(path: str, *, timeout_s: float) -> dict[str, Any]:
    r = requests.get(f"{BASE_URL}{path}", headers=_headers(), timeout=timeout_s)
    r.raise_for_status()
    out = r.json()
    if not isinstance(out, dict):
        raise TypeError(f"expected dict JSON from GET {path}, got {type(out).__name__}")
    return out


def _post_json(path: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    r = requests.post(f"{BASE_URL}{path}", headers=_headers(), json=payload, timeout=timeout_s)
    r.raise_for_status()
    out = r.json()
    if not isinstance(out, dict):
        raise TypeError(f"expected dict JSON from POST {path}, got {type(out).__name__}")
    return out


def _create_sampling_session(*, base_model: str, model_path: str | None) -> str:
    payload: dict[str, Any] = {
        "session_id": str(uuid.uuid4()),
        "sampling_session_seq_id": 0,
        "base_model": base_model,
    }
    if model_path is not None:
        payload["model_path"] = model_path
    out = _post_json("/api/v1/create_sampling_session", payload, timeout_s=120.0)
    sid = out.get("sampling_session_id")
    if not isinstance(sid, str) or not sid:
        raise RuntimeError(f"create_sampling_session missing sampling_session_id: {out!r}")
    return sid


def _asample(sampling_session_id: str, *, max_tokens: int) -> str:
    payload = {
        "sampling_session_id": sampling_session_id,
        "seq_id": 0,
        "num_samples": 1,
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1000, 1000, 1000]}]},
        "sampling_params": {
            "max_tokens": int(max_tokens),
            "temperature": 1.0,
            "top_k": -1,
            "top_p": 1.0,
        },
    }
    out = _post_json("/api/v1/asample", payload, timeout_s=30.0)
    rid = out.get("request_id")
    if not isinstance(rid, str) or not rid:
        raise RuntimeError(f"asample missing request_id: {out!r}")
    return rid


def _retrieve(request_id: str) -> tuple[int, dict[str, Any]]:
    r = requests.post(
        f"{BASE_URL}/api/v1/retrieve_future",
        headers=_headers(),
        json={"request_id": request_id},
        timeout=30.0,
    )
    status = int(r.status_code)
    try:
        data = r.json()
    except Exception:
        data = {"_non_json_body": r.text[:400]}
    if not isinstance(data, dict):
        data = {"_non_dict_json": repr(data)[:400]}
    return status, data


def _poll_done(request_id: str, *, timeout_s: float) -> dict[str, Any]:
    start = time.time()
    while True:
        status, data = _retrieve(request_id)
        if status != 408:
            if status != 200:
                raise RuntimeError(f"retrieve_future status={status} data={data!r}")
            return data
        if time.time() - start > timeout_s:
            raise TimeoutError(f"timeout polling request_id={request_id}")
        time.sleep(POLL_DELAY_S)


def _list_vllm_actor_names() -> list[str]:
    out = _get_json("/api/v1/actors?type=vllm", timeout_s=10.0)
    actors = out.get("actors")
    if not isinstance(actors, list):
        raise TypeError(f"/actors unexpected payload: {out!r}")
    names: list[str] = []
    for a in actors:
        if not isinstance(a, dict):
            continue
        n = a.get("actor_name")
        if isinstance(n, str) and n:
            names.append(n)
    return names


def _create_model_pressure() -> list[str]:
    rids: list[str] = []
    for _ in range(max(0, PRESSURE_CREATE_MODEL_COUNT)):
        payload = {
            "session_id": str(uuid.uuid4()),
            "model_seq_id": 0,
            "base_model": PRESSURE_MODEL,
            "user_metadata": {"repro": "issue_216"},
            "lora_config": {"rank": int(PRESSURE_LORA_RANK)},
        }
        out = _post_json("/api/v1/create_model", payload, timeout_s=30.0)
        rid = out.get("request_id")
        if isinstance(rid, str) and rid:
            rids.append(rid)
    return rids


def main() -> int:
    try:
        health = _get_json("/api/v1/healthz", timeout_s=5.0)
        if health.get("status") != "ready":
            return _fail(f"healthz not ready: {health!r}")

        # Warm: create a base-model sampling session and complete one sample so the vLLM actor exists.
        warm_sid = _create_sampling_session(base_model=BASE_MODEL, model_path=None)
        warm_rid = _asample(warm_sid, max_tokens=1)
        warm = _poll_done(warm_rid, timeout_s=POLL_TIMEOUT_S)
        if "error" in warm:
            return _fail(f"warm sample failed: {warm!r}")

        vllm_names = _list_vllm_actor_names()
        if not vllm_names:
            return _fail("expected at least one vLLM actor after warm sample, got none")
        vllm_name = vllm_names[0]

        # Wait for actor to become evictable under small SESSION_IDLE_TIMEOUT setups.
        time.sleep(max(0.0, IDLE_WAIT_S))

        # Create a LoRA sampling session and start a sample (first /asample triggers LoRA load).
        lora_sid = _create_sampling_session(base_model=BASE_MODEL, model_path=ADAPTER_PATH)
        lora_rid = _asample(lora_sid, max_tokens=1)

        # Let LoRA load start, then start high-GPU training creates to trigger ModelActorRegistry eviction.
        time.sleep(max(0.0, EVICT_DELAY_S))
        _create_model_pressure()

        evicted = False
        start = time.time()
        while True:
            status, data = _retrieve(lora_rid)
            if status == 200:
                if "error" in data:
                    return _fail(f"asample failed: {data!r} (evicted={evicted} vllm={vllm_name})")
                if evicted:
                    return _fail(f"vLLM actor was evicted during in-flight request (vllm={vllm_name})")
                print("PASS")
                return 0
            if status != 408:
                return _fail(f"retrieve_future returned status={status} data={data!r}")

            # While the request is pending, verify the vLLM actor stays registered.
            if vllm_name not in _list_vllm_actor_names():
                evicted = True

            if time.time() - start > POLL_TIMEOUT_S:
                return _fail(f"timeout polling lora request_id={lora_rid} (evicted={evicted} vllm={vllm_name})")
            time.sleep(POLL_DELAY_S)
    except Exception as e:
        return _fail(str(e))


if __name__ == "__main__":
    raise SystemExit(main())

