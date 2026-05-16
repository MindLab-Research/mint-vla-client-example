import os
import sys
import time
import uuid
from typing import Any

import requests


BASE_URL = os.environ.get("TINKER_BASE_URL")
if not BASE_URL:
    port = os.environ.get("TINKER_PORT", "8000")
    BASE_URL = f"http://localhost:{port}"
BASE_URL = BASE_URL.rstrip("/")

API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

MODEL = os.environ.get("TINKER_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
EXPECTED_GPUS_RAW = os.environ.get("TINKER_EXPECTED_GPUS")
POLL_TIMEOUT_S = float(os.environ.get("TINKER_POLL_TIMEOUT_S", "1800"))
POLL_SLEEP_S = float(os.environ.get("TINKER_POLL_SLEEP_S", "2.0"))


def _headers() -> dict[str, str]:
    if not API_KEY:
        return {}
    return {"X-API-Key": API_KEY}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _model_to_actor_name(model_name: str) -> str:
    if "/" in model_name:
        model_part = model_name.split("/")[-1]
    else:
        model_part = model_name
    safe_name = model_part.lower().replace(" ", "_")
    return f"tinker_vllm_{safe_name}"


def _get_json(path: str, *, timeout_s: float) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    resp = requests.get(url, headers=_headers(), timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"GET {path} -> {resp.status_code}: {resp.text[:500]!r}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"GET {path} returned non-dict json: {type(data)}")
    return data


def _post_json(path: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    resp = requests.post(url, headers=_headers(), json=payload, timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"POST {path} -> {resp.status_code}: {resp.text[:500]!r}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"POST {path} returned non-dict json: {type(data)}")
    return data


def _poll_future(request_id: str) -> dict[str, Any]:
    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        resp = requests.post(
            f"{BASE_URL}/api/v1/retrieve_future",
            headers=_headers(),
            json={"request_id": request_id},
            timeout=30.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            if not isinstance(data, dict):
                raise RuntimeError(f"retrieve_future returned non-dict json: {type(data)}")
            return data
        if resp.status_code == 408:
            time.sleep(POLL_SLEEP_S)
            continue
        raise RuntimeError(f"POST /api/v1/retrieve_future -> {resp.status_code}: {resp.text[:500]!r}")
    raise TimeoutError(f"retrieve_future timed out after {POLL_TIMEOUT_S:.1f}s request_id={request_id}")


def _require_model_in_caps(model_name: str) -> None:
    caps = _get_json("/api/v1/get_server_capabilities", timeout_s=30.0)
    models = caps.get("supported_models")
    if not isinstance(models, list):
        raise RuntimeError(f"supported_models missing/invalid: {models!r}")
    for m in models:
        if isinstance(m, dict) and m.get("model_name") == model_name:
            return
    raise RuntimeError(f"model {model_name!r} not present in supported_models (wrong server config?)")


def _get_actor_gpus(actor_name: str) -> int | None:
    status = _get_json("/api/v1/actors?type=vllm", timeout_s=30.0)
    actors = status.get("actors")
    if not isinstance(actors, list):
        raise RuntimeError(f"actors missing/invalid: {actors!r}")
    for a in actors:
        if not isinstance(a, dict):
            continue
        if a.get("actor_name") == actor_name:
            g = a.get("num_gpus")
            if isinstance(g, int):
                return g
            if isinstance(g, float):
                return int(g)
            return None
    return None


def _get_actor_pg_total_gpus(actor_name: str) -> int | None:
    status = _get_json("/api/v1/actors?type=vllm", timeout_s=30.0)
    actors = status.get("actors")
    if not isinstance(actors, list):
        raise RuntimeError(f"actors missing/invalid: {actors!r}")
    for a in actors:
        if not isinstance(a, dict):
            continue
        if a.get("actor_name") != actor_name:
            continue
        v = a.get("pg_total_gpus")
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        return None
    return None


def _expected_gpus() -> int:
    if EXPECTED_GPUS_RAW is not None and EXPECTED_GPUS_RAW.strip():
        return int(EXPECTED_GPUS_RAW)

    if MODEL == "Qwen/Qwen3-30B-A3B-Instruct-2507":
        return 4

    raise RuntimeError(
        f"TINKER_EXPECTED_GPUS is required when TINKER_MODEL is not the default; got TINKER_MODEL={MODEL!r}"
    )


def _try_kill_all_tracked_gpu_actors() -> None:
    try:
        _post_json("/api/v1/actors/kill", {"actor_type": "all"}, timeout_s=30.0)
    except Exception:
        return


def main() -> int:
    try:
        _get_json("/api/v1/healthz", timeout_s=5.0)
        _require_model_in_caps(MODEL)

        expected_gpus = _expected_gpus()
        if expected_gpus <= 0:
            return _fail(f"unexpected expected_gpus={expected_gpus} for model={MODEL!r}")

        _try_kill_all_tracked_gpu_actors()

        session = _post_json(
            "/api/v1/create_session",
            {
                "tags": ["scripts/tools/reproduce_issue_82.py", f"repro-82-{uuid.uuid4().hex[:8]}"],
                "user_metadata": {},
                "sdk_version": "scripts/tools/reproduce_issue_82.py",
            },
            timeout_s=30.0,
        )
        session_id = session.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            return _fail(f"create_session missing session_id: {session!r}")

        sampling = _post_json(
            "/api/v1/create_sampling_session",
            {"session_id": session_id, "sampling_session_seq_id": 0, "base_model": MODEL},
            timeout_s=90.0,
        )
        sampling_session_id = sampling.get("sampling_session_id")
        if not isinstance(sampling_session_id, str) or not sampling_session_id:
            return _fail(f"create_sampling_session missing sampling_session_id: {sampling!r}")

        # End-to-end request: asample -> retrieve_future
        fut = _post_json(
            "/api/v1/asample",
            {
                "sampling_session_id": sampling_session_id,
                "seq_id": 0,
                "num_samples": 1,
                "prompt": {"chunks": [{"tokens": [1, 1, 1], "type": "encoded_text"}]},
                "sampling_params": {"max_tokens": 1, "temperature": 0.0, "top_k": -1, "top_p": 1.0},
            },
            timeout_s=60.0,
        )
        request_id = fut.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return _fail(f"asample missing request_id: {fut!r}")

        out = _poll_future(request_id)
        if "error" in out:
            return _fail(f"retrieve_future error: {out.get('error')!r}")

        # Issue invariant: MultiNodeInferenceEngine must not reserve an extra GPU for controller.
        actor_name = _model_to_actor_name(MODEL)
        actor_gpus = _get_actor_gpus(actor_name)
        if actor_gpus is None:
            return _fail(f"vLLM actor not found in status: {actor_name!r}")

        pg_total_gpus = _get_actor_pg_total_gpus(actor_name)
        if pg_total_gpus is None:
            return _fail(f"placement group info missing for actor: {actor_name!r}")
        if pg_total_gpus != expected_gpus:
            return _fail(
                f"placement group reserves {pg_total_gpus} GPUs for {actor_name} (expected {expected_gpus})"
            )

        if actor_gpus != expected_gpus:
            return _fail(f"model actor registry tracks {actor_gpus} GPUs for {actor_name} (expected {expected_gpus})")

        print("PASS")
        return 0
    except Exception as e:
        return _fail(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
