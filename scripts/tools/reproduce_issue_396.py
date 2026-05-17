from __future__ import annotations

import os
import sys
import time
import uuid
from typing import Any

import requests


BASE_URL = (os.environ.get("TINKER_BASE_URL") or "http://localhost:10396").rstrip("/")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")
MODEL = os.environ.get("TINKER_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")

VALID_TOPK = int(os.environ.get("TINKER_VALID_TOPK", "1"))
INVALID_TOPK = int(os.environ.get("TINKER_INVALID_TOPK", "21"))
HTTP_TIMEOUT_S = float(os.environ.get("TINKER_HTTP_TIMEOUT_S", "60"))
POLL_TIMEOUT_S = float(os.environ.get("TINKER_POLL_TIMEOUT_S", "600"))
POLL_SLEEP_S = float(os.environ.get("TINKER_POLL_SLEEP_S", "1.0"))

PROMPT_TOKENS = [101, 102, 103]


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr, flush=True)
    return 1


def _actor_name_for_model(model_name: str) -> str:
    model_part = model_name.split("/")[-1] if "/" in model_name else model_name
    return f"tinker_vllm_{model_part.lower().replace(' ', '_')}"


def _get_json(path: str) -> tuple[int, dict[str, Any]]:
    resp = requests.get(f"{BASE_URL}{path}", headers=_headers(), timeout=HTTP_TIMEOUT_S)
    try:
        data = resp.json()
    except Exception:
        data = {"_non_json_body": resp.text[:800]}
    if not isinstance(data, dict):
        data = {"_non_dict_json": str(type(data))}
    return resp.status_code, data


def _post_json(path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    resp = requests.post(f"{BASE_URL}{path}", headers=_headers(), json=payload, timeout=HTTP_TIMEOUT_S)
    try:
        data = resp.json()
    except Exception:
        data = {"_non_json_body": resp.text[:800]}
    if not isinstance(data, dict):
        data = {"_non_dict_json": str(type(data))}
    return resp.status_code, data


def _require_healthz_ready() -> None:
    status, body = _get_json("/api/v1/healthz")
    if status != 200 or body.get("status") != "ready":
        raise RuntimeError(f"healthz expected 200/ready, got status={status} body={body!r}")


def _require_model_supported() -> None:
    status, body = _get_json("/api/v1/get_server_capabilities")
    if status != 200:
        raise RuntimeError(f"get_server_capabilities returned {status}: {body!r}")
    models = body.get("supported_models")
    if not isinstance(models, list):
        raise RuntimeError(f"supported_models missing/invalid: {body!r}")
    for entry in models:
        if isinstance(entry, dict) and entry.get("model_name") == MODEL:
            return
    raise RuntimeError(f"model {MODEL!r} not present in supported_models: {models!r}")


def _create_sampling_session() -> str:
    status, session = _post_json(
        "/api/v1/create_session",
        {
            "tags": ["scripts/tools/reproduce_issue_396.py", f"issue396-{uuid.uuid4().hex[:8]}"],
            "user_metadata": {},
            "sdk_version": "repro-396",
        },
    )
    if status != 200:
        raise RuntimeError(f"create_session returned {status}: {session!r}")
    session_id = session.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError(f"create_session missing session_id: {session!r}")

    status, sampling = _post_json(
        "/api/v1/create_sampling_session",
        {"session_id": session_id, "sampling_session_seq_id": 0, "base_model": MODEL},
    )
    if status != 200:
        raise RuntimeError(f"create_sampling_session returned {status}: {sampling!r}")
    sampling_session_id = sampling.get("sampling_session_id")
    if not isinstance(sampling_session_id, str) or not sampling_session_id:
        raise RuntimeError(f"create_sampling_session missing sampling_session_id: {sampling!r}")
    return sampling_session_id


def _submit_request(*, sampling_session_id: str, seq_id: int, topk_prompt_logprobs: int) -> str:
    status, body = _post_json(
        "/api/v1/asample",
        {
            "sampling_session_id": sampling_session_id,
            "seq_id": int(seq_id),
            "num_samples": 1,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": PROMPT_TOKENS}]},
            "sampling_params": {"max_tokens": 1, "temperature": 0.0, "top_k": 1, "top_p": 1.0},
            "prompt_logprobs": True,
            "include_prompt_logprobs": False,
            "topk_prompt_logprobs": int(topk_prompt_logprobs),
        },
    )
    if status != 200:
        raise RuntimeError(f"asample returned {status}: {body!r}")
    request_id = body.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError(f"asample missing request_id: {body!r}")
    return request_id


def _poll_future(request_id: str) -> dict[str, Any]:
    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        resp = requests.post(
            f"{BASE_URL}/api/v1/retrieve_future",
            headers=_headers(),
            json={"request_id": request_id},
            timeout=max(30.0, HTTP_TIMEOUT_S),
        )
        if resp.status_code == 408:
            time.sleep(POLL_SLEEP_S)
            continue
        try:
            body = resp.json()
        except Exception:
            body = {"_non_json_body": resp.text[:800]}
        if not isinstance(body, dict):
            body = {"_non_dict_json": str(type(body))}
        if resp.status_code != 200:
            raise RuntimeError(f"retrieve_future returned {resp.status_code}: {body!r}")
        return body
    raise TimeoutError(f"retrieve_future timed out for request_id={request_id}")


def _list_vllm_actors() -> list[dict[str, Any]]:
    status, body = _get_json("/internal/actors?type=vllm")
    if status != 200:
        raise RuntimeError(f"GET /internal/actors?type=vllm returned {status}: {body!r}")
    actors = body.get("actors")
    if not isinstance(actors, list):
        raise RuntimeError(f"actors payload missing list: {body!r}")
    out: list[dict[str, Any]] = []
    for actor in actors:
        if isinstance(actor, dict):
            out.append(actor)
    return out


def _actor_entry(actor_name: str) -> dict[str, Any] | None:
    for actor in _list_vllm_actors():
        if actor.get("actor_name") == actor_name:
            return actor
    return None


def _assert_valid_topk_result(result: dict[str, Any]) -> None:
    if result.get("error"):
        raise RuntimeError(f"valid request unexpectedly failed: {result['error']!r}")
    topk = result.get("topk_prompt_logprobs")
    if not isinstance(topk, list) or len(topk) != len(PROMPT_TOKENS):
        raise RuntimeError(f"topk_prompt_logprobs shape mismatch: {topk!r}")
    if topk[0] is not None:
        raise RuntimeError(f"topk_prompt_logprobs[0] expected None, got {topk[0]!r}")
    if not isinstance(topk[1], list) or not topk[1]:
        raise RuntimeError(f"topk_prompt_logprobs[1] missing non-empty list: {topk!r}")
    if not isinstance(topk[2], list) or not topk[2]:
        raise RuntimeError(f"topk_prompt_logprobs[2] missing non-empty list: {topk!r}")
    sequences = result.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise RuntimeError(f"valid sample missing sequences: {result!r}")


def main() -> int:
    actor_name = _actor_name_for_model(MODEL)
    print(f"BASE_URL={BASE_URL} model={MODEL} actor_name={actor_name}", flush=True)
    try:
        _require_healthz_ready()
        _require_model_supported()
        sampling_session_id = _create_sampling_session()

        warm_request_id = _submit_request(
            sampling_session_id=sampling_session_id,
            seq_id=0,
            topk_prompt_logprobs=VALID_TOPK,
        )
        warm_result = _poll_future(warm_request_id)
        _assert_valid_topk_result(warm_result)

        warm_actor = _actor_entry(actor_name)
        if warm_actor is None:
            raise RuntimeError(f"expected vLLM actor missing after warmup: {actor_name!r}")
        print(f"warm_actor={actor_name} num_gpus={warm_actor.get('num_gpus')}", flush=True)

        invalid_request_id = _submit_request(
            sampling_session_id=sampling_session_id,
            seq_id=1,
            topk_prompt_logprobs=INVALID_TOPK,
        )
        invalid_result = _poll_future(invalid_request_id)
        invalid_error = invalid_result.get("error")
        if not isinstance(invalid_error, str) or "Requested prompt logprobs" not in invalid_error:
            raise RuntimeError(f"invalid request returned wrong error payload: {invalid_result!r}")
        print(f"invalid_error={invalid_error}", flush=True)

        actor_after_invalid = _actor_entry(actor_name)
        if actor_after_invalid is None:
            raise RuntimeError(
                f"validation error killed actor {actor_name!r}; expected actor to remain present in /internal/actors"
            )

        followup_request_id = _submit_request(
            sampling_session_id=sampling_session_id,
            seq_id=2,
            topk_prompt_logprobs=VALID_TOPK,
        )
        followup_result = _poll_future(followup_request_id)
        _assert_valid_topk_result(followup_result)

        print("PASS", flush=True)
        return 0
    except Exception as exc:
        return _fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
