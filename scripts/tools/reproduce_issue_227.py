import os
import sys
import time
import uuid
from typing import Any

import requests


BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

# This issue is about silent fallbacks inside Megatron training codepaths.
# Use a MoE base model so the backend routes through MegatronWorkerGroup.
BASE_MODEL = os.environ.get("TINKER_MODEL") or "Qwen/Qwen3-30B-A3B-Instruct-2507"

POLL_DELAY_S = float(os.environ.get("TINKER_POLL_DELAY_S", "0.5"))


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _post(url: str, payload: dict[str, Any], timeout_s: float) -> requests.Response:
    return requests.post(url, headers=_headers(), json=payload, timeout=timeout_s)


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    r = _post(url, payload, timeout_s=timeout_s)
    if r.status_code != 200:
        raise RuntimeError(f"POST {url} returned {r.status_code}: {r.text[:400]!r}")
    j = r.json()
    if not isinstance(j, dict):
        raise RuntimeError(f"POST {url} returned non-dict json: {type(j)}")
    return j


def _get_json(url: str, timeout_s: float) -> dict[str, Any]:
    r = requests.get(url, headers=_headers(), timeout=timeout_s)
    if r.status_code != 200:
        raise RuntimeError(f"GET {url} returned {r.status_code}: {r.text[:400]!r}")
    j = r.json()
    if not isinstance(j, dict):
        raise RuntimeError(f"GET {url} returned non-dict json: {type(j)}")
    return j


def _wait_future(request_id: str, *, timeout_s: float) -> dict[str, Any]:
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed > timeout_s:
            raise TimeoutError(f"timeout waiting future request_id={request_id!r} elapsed_s={elapsed:.0f}")
        r = _post(f"{BASE_URL}/api/v1/retrieve_future", {"request_id": request_id}, timeout_s=30.0)
        if r.status_code == 408:
            time.sleep(POLL_DELAY_S)
            continue
        if r.status_code != 200:
            raise RuntimeError(f"retrieve_future {request_id!r} returned {r.status_code}: {r.text[:400]!r}")
        payload = r.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"retrieve_future returned non-dict json: {type(payload)}")
        return payload


def _require_model_in_caps(caps: dict[str, Any], model_name: str) -> None:
    models = caps.get("supported_models")
    if not isinstance(models, list):
        raise RuntimeError(f"supported_models missing/invalid: {models!r}")
    for m in models:
        if isinstance(m, dict) and m.get("model_name") == model_name:
            return
    raise RuntimeError(f"model {model_name!r} not present in supported_models")


def main() -> int:
    try:
        caps = _get_json(f"{BASE_URL}/api/v1/get_server_capabilities", timeout_s=30.0)
        _require_model_in_caps(caps, BASE_MODEL)

        session_id = f"repro227-{uuid.uuid4().hex}"
        create_fut = _post_json(
            f"{BASE_URL}/api/v1/create_model",
            {
                "session_id": session_id,
                "model_seq_id": 0,
                "base_model": BASE_MODEL,
                "lora_config": {"rank": 16},
                "user_metadata": {"tags": ["scripts/tools/reproduce_issue_227.py"]},
            },
            timeout_s=60.0,
        )
        create_request_id = create_fut.get("request_id")
        if not isinstance(create_request_id, str) or not create_request_id:
            return _fail(f"create_model missing request_id: {create_fut!r}")

        created = _wait_future(create_request_id, timeout_s=1800.0)
        if "error" in created:
            return _fail(f"create_model failed: {created!r}")
        model_id = created.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            return _fail(f"create_model result missing model_id: {created!r}")

        # Trigger the silent fallback: provide a datum with empty token list, which is currently
        # skipped during preprocessing. MegatronWorkerGroup then pads loss_fn_outputs to match
        # the original request length.
        #
        # Expected after fix: the operation should fail fast and surface an explicit error.
        data_items = [
            {
                "model_input": {"chunks": [{"tokens": [], "type": "encoded_text"}]},
                "loss_fn_inputs": {},
            },
            {
                "model_input": {"chunks": [{"tokens": [1, 2, 3, 4], "type": "encoded_text"}]},
                "loss_fn_inputs": {
                    "target_tokens": {"data": [2, 3, 4, 5], "shape": [4], "dtype": "int64"},
                    "weights": {"data": [0.0, 1.0, 1.0, 1.0], "shape": [4], "dtype": "float32"},
                },
            },
        ]

        fb_fut = _post_json(
            f"{BASE_URL}/api/v1/forward_backward",
            {
                "model_id": model_id,
                "seq_id": 0,
                "forward_backward_input": {
                    "loss_fn": "cross_entropy",
                    "loss_fn_config": None,
                    "data": data_items,
                },
            },
            timeout_s=60.0,
        )
        fb_request_id = fb_fut.get("request_id")
        if not isinstance(fb_request_id, str) or not fb_request_id:
            return _fail(f"forward_backward missing request_id: {fb_fut!r}")

        fb = _wait_future(fb_request_id, timeout_s=1800.0)
        if "error" in fb:
            print("PASS")
            return 0

        # Old behavior (bug): succeed and silently pad/replace loss_fn_outputs to match request length.
        outputs = fb.get("loss_fn_outputs")
        if not isinstance(outputs, list):
            return _fail(f"expected forward_backward to FAIL; got non-error payload={fb!r}")

        padded = []
        for i, out in enumerate(outputs):
            if not isinstance(out, dict):
                continue
            logprobs = out.get("logprobs")
            if isinstance(logprobs, dict) and isinstance(logprobs.get("shape"), list) and logprobs.get("shape") == [0]:
                padded.append(i)

        return _fail(
            "expected forward_backward to FAIL; got success payload "
            f"loss_fn_outputs_len={len(outputs)} padded_indices={padded} keys={sorted(fb.keys())}"
        )
    except Exception as e:
        return _fail(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
