import json
import os
import sys
import time
from typing import Any

import requests

BASE_URL = os.environ.get("MINT_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("MINT_API_KEY", "dummy")
CREATE_SAMPLING_TIMEOUT_S = float(os.environ.get("MINT_CREATE_SAMPLING_TIMEOUT_S", "1200"))
POLL_TIMEOUT_S = float(os.environ.get("MINT_POLL_TIMEOUT_S", "1200"))


def _headers() -> dict[str, str]:
    if API_KEY:
        return {"X-API-Key": API_KEY}
    return {}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _get_supported_model() -> str:
    url = f"{BASE_URL}/api/v1/get_server_capabilities"
    try:
        resp = requests.get(url, headers=_headers(), timeout=30)
    except Exception as e:
        raise RuntimeError(f"GET {url} failed: {e}")
    if resp.status_code != 200:
        raise RuntimeError(f"GET {url} returned {resp.status_code}: {resp.text[:200]!r}")
    data = resp.json()
    models = data.get("supported_models")
    if not isinstance(models, list) or not models:
        raise RuntimeError(f"supported_models missing/empty: {models!r}")
    first = models[0]
    if not isinstance(first, dict) or not first.get("model_name"):
        raise RuntimeError(f"invalid supported_models entry: {first!r}")
    return str(first["model_name"])


def _post_json(url: str, payload: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
    resp = requests.post(url, headers=_headers(), json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"POST {url} returned {resp.status_code}: {resp.text[:200]!r}")
    return resp.json()


def _poll_future(request_id: str, timeout_s: float = 120.0) -> dict[str, Any]:
    url = f"{BASE_URL}/api/v1/retrieve_future"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = requests.post(url, headers=_headers(), json={"request_id": request_id}, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 408:
            time.sleep(2)
            continue
        raise RuntimeError(f"POST {url} returned {resp.status_code}: {resp.text[:200]!r}")
    raise RuntimeError(f"retrieve_future timed out after {timeout_s}s (request_id={request_id})")


def _validate_topk(topk: Any, prompt_len: int, k: int) -> None:
    if topk is None:
        raise RuntimeError("topk_prompt_logprobs is None")
    if not isinstance(topk, list):
        raise RuntimeError(f"topk_prompt_logprobs is not a list: {type(topk)}")
    if len(topk) != prompt_len:
        raise RuntimeError(f"topk_prompt_logprobs length mismatch: expected {prompt_len}, got {len(topk)}")
    if topk and topk[0] is not None:
        raise RuntimeError(f"topk_prompt_logprobs[0] must be None, got {topk[0]!r}")

    saw_entries = False
    for i, entry in enumerate(topk[1:], start=1):
        if entry is None:
            continue
        if not isinstance(entry, list):
            raise RuntimeError(
                f"topk_prompt_logprobs[{i}] must be list[tuple[int,float]]; got {type(entry)}"
            )
        if len(entry) > k:
            raise RuntimeError(
                f"topk_prompt_logprobs[{i}] has {len(entry)} entries, expected <= {k}"
            )
        for pair in entry:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise RuntimeError(f"topk_prompt_logprobs[{i}] contains invalid pair: {pair!r}")
            tok, lp = pair
            if not isinstance(tok, int):
                raise RuntimeError(f"topk_prompt_logprobs[{i}] token id not int: {tok!r}")
            if not isinstance(lp, (int, float)):
                raise RuntimeError(f"topk_prompt_logprobs[{i}] logprob not float: {lp!r}")
        saw_entries = True

    if not saw_entries:
        raise RuntimeError("topk_prompt_logprobs has no non-None entries")


def main() -> int:
    try:
        base_model = _get_supported_model()
    except Exception as e:
        return _fail(str(e))

    session_id = None
    try:
        session = _post_json(
            f"{BASE_URL}/api/v1/create_session",
            {
                "tags": ["scripts/tools/reproduce_issue_61.py"],
                "user_metadata": {},
                "sdk_version": "scripts/tools/reproduce_issue_61.py",
            },
        )
        session_id = session.get("session_id")
        if not session_id:
            return _fail(f"create_session missing session_id: {session!r}")

        sampling = _post_json(
            f"{BASE_URL}/api/v1/create_sampling_session",
            {
                "session_id": session_id,
                "sampling_session_seq_id": 0,
                "base_model": base_model,
            },
            timeout=CREATE_SAMPLING_TIMEOUT_S,
        )
        sampling_session_id = sampling.get("sampling_session_id")
        if not sampling_session_id:
            return _fail(f"create_sampling_session missing sampling_session_id: {sampling!r}")

        prompt_tokens = [1000, 1001, 1002, 1003]
        k = 5
        sample_payload = {
            "sampling_session_id": sampling_session_id,
            "seq_id": 0,
            "num_samples": 1,
            "prompt": {"chunks": [{"tokens": prompt_tokens, "type": "encoded_text"}]},
            "sampling_params": {
                "max_tokens": 1,
                "temperature": 0.0,
                "top_k": -1,
                "top_p": 1.0,
            },
            "include_prompt_logprobs": True,
            "topk_prompt_logprobs": k,
        }
        fut = _post_json(f"{BASE_URL}/api/v1/asample", sample_payload, timeout=30.0)
        request_id = fut.get("request_id")
        if not request_id:
            return _fail(f"asample missing request_id: {fut!r}")

        result = _poll_future(request_id, timeout_s=POLL_TIMEOUT_S)
        topk = result.get("topk_prompt_logprobs")
        try:
            _validate_topk(topk, prompt_len=len(prompt_tokens), k=k)
        except Exception as e:
            dump = json.dumps(result, indent=2)[:2000]
            return _fail(f"{e}. Response snippet: {dump}")

        print("PASS")
        return 0
    except Exception as e:
        return _fail(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
