from __future__ import annotations

import math
import os
import sys
import time
import uuid
from typing import Any

import requests


BASE_URL = os.environ.get("MINT_BASE_URL")
if not BASE_URL:
    port = os.environ.get("MINT_PORT", "10200")
    BASE_URL = f"http://localhost:{port}"
BASE_URL = BASE_URL.rstrip("/")

API_KEY = os.environ.get("MINT_API_KEY", "dummy")

BASE_MODEL = os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")

# Must be a shared-filesystem path accessible to vLLM workers.
# Default points at a deliberately NaN-corrupted adapter to deterministically hit
# the non-finite sampled-token logprob path.
MODEL_PATH = os.environ.get(
    "MINT_MODEL_PATH",
    "/vePFS-Mindverse/share/mint_checkpoints/issue_200_nan_adapter",
).strip()

# Per official reference: -1 disables top-k.
TOP_K = int(os.environ.get("MINT_TOP_K", "-1"))
TOP_P = float(os.environ.get("MINT_TOP_P", "1.0"))
TEMPERATURE = float(os.environ.get("MINT_TEMPERATURE", "1.0"))
MAX_TOKENS = int(os.environ.get("MINT_MAX_TOKENS", "64"))
NUM_SAMPLES = int(os.environ.get("MINT_NUM_SAMPLES", "8"))
NUM_ATTEMPTS = int(os.environ.get("MINT_NUM_ATTEMPTS", "50"))

POLL_TIMEOUT_S = float(os.environ.get("MINT_POLL_TIMEOUT_S", "1800"))
POLL_SLEEP_S = float(os.environ.get("MINT_POLL_SLEEP_S", "2.0"))

CREATE_SESSION_TIMEOUT_S = float(os.environ.get("MINT_CREATE_SESSION_TIMEOUT_S", "30"))
CREATE_SAMPLING_TIMEOUT_S = float(os.environ.get("MINT_CREATE_SAMPLING_TIMEOUT_S", "120"))
ASAMPLE_TIMEOUT_S = float(os.environ.get("MINT_ASAMPLE_TIMEOUT_S", "60"))


def _headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


def _post_json(path: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    resp = requests.post(url, headers=_headers(), json=payload, timeout=timeout_s)
    if resp.status_code >= 400:
        raise RuntimeError(f"POST {path} -> {resp.status_code}: {resp.text[:400]!r}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"POST {path} -> non-dict json: {data!r}")
    return data


def _get_prompt_tokens() -> list[int]:
    raw = os.environ.get("MINT_PROMPT_TOKENS", "").strip()
    if not raw:
        # Default: a minimal Qwen chat-template prompt that reliably generates
        # `max_tokens` outputs under normal conditions.
        return [
            151644,
            872,
            198,
            2507,
            504,
            220,
            16,
            311,
            220,
            17,
            15,
            15,
            11,
            18663,
            553,
            12621,
            13,
            151645,
            198,
            151644,
            77091,
            198,
        ]
    toks: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        toks.append(int(part))
    return toks


def _create_session() -> str:
    out = _post_json(
        "/api/v1/create_session",
        {
            "tags": [],
            "user_metadata": {},
            "sdk_version": "reproduce_issue_200",
            "type": "create_session",
        },
        timeout_s=CREATE_SESSION_TIMEOUT_S,
    )
    session_id = out.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError(f"create_session missing/invalid session_id: {out!r}")
    return session_id


def _create_sampling_session(session_id: str) -> str:
    payload: dict[str, Any] = {
        "session_id": session_id,
        "sampling_session_seq_id": 0,
        "base_model": BASE_MODEL,
        "lora_rank": 32,
    }
    if MODEL_PATH:
        payload["model_path"] = MODEL_PATH
    out = _post_json(
        "/api/v1/create_sampling_session",
        payload,
        timeout_s=CREATE_SAMPLING_TIMEOUT_S,
    )
    sampling_session_id = out.get("sampling_session_id")
    if not isinstance(sampling_session_id, str) or not sampling_session_id:
        raise RuntimeError(f"create_sampling_session missing/invalid sampling_session_id: {out!r}")
    return sampling_session_id


def _asample_once(*, sampling_session_id: str, seq_id: int) -> str:
    req_id = f"{uuid.uuid4()}_{seq_id}"
    out = _post_json(
        "/api/v1/asample",
        {
            "sampling_session_id": sampling_session_id,
            "seq_id": seq_id,
            "num_samples": NUM_SAMPLES,
            "prompt": {"chunks": [{"tokens": _get_prompt_tokens(), "type": "encoded_text"}]},
            "sampling_params": {
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "top_k": TOP_K,
                "top_p": TOP_P,
                "stop": None,
                "seed": None,
            },
            "prompt_logprobs": False,
            "topk_prompt_logprobs": 0,
            "include_prompt_logprobs": False,
        },
        timeout_s=ASAMPLE_TIMEOUT_S,
    )
    request_id = out.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError(f"asample missing/invalid request_id: {out!r} req_id={req_id!r}")
    return request_id


def _retrieve_future(request_id: str) -> dict[str, Any] | None:
    url = f"{BASE_URL}/api/v1/retrieve_future"
    resp = requests.post(url, headers=_headers(), json={"request_id": request_id}, timeout=30.0)
    if resp.status_code == 408:
        return None
    if resp.status_code >= 400:
        raise RuntimeError(f"retrieve_future -> {resp.status_code}: {resp.text[:400]!r}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"retrieve_future -> non-dict json: {data!r}")
    return data


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    print(
        f"BASE_URL={BASE_URL} base_model={BASE_MODEL} num_samples={NUM_SAMPLES} "
        f"max_tokens={MAX_TOKENS} temperature={TEMPERATURE} top_k={TOP_K} top_p={TOP_P} "
        f"attempts={NUM_ATTEMPTS} model_path={MODEL_PATH!r}",
        flush=True,
    )

    session_id = _create_session()
    sampling_session_id = _create_sampling_session(session_id)

    for attempt in range(NUM_ATTEMPTS):
        request_id = _asample_once(sampling_session_id=sampling_session_id, seq_id=attempt)
        t0 = time.time()
        while True:
            if time.time() - t0 > POLL_TIMEOUT_S:
                return _fail(f"timeout waiting for retrieve_future: request_id={request_id}")
            out = _retrieve_future(request_id)
            if out is None:
                time.sleep(POLL_SLEEP_S)
                continue

            err = out.get("error")
            if err:
                # Expected post-fix: fail loudly instead of clamping to -1e9 and
                # silently returning corrupted logprobs.
                if isinstance(err, str) and "Non-finite sampled-token logprobs:" in err:
                    print("PASS")
                    return 0
                return _fail(f"asample returned error: {err!r}")

            sequences = out.get("sequences")
            if not isinstance(sequences, list) or not sequences:
                return _fail(f"retrieve_future missing/invalid sequences: {out!r}")

            for si, seq in enumerate(sequences):
                if not isinstance(seq, dict):
                    return _fail(f"sequence[{si}] non-dict: {seq!r}")
                toks = seq.get("tokens")
                lps = seq.get("logprobs")
                if not isinstance(toks, list) or not toks:
                    return _fail(f"sequence[{si}] missing/invalid tokens: {seq!r}")
                if not isinstance(lps, list) or len(lps) != len(toks):
                    return _fail(
                        f"sequence[{si}] missing/invalid logprobs: len(tokens)={len(toks)} logprobs={lps!r}"
                    )
                bad = [lp for lp in lps if not isinstance(lp, (float, int)) or not math.isfinite(float(lp))]
                if bad:
                    return _fail(
                        f"attempt={attempt} sequence[{si}] contains non-finite logprobs: {bad[:3]!r}"
                    )
                # Server's current clamp sentinel is -1e9. Treat as repro hit.
                clamped = [lp for lp in lps if float(lp) <= -1e8]
                if clamped:
                    return _fail(
                        "clamped sampled-token logprobs detected (expected fix is finite logprobs or explicit error): "
                        f"attempt={attempt} sequence[{si}] min_lp={min(float(x) for x in lps):.3g} "
                        f"sample={clamped[:3]!r}"
                    )

            break

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
