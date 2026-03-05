#!/usr/bin/env python3
import os
import sys
import time
import json
import requests


BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")
BASE_MODEL = os.environ.get("ISSUE182_BASE_MODEL", "Qwen/Qwen3-0.6B")
SCHEDULER_ENABLED = os.environ.get("MINT_SCHEDULER_ENABLE", "0").strip().lower() in ("1", "true", "yes", "y")


def _headers() -> dict[str, str]:
    if API_KEY:
        return {"X-API-Key": API_KEY}
    return {}


def _post(path: str, payload: dict, timeout_s: float = 60.0) -> requests.Response:
    return requests.post(
        f"{BASE_URL}{path}",
        headers=_headers(),
        json=payload,
        timeout=timeout_s,
    )


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", flush=True)
    return 1


def _ok(msg: str) -> None:
    print(f"OK: {msg}", flush=True)


def _create_sampling_session(session_id: str) -> str:
    r = _post(
        "/api/v1/create_sampling_session",
        {"session_id": session_id, "base_model": BASE_MODEL},
        timeout_s=90.0,
    )
    if r.status_code != 200:
        raise RuntimeError(f"create_sampling_session failed status={r.status_code} body={r.text[:500]!r}")
    data = r.json()
    sid = data.get("sampling_session_id")
    if not isinstance(sid, str) or not sid:
        raise RuntimeError(f"create_sampling_session returned invalid sampling_session_id: {data!r}")
    return sid


def _asample(sampling_session_id: str, seq_id: int, prompt_tokens: list[int], max_tokens: int) -> str:
    payload = {
        "sampling_session_id": sampling_session_id,
        "seq_id": int(seq_id),
        "num_samples": 1,
        "prompt": {"chunks": [{"tokens": prompt_tokens, "type": "encoded_text"}]},
        "sampling_params": {"max_tokens": int(max_tokens), "temperature": 0.7, "top_k": -1, "top_p": 1.0},
    }
    r = _post("/api/v1/asample", payload, timeout_s=60.0)
    if r.status_code != 200:
        raise RuntimeError(f"asample failed status={r.status_code} body={r.text[:500]!r}")
    data = r.json()
    rid = data.get("request_id")
    if not isinstance(rid, str) or not rid:
        raise RuntimeError(f"asample returned invalid request_id: {data!r}")
    return rid


def _check_pending_payload(body: dict, headers: dict[str, str]) -> None:
    if body.get("queue_state") != "active":
        raise RuntimeError(f"queue_state={body.get('queue_state')!r} expected 'active'")

    if body.get("type") != "try_again":
        raise RuntimeError(f"type={body.get('type')!r} expected 'try_again'")
    if not isinstance(body.get("request_id"), str) or not body.get("request_id"):
        raise RuntimeError(f"request_id={body.get('request_id')!r} expected non-empty str")

    status = body.get("status")
    if status not in ("queued", "prefill", "decode"):
        raise RuntimeError(f"status={status!r} expected queued|prefill|decode")

    if headers.get("X-Queue-Status") is None:
        raise RuntimeError("missing X-Queue-Status header")

    qs_reason = body.get("queue_state_reason")
    if qs_reason is not None and (not isinstance(qs_reason, str) or not qs_reason.strip()):
        raise RuntimeError(f"queue_state_reason={qs_reason!r} expected null or non-empty str")

    ra = headers.get("Retry-After")
    try:
        ra_i = int(ra)
    except Exception:
        raise RuntimeError(f"Retry-After={ra!r} expected int header")
    if ra_i < 1:
        raise RuntimeError(f"Retry-After={ra_i!r} expected >= 1")
    if body.get("retry_after_s") != ra_i:
        raise RuntimeError(f"retry_after_s={body.get('retry_after_s')!r} expected {ra_i}")

    qd = body.get("queue_depth")
    if SCHEDULER_ENABLED:
        if qd is not None:
            raise RuntimeError(f"queue_depth={qd!r} expected null when scheduler enabled")
        if headers.get("X-Queue-Depth") is not None:
            raise RuntimeError("unexpected X-Queue-Depth header when scheduler enabled")
    else:
        if qd is None:
            if body.get("queue_state_reason") not in ("queue_position_unknown", "scheduler_enabled"):
                raise RuntimeError(f"queue_depth={qd!r} expected int>=0 or null with queue_state_reason")
        elif not isinstance(qd, int) or qd < 0:
            raise RuntimeError(f"queue_depth={qd!r} expected int>=0")
        if qd is not None and headers.get("X-Queue-Depth") is None:
            raise RuntimeError("missing X-Queue-Depth header when queue_depth is set")

    if status == "queued":
        qp = body.get("queue_position")
        if SCHEDULER_ENABLED:
            if qp is not None:
                raise RuntimeError(f"queue_position={qp!r} expected null when scheduler enabled")
        else:
            if qp is not None:
                if not isinstance(qp, int) or qp < 0:
                    raise RuntimeError(f"queue_position={qp!r} expected int>=0 or null")
                if isinstance(qd, int) and qd < 1:
                    raise RuntimeError(f"queue_depth={qd!r} expected >=1 when queue_position is set")
        if isinstance(qd, int) and qd > 0:
            if body.get("queue_state_reason") != "queue_backlog":
                raise RuntimeError(
                    f"queue_state_reason={body.get('queue_state_reason')!r} expected 'queue_backlog' when queue_depth>0"
                )
        if SCHEDULER_ENABLED and body.get("queue_state_reason") != "scheduler_enabled":
            raise RuntimeError(
                f"queue_state_reason={body.get('queue_state_reason')!r} expected 'scheduler_enabled' when scheduler enabled"
            )

    prog = body.get("progress")
    if isinstance(prog, dict):
        tg = prog.get("tokens_generated")
        mx = prog.get("max_tokens")
        if not isinstance(tg, int) or not isinstance(mx, int) or tg < 0 or mx < 1 or tg > mx:
            raise RuntimeError(f"progress={prog!r} invalid")
        if headers.get("X-Queue-Tokens-Generated") is None or headers.get("X-Queue-Max-Tokens") is None:
            raise RuntimeError("missing progress headers when progress payload is present")
    else:
        if headers.get("X-Queue-Tokens-Generated") is not None or headers.get("X-Queue-Max-Tokens") is not None:
            raise RuntimeError("unexpected progress headers when progress payload is missing")

    eta = body.get("estimated_wait_s")
    if eta is None:
        if headers.get("X-Queue-ETA-S") is not None:
            raise RuntimeError("unexpected X-Queue-ETA-S header when estimated_wait_s is null")
    else:
        if not isinstance(eta, (int, float)) or eta < 0:
            raise RuntimeError(f"estimated_wait_s={eta!r} expected non-negative number")
        if headers.get("X-Queue-ETA-S") is None:
            raise RuntimeError("missing X-Queue-ETA-S header when estimated_wait_s is set")

def _format_pending_line(body: dict, headers: dict[str, str]) -> str:
    return (
        "pending_fields "
        f"request_id={body.get('request_id')!r} "
        f"type={body.get('type')!r} "
        f"status={body.get('status')!r} "
        f"queue_state={body.get('queue_state')!r} "
        f"queue_state_reason={body.get('queue_state_reason')!r} "
        f"queue_depth={body.get('queue_depth')!r} "
        f"queue_position={body.get('queue_position')!r} "
        f"estimated_wait_s={body.get('estimated_wait_s')!r} "
        f"progress={body.get('progress')!r} "
        f"retry_after_s={body.get('retry_after_s')!r} "
        f"x_queue_depth={headers.get('X-Queue-Depth')!r} "
        f"x_queue_status={headers.get('X-Queue-Status')!r} "
        f"x_queue_eta_s={headers.get('X-Queue-ETA-S')!r} "
        f"retry_after={headers.get('Retry-After')!r}"
    )


def main() -> int:
    session_id = f"issue182_{int(time.time())}"
    _ok(f"base_url={BASE_URL} base_model={BASE_MODEL} session_id={session_id}")
    sampling_session_id = _create_sampling_session(session_id)
    _ok(f"sampling_session_id={sampling_session_id}")

    prompt_tokens = [1] * 512
    max_tokens = 512

    request_ids: list[str] = []
    for i in range(3):
        rid = _asample(sampling_session_id, seq_id=i, prompt_tokens=prompt_tokens, max_tokens=max_tokens)
        request_ids.append(rid)
    _ok(f"submitted {len(request_ids)} requests")

    saw_pending = False
    deadline = time.time() + 300
    pending_checked: set[str] = set()

    while request_ids and time.time() < deadline:
        sleep_s = 1
        for rid in list(request_ids):
            r = _post(
                "/api/v1/retrieve_future",
                {"request_id": rid, "model_id": sampling_session_id},
                timeout_s=30.0,
            )
            if r.status_code == 408:
                saw_pending = True
                try:
                    ra_i = int(r.headers.get("Retry-After", "1"))
                except Exception:
                    ra_i = 1
                if ra_i < 1:
                    ra_i = 1
                sleep_s = max(sleep_s, ra_i)
                if rid not in pending_checked:
                    body = r.json()
                    if body.get("request_id") != rid:
                        return _fail(
                            f"request_id mismatch: body={body.get('request_id')!r} expected {rid!r}"
                        )
                    _check_pending_payload(body, r.headers)
                    _ok(_format_pending_line(body, r.headers))
                    pending_checked.add(rid)
                continue
            if r.status_code != 200:
                return _fail(f"retrieve_future status={r.status_code} body={r.text[:500]!r}")
            out = r.json()
            if isinstance(out, dict) and "error" in out:
                return _fail(f"retrieve_future error={out.get('error')!r}")
            request_ids.remove(rid)
        time.sleep(sleep_s)

    if request_ids:
        return _fail(f"timeout waiting for requests: {request_ids!r}")
    if not saw_pending:
        return _fail("no pending 408 observed; cannot validate queue status fields")
    _ok("issue-182 reproduction passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
