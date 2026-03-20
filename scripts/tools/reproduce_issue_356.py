"""Reproduction script for issue #356.

Training model_ids are only unregistered on explicit DELETE or shutdown.
This script verifies that idle training sessions are automatically cleaned up.

Requirements:
  - Server must be started with a short training inactivity timeout, e.g.:
      MINT_TRAINING_INACTIVITY_TIMEOUT=60
  - SSH tunnel to dev server active (localhost:8000 -> mint-dev:8000)

Usage:
  TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy \
    python scripts/tools/reproduce_issue_356.py

Expected:
  - Before fix: FAIL (session persists indefinitely)
  - After fix: PASS (session auto-cleaned after timeout)
"""

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

MODEL = os.environ.get("TINKER_MODEL", "Qwen/Qwen3-0.6B")

# Server must be started with MINT_TRAINING_INACTIVITY_TIMEOUT set to this value.
INACTIVITY_TIMEOUT_S = int(os.environ.get("MINT_TRAINING_INACTIVITY_TIMEOUT", "60"))


def _headers() -> dict[str, str]:
    if not API_KEY:
        return {}
    return {"X-API-Key": API_KEY}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _post_json(path: str, body: dict[str, Any], *, timeout_s: float = 60.0) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    r = requests.post(url, headers=_headers(), json=body, timeout=timeout_s)
    if r.status_code != 200:
        raise RuntimeError(f"POST {path} -> {r.status_code}: {r.text[:500]!r}")
    return r.json()


def _get_json(path: str, *, timeout_s: float = 30.0) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    r = requests.get(url, headers=_headers(), timeout=timeout_s)
    if r.status_code != 200:
        raise RuntimeError(f"GET {path} -> {r.status_code}: {r.text[:500]!r}")
    return r.json()


def _session_exists(model_id: str) -> bool:
    """Check if a training session exists via /training_runs endpoint."""
    try:
        resp = _get_json("/api/v1/training_runs")
        runs = resp.get("training_runs", [])
        return any(r.get("training_run_id") == model_id for r in runs)
    except Exception:
        return False


def main() -> int:
    print(f"Base URL: {BASE_URL}")
    print(f"Model: {MODEL}")
    print(f"Expected inactivity timeout: {INACTIVITY_TIMEOUT_S}s")
    print()

    # Step 1: Health check
    print("[1/5] Health check...")
    try:
        r = requests.get(f"{BASE_URL}/api/v1/healthz", headers=_headers(), timeout=10)
        if r.status_code not in (200, 503):
            return _fail(f"Health check returned {r.status_code}")
    except requests.ConnectionError:
        return _fail(f"Cannot connect to {BASE_URL}")
    print("  OK")

    # Step 2: Create a training session (without ever calling DELETE)
    session_id = f"repro356_{uuid.uuid4().hex[:8]}"
    model_seq_id = 0
    model_id = f"{session_id}_{model_seq_id}"

    print(f"[2/5] Creating session: {session_id} (model_id={model_id})...")
    try:
        # Create parent session
        _post_json("/api/v1/create_session", {
            "type": "create_session",
            "tags": ["repro-356"],
        })
    except Exception as e:
        return _fail(f"create_session failed: {e}")

    try:
        resp = _post_json("/api/v1/create_model", {
            "type": "create_model",
            "session_id": session_id,
            "model_seq_id": model_seq_id,
            "base_model": MODEL,
            "lora_config": {"rank": 16},
        })
        request_id = resp.get("request_id")
        returned_model_id = resp.get("model_id")
        print(f"  create_model submitted: request_id={request_id}, model_id={returned_model_id}")
        if returned_model_id:
            model_id = returned_model_id
    except Exception as e:
        return _fail(f"create_model failed: {e}")

    # The session is registered in TrainingSessionManager as soon as
    # create_model returns (before the async actor init completes).
    # We do NOT need to poll retrieve_future; the cleanup test only
    # cares about whether the in-memory session is auto-evicted.

    # Step 3: Verify session exists
    print(f"[3/5] Verifying session {model_id} exists...")
    if not _session_exists(model_id):
        return _fail(f"Session {model_id} not found after creation")
    print("  Session exists (as expected)")

    # Step 4: Wait for inactivity timeout + cleanup cycle (60s scan interval)
    # Do NOT call any training API or DELETE - simulate client disconnect
    wait_time = INACTIVITY_TIMEOUT_S + 90  # timeout + ~1.5 cleanup cycles
    print(f"[4/5] Waiting {wait_time}s for idle cleanup (timeout={INACTIVITY_TIMEOUT_S}s + margin)...")
    print(f"  NOT calling DELETE - simulating client disconnect")

    poll_interval = 15
    elapsed = 0
    while elapsed < wait_time:
        time.sleep(poll_interval)
        elapsed += poll_interval
        still_exists = _session_exists(model_id)
        print(f"  {elapsed}s elapsed: session {'EXISTS' if still_exists else 'CLEANED UP'}")
        if not still_exists:
            break

    # Step 5: Verify session was auto-cleaned
    print(f"[5/5] Checking if session {model_id} was auto-cleaned...")
    if _session_exists(model_id):
        # Try explicit cleanup before failing
        print(f"  Session still exists after {wait_time}s - attempting explicit DELETE for cleanup...")
        try:
            requests.delete(
                f"{BASE_URL}/api/v1/models/{model_id}",
                headers=_headers(),
                timeout=30,
            )
        except Exception:
            pass
        return _fail(
            f"Session {model_id} still exists after {wait_time}s of inactivity. "
            f"Expected auto-cleanup after {INACTIVITY_TIMEOUT_S}s."
        )

    print(f"  Session {model_id} was auto-cleaned!")
    print()
    print("PASS: Training session idle cleanup is working correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
