import datetime
import os
import re
import subprocess
import sys
import uuid
from typing import Any

import requests

BASE_URL = os.environ.get("MINT_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("MINT_API_KEY", "dummy")
SSH_HOST = os.environ.get("MINT_SSH_HOST", "volcano")


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _get_json(url: str, *, timeout_s: float = 60.0) -> dict[str, Any]:
    r = requests.get(url, headers=_headers(), timeout=timeout_s)
    if r.status_code != 200:
        raise RuntimeError(f"GET {url} returned {r.status_code}: {r.text[:300]!r}")
    out = r.json()
    if not isinstance(out, dict):
        raise RuntimeError(f"GET {url} returned non-dict json: {type(out)}")
    return out


def _infer_remote_log_dir() -> str:
    v = os.environ.get("MINT_USAGE_LOG_DIR")
    if v:
        return v

    try:
        info = _get_json(f"{BASE_URL}/api/v1/server_info", timeout_s=10.0)
        stdout_target = (
            info.get("logging", {})
            .get("stdout", {})
            .get("target")
        )
        if isinstance(stdout_target, str):
            m = re.search(r"/tmp/mint_server_issue_(\d+)\.log$", stdout_target)
            if m:
                return f"/tmp/mint_usage_issue_{m.group(1)}"
    except Exception:
        pass

    return "/tmp/mint_usage"


def _ssh_write_usage_logs(*, user_id: str, date_str: str, remote_log_dir: str) -> None:
    python_lines = [
        "import datetime",
        "import json",
        "import os",
        f"user_id = {user_id!r}",
        f"log_dir = {remote_log_dir!r}",
        f"date_str = {date_str!r}",
        "os.makedirs(log_dir, exist_ok=True)",
        "path = os.path.join(log_dir, f'usage_{date_str}.jsonl')",
        "now = datetime.datetime.now(datetime.timezone.utc).isoformat()",
        "entries = [",
        "  {'user_id': user_id, 'operation_type': 'sample_prefill', 'model_name': 'm', 'token_count': 10, 'session_id': 's', 'request_id': 'r1', 'timestamp': now},",
        "  {'user_id': user_id, 'operation_type': 'sample_prefill', 'model_name': 'm', 'token_count': 20, 'session_id': 's', 'request_id': 'r2', 'timestamp': now},",
        "  {'user_id': user_id, 'operation_type': 'sample_generation', 'model_name': 'm', 'token_count': 100, 'session_id': 's', 'request_id': 'r3', 'timestamp': now},",
        "]",
        "with open(path, 'a', encoding='utf-8') as f:",
        "  for e in entries:",
        "    f.write(json.dumps(e, ensure_ascii=False) + '\\n')",
        "print(path)",
    ]
    python_code = "\n".join(python_lines) + "\n"

    proc = subprocess.run(
        ["ssh", SSH_HOST, "python", "-"],
        input=python_code,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ssh {SSH_HOST} failed: rc={proc.returncode} stderr={proc.stderr.strip()!r}"
        )


def main() -> int:
    try:
        user_id = f"issue64_{uuid.uuid4().hex[:8]}"
        date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        _ssh_write_usage_logs(
            user_id=user_id,
            date_str=date_str,
            remote_log_dir=_infer_remote_log_dir(),
        )

        out = _get_json(f"{BASE_URL}/internal/usage_summary/{user_id}", timeout_s=30)
        total_tokens = out.get("total_tokens")
        op = out.get("operation_counts")

        if total_tokens != 130:
            return _fail(f"total_tokens={total_tokens!r} expected 130")
        if not isinstance(op, dict):
            return _fail(f"operation_counts={op!r} expected dict")
        if op.get("sample_prefill") != 30:
            return _fail(
                f"sample_prefill={op.get('sample_prefill')!r} expected 30 (token totals, not op counts)"
            )
        if op.get("sample_generation") != 100:
            return _fail(
                f"sample_generation={op.get('sample_generation')!r} expected 100 (token totals, not op counts)"
            )

        print("PASS")
        return 0
    except Exception as e:
        return _fail(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
