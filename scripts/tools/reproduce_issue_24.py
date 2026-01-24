import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    from tinker_server.futures_utils import pending_future_http_response  # noqa: E402

    pending = pending_future_http_response()
    if pending.status_code != 408:
        print(f"FAIL: status_code={pending.status_code!r} expected 408", file=sys.stderr)
        return 1
    if pending.headers.get("Retry-After") != "1":
        print(f"FAIL: headers={pending.headers!r} expected Retry-After=1", file=sys.stderr)
        return 1
    if pending.body != {"queue_state": "active"}:
        print(f"FAIL: body={pending.body!r} expected {{'queue_state': 'active'}}", file=sys.stderr)
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

