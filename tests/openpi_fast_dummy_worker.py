from __future__ import annotations

import json
import sys


def _reply(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def main() -> None:
    _reply({"event": "ready", "protocol_version": 1})

    for line in sys.stdin:
        request = json.loads(line)
        request_id = request["id"]
        op = request["op"]
        payload = request.get("payload", {})

        if op == "echo":
            _reply({"id": request_id, "ok": True, "payload": payload})
            continue

        if op == "fail":
            _reply(
                {
                    "id": request_id,
                    "ok": False,
                    "error": {"type": "RuntimeError", "message": payload.get("message", "failed")},
                }
            )
            continue

        if op == "mismatch":
            _reply({"id": f"{request_id}-wrong", "ok": True, "payload": payload})
            continue

        if op == "malformed":
            sys.stdout.write("not-json\n")
            sys.stdout.flush()
            continue

        if op == "shutdown":
            _reply({"id": request_id, "ok": True, "payload": {"stopped": True}})
            break

        _reply({"id": request_id, "ok": True, "payload": {"op": op, "payload": payload}})


if __name__ == "__main__":
    main()
