#!/usr/bin/env python3
import argparse
import json
import os
import sys
import urllib.error
import urllib.request


DEFAULT_FEISHU_WEBHOOK_URL = (
    "https://open.feishu.cn/open-apis/bot/v2/hook/"
    "7cec6e86-8a3a-4d8b-8a11-14b8bdbd5a3c"
)


def _build_payload(*, title: str, markdown: str) -> dict:
    return {
        "msg_type": "interactive",
        "card": {
            "schema": "2.0",
            "config": {"wide_screen_mode": True},
            "header": {"title": {"tag": "plain_text", "content": title}},
            "body": {"elements": [{"tag": "markdown", "content": markdown}]},
        },
    }


def _post_json(*, url: str, payload: dict) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read()
    try:
        return json.loads(body.decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Feishu webhook returned non-JSON: {body[:200]!r}") from e


def main() -> int:
    p = argparse.ArgumentParser(description="Post an interactive card to Feishu bot webhook.")
    p.add_argument(
        "--webhook-url",
        default=os.environ.get("FEISHU_WEBHOOK_URL", DEFAULT_FEISHU_WEBHOOK_URL),
        help="Feishu bot webhook URL (env: FEISHU_WEBHOOK_URL).",
    )
    p.add_argument("--title", required=True, help="Card title.")
    p.add_argument(
        "--markdown",
        required=True,
        help="Markdown content for the card body (Feishu subset).",
    )
    args = p.parse_args()

    payload = _build_payload(title=args.title, markdown=args.markdown)
    try:
        resp = _post_json(url=str(args.webhook_url), payload=payload)
    except urllib.error.HTTPError as e:
        sys.stderr.write(f"Feishu webhook HTTPError: {e.code} {e.reason}\n")
        try:
            sys.stderr.write(e.read().decode("utf-8", errors="replace") + "\n")
        except Exception:  # noqa: BLE001
            pass
        return 3
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"Feishu webhook error: {type(e).__name__}: {e}\n")
        return 3

    if not isinstance(resp, dict):
        sys.stderr.write("Feishu webhook returned non-dict JSON:\n")
        sys.stderr.write(json.dumps(resp, ensure_ascii=False) + "\n")
        return 3

    code = resp.get("StatusCode", resp.get("code"))
    if code is None:
        sys.stderr.write("Feishu webhook response missing status code:\n")
        sys.stderr.write(json.dumps(resp, ensure_ascii=False) + "\n")
        return 3
    if str(code) != "0":
        sys.stderr.write("Feishu webhook returned error:\n")
        sys.stderr.write(json.dumps(resp, ensure_ascii=False) + "\n")
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
