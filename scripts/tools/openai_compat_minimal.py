#!/usr/bin/env python3
"""Minimal OpenAI-compatible example for tinker-server.

Usage:
  TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=your-real-api-key \
    python scripts/tools/openai_compat_minimal.py completions \
      --model tinker:///vePFS-Mindverse/share/tinker_runtime_checkpoints/persistent_cache/anonymous/stress-qwen3-30b_0/sampler \
      --prompt "The capital of France is"

  TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=your-real-api-key \
    python scripts/tools/openai_compat_minimal.py chat \
      --model tinker:///vePFS-Mindverse/share/tinker_runtime_checkpoints/persistent_cache/anonymous/stress-qwen3-30b_0/sampler \
      --user-message "What is 2+2?"
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from openai import OpenAI


DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL = os.environ.get("TINKER_OAI_MODEL")


def _coalesce(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if value:
            return value
    return None


def _oai_base_url(args: argparse.Namespace) -> str:
    base_url = _coalesce(args.base_url, os.environ.get("TINKER_BASE_URL"), os.environ.get("MINT_BASE_URL"))
    if not base_url:
        base_url = DEFAULT_BASE_URL
    return base_url.rstrip("/") + "/oai/api/v1"


def _api_key(args: argparse.Namespace) -> str:
    return _coalesce(args.api_key, os.environ.get("TINKER_API_KEY"), os.environ.get("MINT_API_KEY"), "dummy")


def _mask_secret(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "..." + value[-4:]


def _print_io(*, base_url: str, api_key: str, endpoint: str, payload: dict, response) -> None:
    print("=== input ===")
    print(
        json.dumps(
            {
                "base_url": base_url,
                "api_key": _mask_secret(api_key),
                "endpoint": endpoint,
                "payload": payload,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("=== output ===")
    print(response.model_dump_json(indent=2))


def _print_error(*, base_url: str, api_key: str, endpoint: str, payload: dict, exc: Exception) -> None:
    print("=== input ===")
    print(
        json.dumps(
            {
                "base_url": base_url,
                "api_key": _mask_secret(api_key),
                "endpoint": endpoint,
                "payload": payload,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("=== error ===")
    print(f"{type(exc).__name__}: {exc}")


def _parse_args() -> argparse.Namespace:
    examples = """Examples:
  TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=your-real-api-key \\
    python scripts/tools/openai_compat_minimal.py completions \\
      --model tinker:///vePFS-Mindverse/share/tinker_runtime_checkpoints/persistent_cache/anonymous/stress-qwen3-30b_0/sampler \\
      --prompt "The capital of France is"

  TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=your-real-api-key \\
    python scripts/tools/openai_compat_minimal.py chat \\
      --model tinker:///vePFS-Mindverse/share/tinker_runtime_checkpoints/persistent_cache/anonymous/stress-qwen3-30b_0/sampler \\
      --user-message "What is 2+2?"

Environment:
  TINKER_BASE_URL / MINT_BASE_URL  Base server URL without /oai/api/v1 suffix
  TINKER_API_KEY / MINT_API_KEY    API key
  TINKER_OAI_MODEL                 Default model URI for --model
"""
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=examples,
    )
    p.add_argument("--base-url", default=None, help="Server base URL without /oai/api/v1 suffix")
    p.add_argument("--api-key", default=None)

    sub = p.add_subparsers(dest="cmd", required=True)

    cmpl = sub.add_parser("completions", help="Legacy /completions example")
    cmpl.add_argument("--model", default=DEFAULT_MODEL, help="Sampler checkpoint URI")
    cmpl.add_argument("--prompt", required=True)
    cmpl.add_argument("--max-tokens", type=int, default=32)
    cmpl.add_argument("--temperature", type=float, default=0.2)
    cmpl.add_argument("--top-p", type=float, default=0.9)
    cmpl.add_argument("--stop", default=None, help="Optional stop string")

    chat = sub.add_parser("chat", help="/chat/completions example")
    chat.add_argument("--model", default=DEFAULT_MODEL, help="Sampler checkpoint URI")
    chat.add_argument("--system", "--system-message", dest="system", default=None)
    chat.add_argument("--user", "--user-message", dest="user_message", required=True)
    chat.add_argument("--max-tokens", type=int, default=32)
    chat.add_argument("--temperature", type=float, default=0.2)
    chat.add_argument("--top-p", type=float, default=0.9)
    chat.add_argument("--stop", default=None, help="Optional stop string")

    args = p.parse_args()
    if not args.model:
        p.error("--model is required (or set TINKER_OAI_MODEL)")
    args.oai_base_url = _oai_base_url(args)
    args.resolved_api_key = _api_key(args)
    return args


def _run_completions(client: OpenAI, args: argparse.Namespace) -> None:
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.stop is not None:
        payload["stop"] = args.stop
    try:
        response = client.completions.create(**payload)
    except Exception as exc:
        _print_error(
            base_url=args.oai_base_url,
            api_key=args.resolved_api_key,
            endpoint="/completions",
            payload=payload,
            exc=exc,
        )
        raise
    else:
        _print_io(
            base_url=args.oai_base_url,
            api_key=args.resolved_api_key,
            endpoint="/completions",
            payload=payload,
            response=response,
        )


def _run_chat(client: OpenAI, args: argparse.Namespace) -> None:
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.user_message})

    payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.stop is not None:
        payload["stop"] = args.stop
    try:
        response = client.chat.completions.create(**payload)
    except Exception as exc:
        _print_error(
            base_url=args.oai_base_url,
            api_key=args.resolved_api_key,
            endpoint="/chat/completions",
            payload=payload,
            exc=exc,
        )
        raise
    else:
        _print_io(
            base_url=args.oai_base_url,
            api_key=args.resolved_api_key,
            endpoint="/chat/completions",
            payload=payload,
            response=response,
        )

def main() -> int:
    args = _parse_args()
    client = OpenAI(base_url=args.oai_base_url, api_key=args.resolved_api_key)

    if args.cmd == "completions":
        _run_completions(client, args)
    else:
        _run_chat(client, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
