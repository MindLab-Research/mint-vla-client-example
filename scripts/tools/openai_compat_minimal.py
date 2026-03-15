#!/usr/bin/env python3
"""OpenAI-compatible SDK examples and smoke tests for tinker-server.

Examples:
  TINKER_BASE_URL=http://127.0.0.1:8000 \
    python scripts/tools/openai_compat_minimal.py completions \
      --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
      --prompt "The capital of France is"

  TINKER_BASE_URL=http://127.0.0.1:8000 \
    python scripts/tools/openai_compat_minimal.py chat \
      --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
      --user-message "What is 2+2?"

  TINKER_BASE_URL=http://127.0.0.1:8000 \
    python scripts/tools/openai_compat_minimal.py tool \
      --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
      --user-message "北京天气如何"

  TINKER_BASE_URL=http://127.0.0.1:8000 \
    python scripts/tools/openai_compat_minimal.py smoke \
      --model Qwen/Qwen3-30B-A3B-Instruct-2507
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Optional

from openai import AsyncOpenAI, OpenAI


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = os.environ.get("TINKER_OAI_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: Any


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
    return _coalesce(args.api_key, os.environ.get("TINKER_API_KEY"), os.environ.get("MINT_API_KEY"), "dummy") or "dummy"


def _mask_secret(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "..." + value[-4:]


def _tool_spec(name: str, description: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City or area to get weather for",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]


def _print_io(*, base_url: str, api_key: str, endpoint: str, payload: dict, response: Any) -> None:
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


def _record(results: list[CheckResult], name: str, ok: bool, detail: Any) -> None:
    results.append(CheckResult(name=name, ok=ok, detail=detail))


def _parse_args() -> argparse.Namespace:
    examples = """Examples:
  TINKER_BASE_URL=http://127.0.0.1:8000 \\
    python scripts/tools/openai_compat_minimal.py completions \\
      --model Qwen/Qwen3-30B-A3B-Instruct-2507 \\
      --prompt "The capital of France is"

  TINKER_BASE_URL=http://127.0.0.1:8000 \\
    python scripts/tools/openai_compat_minimal.py chat \\
      --model Qwen/Qwen3-30B-A3B-Instruct-2507 \\
      --user-message "What is 2+2?"

  TINKER_BASE_URL=http://127.0.0.1:8000 \\
    python scripts/tools/openai_compat_minimal.py tool \\
      --model Qwen/Qwen3-30B-A3B-Instruct-2507 \\
      --user-message "北京天气如何"

  TINKER_BASE_URL=http://127.0.0.1:8000 \\
    python scripts/tools/openai_compat_minimal.py smoke \\
      --model Qwen/Qwen3-30B-A3B-Instruct-2507

Environment:
  TINKER_BASE_URL / MINT_BASE_URL  Base server URL without /oai/api/v1 suffix
  TINKER_API_KEY / MINT_API_KEY    API key
  TINKER_OAI_MODEL                 Default model name for --model
"""
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, epilog=examples)
    p.add_argument("--base-url", default=None, help="Server base URL without /oai/api/v1 suffix")
    p.add_argument("--api-key", default=None)

    sub = p.add_subparsers(dest="cmd", required=True)

    cmpl = sub.add_parser("completions", help="Legacy /completions example")
    cmpl.add_argument("--model", default=DEFAULT_MODEL)
    cmpl.add_argument("--prompt", required=True)
    cmpl.add_argument("--max-tokens", type=int, default=32)
    cmpl.add_argument("--temperature", type=float, default=0.2)
    cmpl.add_argument("--top-p", type=float, default=0.9)
    cmpl.add_argument("--stop", default=None, help="Optional stop string")

    chat = sub.add_parser("chat", help="/chat/completions example")
    chat.add_argument("--model", default=DEFAULT_MODEL)
    chat.add_argument("--system", "--system-message", dest="system", default=None)
    chat.add_argument("--user", "--user-message", dest="user_message", required=True)
    chat.add_argument("--max-tokens", type=int, default=32)
    chat.add_argument("--temperature", type=float, default=0.2)
    chat.add_argument("--top-p", type=float, default=0.9)
    chat.add_argument("--stop", default=None, help="Optional stop string")

    tool = sub.add_parser("tool", help="Single tool-calling example")
    tool.add_argument("--model", default=DEFAULT_MODEL)
    tool.add_argument("--system", "--system-message", dest="system", default=None)
    tool.add_argument("--user", "--user-message", dest="user_message", default="北京天气如何")
    tool.add_argument("--tool-name", default="get_weather")
    tool.add_argument("--tool-description", default="Get current weather for a city")
    tool.add_argument("--tool-choice", default="auto", help='OpenAI tool_choice value, e.g. "auto" or "required"')
    tool.add_argument("--max-tokens", type=int, default=128)
    tool.add_argument("--temperature", type=float, default=0.1)
    tool.add_argument("--top-p", type=float, default=0.9)
    tool.add_argument(
        "--allow-no-tool-call",
        action="store_true",
        help="Exit successfully even if the model returns plain text instead of tool_calls",
    )

    smoke = sub.add_parser("smoke", help="More complete real-user SDK smoke test")
    smoke.add_argument("--model", default=DEFAULT_MODEL)
    smoke.add_argument("--tool-name", default="get_weather")
    smoke.add_argument("--tool-description", default="Get current weather for a city")
    smoke.add_argument("--tool-location", default="北京")
    smoke.add_argument("--tool-result", default='{"location":"北京","weather":"晴，18摄氏度"}')
    smoke.add_argument("--max-tokens", type=int, default=128)
    smoke.add_argument("--async-concurrency", type=int, default=3)

    args = p.parse_args()
    if not getattr(args, "model", None):
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
    _print_io(
        base_url=args.oai_base_url,
        api_key=args.resolved_api_key,
        endpoint="/chat/completions",
        payload=payload,
        response=response,
    )


def _run_tool_call(client: OpenAI, args: argparse.Namespace) -> None:
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.user_message})

    payload = {
        "model": args.model,
        "messages": messages,
        "tools": _tool_spec(args.tool_name, args.tool_description),
        "tool_choice": args.tool_choice,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
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

    _print_io(
        base_url=args.oai_base_url,
        api_key=args.resolved_api_key,
        endpoint="/chat/completions",
        payload=payload,
        response=response,
    )

    tool_calls = response.choices[0].message.tool_calls or []
    if tool_calls:
        print("=== tool_call_summary ===")
        for i, tool_call in enumerate(tool_calls, 1):
            print(
                json.dumps(
                    {
                        "index": i,
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return

    if not args.allow_no_tool_call:
        raise SystemExit("No tool_calls returned by the model")


async def _run_async_smoke(client: AsyncOpenAI, args: argparse.Namespace) -> list[CheckResult]:
    results: list[CheckResult] = []

    try:
        resp = await client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Reply with exactly: async-pong"}],
            max_tokens=min(args.max_tokens, 16),
            temperature=0.0,
        )
        content = resp.choices[0].message.content if resp.choices else None
        _record(
            results,
            "async.chat.completions.create",
            isinstance(content, str) and len(content.strip()) > 0,
            {"content": content},
        )
    except Exception as exc:
        _record(results, "async.chat.completions.create", False, f"{type(exc).__name__}: {exc}")

    async def _one_request(idx: int) -> dict[str, Any]:
        resp = await client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": f"Reply with exactly: batch-{idx}"}],
            max_tokens=min(args.max_tokens, 16),
            temperature=0.0,
        )
        return {"index": idx, "content": resp.choices[0].message.content if resp.choices else None}

    try:
        outputs = await asyncio.gather(*[_one_request(i) for i in range(args.async_concurrency)])
        ok = all(isinstance(item["content"], str) and len(item["content"].strip()) > 0 for item in outputs)
        _record(results, "async.chat.completions.concurrent", ok, outputs)
    except Exception as exc:
        _record(results, "async.chat.completions.concurrent", False, f"{type(exc).__name__}: {exc}")

    return results


def _run_smoke(client: OpenAI, args: argparse.Namespace) -> None:
    results: list[CheckResult] = []
    tools = _tool_spec(args.tool_name, args.tool_description)

    try:
        resp = client.models.list()
        ids = [m.id for m in resp.data]
        _record(results, "models.list", args.model in ids, {"count": len(ids), "models": ids})
    except Exception as exc:
        _record(results, "models.list", False, f"{type(exc).__name__}: {exc}")

    try:
        resp = client.models.retrieve(args.model)
        _record(results, "models.retrieve", getattr(resp, "id", None) == args.model, resp.model_dump())
    except Exception as exc:
        _record(results, "models.retrieve", False, f"{type(exc).__name__}: {exc}")

    try:
        resp = client.completions.create(
            model=args.model,
            prompt="The capital of France is",
            max_tokens=min(args.max_tokens, 16),
            temperature=0.1,
        )
        text = resp.choices[0].text if resp.choices else None
        _record(results, "completions.create", isinstance(text, str) and len(text.strip()) > 0, {"text": text})
    except Exception as exc:
        _record(results, "completions.create", False, f"{type(exc).__name__}: {exc}")

    try:
        resp = client.completions.create(
            model=args.model,
            prompt="Count: 1, 2, 3,",
            max_tokens=min(args.max_tokens, 16),
            temperature=0.0,
            stop=[" 5", "5"],
        )
        text = resp.choices[0].text if resp.choices else None
        _record(results, "completions.create.stop", isinstance(text, str), {"text": text})
    except Exception as exc:
        _record(results, "completions.create.stop", False, f"{type(exc).__name__}: {exc}")

    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Reply with exactly: pong"}],
            max_tokens=min(args.max_tokens, 16),
            temperature=0.0,
        )
        content = resp.choices[0].message.content if resp.choices else None
        _record(results, "chat.completions.create", isinstance(content, str) and len(content.strip()) > 0, {"content": content})
    except Exception as exc:
        _record(results, "chat.completions.create", False, f"{type(exc).__name__}: {exc}")

    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": f"{args.tool_location}天气如何？如果需要，请调用工具。"}],
            tools=tools,
            tool_choice="auto",
            max_tokens=args.max_tokens,
            temperature=0.1,
        )
        tool_calls = resp.choices[0].message.tool_calls or []
        _record(
            results,
            "chat.completions.tool_call_auto",
            len(tool_calls) >= 1,
            {
                "tool_calls": [
                    {
                        "id": call.id,
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    }
                    for call in tool_calls
                ],
                "content": resp.choices[0].message.content,
            },
        )
    except Exception as exc:
        _record(results, "chat.completions.tool_call_auto", False, f"{type(exc).__name__}: {exc}")

    try:
        first = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": f"{args.tool_location}天气如何？请调用工具后再回答。"}],
            tools=tools,
            tool_choice="required",
            max_tokens=args.max_tokens,
            temperature=0.1,
        )
        tool_calls = first.choices[0].message.tool_calls or []
        if not tool_calls:
            raise RuntimeError("no tool_calls returned")
        call = tool_calls[0]
        second = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "user", "content": f"{args.tool_location}天气如何？请调用工具后再回答。"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": args.tool_result,
                },
            ],
            tools=tools,
            max_tokens=args.max_tokens,
            temperature=0.1,
        )
        final_content = second.choices[0].message.content if second.choices else None
        _record(results, "chat.completions.tool_roundtrip", isinstance(final_content, str) and len(final_content.strip()) > 0, {"final": final_content})
    except Exception as exc:
        _record(results, "chat.completions.tool_roundtrip", False, f"{type(exc).__name__}: {exc}")

    try:
        client.chat.completions.create(model=args.model, messages=[{"role": "user", "content": "hi"}], stream=True)
        _record(results, "chat.completions.stream_true", False, "unexpected success")
    except Exception as exc:
        message = str(exc).lower()
        _record(results, "chat.completions.stream_true", "stream=true is not supported" in message, f"{type(exc).__name__}: {exc}")

    try:
        client.chat.completions.create(model=args.model, messages=[{"role": "user", "content": "hi"}], n=2)
        _record(results, "chat.completions.n_gt_1", False, "unexpected success")
    except Exception as exc:
        message = str(exc).lower()
        _record(results, "chat.completions.n_gt_1", "only n=1 is supported" in message, f"{type(exc).__name__}: {exc}")

    try:
        client.responses.create(model=args.model, input="hi")
        _record(results, "responses.create", False, "unexpected success")
    except Exception as exc:
        message = str(exc).lower()
        _record(results, "responses.create", ("404" in message) or ("not found" in message) or ("unsupported" in message), f"{type(exc).__name__}: {exc}")

    try:
        client.embeddings.create(model=args.model, input="hello")
        _record(results, "embeddings.create", False, "unexpected success")
    except Exception as exc:
        message = str(exc).lower()
        _record(results, "embeddings.create", ("404" in message) or ("not found" in message) or ("unsupported" in message), f"{type(exc).__name__}: {exc}")

    async_results = asyncio.run(_run_async_smoke(AsyncOpenAI(base_url=args.oai_base_url, api_key=args.resolved_api_key), args))
    results.extend(async_results)

    print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
    if not all(result.ok for result in results):
        raise SystemExit(1)


def main() -> int:
    args = _parse_args()
    client = OpenAI(base_url=args.oai_base_url, api_key=args.resolved_api_key)

    if args.cmd == "completions":
        _run_completions(client, args)
    elif args.cmd == "chat":
        _run_chat(client, args)
    elif args.cmd == "tool":
        _run_tool_call(client, args)
    else:
        _run_smoke(client, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
