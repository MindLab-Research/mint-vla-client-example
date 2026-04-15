#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from typing import Any

import requests


DEFAULT_BASE_URL = "http://localhost:8000"
HTTP_TIMEOUT_S = float(os.environ.get("TINKER_HTTP_TIMEOUT_S", "120"))
DEFAULT_PROMPT = "Reply with exactly: hi"


def _coalesce(*values: str | None) -> str | None:
    for value in values:
        if value:
            return value
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce issue #500: native /api/v1/create_sampling_session succeeds while "
            "/oai/api/v1/completions and /oai/api/v1/chat/completions must not fail with "
            "'Session manager not initialized'."
        )
    )
    parser.add_argument("--base-url", default=None, help="Server base URL without /oai/api/v1 suffix")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None, help="Model to test; defaults to an advertised supported model")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=8)
    return parser.parse_args()


def _base_url(args: argparse.Namespace) -> str:
    return (
        _coalesce(args.base_url, os.environ.get("TINKER_BASE_URL"), os.environ.get("MINT_BASE_URL"), DEFAULT_BASE_URL)
        or DEFAULT_BASE_URL
    ).rstrip("/")


def _headers(args: argparse.Namespace) -> dict[str, str]:
    api_key = _coalesce(args.api_key, os.environ.get("TINKER_API_KEY"), os.environ.get("MINT_API_KEY"))
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _json_or_text(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except Exception:
        payload = {"_non_json_body": response.text[:2000]}
    if isinstance(payload, dict):
        return payload
    return {"_non_dict_json": payload}


def _get_json(base_url: str, headers: dict[str, str], path: str) -> tuple[int, dict[str, Any]]:
    response = requests.get(f"{base_url}{path}", headers=headers, timeout=HTTP_TIMEOUT_S)
    return response.status_code, _json_or_text(response)


def _post_json(base_url: str, headers: dict[str, str], path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    response = requests.post(f"{base_url}{path}", headers=headers, json=payload, timeout=HTTP_TIMEOUT_S)
    return response.status_code, _json_or_text(response)


def _extract_supported_models(body: dict[str, Any]) -> list[str]:
    models: list[str] = []
    supported = body.get("supported_models")
    if isinstance(supported, list):
        for entry in supported:
            if isinstance(entry, str) and entry:
                models.append(entry)
            elif isinstance(entry, dict):
                name = entry.get("model_name")
                if isinstance(name, str) and name:
                    models.append(name)
    return models


def _select_model(args: argparse.Namespace, base_url: str, headers: dict[str, str]) -> str:
    explicit = _coalesce(args.model, os.environ.get("TINKER_MODEL"), os.environ.get("MINT_MODEL"))
    status, body = _get_json(base_url, headers, "/api/v1/get_server_capabilities")
    if status != 200:
        raise RuntimeError(f"get_server_capabilities returned {status}: {body!r}")
    models = _extract_supported_models(body)
    if explicit:
        if models and explicit not in models:
            raise RuntimeError(f"requested model {explicit!r} not in supported_models={models!r}")
        return explicit
    if not models:
        raise RuntimeError(f"supported_models missing or empty: {body!r}")
    return models[0]


def _create_sampling_session(base_url: str, headers: dict[str, str], model: str) -> str:
    status, body = _post_json(
        base_url,
        headers,
        "/api/v1/create_session",
        {
            "tags": ["scripts/tools/reproduce_issue_500.py", f"issue500-{uuid.uuid4().hex[:8]}"],
            "user_metadata": {},
            "sdk_version": "repro-500",
        },
    )
    if status != 200:
        raise RuntimeError(f"create_session returned {status}: {body!r}")
    session_id = body.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError(f"create_session missing session_id: {body!r}")

    status, body = _post_json(
        base_url,
        headers,
        "/api/v1/create_sampling_session",
        {
            "session_id": session_id,
            "sampling_session_seq_id": 0,
            "base_model": model,
        },
    )
    if status != 200:
        raise RuntimeError(f"create_sampling_session returned {status}: {body!r}")
    sampling_session_id = body.get("sampling_session_id")
    if not isinstance(sampling_session_id, str) or not sampling_session_id:
        raise RuntimeError(f"create_sampling_session missing sampling_session_id: {body!r}")
    return sampling_session_id


def _assert_oai_success(
    *,
    name: str,
    status: int,
    body: dict[str, Any],
) -> None:
    if status != 200:
        raise RuntimeError(f"{name} returned {status}: {body!r}")
    error = body.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if message == "Session manager not initialized":
            raise RuntimeError(f"{name} still fails with split-state error: {body!r}")


def main() -> int:
    args = _parse_args()
    base_url = _base_url(args)
    headers = _headers(args)

    print(f"base_url={base_url}", flush=True)
    status, body = _get_json(base_url, headers, "/api/v1/healthz")
    if status != 200:
        print(f"FAIL: healthz returned {status}: {body!r}", file=sys.stderr, flush=True)
        return 1
    print(f"healthz={json.dumps(body, ensure_ascii=False)}", flush=True)

    try:
        model = _select_model(args, base_url, headers)
        sampling_session_id = _create_sampling_session(base_url, headers, model)
        print(
            f"native create_sampling_session ok model={model} sampling_session_id={sampling_session_id}",
            flush=True,
        )

        completions_status, completions_body = _post_json(
            base_url,
            headers,
            "/oai/api/v1/completions",
            {
                "model": model,
                "prompt": args.prompt,
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
            },
        )
        print(
            f"oai completions status={completions_status} body={json.dumps(completions_body, ensure_ascii=False)}",
            flush=True,
        )
        _assert_oai_success(name="oai completions", status=completions_status, body=completions_body)

        chat_status, chat_body = _post_json(
            base_url,
            headers,
            "/oai/api/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": args.prompt}],
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
            },
        )
        print(f"oai chat status={chat_status} body={json.dumps(chat_body, ensure_ascii=False)}", flush=True)
        _assert_oai_success(name="oai chat completions", status=chat_status, body=chat_body)
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1

    print("PASS: native and OAI sampling paths all succeeded", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
