#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import sys


async def _health_check(base_url: str, api_key: str) -> str | None:
    import mint  # noqa: F401
    from tinker._client import AsyncTinker

    async with AsyncTinker(api_key=api_key, base_url=base_url) as client:
        health = await client.service.health_check()
    return getattr(health, "status", None)

def _get_capabilities(base_url: str, api_key: str) -> list[str]:
    import mint

    service_client = mint.ServiceClient(api_key=api_key, base_url=base_url)
    caps = service_client.get_server_capabilities()
    models: list[str] = []
    for m in getattr(caps, "supported_models", []) or []:
        name = getattr(m, "model_name", None)
        if name:
            models.append(name)
    return models


def _format_models(models: list[str], limit: int = 20) -> str:
    if not models:
        return "supported_models_count=0"
    shown = models[:limit]
    suffix = ",..." if len(models) > len(shown) else ""
    return f"supported_models_count={len(models)} supported_models={','.join(shown)}{suffix}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-url",
        default=None,
        help="Defaults to $MINT_BASE_URL, then $TINKER_BASE_URL, then https://mint.macaron.im",
    )
    args = ap.parse_args()

    base_url = args.base_url or os.environ.get("MINT_BASE_URL") or os.environ.get("TINKER_BASE_URL")
    if not base_url:
        base_url = "https://mint.macaron.im"

    api_key = os.environ.get("MINT_API_KEY")
    if not api_key:
        print("error: MINT_API_KEY missing in environment", file=sys.stderr)
        return 1

    try:
        status = asyncio.run(_health_check(base_url, api_key))
        models = _get_capabilities(base_url, api_key)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"status={status} base_url={str(base_url).rstrip('/')} {_format_models(models)}")
    return 0 if status in ("ready", "ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
