#!/usr/bin/env python3
"""Run a real dev sampling E2E smoke against a live Mint API server.

This script is intentionally small and explicit: it does not read .env files,
does not start the server, and returns 0 only after retrieve_future produces
non-empty generated token sequences.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:18080"
DEFAULT_HF_HOME = "/vePFS-Mindverse/share/huggingface"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_PROMPT = (
    "用一句话说明调度器为什么需要在执行 lease 时持续续租和上报 liveness。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("MINT_BASE_URL") or DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=os.environ.get("MINT_API_KEY") or "dummy")
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--http-timeout", type=float, default=60.0)
    parser.add_argument("--poll-timeout", type=float, default=360.0)
    parser.add_argument("--poll-interval", type=float, default=0.2)
    parser.add_argument("--evidence-jsonl", type=Path)
    return parser.parse_args()


class EventWriter:
    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("w", encoding="utf-8")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def emit(self, event: str, **fields: Any) -> None:
        payload = {"event": event, "ts": time.time(), **fields}
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        print(line, flush=True)
        if self._file is not None:
            self._file.write(line + "\n")
            self._file.flush()


def cached_tokenizer_dir(model_name: str) -> Path | None:
    if "/" not in model_name:
        return None
    hf_home = Path(os.environ.get("HF_HOME", DEFAULT_HF_HOME)).expanduser()
    hub_root = hf_home / "hub" if (hf_home / "hub").exists() else hf_home / ".cache" / "huggingface" / "hub"
    org, repo = model_name.split("/", 1)
    repo_dir = hub_root / f"models--{org.replace('/', '--')}--{repo.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    def has_tokenizer_files(path: Path) -> bool:
        if not path.is_dir():
            return False
        files = {f.name for f in path.iterdir() if f.is_file()}
        if "tokenizer_config.json" not in files:
            return False
        return "tokenizer.json" in files or {"vocab.json", "merges.txt"}.issubset(files)

    candidates = sorted((p for p in snapshots_dir.iterdir() if has_tokenizer_files(p)), reverse=True)
    return candidates[0] if candidates else None


def load_tokenizer(model_name: str, events: EventWriter) -> Any:
    from transformers import AutoTokenizer

    os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    cache_dir = cached_tokenizer_dir(model_name)
    if cache_dir is not None:
        events.emit("tokenizer_cache", model=model_name, path=str(cache_dir))
        return AutoTokenizer.from_pretrained(str(cache_dir), fast=True, local_files_only=True)
    events.emit("tokenizer_remote_or_default_cache", model=model_name)
    return AutoTokenizer.from_pretrained(model_name, fast=True)


def prompt_tokens(tokenizer: Any, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    apply_fn = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_fn):
        toks = apply_fn(messages, tokenize=True, add_generation_prompt=True)
        if hasattr(toks, "input_ids"):
            toks = toks.input_ids
        elif isinstance(toks, dict) and "input_ids" in toks:
            toks = toks["input_ids"]
        if hasattr(toks, "tolist"):
            toks = toks.tolist()
        if isinstance(toks, list) and toks and isinstance(toks[0], list):
            toks = toks[0]
        return [int(tok) for tok in toks]
    text = "\n\n".join(f"{message['role']}:\n{message['content']}" for message in messages)
    return [int(tok) for tok in tokenizer.encode(text + "\n\nassistant:", add_special_tokens=True)]


def headers(api_key: str) -> dict[str, str]:
    key = api_key.strip()
    return {"X-API-Key": key} if key else {}


async def post_json(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    api_key: str,
    path: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> tuple[int, dict[str, Any]]:
    response = await client.post(
        f"{base_url.rstrip('/')}{path}",
        json=payload,
        headers=headers(api_key),
        timeout=timeout_s,
    )
    try:
        data = response.json()
    except Exception:
        data = {"_non_json_body": response.text[:1000]}
    if not isinstance(data, dict):
        data = {"_non_dict_json": repr(data)}
    return response.status_code, data


async def create_sampling_session(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    events: EventWriter,
) -> str:
    status, session = await post_json(
        client,
        base_url=args.base_url,
        api_key=args.api_key,
        path="/api/v1/create_session",
        payload={
            "tags": ["scripts/tools/smoke_dev_sampling_e2e.py"],
            "user_metadata": {},
            "sdk_version": "dev-sampling-e2e-smoke",
        },
        timeout_s=args.http_timeout,
    )
    if status != 200:
        raise RuntimeError(f"create_session returned {status}: {session!r}")
    session_id = session.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError(f"create_session missing session_id: {session!r}")
    events.emit("create_session_ok", session_id=session_id)

    status, sampling = await post_json(
        client,
        base_url=args.base_url,
        api_key=args.api_key,
        path="/api/v1/create_sampling_session",
        payload={"session_id": session_id, "sampling_session_seq_id": 0, "base_model": args.base_model},
        timeout_s=args.http_timeout,
    )
    if status != 200:
        raise RuntimeError(f"create_sampling_session returned {status}: {sampling!r}")
    sampling_session_id = sampling.get("sampling_session_id")
    if not isinstance(sampling_session_id, str) or not sampling_session_id:
        raise RuntimeError(f"create_sampling_session missing sampling_session_id: {sampling!r}")
    events.emit("create_sampling_session_ok", sampling_session_id=sampling_session_id)
    return sampling_session_id


async def submit_sample(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    *,
    sampling_session_id: str,
    tokens: list[int],
    events: EventWriter,
) -> str:
    status, out = await post_json(
        client,
        base_url=args.base_url,
        api_key=args.api_key,
        path="/api/v1/asample",
        payload={
            "sampling_session_id": sampling_session_id,
            "seq_id": 1,
            "num_samples": 1,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
            "sampling_params": {
                "max_tokens": int(args.max_tokens),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "top_k": int(args.top_k),
            },
        },
        timeout_s=args.http_timeout,
    )
    if status != 200:
        raise RuntimeError(f"asample returned {status}: {out!r}")
    request_id = out.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError(f"asample missing request_id: {out!r}")
    events.emit("asample_ok", request_id=request_id)
    return request_id


def extract_first_sequence_tokens(payload: dict[str, Any]) -> list[int]:
    if payload.get("error"):
        raise RuntimeError(f"retrieve_future returned error: {payload['error']!r}")
    sequences = payload.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise RuntimeError(f"retrieve_future missing non-empty sequences: {payload!r}")
    first = sequences[0]
    if not isinstance(first, dict):
        raise RuntimeError(f"retrieve_future sequence is not an object: {first!r}")
    tokens = first.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise RuntimeError(f"retrieve_future sequence missing non-empty tokens: {first!r}")
    return [int(tok) for tok in tokens]


async def poll_future(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    *,
    request_id: str,
    events: EventWriter,
) -> dict[str, Any]:
    start = time.monotonic()
    polls = 0
    while True:
        elapsed_s = time.monotonic() - start
        if elapsed_s > args.poll_timeout:
            raise TimeoutError(f"retrieve_future timeout after {elapsed_s:.1f}s request_id={request_id}")
        status, out = await post_json(
            client,
            base_url=args.base_url,
            api_key=args.api_key,
            path="/api/v1/retrieve_future",
            payload={"request_id": request_id},
            timeout_s=args.http_timeout,
        )
        polls += 1
        if status == 408:
            if polls == 1 or polls % 50 == 0:
                events.emit("retrieve_future_pending", request_id=request_id, polls=polls, elapsed_s=round(elapsed_s, 3))
            await asyncio.sleep(args.poll_interval)
            continue
        if status != 200:
            raise RuntimeError(f"retrieve_future returned {status}: {out!r}")
        events.emit("retrieve_future_ok", request_id=request_id, polls=polls, elapsed_s=round(elapsed_s, 3))
        return out


async def run(args: argparse.Namespace, events: EventWriter) -> int:
    tokenizer = load_tokenizer(args.base_model, events)
    tokens = prompt_tokens(tokenizer, args.prompt)
    events.emit(
        "smoke_start",
        base_url=args.base_url,
        base_model=args.base_model,
        prompt_tokens=len(tokens),
        max_tokens=args.max_tokens,
    )
    limits = httpx.Limits(max_connections=4, max_keepalive_connections=4)
    async with httpx.AsyncClient(http2=False, limits=limits, trust_env=False) as client:
        sampling_session_id = await create_sampling_session(client, args, events)
        request_id = await submit_sample(client, args, sampling_session_id=sampling_session_id, tokens=tokens, events=events)
        result = await poll_future(client, args, request_id=request_id, events=events)
    generated_tokens = extract_first_sequence_tokens(result)
    preview = tokenizer.decode(generated_tokens).replace("\n", "\\n")[:200]
    events.emit(
        "smoke_success",
        sampling_session_id=sampling_session_id,
        request_id=request_id,
        generated_token_count=len(generated_tokens),
        preview=preview,
    )
    return 0


def main() -> int:
    args = parse_args()
    events = EventWriter(args.evidence_jsonl)
    try:
        return asyncio.run(run(args, events))
    except Exception as exc:
        events.emit("smoke_failure", error_type=type(exc).__name__, error=str(exc))
        return 1
    finally:
        events.close()


if __name__ == "__main__":
    raise SystemExit(main())
