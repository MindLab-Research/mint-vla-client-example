#!/usr/bin/env python3
"""Reproduce duplicate-prompt coalescing failure via raw HTTP calls.

This creates one sampling session, then submits two concurrent /api/v1/asample
requests with identical prompt tokens and different seq_id values. On the
affected MinT deployment, one of the futures returns an error like:

    coalesce: got 1 results for total_samples=2
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path
from typing import Any

import httpx
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HF_HOME = "/vePFS-Mindverse/share/huggingface"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ENV_KEYS = {"MINT_BASE_URL", "MINT_API_KEY"}

DEFAULT_DUPLICATE_PROMPT = (
    "依据给出的实体类型提取句子的实体信息，实体类型包括:犯罪嫌疑人、受害人、被盗货币、物品价值、"
    "盗窃获利、被盗物品、作案工具、时间、地点、组织机构。逐个列出实体信息。\n"
    "句子:经黑河市爱辉区价格认证中心价格鉴定：被盗虾仁价值人民币200.00元。"
)
DEFAULT_DISTINCT_PROMPT = (
    "依据给出的实体类型提取句子的实体信息，实体类型包括:犯罪嫌疑人、受害人、被盗货币、物品价值、"
    "盗窃获利、被盗物品、作案工具、时间、地点、组织机构。逐个列出实体信息。\n"
    "句子:破案后，公安机关将查获手机依法返还给了被害人严某某、肖某某。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--http-timeout", type=float, default=30.0)
    parser.add_argument("--poll-timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--mode", choices=("duplicate", "distinct"), default="duplicate")
    parser.add_argument("--prompt", default=DEFAULT_DUPLICATE_PROMPT)
    parser.add_argument("--other-prompt", default=DEFAULT_DISTINCT_PROMPT)
    return parser.parse_args()


def load_local_env(path: Path) -> None:
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[len("export ") :].lstrip()
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            if key not in ENV_KEYS or key in os.environ:
                continue
            os.environ[key] = value.strip().strip('"').strip("'")
    os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


load_local_env(REPO_ROOT / ".env")


def base_url() -> str:
    return (os.environ.get("MINT_BASE_URL") or os.environ.get("TINKER_BASE_URL") or "http://localhost:8000").rstrip("/")


def api_key() -> str:
    return os.environ.get("MINT_API_KEY") or os.environ.get("TINKER_API_KEY") or "dummy"


def headers() -> dict[str, str]:
    key = api_key().strip()
    return {"X-API-Key": key} if key else {}


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


def get_tokenizer(model_name: str) -> Any:
    cache_dir = cached_tokenizer_dir(model_name)
    if cache_dir is not None:
        print(f"@@ tokenizer_cache model={model_name} path={cache_dir}")
        return AutoTokenizer.from_pretrained(str(cache_dir), fast=True, local_files_only=True)
    return AutoTokenizer.from_pretrained(model_name, fast=True)


def prompt_tokens(tokenizer: Any, prompt: str) -> list[int]:
    apply_fn = getattr(tokenizer, "apply_chat_template", None)
    messages = [{"role": "user", "content": prompt}]
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
        return [int(t) for t in toks]
    text = "\n\n".join(f"{m['role']}:\n{m['content']}" for m in messages)
    return [int(t) for t in tokenizer.encode(text + "\n\nassistant:", add_special_tokens=True)]


async def post_json(client: httpx.AsyncClient, path: str, payload: dict[str, Any], timeout_s: float) -> tuple[int, dict[str, Any]]:
    response = await client.post(f"{base_url()}{path}", json=payload, headers=headers(), timeout=timeout_s)
    try:
        data = response.json()
    except Exception:
        data = {"_non_json_body": response.text[:400]}
    if not isinstance(data, dict):
        data = {"_non_dict_json": str(type(data))}
    return response.status_code, data


async def create_sampling_session(client: httpx.AsyncClient, base_model_name: str, timeout_s: float) -> str:
    status, sess = await post_json(
        client,
        "/api/v1/create_session",
        {"tags": ["scripts/tools/reproduce_issue_510.py"], "user_metadata": {}, "sdk_version": "repro-issue-510"},
        timeout_s,
    )
    if status != 200:
        raise RuntimeError(f"create_session returned {status}: {sess!r}")
    session_id = sess.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError(f"create_session missing session_id: {sess!r}")
    status, out = await post_json(
        client,
        "/api/v1/create_sampling_session",
        {"session_id": session_id, "sampling_session_seq_id": 0, "base_model": base_model_name},
        timeout_s,
    )
    if status != 200:
        raise RuntimeError(f"create_sampling_session returned {status}: {out!r}")
    sampling_session_id = out.get("sampling_session_id")
    if not isinstance(sampling_session_id, str) or not sampling_session_id:
        raise RuntimeError(f"create_sampling_session missing sampling_session_id: {out!r}")
    return sampling_session_id


async def asample(client: httpx.AsyncClient, *, sampling_session_id: str, seq_id: int, tokens: list[int], args: argparse.Namespace) -> str:
    status, out = await post_json(
        client,
        "/api/v1/asample",
        {
            "sampling_session_id": sampling_session_id,
            "seq_id": int(seq_id),
            "num_samples": 1,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
            "sampling_params": {
                "max_tokens": int(args.max_tokens),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "top_k": int(args.top_k),
            },
        },
        args.http_timeout,
    )
    if status != 200:
        raise RuntimeError(f"asample returned {status}: {out!r}")
    request_id = out.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError(f"asample missing request_id: {out!r}")
    return request_id


async def poll_future(client: httpx.AsyncClient, request_id: str, timeout_s: float) -> dict[str, Any]:
    start = time.time()
    while True:
        if time.time() - start > timeout_s:
            raise TimeoutError(f"retrieve_future timeout after {timeout_s:.1f}s request_id={request_id}")
        status, out = await post_json(client, "/api/v1/retrieve_future", {"request_id": request_id}, timeout_s)
        if status == 408:
            await asyncio.sleep(0.2)
            continue
        if status != 200:
            raise RuntimeError(f"retrieve_future returned {status}: {out!r}")
        return out


async def main_async(args: argparse.Namespace) -> int:
    tokenizer = get_tokenizer(args.base_model)
    prompt_a = args.prompt
    prompt_b = args.prompt if args.mode == "duplicate" else args.other_prompt
    tokens_a = prompt_tokens(tokenizer, prompt_a)
    tokens_b = prompt_tokens(tokenizer, prompt_b)

    print(f"base_url={base_url()}")
    print(f"mode={args.mode} attempts={args.attempts} same_prompt={prompt_a == prompt_b}")

    limits = httpx.Limits(max_connections=10, max_keepalive_connections=10)
    async with httpx.AsyncClient(http2=False, limits=limits) as client:
        sampling_session_id = await create_sampling_session(client, args.base_model, args.http_timeout)
        print(f"sampling_session_id={sampling_session_id}")
        for attempt in range(1, max(int(args.attempts), 1) + 1):
            print(f"===== attempt {attempt} =====")
            req1, req2 = await asyncio.gather(
                asample(client, sampling_session_id=sampling_session_id, seq_id=attempt * 2 - 1, tokens=tokens_a, args=args),
                asample(client, sampling_session_id=sampling_session_id, seq_id=attempt * 2, tokens=tokens_b, args=args),
            )
            fut1, fut2 = await asyncio.gather(
                poll_future(client, req1, args.poll_timeout),
                poll_future(client, req2, args.poll_timeout),
            )
            errors = [fut.get("error") for fut in (fut1, fut2) if isinstance(fut, dict) and fut.get("error")]
            if errors:
                print(f"request_ids={[req1, req2]}")
                for idx, err in enumerate(errors, start=1):
                    print(f"error_{idx}={err}")
                return 1
            for idx, fut in enumerate((fut1, fut2), start=1):
                sequences = fut.get("sequences") if isinstance(fut, dict) else None
                if not isinstance(sequences, list) or not sequences:
                    raise RuntimeError(f"retrieve_future missing sequences: {fut!r}")
                seq0 = sequences[0]
                tokens = seq0.get("tokens") if isinstance(seq0, dict) else None
                if not isinstance(tokens, list):
                    raise RuntimeError(f"sequence missing tokens: {seq0!r}")
                preview = tokenizer.decode(tokens).replace("\n", "\\n")[:120]
                print(f"result_{idx}={preview}")
    print("No failure reproduced within the configured attempts.")
    return 2


def main() -> int:
    args = parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
