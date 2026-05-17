#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import uuid

from requests import HTTPError
from pathlib import Path
from typing import Any

import numpy as np
import requests

from openpi_libero_sft import (
    _build_transform,
    _collect_transformed_items,
    _decode_image,
    _encode_png_base64,
    _episode_path,
    _iter_windows_for_task,
    _load_tasks,
    _plot_curve,
    CONFIG_NAME_BY_BASE_MODEL,
)


def _request_headers() -> dict[str, str]:
    headers = {"Connection": "close"}
    api_key = (os.environ.get("TINKER_API_KEY") or os.environ.get("MINT_API_KEY") or "").strip()
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _http_post(url: str, *, payload: dict[str, Any], timeout: float) -> requests.Response:
    return requests.post(url, json=payload, timeout=timeout, headers=_request_headers())


def _http_delete(url: str, *, timeout: float) -> requests.Response:
    return requests.delete(url, timeout=timeout, headers=_request_headers())


def _poll_future(base_url: str, request_id: str, *, timeout_s: float = 3600.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            resp = _http_post(f"{base_url}/api/v1/retrieve_future", payload={"request_id": request_id}, timeout=120)
        except requests.Timeout:
            time.sleep(1.0)
            continue
        if resp.status_code in {408, 503}:
            time.sleep(1.0)
            continue
        resp.raise_for_status()
        return resp.json()
    raise TimeoutError(f"timed out waiting for {request_id}")


def _create_model(base_url: str, base_model: str) -> str:
    payload = {
        "session_id": f"rl-{uuid.uuid4().hex[:12]}",
        "model_seq_id": 0,
        "base_model": base_model,
        "lora_config": {"rank": 16, "train_attn": True, "train_mlp": True, "train_unembed": True},
        "user_metadata": {"script": "scripts/wip/openpi_libero_fast_rl.py"},
    }
    resp = _http_post(f"{base_url}/api/v1/create_model", payload=payload, timeout=120)
    resp.raise_for_status()
    result = _poll_future(base_url, resp.json()["request_id"], timeout_s=3600)
    model_id = result.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"create_model missing model_id: {result!r}")
    return model_id


def _create_model_from_state(
    base_url: str,
    *,
    base_model: str,
    state_path: str,
    load_optimizer: bool = False,
) -> str:
    payload = {
        "session_id": f"rl-{uuid.uuid4().hex[:12]}",
        "model_seq_id": 0,
        "base_model": base_model,
        "state_path": state_path,
        "lora_config": {"rank": 16, "train_attn": True, "train_mlp": True, "train_unembed": True},
        "load_optimizer": bool(load_optimizer),
        "user_metadata": {"script": "scripts/wip/openpi_libero_fast_rl.py"},
    }
    owner_id = _checkpoint_owner_id_from_uri(state_path)
    if owner_id is not None:
        payload["owner_id"] = owner_id
    resp = _http_post(f"{base_url}/api/v1/create_model_from_state", payload=payload, timeout=120)
    resp.raise_for_status()
    result = _poll_future(base_url, resp.json()["request_id"], timeout_s=3600)
    model_id = result.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"create_model_from_state missing model_id: {result!r}")
    return model_id


def _delete_model(base_url: str, model_id: str) -> None:
    try:
        _http_delete(f"{base_url}/api/v1/models/{model_id}", timeout=300)
    except Exception:
        pass


def _save_weights_for_sampler(base_url: str, model_id: str, checkpoint_name: str) -> str:
    resp = _http_post(
        f"{base_url}/api/v1/save_weights_for_sampler",
        payload={"model_id": model_id, "path": checkpoint_name},
        timeout=120,
    )
    resp.raise_for_status()
    result = _poll_future(base_url, resp.json()["request_id"], timeout_s=3600)
    path = result.get("path")
    if not isinstance(path, str) or not path:
        raise RuntimeError(f"save_weights_for_sampler missing path: {result!r}")
    return path


def _checkpoint_owner_id_from_uri(checkpoint_uri: str) -> str | None:
    if not checkpoint_uri.startswith(("mint://", "tinker://", "ckpt_")):
        return None

    env_override = (
        os.environ.get("OPENPI_VLA_CHECKPOINT_OWNER_ID")
        or os.environ.get("TINKER_CHECKPOINT_OWNER_ID")
        or ""
    ).strip()
    if env_override:
        return env_override

    search_roots = [
        Path("/vePFS-Mindverse/share/mint/prod/data/runtime-checkpoints/persistent_cache"),
        Path("/vePFS-Mindverse/share/mint/prod/data/runtime-checkpoints/ephemeral"),
        Path("/vePFS-Mindverse/share/mint/dev/data/runtime-checkpoints/persistent_cache"),
        Path("/vePFS-Mindverse/share/mint/dev/data/runtime-checkpoints/ephemeral"),
        Path("/tos-mindverse/tinker_checkpoints"),
        Path("/vePFS-Mindverse/share/tinker_checkpoints"),
    ]
    owner_ids: set[str] = set()

    if checkpoint_uri.startswith("ckpt_"):
        patterns = [f"*/{checkpoint_uri}/**/metadata.json"]
    else:
        raw_path = checkpoint_uri.split("://", 1)[1]
        parts = [part for part in raw_path.split("/") if part]
        if len(parts) < 3:
            return None
        model_id = parts[0]
        checkpoint_kind = parts[1]
        checkpoint_name = "/".join(parts[2:])
        checkpoint_type = {
            "weights": "training",
            "sampler_weights": "sampler",
        }.get(checkpoint_kind)
        if checkpoint_type is None:
            return None
        patterns = [f"*/{model_id}/{checkpoint_name}/{checkpoint_type}/metadata.json"]

    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for metadata_path in root.glob(pattern):
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                owner_id = str(metadata.get("owner_id") or "anonymous").strip() or "anonymous"
                owner_ids.add(owner_id)

    if not owner_ids:
        return None
    if len(owner_ids) > 1:
        raise RuntimeError(
            f"checkpoint owner is ambiguous for {checkpoint_uri!r}: {sorted(owner_ids)}"
        )
    return next(iter(owner_ids))


def _create_action_session(base_url: str, base_model: str, model_path: str, *, timeout_s: float = 3600.0) -> str:
    deadline = time.time() + timeout_s
    payload = {
        "session_id": f"act-{uuid.uuid4().hex[:12]}",
        "base_model": base_model,
        "model_path": model_path,
    }
    owner_id = _checkpoint_owner_id_from_uri(model_path)
    if owner_id is not None:
        payload["owner_id"] = owner_id
    while True:
        resp = _http_post(
            f"{base_url}/api/v1/mint/action_sessions",
            payload=payload,
            timeout=3600,
        )
        if resp.status_code in {429, 503} and time.time() < deadline:
            time.sleep(2.0)
            continue
        try:
            resp.raise_for_status()
        except HTTPError:
            raise
        result = resp.json()
        session_id = result.get("action_session_id")
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError(f"create_action_session missing action_session_id: {result!r}")
        return session_id


def _delete_action_session(base_url: str, action_session_id: str) -> None:
    try:
        _http_delete(f"{base_url}/api/v1/mint/action_sessions/{action_session_id}", timeout=120)
    except Exception:
        pass


def _resolve_fast_tokenizer_path() -> str:
    hf_home = Path("/vePFS-Mindverse/share/huggingface")
    repo_root = hf_home / "hub" / "models--physical-intelligence--fast"
    refs_main = repo_root / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        if revision:
            snapshot_dir = repo_root / "snapshots" / revision
            if snapshot_dir.exists():
                return str(snapshot_dir)
    snapshots = sorted((repo_root / "snapshots").glob("*"))
    if snapshots:
        return str(snapshots[-1])
    raise FileNotFoundError(f"FAST tokenizer snapshot not found under {repo_root}")


def _prompt_tokens(item: dict[str, Any]) -> list[int]:
    prompt_tokens = np.asarray(item["tokenized_prompt"])[
        np.asarray(item["tokenized_prompt_mask"]).astype(bool)
        & ~np.asarray(item["token_loss_mask"]).astype(bool)
    ].astype(int)
    return prompt_tokens.tolist()


def _observation_chunks(item: dict[str, Any], prompt_tokens: list[int]) -> list[dict[str, Any]]:
    image = item["image"]
    image_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    missing = [key for key in image_keys if key not in image]
    if missing:
        raise KeyError(f"missing camera keys in transformed item: {missing}")
    chunks: list[dict[str, Any]] = [
        {
            "type": "image",
            "data": _encode_png_base64(np.asarray(image[key])),
            "format": "png",
            "expected_tokens": 256,
        }
        for key in image_keys
    ]
    chunks.append({"type": "encoded_text", "tokens": prompt_tokens})
    return chunks


def _make_action_observation(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "observation": {
            "state": {
                "data": np.asarray(item["state"], dtype=np.float32).reshape(-1).tolist(),
                "shape": list(np.asarray(item["state"]).shape),
                "dtype": "float32",
            },
            "model_input": {"chunks": _observation_chunks(item, _prompt_tokens(item))},
        }
    }


def _sample_actions(base_url: str, action_session_id: str, item: dict[str, Any], *, temperature: float = 0.0) -> np.ndarray:
    payload = _make_action_observation(item)
    payload["temperature"] = float(temperature)
    resp = _http_post(
        f"{base_url}/api/v1/mint/action_sessions/{action_session_id}/act",
        payload=payload,
        timeout=120,
    )
    resp.raise_for_status()
    result = _poll_future(base_url, resp.json()["request_id"], timeout_s=3600)
    actions = result.get("actions")
    if not isinstance(actions, dict):
        raise RuntimeError(f"act future missing actions payload: {result!r}")
    arr = np.asarray(actions["data"], dtype=np.float32)
    shape = list(actions["shape"])
    return arr.reshape(shape)


def _tokenize_sampled_actions(tokenizer, task_text: str, item: dict[str, Any], actions: np.ndarray):
    tokens, token_mask, token_ar_mask, loss_mask = tokenizer.tokenize(
        task_text,
        np.asarray(item["state"], dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
    )
    prefix_mask = np.asarray(token_mask).astype(bool) & ~np.asarray(loss_mask).astype(bool)
    target_mask = np.asarray(loss_mask).astype(bool)
    prefix_tokens = np.asarray(tokens)[prefix_mask].astype(int).tolist()
    target_tokens = np.asarray(tokens)[target_mask].astype(int).tolist()
    suffix_token_ar_mask = np.asarray(token_ar_mask)[target_mask].astype(int).tolist()
    return prefix_tokens, target_tokens, suffix_token_ar_mask


def _make_rl_datum(item: dict[str, Any], prefix_tokens: list[int], target_tokens: list[int], suffix_token_ar_mask: list[int], *, logprobs: list[float], advantages: list[float]) -> dict[str, Any]:
    if not (len(target_tokens) == len(logprobs) == len(advantages) == len(suffix_token_ar_mask)):
        raise ValueError("target_tokens/logprobs/advantages/token_ar_mask length mismatch")
    return {
        "model_input": {"chunks": _observation_chunks(item, prefix_tokens)},
        "loss_fn_inputs": {
            "state": {"data": np.asarray(item["state"], dtype=np.float32).reshape(-1).tolist(), "shape": list(np.asarray(item["state"]).shape), "dtype": "float32"},
            "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
            "weights": {"data": [1.0] * len(target_tokens), "shape": [len(target_tokens)], "dtype": "float32"},
            "token_ar_mask": {"data": suffix_token_ar_mask, "shape": [len(suffix_token_ar_mask)], "dtype": "int64"},
            "logprobs": {"data": logprobs, "shape": [len(logprobs)], "dtype": "float32"},
            "advantages": {"data": advantages, "shape": [len(advantages)], "dtype": "float32"},
        },
    }


def _forward_logprobs(base_url: str, model_id: str, datum: dict[str, Any]) -> list[float]:
    zero_len = len(datum["loss_fn_inputs"]["target_tokens"]["data"])
    datum = json.loads(json.dumps(datum))
    datum["loss_fn_inputs"]["logprobs"]["data"] = [0.0] * zero_len
    datum["loss_fn_inputs"]["advantages"]["data"] = [0.0] * zero_len
    resp = _http_post(
        f"{base_url}/api/v1/forward_backward",
        payload={"model_id": model_id, "forward_backward_input": {"loss_fn": "importance_sampling", "data": [datum]}},
        timeout=120,
    )
    resp.raise_for_status()
    result = _poll_future(base_url, resp.json()["request_id"], timeout_s=3600)
    outputs = result["loss_fn_outputs"]
    return [float(x) for x in outputs[0]["logprobs"]["data"]]


def _ppo_train_step(base_url: str, model_id: str, datum: dict[str, Any]) -> dict[str, Any]:
    resp = _http_post(
        f"{base_url}/api/v1/train_step",
        payload={
            "model_id": model_id,
            "forward_backward_input": {"loss_fn": "ppo", "loss_fn_config": {"epsilon": 0.2}, "data": [datum]},
            "adam_params": {"learning_rate": 1e-4},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return _poll_future(base_url, resp.json()["request_id"], timeout_s=3600)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    base_model = "openpi/pi0-fast-libero-low-mem-finetune"
    base_url = args.base_url.rstrip("/")
    tasks = _load_tasks()
    task_text = tasks[args.task_index]
    cfg, tx = _build_transform(base_model)
    items, pool_meta = _collect_transformed_items(base_model, tx, task_text, int(cfg.model.action_horizon), max_episodes=args.max_episodes, stride=args.stride)
    rng = random.Random(args.seed)

    from openpi.models.tokenizer import FASTTokenizer
    tokenizer = FASTTokenizer(cfg.model.max_token_len, fast_tokenizer_path=_resolve_fast_tokenizer_path())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    reward_steps: list[int] = []
    rewards: list[float] = []
    losses: list[float] = []

    model_id = _create_model(base_url, base_model)
    action_session_id = None
    try:
        for step in range(1, args.steps + 1):
            checkpoint_path = _save_weights_for_sampler(base_url, model_id, f"rl-step-{step}-{uuid.uuid4().hex[:6]}")
            if action_session_id:
                _delete_action_session(base_url, action_session_id)
            action_session_id = _create_action_session(base_url, base_model, checkpoint_path)

            step_rewards = []
            step_losses = []
            for _ in range(args.batch_size):
                item = items[rng.randrange(len(items))]
                sampled_actions = _sample_actions(base_url, action_session_id, item)
                expert_actions = np.asarray(item["actions"], dtype=np.float32)
                mse = float(np.mean((sampled_actions - expert_actions) ** 2))
                reward = math.exp(-5.0 * mse)
                prefix_tokens, target_tokens, suffix_token_ar_mask = _tokenize_sampled_actions(
                    tokenizer, task_text, item, sampled_actions
                )
                if not target_tokens:
                    continue
                probe_datum = _make_rl_datum(item, prefix_tokens, target_tokens, suffix_token_ar_mask, logprobs=[0.0] * len(target_tokens), advantages=[0.0] * len(target_tokens))
                old_logprobs = _forward_logprobs(base_url, model_id, probe_datum)
                datum = _make_rl_datum(item, prefix_tokens, target_tokens, suffix_token_ar_mask, logprobs=old_logprobs, advantages=[reward] * len(target_tokens))
                result = _ppo_train_step(base_url, model_id, datum)
                step_rewards.append(reward)
                step_losses.append(float(result["metrics"]["loss:mean"]))

            mean_reward = float(sum(step_rewards) / max(len(step_rewards), 1))
            mean_loss = float(sum(step_losses) / max(len(step_losses), 1))
            record = {"step": step, "reward": mean_reward, "loss": mean_loss, "num_samples": len(step_rewards)}
            metrics_path.open("a", encoding="utf-8").write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            reward_steps.append(step)
            rewards.append(mean_reward)
            losses.append(mean_loss)

        _plot_curve(reward_steps, rewards, out_dir / "reward_curve.png", f"{base_model} | reward task={args.task_index}")
        _plot_curve(reward_steps, losses, out_dir / "loss_curve.png", f"{base_model} | ppo loss task={args.task_index}")
        summary = {
            "base_model": base_model,
            "task_index": args.task_index,
            "task": task_text,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "initial_reward": rewards[0] if rewards else None,
            "final_reward": rewards[-1] if rewards else None,
            "max_reward": max(rewards) if rewards else None,
            "initial_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None,
            "reward_curve_path": str(out_dir / "reward_curve.png"),
            "loss_curve_path": str(out_dir / "loss_curve.png"),
            **pool_meta,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"event": "done", **summary}), flush=True)
    finally:
        if action_session_id:
            _delete_action_session(base_url, action_session_id)
        _delete_model(base_url, model_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
