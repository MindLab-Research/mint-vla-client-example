"""Bumblebee-native LoRA adapter conversion helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


RANK_SHARDED_MANIFEST = "bumblebee_rank_sharded_adapter.json"
STREAMED_SHARDED_INDEX = "adapter_model.safetensors.index.json"


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _artifact_fingerprint(adapter_dir: Path, manifest_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(str(adapter_dir.resolve()).encode("utf-8"))
    digest.update(b"\0")
    digest.update(manifest_path.read_bytes())
    return digest.hexdigest()[:24]


def _cache_root() -> Path:
    raw = os.environ.get("MINT_BUMBLEBEE_LORA_CACHE_DIR", "").strip()
    if raw:
        return Path(raw)
    return Path(os.environ.get("TMPDIR", "/tmp")) / "mint_bumblebee_lora_cache"


def _load_shard_tensor(shard_cache: dict[str, dict[str, Any]], shard_path: Path, name: str):
    from safetensors.torch import load_file

    key = str(shard_path)
    tensors = shard_cache.get(key)
    if tensors is None:
        tensors = load_file(str(shard_path), device="cpu")
        shard_cache[key] = tensors
    try:
        return tensors[name]
    except KeyError as exc:
        raise KeyError(f"Missing tensor {name!r} in Bumblebee shard {shard_path}") from exc


def _select_records(records: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if not records:
        return []
    kind = str(records[0][1].get("kind") or "")
    if kind == "replicated":
        canonical = records[0][1].get("canonical") or {}
        for record, placement in records:
            ranks = record.get("parallel_rank") or {}
            if all(int(ranks.get(axis, 0)) == int(value) for axis, value in canonical.items()):
                return [(record, placement)]
        return [records[0]]
    if kind == "unique":
        canonical = records[0][1].get("canonical") or {}
        for record, placement in records:
            ranks = record.get("parallel_rank") or {}
            if all(int(ranks.get(axis, 0)) == int(value) for axis, value in canonical.items()):
                return [(record, placement)]
        return [records[0]]
    if kind == "concat":
        canonical = records[0][1].get("canonical") or {}
        if canonical:
            records = [
                (record, placement)
                for record, placement in records
                if all(
                    int((record.get("parallel_rank") or {}).get(axis, 0)) == int(value)
                    for axis, value in canonical.items()
                )
            ]
        return sorted(records, key=lambda pair: int(pair[1].get("index", 0)))
    raise ValueError(f"Unsupported Bumblebee LoRA tensor placement kind={kind!r}")


def _merge_rank_sharded_tensors(adapter_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    import torch

    by_name: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for shard in manifest.get("shards") or []:
        if not isinstance(shard, dict):
            continue
        for entry in shard.get("tensors") or []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "")
            if not name:
                continue
            placement = entry.get("placement")
            if not isinstance(placement, dict):
                raise ValueError(f"Bumblebee tensor {name!r} missing placement metadata")
            by_name.setdefault(name, []).append((shard, placement))

    shard_cache: dict[str, dict[str, Any]] = {}
    merged: dict[str, Any] = {}
    for name, records in by_name.items():
        selected = _select_records(records)
        if len(selected) == 1 and str(selected[0][1].get("kind") or "") != "concat":
            shard, _placement = selected[0]
            merged[name] = _load_shard_tensor(shard_cache, adapter_dir / str(shard["file"]), name)
            continue
        tensors = []
        axis = int(selected[0][1].get("axis", 0))
        expected_parts = int(selected[0][1].get("parts", len(selected)))
        if len(selected) != expected_parts:
            raise ValueError(
                f"Bumblebee tensor {name!r} has {len(selected)} concat shards, expected {expected_parts}"
            )
        for shard, _placement in selected:
            tensors.append(_load_shard_tensor(shard_cache, adapter_dir / str(shard["file"]), name))
        merged[name] = torch.cat(tensors, dim=axis).contiguous()
    return merged


def _ensure_complete_peft_dir(output_dir: Path) -> bool:
    return (output_dir / "adapter_model.safetensors").is_file() and (
        output_dir / "adapter_config.json"
    ).is_file()


def prepare_lora_adapter_for_vllm(adapter_path: str) -> str:
    """Return a vLLM-compatible PEFT adapter directory for ``adapter_path``.

    Standard PEFT directories are returned unchanged. Bumblebee rank-sharded
    artifacts are converted on CPU into a cached PEFT directory. Streamed size
    shards intentionally fail fast because they are an intermediate write format
    without rank/load semantics.
    """

    adapter_dir = Path(adapter_path)
    if (adapter_dir / "adapter_model.safetensors").is_file():
        return str(adapter_dir)

    rank_manifest_path = adapter_dir / RANK_SHARDED_MANIFEST
    if rank_manifest_path.is_file():
        manifest = _read_json_object(rank_manifest_path)
        if manifest.get("format") != "bumblebee_qwen3_moe_lora_rank_sharded_v1":
            raise ValueError(f"Unsupported Bumblebee LoRA manifest format: {manifest.get('format')!r}")
        if manifest.get("sharding_kind") != "rank":
            raise ValueError("Bumblebee LoRA manifest is not rank-sharded and cannot be converted")
        config_path = adapter_dir / "adapter_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing adapter_config.json for Bumblebee adapter: {adapter_dir}")

        cache_dir = _cache_root() / _artifact_fingerprint(adapter_dir, rank_manifest_path)
        if _ensure_complete_peft_dir(cache_dir):
            return str(cache_dir)

        tmp_dir = cache_dir.with_name(cache_dir.name + ".tmp")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        from safetensors.torch import save_file

        state_dict = _merge_rank_sharded_tensors(adapter_dir, manifest)
        save_file(state_dict, str(tmp_dir / "adapter_model.safetensors"))
        shutil.copy2(config_path, tmp_dir / "adapter_config.json")
        meta_path = adapter_dir / "bumblebee_adapter_meta.json"
        if meta_path.is_file():
            shutil.copy2(meta_path, tmp_dir / "bumblebee_adapter_meta.json")
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        os.replace(tmp_dir, cache_dir)
        return str(cache_dir)

    if (adapter_dir / STREAMED_SHARDED_INDEX).is_file():
        raise ValueError(
            "Bumblebee streamed_sharded adapter is an intermediate size-sharded artifact "
            "and is not directly loadable by vLLM; export rank_sharded or peft instead."
        )

    return str(adapter_dir)
