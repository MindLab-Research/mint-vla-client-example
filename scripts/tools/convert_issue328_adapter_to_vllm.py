from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


MODULE_MAP = {
    ".mlp.experts.w1.": ".mlp.experts.0.gate_proj.",
    ".mlp.experts.w2.": ".mlp.experts.0.down_proj.",
    ".mlp.experts.w3.": ".mlp.experts.0.up_proj.",
}


def _convert_key(key: str) -> str:
    if ".unembed_tokens." in key:
        return ""
    out = key
    for src, dst in MODULE_MAP.items():
        if src in out:
            out = out.replace(src, dst)
            break
    return out


def _convert_tensor(key: str, tensor):
    if ".mlp.experts." in key and tensor.ndim >= 3:
        # Old issue328-style export stores expert-stacked routed tensors on one side
        # of the LoRA pair. For the sparse expert-0 path, keep only one representative
        # expert slice and let the runtime sparse patch broadcast it.
        return tensor[0].contiguous()
    if tensor.ndim >= 3 and tensor.shape[0] == 1:
        return tensor[0].contiguous()
    return tensor


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: convert_issue328_adapter_to_vllm.py <src_dir> <dst_dir>")

    src_dir = Path(sys.argv[1]).resolve()
    dst_dir = Path(sys.argv[2]).resolve()
    src_model = src_dir / "adapter_model.safetensors"
    src_cfg = src_dir / "adapter_config.json"
    if not src_model.is_file():
        raise SystemExit(f"missing adapter_model.safetensors: {src_model}")
    if not src_cfg.is_file():
        raise SystemExit(f"missing adapter_config.json: {src_cfg}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    tensors = {}
    with safe_open(str(src_model), framework="pt", device="cpu") as f:
        for key in f.keys():
            new_key = _convert_key(key)
            if not new_key:
                continue
            tensors[new_key] = _convert_tensor(key, f.get_tensor(key))
    save_file(tensors, str(dst_dir / "adapter_model.safetensors"))

    cfg = json.loads(src_cfg.read_text())
    cfg["base_model_name_or_path"] = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    cfg["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    (dst_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    checkpoint_complete = src_dir / "checkpoint_complete"
    if checkpoint_complete.exists():
        data = checkpoint_complete.read_bytes()
        (dst_dir / "checkpoint_complete").write_bytes(data)
    else:
        (dst_dir / "checkpoint_complete").write_text("", encoding="utf-8")

    print(f"converted {src_dir} -> {dst_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
