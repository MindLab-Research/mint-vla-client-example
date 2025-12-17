"""Test LoRA dimension mismatch when loading Qwen3 LoRA into Qwen2.5 vLLM engine."""
import torch
import tempfile
import os
import json
import requests
import uuid
from safetensors.torch import save_file

# Create fake LoRA weights with Qwen3 hidden_size (2048) - WRONG for Qwen2.5 (3584)
lora_rank = 32
hidden_size_qwen3 = 2048  # Qwen3-30B-A3B hidden_size

tmpdir = tempfile.mkdtemp(prefix="test_lora_mismatch_")
print(f"Temp dir: {tmpdir}")

# Create minimal state_dict - just one layer
state_dict = {}
state_dict["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"] = torch.randn(lora_rank, hidden_size_qwen3)
state_dict["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"] = torch.randn(hidden_size_qwen3, lora_rank)

save_file(state_dict, os.path.join(tmpdir, "adapter_model.safetensors"))
print(f"Saved weights: lora_A shape [{lora_rank}, {hidden_size_qwen3}]")

peft_config = {
    "base_model_name_or_path": "Qwen/Qwen3-30B-A3B",
    "target_modules": ["q_proj"],
    "r": lora_rank,
    "lora_alpha": 32,
    "peft_type": "LORA",
}
with open(os.path.join(tmpdir, "adapter_config.json"), "w") as f:
    json.dump(peft_config, f)

# Create session
session_id = str(uuid.uuid4())
requests.post("http://localhost:8000/api/v1/create_session", json={"session_id": session_id, "model_name": "Qwen/Qwen3-30B-A3B"})

# Try sampling session with mismatched LoRA
print(f"\nCalling create_sampling_session with file://{tmpdir}...")
try:
    response = requests.post(
        "http://localhost:8000/api/v1/create_sampling_session",
        json={
            "session_id": session_id,
            "base_model": "Qwen/Qwen3-30B-A3B",
            "model_path": f"file://{tmpdir}",
            "lora_rank": lora_rank,
        },
        timeout=60
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Exception: {e}")

# Keep temp dir for debugging
print(f"\nKeeping temp dir: {tmpdir}")
