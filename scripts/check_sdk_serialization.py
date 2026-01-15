#!/usr/bin/env python3
"""Check what SDK actually serializes to JSON."""

import json
from tinker import types, TensorData
import torch

# Create datum with plain lists (like SFT section of notebook)
sft_datum = types.Datum(
    model_input=types.ModelInput.from_ints(tokens=[1, 2, 3]),
    loss_fn_inputs={
        "target_tokens": [2, 3, 4],
        "weights": [0.0, 1.0, 1.0],
    }
)

# Create datum with TensorData.from_torch (like RL section of notebook)
rl_datum = types.Datum(
    model_input=types.ModelInput.from_ints(tokens=[1, 2, 3]),
    loss_fn_inputs={
        "target_tokens": TensorData.from_torch(torch.tensor([2, 3, 4], dtype=torch.int64)),
        "weights": TensorData.from_torch(torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32)),
        "logprobs": TensorData.from_torch(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)),
        "advantages": TensorData.from_torch(torch.tensor([0.0, 0.5, 0.5], dtype=torch.float32)),
    }
)

print("SFT Datum (plain lists) serialized:")
print(json.dumps(sft_datum.model_dump(mode="json"), indent=2))
print()

print("RL Datum (TensorData.from_torch) serialized:")
print(json.dumps(rl_datum.model_dump(mode="json"), indent=2))
