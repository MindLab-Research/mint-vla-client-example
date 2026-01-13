#!/usr/bin/env python3
"""Get raw logits from HuggingFace model with LoRA for comparison.

This bypasses vLLM and loads the model directly via transformers.
Runs on volcano server via Ray task (needs GPU).

Usage (on volcano):
    python scripts/get_hf_raw_logits.py
"""
import os
import sys
import json
import torch
import ray

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"
CHECKPOINT_PATH = "/vePFS-Mindverse/share/code/tinker-server/checkpoints/80e6bf97-56d9-4f3a-9872-7cc4b80a7379_0/debug_checkpoint_20260110_182006/"
DUMP_PATH = "/vePFS-Mindverse/share/code/hf_raw_logits.pt"
MEGATRON_DUMP = "/vePFS-Mindverse/share/code/logits_processor_input.pt"

# Same test text as Megatron comparison
TEST_TEXT = """<|im_start|>user
Count down from 10 to 1, one number per line.<|im_end|>
<|im_start|>assistant
10
9
8
7
6
5
4
3
2
1<|im_end|>"""


@ray.remote(num_gpus=1)
def get_hf_logits(model_name: str, lora_path: str, test_text: str, dump_path: str) -> dict:
    """Load model with LoRA and get raw logits."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Tokenizing input...")
    tokens = tokenizer.encode(test_text, add_special_tokens=False)
    input_tokens = tokens[:-1]  # Remove last token (we predict it)
    print(f"Input tokens ({len(input_tokens)}): {input_tokens[:20]}...")

    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Check what's in the LoRA checkpoint
    print(f"\nLoRA checkpoint contents: {lora_path}")
    import os
    lora_files = os.listdir(lora_path)
    print(f"Files: {lora_files}")

    # Check if it's a standard LoRA format
    if "adapter_config.json" in lora_files:
        print("Found adapter_config.json - loading as PEFT LoRA")
        model = PeftModel.from_pretrained(model, lora_path)
    elif any(f.endswith('.safetensors') for f in lora_files):
        print("Found safetensors - checking format...")
        from safetensors import safe_open
        st_files = [f for f in lora_files if f.endswith('.safetensors')]
        with safe_open(os.path.join(lora_path, st_files[0]), framework="pt") as f:
            keys = list(f.keys())
            print(f"Keys in safetensors: {keys[:10]}...")
        # May need custom loading logic
    else:
        print(f"Unknown checkpoint format. Files: {lora_files}")

    model.eval()

    print(f"\nRunning forward pass...")
    input_ids = torch.tensor([input_tokens], device=model.device)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=False)
        logits = outputs.logits  # Shape: [1, seq_len, vocab_size]

    print(f"Logits shape: {logits.shape}")

    # Get top-k predictions at each position
    topk = 10
    topk_values, topk_indices = torch.topk(logits[0], topk, dim=-1)

    # Decode tokens for interpretability
    results = {
        "input_tokens": input_tokens,
        "target_tokens": tokens[1:],  # What should be predicted
        "logits_shape": list(logits.shape),
        "positions": [],
    }

    for pos in range(len(input_tokens)):
        target_token = tokens[pos + 1]
        pos_logits = logits[0, pos]  # [vocab_size]

        # Get logit for target token
        target_logit = pos_logits[target_token].item()
        max_logit = pos_logits.max().item()
        argmax_token = pos_logits.argmax().item()

        # Top-k
        topk_tokens = topk_indices[pos].tolist()
        topk_logits = topk_values[pos].tolist()

        pos_info = {
            "position": pos,
            "target_token": target_token,
            "target_text": tokenizer.decode([target_token]),
            "target_logit": target_logit,
            "max_logit": max_logit,
            "argmax_token": argmax_token,
            "argmax_text": tokenizer.decode([argmax_token]),
            "correct": argmax_token == target_token,
            "topk_tokens": topk_tokens,
            "topk_logits": topk_logits,
            "topk_texts": [tokenizer.decode([t]) for t in topk_tokens],
        }
        results["positions"].append(pos_info)

        # Print problematic positions
        if not pos_info["correct"]:
            print(f"Pos {pos}: target='{pos_info['target_text']}' ({target_token}), "
                  f"predicted='{pos_info['argmax_text']}' ({argmax_token}), "
                  f"target_logit={target_logit:.2f}, max_logit={max_logit:.2f}")

    # Save full logits tensor
    torch.save({
        "logits": logits.cpu(),
        "input_tokens": input_tokens,
        "target_tokens": tokens[1:],
        "results": results,
    }, dump_path)
    print(f"\nSaved to {dump_path}")

    return results


def main():
    print("=" * 60)
    print("HuggingFace Raw Logits Extraction")
    print("=" * 60)

    ray.init(address="auto", namespace="tinker", ignore_reinit_error=True)

    print(f"\nModel: {MODEL_NAME}")
    print(f"LoRA: {CHECKPOINT_PATH}")
    print(f"Output: {DUMP_PATH}")

    result = ray.get(get_hf_logits.remote(MODEL_NAME, CHECKPOINT_PATH, TEST_TEXT, DUMP_PATH))

    # Print summary
    print("\n" + "=" * 60)
    print("Summary of Predictions")
    print("=" * 60)

    correct = sum(1 for p in result["positions"] if p["correct"])
    total = len(result["positions"])
    print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")

    print("\nIncorrect positions:")
    for p in result["positions"]:
        if not p["correct"]:
            print(f"  Pos {p['position']}: target='{p['target_text']}' -> predicted='{p['argmax_text']}'")


if __name__ == "__main__":
    main()
