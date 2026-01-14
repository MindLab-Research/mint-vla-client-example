#!/usr/bin/env python3
"""K2 RL Training Loop - 12+ hour continuous run.

Full RL loop: train -> save_weights -> vLLM sample -> repeat

K2 configuration:
- Training: 64 GPUs (TP=8, EP=8) with Megatron
- Inference: 16 GPUs (TP=16) with vLLM + LoRA
- Resource eviction: Megatron evicted when vLLM starts
"""

import os
import sys
import time
import json
import uuid
from datetime import datetime

import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")
K2_MODEL = "moonshotai/Kimi-K2-Thinking"
# Local path for tokenizer (server has no internet)
K2_TOKENIZER_PATH = "/vePFS-Mindverse/share/huggingface/hub/models--moonshotai--Kimi-K2-Thinking/snapshots/612681931a8c906ddb349f8ad0f582cb552189cd"

LOG_FILE = "/tmp/k2_rl_12hr.jsonl"

def get_headers():
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def log_metrics(step: int, phase: str, metrics: dict, elapsed: float):
    """Append metrics to JSONL log."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "step": step,
        "phase": phase,
        "elapsed_sec": elapsed,
        **metrics
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[Step {step}] {phase}: {metrics} ({elapsed:.1f}s)")


def poll_future(request_id: str, timeout: int = 600) -> dict:
    """Poll for async operation result."""
    poll_url = f"{BASE_URL}/api/v1/retrieve_future"
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.post(poll_url, json={"request_id": request_id}, headers=get_headers(), timeout=300)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            time.sleep(2)
            continue
        else:
            return {"error": f"HTTP {resp.status_code}: {resp.text}"}
    return {"error": f"Timeout after {timeout}s"}


def create_training_model(session_id: str, model_seq_id: int, lr: float = 1e-4) -> str:
    """Create training model, returns model_id."""
    resp = requests.post(
        f"{BASE_URL}/api/v1/create_model",
        json={
            "session_id": session_id,
            "model_seq_id": model_seq_id,
            "base_model": K2_MODEL,
            "lora_config": {"rank": 32},
            "learning_rate": lr,
        },
        headers=get_headers(),
        timeout=300,  # K2 model creation may be slow
    )
    if resp.status_code != 200:
        raise RuntimeError(f"create_model failed: {resp.status_code}: {resp.text}")

    request_id = resp.json().get("request_id")
    result = poll_future(request_id, timeout=900)
    if "error" in result:
        raise RuntimeError(f"create_model poll failed: {result['error']}")

    return result.get("model_id")


def train_step(model_id: str, data: list, lr: float = 1e-4) -> dict:
    """Run one training step using combined forward-backward + optim step API."""
    resp = requests.post(
        f"{BASE_URL}/api/v1/train_step",
        json={
            "model_id": model_id,
            "forward_backward_input": {
                "data": data,
                "loss_fn": "cross_entropy",
            },
            "adam_params": {
                "learning_rate": lr,
                "beta1": 0.9,
                "beta2": 0.95,
                "eps": 1e-12,
            },
        },
        headers=get_headers(),
        timeout=300,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"train_step failed: {resp.status_code}: {resp.text}")

    # Synchronous endpoint - return directly
    if "metrics" in resp.json():
        return resp.json()

    # Async endpoint - poll for result
    request_id = resp.json().get("request_id")
    if request_id:
        result = poll_future(request_id, timeout=600)
        if "error" in result:
            raise RuntimeError(f"train_step poll failed: {result['error']}")
        return result
    return resp.json()


def save_weights_for_sampler(model_id: str) -> str:
    """Save weights and get sampling session ID."""
    resp = requests.post(
        f"{BASE_URL}/api/v1/save_weights_for_sampler",
        json={"model_id": model_id},
        headers=get_headers(),
        timeout=300,  # Increased for K2 vLLM init
    )
    if resp.status_code != 200:
        raise RuntimeError(f"save_weights_for_sampler failed: {resp.status_code}: {resp.text}")

    request_id = resp.json().get("request_id")
    result = poll_future(request_id, timeout=2400)  # vLLM init can take 20+ min for K2
    if "error" in result:
        raise RuntimeError(f"save_weights poll failed: {result['error']}")

    return result.get("sampling_session_id")


def sample(sampling_session_id: str, prompt_tokens: list, max_tokens: int = 100) -> dict:
    """Sample from model."""
    resp = requests.post(
        f"{BASE_URL}/api/v1/asample",
        json={
            "sampling_session_id": sampling_session_id,
            "prompt": {"chunks": [{"tokens": prompt_tokens, "type": "encoded_text"}]},
            "sampling_params": {"max_tokens": max_tokens, "temperature": 0.7},
            "num_samples": 1,
        },
        headers=get_headers(),
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"asample failed: {resp.status_code}: {resp.text}")

    request_id = resp.json().get("request_id")
    return poll_future(request_id, timeout=300)


def prepare_training_data(tokenizer, prompts_and_targets: list) -> list:
    """Prepare training data from prompt-target pairs."""
    data = []
    for prompt, target in prompts_and_targets:
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        target_tokens = tokenizer.encode(target, add_special_tokens=False)
        full_tokens = prompt_tokens + target_tokens
        loss_mask = [0.0] * len(prompt_tokens) + [1.0] * len(target_tokens)

        data.append({
            "model_input": {"chunks": [{"tokens": full_tokens[:-1], "type": "encoded_text"}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": full_tokens[1:], "shape": [len(full_tokens) - 1], "dtype": "int64"},
                "loss_mask": {"data": loss_mask[1:], "shape": [len(loss_mask) - 1], "dtype": "float32"},
            },
        })
    return data


def main():
    import sys
    print("=" * 70, flush=True)
    print("K2 RL Training Loop - 12+ Hour Run", flush=True)
    print("=" * 70, flush=True)
    print(f"Model: {K2_MODEL}", flush=True)
    print(f"Training: 64 GPUs (TP=8, EP=8)", flush=True)
    print(f"Inference: 16 GPUs (TP=16) with LoRA", flush=True)
    print(f"Log file: {LOG_FILE}", flush=True)
    print(flush=True)

    # Load tokenizer from local path (server has no internet)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(K2_TOKENIZER_PATH, trust_remote_code=True, local_files_only=True)

    # Training data (simple math problems)
    prompts_and_targets = [
        ("Q: What is 2+2?\nA: ", "4"),
        ("Q: What is 3*4?\nA: ", "12"),
        ("Q: What is 10-7?\nA: ", "3"),
        ("Q: What is 15/3?\nA: ", "5"),
        ("Q: What is 8+9?\nA: ", "17"),
    ]
    training_data = prepare_training_data(tokenizer, prompts_and_targets)

    # Sample prompts
    sample_prompts = [
        "Q: What is 5+5?\nA:",
        "Q: What is 6*7?\nA:",
        "Q: What is 20-8?\nA:",
    ]
    sample_prompt_tokens = [tokenizer.encode(p, add_special_tokens=True) for p in sample_prompts]

    # Session setup
    session_id = f"k2_rl_{uuid.uuid4().hex[:8]}"
    model_seq_id = 0
    run_duration = 12 * 3600  # 12 hours in seconds
    start_time = time.time()

    print(f"Session: {session_id}")
    print(f"Target duration: {run_duration/3600:.1f} hours")
    print()

    step = 0
    losses = []

    while time.time() - start_time < run_duration:
        loop_start = time.time()
        model_seq_id += 1

        print(f"\n{'='*70}")
        print(f"RL Loop Step {step} (elapsed: {(time.time()-start_time)/3600:.2f}h)")
        print(f"{'='*70}")

        try:
            # Phase 1: Create training model
            t0 = time.time()
            print(f"Creating training model (seq={model_seq_id})...")
            model_id = create_training_model(session_id, model_seq_id, lr=1e-4)
            log_metrics(step, "create_model", {"model_id": model_id}, time.time() - t0)

            # Phase 2: Train (multiple steps per RL iteration)
            train_losses = []
            for i in range(3):
                t0 = time.time()
                result = train_step(model_id, training_data, lr=1e-4)
                loss = result.get("metrics", {}).get("loss:mean", 0)
                grad_norm = result.get("metrics", {}).get("grad_norm:mean", 0)
                train_losses.append(loss)
                log_metrics(step, f"train_{i}", {"loss": loss, "grad_norm": grad_norm}, time.time() - t0)

            losses.append(train_losses[-1])

            # Phase 3: Save weights for inference
            t0 = time.time()
            print("Saving weights for sampler (vLLM init)...")
            sampling_session_id = save_weights_for_sampler(model_id)
            log_metrics(step, "save_weights", {"sampling_session_id": sampling_session_id}, time.time() - t0)

            # Phase 4: Sample
            for j, prompt_tokens in enumerate(sample_prompt_tokens):
                t0 = time.time()
                result = sample(sampling_session_id, prompt_tokens, max_tokens=50)
                if "error" not in result:
                    sequences = result.get("sequences", [])
                    if sequences:
                        tokens = sequences[0].get("tokens", [])
                        text = tokenizer.decode(tokens, skip_special_tokens=True)
                        log_metrics(step, f"sample_{j}", {"prompt": sample_prompts[j], "output": text[:100]}, time.time() - t0)
                else:
                    log_metrics(step, f"sample_{j}", {"error": result["error"]}, time.time() - t0)

            # Summary
            loop_time = time.time() - loop_start
            log_metrics(step, "loop_complete", {
                "loop_time_sec": loop_time,
                "total_elapsed_hr": (time.time() - start_time) / 3600,
                "final_loss": train_losses[-1],
            }, loop_time)

            step += 1

        except Exception as e:
            log_metrics(step, "error", {"exception": str(e)}, time.time() - loop_start)
            import traceback
            traceback.print_exc()
            # Continue to next iteration after short delay
            time.sleep(60)
            step += 1

    # Final summary
    print("\n" + "=" * 70)
    print("K2 RL 12-Hour Run Complete")
    print("=" * 70)
    print(f"Total steps: {step}")
    print(f"Total time: {(time.time() - start_time) / 3600:.2f} hours")
    if losses:
        print(f"Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
        print(f"Loss reduction: {(1 - losses[-1]/losses[0])*100:.1f}%")
    print(f"Log file: {LOG_FILE}")


if __name__ == "__main__":
    main()
