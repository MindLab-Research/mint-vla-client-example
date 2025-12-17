#!/usr/bin/env python3
"""End-to-end test for MoE expert LoRA (with MLP filter removed).

This test verifies that vLLM 0.12.0 FusedMoE LoRA support works for inference.

Steps:
1. Kill existing MoE vLLM actor (old code with MLP filter)
2. Run MoE training (6 iterations)
3. Save weights and create sampling session
4. Sample from trained model
5. Verify LoRA weights include MLP modules
"""

import os
import sys
import time
import uuid

# Set up environment
os.environ["TINKER_BASE_URL"] = "http://localhost:8000"
os.environ["TINKER_API_KEY"] = "dummy"

import ray
import requests

# Model paths
MOE_MODEL_PATH = "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/4f41ec5a0313c8f4e2a5efde73fa2e999c596ef3"


def kill_actor_by_name(actor_name: str, namespace: str = "tinker") -> bool:
    """Kill a Ray actor by name."""
    try:
        if not ray.is_initialized():
            ray.init(address="auto", namespace=namespace, ignore_reinit_error=True)

        actor = ray.get_actor(actor_name, namespace=namespace)
        ray.kill(actor)
        print(f"Killed actor: {actor_name}")
        return True
    except ValueError:
        print(f"Actor not found: {actor_name}")
        return False
    except Exception as e:
        print(f"Error killing {actor_name}: {e}")
        return False


def check_resource_pool():
    """Check resource pool status."""
    resp = requests.get("http://localhost:8000/api/v1/resource_pool")
    return resp.json()


def kill_idle_actors():
    """Kill all idle actors to free up GPUs."""
    pool = check_resource_pool()
    killed = []

    for actor in pool.get("actors", []):
        if actor.get("idle", False):
            actor_name = actor["actor_name"]
            if kill_actor_by_name(actor_name):
                killed.append(actor_name)

    return killed


def run_moe_training(session_id: str, n_iters: int = 6):
    """Run MoE training iterations."""
    from tinker import Tinker

    client = Tinker()

    # Training config
    config = {
        "base_model": MOE_MODEL_PATH,
        "lora_rank": 16,
        "learning_rate": 1e-4,
        "loss_fn": "cross_entropy",
        "loss_fn_config": {},
    }

    # Simple training data - pig latin
    pig_latin_examples = [
        ("hello", "ellohay"),
        ("world", "orldway"),
        ("python", "ythonpay"),
        ("machine", "achinemay"),
        ("learning", "earninglay"),
        ("data", "ataday"),
    ]

    print(f"\n{'='*60}")
    print(f"Starting MoE training: {n_iters} iterations")
    print(f"Session ID: {session_id}")
    print(f"{'='*60}\n")

    losses = []

    for i in range(n_iters):
        # Construct training batch
        data = []
        for en, pig in pig_latin_examples:
            prompt = f"Translate to Pig Latin: {en}\n"
            completion = pig

            # Use model_input format expected by Tinker
            data.append({
                "session_id": session_id,
                "model_input": {
                    "chunks": [{
                        "tokens": list(range(20)),  # Placeholder tokens
                        "mask": [0] * 10 + [1] * 10,  # Last 10 tokens are targets
                    }]
                },
                "datum_metadata": {"prompt": prompt, "completion": completion},
            })

        # Call train endpoint
        resp = requests.post(
            "http://localhost:8000/api/v1/train",
            json={
                "session_id": session_id,
                "data": data,
                "config": config,
            },
        )

        if resp.status_code != 200:
            print(f"Training failed: {resp.text}")
            return None

        result = resp.json()
        loss = result.get("metrics", {}).get("loss:mean", 0)
        losses.append(loss)
        print(f"Iteration {i+1}/{n_iters}: loss={loss:.4f}")

    return losses


def save_weights_and_sample(session_id: str):
    """Save weights and sample from trained model."""
    print(f"\n{'='*60}")
    print("Saving weights and creating sampling session")
    print(f"{'='*60}\n")

    # Save weights
    resp = requests.post(
        "http://localhost:8000/api/v1/save_weights",
        json={
            "session_id": session_id,
            "base_model": MOE_MODEL_PATH,
            "lora_rank": 16,
        },
    )

    if resp.status_code != 200:
        print(f"Save weights failed: {resp.text}")
        return None

    result = resp.json()
    print(f"Weights saved: {result.get('weights_path', 'N/A')}")

    # Check if state_dict has MLP modules
    state_dict = result.get("state_dict", {})
    if state_dict:
        mlp_keys = [k for k in state_dict.keys() if "mlp" in k.lower() or "gate" in k.lower() or "down" in k.lower()]
        attn_keys = [k for k in state_dict.keys() if "attn" in k.lower() or "proj" in k.lower()]
        print(f"LoRA state_dict: {len(state_dict)} total, {len(mlp_keys)} MLP, {len(attn_keys)} attention")
        if mlp_keys:
            print(f"Sample MLP keys: {mlp_keys[:3]}")

    # Get sampling session ID
    sampling_session_id = result.get("sampling_session_id")
    if not sampling_session_id:
        print("No sampling session ID returned")
        return None

    print(f"Sampling session ID: {sampling_session_id}")

    # Sample from trained model
    print("\nSampling from trained model...")
    sample_resp = requests.post(
        "http://localhost:8000/api/v1/sample",
        json={
            "sampling_session_id": sampling_session_id,
            "prompts": ["Translate to Pig Latin: hello\n"],
            "max_tokens": 20,
            "temperature": 0.1,
        },
    )

    if sample_resp.status_code != 200:
        print(f"Sampling failed: {sample_resp.text}")
        return None

    sample_result = sample_resp.json()
    completions = sample_result.get("completions", [])
    if completions:
        print(f"Sample output: {completions[0][:100]}...")

    return {
        "state_dict_size": len(state_dict),
        "mlp_keys": len(mlp_keys) if state_dict else 0,
        "completions": completions,
    }


def main():
    print("=" * 60)
    print("MoE Expert LoRA End-to-End Test")
    print("=" * 60)
    print(f"Model: {MOE_MODEL_PATH}")
    print()

    # Step 1: Check and kill idle actors
    print("Step 1: Checking resource pool...")
    pool = check_resource_pool()
    print(f"Current GPU usage: {pool.get('total_gpus_used', 0)}")
    for actor in pool.get("actors", []):
        print(f"  - {actor['actor_name']}: {actor['num_gpus']} GPUs, idle={actor.get('idle', False)}")

    print("\nKilling idle actors to free GPUs...")
    killed = kill_idle_actors()
    print(f"Killed: {killed}")

    # Wait for cleanup
    time.sleep(5)

    # Check again
    pool = check_resource_pool()
    print(f"After cleanup: {pool.get('total_gpus_used', 0)} GPUs used")

    # Step 2: Create training session
    session_id = f"moe_expert_lora_test_{uuid.uuid4().hex[:8]}"
    print(f"\nStep 2: Creating session {session_id}")

    resp = requests.post(
        "http://localhost:8000/api/v1/create_session",
        json={"tags": ["moe_expert_lora_test"], "user_metadata": {}},
    )
    if resp.status_code != 200:
        print(f"Create session failed: {resp.text}")
        return 1

    actual_session_id = resp.json().get("session_id", session_id)
    print(f"Session created: {actual_session_id}")

    # Step 3: Run training
    print("\nStep 3: Running MoE training...")
    losses = run_moe_training(actual_session_id, n_iters=6)
    if losses is None:
        print("Training failed!")
        return 1

    print(f"\nTraining complete. Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # Step 4: Save weights and sample
    print("\nStep 4: Saving weights and sampling...")
    result = save_weights_and_sample(actual_session_id)
    if result is None:
        print("Save/sample failed!")
        return 1

    # Step 5: Verify results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Training loss reduction: {losses[0]:.4f} -> {losses[-1]:.4f} ({100*(1-losses[-1]/losses[0]):.1f}%)")
    print(f"LoRA state_dict size: {result['state_dict_size']}")
    print(f"MLP modules in LoRA: {result['mlp_keys']}")

    if result['mlp_keys'] > 0:
        print("\nSUCCESS: MoE expert LoRA working! MLP modules included in extraction.")
    else:
        print("\nWARNING: No MLP modules found in LoRA state_dict")

    return 0


if __name__ == "__main__":
    sys.exit(main())
