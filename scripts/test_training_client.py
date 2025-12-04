#!/usr/bin/env python
"""Test tinker-server training using tinker's actual API.

Based on tinker's math_rl example.

Usage:
    export TINKER_BASE_URL=http://localhost:8000
    python scripts/test_training_client.py
"""

import os
import sys
import time

# Add tinker to path if not installed
tinker_path = os.path.join(os.path.dirname(__file__), "../../tinker/src")
if os.path.exists(tinker_path):
    sys.path.insert(0, tinker_path)

# Set default base URL to local server
os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")

import torch
from tinker import types
from tinker.lib.public_interfaces.service_client import ServiceClient
from tinker.types.tensor_data import TensorData
from transformers import AutoTokenizer


def timed(name):
    """Context manager to time operations."""
    class Timer:
        def __init__(self, name):
            self.name = name
            self.elapsed = 0
        def __enter__(self):
            self.start = time.time()
            return self
        def __exit__(self, *args):
            self.elapsed = time.time() - self.start
            print(f"    [{self.name}] took {self.elapsed:.2f}s")
    return Timer(name)


def main():
    print(f"Connecting to: {os.environ.get('TINKER_BASE_URL')}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Create service client
    service_client = ServiceClient(base_url=os.environ.get("TINKER_BASE_URL"))
    print(f"ServiceClient created")

    # ========================================
    # Step 1: Create LoRA training client
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 1: Creating LoRA training client")
    print("=" * 60)

    print("\nCreating training client...")
    training_client = service_client.create_lora_training_client(
        base_model=model_name,
        rank=32,  # LoRA rank
    )
    print("✓ Training client created")

    # ========================================
    # Step 2: Prepare training data
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 2: Preparing training data")
    print("=" * 60)

    # Simple training example
    training_examples = [
        {
            "prompt": "What is 2 + 2?",
            "response": "2 + 2 equals 4.",
        },
        {
            "prompt": "What is the capital of France?",
            "response": "The capital of France is Paris.",
        },
    ]

    print(f"\nPreparing {len(training_examples)} training examples...")

    # Convert to Datum objects (tinker's training format)
    training_datums: list[types.Datum] = []

    for i, example in enumerate(training_examples):
        # Format as chat
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]

        # Apply chat template
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Tokenize
        tokens = tokenizer.encode(formatted, add_special_tokens=True)

        print(f"\n  Example {i + 1}:")
        print(f"    Prompt: {example['prompt']}")
        print(f"    Response: {example['response']}")
        print(f"    Total tokens: {len(tokens)}")

        # Create Datum (tinker's training data format)
        # For supervised learning:
        # - input_tokens: tokens[:-1] (all but last)
        # - target_tokens: tokens[1:] (all but first, shifted by 1)
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]

        # Create loss function inputs
        # For simple supervised learning, we can use uniform weights
        loss_mask = [1.0] * len(target_tokens)  # Train on all tokens

        datum = types.Datum(
            model_input=types.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
                "loss_mask": TensorData.from_torch(torch.tensor(loss_mask, dtype=torch.float32)),
            },
        )
        training_datums.append(datum)

    print(f"\n✓ Prepared {len(training_datums)} training datums")

    # ========================================
    # Step 3: Training loop
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 3: Training loop")
    print("=" * 60)

    num_steps = 3  # Train for 3 steps
    learning_rate = 1e-5

    # Create Adam optimizer parameters
    adam_params = types.AdamParams(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )

    for step in range(num_steps):
        print(f"\n--- Training Step {step + 1}/{num_steps} ---")

        # Forward-backward pass
        print(f"\n  [3a] Computing gradients...")
        
        # Note: tinker uses a specific loss_fn
        # Common options: "cross_entropy", "importance_sampling"
        fwd_bwd_future = training_client.forward_backward(
            training_datums,
            loss_fn="cross_entropy",  # Standard supervised learning
        )

        # Optimizer step
        print(f"  [3b] Updating weights...")
        optim_step_future = training_client.optim_step(adam_params)

        # Wait for results
        fwd_bwd_result = fwd_bwd_future.result()
        optim_result = optim_step_future.result()

        print(f"    ✓ Forward-backward completed")
        print(f"    ✓ Optimizer step completed")

    print("\n Training loop completed")

    # ========================================
    # Step 4: Save weights for inference
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 4: Saving weights for inference")
    print("=" * 60)

    print("\nSaving trained weights...")
    save_future = training_client.save_weights_for_sampler(name="test_checkpoint")
    save_result = save_future.result()

    print(f" Weights saved to: {save_result.path}")

    # ========================================
    # Step 5: Test inference with trained model (Named flow)
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 5: Testing inference with trained model (NAMED FLOW)")
    print("=" * 60)

    print("\nCreating sampling client with trained weights...")
    with timed("create_sampling_client (named)") as t_named:
        sampling_client = service_client.create_sampling_client(
            model_path=save_result.path
        )
    print(f" Sampling client created")

    # Test prompt
    test_prompt = "What is 2 + 2?"
    messages = [{"role": "user", "content": test_prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    model_input = types.ModelInput.from_ints(prompt_tokens)

    print(f"\nTest prompt: {test_prompt}")
    print("Generating response...")

    sampling_params = types.SamplingParams(
        max_tokens=64,
        temperature=0.7,
    )

    sample_future = sampling_client.sample(
        prompt=model_input,
        num_samples=1,
        sampling_params=sampling_params,
    )

    sample_result = sample_future.result()
    response_tokens = sample_result.sequences[0].tokens
    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

    print(f"\nGenerated response:\n{response_text}")

    print("\n Named flow test passed!")

    # ========================================
    # Step 6: Test ephemeral flow (should be FAST after first call)
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 6: Testing EPHEMERAL FLOW (should be fast after first call)")
    print("=" * 60)

    print("\nFirst ephemeral call (initializes shared engine - may be slow)...")
    with timed("save_weights_and_get_sampling_client #1 (init shared engine)") as t_eph1:
        ephemeral_sampling_client = training_client.save_weights_and_get_sampling_client()
    print(f" Ephemeral sampling client created")

    # Test with same prompt
    print(f"\nTest prompt: {test_prompt}")
    print("Generating response with ephemeral weights...")

    ephemeral_sample_future = ephemeral_sampling_client.sample(
        prompt=model_input,
        num_samples=1,
        sampling_params=sampling_params,
    )

    ephemeral_sample_result = ephemeral_sample_future.result()
    ephemeral_response_tokens = ephemeral_sample_result.sequences[0].tokens
    ephemeral_response_text = tokenizer.decode(ephemeral_response_tokens, skip_special_tokens=True)

    print(f"\nGenerated response (ephemeral):\n{ephemeral_response_text}")

    print("\n Ephemeral flow test passed!")

    # ========================================
    # Step 6b: Second ephemeral call (should be FAST - hot reload only)
    # ========================================
    print("\n" + "-" * 60)
    print("STEP 6b: Second ephemeral call (should be FAST - hot reload)")
    print("-" * 60)

    print("\nSecond ephemeral call (reuses shared engine, hot-reload only)...")
    with timed("save_weights_and_get_sampling_client #2 (hot reload)") as t_eph2:
        ephemeral_sampling_client2 = training_client.save_weights_and_get_sampling_client()
    print(f" Second ephemeral sampling client created")

    # Compare times
    print(f"\n" + "=" * 60)
    print("TIMING COMPARISON:")
    print(f"  Named flow (create_sampling_client):    {t_named.elapsed:.2f}s")
    print(f"  Ephemeral #1 (init shared engine):      {t_eph1.elapsed:.2f}s")
    print(f"  Ephemeral #2 (hot reload only):         {t_eph2.elapsed:.2f}s")
    if t_eph1.elapsed > 0:
        speedup = t_eph1.elapsed / t_eph2.elapsed if t_eph2.elapsed > 0 else float('inf')
        print(f"  Speedup (eph#1 / eph#2):                {speedup:.1f}x")
    print("=" * 60)

    # ========================================
    # Step 7: Cleanup
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 7: Cleanup")
    print("=" * 60)

    # Note: Depending on tinker's API, you might need to delete the training client
    # training_client.delete() might not exist
    print("\n Test completed (cleanup not implemented)")

    print("\n" + "=" * 60)
    print("Training test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()