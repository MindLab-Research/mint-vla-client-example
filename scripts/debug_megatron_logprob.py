#!/usr/bin/env python3
"""Debug script to compare vLLM and Megatron logprobs directly.

This isolates whether the issue is in:
1. LoRA weights (both should have same)
2. Forward pass computation
3. Token alignment
4. Something else
"""

import asyncio
import os
import sys

# Setup
os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
from transformers import AutoTokenizer


async def main():
    model_name = "moonshotai/Moonlight-16B-A3B-Instruct"

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Create a simple test sequence
    prompt = "What is 2 + 2?"
    response = "<think>\nLet me calculate: 2 + 2 = 4\n</think>\n<answer>4</answer>"
    full_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"

    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    print(f"\nTest sequence: {len(tokens)} tokens")
    print(f"Text: {full_text[:100]}...")

    # Find where response starts (action tokens)
    prompt_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    action_start = len(prompt_tokens)
    print(f"Action starts at position: {action_start}")

    # Show tokens around action start
    print("\nTokens around action start:")
    for i in range(max(0, action_start-3), min(len(tokens), action_start+10)):
        tok = tokens[i]
        text = tokenizer.decode([tok])
        marker = " <-- ACTION START" if i == action_start else ""
        print(f"  pos[{i}]: token={tok}, text={repr(text)}{marker}")

    # Create training session
    print("\n--- Creating training session ---")
    client = tinker.AsyncTinker()

    session = await client.training.create_session(
        model_path=model_name,
        lora_rank=16,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    print(f"Session: {session.session_id}")

    training_client = session.get_training_client()

    # Get Megatron logprobs for the sequence (no training, just forward pass)
    print("\n--- Getting Megatron logprobs (forward only) ---")

    # Create a minimal batch
    import torch
    from tensordict import TensorDict

    input_ids = torch.tensor([tokens], dtype=torch.long)

    # We need to use the actual training API to get logprobs
    # Let's check what the forward_backward returns

    # Actually, let's just run a single forward pass through the training client
    # The train step computes logprobs as part of the loss computation

    # For now, let's just sample from the model and compare the logprobs
    print("\n--- Getting vLLM logprobs via sampling ---")

    # Create sampling session
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    print(f"Sampling client ready")

    # Sample with logprobs
    result = await sampling_client.asample(
        prompt=prompt_text,
        n=1,
        max_tokens=50,
        temperature=1.0,
        logprobs=True,
        top_logprobs=5,
    )

    print("\n--- vLLM sampling result ---")
    generated = result.completions[0].text
    print(f"Generated: {generated[:100]}...")

    if result.completions[0].logprobs:
        print("\nvLLM logprobs for first 20 tokens:")
        for i, lp in enumerate(result.completions[0].logprobs[:20]):
            if lp and lp.token:
                print(f"  pos[{i}]: token={lp.token}, logprob={lp.logprob:.4f}, text={repr(tokenizer.decode([lp.token]))}")

    # Now we need to get Megatron's logprobs for the SAME sequence
    # The issue is that Megatron computes logprobs during train_step
    # Let's create a batch with reward=0 (no gradient update) just to get logprobs

    print("\n--- Getting Megatron logprobs via train_step ---")

    # Prepare the exact sequence that was generated
    full_sequence = prompt_text + generated
    full_tokens = tokenizer.encode(full_sequence, add_special_tokens=False)

    print(f"Full sequence: {len(full_tokens)} tokens")

    # Create training batch
    # This is a hack - we're using train_step just to get logprobs
    # The reward is 0 so no actual learning happens

    from tinker_cookbook.rl.trajectory import TrajectoryStep, Trajectory, TrajectoryGroup
    from tinker_cookbook.rl.problem_env import create_trajectory_group

    # Create a single trajectory
    trajectory = Trajectory(
        datum={"prompt": prompt_text},
        steps=[TrajectoryStep(
            observation=prompt_text,
            action=generated,
            reward=0.0,
            metrics={"format": 1.0, "correct": 0.0},
        )],
        ob_token_ids=tokenizer.encode(prompt_text, add_special_tokens=False),
    )

    group = TrajectoryGroup(
        trajectories=[trajectory],
        datum={"prompt": prompt_text},
    )

    print(f"Created trajectory group with 1 trajectory")

    # Use the actual training code path to get logprobs
    # This requires importing from tinker_cookbook

    from tinker_cookbook.rl.train import do_train_step_and_get_sampling_client
    from tinker_cookbook.rl.tokenize_trajectories import tokenize_groups
    from tinker_cookbook.rl.trajectory import collect_sampling_logprobs

    # Tokenize
    tokenized = tokenize_groups(
        groups=[group],
        tokenizer=tokenizer,
        max_tokens=8192,
    )

    print(f"Tokenized: {len(tokenized)} groups")

    # Get sampling logprobs from vLLM
    if result.completions[0].logprobs:
        sampling_logprobs = [lp.logprob for lp in result.completions[0].logprobs if lp]
        print(f"vLLM logprobs: {sampling_logprobs[:10]}...")

    # The key question: what does Megatron compute for these same tokens?
    # We need to trace through the actual forward pass

    print("\n" + "="*60)
    print("KEY QUESTION: Why does Megatron give ~-17 logprob when vLLM gives ~-0.2?")
    print("="*60)
    print("\nPossible causes:")
    print("1. Token indexing mismatch (Megatron computing logprob for wrong token)")
    print("2. LoRA not applied during Megatron forward pass")
    print("3. Vocab parallel gathering bug")
    print("4. Label shift still wrong")
    print("\nNext step: Add debug logging in verl's forward pass to see:")
    print("- What token IDs are being processed")
    print("- What logits Megatron produces")
    print("- What labels are being used for logprob extraction")


if __name__ == "__main__":
    asyncio.run(main())
