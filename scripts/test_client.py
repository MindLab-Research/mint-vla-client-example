#!/usr/bin/env python
"""Test tinker-server using the tinker SamplingClient API.

Usage:
    # Set the server URL (defaults to localhost:8000)
    export TINKER_BASE_URL=http://localhost:8000

    python scripts/test_client.py
"""

import os
import sys

# Add tinker to path if not installed
tinker_path = os.path.join(os.path.dirname(__file__), "../../tinker/src")
if os.path.exists(tinker_path):
    sys.path.insert(0, tinker_path)

# Set default base URL to local server
os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")

from tinker import types
from tinker.lib.public_interfaces.service_client import ServiceClient


def main():
    print(f"Connecting to: {os.environ.get('TINKER_BASE_URL')}")

    # Test model - uses same model for tokenizer and sampling
    base_model = "Qwen/Qwen2.5-7B-Instruct"

    # Load tokenizer for decoding
    print("\nLoading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Create service client
    client = ServiceClient()
    print(f"ServiceClient created, session_id: {client.holder._session_id}")

    # Create sampling client
    print("\nCreating SamplingClient...")
    sampling_client = client.create_sampling_client(
        base_model=base_model
    )
    print(f"SamplingClient created, sampling_session_id: {sampling_client._sampling_session_id}")

    # Prepare prompt using chat template
    prompt_text = "What is the capital of France?"
    messages = [
        {"role": "user", "content": prompt_text}
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    prompt_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    prompt = types.ModelInput.from_ints(prompt_tokens)

    print(f"\n{'='*60}")
    print(f"PROMPT ({len(prompt_tokens)} tokens):")
    print(f"{'='*60}")
    print(f"User message: {prompt_text}")
    print(f"\nFormatted with chat template:")
    print(formatted_prompt)
    print(f"\nToken IDs: {prompt_tokens[:20]}{'...' if len(prompt_tokens) > 20 else ''}")

    # Sampling parameters
    params = types.SamplingParams(
        max_tokens=64,
        temperature=0.7,
        top_p=0.9,
    )

    # Generate sample
    print(f"\n{'='*60}")
    print("GENERATING...")
    print(f"{'='*60}")
    future = sampling_client.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=params,
    )

    # Wait for result
    result = future.result(timeout=60)

    # Print results
    print(f"\n{'='*60}")
    print(f"RESPONSE ({len(result.sequences)} sequence(s)):")
    print(f"{'='*60}")

    # EOS token diagnostic
    eos_token_id = tokenizer.eos_token_id
    print(f"\nEOS token ID: {eos_token_id}")

    for i, seq in enumerate(result.sequences):
        response_text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
        print(f"\nSequence {i}:")
        print(f"  Length: {len(seq.tokens)} tokens")
        print(f"  Stop reason: {seq.stop_reason}")

        # Check if EOS is in the sequence
        if eos_token_id in seq.tokens:
            eos_position = seq.tokens.index(eos_token_id)
            print(f"  EOS found at position: {eos_position}")
        else:
            print(f"  EOS NOT found in {len(seq.tokens)} generated tokens")

        print(f"\n  First 20 token IDs: {seq.tokens[:20]}")
        print(f"  Last 20 token IDs: {seq.tokens[-20:]}")
        print(f"\n  Text:\n{response_text}")
        if seq.logprobs:
            print(f"\n  Logprobs (first 5): {[f'{lp:.6g}' for lp in seq.logprobs[:5]]}")

    print(f"\n{'='*60}")
    print("Test passed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
