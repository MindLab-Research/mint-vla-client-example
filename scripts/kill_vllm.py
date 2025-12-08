#!/usr/bin/env python3
"""Kill the persistent vLLM actor.

Usage:
    python scripts/kill_vllm.py

This kills the detached Ray actor running vLLM, forcing a full reinit (~80s)
on next server start. Use when you need to reload the base model or vLLM
is in a bad state.
"""

import ray

# Must match the constants in multi_lora_engine.py
PERSISTENT_VLLM_ACTOR_NAME = "tinker_vllm_server"
PERSISTENT_NAMESPACE = "tinker"


def main():
    ray.init(address="auto", namespace=PERSISTENT_NAMESPACE, ignore_reinit_error=True)

    try:
        actor = ray.get_actor(PERSISTENT_VLLM_ACTOR_NAME, namespace=PERSISTENT_NAMESPACE)
        ray.kill(actor)
        print(f"Killed vLLM actor: {PERSISTENT_VLLM_ACTOR_NAME}")
    except ValueError:
        print(f"No vLLM actor found: {PERSISTENT_VLLM_ACTOR_NAME}")


if __name__ == "__main__":
    main()
