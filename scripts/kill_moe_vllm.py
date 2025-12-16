#!/usr/bin/env python3
"""Kill the MoE vLLM actor to force code reload."""
import ray

NAMESPACE = "tinker"
MOE_ACTOR_NAME = "tinker_vllm_qwen3-30b-a3b-instruct-2507"

def main():
    ray.init(address="auto", namespace=NAMESPACE, ignore_reinit_error=True)

    try:
        actor = ray.get_actor(MOE_ACTOR_NAME, namespace=NAMESPACE)
        ray.kill(actor)
        print(f"Killed: {MOE_ACTOR_NAME}")
    except ValueError:
        print(f"Not found: {MOE_ACTOR_NAME}")

    # Also kill shared vLLM if exists
    try:
        shared = ray.get_actor("tinker_vllm", namespace=NAMESPACE)
        ray.kill(shared)
        print("Killed: tinker_vllm")
    except ValueError:
        print("Not found: tinker_vllm")

if __name__ == "__main__":
    main()
