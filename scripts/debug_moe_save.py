#!/usr/bin/env python3
"""Debug MoE save_weights to understand why state_dict might be None."""

import os
import time
import uuid

import requests

os.environ.setdefault("HF_HUB_CACHE", "/vePFS-Mindverse/share/huggingface/hub")

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API = f"{BASE_URL}/api/v1"
MOE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def poll_future(request_id, timeout=300):
    poll_url = f"{API}/retrieve_future"
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.post(poll_url, json={"request_id": request_id}, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            time.sleep(0.5)
            continue
        else:
            resp.raise_for_status()
    raise TimeoutError(f"Operation did not complete within {timeout}s")


def create_session(base_model, lora_rank=32, lr=1e-4):
    session_id = f"debug_{uuid.uuid4().hex[:8]}"
    resp = requests.post(f"{API}/create_model", json={
        "session_id": session_id,
        "model_seq_id": 1,
        "base_model": base_model,
        "lora_config": {"rank": lora_rank},
        "learning_rate": lr,
    }, timeout=300)
    resp.raise_for_status()
    result = poll_future(resp.json().get("request_id"), timeout=300)
    if "error" in result:
        raise RuntimeError(f"Create failed: {result['error']}")
    return session_id, result.get("model_id")


def save_weights_debug(model_id, name="test"):
    """Call save_weights and print full result."""
    resp = requests.post(f"{API}/save_weights", json={"model_id": model_id, "name": name}, timeout=120)
    resp.raise_for_status()
    result = poll_future(resp.json().get("request_id"), timeout=120)
    return result


def main():
    import ray

    # Connect to existing ray cluster
    ray.init(address="auto", ignore_reinit_error=True, namespace="tinker")

    # Check if MegatronWorkerGroup exists
    try:
        actor = ray.get_actor("persistent_megatron_worker_group_v2")
        print(f"MegatronWorkerGroup actor found: {actor}")

        # Check what save_checkpoint returns directly
        import tempfile
        test_path = tempfile.mkdtemp(prefix="debug_save_")
        print(f"\nCalling save_checkpoint directly with path: {test_path}")
        result = ray.get(actor.save_checkpoint.remote(test_path))
        print(f"save_checkpoint result keys: {list(result.keys())}")
        print(f"  state_dict present: {'state_dict' in result}")
        print(f"  peft_config present: {'peft_config' in result}")
        if "state_dict" in result:
            sd = result["state_dict"]
            print(f"  state_dict type: {type(sd)}")
            if sd is not None:
                print(f"  state_dict keys (first 5): {list(sd.keys())[:5]}")
        if "peft_config" in result:
            print(f"  peft_config: {result['peft_config']}")

    except Exception as e:
        print(f"Error checking actor: {e}")

    # Also test via API
    print(f"\n--- Testing via API ---")
    print(f"Creating MoE session...")
    session_id, model_id = create_session(MOE_MODEL)
    print(f"  model_id: {model_id}")

    print(f"\nSaving weights via API...")
    result = save_weights_debug(model_id, name="debug_test")
    print(f"API save_weights result: {result}")

    # Check the LoRA registry
    try:
        from tinker_server.backend.session_manager import InferenceSessionManager
        # This would need to be accessed from within the server process
        print("\n(Cannot access inference_manager from client script)")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
