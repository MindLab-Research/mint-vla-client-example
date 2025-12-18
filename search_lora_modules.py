import ray

@ray.remote
def find_lora_modules():
    import subprocess
    # Search for supported_lora_modules in vLLM
    result = subprocess.run(
        ["grep", "-r", "supported_lora_modules", "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/"],
        capture_output=True, text=True
    )

    # Also look for qwen3_moe.py content
    qwen3_moe = subprocess.run(
        ["cat", "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/qwen3_moe.py"],
        capture_output=True, text=True
    )

    # Check qwen2_moe.py too
    qwen2_moe = subprocess.run(
        ["cat", "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/qwen2_moe.py"],
        capture_output=True, text=True
    )

    return {
        "grep_results": result.stdout + result.stderr,
        "qwen3_moe": qwen3_moe.stdout[:10000],
        "qwen2_moe": qwen2_moe.stdout[:10000]
    }

ray.init(address="auto", ignore_reinit_error=True)
result = ray.get(find_lora_modules.remote())
print("=== supported_lora_modules grep ===")
print(result["grep_results"][:5000])
print("\n=== qwen3_moe.py (first 10000 chars) ===")
print(result["qwen3_moe"])
print("\n=== qwen2_moe.py (first 10000 chars) ===")
print(result["qwen2_moe"])
