#!/usr/bin/env python3
"""Patch system vLLM to dump raw logits on demand.

This patches /usr/local/lib/python3.10/dist-packages/vllm/v1/worker/gpu/model_runner.py
on GPU nodes to check for a dump request file and save raw logits.

Usage:
    python scripts/patch_vllm_logits_dump.py --check    # Check if patched
    python scripts/patch_vllm_logits_dump.py --patch    # Apply patch
    python scripts/patch_vllm_logits_dump.py --unpatch  # Remove patch
"""
import ray
import sys

MODEL_RUNNER_PATH = "/usr/local/lib/python3.10/dist-packages/vllm/v1/worker/gpu/model_runner.py"

# Patch code to add at module level
PATCH_MODULE_CODE = '''
# === RAW LOGITS DUMP PATCH START ===
_DUMP_LOGITS_REQUEST = "/vePFS-Mindverse/share/code/vllm_logits_dump_request.json"
_DUMP_LOGITS_OUTPUT = "/vePFS-Mindverse/share/code/vllm_raw_logits.pt"

def _maybe_dump_raw_logits(logits, extra_info=None):
    """Check if dump requested and save raw logits."""
    import os
    if not os.path.exists(_DUMP_LOGITS_REQUEST):
        return
    try:
        import json
        import torch
        with open(_DUMP_LOGITS_REQUEST) as f:
            config = json.load(f)
        os.remove(_DUMP_LOGITS_REQUEST)  # Only dump once
        save_data = {
            "logits": logits.cpu().float(),
            "shape": list(logits.shape),
        }
        if extra_info:
            save_data.update(extra_info)
        torch.save(save_data, _DUMP_LOGITS_OUTPUT)
        print(f"[VLLM_LOGITS_DUMP] Saved logits {logits.shape} to {_DUMP_LOGITS_OUTPUT}", flush=True)
    except Exception as e:
        print(f"[VLLM_LOGITS_DUMP] Error: {e}", flush=True)
# === RAW LOGITS DUMP PATCH END ===
'''

# Line to add after logits computation
PATCH_CALL = "        _maybe_dump_raw_logits(logits)"


@ray.remote(num_gpus=0.01)
def check_patch_status():
    """Check if the patch is applied."""
    import os
    hostname = os.uname().nodename

    try:
        with open(MODEL_RUNNER_PATH, "r") as f:
            content = f.read()

        is_patched = "_maybe_dump_raw_logits" in content
        return {
            "hostname": hostname,
            "path": MODEL_RUNNER_PATH,
            "is_patched": is_patched,
            "exists": True,
        }
    except Exception as e:
        return {
            "hostname": hostname,
            "path": MODEL_RUNNER_PATH,
            "error": str(e),
            "exists": False,
        }


@ray.remote(num_gpus=0.01)
def apply_patch():
    """Apply the raw logits dump patch."""
    import os
    hostname = os.uname().nodename

    try:
        with open(MODEL_RUNNER_PATH, "r") as f:
            content = f.read()

        if "_maybe_dump_raw_logits" in content:
            return {"hostname": hostname, "status": "already_patched"}

        # Find where to insert module-level code (after imports, before first class)
        class_idx = content.find("class GPUModelRunner")
        if class_idx == -1:
            # Try alternative class name
            class_idx = content.find("class ModelRunner")
        if class_idx == -1:
            return {"hostname": hostname, "status": "error", "msg": "Could not find ModelRunner class"}

        # Insert module-level patch before class
        new_content = content[:class_idx] + PATCH_MODULE_CODE + "\n" + content[class_idx:]

        # Find execute_model method's logits line and add call after it
        # Look for pattern like: logits = self.model(
        search_start = 0
        logits_patterns = [
            "logits = self.model(",
            "logits = self._model(",
            "logits = model(",
        ]

        logits_idx = -1
        for pattern in logits_patterns:
            idx = new_content.find(pattern, search_start)
            if idx != -1:
                logits_idx = idx
                break

        if logits_idx == -1:
            # Try to find any logits assignment
            logits_idx = new_content.find("logits = ", search_start)

        if logits_idx == -1:
            # Still insert module code but note we couldn't add the call
            with open(MODEL_RUNNER_PATH, "w") as f:
                f.write(new_content)
            return {
                "hostname": hostname,
                "status": "partial_patch",
                "msg": "Added module code but could not find logits assignment to add call"
            }

        # Find end of the logits statement (handle multiline)
        # Look for the closing parenthesis at the right indentation level
        paren_count = 0
        in_string = False
        string_char = None
        i = logits_idx
        started = False

        while i < len(new_content):
            char = new_content[i]

            # Handle strings
            if char in '"\'':
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char and new_content[i-1] != '\\':
                    in_string = False

            if not in_string:
                if char == '(':
                    paren_count += 1
                    started = True
                elif char == ')':
                    paren_count -= 1
                    if started and paren_count == 0:
                        break
                elif char == '\n' and paren_count == 0 and started:
                    break
            i += 1

        # Find the newline after the statement
        end_idx = new_content.find('\n', i)
        if end_idx == -1:
            end_idx = len(new_content)

        # Insert the call
        new_content = new_content[:end_idx] + "\n" + PATCH_CALL + new_content[end_idx:]

        # Write patched file
        with open(MODEL_RUNNER_PATH, "w") as f:
            f.write(new_content)

        return {"hostname": hostname, "status": "patched"}

    except PermissionError:
        return {"hostname": hostname, "status": "error", "msg": "Permission denied - need root"}
    except Exception as e:
        return {"hostname": hostname, "status": "error", "msg": str(e)}


@ray.remote(num_gpus=0.01)
def remove_patch():
    """Remove the raw logits dump patch."""
    import os
    import re
    hostname = os.uname().nodename

    try:
        with open(MODEL_RUNNER_PATH, "r") as f:
            content = f.read()

        if "_maybe_dump_raw_logits" not in content:
            return {"hostname": hostname, "status": "not_patched"}

        # Remove module-level patch
        pattern = r'# === RAW LOGITS DUMP PATCH START ===.*?# === RAW LOGITS DUMP PATCH END ===\n'
        new_content = re.sub(pattern, '', content, flags=re.DOTALL)

        # Remove the call
        new_content = new_content.replace(PATCH_CALL + "\n", "")
        new_content = new_content.replace(PATCH_CALL, "")

        with open(MODEL_RUNNER_PATH, "w") as f:
            f.write(new_content)

        return {"hostname": hostname, "status": "unpatched"}

    except Exception as e:
        return {"hostname": hostname, "status": "error", "msg": str(e)}


def main():
    if len(sys.argv) < 2:
        print("Usage: python patch_vllm_logits_dump.py [--check|--patch|--unpatch]")
        sys.exit(1)

    action = sys.argv[1]

    ray.init(address="auto", namespace="tinker", ignore_reinit_error=True)

    # Get all nodes with GPUs
    nodes = ray.nodes()
    gpu_nodes = [n for n in nodes if n.get("Resources", {}).get("GPU", 0) > 0 and n["Alive"]]
    print(f"Found {len(gpu_nodes)} GPU nodes")

    if action == "--check":
        # Check on one node (they should all be the same)
        result = ray.get(check_patch_status.remote())
        print(f"Node {result['hostname']}: patched={result.get('is_patched', 'unknown')}")

    elif action == "--patch":
        # Patch on one node first to test
        result = ray.get(apply_patch.remote())
        print(f"Node {result['hostname']}: {result['status']}")
        if result['status'] == 'error':
            print(f"  Error: {result.get('msg', 'unknown')}")

    elif action == "--unpatch":
        result = ray.get(remove_patch.remote())
        print(f"Node {result['hostname']}: {result['status']}")

    else:
        print(f"Unknown action: {action}")
        sys.exit(1)


if __name__ == "__main__":
    main()
