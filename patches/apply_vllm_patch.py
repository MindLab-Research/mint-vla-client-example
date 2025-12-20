#!/usr/bin/env python3
"""Patch vLLM models.py to support MoE expert LoRA weights."""

import re
import sys

VLLM_MODELS_PATH = "/root/tinker_project/vllm/vllm/lora/models.py"

# The old code block to replace
OLD_CODE = '''                # Case for expert lora weights
                if ".experts" in module_name:
                    expert_idx = module_name.find(".experts")
                    expert_suffix = module_name[expert_idx + 1 :]
                    if expert_suffix not in expected_lora_modules:
                        unexpected_modules.append(module_name)'''

# The new code block
NEW_CODE = '''                # Case for expert lora weights
                if ".experts" in module_name:
                    # Handle expert patterns like: experts.0.gate_proj, experts.1.down_proj
                    # Extract the module name after experts.{N}.
                    import re as re_mod
                    VALID_EXPERT_SUFFIXES = {"gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"}
                    expert_match = re_mod.search(r"\\.experts\\.(\\d+)\\.(\\w+)$", module_name)
                    if expert_match:
                        expert_module = expert_match.group(2)  # e.g., "gate_proj"
                        # Accept if it is a valid expert suffix or in expected modules
                        if expert_module not in VALID_EXPERT_SUFFIXES and \\
                           expert_module not in expected_lora_modules and \\
                           "experts" not in expected_lora_modules:
                            unexpected_modules.append(module_name)
                    else:
                        # Fallback: check if "experts" is in expected modules
                        if "experts" not in expected_lora_modules:
                            unexpected_modules.append(module_name)'''


def main():
    # Read the original file
    with open(VLLM_MODELS_PATH, "r") as f:
        content = f.read()

    if OLD_CODE in content:
        content = content.replace(OLD_CODE, NEW_CODE)
        with open(VLLM_MODELS_PATH, "w") as f:
            f.write(content)
        print("SUCCESS: Patched vLLM models.py for MoE expert LoRA support")
        return 0
    elif "VALID_EXPERT_SUFFIXES" in content:
        print("INFO: Patch already applied")
        return 0
    else:
        print("ERROR: Could not find the expected code block to patch")
        # Try to find partial matches for debugging
        if "expert_suffix = module_name[expert_idx + 1 :]" in content:
            print("Found expert_suffix line but context differs")
        if ".experts" in content and "module_name" in content:
            print("Found related code but exact match failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
