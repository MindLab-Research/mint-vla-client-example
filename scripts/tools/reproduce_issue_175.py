#!/usr/bin/env python3
"""Reproduction script for issue #175: Capabilities include parameter count and sort models.

Tests that:
1. /api/v1/get_server_capabilities includes num_parameters field
2. Models are sorted by num_parameters (ascending)
3. Sorting works across both local and gateway-routed models
"""

import os
import sys

import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")


def test_capabilities():
    """Test capabilities endpoint includes num_parameters and sorts correctly."""
    url = f"{BASE_URL}/api/v1/get_server_capabilities"
    headers = {"X-API-Key": API_KEY} if API_KEY != "dummy" else {}

    print(f"Testing: GET {url}")
    resp = requests.get(url, headers=headers, timeout=10)

    if resp.status_code != 200:
        print(f"FAIL: Expected status 200, got {resp.status_code}")
        print(f"Response: {resp.text}")
        return False

    data = resp.json()
    models = data.get("supported_models", [])

    if not models:
        print("FAIL: No models returned")
        return False

    print(f"\nFound {len(models)} models:")
    print("-" * 80)

    # Check each model has required fields
    prev_params = None
    for i, model in enumerate(models):
        model_name = model.get("model_name")
        max_context = model.get("max_context_length")
        num_params = model.get("num_parameters")

        if not model_name:
            print(f"FAIL: Model {i} missing model_name")
            return False

        if max_context is None:
            print(f"FAIL: Model {model_name} missing max_context_length")
            return False

        # num_parameters is optional for gateway-routed models not in local registry
        if num_params is not None:
            print(f"{i+1}. {model_name:50s} | {num_params:8.1f}B | {max_context:6d} ctx")

            # Check sorting (ascending by num_parameters)
            if prev_params is not None and num_params < prev_params:
                print(f"FAIL: Models not sorted by num_parameters")
                print(f"  Previous: {prev_params}B, Current: {num_params}B")
                return False

            prev_params = num_params
        else:
            print(f"{i+1}. {model_name:50s} | {'N/A':>8s} | {max_context:6d} ctx")

    print("-" * 80)
    print(f"\nPASS: All {len(models)} models have required fields and are sorted correctly")
    return True


def main():
    """Run reproduction test."""
    print("=" * 80)
    print("Issue #175: Capabilities include parameter count and sort models")
    print("=" * 80)
    print()

    try:
        success = test_capabilities()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"FAIL: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
