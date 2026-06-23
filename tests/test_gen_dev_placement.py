from __future__ import annotations

import json

from scripts.tools import gen_dev_placement


def test_get_worker_ips_filters_to_alive_gpu_workers_and_excludes_head(monkeypatch):
    payload = {
        "data": {
            "result": {
                "result": [
                    {
                        "node_ip": "10.0.0.1",
                        "state": "ALIVE",
                        "is_head_node": True,
                        "resources_total": {"CPU": 8, "GPU": 0},
                    },
                    {
                        "node_ip": "10.0.0.2",
                        "state": "ALIVE",
                        "is_head_node": False,
                        "resources_total": {"CPU": 64, "GPU": 8},
                    },
                    {
                        "node_ip": "10.0.0.3",
                        "state": "DEAD",
                        "is_head_node": False,
                        "resources_total": {"CPU": 64, "GPU": 8},
                    },
                ]
            }
        }
    }

    monkeypatch.setattr(gen_dev_placement, "_read_json_url", lambda _url, timeout_s=10.0: payload)

    assert gen_dev_placement.get_worker_ips("10.0.0.1") == [("10.0.0.2", 8)]


def test_models_from_env_prefers_persistent_models(monkeypatch):
    monkeypatch.setenv("MINT_SUPPORTED_MODELS", "Qwen/A,Qwen/B")
    monkeypatch.setenv("MINT_PERSISTENT_MODELS", "Qwen/C, Qwen/D ")

    assert gen_dev_placement.models_from_env() == ["Qwen/C", "Qwen/D"]


def test_write_env_file_exports_all_canonical_placement_vars(tmp_path):
    output = tmp_path / "auto.env"

    gen_dev_placement.write_env_file(
        output,
        head_ip="10.0.0.1",
        models=["Qwen/A", "Qwen/B"],
        workers=[("10.0.0.2", 8), ("10.0.0.3", 8)],
        gpu_count=2,
        max_model_len=4096,
    )

    text = output.read_text(encoding="utf-8")
    placement = {
        "Qwen/A": {"replica": 0, "node_ip": "10.0.0.2", "gpu_count": 2},
        "Qwen/B": {"replica": 0, "node_ip": "10.0.0.3", "gpu_count": 2},
    }
    placement_json = json.dumps(placement, sort_keys=True, separators=(",", ":"))
    assert f"export MINT_MODEL_PLACEMENT_JSON='{placement_json}'" in text
    assert f"export MINT_DENSE_MODEL_PLACEMENT_JSON='{placement_json}'" in text
    assert f"export MINT_VLLM_MODEL_PLACEMENT_JSON='{placement_json}'" in text
    assert f"export MINT_MEGATRON_MODEL_PLACEMENT_JSON='{placement_json}'" in text
    assert "MINT_MODEL_CONFIG_OVERRIDES_JSON" in text


def test_write_env_file_uses_known_model_gpu_count_defaults(tmp_path):
    output = tmp_path / "auto.env"

    gen_dev_placement.write_env_file(
        output,
        head_ip="10.0.0.1",
        models=["Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen/Qwen3-0.6B"],
        workers=[("10.0.0.2", 8), ("10.0.0.3", 8)],
        gpu_count=1,
        max_model_len=4096,
    )

    text = output.read_text(encoding="utf-8")
    placement = {
        "Qwen/Qwen3-30B-A3B-Instruct-2507": {
            "replica": 0,
            "node_ip": "10.0.0.2",
            "gpu_count": 4,
        },
        "Qwen/Qwen3-0.6B": {"replica": 0, "node_ip": "10.0.0.3", "gpu_count": 1},
    }
    placement_json = json.dumps(placement, sort_keys=True, separators=(",", ":"))
    assert f"export MINT_MODEL_PLACEMENT_JSON='{placement_json}'" in text


def test_write_env_file_fails_when_placement_overcommits_worker(tmp_path):
    output = tmp_path / "auto.env"

    try:
        gen_dev_placement.write_env_file(
            output,
            head_ip="10.0.0.1",
            models=["Qwen/A", "Qwen/B", "Qwen/Qwen3-30B-A3B-Instruct-2507"],
            workers=[("10.0.0.2", 4)],
            gpu_count=1,
            max_model_len=4096,
        )
    except RuntimeError as exc:
        assert "not enough alive GPU worker capacity" in str(exc)
    else:
        raise AssertionError("expected overcommitted placement to fail")


def test_write_env_file_refuses_to_overwrite_existing_file_without_force(tmp_path):
    output = tmp_path / "auto.env"
    output.write_text("# existing manual config\n", encoding="utf-8")

    try:
        gen_dev_placement.write_env_file(
            output,
            head_ip="10.0.0.1",
            models=["Qwen/A"],
            workers=[("10.0.0.2", 8)],
            gpu_count=1,
            max_model_len=4096,
        )
    except RuntimeError as exc:
        assert "already exists" in str(exc)
        assert "--force" in str(exc)
    else:
        raise AssertionError("expected write_env_file to refuse overwriting without force")

    # File content should be unchanged.
    assert output.read_text(encoding="utf-8") == "# existing manual config\n"


def test_write_env_file_overwrites_existing_file_with_force(tmp_path):
    output = tmp_path / "auto.env"
    output.write_text("# existing manual config\n", encoding="utf-8")

    gen_dev_placement.write_env_file(
        output,
        head_ip="10.0.0.1",
        models=["Qwen/A"],
        workers=[("10.0.0.2", 8)],
        gpu_count=1,
        max_model_len=4096,
        force=True,
    )

    text = output.read_text(encoding="utf-8")
    assert "Auto-generated" in text
    assert "MINT_MODEL_PLACEMENT_JSON" in text


def test_get_worker_ips_raises_when_dashboard_unreachable(monkeypatch):
    def _fail(_url, timeout_s=10.0):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(gen_dev_placement, "_read_json_url", _fail)

    try:
        gen_dev_placement.get_worker_ips("10.0.0.1")
    except RuntimeError as exc:
        assert "failed to query Ray dashboard" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when dashboard is unreachable")


def test_get_worker_ips_returns_empty_when_no_alive_gpu_workers(monkeypatch):
    payload = {
        "data": {
            "result": {
                "result": [
                    {
                        "node_ip": "10.0.0.1",
                        "state": "ALIVE",
                        "is_head_node": True,
                        "resources_total": {"CPU": 8, "GPU": 0},
                    },
                ]
            }
        }
    }

    monkeypatch.setattr(gen_dev_placement, "_read_json_url", lambda _url, timeout_s=10.0: payload)

    assert gen_dev_placement.get_worker_ips("10.0.0.1") == []


def test_write_env_file_fails_with_no_workers(tmp_path):
    output = tmp_path / "auto.env"

    try:
        gen_dev_placement.write_env_file(
            output,
            head_ip="10.0.0.1",
            models=["Qwen/A"],
            workers=[],
            gpu_count=1,
            max_model_len=4096,
        )
    except RuntimeError as exc:
        assert "no alive GPU workers" in str(exc)
    else:
        raise AssertionError("expected failure when no workers are provided")


def test_write_env_file_distributes_models_across_multiple_workers(tmp_path):
    output = tmp_path / "auto.env"

    placement = gen_dev_placement.write_env_file(
        output,
        head_ip="10.0.0.1",
        models=["Qwen/A", "Qwen/B", "Qwen/C", "Qwen/D"],
        workers=[("10.0.0.2", 4), ("10.0.0.3", 4)],
        gpu_count=1,
        max_model_len=4096,
        force=True,
    )

    # Each model needs 1 GPU; workers have 4 each. Models should be spread
    # across both workers via the most-remaining heuristic.
    worker_ips_used = {p["node_ip"] for p in placement.values()}
    assert worker_ips_used == {"10.0.0.2", "10.0.0.3"}

    # No worker should be over capacity (4 GPUs, 1 per model).
    from collections import Counter

    ip_counts = Counter(p["node_ip"] for p in placement.values())
    assert all(count <= 4 for count in ip_counts.values())
