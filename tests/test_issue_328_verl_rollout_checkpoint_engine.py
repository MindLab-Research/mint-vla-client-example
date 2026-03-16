from __future__ import annotations

from pathlib import Path


def test_issue_328_rollout_config_sets_checkpoint_engine_backend() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    required = (
        "tinker_server/backend/verl_inference.py",
        "tinker_server/backend/multi_lora_engine.py",
    )
    needle = 'CheckpointEngineConfig(backend="naive")'

    for rel_path in required:
        text = (repo_root / rel_path).read_text(encoding="utf-8")
        assert needle in text, f"{rel_path} must set a checkpoint_engine backend for new verl RolloutConfig"


def test_issue_328_actor_startup_does_not_apply_vllm_hijack() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "tinker_server/backend/verl_inference.py").read_text(encoding="utf-8")
    assert "VLLMHijack.hijack()" not in text


def test_issue_328_vllm_actors_force_spawn_mode() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    required = (
        "tinker_server/backend/multi_lora_engine.py",
        "tinker_server/backend/verl_inference.py",
        "tinker_server/backend/multinode_inference.py",
    )
    needle = '"VLLM_WORKER_MULTIPROC_METHOD": "spawn"'
    for rel_path in required:
        text = (repo_root / rel_path).read_text(encoding="utf-8")
        assert needle in text, f"{rel_path} must force spawn mode for vLLM worker subprocesses"


def test_issue_328_single_gpu_standalone_has_no_bespoke_backend_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "tinker_server/backend/verl_inference.py").read_text(encoding="utf-8")
    assert 'args.distributed_executor_backend = "uni"' not in text
    assert 'args.worker_extension_cls = ""' not in text
    assert "single_gpu_standalone" not in text
