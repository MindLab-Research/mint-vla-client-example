from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _tracked_text_files() -> list[Path]:
    roots = [
        "mint_server",
        "scripts",
        "ops",
        "tests",
        ".claude/skills",
        "pyproject.toml",
        "README.md",
        "CLAUDE.md",
        "AGENTS.md",
        "PLAN.md",
        "Dockerfile",
        "sitecustomize.py",
    ]
    excluded_prefixes = {
        ".claude/skills/tinker-official-reference/",
        "tinker" + "_server/",
    }
    excluded_exact = {
        "configs/" + "tinker" + "-server-auth.supervisord.conf",
    }
    text_suffixes = {
        ".cfg",
        ".conf",
        ".css",
        ".html",
        ".in",
        ".json",
        ".md",
        ".py",
        ".sh",
        ".toml",
        ".tsx",
        ".ts",
        ".txt",
        ".yaml",
        ".yml",
        ".zsh",
    }
    tracked = subprocess.check_output(["git", "ls-files", *roots], cwd=REPO_ROOT, text=True).splitlines()
    out: list[Path] = []
    for rel_path in tracked:
        p = REPO_ROOT / rel_path
        if not p.is_file() or p.suffix not in text_suffixes:
            continue
        if rel_path in excluded_exact:
            continue
        if any(rel_path.startswith(prefix) for prefix in excluded_prefixes):
            continue
        out.append(p)
    return sorted(out)


def _current_service_text_files() -> list[Path]:
    roots = [
        "mint_server",
        ".claude/skills/architecture-design/references",
        "pyproject.toml",
        "README.md",
        "CLAUDE.md",
        "AGENTS.md",
        "PLAN.md",
        "Dockerfile",
    ]
    excluded_exact = {
        ".claude/skills/architecture-design/references/architecture.md",
        ".claude/skills/architecture-design/references/checkpoint-compat-matrix.md",
        ".claude/skills/architecture-design/references/weights-checkpoints.md",
        "mint_server/app.py",
        "mint_server/client_compat.py",
        "mint_server/compatibility.py",
        "mint_server/routes/service.py",
        "mint_server/routes/training.py",
        "mint_server/routes/weights.py",
    }
    text_suffixes = {
        ".cfg",
        ".conf",
        ".in",
        ".md",
        ".py",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
    out: list[Path] = []
    for rel in roots:
        path = REPO_ROOT / rel
        if path.is_dir():
            for p in path.rglob("*"):
                if not p.is_file() or p.suffix not in text_suffixes:
                    continue
                rel_path = str(p.relative_to(REPO_ROOT))
                if rel_path in excluded_exact:
                    continue
                out.append(p)
        elif path.is_file():
            out.append(path)
    return sorted(out)


ENV_ALIAS_TOKEN = "TINKER" + "_"
URI_COMPAT_TOKEN = "tinker" + "://"


def test_core_service_no_legacy_names() -> None:
    forbidden = (
        "tinker" + "_server",
        "tinker" + "-server",
        "PFS_" + "TINKER" + "_PATH",
        "tool." + "tinker",
        "[tool." + "tinker",
        "load_" + "tinker" + "_config_file",
        "Tinker" + "ConfigFile",
        "tinker" + "_vllm_",
        "mint" + "_vllm" + "_server",
        "multinode" + "_vllm_",
        "peft" + "_trainer_",
        "tinker" + "_training_session_store",
        "tinker" + "_gateway_session_store",
        "tinker" + "_training_cleanup_executor",
        "namespace=" + '"' + "tinker" + '"',
        "namespace=" + "'" + "tinker" + "'",
        '"namespace": "' + "tinker" + '"',
        "MINT_RAY_NAMESPACE" + '": "' + "tinker" + '"',
        "openpi_action_session_state/" + "tinker" + "/",
        "tinker" + "_startup_lease",
        "ns_" + "tinker",
    )
    hits: list[str] = []
    for path in _tracked_text_files():
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                hits.append(f"{path.relative_to(REPO_ROOT)} contains {token}")

    assert hits == []


def test_current_service_no_internal_tinker_or_direct_billing_surface() -> None:
    forbidden = (
        "tinker" + "_to_tensordict",
        "_tinker" + "_",
        "tinker" + "_lora_",
        "schedule_usage_events(",
        "persist_usage_events(",
        "direct-PG",
        "direct PG",
        "directly to PostgreSQL",
    )
    hits: list[str] = []
    for path in _current_service_text_files():
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                hits.append(f"{path.relative_to(REPO_ROOT)} contains {token}")

    assert hits == []


def test_env_aliases_stay_in_compatibility_boundary() -> None:
    allowed = {
        "mint_server/runtime_env.py",
        "mint_server/runtime_config.py",
        "scripts/run_server.py",
        "tests/test_external_compatibility.py",
        "tests/test_openai_compat_helpers.py",
    }
    hits: list[str] = []
    for path in _tracked_text_files():
        rel = str(path.relative_to(REPO_ROOT))
        if rel in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if ENV_ALIAS_TOKEN in text:
            hits.append(rel)

    assert hits == []


def test_vllm_requires_model_specific_actor_name() -> None:
    from mint_server.backend.inference.multi_lora_engine import MultiLoRAInferenceEngine
    from mint_server.backend.ray_cluster.model_actor_names import vllm_actor_name as _model_to_actor_name

    actor_name = _model_to_actor_name("Qwen/Qwen3-30B-A3B-Instruct-2507")
    assert actor_name == "mint_vllm_qwen3-30b-a3b-instruct-2507"
    assert actor_name != "mint" + "_vllm" + "_server"

    try:
        MultiLoRAInferenceEngine(model_path="/tmp/model")
    except ValueError as exc:
        assert "model-specific actor_name" in str(exc)
    else:
        raise AssertionError("MultiLoRAInferenceEngine must not create a default vLLM actor")


def test_compat_uris_stay_in_compatibility_boundary() -> None:
    allowed = {
        ".claude/skills/architecture-design/references/checkpoint-compat-matrix.md",
        ".claude/skills/architecture-design/references/weights-checkpoints.md",
        "mint_server/compatibility.py",
        "tests/test_client_compat_user_agent.py",
        "tests/test_external_compatibility.py",
    }
    hits: list[str] = []
    for path in _tracked_text_files():
        rel = str(path.relative_to(REPO_ROOT))
        if rel in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if URI_COMPAT_TOKEN in text:
            hits.append(rel)

    assert hits == []
