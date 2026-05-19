from __future__ import annotations

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
        "requirements/api_server_driver_py31213.requirements.in",
        "README.md",
        "CLAUDE.md",
        "AGENTS.md",
        "PLAN.md",
        "Dockerfile",
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
                if any(rel_path.startswith(prefix) for prefix in excluded_prefixes):
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
