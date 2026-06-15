from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "tools" / "dev_ray_cleanup.py"


@pytest.fixture(scope="module")
def cleanup_module():
    spec = importlib.util.spec_from_file_location("dev_ray_cleanup", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stale_actor_candidates_selects_old_foreign_mint_owned_actors(cleanup_module):
    now_ms = 1_000_000.0
    rows = [
        {
            "name": "mint_config",
            "class_name": "ConfigActor",
            "ray_namespace": "other",
            "start_time_ms": now_ms - 400_000.0,
            "state": "ALIVE",
        },
        {
            "name": "not_mint",
            "class_name": "PlainActor",
            "ray_namespace": "other",
            "start_time_ms": now_ms - 400_000.0,
            "state": "ALIVE",
        },
        {
            "name": "mint_current",
            "class_name": "ConfigActor",
            "ray_namespace": "current",
            "start_time_ms": now_ms - 400_000.0,
            "state": "ALIVE",
        },
        SimpleNamespace(
            name="mint_recent",
            class_name="ConfigActor",
            ray_namespace="other",
            start_time_ms=now_ms - 1_000.0,
            state="ALIVE",
        ),
        {
            "name": "mint_dead",
            "class_name": "ConfigActor",
            "ray_namespace": "other",
            "start_time_ms": now_ms - 400_000.0,
            "state": "DEAD",
        },
    ]

    candidates = cleanup_module.stale_actor_candidates(
        rows,
        driver_namespace="current",
        now_ms=now_ms,
        max_age_s=300.0,
    )

    assert [candidate["name"] for candidate in candidates] == ["mint_config"]
    assert candidates[0]["age_s"] == 400.0


def test_clear_task_state_dir_requires_safe_descendant(cleanup_module, tmp_path):
    safe_root = tmp_path / "safe"
    task_state = safe_root / "issue" / "task-state"
    payload = task_state / "payloads" / "payload.json"
    payload.parent.mkdir(parents=True)
    payload.write_text("{}", encoding="utf-8")

    record = cleanup_module.clear_task_state_dir(task_state, safe_root)

    assert record["status"] == "removed"
    assert record["file_count"] == 1
    assert (task_state / "payloads").is_dir()
    assert not payload.exists()


def test_clear_task_state_dir_refuses_safe_root_or_sibling(cleanup_module, tmp_path):
    safe_root = tmp_path / "safe"
    safe_root.mkdir()

    with pytest.raises(RuntimeError, match="outside safe root"):
        cleanup_module.clear_task_state_dir(safe_root, safe_root)

    with pytest.raises(RuntimeError, match="outside safe root"):
        cleanup_module.clear_task_state_dir(tmp_path / "sibling", safe_root)


def test_dev_ray_cleanup_script_has_no_dotenv_or_legacy_issue_launcher_dependencies():
    text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "dotenv" not in text.lower()
    assert "start_issue729" not in text
    assert "MINT_ISSUE729" not in text
    assert "secrets.env" not in text
