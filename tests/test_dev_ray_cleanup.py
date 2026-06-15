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


def test_force_reset_skips_actor_and_task_state_probes(cleanup_module, monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("force reset must not run auto probes")

    monkeypatch.setattr(cleanup_module, "_expected_config_snapshot", fail)
    monkeypatch.setattr(cleanup_module, "_active_task_probe", fail)

    result = cleanup_module._probe_should_reset(
        ray=SimpleNamespace(),
        namespace="mint_test",
        mode="force",
    )

    assert result == {
        "expected_config_actor": None,
        "expected_fingerprint": None,
        "actual_fingerprint": None,
        "actual_snapshot_error": None,
        "active_task_probe": None,
        "active_task_probe_error": None,
        "should_reset": True,
    }


def test_dashboard_fast_reset_snapshot_requires_no_alive_actors_or_active_pgs(
    cleanup_module,
    monkeypatch,
):
    monkeypatch.setenv("MINT_DEV_RESET_SKIP_RAY_WHEN_NO_ALIVE", "1")
    calls: list[str] = []

    def fake_dashboard(path: str):
        calls.append(path)
        if path.startswith("/api/v0/actors?"):
            return [
                {"name": "mint_config", "state": "DEAD"},
                {"name": "mint_model_work_scheduler", "state": "DEAD"},
            ]
        if path == "/api/v0/placement_groups?limit=10000":
            return [
                {"name": "target_pg", "state": "REMOVED", "placement_group_id": "pg1"},
                {"name": "other_pg", "state": "CREATED", "placement_group_id": "pg2"},
            ]
        raise AssertionError(path)

    monkeypatch.setattr(cleanup_module, "_dashboard_result", fake_dashboard)

    result = cleanup_module._dashboard_no_alive_reset_snapshot(
        "mint_test",
        ["target_pg"],
    )

    assert result["fast_reset_available"] is True
    assert result["dashboard_actor_count"] == 2
    assert result["dashboard_alive_actor_count"] == 0
    assert result["dashboard_active_actor_count"] == 0
    assert result["dashboard_active_actors"] == []
    assert result["dashboard_active_reset_pg_count"] == 0
    assert calls[0].startswith("/api/v0/actors?")
    assert calls[1] == "/api/v0/placement_groups?limit=10000"


def test_dashboard_fast_reset_snapshot_blocks_alive_actor(cleanup_module, monkeypatch):
    monkeypatch.setenv("MINT_DEV_RESET_SKIP_RAY_WHEN_NO_ALIVE", "1")

    def fake_dashboard(path: str):
        if path.startswith("/api/v0/actors?"):
            return [{"name": "mint_config", "state": "ALIVE"}]
        return []

    monkeypatch.setattr(cleanup_module, "_dashboard_result", fake_dashboard)

    result = cleanup_module._dashboard_no_alive_reset_snapshot("mint_test", [])

    assert result["fast_reset_available"] is False
    assert result["dashboard_alive_actor_count"] == 1
    assert result["dashboard_active_actor_count"] == 1


def test_dashboard_fast_reset_snapshot_blocks_pending_actor(cleanup_module, monkeypatch):
    monkeypatch.setenv("MINT_DEV_RESET_SKIP_RAY_WHEN_NO_ALIVE", "1")

    def fake_dashboard(path: str):
        if path.startswith("/api/v0/actors?"):
            return [
                {
                    "name": "mint_config",
                    "state": "PENDING_CREATION",
                    "actor_id": "actor-1",
                    "class_name": "ConfigActor",
                }
            ]
        return []

    monkeypatch.setattr(cleanup_module, "_dashboard_result", fake_dashboard)

    result = cleanup_module._dashboard_no_alive_reset_snapshot("mint_test", [])

    assert result["fast_reset_available"] is False
    assert result["dashboard_alive_actor_count"] == 0
    assert result["dashboard_active_actor_count"] == 1
    assert result["dashboard_active_actors"] == [
        {
            "name": "mint_config",
            "state": "PENDING_CREATION",
            "actor_id": "actor-1",
            "class_name": "ConfigActor",
        }
    ]


def test_gc_stale_actors_skips_ray_init_when_dashboard_has_no_candidates(
    cleanup_module,
    monkeypatch,
    tmp_path,
):
    summary_path = tmp_path / "summary.json"
    results_path = tmp_path / "results.jsonl"

    monkeypatch.setenv("MINT_RAY_NAMESPACE", "current")
    monkeypatch.setattr(
        cleanup_module,
        "_dashboard_result",
        lambda _path: [
            {
                "name": "mint_recent",
                "class_name": "ConfigActor",
                "ray_namespace": "other",
                "start_time_ms": 999_000.0,
                "state": "ALIVE",
            }
        ],
    )
    monkeypatch.setattr(cleanup_module.time, "time", lambda: 1_000.0)

    def fail_init(_namespace: str):
        raise AssertionError("no stale candidates should avoid ray.init")

    monkeypatch.setattr(cleanup_module, "_init_ray", fail_init)

    rc = cleanup_module.cmd_gc_stale_actors(
        SimpleNamespace(
            namespace=None,
            summary=str(summary_path),
            results=str(results_path),
            max_age_s=300.0,
            limit=100,
        )
    )

    assert rc == 0
    assert summary_path.exists()
    assert results_path.read_text(encoding="utf-8") == ""


def test_dev_ray_cleanup_script_has_no_dotenv_or_legacy_issue_launcher_dependencies():
    text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "dotenv" not in text.lower()
    assert "start_issue729" not in text
    assert "MINT_ISSUE729" not in text
    assert "secrets.env" not in text
