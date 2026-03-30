from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("ray")

from tinker_server.backend.megatron_distributed import MegatronSessionStateManager, MegatronWorkerGroup


def _write_adapter(session_path: Path, *, size: int = 64) -> None:
    session_path.mkdir(parents=True, exist_ok=True)
    (session_path / "mp_rank_00_adapter.pt").write_bytes(b"a" * size)


def _set_tree_mtime(path: Path, when: float) -> None:
    for root, dirs, files in os.walk(path):
        for name in files:
            os.utime(Path(root) / name, (when, when))
        for name in dirs:
            os.utime(Path(root) / name, (when, when))
    os.utime(path, (when, when))


def test_issue_414_cache_usage_reports_skipped_and_evictable(tmp_path: Path):
    manager = MegatronSessionStateManager(base_path=str(tmp_path))

    safe_path = Path(manager.get_session_path("session_safe"))
    dirty_path = Path(manager.get_session_path("session_dirty"))
    _write_adapter(safe_path, size=128)
    _write_adapter(dirty_path, size=256)

    manager.save_metadata("session_safe", step=1, lr=1e-4, actual_rank=8)
    manager.mark_external_checkpoint(
        "session_safe",
        checkpoint_path="/checkpoints/alice/model_a/ckpt_1",
        reason="save_checkpoint",
        actor_name="shared-megatron-actor",
    )
    manager.mark_actor_only_state(
        "session_dirty",
        reason="forward_backward",
        actor_name="shared-megatron-actor",
    )

    usage = manager.get_cache_usage(actor_name="shared-megatron-actor")

    assert usage["session_count"] == 2
    assert usage["skipped_not_cold_safe_count"] == 1
    assert usage["skipped_not_cold_safe_sessions"] == ["session_dirty"]
    assert usage["evictable_session_count"] == 1
    assert usage["evictable_bytes"] >= 128
    assert usage["total_bytes"] >= 128 + 256


def test_issue_414_recycle_cache_respects_age_and_dirty_markers(tmp_path: Path):
    manager = MegatronSessionStateManager(base_path=str(tmp_path))

    old_safe = Path(manager.get_session_path("session_old_safe"))
    new_safe = Path(manager.get_session_path("session_new_safe"))
    dirty = Path(manager.get_session_path("session_dirty"))
    _write_adapter(old_safe, size=111)
    _write_adapter(new_safe, size=222)
    _write_adapter(dirty, size=333)

    manager.mark_external_checkpoint(
        "session_old_safe",
        checkpoint_path="/checkpoints/alice/model_a/ckpt_old",
        reason="save_checkpoint",
        actor_name="actor-a",
    )
    manager.mark_external_checkpoint(
        "session_new_safe",
        checkpoint_path="/checkpoints/alice/model_a/ckpt_new",
        reason="save_checkpoint",
        actor_name="actor-a",
    )
    manager.mark_actor_only_state(
        "session_dirty",
        reason="forward_backward",
        actor_name="actor-a",
    )

    old_ts = time.time() - 7200
    _set_tree_mtime(old_safe, old_ts)

    result = manager.recycle_cache(max_age_s=3600, max_total_bytes=0, max_bytes_per_actor=0)

    evicted_ids = [item["session_id"] for item in result["evicted_sessions"]]
    assert evicted_ids == ["session_old_safe"]
    assert not old_safe.exists()
    assert new_safe.exists()
    assert dirty.exists()
    assert result["after"]["skipped_not_cold_safe_count"] == 1


def test_issue_414_recycle_cache_enforces_per_actor_budget(tmp_path: Path):
    manager = MegatronSessionStateManager(base_path=str(tmp_path))

    for session_id, actor_name, size in (
        ("session_a_old", "actor-a", 200),
        ("session_a_new", "actor-a", 200),
        ("session_b", "actor-b", 200),
    ):
        session_path = Path(manager.get_session_path(session_id))
        _write_adapter(session_path, size=size)
        manager.mark_external_checkpoint(
            session_id,
            checkpoint_path=f"/checkpoints/{actor_name}/{session_id}",
            reason="save_checkpoint",
            actor_name=actor_name,
        )

    _set_tree_mtime(Path(manager.get_session_path("session_a_old")), time.time() - 7200)

    result = manager.recycle_cache(max_total_bytes=0, max_age_s=0, max_bytes_per_actor=350)

    evicted_ids = {item["session_id"] for item in result["evicted_sessions"]}
    assert evicted_ids == {"session_a_old"}
    assert not Path(manager.get_session_path("session_a_old")).exists()
    assert Path(manager.get_session_path("session_a_new")).exists()
    assert Path(manager.get_session_path("session_b")).exists()


def test_issue_414_save_checkpoint_marks_external_checkpoint(monkeypatch: pytest.MonkeyPatch):
    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)
    group.workers = []
    group.base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    group.config = type("Cfg", (), {"world_size": 1})()
    group._actual_rank = 8
    group._bind_traceparent = lambda traceparent: None
    group._resolve_required_session_id = lambda session_id, op: session_id
    group._ensure_session_loaded = lambda session_id, **kwargs: None

    calls: list[tuple[str, str, str, str]] = []
    group._session_manager = type(
        "SessionMgr",
        (),
        {
            "mark_external_checkpoint": staticmethod(
                lambda session_id, checkpoint_path, reason, actor_name=None: calls.append(
                    (session_id, checkpoint_path, reason, actor_name or "")
                )
            )
        },
    )()

    monkeypatch.setattr(sys.modules[MegatronWorkerGroup.__module__].ray, "get", lambda refs, timeout=None: [{}])

    out = group.save_checkpoint("/checkpoints/alice/model_x/ckpt_7", session_id="sess-1")

    assert out == {}
    assert calls == [
        (
            "sess-1",
            "/checkpoints/alice/model_x/ckpt_7",
            "save_checkpoint",
            "megatron_qwen3_30b_a3b_instruct_2507",
        )
    ]
