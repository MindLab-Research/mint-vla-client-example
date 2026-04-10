from __future__ import annotations

import os
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


def test_issue_414_stale_external_checkpoint_is_not_evictable(tmp_path: Path):
    manager = MegatronSessionStateManager(base_path=str(tmp_path))

    session_path = Path(manager.get_session_path("session_stale"))
    _write_adapter(session_path, size=144)
    manager.mark_external_checkpoint(
        "session_stale",
        checkpoint_path="/checkpoints/actor-a/session_stale",
        reason="save_checkpoint",
        actor_name="actor-a",
    )
    manager.invalidate_external_checkpoint("session_stale", reason="forward_backward")
    _set_tree_mtime(session_path, time.time() - 7200)

    usage = manager.get_cache_usage(actor_name="actor-a")
    assert usage["stale_external_checkpoint_count"] == 1
    assert usage["stale_external_checkpoint_sessions"] == ["session_stale"]
    assert usage["evictable_session_count"] == 0

    result = manager.recycle_cache(max_age_s=3600, max_total_bytes=0, max_bytes_per_actor=0)

    assert result["evicted_sessions"] == []
    assert session_path.exists()
    assert result["after"]["stale_external_checkpoint_sessions"] == ["session_stale"]


def test_issue_414_fresh_external_checkpoint_restores_evictability(tmp_path: Path):
    manager = MegatronSessionStateManager(base_path=str(tmp_path))

    session_path = Path(manager.get_session_path("session_saved"))
    _write_adapter(session_path, size=144)
    manager.mark_actor_only_state(
        "session_saved",
        reason="forward_backward",
        actor_name="actor-a",
    )
    manager.mark_external_checkpoint(
        "session_saved",
        checkpoint_path="/checkpoints/actor-a/session_saved",
        reason="save_checkpoint",
        actor_name="actor-a",
    )

    usage = manager.get_cache_usage(actor_name="actor-a")

    assert usage["evictable_session_count"] == 1
    assert usage["evictable_bytes"] >= 144
    assert usage["actor_only_state_dirty_count"] == 0


def test_issue_414_load_checkpoint_without_optimizer_invalidates_existing_external_checkpoint(tmp_path: Path):
    load_path = tmp_path / "checkpoint"
    load_path.mkdir()
    (load_path / "mp_rank_00_adapter.pt").write_bytes(b"adapter")
    (load_path / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
    (load_path / "training_meta.json").write_text('{"current_step": 5, "learning_rate": 0.0002}', encoding="utf-8")

    manager = MegatronSessionStateManager(base_path=str(tmp_path / "session_store"))
    cached_session_path = Path(manager.get_session_path("sess-2"))
    _write_adapter(cached_session_path, size=99)
    manager.mark_external_checkpoint(
        "sess-2",
        checkpoint_path="/checkpoints/actor-a/sess-2",
        reason="save_checkpoint",
        actor_name="megatron_qwen3_30b_a3b_instruct_2507",
    )

    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)
    group.workers = []
    group.base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    group.learning_rate = 1e-4
    group._step_count = 0
    group.lora_rank = 8
    group._bind_traceparent = lambda traceparent: None
    group._resolve_required_session_id = lambda session_id, op: session_id
    group._prepare_session_for_explicit_load = lambda session_id, traceparent=None: None
    group.load_adapter_state = lambda *args, **kwargs: {"status": "ok"}
    group.reset_optimizer = lambda learning_rate, traceparent=None, zero_grad_buffers=True: {"status": "ok", "learning_rate": learning_rate}
    group._session_manager = manager

    out = group.load_checkpoint(str(load_path), load_optimizer=False, session_id="sess-2")

    assert out["optimizer_restored"] is False
    assert out["optimizer_reset"] is True
    marker = manager.get_external_checkpoint("sess-2")
    assert marker is not None
    assert marker["is_fresh"] is False
    assert marker["invalidated_reason"] == "load_checkpoint_without_optimizer"

    usage = manager.get_cache_usage(actor_name="megatron_qwen3_30b_a3b_instruct_2507")
    assert usage["stale_external_checkpoint_count"] == 1
    assert usage["evictable_session_count"] == 0


def test_issue_414_load_checkpoint_without_optimizer_stales_marker_before_reset(tmp_path: Path):
    load_path = tmp_path / "checkpoint"
    load_path.mkdir()
    (load_path / "mp_rank_00_adapter.pt").write_bytes(b"adapter")
    (load_path / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
    (load_path / "training_meta.json").write_text('{"current_step": 5, "learning_rate": 0.0002}', encoding="utf-8")

    manager = MegatronSessionStateManager(base_path=str(tmp_path / "session_store"))
    cached_session_path = Path(manager.get_session_path("sess-3"))
    _write_adapter(cached_session_path, size=64)
    manager.mark_external_checkpoint(
        "sess-3",
        checkpoint_path="/checkpoints/actor-a/sess-3",
        reason="save_checkpoint",
        actor_name="megatron_qwen3_30b_a3b_instruct_2507",
    )

    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)
    group.workers = []
    group.base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    group.learning_rate = 1e-4
    group._step_count = 0
    group.lora_rank = 8
    group._bind_traceparent = lambda traceparent: None
    group._resolve_required_session_id = lambda session_id, op: session_id
    group._prepare_session_for_explicit_load = lambda session_id, traceparent=None: None
    group.load_adapter_state = lambda *args, **kwargs: {"status": "ok"}
    group.reset_optimizer = lambda learning_rate, traceparent=None, zero_grad_buffers=True: (_ for _ in ()).throw(RuntimeError("reset failed"))
    group._session_manager = manager

    with pytest.raises(RuntimeError, match="reset failed"):
        group.load_checkpoint(str(load_path), load_optimizer=False, session_id="sess-3")

    marker = manager.get_external_checkpoint("sess-3")
    assert marker is not None
    assert marker["is_fresh"] is False
    assert marker["invalidated_reason"] == "load_checkpoint_without_optimizer"


def test_issue_414_reinit_lora_weights_invalidates_existing_external_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manager = MegatronSessionStateManager(base_path=str(tmp_path / "session_store"))
    session_path = Path(manager.get_session_path("sess-4"))
    _write_adapter(session_path, size=64)
    manager.mark_external_checkpoint(
        "sess-4",
        checkpoint_path="/checkpoints/actor-a/sess-4",
        reason="save_checkpoint",
        actor_name="megatron_qwen3_30b_a3b_instruct_2507",
    )

    class _RemoteCall:
        def __init__(self, result):
            self._result = result

        def remote(self, *args, **kwargs):
            return self._result

    class _Worker:
        reinit_lora_weights = _RemoteCall({"reinit_count": 1, "opt_state_reset": 1, "lr_updated": False})

    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)
    group.workers = [_Worker()]
    group.base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    group.learning_rate = 1e-4
    group._step_count = 7
    group._actual_rank = 8
    group.lora_rank = 8
    group._current_session = "sess-4"
    group._bind_traceparent = lambda traceparent: None
    group._session_manager = manager

    monkeypatch.setattr("tinker_server.backend.megatron_distributed.ray.get", lambda refs, timeout=None: refs)

    out = group.reinit_lora_weights()

    assert out["status"] == "ok"
    marker = manager.get_external_checkpoint("sess-4")
    assert marker is not None
    assert marker["is_fresh"] is False
    assert marker["invalidated_reason"] == "reinit_lora_weights"


def test_issue_414_save_checkpoint_marks_external_checkpoint(monkeypatch: pytest.MonkeyPatch):
    class _RemoteSaveCheckpoint:
        def remote(self, *args, **kwargs):
            return {}

    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)
    group.workers = [type("W", (), {"save_checkpoint": _RemoteSaveCheckpoint()})()]
    group.base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    group.config = type("Cfg", (), {"world_size": 1})()
    group._step_count = 0
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

    monkeypatch.setattr("tinker_server.backend.megatron_distributed.ray.get", lambda refs, timeout=None: [{}])

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
