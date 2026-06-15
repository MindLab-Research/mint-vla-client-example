from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

pytest.importorskip("ray")

from mint_server.backend.training.megatron.megatron_distributed import MegatronSessionStateManager, MegatronWorkerGroup


def _write_adapter(session_path: Path, *, size: int = 64) -> None:
    session_path.mkdir(parents=True, exist_ok=True)
    (session_path / "mp_rank_00_adapter.pt").write_bytes(b"a" * size)


def _write_training_checkpoint(path: Path, *, size: int = 64) -> None:
    _write_adapter(path, size=size)
    (path / "mp_rank_00_optimizer.pt").write_bytes(b"optimizer")


def _mark_checkpoint_authority(
    manager: MegatronSessionStateManager,
    session_id: str,
    checkpoint_path: Path,
    actor_name: str,
) -> None:
    manager.save_metadata(
        session_id,
        step=0,
        lr=1e-4,
        actual_rank=8,
        checkpoint_path=str(checkpoint_path),
        optimizer_restored=True,
    )
    manager.mark_external_checkpoint(
        session_id,
        checkpoint_path=str(checkpoint_path),
        reason="save_checkpoint",
        actor_name=actor_name,
    )


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
    safe_checkpoint = tmp_path / "external" / "session_safe"
    _write_training_checkpoint(safe_checkpoint, size=128)

    manager.save_metadata(
        "session_safe",
        step=1,
        lr=1e-4,
        actual_rank=8,
        checkpoint_path=str(safe_checkpoint),
    )
    _mark_checkpoint_authority(
        manager,
        "session_safe",
        safe_checkpoint,
        "shared-megatron-actor",
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
    old_checkpoint = tmp_path / "external" / "session_old_safe"
    new_checkpoint = tmp_path / "external" / "session_new_safe"
    _write_training_checkpoint(old_checkpoint, size=111)
    _write_training_checkpoint(new_checkpoint, size=222)

    _mark_checkpoint_authority(
        manager,
        "session_old_safe",
        old_checkpoint,
        "actor-a",
    )
    _mark_checkpoint_authority(
        manager,
        "session_new_safe",
        new_checkpoint,
        "actor-a",
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
        checkpoint_path = tmp_path / "external" / actor_name / session_id
        _write_training_checkpoint(checkpoint_path, size=size)
        _mark_checkpoint_authority(
            manager,
            session_id,
            checkpoint_path,
            actor_name,
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
    checkpoint_path = tmp_path / "external" / "session_saved"
    _write_training_checkpoint(checkpoint_path, size=144)
    manager.mark_actor_only_state(
        "session_saved",
        reason="forward_backward",
        actor_name="actor-a",
    )
    manager.clear_actor_only_state("session_saved")
    _mark_checkpoint_authority(
        manager,
        "session_saved",
        checkpoint_path,
        "actor-a",
    )

    usage = manager.get_cache_usage(actor_name="actor-a")

    assert usage["evictable_session_count"] == 1
    assert usage["evictable_bytes"] >= 144
    assert usage["actor_only_state_dirty_count"] == 0


def test_issue_417_missing_external_checkpoint_is_not_evictable(tmp_path: Path):
    manager = MegatronSessionStateManager(base_path=str(tmp_path / "session_store"))
    original_checkpoint = tmp_path / "external" / "deleted_checkpoint"
    _write_training_checkpoint(original_checkpoint, size=128)

    session_path = Path(
        manager.prime_session(
            "session_deleted_external",
            str(original_checkpoint),
            step=5,
            lr=2e-4,
            actual_rank=8,
            optimizer_restored=True,
        )
    )
    manager.mark_external_checkpoint(
        "session_deleted_external",
        checkpoint_path=str(original_checkpoint),
        reason="save_checkpoint",
        actor_name="actor-a",
    )
    for child in original_checkpoint.iterdir():
        child.unlink()
    original_checkpoint.rmdir()

    authority = manager.get_authority_record("session_deleted_external")
    usage = manager.get_cache_usage(actor_name="actor-a")
    result = manager.recycle_cache(max_age_s=0, max_total_bytes=1, max_bytes_per_actor=0)

    assert session_path.exists()
    assert authority.weights_source.kind == "session_cache"
    assert authority.optimizer_source.kind == "none"
    assert usage["evictable_session_count"] == 0
    assert result["evicted_sessions"] == []
    assert session_path.exists()


def test_issue_417_metadata_less_external_checkpoint_is_not_cold_safe(tmp_path: Path):
    manager = MegatronSessionStateManager(base_path=str(tmp_path / "session_store"))
    session_path = Path(manager.get_session_path("session_without_metadata"))
    _write_adapter(session_path, size=128)
    external_checkpoint = tmp_path / "external" / "metadata_less"
    _write_training_checkpoint(external_checkpoint, size=128)
    manager.mark_external_checkpoint(
        "session_without_metadata",
        checkpoint_path=str(external_checkpoint),
        reason="save_checkpoint",
        actor_name="actor-a",
    )
    _set_tree_mtime(session_path, time.time() - 7200)

    authority = manager.get_authority_record("session_without_metadata")
    usage = manager.get_cache_usage(actor_name="actor-a")
    result = manager.recycle_cache(max_age_s=0, max_total_bytes=1, max_bytes_per_actor=0)

    assert authority.weights_source.kind == "session_cache"
    assert authority.optimizer_source.kind == "none"
    assert usage["evictable_session_count"] == 0
    assert usage["no_external_checkpoint_sessions"] == ["session_without_metadata"]
    assert result["evicted_sessions"] == []
    assert session_path.exists()


def test_issue_417_external_checkpoint_identity_mismatch_is_not_evictable(tmp_path: Path):
    manager = MegatronSessionStateManager(base_path=str(tmp_path / "session_store"))
    original_checkpoint = tmp_path / "external" / "mutated_checkpoint"
    _write_training_checkpoint(original_checkpoint, size=128)

    session_path = Path(
        manager.prime_session(
            "session_mutated_external",
            str(original_checkpoint),
            step=5,
            lr=2e-4,
            actual_rank=8,
            optimizer_restored=True,
        )
    )
    manager.mark_external_checkpoint(
        "session_mutated_external",
        checkpoint_path=str(original_checkpoint),
        reason="save_checkpoint",
        actor_name="actor-a",
    )
    (original_checkpoint / "mp_rank_00_adapter.pt").write_bytes(b"changed adapter")

    authority = manager.get_authority_record("session_mutated_external")
    usage = manager.get_cache_usage(actor_name="actor-a")
    result = manager.recycle_cache(max_age_s=0, max_total_bytes=1, max_bytes_per_actor=0)

    assert session_path.exists()
    assert authority.weights_source.kind == "session_cache"
    assert authority.optimizer_source.kind == "none"
    assert usage["evictable_session_count"] == 0
    assert result["evicted_sessions"] == []
    assert session_path.exists()


def test_issue_417_checkpoint_identity_hashes_content_and_ignores_route_metadata(tmp_path: Path):
    manager = MegatronSessionStateManager(base_path=str(tmp_path / "session_store"))
    checkpoint_path = tmp_path / "external" / "checkpoint"
    _write_training_checkpoint(checkpoint_path, size=4)

    original_identity = manager.checkpoint_identity(str(checkpoint_path))
    (checkpoint_path / "metadata.json").write_text('{"owner_id": "user-a"}', encoding="utf-8")

    adapter_path = checkpoint_path / "mp_rank_00_adapter.pt"
    stat = adapter_path.stat()
    adapter_path.write_bytes(b"bbbb")
    os.utime(adapter_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    assert manager.checkpoint_identity(str(checkpoint_path)) != original_identity
    adapter_path.write_bytes(b"aaaa")
    os.utime(adapter_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert manager.checkpoint_identity(str(checkpoint_path)) == original_identity


def test_issue_417_route_metadata_write_does_not_break_external_authority(tmp_path: Path):
    manager = MegatronSessionStateManager(base_path=str(tmp_path / "session_store"))
    checkpoint_path = tmp_path / "external" / "checkpoint"
    _write_training_checkpoint(checkpoint_path, size=128)

    session_path = Path(
        manager.prime_session(
            "session_with_route_metadata",
            str(checkpoint_path),
            step=5,
            lr=2e-4,
            actual_rank=8,
            optimizer_restored=True,
        )
    )
    manager.mark_external_checkpoint(
        "session_with_route_metadata",
        checkpoint_path=str(checkpoint_path),
        reason="save_checkpoint",
        actor_name="actor-a",
    )
    (checkpoint_path / "metadata.json").write_text('{"checkpoint_type": "training"}', encoding="utf-8")

    authority = manager.get_authority_record("session_with_route_metadata")
    usage = manager.get_cache_usage(actor_name="actor-a")

    assert session_path.exists()
    assert authority.weights_source.kind == "checkpoint"
    assert authority.optimizer_source.kind == "checkpoint"
    assert usage["evictable_session_count"] == 1


def test_issue_417_optim_step_marks_live_actor_weights_authoritative(tmp_path: Path):
    manager = MegatronSessionStateManager(base_path=str(tmp_path / "session_store"))
    checkpoint_path = tmp_path / "external" / "checkpoint"
    _write_training_checkpoint(checkpoint_path, size=128)
    manager.prime_session(
        "session_after_optim",
        str(checkpoint_path),
        step=5,
        lr=2e-4,
        actual_rank=8,
        optimizer_restored=True,
    )
    manager.mark_actor_only_state(
        "session_after_optim",
        reason="optim_step",
        actor_name="actor-a",
    )

    authority = manager.get_authority_record("session_after_optim")

    assert authority.weights_source.kind == "live_actor"
    assert authority.optimizer_source.kind == "live_actor"
    assert authority.gradient_source.kind == "none"
    assert authority.gradient_source.actor_name is None
    assert authority.scheduler_source.kind == "live_actor"


def test_issue_417_forward_backward_preserves_loaded_optimizer_authority(tmp_path: Path):
    manager = MegatronSessionStateManager(base_path=str(tmp_path / "session_store"))
    checkpoint_path = tmp_path / "loaded_checkpoint"
    _write_training_checkpoint(checkpoint_path, size=128)
    manager.prime_session(
        "session_loaded_then_backward",
        str(checkpoint_path),
        step=5,
        lr=2e-4,
        actual_rank=8,
        optimizer_restored=True,
    )
    manager.mark_external_checkpoint(
        "session_loaded_then_backward",
        checkpoint_path=str(checkpoint_path),
        reason="load_checkpoint",
        actor_name="actor-a",
    )
    manager.mark_actor_only_state(
        "session_loaded_then_backward",
        reason="load_weights",
        actor_name="actor-a",
    )
    manager.mark_actor_only_state(
        "session_loaded_then_backward",
        reason="forward_backward",
        actor_name="actor-a",
    )

    authority = manager.get_authority_record("session_loaded_then_backward")

    assert authority.weights_source.kind == "checkpoint"
    assert authority.optimizer_source.kind == "live_actor"
    assert authority.gradient_source.kind == "live_actor"
    assert authority.scheduler_source.kind == "none"
    assert authority.scheduler_source.actor_name is None


def test_issue_417_load_weights_marker_records_current_checkpoint_weights(tmp_path: Path):
    manager = MegatronSessionStateManager(base_path=str(tmp_path / "session_store"))
    old_checkpoint = tmp_path / "adapter_only_checkpoint"
    new_checkpoint = tmp_path / "optimizer_checkpoint"
    _write_training_checkpoint(old_checkpoint, size=128)
    _write_training_checkpoint(new_checkpoint, size=256)
    manager.prime_session(
        "session_reloaded",
        str(old_checkpoint),
        step=0,
        lr=1e-4,
        actual_rank=8,
        optimizer_restored=False,
    )
    manager.mark_actor_only_state(
        "session_reloaded",
        reason="load_weights",
        actor_name="actor-a",
        checkpoint_path=str(new_checkpoint),
    )

    authority = manager.get_authority_record("session_reloaded")

    assert authority.weights_source.kind == "checkpoint"
    assert authority.weights_source.path == str(new_checkpoint.resolve())
    assert authority.weights_source.identity == manager.checkpoint_identity(str(new_checkpoint))
    assert authority.optimizer_source.kind == "live_actor"
    assert authority.gradient_source.kind == "none"
    assert authority.scheduler_source.kind == "none"


def test_issue_417_save_metadata_preserves_existing_train_flags(tmp_path: Path):
    manager = MegatronSessionStateManager(base_path=str(tmp_path / "session_store"))
    checkpoint_path = tmp_path / "external" / "checkpoint"
    _write_training_checkpoint(checkpoint_path, size=128)

    manager.save_metadata(
        "session_train_flags",
        step=1,
        lr=1e-4,
        actual_rank=8,
        checkpoint_path=str(checkpoint_path),
        train_attn=False,
        train_mlp=True,
        train_unembed=False,
    )
    manager.save_metadata(
        "session_train_flags",
        step=2,
        lr=2e-4,
        actual_rank=8,
        checkpoint_path=str(checkpoint_path),
    )

    metadata = manager.get_metadata("session_train_flags")
    assert metadata is not None
    assert metadata["step"] == 2
    assert metadata["lr"] == pytest.approx(2e-4)
    assert metadata["train_attn"] is False
    assert metadata["train_mlp"] is True
    assert metadata["train_unembed"] is False


def test_issue_417_authority_record_keeps_live_actor_state_non_evictable(tmp_path: Path):
    manager = MegatronSessionStateManager(base_path=str(tmp_path / "session_store"))
    checkpoint_path = tmp_path / "loaded_checkpoint"
    _write_adapter(checkpoint_path, size=128)
    (checkpoint_path / "mp_rank_00_optimizer.pt").write_bytes(b"optimizer")

    session_path = Path(
        manager.prime_session(
            "session_loaded",
            str(checkpoint_path),
            step=5,
            lr=2e-4,
            actual_rank=8,
            optimizer_restored=True,
        )
    )
    manager.mark_external_checkpoint(
        "session_loaded",
        checkpoint_path=str(checkpoint_path),
        reason="load_checkpoint",
        actor_name="actor-a",
    )
    manager.mark_actor_only_state(
        "session_loaded",
        reason="load_weights",
        actor_name="actor-a",
    )

    authority = manager.get_authority_record("session_loaded")

    assert session_path.exists()
    assert authority.weights_source.kind == "checkpoint"
    assert authority.weights_source.path == str(checkpoint_path)
    assert authority.optimizer_source.kind == "live_actor"
    assert authority.gradient_source.kind == "none"
    assert authority.gradient_source.actor_name is None
    assert authority.scheduler_source.kind == "none"
    assert authority.scheduler_source.actor_name is None

    usage = manager.get_cache_usage(actor_name="actor-a")
    assert usage["evictable_session_count"] == 0
    assert usage["actor_only_state_dirty_sessions"] == ["session_loaded"]


def test_issue_417_actor_snapshot_manifest_records_exact_sources(tmp_path: Path):
    manager = MegatronSessionStateManager(base_path=str(tmp_path / "session_store"))
    session_path = Path(manager.get_session_path("session_snapshot"))
    _write_adapter(session_path, size=128)
    snapshot_dir = session_path / "actor_only_state"
    snapshot_dir.mkdir()
    rank_path = snapshot_dir / "rank_0.pt"
    rank_path.write_bytes(b"snapshot")
    manager.save_persisted_actor_only_state(
        "session_snapshot",
        actor_name="actor-a",
        worker_entries=[
            {
                "rank": 0,
                "path": str(rank_path),
                "bytes": len(b"snapshot"),
                "gradient_kind": "consumed",
                "optimizer_state_present": True,
                "scheduler_state_present": False,
            }
        ],
    )

    authority = manager.get_authority_record("session_snapshot")

    assert authority.weights_source.kind == "session_cache"
    assert authority.optimizer_source.kind == "actor_snapshot"
    assert authority.gradient_source.kind == "none"
    assert authority.gradient_source.manifest_path is None
    assert authority.scheduler_source.kind == "none"
    assert authority.scheduler_source.manifest_path is None


def test_issue_414_load_checkpoint_without_optimizer_invalidates_existing_external_checkpoint(tmp_path: Path):
    load_path = tmp_path / "checkpoint"
    load_path.mkdir()
    (load_path / "mp_rank_00_adapter.pt").write_bytes(b"adapter")
    (load_path / "adapter_config.json").write_text(
        '{"r": 8, "target_modules": ["gate_proj", "up_proj", "down_proj"]}',
        encoding="utf-8",
    )
    (load_path / "training_meta.json").write_text('{"current_step": 5, "learning_rate": 0.0002}', encoding="utf-8")

    manager = MegatronSessionStateManager(base_path=str(tmp_path / "session_store"))
    cached_session_path = Path(manager.get_session_path("sess-2"))
    _write_adapter(cached_session_path, size=99)
    manager.mark_external_checkpoint(
        "sess-2",
        checkpoint_path="/checkpoints/actor-a/sess-2",
        reason="save_checkpoint",
        actor_name="mint_megatron_qwen3_30b_a3b_instruct_2507",
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

    usage = manager.get_cache_usage(actor_name="mint_megatron_qwen3_30b_a3b_instruct_2507")
    assert usage["stale_external_checkpoint_count"] == 1
    assert usage["evictable_session_count"] == 0


def test_issue_414_load_checkpoint_without_optimizer_stales_marker_before_reset(tmp_path: Path):
    load_path = tmp_path / "checkpoint"
    load_path.mkdir()
    (load_path / "mp_rank_00_adapter.pt").write_bytes(b"adapter")
    (load_path / "adapter_config.json").write_text(
        '{"r": 8, "target_modules": ["gate_proj", "up_proj", "down_proj"]}',
        encoding="utf-8",
    )
    (load_path / "training_meta.json").write_text('{"current_step": 5, "learning_rate": 0.0002}', encoding="utf-8")

    manager = MegatronSessionStateManager(base_path=str(tmp_path / "session_store"))
    cached_session_path = Path(manager.get_session_path("sess-3"))
    _write_adapter(cached_session_path, size=64)
    manager.mark_external_checkpoint(
        "sess-3",
        checkpoint_path="/checkpoints/actor-a/sess-3",
        reason="save_checkpoint",
        actor_name="mint_megatron_qwen3_30b_a3b_instruct_2507",
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
        actor_name="mint_megatron_qwen3_30b_a3b_instruct_2507",
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

    monkeypatch.setattr("mint_server.backend.training.megatron.megatron_distributed.ray.get", lambda refs, timeout=None: refs)

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

    monkeypatch.setattr("mint_server.backend.training.megatron.megatron_distributed.ray.get", lambda refs, timeout=None: [{}])

    out = group.save_checkpoint("/checkpoints/alice/model_x/ckpt_7", session_id="sess-1")

    assert out == {}
    assert calls == [
        (
            "sess-1",
            "/checkpoints/alice/model_x/ckpt_7",
            "save_checkpoint",
            "mint_megatron_qwen3_30b_a3b_instruct_2507",
        )
    ]
