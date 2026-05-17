import json
from pathlib import Path


def test_issue_18_resolve_and_materialize_persistent_checkpoint(tmp_path, monkeypatch) -> None:
    from tinker_server import checkpoints

    legacy_root = tmp_path / "legacy"
    persistent_root = tmp_path / "tos"
    runtime_root = tmp_path / "runtime"
    checkpoint_dir = persistent_root / "owner-a" / "run-18" / "ckpt-final"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "adapter_model.safetensors").write_text("x", encoding="utf-8")
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "ckpt-final",
                "owner_id": "owner-a",
                "model_id": "run-18",
                "checkpoint_type": "sampler",
                "optimizer_present": False,
                "type": "sampler",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(legacy_root))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(persistent_root))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(runtime_root))

    resolved = checkpoints.resolve_checkpoint_path(
        "mint://run-18/sampler_weights/ckpt-final",
        user_id="owner-a",
    )
    assert resolved == str(checkpoint_dir)

    local_path = checkpoints.materialize_persistent_checkpoint(resolved)
    assert local_path == str(runtime_root / "persistent_cache" / "owner-a" / "run-18" / "ckpt-final")
    assert (Path(local_path) / "adapter_model.safetensors").exists()


def test_issue_18_resolve_prefers_cache_when_persistent_view_is_partial(tmp_path, monkeypatch) -> None:
    from tinker_server import checkpoints

    persistent_root = tmp_path / "tos"
    runtime_root = tmp_path / "runtime"

    persistent_dir = persistent_root / "owner-a" / "run-18" / "ckpt-race"
    persistent_dir.mkdir(parents=True)
    (persistent_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "ckpt-race",
                "owner_id": "owner-a",
                "model_id": "run-18",
                "checkpoint_type": "training",
                "optimizer_present": True,
                "type": "training",
            }
        ),
        encoding="utf-8",
    )

    cache_dir = runtime_root / "persistent_cache" / "owner-a" / "run-18" / "ckpt-race"
    cache_dir.mkdir(parents=True)
    (cache_dir / "adapter_model.safetensors").write_text("x", encoding="utf-8")
    (cache_dir / "mp_rank_00_optimizer.pt").write_text("x", encoding="utf-8")
    (cache_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "ckpt-race",
                "owner_id": "owner-a",
                "model_id": "run-18",
                "checkpoint_type": "training",
                "optimizer_present": True,
                "type": "training",
                "storage_tier": "persistent_cache",
                "mirror_status": "pending",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(persistent_root))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(persistent_root))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(runtime_root))

    resolved = checkpoints.resolve_checkpoint_path(
        "mint://run-18/weights/ckpt-race",
        user_id="owner-a",
    )
    assert resolved == str(cache_dir)


def test_issue_18_reap_runtime_checkpoints(tmp_path, monkeypatch) -> None:
    from tinker_server import checkpoints

    persistent_root = tmp_path / "tos"
    runtime_root = tmp_path / "runtime"
    ephemeral = runtime_root / "ephemeral" / "owner-a" / "run-18" / "_ephemeral_dead"
    cache = runtime_root / "persistent_cache" / "owner-a" / "run-18" / "ckpt-final"
    persistent = persistent_root / "owner-a" / "run-18" / "ckpt-expired"
    ephemeral.mkdir(parents=True)
    cache.mkdir(parents=True)
    persistent.mkdir(parents=True)
    (ephemeral / "adapter_model.safetensors").write_text("x", encoding="utf-8")
    (cache / "adapter_model.safetensors").write_text("x", encoding="utf-8")
    (persistent / "adapter_model.safetensors").write_text("x", encoding="utf-8")
    (persistent / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "ckpt-expired",
                "owner_id": "owner-a",
                "model_id": "run-18",
                "checkpoint_type": "sampler",
                "optimizer_present": False,
                "type": "sampler",
                "created_at": "2026-03-01T00:00:00Z",
                "ttl_seconds": 1,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(persistent_root))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(runtime_root))
    monkeypatch.setenv("MINT_EPHEMERAL_CHECKPOINT_TTL_S", "1")
    monkeypatch.setenv("MINT_PERSISTENT_CHECKPOINT_CACHE_TTL_S", "1")

    old_time = 1_700_000_000
    import os

    os.utime(ephemeral, (old_time, old_time))
    os.utime(cache, (old_time, old_time))

    reaped = checkpoints.reap_runtime_checkpoints(now=1_800_000_000)
    assert str(ephemeral) in reaped["ephemeral"]
    assert str(cache) in reaped["persistent_cache"]
    assert str(persistent) in reaped["persistent"]


def test_issue_18_no_legacy_fallback_resolution(tmp_path, monkeypatch) -> None:
    from tinker_server import checkpoints

    legacy_root = tmp_path / "legacy"
    persistent_root = tmp_path / "tos"
    runtime_root = tmp_path / "runtime"
    legacy_dir = legacy_root / "owner-a" / "run-18" / "ckpt-final"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "adapter_model.safetensors").write_text("x", encoding="utf-8")
    (legacy_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "ckpt-final",
                "owner_id": "owner-a",
                "model_id": "run-18",
                "checkpoint_type": "sampler",
                "optimizer_present": False,
                "type": "sampler",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(persistent_root))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(persistent_root))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(runtime_root))
    monkeypatch.setenv("MINT_LEGACY_CHECKPOINT_DIRS", str(legacy_root))

    unresolved = checkpoints.resolve_checkpoint_path(
        "mint://run-18/sampler_weights/ckpt-final",
        user_id="owner-a",
    )
    assert unresolved == str(persistent_root / "owner-a" / "run-18" / "ckpt-final" / "sampler")


def test_issue_18_session_manager_rejects_checkpoint_uri() -> None:
    from tinker_server.backend.session_manager import SessionManager

    manager = SessionManager()
    try:
        manager._resolve_model_path("mint://run-18/sampler_weights/ckpt-final")
    except ValueError as e:
        assert "must be resolved" in str(e)
    else:
        raise AssertionError("expected ValueError for raw checkpoint URI")
