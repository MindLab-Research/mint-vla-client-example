from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _touch(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_issue_317_begin_async_checkpoint_mirror_marks_pending(monkeypatch, tmp_path: Path) -> None:
    from tinker_server import checkpoints

    runtime_root = tmp_path / "runtime"
    persistent_root = tmp_path / "tos"
    cache_dir = runtime_root / "persistent_cache" / "owner-a" / "run-317" / "ckpt-a"
    _touch(cache_dir / "adapter_model.safetensors")
    _touch(cache_dir / "optimizer.pt")
    checkpoints.write_checkpoint_metadata(
        str(cache_dir),
        {
            "checkpoint_id": "ckpt-a",
            "owner_id": "owner-a",
            "model_id": "run-317",
            "checkpoint_type": "training",
            "optimizer_present": True,
            "storage_tier": "persistent_cache",
            "type": "training",
        },
    )

    kicked: list[str] = []
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(runtime_root))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(persistent_root))
    monkeypatch.setattr(checkpoints, "_kickoff_pending_checkpoint_mirrors", lambda: kicked.append("kicked"))

    persistent_path = checkpoints.begin_async_checkpoint_mirror(
        str(cache_dir),
        user_id="owner-a",
        model_id="run-317",
        checkpoint_name="ckpt-a",
    )

    meta = checkpoints.read_checkpoint_metadata(str(cache_dir))
    assert meta["mirror_status"] == checkpoints.MIRROR_STATUS_PENDING
    assert meta["persistent_mirror_path"] == persistent_path
    assert meta["checkpoint_type"] == "training"
    assert meta["type"] == "training"
    assert kicked == ["kicked"]


def test_issue_317_update_checkpoint_metadata_refuses_to_clobber_invalid_json(tmp_path: Path) -> None:
    from tinker_server import checkpoints

    cache_dir = tmp_path / "runtime" / "persistent_cache" / "owner-a" / "run-317" / "ckpt-a"
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "metadata.json"
    meta_path.write_text('{"checkpoint_type": "training"', encoding="utf-8")

    with pytest.raises(RuntimeError, match="Refusing to overwrite invalid checkpoint metadata"):
        checkpoints.update_checkpoint_metadata(
            str(cache_dir),
            {"mirror_status": checkpoints.MIRROR_STATUS_PENDING},
        )

    assert meta_path.read_text(encoding="utf-8") == '{"checkpoint_type": "training"'


def test_issue_317_process_pending_checkpoint_mirrors_updates_metadata(monkeypatch, tmp_path: Path) -> None:
    from tinker_server import checkpoints

    runtime_root = tmp_path / "runtime"
    persistent_root = tmp_path / "tos"
    cache_dir = runtime_root / "persistent_cache" / "owner-a" / "run-317" / "ckpt-a"
    _touch(cache_dir / "adapter_model.safetensors")
    _touch(cache_dir / "optimizer.pt")
    checkpoints.write_checkpoint_metadata(
        str(cache_dir),
        {
            "checkpoint_id": "ckpt-a",
            "owner_id": "owner-a",
            "model_id": "run-317",
            "checkpoint_type": "training",
            "optimizer_present": True,
            "storage_tier": "persistent_cache",
            "mirror_status": checkpoints.MIRROR_STATUS_PENDING,
            "type": "training",
        },
    )

    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(runtime_root))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(persistent_root))

    result = checkpoints.process_pending_checkpoint_mirrors()
    assert len(result["mirrored"]) == 1
    assert result["failed"] == []

    cache_meta = checkpoints.read_checkpoint_metadata(str(cache_dir))
    assert cache_meta["mirror_status"] == checkpoints.MIRROR_STATUS_COMPLETE

    persistent_dir = persistent_root / "owner-a" / "run-317" / "ckpt-a" / "training"
    persistent_meta = checkpoints.read_checkpoint_metadata(str(persistent_dir))
    assert persistent_meta["storage_tier"] == "persistent_tos"
    assert persistent_meta["mirror_status"] == checkpoints.MIRROR_STATUS_COMPLETE


def test_issue_317_reaper_keeps_pending_cache(monkeypatch, tmp_path: Path) -> None:
    from tinker_server import checkpoints

    runtime_root = tmp_path / "runtime"
    persistent_root = tmp_path / "tos"
    cache_dir = runtime_root / "persistent_cache" / "owner-a" / "run-317" / "ckpt-a"
    _touch(cache_dir / "adapter_model.safetensors")
    _touch(cache_dir / "optimizer.pt")
    checkpoints.write_checkpoint_metadata(
        str(cache_dir),
        {
            "checkpoint_id": "ckpt-a",
            "owner_id": "owner-a",
            "model_id": "run-317",
            "checkpoint_type": "training",
            "optimizer_present": True,
            "storage_tier": "persistent_cache",
            "mirror_status": checkpoints.MIRROR_STATUS_PENDING,
            "type": "training",
        },
    )

    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(runtime_root))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(persistent_root))
    monkeypatch.setenv("MINT_PERSISTENT_CHECKPOINT_CACHE_TTL_S", "1")

    old = 1_700_000_000
    import os

    os.utime(cache_dir, (old, old))

    reaped = checkpoints.reap_runtime_checkpoints(now=1_800_000_000)
    assert reaped["persistent_cache"] == []
    assert cache_dir.exists()


def test_issue_317_failed_mirror_status_is_stable(monkeypatch, tmp_path: Path) -> None:
    from tinker_server import checkpoints

    runtime_root = tmp_path / "runtime"
    persistent_root = tmp_path / "tos"
    cache_dir = runtime_root / "persistent_cache" / "owner-a" / "run-317" / "ckpt-a"
    _touch(cache_dir / "adapter_model.safetensors")
    _touch(cache_dir / "optimizer.pt")
    checkpoints.write_checkpoint_metadata(
        str(cache_dir),
        {
            "checkpoint_id": "ckpt-a",
            "owner_id": "owner-a",
            "model_id": "run-317",
            "checkpoint_type": "training",
            "optimizer_present": True,
            "storage_tier": "persistent_cache",
            "mirror_status": checkpoints.MIRROR_STATUS_FAILED,
            "mirror_error": "RuntimeError: previous failure",
            "type": "training",
        },
    )

    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(runtime_root))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(persistent_root))
    monkeypatch.setattr(
        checkpoints,
        "_process_pending_checkpoint_mirror",
        lambda _path: (_ for _ in ()).throw(AssertionError("failed mirrors must stay observable")),
    )

    result = checkpoints.process_pending_checkpoint_mirrors()
    assert result == {"mirrored": [], "failed": []}
    meta = checkpoints.read_checkpoint_metadata(str(cache_dir))
    assert meta["mirror_status"] == checkpoints.MIRROR_STATUS_FAILED
    assert meta["mirror_error"] == "RuntimeError: previous failure"


def test_issue_317_publish_not_found_stays_failed(monkeypatch, tmp_path: Path) -> None:
    from tinker_server import checkpoints
    from tinker_server.checkpoint_index import CheckpointNotFoundError

    runtime_root = tmp_path / "runtime"
    persistent_root = tmp_path / "tos"
    cache_dir = runtime_root / "persistent_cache" / "owner-a" / "run-317" / "ckpt-not-found"
    _touch(cache_dir / "adapter_model.safetensors")
    _touch(cache_dir / "optimizer.pt")
    checkpoints.write_checkpoint_metadata(
        str(cache_dir),
        {
            "checkpoint_id": "ckpt-not-found",
            "owner_id": "owner-a",
            "model_id": "run-317",
            "checkpoint_type": "training",
            "optimizer_present": True,
            "storage_tier": "persistent_cache",
            "mirror_status": checkpoints.MIRROR_STATUS_PENDING,
            "type": "training",
            "ckpt_id": "11111111-1111-1111-1111-111111111111",
        },
    )

    async def _raise_not_found(*_args, **_kwargs):
        raise CheckpointNotFoundError("staging row missing")

    async def _mark_failed(*_args, **_kwargs):
        return None

    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(runtime_root))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(persistent_root))
    monkeypatch.setattr(checkpoints, "checkpoint_index_enabled", lambda: True)
    monkeypatch.setattr(checkpoints, "publish_checkpoint_catalog", _raise_not_found)
    monkeypatch.setattr(checkpoints, "mark_checkpoint_failed", _mark_failed)

    result = checkpoints.process_pending_checkpoint_mirrors()
    assert result["mirrored"] == []
    assert str(cache_dir) in result["failed"]
    meta = checkpoints.read_checkpoint_metadata(str(cache_dir))
    assert meta["mirror_status"] == checkpoints.MIRROR_STATUS_FAILED
    assert "CheckpointNotFoundError" in str(meta.get("mirror_error"))


def test_issue_317_publish_retry_backoff_skips_immediate_retries(monkeypatch, tmp_path: Path) -> None:
    from tinker_server import checkpoints

    runtime_root = tmp_path / "runtime"
    persistent_root = tmp_path / "tos"
    cache_dir = runtime_root / "persistent_cache" / "owner-a" / "run-317" / "ckpt-retry"
    _touch(cache_dir / "adapter_model.safetensors")
    _touch(cache_dir / "optimizer.pt")
    checkpoints.write_checkpoint_metadata(
        str(cache_dir),
        {
            "checkpoint_id": "ckpt-retry",
            "owner_id": "owner-a",
            "model_id": "run-317",
            "checkpoint_type": "training",
            "optimizer_present": True,
            "storage_tier": "persistent_cache",
            "mirror_status": checkpoints.MIRROR_STATUS_PENDING,
            "type": "training",
            "ckpt_id": "22222222-2222-2222-2222-222222222222",
        },
    )

    async def _raise_publish(*_args, **_kwargs):
        raise RuntimeError("pg unavailable")

    async def _mark_failed(*_args, **_kwargs):
        return None

    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(runtime_root))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(persistent_root))
    monkeypatch.setattr(checkpoints, "checkpoint_index_enabled", lambda: True)
    monkeypatch.setattr(checkpoints, "publish_checkpoint_catalog", _raise_publish)
    monkeypatch.setattr(checkpoints, "mark_checkpoint_failed", _mark_failed)
    monkeypatch.setenv("MINT_CHECKPOINT_INDEX_PUBLISH_RETRY_S", "60")

    first = checkpoints.process_pending_checkpoint_mirrors()
    assert first["mirrored"] == []
    assert str(cache_dir) in first["failed"]

    meta = checkpoints.read_checkpoint_metadata(str(cache_dir))
    assert meta["mirror_status"] == checkpoints.MIRROR_STATUS_PENDING
    assert meta["mirror_error"] == "checkpoint_index_publish_failed"
    assert isinstance(meta.get("next_publish_retry_at"), str)

    second = checkpoints.process_pending_checkpoint_mirrors()
    assert second == {"mirrored": [], "failed": []}


def test_issue_317_list_checkpoints_includes_pending_cache_status(monkeypatch, tmp_path: Path) -> None:
    from tinker_server import checkpoints
    from tinker_server.routes import weights as wt

    root = tmp_path / "tos"
    runtime_root = tmp_path / "runtime"
    cache_dir = runtime_root / "persistent_cache" / "anonymous" / "run-317" / "ckpt-a"
    _touch(cache_dir / "adapter_model.safetensors")
    _touch(cache_dir / "optimizer.pt")
    checkpoints.write_checkpoint_metadata(
        str(cache_dir),
        {
            "checkpoint_id": "ckpt-a",
            "owner_id": None,
            "model_id": "run-317",
            "model_name": "Qwen/Qwen3-0.6B",
            "checkpoint_type": "training",
            "optimizer_present": True,
            "storage_tier": "persistent_cache",
            "mirror_status": checkpoints.MIRROR_STATUS_PENDING,
            "type": "training",
            "created_at": "2026-03-15T00:00:00Z",
        },
    )

    async def _no_remote(**_kwargs):
        return None

    async def _list_catalog_checkpoints_for_model(*, model_id: str, owner_id: str | None, is_admin: bool):
        assert model_id == "run-317"
        assert owner_id is None
        assert is_admin is False
        return [
            {
                "ckpt_id": "31700000-0000-0000-0000-000000000001",
                "owner_id": "anonymous",
                "model_id": "run-317",
                "raw_checkpoint_id": "ckpt-a",
                "checkpoint_type": "training",
                "storage_root": str(runtime_root / "persistent_cache"),
                "checkpoint_created_at": "2026-03-15T00:00:00Z",
            }
        ]

    monkeypatch.setattr(wt, "CHECKPOINTS_DIR", str(root))
    monkeypatch.setattr(wt, "PERSISTENT_CHECKPOINTS_DIR", str(root), raising=False)
    monkeypatch.setattr(wt, "RUNTIME_CHECKPOINTS_DIR", str(runtime_root), raising=False)
    monkeypatch.setattr(wt, "get_persistent_search_roots", lambda primary_root=None: [str(root)])
    monkeypatch.setattr(wt, "build_persistent_cache_dir", checkpoints.build_persistent_cache_dir)
    monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(root))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(root))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(runtime_root))
    monkeypatch.setattr(wt, "checkpoint_index_enabled", lambda: True)
    monkeypatch.setattr(wt, "list_catalog_checkpoints_for_model", _list_catalog_checkpoints_for_model)
    monkeypatch.setattr(wt, "_forward_remote_checkpoint_route", _no_remote)

    app = FastAPI()
    app.include_router(wt.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.get("/api/v1/training_runs/run-317/checkpoints")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert len(payload["checkpoints"]) == 1
    entry = payload["checkpoints"][0]
    assert entry["checkpoint_id"] == "weights/ckpt-a"
    assert entry["storage_tier"] == "persistent_cache"
    assert entry["mirror_status"] == checkpoints.MIRROR_STATUS_PENDING


def test_issue_317_list_checkpoints_finds_owner_scoped_pending_cache(
    monkeypatch, tmp_path: Path
) -> None:
    from tinker_server import checkpoints
    from tinker_server.routes import weights as wt

    root = tmp_path / "tos"
    runtime_root = tmp_path / "runtime"
    cache_dir = runtime_root / "persistent_cache" / "user-a" / "run-317" / "ckpt-a"
    _touch(cache_dir / "adapter_model.safetensors")
    _touch(cache_dir / "optimizer.pt")
    checkpoints.write_checkpoint_metadata(
        str(cache_dir),
        {
            "checkpoint_id": "ckpt-a",
            "owner_id": "user-a",
            "model_id": "run-317",
            "model_name": "Qwen/Qwen3-0.6B",
            "checkpoint_type": "training",
            "optimizer_present": True,
            "storage_tier": "persistent_cache",
            "mirror_status": checkpoints.MIRROR_STATUS_PENDING,
            "type": "training",
            "created_at": "2026-03-15T00:00:00Z",
        },
    )

    async def _no_remote(**_kwargs):
        return None

    async def _list_catalog_checkpoints_for_model(*, model_id: str, owner_id: str | None, is_admin: bool):
        assert model_id == "run-317"
        assert owner_id == "user-a"
        assert is_admin is False
        return [
            {
                "ckpt_id": "31700000-0000-0000-0000-000000000002",
                "owner_id": "user-a",
                "model_id": "run-317",
                "raw_checkpoint_id": "ckpt-a",
                "checkpoint_type": "training",
                "storage_root": str(runtime_root / "persistent_cache"),
                "checkpoint_created_at": "2026-03-15T00:00:00Z",
            }
        ]

    monkeypatch.setattr(wt, "CHECKPOINTS_DIR", str(root))
    monkeypatch.setattr(wt, "PERSISTENT_CHECKPOINTS_DIR", str(root), raising=False)
    monkeypatch.setattr(wt, "get_persistent_search_roots", lambda primary_root=None: [str(root)])
    monkeypatch.setattr(wt, "get_persistent_cache_dir", lambda: str(runtime_root / "persistent_cache"))
    monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(root))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(root))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(runtime_root))
    monkeypatch.setattr(wt, "checkpoint_index_enabled", lambda: True)
    monkeypatch.setattr(wt, "list_catalog_checkpoints_for_model", _list_catalog_checkpoints_for_model)
    monkeypatch.setattr(wt, "_forward_remote_checkpoint_route", _no_remote)
    monkeypatch.setattr(wt, "_get_user_id", lambda _request: "user-a")

    app = FastAPI()
    app.include_router(wt.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.get("/api/v1/training_runs/run-317/checkpoints")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert [entry["checkpoint_id"] for entry in payload["checkpoints"]] == ["weights/ckpt-a"]
    assert payload["checkpoints"][0]["storage_tier"] == "persistent_cache"
    assert payload["checkpoints"][0]["mirror_status"] == checkpoints.MIRROR_STATUS_PENDING


@pytest.mark.anyio
async def test_issue_317_save_state_does_not_wait_for_sampling_registration(
    monkeypatch, tmp_path: Path
) -> None:
    from tinker_server.models.types import SaveStateRequest
    from tinker_server.routes import weights as wt

    ckpt_dir = tmp_path / "training_named"
    resolved: dict[str, dict] = {}

    async def _fake_save_weights(_session, _save_path):
        _touch(ckpt_dir / "adapter_model.safetensors")
        _touch(ckpt_dir / "optimizer.pt")
        return str(ckpt_dir)

    async def _async_resolve(request_id: str, response: dict) -> None:
        resolved["request_id"] = request_id
        resolved["response"] = response

    async def _async_fail(request_id: str, error: str) -> None:
        raise AssertionError(f"unexpected async_fail({request_id}): {error}")

    class _ForbiddenInferenceManager:
        async def get_engine_for_model(self, _base_model):
            raise AssertionError("save_state must not wait for inference registration")

    monkeypatch.setattr(
        wt,
        "training_manager",
        SimpleNamespace(
            get_session=lambda _model_id: SimpleNamespace(
                model_id="run-317",
                base_model="Qwen/Qwen3-0.6B",
                current_step=11,
                backend="dense",
            ),
            mark_inflight=lambda *_args, **_kwargs: None,
        ),
    )
    monkeypatch.setattr(wt, "training_engine", SimpleNamespace(save_weights=_fake_save_weights))
    monkeypatch.setattr(
        wt,
        "task_state_futures",
        SimpleNamespace(async_resolve=_async_resolve, async_fail=_async_fail),
    )
    monkeypatch.setattr(wt, "build_persistent_cache_dir", lambda **_kwargs: str(ckpt_dir))
    monkeypatch.setattr(
        wt,
        "begin_async_checkpoint_mirror",
        lambda *_args, **_kwargs: "/tos-mindverse/tinker_checkpoints/user-a/run-317/ckpt-a",
    )
    monkeypatch.setattr(wt, "inference_manager", _ForbiddenInferenceManager())

    request = SaveStateRequest(model_id="run-317", path="ckpt-a")
    await wt._do_save_state(
        request_id="req-317-save-state",
        request=request,
        user_id="user-a",
        webhook_url=None,
        prefer_tinker=True,
    )

    assert resolved["request_id"] == "req-317-save-state"
    assert resolved["response"]["storage_tier"] == "persistent_cache"
    assert resolved["response"]["mirror_status"] == wt.MIRROR_STATUS_PENDING
    assert resolved["response"]["sampling_registered"] is False


@pytest.mark.anyio
async def test_issue_317_named_save_weights_for_sampler_preserves_type(
    monkeypatch, tmp_path: Path
) -> None:
    from tinker_server.models.types import SaveWeightsForSamplerRequest
    from tinker_server.routes import training as tr

    async def _identity_materialize(session):
        return session

    monkeypatch.setattr(tr, "_materialize_training_session_for_stateful_use", _identity_materialize)

    ckpt_dir = tmp_path / "runtime" / "persistent_cache" / "user-a" / "run-317" / "sampler-a" / "sampler"
    resolved: dict[str, dict] = {}
    save_kwargs: dict[str, object] = {}

    async def _fake_save_weights_for_sampler(**kwargs):
        import numpy as np
        from safetensors.numpy import save_file

        save_kwargs.update(kwargs)
        export_dir = Path(kwargs["checkpoint_base_dir"]) / "run-317" / "sampler-a" / "sampler"
        export_dir.mkdir(parents=True, exist_ok=True)
        save_file(
            {"lora_A.weight": np.zeros((1, 1), dtype=np.float32)},
            str(export_dir / "adapter_model.safetensors"),
        )
        return str(export_dir)

    async def _async_resolve(request_id: str, response: dict) -> None:
        resolved["request_id"] = request_id
        resolved["response"] = response

    async def _async_fail(request_id: str, error: str) -> None:
        raise AssertionError(f"unexpected async_fail({request_id}): {error}")

    monkeypatch.setattr(
        tr,
        "training_manager",
        SimpleNamespace(
            get_session=lambda _model_id: SimpleNamespace(
                model_id="run-317",
                base_model="Qwen/Qwen3-0.6B",
                current_step=9,
                backend="dense",
                lora_config=SimpleNamespace(rank=8, train_mlp=False),
            ),
            mark_inflight=lambda *_args, **_kwargs: None,
        ),
    )
    monkeypatch.setattr(
        tr,
        "training_engine",
        SimpleNamespace(save_weights_for_sampler=_fake_save_weights_for_sampler),
    )
    monkeypatch.setattr(
        tr,
        "task_state_futures",
        SimpleNamespace(async_resolve=_async_resolve, async_fail=_async_fail),
    )
    monkeypatch.setattr(tr, "build_persistent_cache_dir", lambda **_kwargs: str(ckpt_dir))
    monkeypatch.setattr(
        tr,
        "begin_async_checkpoint_mirror",
        lambda *_args, **_kwargs: "/tos-mindverse/tinker_checkpoints/user-a/run-317/sampler-a",
    )

    request = SaveWeightsForSamplerRequest(model_id="run-317", seq_id=0, path="sampler-a")
    await tr._do_save_weights_for_sampler(
        request_id="req-317-sampler",
        request=request,
        user_id="user-a",
        prefer_tinker=True,
    )

    assert resolved["request_id"] == "req-317-sampler"
    assert resolved["response"]["type"] == "save_weights_for_sampler"
    assert resolved["response"]["storage_tier"] == "persistent_cache"
    assert resolved["response"]["mirror_status"] == tr.MIRROR_STATUS_PENDING
    assert save_kwargs["checkpoint_type"] == "sampler"
    assert save_kwargs["checkpoint_base_dir"] == str(tmp_path / "runtime" / "persistent_cache" / "user-a")


@pytest.mark.anyio
async def test_issue_317_named_save_weights_for_sampler_admin_owner_is_anonymous(
    monkeypatch, tmp_path: Path
) -> None:
    from tinker_server.checkpoints import read_checkpoint_metadata
    from tinker_server.models.types import SaveWeightsForSamplerRequest
    from tinker_server.routes import training as tr

    async def _identity_materialize(session):
        return session

    monkeypatch.setattr(tr, "_materialize_training_session_for_stateful_use", _identity_materialize)

    ckpt_dir = tmp_path / "runtime" / "persistent_cache" / "anonymous" / "run-317" / "sampler-admin" / "sampler"

    async def _fake_save_weights_for_sampler(**kwargs):
        import numpy as np
        from safetensors.numpy import save_file

        export_dir = Path(kwargs["checkpoint_base_dir"]) / "run-317" / "sampler-admin" / "sampler"
        export_dir.mkdir(parents=True, exist_ok=True)
        save_file(
            {"lora_A.weight": np.zeros((1, 1), dtype=np.float32)},
            str(export_dir / "adapter_model.safetensors"),
        )
        return str(export_dir)

    async def _async_resolve(_request_id: str, _response: dict) -> None:
        return None

    async def _async_fail(request_id: str, error: str) -> None:
        raise AssertionError(f"unexpected async_fail({request_id}): {error}")

    monkeypatch.setattr(
        tr,
        "training_manager",
        SimpleNamespace(
            get_session=lambda _model_id: SimpleNamespace(
                model_id="run-317",
                base_model="Qwen/Qwen3-0.6B",
                current_step=9,
                backend="dense",
                lora_config=SimpleNamespace(rank=8, train_mlp=False),
            ),
            mark_inflight=lambda *_args, **_kwargs: None,
        ),
    )
    monkeypatch.setattr(
        tr,
        "training_engine",
        SimpleNamespace(save_weights_for_sampler=_fake_save_weights_for_sampler),
    )
    monkeypatch.setattr(
        tr,
        "task_state_futures",
        SimpleNamespace(async_resolve=_async_resolve, async_fail=_async_fail),
    )
    monkeypatch.setattr(tr, "build_persistent_cache_dir", lambda **_kwargs: str(ckpt_dir))
    monkeypatch.setattr(
        tr,
        "begin_async_checkpoint_mirror",
        lambda *_args, **_kwargs: "/tos-mindverse/tinker_checkpoints/anonymous/run-317/sampler-admin",
    )

    request = SaveWeightsForSamplerRequest(model_id="run-317", seq_id=0, path="sampler-admin")
    await tr._do_save_weights_for_sampler(
        request_id="req-317-sampler-admin",
        request=request,
        user_id="admin-user",
        prefer_tinker=True,
        is_admin=True,
    )

    metadata = read_checkpoint_metadata(str(ckpt_dir))
    assert metadata["owner_id"] is None
