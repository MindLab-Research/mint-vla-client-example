from __future__ import annotations

from types import SimpleNamespace
import json
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
    assert kicked == ["kicked"]


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

    persistent_dir = persistent_root / "owner-a" / "run-317" / "ckpt-a"
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

    monkeypatch.setattr(wt, "CHECKPOINTS_DIR", str(root))
    monkeypatch.setattr(wt, "PERSISTENT_CHECKPOINTS_DIR", str(root), raising=False)
    monkeypatch.setattr(wt, "RUNTIME_CHECKPOINTS_DIR", str(runtime_root), raising=False)
    monkeypatch.setattr(wt, "get_persistent_search_roots", lambda primary_root=None: [str(root)])
    monkeypatch.setattr(wt, "build_persistent_cache_dir", checkpoints.build_persistent_cache_dir)
    monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(root))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(root))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(runtime_root))
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

    monkeypatch.setattr(wt, "CHECKPOINTS_DIR", str(root))
    monkeypatch.setattr(wt, "PERSISTENT_CHECKPOINTS_DIR", str(root), raising=False)
    monkeypatch.setattr(wt, "get_persistent_search_roots", lambda primary_root=None: [str(root)])
    monkeypatch.setattr(wt, "get_persistent_cache_dir", lambda: str(runtime_root / "persistent_cache"))
    monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(root))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(root))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(runtime_root))
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

    def _resolve(request_id: str, response: dict) -> None:
        resolved["request_id"] = request_id
        resolved["response"] = response

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
                )
            ),
        )
    monkeypatch.setattr(wt, "training_engine", SimpleNamespace(save_weights=_fake_save_weights))
    monkeypatch.setattr(
        wt,
        "future_store",
        SimpleNamespace(resolve=_resolve, fail=lambda *_args, **_kwargs: None),
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

    ckpt_dir = tmp_path / "sampler_named"
    resolved: dict[str, dict] = {}

    async def _fake_save_weights_for_sampler(**_kwargs):
        import numpy as np
        from safetensors.numpy import save_file

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_file({"lora_A.weight": np.zeros((1, 1), dtype=np.float32)}, str(ckpt_dir / "adapter_model.safetensors"))
        return str(ckpt_dir)

    def _resolve(request_id: str, response: dict) -> None:
        resolved["request_id"] = request_id
        resolved["response"] = response

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
            )
        ),
    )
    monkeypatch.setattr(
        tr,
        "training_engine",
        SimpleNamespace(save_weights_for_sampler=_fake_save_weights_for_sampler),
    )
    monkeypatch.setattr(
        tr,
        "future_store",
        SimpleNamespace(resolve=_resolve, fail=lambda *_args, **_kwargs: None),
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
