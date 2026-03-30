from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _touch(path: Path, data: bytes = b"") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_issue_319_save_weights_for_sampler_fails_before_metadata(monkeypatch, tmp_path: Path) -> None:
    from tinker_server.models.types import SaveWeightsForSamplerRequest
    from tinker_server.routes import training as tr

    ckpt_dir = tmp_path / "sampler_missing_lora"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    failed: dict[str, str] = {}

    async def _fake_save_weights_for_sampler(**_kwargs):
        return str(ckpt_dir)

    async def _async_fail(request_id: str, error: str) -> None:
        failed["request_id"] = request_id
        failed["error"] = error

    monkeypatch.setattr(
        tr,
        "training_manager",
        SimpleNamespace(
            get_session=lambda _model_id: SimpleNamespace(
                model_id="run-319",
                base_model="Qwen/Qwen3-0.6B",
                current_step=5,
                backend="dense",
                lora_config=SimpleNamespace(rank=8, train_mlp=False),
            ),
            mark_inflight=lambda _model_id, _delta: None,
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
        SimpleNamespace(resolve=lambda *_args, **_kwargs: None, async_fail=_async_fail),
    )
    monkeypatch.setattr(tr, "build_persistent_cache_dir", lambda **_kwargs: str(ckpt_dir))
    request = SaveWeightsForSamplerRequest(model_id="run-319", seq_id=0, path="sampler-bad")
    await tr._do_save_weights_for_sampler(
        request_id="req-319-sampler",
        request=request,
        user_id="owner-a",
        prefer_tinker=True,
    )

    assert failed["request_id"] == "req-319-sampler"
    assert "invalid sampler checkpoint" in failed["error"]
    assert not (ckpt_dir / "metadata.json").exists()


@pytest.mark.anyio
async def test_issue_319_save_weights_for_sampler_rejects_corrupt_safetensors(
    monkeypatch, tmp_path: Path
) -> None:
    from tinker_server.models.types import SaveWeightsForSamplerRequest
    from tinker_server.routes import training as tr

    ckpt_dir = tmp_path / "sampler_corrupt_lora"
    _touch(ckpt_dir / "adapter_model.safetensors", b"")

    failed: dict[str, str] = {}

    async def _fake_save_weights_for_sampler(**_kwargs):
        return str(ckpt_dir)

    async def _async_fail(request_id: str, error: str) -> None:
        failed["request_id"] = request_id
        failed["error"] = error

    monkeypatch.setattr(
        tr,
        "training_manager",
        SimpleNamespace(
            get_session=lambda _model_id: SimpleNamespace(
                model_id="run-319",
                base_model="Qwen/Qwen3-0.6B",
                current_step=5,
                backend="dense",
                lora_config=SimpleNamespace(rank=8, train_mlp=False),
            ),
            mark_inflight=lambda _model_id, _delta: None,
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
        SimpleNamespace(resolve=lambda *_args, **_kwargs: None, async_fail=_async_fail),
    )
    monkeypatch.setattr(tr, "build_persistent_cache_dir", lambda **_kwargs: str(ckpt_dir))
    request = SaveWeightsForSamplerRequest(model_id="run-319", seq_id=0, path="sampler-corrupt")
    await tr._do_save_weights_for_sampler(
        request_id="req-319-sampler-corrupt",
        request=request,
        user_id="owner-a",
        prefer_tinker=True,
    )

    assert failed["request_id"] == "req-319-sampler-corrupt"
    assert "Unreadable adapter_model.safetensors" in failed["error"]
    assert not (ckpt_dir / "metadata.json").exists()


@pytest.mark.anyio
async def test_issue_319_save_state_fails_before_metadata(monkeypatch, tmp_path: Path) -> None:
    from tinker_server.models.types import SaveStateRequest
    from tinker_server.routes import weights as wt

    ckpt_dir = tmp_path / "training_missing_lora"
    _touch(ckpt_dir / "optimizer.pt", b"optimizer")

    failed: dict[str, str] = {}

    async def _fake_save_weights(_session, _save_path):
        return str(ckpt_dir)

    async def _async_fail(request_id: str, error: str) -> None:
        failed["request_id"] = request_id
        failed["error"] = error

    monkeypatch.setattr(
        wt,
        "training_manager",
        SimpleNamespace(
            get_session=lambda _model_id: SimpleNamespace(
                model_id="run-319",
                base_model="Qwen/Qwen3-0.6B",
                current_step=7,
                backend="dense",
            ),
            mark_inflight=lambda _model_id, _delta: None,
        ),
    )
    monkeypatch.setattr(wt, "training_engine", SimpleNamespace(save_weights=_fake_save_weights))
    monkeypatch.setattr(
        wt,
        "future_store",
        SimpleNamespace(resolve=lambda *_args, **_kwargs: None, async_fail=_async_fail),
    )
    monkeypatch.setattr(wt, "build_persistent_cache_dir", lambda **_kwargs: str(ckpt_dir))
    request = SaveStateRequest(model_id="run-319", path="training-bad")
    await wt._do_save_state(
        request_id="req-319-training",
        request=request,
        user_id="owner-a",
        webhook_url=None,
        prefer_tinker=False,
    )

    assert failed["request_id"] == "req-319-training"
    assert "invalid training checkpoint" in failed["error"]
    assert not (ckpt_dir / "metadata.json").exists()


def test_issue_319_list_checkpoints_skips_invalid_sampler_dirs(monkeypatch, tmp_path: Path) -> None:
    from tinker_server.checkpoints import write_checkpoint_metadata
    from tinker_server.routes import weights as wt

    root = tmp_path / "checkpoints"
    model_id = "run-319"
    owner_dir = root / "anonymous" / model_id

    bad = owner_dir / "sampler-bad"
    bad.mkdir(parents=True, exist_ok=True)
    write_checkpoint_metadata(
        str(bad),
        {
            "checkpoint_id": "sampler-bad",
            "owner_id": None,
            "model_id": model_id,
            "model_name": "Qwen/Qwen3-0.6B",
            "created_at": "2026-03-14T00:00:00Z",
            "step": 5,
            "checkpoint_type": "sampler",
            "optimizer_present": False,
            "backend": "dense",
            "type": "sampler",
        },
    )

    good = owner_dir / "training-good"
    _touch(good / "adapter_model.safetensors", b"lora")
    _touch(good / "optimizer.pt", b"optimizer")
    write_checkpoint_metadata(
        str(good),
        {
            "checkpoint_id": "training-good",
            "owner_id": None,
            "model_id": model_id,
            "model_name": "Qwen/Qwen3-0.6B",
            "created_at": "2026-03-14T00:00:00Z",
            "step": 6,
            "checkpoint_type": "training",
            "optimizer_present": True,
            "backend": "dense",
            "type": "training",
        },
    )

    async def _no_remote(**_kwargs):
        return None

    monkeypatch.setattr(wt, "CHECKPOINTS_DIR", str(root))
    monkeypatch.setattr(wt, "get_persistent_search_roots", lambda primary_root=None: [str(root)])
    monkeypatch.setattr(wt, "_forward_remote_checkpoint_route", _no_remote)

    app = FastAPI()
    app.include_router(wt.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.get(f"/api/v1/training_runs/{model_id}/checkpoints")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    checkpoint_ids = [item["checkpoint_id"] for item in payload["checkpoints"]]

    assert checkpoint_ids == ["weights/training-good"]
    assert "sampler_weights/sampler-bad" not in checkpoint_ids


def test_issue_319_list_checkpoints_skips_shard_only_sampler_dirs(monkeypatch, tmp_path: Path) -> None:
    from tinker_server.checkpoints import write_checkpoint_metadata
    from tinker_server.routes import weights as wt

    root = tmp_path / "checkpoints"
    model_id = "run-319-mismatch"
    owner_dir = root / "anonymous" / model_id

    shard_only = owner_dir / "sampler-sharded-only"
    _touch(shard_only / "mp_rank_00_adapter.pt", b"lora-shard")
    write_checkpoint_metadata(
        str(shard_only),
        {
            "checkpoint_id": "sampler-sharded-only",
            "owner_id": None,
            "model_id": model_id,
            "model_name": "Qwen/Qwen3-0.6B",
            "created_at": "2026-03-14T00:00:00Z",
            "step": 8,
            "checkpoint_type": "sampler",
            "optimizer_present": False,
            "backend": "megatron",
            "type": "sampler",
        },
    )

    async def _no_remote(**_kwargs):
        return None

    monkeypatch.setattr(wt, "CHECKPOINTS_DIR", str(root))
    monkeypatch.setattr(wt, "get_persistent_search_roots", lambda primary_root=None: [str(root)])
    monkeypatch.setattr(wt, "_forward_remote_checkpoint_route", _no_remote)

    app = FastAPI()
    app.include_router(wt.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.get(f"/api/v1/training_runs/{model_id}/checkpoints")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["checkpoints"] == []


def test_issue_319_list_checkpoints_skips_corrupt_sampler_dirs(monkeypatch, tmp_path: Path) -> None:
    from tinker_server.checkpoints import write_checkpoint_metadata
    from tinker_server.routes import weights as wt

    root = tmp_path / "checkpoints"
    model_id = "run-319-corrupt"
    owner_dir = root / "anonymous" / model_id

    corrupt = owner_dir / "sampler-corrupt"
    _touch(corrupt / "adapter_model.safetensors", b"")
    write_checkpoint_metadata(
        str(corrupt),
        {
            "checkpoint_id": "sampler-corrupt",
            "owner_id": None,
            "model_id": model_id,
            "model_name": "Qwen/Qwen3-0.6B",
            "created_at": "2026-03-14T00:00:00Z",
            "step": 9,
            "checkpoint_type": "sampler",
            "optimizer_present": False,
            "backend": "dense",
            "type": "sampler",
        },
    )

    async def _no_remote(**_kwargs):
        return None

    monkeypatch.setattr(wt, "CHECKPOINTS_DIR", str(root))
    monkeypatch.setattr(wt, "get_persistent_search_roots", lambda primary_root=None: [str(root)])
    monkeypatch.setattr(wt, "_forward_remote_checkpoint_route", _no_remote)

    app = FastAPI()
    app.include_router(wt.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.get(f"/api/v1/training_runs/{model_id}/checkpoints")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["checkpoints"] == []
