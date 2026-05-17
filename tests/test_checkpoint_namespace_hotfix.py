import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from safetensors.numpy import save_file


def _mk_checkpoint_view(
    root: Path,
    *,
    owner: str,
    run_id: str,
    name: str,
    checkpoint_type: str,
    typed: bool,
) -> Path:
    ckpt_root = root / owner / run_id / name
    ckpt_dir = ckpt_root / checkpoint_type if typed else ckpt_root
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_file({"lora": np.zeros((1,), dtype=np.float32)}, str(ckpt_dir / "adapter_model.safetensors"))
    if checkpoint_type == "training":
        (ckpt_dir / "optimizer.pt").write_bytes(b"dummy-optim")
    (ckpt_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": name,
                "owner_id": None if owner == "anonymous" else owner,
                "model_id": run_id,
                "model_name": "dummy",
                "created_at": "2026-04-07T00:00:00Z",
                "step": 1,
                "checkpoint_type": checkpoint_type,
                "optimizer_present": checkpoint_type == "training",
                "backend": "dense",
                "type": checkpoint_type,
            }
        ),
        encoding="utf-8",
    )
    return ckpt_dir


def test_checkpoint_namespace_resolution_prefers_typed_view(tmp_path, monkeypatch) -> None:
    from tinker_server import checkpoints

    persistent_root = tmp_path / "tos"
    runtime_root = tmp_path / "runtime"
    training_dir = _mk_checkpoint_view(
        persistent_root,
        owner="owner-a",
        run_id="run-hotfix",
        name="0001",
        checkpoint_type="training",
        typed=True,
    )
    sampler_dir = _mk_checkpoint_view(
        persistent_root,
        owner="owner-a",
        run_id="run-hotfix",
        name="0001",
        checkpoint_type="sampler",
        typed=True,
    )

    monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(persistent_root))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(persistent_root))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(runtime_root))

    assert checkpoints.resolve_checkpoint_path(
        "mint://run-hotfix/weights/0001",
        user_id="owner-a",
    ) == str(training_dir)
    assert checkpoints.resolve_checkpoint_path(
        "mint://run-hotfix/sampler_weights/0001",
        user_id="owner-a",
    ) == str(sampler_dir)


def test_checkpoint_namespace_resolution_falls_back_to_legacy_flat_dir(tmp_path, monkeypatch) -> None:
    from tinker_server import checkpoints

    persistent_root = tmp_path / "tos"
    runtime_root = tmp_path / "runtime"
    legacy_dir = _mk_checkpoint_view(
        persistent_root,
        owner="owner-a",
        run_id="run-hotfix",
        name="0002",
        checkpoint_type="sampler",
        typed=False,
    )

    monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(persistent_root))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(persistent_root))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(runtime_root))

    assert checkpoints.resolve_checkpoint_path(
        "mint://run-hotfix/sampler_weights/0002",
        user_id="owner-a",
    ) == str(legacy_dir)


def test_checkpoint_namespace_resolution_allows_legacy_anonymous_cache(tmp_path, monkeypatch) -> None:
    from tinker_server import checkpoints

    persistent_root = tmp_path / "tos"
    runtime_root = tmp_path / "runtime"
    anonymous_dir = _mk_checkpoint_view(
        runtime_root / "persistent_cache",
        owner="anonymous",
        run_id="run-hotfix",
        name="0003",
        checkpoint_type="sampler",
        typed=True,
    )

    monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(persistent_root))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(persistent_root))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(runtime_root))

    assert checkpoints.resolve_checkpoint_path(
        "mint://run-hotfix/sampler_weights/0003",
        user_id="owner-a",
    ) == str(anonymous_dir)


def test_checkpoint_namespace_rejects_untyped_uri(tmp_path, monkeypatch) -> None:
    from tinker_server import checkpoints

    persistent_root = tmp_path / "tos"
    runtime_root = tmp_path / "runtime"
    _mk_checkpoint_view(
        persistent_root,
        owner="owner-a",
        run_id="run-hotfix",
        name="0002",
        checkpoint_type="sampler",
        typed=False,
    )

    monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(persistent_root))
    monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(persistent_root))
    monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(runtime_root))

    with pytest.raises(ValueError, match="explicit checkpoint type"):
        checkpoints.resolve_checkpoint_path(
            "mint://run-hotfix/0002",
            user_id="owner-a",
        )


def test_checkpoint_namespace_reap_typed_ephemeral_leaf(monkeypatch) -> None:
    from tinker_server import checkpoints

    with TemporaryDirectory() as td:
        root = Path(td)
        runtime_root = root / "runtime"
        persistent_root = root / "persistent"
        ephemeral_dir = _mk_checkpoint_view(
            runtime_root / "ephemeral",
            owner="owner-a",
            run_id="run-hotfix",
            name="_ephemeral_dead",
            checkpoint_type="sampler",
            typed=True,
        )

        monkeypatch.setattr(checkpoints, "CHECKPOINTS_DIR", str(persistent_root))
        monkeypatch.setattr(checkpoints, "PERSISTENT_CHECKPOINTS_DIR", str(persistent_root))
        monkeypatch.setattr(checkpoints, "RUNTIME_CHECKPOINTS_DIR", str(runtime_root))
        monkeypatch.setenv("MINT_EPHEMERAL_CHECKPOINT_TTL_S", "1")

        old_time = 1_700_000_000
        import os

        os.utime(ephemeral_dir, (old_time, old_time))

        reaped = checkpoints.reap_runtime_checkpoints(now=1_800_000_000)
        assert str(ephemeral_dir) in reaped["ephemeral"]


def test_checkpoint_namespace_list_exposes_training_and_sampler_views(tmp_path: Path, monkeypatch) -> None:
    from tinker_server import checkpoints
    from tinker_server.routes import weights as weights_routes

    weights_routes.CHECKPOINTS_DIR = str(tmp_path)
    checkpoints.CHECKPOINTS_DIR = str(tmp_path)
    checkpoints.PERSISTENT_CHECKPOINTS_DIR = str(tmp_path)
    training_dir = _mk_checkpoint_view(
        tmp_path,
        owner="anonymous",
        run_id="run-hotfix",
        name="0003",
        checkpoint_type="training",
        typed=True,
    )
    sampler_dir = _mk_checkpoint_view(
        tmp_path,
        owner="anonymous",
        run_id="run-hotfix",
        name="0003",
        checkpoint_type="sampler",
        typed=True,
    )

    async def _no_remote(**_kwargs):
        return None

    async def _list_catalog_checkpoints_for_model(model_id: str, *, owner_id: str | None, is_admin: bool):
        assert model_id == "run-hotfix"
        assert owner_id is None
        assert is_admin is False
        return [
            {
                "ckpt_id": "11111111-1111-1111-1111-111111111111",
                "owner_id": "anonymous",
                "model_id": "run-hotfix",
                "raw_checkpoint_id": "0003",
                "checkpoint_type": "training",
                "storage_root": str(tmp_path),
                "checkpoint_created_at": "2026-04-07T00:00:00Z",
            },
            {
                "ckpt_id": "22222222-2222-2222-2222-222222222222",
                "owner_id": "anonymous",
                "model_id": "run-hotfix",
                "raw_checkpoint_id": "0003",
                "checkpoint_type": "sampler",
                "storage_root": str(tmp_path),
                "checkpoint_created_at": "2026-04-07T00:00:00Z",
            },
        ]

    monkeypatch.setattr(weights_routes, "checkpoint_index_enabled", lambda: True)
    monkeypatch.setattr(weights_routes, "list_catalog_checkpoints_for_model", _list_catalog_checkpoints_for_model)
    monkeypatch.setattr(weights_routes, "_forward_remote_checkpoint_route", _no_remote)

    app = FastAPI()
    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.get("/api/v1/training_runs/run-hotfix/checkpoints")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    ids = {item["checkpoint_id"] for item in payload["checkpoints"]}
    assert ids == {"weights/0003", "sampler_weights/0003"}


def test_checkpoint_namespace_archive_prefers_requested_type(tmp_path: Path) -> None:
    from tinker_server import checkpoints
    from tinker_server.routes import weights as weights_routes

    weights_routes.CHECKPOINTS_DIR = str(tmp_path)
    checkpoints.CHECKPOINTS_DIR = str(tmp_path)
    checkpoints.PERSISTENT_CHECKPOINTS_DIR = str(tmp_path)
    _mk_checkpoint_view(
        tmp_path,
        owner="anonymous",
        run_id="run-hotfix",
        name="0004",
        checkpoint_type="training",
        typed=True,
    )
    _mk_checkpoint_view(
        tmp_path,
        owner="anonymous",
        run_id="run-hotfix",
        name="0004",
        checkpoint_type="sampler",
        typed=True,
    )

    app = FastAPI()
    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.get(
        "/api/v1/training_runs/run-hotfix/checkpoints/weights/0004/archive",
        headers={"User-Agent": "AsyncTinker/Python 0.13.1"},
        follow_redirects=False,
    )
    assert resp.status_code == 302
    direct = client.get(resp.headers["location"])
    assert direct.status_code == 200
    assert direct.content[:2] == b"\x1f\x8b"
