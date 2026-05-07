from __future__ import annotations

import importlib.machinery
import sys
import types
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _install_ray_stub(monkeypatch) -> None:
    ray = types.ModuleType("ray")
    ray.__spec__ = importlib.machinery.ModuleSpec("ray", loader=None)
    ray.is_initialized = lambda: False  # type: ignore[attr-defined]
    ray.actor = SimpleNamespace(ActorHandle=object)
    ray.util = SimpleNamespace(
        get_placement_group=lambda *_args, **_kwargs: None,
        remove_placement_group=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "ray", ray)


class _FakePool:
    def __init__(self, *, actors: list[dict], entries: list[object]) -> None:
        self._actors = list(actors)
        self._entries = list(entries)
        self.unregister_calls: list[str] = []
        self.list_actor_refresh_metadata_calls: list[bool] = []
        self.list_actor_filter_calls: list[dict[str, object]] = []

    def list_actors(self, *, refresh_metadata: bool = False, actor_type=None, model_name: str | None = None) -> list[dict]:
        self.list_actor_refresh_metadata_calls.append(bool(refresh_metadata))
        self.list_actor_filter_calls.append({"actor_type": actor_type, "model_name": model_name})
        return list(self._actors)

    def total_gpus_used(self) -> int:
        return 0

    def iter_entries(self) -> list[object]:
        return list(self._entries)

    def get(self, actor_name: str):
        for entry in self._entries:
            if getattr(entry, "actor_name", None) == actor_name:
                return entry
        return None

    def unregister(self, actor_name: str) -> None:
        self.unregister_calls.append(actor_name)


def _build_client(monkeypatch, pool: _FakePool) -> TestClient:
    from tinker_server.routes import service as service_routes
    import tinker_server.backend.resource_pool as resource_pool

    monkeypatch.setattr(service_routes, "_require_admin", lambda _request: None)
    monkeypatch.setattr(resource_pool, "get_resource_pool", lambda: pool)

    app = FastAPI()
    app.include_router(service_routes.router, prefix="/api/v1")
    return TestClient(app)


def _raise_missing_ray_address(*_args, **_kwargs):
    from tinker_server.ray_utils import MissingRayAddressError

    raise MissingRayAddressError("RAY_ADDRESS must be set before initializing Ray")


def test_list_actors_returns_503_when_ray_init_contract_fails(monkeypatch) -> None:
    import tinker_server.ray_utils as ray_utils

    _install_ray_stub(monkeypatch)
    monkeypatch.setattr(ray_utils, "init_ray", _raise_missing_ray_address)

    client = _build_client(
        monkeypatch,
        _FakePool(
            actors=[{"actor_name": "dense-a", "actor_type": "dense", "base_model": "Qwen/Qwen3-0.6B"}],
            entries=[],
        ),
    )

    resp = client.get("/api/v1/actors")

    assert resp.status_code == 503, resp.text
    assert "RAY_ADDRESS must be set" in resp.text


def test_list_actors_refreshes_metadata_by_default(monkeypatch) -> None:
    import tinker_server.ray_utils as ray_utils

    _install_ray_stub(monkeypatch)
    monkeypatch.setattr(ray_utils, "init_ray", lambda *_args, **_kwargs: None)
    pool = _FakePool(
        actors=[{"actor_name": "vllm-a", "actor_type": "vllm", "base_model": "Qwen/Qwen3-4B-Instruct-2507"}],
        entries=[],
    )
    client = _build_client(monkeypatch, pool)

    resp = client.get("/api/v1/actors")

    assert resp.status_code == 200, resp.text
    assert pool.list_actor_refresh_metadata_calls == [True]


def test_list_actors_can_skip_metadata_refresh(monkeypatch) -> None:
    import tinker_server.ray_utils as ray_utils

    _install_ray_stub(monkeypatch)
    monkeypatch.setattr(ray_utils, "init_ray", lambda *_args, **_kwargs: None)
    pool = _FakePool(
        actors=[{"actor_name": "vllm-a", "actor_type": "vllm", "base_model": "Qwen/Qwen3-4B-Instruct-2507"}],
        entries=[],
    )
    client = _build_client(monkeypatch, pool)

    resp = client.get("/api/v1/actors?refresh_metadata=false")

    assert resp.status_code == 200, resp.text
    assert pool.list_actor_refresh_metadata_calls == [False]


def test_list_actors_passes_filters_to_resource_pool_before_refresh(monkeypatch) -> None:
    import tinker_server.ray_utils as ray_utils
    from tinker_server.backend.resource_pool import ActorType

    _install_ray_stub(monkeypatch)
    monkeypatch.setattr(ray_utils, "init_ray", lambda *_args, **_kwargs: None)
    pool = _FakePool(
        actors=[{"actor_name": "vllm-a", "actor_type": "vllm", "base_model": "Qwen/Qwen3-4B-Instruct-2507"}],
        entries=[],
    )
    client = _build_client(monkeypatch, pool)

    resp = client.get("/api/v1/actors?type=vllm&model_name=Qwen/Qwen3-4B-Instruct-2507")

    assert resp.status_code == 200, resp.text
    assert pool.list_actor_refresh_metadata_calls == [True]
    assert pool.list_actor_filter_calls == [
        {
            "actor_type": ActorType.VLLM,
            "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        }
    ]


def test_list_actors_runs_resource_pool_inventory_in_threadpool(monkeypatch) -> None:
    from tinker_server.routes import service as service_routes
    import tinker_server.ray_utils as ray_utils

    _install_ray_stub(monkeypatch)
    monkeypatch.setattr(ray_utils, "init_ray", lambda *_args, **_kwargs: None)
    pool = _FakePool(
        actors=[{"actor_name": "vllm-a", "actor_type": "vllm", "base_model": "Qwen/Qwen3-4B-Instruct-2507"}],
        entries=[],
    )
    threadpool_calls = []

    async def _run_in_threadpool(fn, *args, **kwargs):
        threadpool_calls.append((fn, args, kwargs))
        return fn(*args, **kwargs)

    monkeypatch.setattr(service_routes, "run_in_threadpool", _run_in_threadpool)
    client = _build_client(monkeypatch, pool)

    resp = client.get("/api/v1/actors")

    assert resp.status_code == 200, resp.text
    assert len(threadpool_calls) == 1
    assert pool.list_actor_refresh_metadata_calls == [True]


def test_kill_dense_actors_returns_503_without_unregistering_when_ray_init_fails(monkeypatch) -> None:
    import tinker_server.ray_utils as ray_utils
    from tinker_server.backend.resource_pool import ActorType

    _install_ray_stub(monkeypatch)
    monkeypatch.setattr(ray_utils, "init_ray", _raise_missing_ray_address)

    pool = _FakePool(
        actors=[],
        entries=[
            SimpleNamespace(
                actor_type=ActorType.DENSE,
                actor_name="dense-a",
                namespace="ns",
                base_model="Qwen/Qwen3-0.6B",
            )
        ],
    )
    client = _build_client(monkeypatch, pool)

    resp = client.post("/api/v1/actors/kill", json={"actor_type": "dense"})

    assert resp.status_code == 503, resp.text
    assert "RAY_ADDRESS must be set" in resp.text
    assert pool.unregister_calls == []


def test_kill_exact_dense_actor_returns_503_without_unregistering_when_kill_fails(monkeypatch) -> None:
    from tinker_server.routes import service as service_routes
    from tinker_server.backend.resource_pool import ActorType

    _install_ray_stub(monkeypatch)
    remove_pg_calls: list[str] = []

    async def _lookup(*_args, **_kwargs):
        return object()

    async def _raise_kill(*_args, **_kwargs):
        raise RuntimeError("kill failed")

    pool = _FakePool(
        actors=[],
        entries=[
            SimpleNamespace(
                actor_type=ActorType.DENSE,
                actor_name="dense-a",
                namespace="ns",
                base_model="Qwen/Qwen3-0.6B",
                actor_handle=None,
            )
        ],
    )
    monkeypatch.setattr(service_routes, "async_lookup_actor_handle", _lookup)
    monkeypatch.setattr(service_routes, "async_kill_named_actor", _raise_kill)
    monkeypatch.setattr(service_routes, "_remove_actor_pg", lambda actor_name: remove_pg_calls.append(actor_name))
    client = _build_client(monkeypatch, pool)

    resp = client.post("/api/v1/actors/kill", json={"actor_type": "dense", "actor_name": "dense-a"})

    assert resp.status_code == 503, resp.text
    assert "kill failed" in resp.text
    assert pool.unregister_calls == []
    assert remove_pg_calls == []


def test_kill_dense_actors_returns_503_without_unregistering_when_kill_fails(monkeypatch) -> None:
    from tinker_server.routes import service as service_routes
    from tinker_server.backend.resource_pool import ActorType
    import tinker_server.ray_utils as ray_utils

    _install_ray_stub(monkeypatch)
    remove_pg_calls: list[str] = []

    async def _raise_kill(*_args, **_kwargs):
        raise RuntimeError("kill failed")

    pool = _FakePool(
        actors=[],
        entries=[
            SimpleNamespace(
                actor_type=ActorType.DENSE,
                actor_name="dense-a",
                namespace="ns",
                base_model="Qwen/Qwen3-0.6B",
                actor_handle=None,
            )
        ],
    )
    monkeypatch.setattr(service_routes, "async_kill_named_actor", _raise_kill)
    monkeypatch.setattr(service_routes, "_remove_actor_pg", lambda actor_name: remove_pg_calls.append(actor_name))
    monkeypatch.setattr(ray_utils, "init_ray", lambda *_args, **_kwargs: None)
    client = _build_client(monkeypatch, pool)

    resp = client.post("/api/v1/actors/kill", json={"actor_type": "dense"})

    assert resp.status_code == 503, resp.text
    assert "kill failed" in resp.text
    assert pool.unregister_calls == []
    assert remove_pg_calls == []


def test_kill_dense_actors_returns_503_when_pg_removal_fails(monkeypatch) -> None:
    from tinker_server.routes import service as service_routes
    from tinker_server.backend.resource_pool import ActorType
    import tinker_server.ray_utils as ray_utils

    _install_ray_stub(monkeypatch)

    async def _kill_ok(*_args, **_kwargs):
        return None

    def _raise_remove_pg(*_args, **_kwargs):
        raise RuntimeError("pg failed")

    pool = _FakePool(
        actors=[],
        entries=[
            SimpleNamespace(
                actor_type=ActorType.DENSE,
                actor_name="dense-a",
                namespace="ns",
                base_model="Qwen/Qwen3-0.6B",
                actor_handle=None,
            )
        ],
    )
    monkeypatch.setattr(service_routes, "async_kill_named_actor", _kill_ok)
    monkeypatch.setattr(ray_utils, "init_ray", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(service_routes, "_remove_actor_pg", _raise_remove_pg)
    client = _build_client(monkeypatch, pool)

    resp = client.post("/api/v1/actors/kill", json={"actor_type": "dense"})

    assert resp.status_code == 503, resp.text
    assert "pg failed" in resp.text
    assert pool.unregister_calls == []


def test_infer_base_model_from_checkpoint_passes_admin_scope(monkeypatch, tmp_path) -> None:
    from tinker_server.routes import service as service_routes
    import tinker_server.checkpoints as checkpoints

    checkpoint_dir = tmp_path / "admin-ckpt"
    checkpoint_dir.mkdir()
    resolve_calls: list[dict[str, object]] = []

    monkeypatch.setattr(checkpoints, "get_checkpoints_dir", lambda: str(tmp_path / "root"))
    monkeypatch.setattr(
        checkpoints,
        "resolve_checkpoint_uri",
        lambda model_path, checkpoints_dir, *, user_id=None, is_admin=False: (
            resolve_calls.append(
                {
                    "model_path": model_path,
                    "checkpoints_dir": checkpoints_dir,
                    "user_id": user_id,
                    "is_admin": is_admin,
                }
            ),
            str(checkpoint_dir),
        )[1],
    )
    monkeypatch.setattr(
        checkpoints,
        "read_checkpoint_metadata",
        lambda _path: {"model_name": "openpi/pi0-fast-libero-low-mem-finetune"},
    )

    base_model = service_routes._infer_base_model_from_checkpoint(
        "mint://run-1/sampler_weights/export-1",
        user_id="admin",
        is_admin=True,
    )

    assert base_model == "openpi/pi0-fast-libero-low-mem-finetune"
    assert resolve_calls == [
        {
            "model_path": "mint://run-1/sampler_weights/export-1",
            "checkpoints_dir": str(tmp_path / "root"),
            "user_id": "admin",
            "is_admin": True,
        }
    ]


def test_kill_dense_actors_returns_503_when_pg_lookup_mismatches_namespace(monkeypatch) -> None:
    from tinker_server.routes import service as service_routes
    import tinker_server.backend.ray_placement_groups as ray_placement_groups
    from tinker_server.backend.resource_pool import ActorType
    import tinker_server.ray_utils as ray_utils

    _install_ray_stub(monkeypatch)

    async def _kill_ok(*_args, **_kwargs):
        return None

    def _raise_lookup_mismatch(*_args, **_kwargs):
        raise ValueError("placement group 'dense-a_pg' exists in namespace='ns-b', not target_namespace='ns-a'")

    pool = _FakePool(
        actors=[],
        entries=[
            SimpleNamespace(
                actor_type=ActorType.DENSE,
                actor_name="dense-a",
                namespace="ns-a",
                base_model="Qwen/Qwen3-0.6B",
                actor_handle=None,
            )
        ],
    )
    monkeypatch.setattr(service_routes, "async_kill_named_actor", _kill_ok)
    monkeypatch.setattr(ray_utils, "init_ray", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ray_placement_groups, "get_named_placement_group", _raise_lookup_mismatch)
    client = _build_client(monkeypatch, pool)

    resp = client.post("/api/v1/actors/kill", json={"actor_type": "dense"})

    assert resp.status_code == 503, resp.text
    assert "target_namespace='ns-a'" in resp.text
    assert pool.unregister_calls == []
