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
    ray.exceptions = SimpleNamespace(
        GetTimeoutError=RuntimeError,
        RayActorError=RuntimeError,
    )
    ray.util = SimpleNamespace(
        get_placement_group=lambda *_args, **_kwargs: None,
        remove_placement_group=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.exceptions", ray.exceptions)


class _FakePool:
    def __init__(self, *, actors: list[dict], entries: list[object]) -> None:
        self._actors = list(actors)
        self._entries = list(entries)
        self.unregister_calls: list[str] = []
        self.list_actor_refresh_metadata_calls: list[bool] = []
        self.list_actor_filter_calls: list[dict[str, object]] = []
        self.total_gpus_used_calls = 0

    def list_actors(self, *, refresh_metadata: bool = False, actor_type=None, model_name: str | None = None) -> list[dict]:
        raise AssertionError("/actors route must use ModelActorRegistry.async_list_actors")

    async def async_list_actors(
        self,
        *,
        refresh_metadata: bool = False,
        actor_type=None,
        model_name: str | None = None,
    ) -> list[dict]:
        self.list_actor_refresh_metadata_calls.append(bool(refresh_metadata))
        self.list_actor_filter_calls.append({"actor_type": actor_type, "model_name": model_name})
        return list(self._actors)

    def total_gpus_used(self) -> int:
        raise AssertionError("/actors route must use ModelActorRegistry.async_total_gpus_used")

    async def async_total_gpus_used(self) -> int:
        self.total_gpus_used_calls += 1
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


def _build_client(monkeypatch, pool: _FakePool, *, patch_placement_groups: bool = True) -> TestClient:
    from tinker_server.routes import service as service_routes
    import tinker_server.backend.model_actor_registry as model_actor_registry

    monkeypatch.setattr(service_routes, "_require_admin", lambda _request: None)
    monkeypatch.setattr(model_actor_registry, "get_model_actor_registry", lambda: pool)
    if patch_placement_groups:
        async def _empty_placement_group_table(*_args, **_kwargs):
            return {}

        monkeypatch.setattr(service_routes, "async_placement_group_table", _empty_placement_group_table)

    app = FastAPI()
    app.include_router(service_routes.router, prefix="/api/v1")
    return TestClient(app)


def test_list_actors_uses_startup_ray_driver_without_request_path_init(monkeypatch) -> None:
    import tinker_server.ray_utils as ray_utils

    _install_ray_stub(monkeypatch)
    init_ray_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(ray_utils, "init_ray", lambda *args, **kwargs: init_ray_calls.append((args, kwargs)))

    client = _build_client(
        monkeypatch,
        _FakePool(
            actors=[{"actor_name": "dense-a", "actor_type": "dense", "base_model": "Qwen/Qwen3-0.6B"}],
            entries=[],
        ),
    )

    resp = client.get("/api/v1/actors")

    assert resp.status_code == 200, resp.text
    assert init_ray_calls == []


def test_list_actors_refreshes_metadata_by_default(monkeypatch) -> None:
    _install_ray_stub(monkeypatch)
    pool = _FakePool(
        actors=[{"actor_name": "vllm-a", "actor_type": "vllm", "base_model": "Qwen/Qwen3-4B-Instruct-2507"}],
        entries=[],
    )
    client = _build_client(monkeypatch, pool)

    resp = client.get("/api/v1/actors")

    assert resp.status_code == 200, resp.text
    assert pool.list_actor_refresh_metadata_calls == [True]


def test_list_actors_can_skip_metadata_refresh(monkeypatch) -> None:
    _install_ray_stub(monkeypatch)
    pool = _FakePool(
        actors=[{"actor_name": "vllm-a", "actor_type": "vllm", "base_model": "Qwen/Qwen3-4B-Instruct-2507"}],
        entries=[],
    )
    client = _build_client(monkeypatch, pool)

    resp = client.get("/api/v1/actors?refresh_metadata=false")

    assert resp.status_code == 200, resp.text
    assert pool.list_actor_refresh_metadata_calls == [False]


def test_list_actors_passes_filters_to_model_actor_registry_before_refresh(monkeypatch) -> None:
    from tinker_server.backend.model_actor_registry import ActorType

    _install_ray_stub(monkeypatch)
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


def test_list_actors_uses_async_model_actor_registry_inventory(monkeypatch) -> None:
    _install_ray_stub(monkeypatch)
    pool = _FakePool(
        actors=[{"actor_name": "vllm-a", "actor_type": "vllm", "base_model": "Qwen/Qwen3-4B-Instruct-2507"}],
        entries=[],
    )
    client = _build_client(monkeypatch, pool)

    resp = client.get("/api/v1/actors")

    assert resp.status_code == 200, resp.text
    assert pool.list_actor_refresh_metadata_calls == [True]
    assert pool.total_gpus_used_calls == 1


def test_list_actors_returns_503_when_model_actor_registry_inventory_fails(monkeypatch) -> None:
    class _FailingListPool(_FakePool):
        async def async_list_actors(self, **_kwargs) -> list[dict]:
            raise RuntimeError("ray disconnected")

    _install_ray_stub(monkeypatch)
    client = _build_client(monkeypatch, _FailingListPool(actors=[], entries=[]))

    resp = client.get("/api/v1/actors")

    assert resp.status_code == 503, resp.text
    assert "Ray unavailable for actor inventory" in resp.text
    assert "ray disconnected" in resp.text


def test_list_actors_returns_503_when_model_actor_registry_gpu_total_fails(monkeypatch) -> None:
    class _FailingGpuTotalPool(_FakePool):
        async def async_total_gpus_used(self) -> int:
            raise RuntimeError("model actor registry unavailable")

    _install_ray_stub(monkeypatch)
    client = _build_client(
        monkeypatch,
        _FailingGpuTotalPool(
            actors=[{"actor_name": "vllm-a", "actor_type": "vllm", "base_model": "Qwen/Qwen3-4B-Instruct-2507"}],
            entries=[],
        ),
    )

    resp = client.get("/api/v1/actors")

    assert resp.status_code == 503, resp.text
    assert "Ray unavailable for actor inventory" in resp.text
    assert "model actor registry unavailable" in resp.text


def test_kill_dense_actors_returns_503_without_unregistering_when_ray_driver_is_unavailable(monkeypatch) -> None:
    from tinker_server.backend.model_actor_registry import ActorType

    _install_ray_stub(monkeypatch)

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
    assert "Ray is not initialized" in resp.text
    assert pool.unregister_calls == []


def test_kill_exact_dense_actor_returns_503_without_unregistering_when_kill_fails(monkeypatch) -> None:
    from tinker_server.routes import service as service_routes
    from tinker_server.backend.model_actor_registry import ActorType

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
    from tinker_server.backend.model_actor_registry import ActorType

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
    client = _build_client(monkeypatch, pool)

    resp = client.post("/api/v1/actors/kill", json={"actor_type": "dense"})

    assert resp.status_code == 503, resp.text
    assert "kill failed" in resp.text
    assert pool.unregister_calls == []
    assert remove_pg_calls == []


def test_kill_dense_actors_returns_503_when_pg_removal_fails(monkeypatch) -> None:
    from tinker_server.routes import service as service_routes
    from tinker_server.backend.model_actor_registry import ActorType

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
    from tinker_server.backend.model_actor_registry import ActorType

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
    monkeypatch.setattr(ray_placement_groups, "get_named_placement_group", _raise_lookup_mismatch)
    client = _build_client(monkeypatch, pool)

    resp = client.post("/api/v1/actors/kill", json={"actor_type": "dense"})

    assert resp.status_code == 503, resp.text
    assert "target_namespace='ns-a'" in resp.text
    assert pool.unregister_calls == []
