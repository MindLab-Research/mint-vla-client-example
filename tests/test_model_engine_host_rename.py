from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_model_engine_host_is_primary_runtime_entrypoint() -> None:
    from mint_server.backend import model_engine_host
    from mint_server.backend import model_runtime_actor

    assert hasattr(model_engine_host, "ModelEngineHost")
    assert hasattr(model_engine_host, "get_or_create_model_engine_host")
    assert model_runtime_actor.ModelRuntimeActor is model_engine_host.ModelEngineHost
    assert (
        model_runtime_actor.get_or_create_model_runtime_actor
        is model_engine_host.get_or_create_model_engine_host
    )


def test_model_actor_launcher_imports_engine_host_not_runtime_actor() -> None:
    launcher_source = (
        REPO_ROOT / "mint_server" / "backend" / "model_actor_launchers.py"
    ).read_text()

    assert "from .model_engine_host import get_or_create_model_engine_host" in launcher_source
    assert "from .model_runtime_actor import get_or_create_model_runtime_actor" not in launcher_source
