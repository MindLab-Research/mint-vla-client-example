import pytest


@pytest.fixture(autouse=True)
def _component_scheduler_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MINT_MODEL_WORK_SCHEDULER_ASSIGNMENT_INTERVAL_S", "0")
    monkeypatch.setenv("MINT_MODEL_WORK_SCHEDULER_OWNER_HEARTBEAT_INTERVAL_S", "0")
    monkeypatch.setattr(
        "mint_server.logging_context.init_actor_observability",
        lambda: None,
    )
    try:
        import ray

        monkeypatch.setattr(ray, "init", lambda *args, **kwargs: None)
    except Exception:
        pass


@pytest.fixture
def anyio_backend():
    return "asyncio"
