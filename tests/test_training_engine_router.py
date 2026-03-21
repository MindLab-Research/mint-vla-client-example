import asyncio
from types import SimpleNamespace

from tinker_server.backend.training_engine_router import TrainingEngineRouter


class _RecordingEngine:
    def __init__(self, label: str) -> None:
        self.label = label
        self.calls: list[tuple[str, str]] = []

    async def initialize(self) -> None:
        self.calls.append(("initialize", self.label))

    async def create_training_session(self, session):
        self.calls.append(("create_training_session", session.base_model))
        return self.label

    async def forward_backward(self, session, request):
        self.calls.append(("forward_backward", session.base_model))
        return {"engine": self.label, "request": request}

    def _resolve_hf_model_path(self, model_name: str):
        self.calls.append(("resolve", model_name))
        return f"/models/{model_name}"


def test_training_engine_router_delegates_text_models_to_verl_engine() -> None:
    text_engine = _RecordingEngine("text")
    openpi_fast_engine = _RecordingEngine("openpi-fast")
    router = TrainingEngineRouter(text_engine=text_engine, openpi_fast_engine=openpi_fast_engine)
    session = SimpleNamespace(base_model="Qwen/Qwen3-0.6B")

    result = asyncio.run(router.create_training_session(session))

    assert result == "text"
    assert text_engine.calls == [("create_training_session", "Qwen/Qwen3-0.6B")]
    assert openpi_fast_engine.calls == []


def test_training_engine_router_delegates_openpi_fast_models_by_training_backend() -> None:
    text_engine = _RecordingEngine("text")
    openpi_fast_engine = _RecordingEngine("openpi-fast")
    router = TrainingEngineRouter(text_engine=text_engine, openpi_fast_engine=openpi_fast_engine)
    session = SimpleNamespace(base_model="openpi/pi0-fast-libero-low-mem-finetune")

    result = asyncio.run(router.forward_backward(session, request={"batch": 1}))

    assert result == {"engine": "openpi-fast", "request": {"batch": 1}}
    assert openpi_fast_engine.calls == [
        ("forward_backward", "openpi/pi0-fast-libero-low-mem-finetune")
    ]
    assert text_engine.calls == []


def test_training_engine_router_delegates_openpi_pi05_models_by_training_backend(monkeypatch) -> None:
    text_engine = _RecordingEngine("text")
    openpi_fast_engine = _RecordingEngine("openpi-fast")
    openpi_pi05_engine = _RecordingEngine("openpi-pi05")
    router = TrainingEngineRouter(
        text_engine=text_engine,
        openpi_fast_engine=openpi_fast_engine,
        openpi_pi05_engine=openpi_pi05_engine,
    )
    session = SimpleNamespace(base_model="openpi/pi05-libero-low-mem-finetune")

    monkeypatch.setattr(
        "tinker_server.backend.training_engine_router.get_model_config",
        lambda base_model: SimpleNamespace(training_backend="openpi_pi05"),
    )

    result = asyncio.run(router.forward_backward(session, request={"batch": 2}))

    assert result == {"engine": "openpi-pi05", "request": {"batch": 2}}
    assert openpi_pi05_engine.calls == [("forward_backward", "openpi/pi05-libero-low-mem-finetune")]
    assert openpi_fast_engine.calls == []
    assert text_engine.calls == []


def test_training_engine_router_forwards_hf_path_resolution_to_text_engine() -> None:
    text_engine = _RecordingEngine("text")
    router = TrainingEngineRouter(text_engine=text_engine, openpi_fast_engine=_RecordingEngine("openpi-fast"))

    resolved = router._resolve_hf_model_path("Qwen/Qwen3-0.6B")

    assert resolved == "/models/Qwen/Qwen3-0.6B"
    assert text_engine.calls == [("resolve", "Qwen/Qwen3-0.6B")]
