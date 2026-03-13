import anyio
from types import SimpleNamespace

from tinker_server.models.types import ComputeLogprobsRequest, ModelInput, SampleRequest, SamplingParams
from tinker_server.routes import sampling as sampling_route


class _StubFutureStore:
    def __init__(self, order: list[str] | None = None):
        self.resolved: dict[str, dict] = {}
        self.failed: dict[str, str] = {}
        self.order = order if order is not None else []

    def resolve(self, request_id: str, payload: dict) -> None:
        self.order.append("resolve")
        self.resolved[request_id] = dict(payload)

    def fail(self, request_id: str, error: str) -> None:
        self.failed[request_id] = str(error)


class _StubUsageStore:
    def __init__(self, order: list[str] | None = None):
        self.events = []
        self.order = order if order is not None else []

    async def write_events(self, events) -> None:
        self.order.append("write_events")
        self.events.extend(list(events))


class _StubSamplingEngine:
    actor_name = "actor-test"

    async def generate(self, **_kwargs):
        return SimpleNamespace(
            token_ids=[101, 102],
            logprobs=[-0.1, -0.2],
            routed_experts=None,
            stop_reason="length",
        )

    async def compute_logprobs(self, **_kwargs):
        return [-0.1, -0.2, -0.3, -0.4]


class _StubSessionManager:
    def __init__(self):
        self.engine = _StubSamplingEngine()
        self.inflight: list[tuple[str, int]] = []

    def mark_session_inflight(self, session_id: str, delta: int) -> None:
        self.inflight.append((session_id, int(delta)))

    def is_multi_lora_session(self, _session_id: str) -> bool:
        return False

    def get_engine(self, _session_id: str):
        return self.engine

    def get_session_base_model(self, _session_id: str) -> str:
        return "Qwen/Test"


def _gateway_auth() -> dict[str, str]:
    return {
        "user_id": "aaaaaaaaaaaaaaaaaaaaaaaa",
        "user_role": "user",
        "account_id": "aaaaaaaaaaaaaaaaaaaaaaaa",
        "apikey_id": "bbbbbbbbbbbbbbbbbbbbbbbb",
        "request_id": "req-billing-test",
    }


def test_asample_logs_prefill_and_sample_dimensions(monkeypatch):
    future_store = _StubFutureStore()
    usage_store = _StubUsageStore()

    monkeypatch.setattr(sampling_route, "session_manager", _StubSessionManager())
    monkeypatch.setattr(sampling_route, "future_store", future_store)

    async def _get_usage_store():
        return usage_store

    monkeypatch.setattr(sampling_route, "get_usage_store", _get_usage_store)

    request = SampleRequest(
        sampling_session_id="sess-1",
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=8),
    )

    anyio.run(sampling_route._do_sample, "req-sample", request, None, _gateway_auth())

    assert "req-sample" in future_store.resolved
    assert future_store.failed == {}
    assert [event.charge_item for event in usage_store.events] == ["sampling", "sampling"]
    assert [event.quantity for event in usage_store.events] == [3, 2]
    assert [event.label for event in usage_store.events] == [
        "model=Qwen/Test,route=sampling.asample,dimension=prefill",
        "model=Qwen/Test,route=sampling.asample,dimension=sample",
    ]


def test_asample_persists_usage_before_resolving_future(monkeypatch):
    order: list[str] = []
    future_store = _StubFutureStore(order=order)
    usage_store = _StubUsageStore(order=order)

    monkeypatch.setattr(sampling_route, "session_manager", _StubSessionManager())
    monkeypatch.setattr(sampling_route, "future_store", future_store)

    async def _get_usage_store():
        return usage_store

    monkeypatch.setattr(sampling_route, "get_usage_store", _get_usage_store)

    request = SampleRequest(
        sampling_session_id="sess-1",
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=8),
    )

    anyio.run(sampling_route._do_sample, "req-order", request, None, _gateway_auth())

    assert order == ["write_events", "resolve"]


def test_compute_logprobs_logs_prefill_dimension(monkeypatch):
    future_store = _StubFutureStore()
    usage_store = _StubUsageStore()

    monkeypatch.setattr(sampling_route, "session_manager", _StubSessionManager())
    monkeypatch.setattr(sampling_route, "future_store", future_store)

    async def _get_usage_store():
        return usage_store

    monkeypatch.setattr(sampling_route, "get_usage_store", _get_usage_store)

    request = ComputeLogprobsRequest(
        sampling_session_id="sess-1",
        seq_id=0,
        sequence=ModelInput.from_ints([11, 12, 13, 14]),
    )

    anyio.run(sampling_route._do_compute_logprobs, "req-logprobs", request, None, _gateway_auth())

    assert "req-logprobs" in future_store.resolved
    assert future_store.failed == {}
    assert len(usage_store.events) == 1
    assert usage_store.events[0].charge_item == "sampling"
    assert usage_store.events[0].quantity == 4
    assert usage_store.events[0].label == "model=Qwen/Test,route=sampling.compute_logprobs,dimension=prefill"
