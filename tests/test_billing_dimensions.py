import anyio
from types import SimpleNamespace

from mint_server.models.types import (
    ComputeLogprobsRequest,
    ModelInput,
    SampleRequest,
    SamplingParams,
)
from mint_server.routes import sampling as sampling_route


class _StubTaskFutureService:
    def __init__(self, order: list[str] | None = None):
        self.resolved: dict[str, dict] = {}
        self.failed: dict[str, str] = {}
        self.billing_observations: dict[str, list[dict]] = {}
        self.outbox_observations: list[dict] = []
        self.order = order if order is not None else []

    def resolve(self, request_id: str, payload: dict) -> None:
        self.order.append("resolve")
        self.resolved[request_id] = dict(payload)

    async def async_resolve(
        self, request_id: str, payload: dict, *, billing_observations=None
    ) -> None:
        self.resolve(request_id, payload)
        self.billing_observations[request_id] = list(billing_observations or [])

    async def async_append_billing_outbox(
        self, observations, *, source: str = "unknown"
    ) -> dict:
        self.order.append(f"append:{source}")
        self.outbox_observations.extend(list(observations or []))
        return {"ok": True, "inserted": len(list(observations or []))}

    def fail(self, request_id: str, error: str) -> None:
        self.failed[request_id] = str(error)

    async def async_fail(self, request_id: str, error: str) -> None:
        self.fail(request_id, error)


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
    task_futures = _StubTaskFutureService()

    monkeypatch.setattr(sampling_route, "session_manager", _StubSessionManager())
    monkeypatch.setattr(sampling_route, "task_futures", task_futures)

    request = SampleRequest(
        sampling_session_id="sess-1",
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=8),
    )

    anyio.run(sampling_route._do_sample, "req-sample", request, None, _gateway_auth())

    assert "req-sample" in task_futures.resolved
    assert task_futures.failed == {}
    observations = task_futures.billing_observations["req-sample"]
    assert [event["charge_item"] for event in observations] == ["sampling", "sampling"]
    assert [event["quantity"] for event in observations] == [3, 2]
    assert [event["route"] for event in observations] == [
        "sampling.asample",
        "sampling.asample",
    ]
    assert [event["dimension"] for event in observations] == [
        "prefill",
        "sample",
    ]
    assert [event["model"] for event in observations] == ["Qwen/Test", "Qwen/Test"]


def test_asample_attaches_billing_to_future_resolve(monkeypatch):
    order: list[str] = []
    task_futures = _StubTaskFutureService(order=order)

    monkeypatch.setattr(sampling_route, "session_manager", _StubSessionManager())
    monkeypatch.setattr(sampling_route, "task_futures", task_futures)

    request = SampleRequest(
        sampling_session_id="sess-1",
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=8),
    )

    anyio.run(sampling_route._do_sample, "req-order", request, None, _gateway_auth())

    assert order == ["resolve"]
    assert len(task_futures.billing_observations["req-order"]) == 2


def test_asample_suppresses_billing_when_requested(monkeypatch):
    task_futures = _StubTaskFutureService()

    monkeypatch.setattr(sampling_route, "session_manager", _StubSessionManager())
    monkeypatch.setattr(sampling_route, "task_futures", task_futures)

    request = SampleRequest(
        sampling_session_id="sess-1",
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=8),
    )

    anyio.run(
        sampling_route._do_sample,
        "req-suppressed",
        request,
        None,
        _gateway_auth(),
        True,
    )

    assert "req-suppressed" in task_futures.resolved
    assert task_futures.failed == {}
    assert task_futures.billing_observations["req-suppressed"] == []


def test_compute_logprobs_logs_prefill_dimension(monkeypatch):
    task_futures = _StubTaskFutureService()

    monkeypatch.setattr(sampling_route, "session_manager", _StubSessionManager())
    monkeypatch.setattr(sampling_route, "task_futures", task_futures)

    request = ComputeLogprobsRequest(
        sampling_session_id="sess-1",
        seq_id=0,
        sequence=ModelInput.from_ints([11, 12, 13, 14]),
    )

    anyio.run(
        sampling_route._do_compute_logprobs,
        "req-logprobs",
        request,
        None,
        _gateway_auth(),
    )

    assert "req-logprobs" in task_futures.resolved
    assert task_futures.failed == {}
    observations = task_futures.billing_observations["req-logprobs"]
    assert len(observations) == 1
    assert observations[0]["charge_item"] == "sampling"
    assert observations[0]["quantity"] == 4
    assert observations[0]["route"] == "sampling.compute_logprobs"
    assert observations[0]["dimension"] == "prefill"
