import asyncio

import anyio
import pytest

from mint_server.backend.training.megatron.megatron_distributed import (
    MegatronRankWorker,
    MegatronSessionStateManager,
    MegatronWorkerGroup,
)
from mint_server.backend.training.bumblebee.bumblebee_distributed import BumblebeeRankWorker, BumblebeeWorkerGroup
from mint_server.backend.core.runtime_observability import RuntimeObservability
from mint_server.backend.training.verl.verl_training import TrainingWorker, VerlTrainingEngine
from mint_server.models.types import ComputeLogprobsRequest, ModelInput, SampleRequest, SamplingParams
from mint_server.routes import sampling as sampling_route


class _StubTaskFutureService:
    def __init__(self):
        self.failed: dict[str, str] = {}

    def fail(self, request_id: str, error: str) -> None:
        self.failed[request_id] = str(error)

    async def async_fail(self, request_id: str, error: str) -> None:
        self.fail(request_id, error)

    def resolve(self, request_id: str, payload: dict) -> None:
        raise AssertionError(f"did not expect resolve for canceled request {request_id}: {payload!r}")


class _CanceledSamplingEngine:
    actor_name = "actor-test"

    async def generate(self, **_kwargs):
        raise asyncio.CancelledError()

    async def compute_logprobs(self, **_kwargs):
        raise asyncio.CancelledError()


class _StubSessionManager:
    def __init__(self):
        self.engine = _CanceledSamplingEngine()
        self.inflight: list[tuple[str, int]] = []

    def mark_session_inflight(self, session_id: str, delta: int) -> None:
        self.inflight.append((session_id, int(delta)))

    def is_multi_lora_session(self, _session_id: str) -> bool:
        return False

    def get_engine(self, _session_id: str):
        return self.engine

    def get_session_base_model(self, _session_id: str) -> str:
        return "Qwen/Test"


def test_issue_439_reverse_kl_entrypoints_exist() -> None:
    assert callable(getattr(TrainingWorker, "forward_backward_reverse_kl", None))
    assert callable(getattr(VerlTrainingEngine, "forward_backward_reverse_kl", None))
    assert callable(getattr(MegatronRankWorker, "forward_backward_reverse_kl", None))
    assert callable(getattr(MegatronRankWorker, "forward_reference_full_log_probs", None))
    assert callable(getattr(MegatronSessionStateManager, "prime_session", None))
    assert callable(getattr(MegatronWorkerGroup, "forward_backward_reverse_kl", None))
    assert callable(getattr(MegatronWorkerGroup, "forward_reference_full_log_probs", None))
    assert callable(getattr(MegatronWorkerGroup, "prime_session_checkpoint", None))
    assert callable(getattr(MegatronWorkerGroup, "delete_session", None))
    assert callable(getattr(BumblebeeRankWorker, "forward_backward_reverse_kl", None))
    assert callable(getattr(BumblebeeRankWorker, "forward_reference_full_log_probs", None))
    assert callable(getattr(BumblebeeWorkerGroup, "forward_backward_reverse_kl", None))
    assert callable(getattr(BumblebeeWorkerGroup, "forward_reference_full_log_probs", None))
    assert callable(getattr(BumblebeeWorkerGroup, "delete_session", None))


def test_issue_439_asample_cancellation_decrements_active_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    task_futures = _StubTaskFutureService()
    session_manager = _StubSessionManager()
    obs = RuntimeObservability()

    monkeypatch.setattr(sampling_route, "task_futures", task_futures)
    monkeypatch.setattr(sampling_route, "session_manager", session_manager)
    monkeypatch.setattr(
        sampling_route,
        "_abort_engine_request",
        lambda *_args, **_kwargs: anyio.sleep(0),
    )
    monkeypatch.setattr(
        "mint_server.backend.core.runtime_observability.runtime_observability",
        obs,
    )

    request = SampleRequest(
        sampling_session_id="sess-1",
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=8),
    )

    with pytest.raises(asyncio.CancelledError):
        anyio.run(sampling_route._do_sample, "req-sample-cancel", request, None, None)

    snap = obs.snapshot()
    assert task_futures.failed == {"req-sample-cancel": "sampling task cancelled"}
    assert session_manager.inflight == [("sess-1", 1), ("sess-1", -1)]
    assert snap["vllm_active_requests"] == [
        {
            "actor_name": "actor-test",
            "base_model": "Qwen/Test",
            "op": "asample",
            "active_requests": 0,
        }
    ]
    assert len(snap["vllm_workload"]) == 1
    row = snap["vllm_workload"][0]
    assert row["base_model"] == "Qwen/Test"
    assert row["op"] == "asample"
    assert row["status"] == "canceled"
    assert row["requests_total"] == 1
    assert row["prompt_tokens_total"] == 3
    assert row["generated_tokens_total"] == 0
    assert row["duration_s_total"] >= 0.0
    assert row["duration_s_max"] >= 0.0


def test_issue_439_compute_logprobs_cancellation_decrements_active_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    task_futures = _StubTaskFutureService()
    session_manager = _StubSessionManager()
    obs = RuntimeObservability()

    monkeypatch.setattr(sampling_route, "task_futures", task_futures)
    monkeypatch.setattr(sampling_route, "session_manager", session_manager)
    monkeypatch.setattr(
        "mint_server.backend.core.runtime_observability.runtime_observability",
        obs,
    )

    request = ComputeLogprobsRequest(
        sampling_session_id="sess-1",
        seq_id=0,
        sequence=ModelInput.from_ints([11, 12, 13, 14]),
    )

    with pytest.raises(asyncio.CancelledError):
        anyio.run(sampling_route._do_compute_logprobs, "req-logprobs-cancel", request, None, None)

    snap = obs.snapshot()
    assert task_futures.failed == {"req-logprobs-cancel": "compute_logprobs task cancelled"}
    assert session_manager.inflight == [("sess-1", 1), ("sess-1", -1)]
    assert snap["vllm_active_requests"] == [
        {
            "actor_name": "actor-test",
            "base_model": "Qwen/Test",
            "op": "compute_logprobs",
            "active_requests": 0,
        }
    ]
    assert len(snap["vllm_workload"]) == 1
    row = snap["vllm_workload"][0]
    assert row["base_model"] == "Qwen/Test"
    assert row["op"] == "compute_logprobs"
    assert row["status"] == "canceled"
    assert row["requests_total"] == 1
    assert row["prompt_tokens_total"] == 4
    assert row["generated_tokens_total"] == 0
    assert row["duration_s_total"] >= 0.0
    assert row["duration_s_max"] >= 0.0
