from __future__ import annotations

import json
from types import SimpleNamespace

import anyio

from mint_server.backend.stores.task_state_store import TaskStateStore
from mint_server.models.types import ActRequest, ModelInput
from mint_server.routes import action_sampling
from mint_server.routes import training as training_routes


class _StubTaskFutureService:
    def __init__(self) -> None:
        self.resolved: dict[str, dict] = {}
        self.failed: dict[str, str] = {}
        self.billing_observations: dict[str, list[dict]] = {}

    async def async_resolve(
        self, request_id: str, payload: dict, *, billing_observations=None
    ) -> None:
        self.resolved[str(request_id)] = dict(payload)
        self.billing_observations[str(request_id)] = list(billing_observations or [])

    async def async_fail(self, request_id: str, error: str) -> None:
        self.failed[str(request_id)] = str(error)


def _billing_observation(
    *,
    request_id: str,
    charge_item: str,
    quantity: int,
    route: str,
    dimension: str,
) -> dict:
    return {
        "account_id": "acct-1",
        "apikey_id": "key-1",
        "request_id": request_id,
        "charge_item": charge_item,
        "quantity": int(quantity),
        "unit": "estimated_tokens",
        "route": route,
        "dimension": dimension,
        "model": "openpi/pi0-fast-libero-low-mem-finetune",
        "metadata": {"issue": 630},
        "observed_at": 100.0,
    }


def _gateway_auth() -> dict:
    return {
        "user_id": "user-1",
        "user_role": "user",
        "account_id": "acct-1",
        "apikey_id": "key-1",
        "request_id": "gateway-req-1",
        "cap_write": True,
    }


def _commit_success_with_observations(
    *, request_id: str, observations: list[dict]
) -> list[dict]:
    store = TaskStateStore.in_memory()
    try:
        store.ensure_task(
            request_id=request_id,
            op="mint.test",
            domain_key="internal:runtime",
            status="running",
            metadata={},
            now=100.0,
        )
        store.complete_task_success(
            request_id=request_id,
            result_path="/tmp/result.json",
            result_checksum="sha256:abc",
            result_size_bytes=17,
            metadata={"done_at": 101.0},
            billing_observations=observations,
            now=101.0,
        )
        claimed = store.claim_billing_outbox(
            claim_id="claim-1", limit=10, lease_ttl_s=30.0, now=102.0
        )
        return [dict(row["event"]) for row in claimed]
    finally:
        store.close()


def test_issue_630_smoke_sft_vla_success_writes_training_outbox() -> None:
    observations = [
        _billing_observation(
            request_id="mint-vla-sft-1",
            charge_item="training",
            quantity=261,
            route="mint.vla.train_step",
            dimension="train",
        )
    ]

    events = _commit_success_with_observations(
        request_id="mint-vla-sft-1", observations=observations
    )

    assert len(events) == 1
    assert events[0]["charge_item"] == "training"
    assert events[0]["quantity"] == 261
    assert events[0]["label"] == (
        "model=openpi/pi0-fast-libero-low-mem-finetune,"
        "route=mint.vla.train_step,dimension=train,unit=estimated_tokens"
    )


def test_issue_630_smoke_rl_rollout_act_and_train_step_are_separate_outbox_events() -> (
    None
):
    observations = [
        _billing_observation(
            request_id="mint-rl-act-1",
            charge_item="inference",
            quantity=323,
            route="mint.action.act",
            dimension="action",
        ),
        _billing_observation(
            request_id="mint-rl-act-2",
            charge_item="inference",
            quantity=323,
            route="mint.action.act",
            dimension="action",
        ),
        _billing_observation(
            request_id="mint-rl-train-1",
            charge_item="training",
            quantity=261,
            route="mint.vla.train_step",
            dimension="train",
        ),
    ]

    events = _commit_success_with_observations(
        request_id="mint-rl-rollout-1", observations=observations
    )

    assert [event["charge_item"] for event in events] == [
        "inference",
        "inference",
        "training",
    ]
    assert [event["request_id"] for event in events] == [
        "mint-rl-act-1",
        "mint-rl-act-2",
        "mint-rl-train-1",
    ]


def test_issue_630_action_executor_attaches_billing_observation_on_success(
    monkeypatch,
) -> None:
    task_futures = _StubTaskFutureService()

    class _ActionSessionManager:
        async def act(self, **_kwargs):
            return {"actions": [[0.0]], "meta": {"ok": True}}

    monkeypatch.setattr(action_sampling, "task_futures", task_futures)
    monkeypatch.setattr(
        action_sampling, "action_session_manager", _ActionSessionManager()
    )

    billing_input = {
        "charge_item": "inference",
        "quantity": 323,
        "unit": "estimated_tokens",
        "route": "mint.action.act",
        "dimension": "action",
        "model": "openpi/pi0-fast-libero-low-mem-finetune",
        "metadata": {"issue": 630},
    }
    request = ActRequest(
        action_session_id="action-session-1",
        observation=ModelInput.from_ints([1, 2, 3]),
        extra_inputs={},
    )

    anyio.run(
        action_sampling._do_act,
        "act-success-1",
        request,
        None,
        _gateway_auth(),
        billing_input,
    )

    assert task_futures.failed == {}
    assert task_futures.resolved["act-success-1"]["type"] == "act"
    observations = task_futures.billing_observations["act-success-1"]
    assert len(observations) == 1
    assert observations[0]["account_id"] == "acct-1"
    assert observations[0]["apikey_id"] == "key-1"
    assert observations[0]["request_id"] == "gateway-req-1"
    assert observations[0]["charge_item"] == "inference"
    assert observations[0]["quantity"] == 323
    assert observations[0]["route"] == "mint.action.act"
    assert observations[0]["dimension"] == "action"


def test_issue_630_vla_train_step_uses_mint_billing_observation_override(
    monkeypatch,
) -> None:
    task_futures = _StubTaskFutureService()
    inflight_calls: list[tuple[str, int]] = []

    class _TrainingManager:
        def __init__(self) -> None:
            self.inflight: list[tuple[str, int]] = []

        def get_session(self, model_id: str):
            assert model_id == "model-rl"
            return SimpleNamespace(
                model_id="model-rl",
                base_model="openpi/pi0-fast-libero-low-mem-finetune",
                backend="openpi_fast",
            )

        def mark_inflight(self, model_id: str, delta: int) -> None:
            self.inflight.append((model_id, int(delta)))

    class _TrainingEngine:
        async def train_step(self, session, request):
            assert session.model_id == "model-rl"
            assert request.model_id == "model-rl"
            return {"metrics": {"loss": 1.0}}

    async def _mark_training_inflight(model_id: str, delta: int) -> None:
        inflight_calls.append((model_id, delta))

    monkeypatch.setattr(training_routes, "task_futures", task_futures)
    monkeypatch.setattr(training_routes, "training_manager", _TrainingManager())
    monkeypatch.setattr(training_routes, "training_engine", _TrainingEngine())
    monkeypatch.setattr(training_routes, "_get_max_model_len", lambda _base_model: 2048)
    monkeypatch.setattr(
        training_routes, "_mark_training_inflight", _mark_training_inflight
    )

    async def _materialize(session):
        return session

    monkeypatch.setattr(
        training_routes, "_materialize_training_session_for_stateful_use", _materialize
    )

    billing_input = {
        "charge_item": "training",
        "quantity": 261,
        "unit": "estimated_tokens",
        "route": "mint.vla.train_step",
        "dimension": "train",
        "model": "openpi/pi0-fast-libero-low-mem-finetune",
        "metadata": {"issue": 630},
    }
    payload = {
        "model_id": "model-rl",
        "forward_backward_input": {
            "loss_fn": "cross_entropy",
            "data": [
                {
                    "model_input": {
                        "chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]
                    },
                    "loss_fn_inputs": {
                        "target_tokens": {
                            "data": [11, 12],
                            "shape": [2],
                            "dtype": "int64",
                        }
                    },
                }
            ],
        },
    }
    from mint_server.models.types import TrainStepRequest

    request = TrainStepRequest.model_validate_json(json.dumps(payload))

    anyio.run(
        training_routes._do_train_step,
        "mint-rl-train-override",
        request,
        "user-a",
        _gateway_auth(),
        None,
        billing_input,
    )

    assert task_futures.failed == {}
    assert inflight_calls == [("model-rl", -1)]
    observations = task_futures.billing_observations["mint-rl-train-override"]
    assert len(observations) == 1
    assert observations[0]["account_id"] == "acct-1"
    assert observations[0]["apikey_id"] == "key-1"
    assert observations[0]["request_id"] == "gateway-req-1"
    assert observations[0]["charge_item"] == "training"
    assert observations[0]["quantity"] == 261
    assert observations[0]["route"] == "mint.vla.train_step"
    assert observations[0]["dimension"] == "train"
