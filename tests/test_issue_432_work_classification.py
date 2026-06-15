from __future__ import annotations

from types import SimpleNamespace

from mint_server.backend.core.work_classification import (
    WorkClassification,
    build_sampling_work_classification,
    build_training_work_classification,
)


def test_issue_432_training_work_classification_is_server_derived(monkeypatch) -> None:
    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")

    session = SimpleNamespace(backend="megatron", base_model="Qwen/Qwen3-30B-A3B-Instruct-2507")
    classification = build_training_work_classification(
        session=session,
        model_id="model-432",
    )

    assert classification.queue_kind == "scheduled"
    assert classification.scheduler_domain == "megatron:Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert classification.scheduler_session_key == "model-432"
    assert classification.scheduler_domain_key_source == "backend_base_model"
    assert classification.scheduler_capacity_owner == "single_worker"


def test_issue_432_sampling_work_classification_uses_replica_key_then_base_model(monkeypatch) -> None:
    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")

    with_replica = SimpleNamespace(
        is_multi_lora_session=lambda _session_id: True,
        get_session_replica_key=lambda _session_id: "Qwen/Qwen3-0.6B::replica::1",
        get_session_base_model=lambda _session_id: "Qwen/Qwen3-0.6B",
    )
    classification = build_sampling_work_classification(
        session_manager=with_replica,
        session_id="sess-432",
    )
    assert classification.queue_kind == "scheduled"
    assert classification.scheduler_domain == "vllm:Qwen/Qwen3-0.6B::replica::1"
    assert classification.scheduler_session_key == "sess-432"
    assert classification.scheduler_domain_key_source == "replica_key"
    assert classification.scheduler_capacity_owner == "vllm_replica_single_worker"

    without_replica = SimpleNamespace(
        is_multi_lora_session=lambda _session_id: True,
        get_session_replica_key=lambda _session_id: None,
        get_session_base_model=lambda _session_id: "Qwen/Qwen3-0.6B",
    )
    fallback = build_sampling_work_classification(
        session_manager=without_replica,
        session_id="sess-432",
    )
    assert fallback.queue_kind == "scheduled"
    assert fallback.scheduler_domain == "vllm:Qwen/Qwen3-0.6B"
    assert fallback.scheduler_domain_key_source == "base_model_fallback"
    assert fallback.scheduler_capacity_owner == "model_registry_inference_dp"


def test_issue_432_unschedulable_classification_stays_direct() -> None:
    classification = WorkClassification.build(
        scheduler_enabled=True,
        scheduler_domain="megatron:Qwen/Qwen3-30B-A3B-Instruct-2507",
        scheduler_session_key=None,
        scheduler_domain_key_source="backend_base_model",
    )

    assert classification.queue_kind == "direct"
    assert classification.scheduler_domain == "megatron:Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert classification.scheduler_session_key is None

    round_trip = WorkClassification.from_queue_extra(
        {
            "scheduler_enabled": True,
            "scheduler_domain": "megatron:Qwen/Qwen3-30B-A3B-Instruct-2507",
        },
        scheduler_enabled_default=False,
    )
    assert round_trip.queue_kind == "direct"


def test_issue_432_unknown_capacity_owner_stays_direct() -> None:
    classification = WorkClassification.build(
        scheduler_enabled=True,
        scheduler_domain="custom_backend:tenant-1",
        scheduler_session_key="sess-unknown",
        scheduler_domain_key_source="custom",
    )

    assert classification.queue_kind == "direct"
    assert classification.scheduler_capacity_owner is None


def test_issue_432_queued_meta_uses_normalized_scheduler_fields() -> None:
    classification = WorkClassification.build(
        scheduler_enabled=True,
        scheduler_domain="vllm:Qwen/Qwen3-0.6B",
        scheduler_session_key="sess-432",
        scheduler_domain_key_source="base_model_fallback",
    )

    meta = classification.queued_meta(
        op="sampling.asample",
        queued_at=123.0,
        extra_meta={"model_id": "model-432"},
    )

    assert meta == {
        "op": "sampling.asample",
        "model_id": "model-432",
        "queue_state": "queued",
        "queued_at": 123.0,
        "stage": "queued",
        "queue_kind": "scheduled",
        "scheduler_domain": "vllm:Qwen/Qwen3-0.6B",
        "scheduler_session_id": "sess-432",
        "scheduler_domain_key_source": "base_model_fallback",
        "scheduler_capacity_owner": "model_registry_inference_dp",
    }
