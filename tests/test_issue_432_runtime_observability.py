from types import SimpleNamespace

import tinker_server.logging_context as logging_context
from tinker_server.backend.runtime_observability import RuntimeObservability
from tinker_server.backend.verl_training import VerlTrainingEngine
from tinker_server.backend.vllm_scheduler_observability import VllmStatsObserver


def test_issue_432_verl_training_records_megatron_switch_metrics(monkeypatch) -> None:
    obs = RuntimeObservability()
    import tinker_server.backend.runtime_observability as runtime_obs_mod

    monkeypatch.setattr(runtime_obs_mod, "runtime_observability", obs)
    engine = VerlTrainingEngine()
    session = SimpleNamespace(backend="megatron", base_model="Qwen/Qwen3-30B-A3B-Instruct-2507")
    result = {
        "metrics": {
            "session_switch_total:sum": 1.0,
            "session_switch_existing_session:mean": 1.0,
            "session_switch_save_s:sum": 1.5,
            "session_switch_swap_s:sum": 2.0,
            "session_switch_load_s:sum": 2.5,
            "session_switch_reset_bias_s:sum": 0.25,
            "session_switch_total_s:sum": 6.25,
        }
    }

    engine._record_megatron_result_metrics(session, result)

    snap = obs.snapshot()
    assert snap["megatron_session_switch"][0]["count"] == 1
    assert snap["megatron_session_switch"][0]["base_model"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert snap["megatron_session_switch"][0]["session_state"] == "existing"
    assert snap["megatron_session_switch"][0]["total_s_total"] == 6.25



def test_issue_432_runtime_observability_tracks_megatron_session_switch() -> None:
    obs = RuntimeObservability()

    obs.record_megatron_session_switch(
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        session_state="existing",
        save_s=1.0,
        swap_s=2.0,
        load_s=3.0,
        reset_bias_s=0.5,
        total_s=6.5,
    )
    obs.record_megatron_session_switch(
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        session_state="existing",
        save_s=0.5,
        swap_s=1.0,
        load_s=1.5,
        reset_bias_s=0.25,
        total_s=3.25,
    )
    obs.record_megatron_session_switch_failure(
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        reason="partial_swap",
    )
    obs.record_megatron_actor_lifecycle(
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        event="startup_timeout",
    )

    snap = obs.snapshot()
    rows = snap["megatron_session_switch"]
    assert rows == [
        {
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "session_state": "existing",
            "count": 2,
            "save_s_total": 1.5,
            "save_s_max": 1.0,
            "swap_s_total": 3.0,
            "swap_s_max": 2.0,
            "load_s_total": 4.5,
            "load_s_max": 3.0,
            "reset_bias_s_total": 0.75,
            "reset_bias_s_max": 0.5,
            "total_s_total": 9.75,
            "total_s_max": 6.5,
        }
    ]
    assert snap["megatron_session_switch_failures"] == [
        {
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "reason": "partial_swap",
            "count": 1,
        }
    ]
    assert snap["megatron_actor_lifecycle"] == [
        {
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "event": "startup_timeout",
            "count": 1,
        }
    ]


def test_issue_432_runtime_observability_tracks_vllm_workload_and_active_requests() -> None:
    obs = RuntimeObservability()

    obs.begin_vllm_request(actor_name="vllm-1", base_model="Qwen/Qwen3-0.6B", op="asample")
    obs.begin_vllm_request(actor_name="vllm-1", base_model="Qwen/Qwen3-0.6B", op="asample")
    obs.finish_vllm_request(
        actor_name="vllm-1",
        base_model="Qwen/Qwen3-0.6B",
        op="asample",
        status="ok",
        prompt_tokens=128,
        generated_tokens=32,
        duration_s=1.25,
        ttft_s=0.4,
        tpot_s=0.03,
    )
    obs.finish_vllm_request(
        actor_name="vllm-1",
        base_model="Qwen/Qwen3-0.6B",
        op="asample",
        status="error",
        prompt_tokens=64,
        generated_tokens=0,
        duration_s=0.5,
    )

    snap = obs.snapshot()
    assert snap["vllm_active_requests"] == [
        {
            "actor_name": "vllm-1",
            "base_model": "Qwen/Qwen3-0.6B",
            "op": "asample",
            "active_requests": 0,
        }
    ]
    assert snap["vllm_workload"] == [
        {
            "actor_name": "vllm-1",
            "base_model": "Qwen/Qwen3-0.6B",
            "op": "asample",
            "status": "error",
            "requests_total": 1,
            "prompt_tokens_total": 64,
            "generated_tokens_total": 0,
            "duration_s_total": 0.5,
            "duration_s_max": 0.5,
            "ttft_s_total": 0.0,
            "ttft_s_max": 0.0,
            "ttft_s_count": 0,
            "tpot_s_total": 0.0,
            "tpot_s_max": 0.0,
            "tpot_s_count": 0,
        },
        {
            "actor_name": "vllm-1",
            "base_model": "Qwen/Qwen3-0.6B",
            "op": "asample",
            "status": "ok",
            "requests_total": 1,
            "prompt_tokens_total": 128,
            "generated_tokens_total": 32,
            "duration_s_total": 1.25,
            "duration_s_max": 1.25,
            "ttft_s_total": 0.4,
            "ttft_s_max": 0.4,
            "ttft_s_count": 1,
            "tpot_s_total": 0.03,
            "tpot_s_max": 0.03,
            "tpot_s_count": 1,
        },
    ]
    assert "vllm_actor_latency" not in snap
    assert "training_operation_latency" not in snap


class _Recorder:
    def __init__(self) -> None:
        self.calls = []

    def add(self, value, attributes=None):
        self.calls.append(("add", value, attributes))

    def record(self, value, attributes=None):
        self.calls.append(("record", value, attributes))


def test_issue_432_otel_latency_metrics_ignore_failures(monkeypatch) -> None:
    events = []
    counter = _Recorder()
    hist = _Recorder()

    monkeypatch.setattr(logging_context, "_OTEL_ENABLED", True)
    monkeypatch.setattr(logging_context, "_VLLM_ACTOR_REQUEST_COUNTER", counter)
    monkeypatch.setattr(logging_context, "_VLLM_ACTOR_REQUEST_DURATION_HISTOGRAM", hist)
    monkeypatch.setattr(logging_context, "_TRAINING_OPERATION_COUNTER", counter)
    monkeypatch.setattr(logging_context, "_TRAINING_OPERATION_DURATION_HISTOGRAM", hist)
    monkeypatch.setattr(logging_context, "_record_current_span_event", lambda name, attrs: events.append((name, attrs)))

    logging_context.record_vllm_actor_latency_otel(
        actor_name="vllm-1",
        base_model="Qwen/Qwen3-0.6B",
        op="asample",
        status="ok",
        duration_s=1.25,
    )
    logging_context.record_vllm_actor_latency_otel(
        actor_name="vllm-1",
        base_model="Qwen/Qwen3-0.6B",
        op="asample",
        status="error",
        duration_s=0.5,
    )
    logging_context.record_training_operation_latency_otel(
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
        op="forward_backward",
        status="ok",
        duration_s=6.0,
    )
    logging_context.record_training_operation_latency_otel(
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
        op="forward_backward",
        status="canceled",
        duration_s=3.0,
    )

    assert counter.calls == [
        (
            "add",
            1,
            {"actor_name": "vllm-1", "base_model": "Qwen/Qwen3-0.6B", "op": "asample"},
        ),
        (
            "add",
            1,
            {
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "backend": "megatron",
                "op": "forward_backward",
            },
        ),
    ]
    assert hist.calls == [
        (
            "record",
            1.25,
            {"actor_name": "vllm-1", "base_model": "Qwen/Qwen3-0.6B", "op": "asample"},
        ),
        (
            "record",
            6.0,
            {
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "backend": "megatron",
                "op": "forward_backward",
            },
        ),
    ]
    assert events == [
        (
            "mint.vllm_actor_request.failure",
            {
                "actor_name": "vllm-1",
                "base_model": "Qwen/Qwen3-0.6B",
                "op": "asample",
                "status": "error",
                "duration_s": 0.5,
            },
        ),
        (
            "mint.training_operation.failure",
            {
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "backend": "megatron",
                "op": "forward_backward",
                "status": "canceled",
                "duration_s": 3.0,
            },
        ),
    ]


def test_issue_432_vllm_stats_observer_tracks_scheduler_and_finished_request_metrics() -> None:
    obs = VllmStatsObserver()
    scheduler_stats = SimpleNamespace(
        num_waiting_reqs=4,
        num_running_reqs=2,
        kv_cache_usage=0.75,
        prefix_cache_stats=SimpleNamespace(queries=100, hits=60),
    )
    iteration_stats = SimpleNamespace(
        num_preempted_reqs=3,
        finished_requests=[
            SimpleNamespace(
                queued_time=1.5,
                prefill_time=2.0,
                decode_time=5.0,
                mean_time_per_output_token=0.08,
            ),
            SimpleNamespace(
                queued_time=2.5,
                prefill_time=3.0,
                decode_time=7.0,
                mean_time_per_output_token=0.12,
            ),
        ],
    )

    obs.record(scheduler_stats, iteration_stats)
    snap = obs.snapshot()

    assert snap == {
        "scheduler_waiting_requests": 4,
        "scheduler_running_requests": 2,
        "scheduler_kv_cache_usage_ratio": 0.75,
        "prefix_cache_queries_total": 100,
        "prefix_cache_hits_total": 60,
        "prefix_cache_hit_ratio": 0.6,
        "preemptions_total": 3,
        "queue_time_s_total": 4.0,
        "queue_time_s_count": 2,
        "queue_time_s_max": 2.5,
        "prefill_time_s_total": 5.0,
        "prefill_time_s_count": 2,
        "prefill_time_s_max": 3.0,
        "decode_time_s_total": 12.0,
        "decode_time_s_count": 2,
        "decode_time_s_max": 7.0,
        "time_per_output_token_s_total": 0.2,
        "time_per_output_token_s_count": 2,
        "time_per_output_token_s_max": 0.12,
    }


def test_issue_432_scheduler_decision_otel_records_experience_metrics(monkeypatch) -> None:
    decision_counter = _Recorder()
    switch_counter = _Recorder()
    wait_hist = _Recorder()
    ready_hist = _Recorder()
    depth_hist = _Recorder()

    monkeypatch.setattr(logging_context, "_OTEL_ENABLED", True)
    monkeypatch.setattr(logging_context, "_SCHEDULER_DECISION_COUNTER", decision_counter)
    monkeypatch.setattr(logging_context, "_SCHEDULER_SWITCH_COUNTER", switch_counter)
    monkeypatch.setattr(logging_context, "_SCHEDULER_QUEUE_WAIT_HISTOGRAM", wait_hist)
    monkeypatch.setattr(logging_context, "_SCHEDULER_READY_SESSIONS_HISTOGRAM", ready_hist)
    monkeypatch.setattr(logging_context, "_SCHEDULER_CHOSEN_QUEUE_DEPTH_HISTOGRAM", depth_hist)

    logging_context.record_scheduler_decision_otel(
        op="training.forward_backward",
        backend="megatron",
        queue_kind="scheduled",
        reason="fairness_oldest",
        queue_wait_s=12.5,
        switched=True,
        ready_sessions=3,
        chosen_queue_depth=2,
    )

    attrs = {
        "op": "training.forward_backward",
        "backend": "megatron",
        "queue_kind": "scheduled",
        "reason": "fairness_oldest",
    }
    assert decision_counter.calls == [("add", 1, attrs)]
    assert switch_counter.calls == [("add", 1, attrs)]
    assert wait_hist.calls == [("record", 12.5, attrs)]
    assert ready_hist.calls == [("record", 3, attrs)]
    assert depth_hist.calls == [("record", 2, attrs)]


def test_issue_432_megatron_switch_otel_flushes_pending_aggregate(monkeypatch) -> None:
    event_counter = _Recorder()
    duration_counter = _Recorder()

    monkeypatch.setattr(logging_context, "_OTEL_ENABLED", True)
    monkeypatch.setattr(logging_context, "_MEGATRON_SESSION_SWITCH_COUNTER", event_counter)
    monkeypatch.setattr(logging_context, "_MEGATRON_SESSION_SWITCH_DURATION_COUNTER", duration_counter)

    obs = RuntimeObservability()
    obs.record_megatron_session_switch(
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        session_state="existing",
        save_s=1.0,
        swap_s=2.0,
        load_s=3.0,
        reset_bias_s=0.5,
        total_s=6.5,
    )
    obs.record_megatron_session_switch(
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        session_state="existing",
        save_s=0.5,
        swap_s=1.0,
        load_s=1.5,
        reset_bias_s=0.25,
        total_s=3.25,
    )

    obs.flush_otel()
    obs.flush_otel()

    assert event_counter.calls == [
        (
            "add",
            2,
            {
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "session_state": "existing",
            },
        )
    ]
    assert duration_counter.calls == [
        (
            "add",
            1.5,
            {
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "session_state": "existing",
                "phase": "save",
            },
        ),
        (
            "add",
            3.0,
            {
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "session_state": "existing",
                "phase": "swap",
            },
        ),
        (
            "add",
            4.5,
            {
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "session_state": "existing",
                "phase": "load",
            },
        ),
        (
            "add",
            0.75,
            {
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "session_state": "existing",
                "phase": "reset_bias",
            },
        ),
        (
            "add",
            9.75,
            {
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "session_state": "existing",
                "phase": "total",
            },
        ),
    ]
