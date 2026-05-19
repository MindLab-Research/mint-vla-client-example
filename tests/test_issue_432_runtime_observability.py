import sys
import types
from types import SimpleNamespace

import pytest
import mint_server.logging_context as logging_context
import mint_server.backend.vllm_scheduler_observability as vllm_obs_mod
from mint_server.backend.runtime_observability import RuntimeObservability
from mint_server.backend.verl_training import VerlTrainingEngine
from mint_server.backend.vllm_scheduler_observability import VllmStatsObserver


def test_issue_432_verl_training_records_megatron_switch_metrics(monkeypatch) -> None:
    obs = RuntimeObservability()
    import mint_server.backend.runtime_observability as runtime_obs_mod

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



def test_issue_432_vllm_stats_observer_tracks_request_stage_timings() -> None:
    obs = VllmStatsObserver()

    obs.observe_actor_timing(
        seq_slot_wait_s=0.4,
        generate_lock_wait_s=0.1,
        engine_read_lock_wait_s=0.05,
        add_request_wait_s=0.2,
        add_request_exec_s=0.08,
        first_token_observed_s=1.3,
    )
    obs.observe_actor_timing(
        seq_slot_wait_s=1.0,
        generate_lock_wait_s=0.3,
        engine_read_lock_wait_s=0.15,
        add_request_wait_s=0.5,
        add_request_exec_s=0.12,
        first_token_observed_s=1.8,
    )

    snap = obs.snapshot()
    assert snap["seq_slot_wait_s_count"] == 2
    assert snap["seq_slot_wait_s_total"] == pytest.approx(1.4)
    assert snap["seq_slot_wait_s_max"] == pytest.approx(1.0)
    assert snap["seq_slot_wait_s_p50_recent"] == pytest.approx(0.7)
    assert snap["seq_slot_wait_s_p95_recent"] == pytest.approx(0.97)
    assert snap["add_request_wait_s_total"] == pytest.approx(0.7)
    assert snap["add_request_exec_s_max"] == pytest.approx(0.12)
    assert snap["first_token_observed_s_p50_recent"] == pytest.approx(1.55)
    assert snap["first_token_observed_s_p95_recent"] == pytest.approx(1.775)


def test_issue_432_vllm_stats_observer_drops_non_finite_actor_timings() -> None:
    obs = VllmStatsObserver()

    obs.observe_actor_timing(
        seq_slot_wait_s=float("nan"),
        add_request_wait_s=float("inf"),
        add_request_exec_s=float("-inf"),
    )

    snap = obs.snapshot()
    assert snap["seq_slot_wait_s_count"] == 0
    assert snap["add_request_wait_s_count"] == 0
    assert snap["add_request_exec_s_count"] == 0


def test_issue_432_install_vllm_iteration_observability_patches_fails_open(monkeypatch) -> None:
    monkeypatch.setattr(vllm_obs_mod, "_VLLM_PATCHES_INSTALLED", False)

    original_import = __import__

    def _explode(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm.v1.core.sched.scheduler":
            raise ImportError("boom")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", _explode)

    # Missing private vLLM symbols must not make engine startup fail.
    vllm_obs_mod.install_vllm_iteration_observability_patches()

    assert vllm_obs_mod._VLLM_PATCHES_INSTALLED is False


def test_issue_432_vllm_iteration_patch_resets_stale_step_timings(monkeypatch) -> None:
    monkeypatch.setattr(vllm_obs_mod, "_VLLM_PATCHES_INSTALLED", False)

    fake_sched_mod = types.ModuleType("vllm.v1.core.sched.scheduler")
    fake_core_mod = types.ModuleType("vllm.v1.engine.core")
    fake_output_mod = types.ModuleType("vllm.v1.engine.output_processor")
    fake_worker_mod = types.ModuleType("vllm.v1.worker.gpu_worker")

    class FakeScheduler:
        def __init__(self) -> None:
            self._mode = "first"

        def schedule(self):
            if self._mode == "first":
                return SimpleNamespace(
                    total_num_scheduled_tokens=8,
                    scheduled_new_reqs=[object()],
                    scheduled_cached_reqs=SimpleNamespace(num_reqs=0),
                )
            return SimpleNamespace(
                total_num_scheduled_tokens=4,
                scheduled_new_reqs=[],
                scheduled_cached_reqs=SimpleNamespace(num_reqs=1),
            )

        def make_stats(self):
            return SimpleNamespace()

    class FakeEngineCore:
        def __init__(self, scheduler) -> None:
            self.scheduler = scheduler

        def execute_model_with_error_logging(self, *_args, **_kwargs):
            return SimpleNamespace(_mint_worker_execute_model_s=0.3)

    class FakeOutputProcessor:
        def _update_stats_from_output(self, *_args, **_kwargs):
            return None

    class FakeWorker:
        def execute_model(self, *_args, **_kwargs):
            return SimpleNamespace()

    fake_sched_mod.Scheduler = FakeScheduler
    fake_core_mod.EngineCore = FakeEngineCore
    fake_output_mod.OutputProcessor = FakeOutputProcessor
    fake_worker_mod.Worker = FakeWorker

    monkeypatch.setitem(sys.modules, "vllm.v1.core.sched.scheduler", fake_sched_mod)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", fake_core_mod)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.output_processor", fake_output_mod)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_worker", fake_worker_mod)

    vllm_obs_mod.install_vllm_iteration_observability_patches()

    scheduler = FakeScheduler()
    engine_core = FakeEngineCore(scheduler)
    scheduler.schedule()
    engine_core.execute_model_with_error_logging(lambda *_a, **_k: None, None)
    stats = scheduler.make_stats()
    assert stats.mint_executor_execute_model_s >= 0.0
    assert stats.mint_worker_execute_model_s == pytest.approx(0.3)

    scheduler._mode = "second"
    scheduler.schedule()
    stats = scheduler.make_stats()
    assert not hasattr(stats, "mint_executor_execute_model_s")
    assert not hasattr(stats, "mint_worker_execute_model_s")


def test_issue_432_vllm_iteration_patch_tolerates_missing_engine_timing_hook(monkeypatch) -> None:
    monkeypatch.setattr(vllm_obs_mod, "_VLLM_PATCHES_INSTALLED", False)

    fake_sched_mod = types.ModuleType("vllm.v1.core.sched.scheduler")
    fake_core_mod = types.ModuleType("vllm.v1.engine.core")
    fake_output_mod = types.ModuleType("vllm.v1.engine.output_processor")

    class FakeScheduler:
        def schedule(self):
            return SimpleNamespace(
                total_num_scheduled_tokens=8,
                scheduled_new_reqs=[object()],
                scheduled_cached_reqs=SimpleNamespace(num_reqs=0),
            )

        def make_stats(self):
            return SimpleNamespace()

    class FakeEngineCore:
        pass

    class FakeOutputProcessor:
        def _update_stats_from_output(self, *_args, **_kwargs):
            return None

    fake_sched_mod.Scheduler = FakeScheduler
    fake_core_mod.EngineCore = FakeEngineCore
    fake_output_mod.OutputProcessor = FakeOutputProcessor

    monkeypatch.setitem(sys.modules, "vllm.v1.core.sched.scheduler", fake_sched_mod)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", fake_core_mod)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.output_processor", fake_output_mod)
    monkeypatch.delitem(sys.modules, "vllm.v1.worker.gpu_worker", raising=False)

    vllm_obs_mod.install_vllm_iteration_observability_patches()

    scheduler = FakeScheduler()
    scheduler.schedule()
    stats = scheduler.make_stats()
    assert stats.mint_total_scheduled_tokens == 8
    assert not hasattr(stats, "mint_executor_execute_model_s")


def test_issue_432_vllm_iteration_timings_do_not_replay_stale_values() -> None:
    obs = VllmStatsObserver()
    scheduler_stats = SimpleNamespace(
        num_waiting_reqs=1,
        num_running_reqs=1,
        kv_cache_usage=0.25,
        prefix_cache_stats=SimpleNamespace(queries=10, hits=4),
        mint_total_scheduled_tokens=8,
        mint_scheduled_new_requests=1,
        mint_scheduled_cached_requests=0,
        mint_executor_execute_model_s=0.4,
        mint_worker_execute_model_s=0.3,
    )
    iteration_stats = SimpleNamespace(
        num_preempted_reqs=0,
        num_prompt_tokens=8,
        num_generation_tokens=1,
        mint_prefill_requests=1,
        mint_decode_requests=0,
        time_to_first_tokens_iter=[],
        inter_token_latencies_iter=[],
        finished_requests=[],
    )

    obs.record(scheduler_stats, iteration_stats)
    snap = obs.snapshot()
    assert snap["executor_execute_model_s_count"] == 1
    assert snap["worker_execute_model_s_count"] == 1

    scheduler_stats_2 = SimpleNamespace(
        num_waiting_reqs=0,
        num_running_reqs=1,
        kv_cache_usage=0.1,
        prefix_cache_stats=SimpleNamespace(queries=0, hits=0),
        mint_total_scheduled_tokens=4,
        mint_scheduled_new_requests=0,
        mint_scheduled_cached_requests=1,
    )
    obs.record(scheduler_stats_2, iteration_stats)
    snap = obs.snapshot()
    assert snap["executor_execute_model_s_count"] == 1
    assert snap["worker_execute_model_s_count"] == 1


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


def test_issue_432_runtime_observability_tracks_training_ops_and_dense_actor_events() -> None:
    obs = RuntimeObservability()

    obs.record_training_operation(
        base_model="Qwen/Qwen3-0.6B",
        backend="peft",
        op="forward_backward",
        status="ok",
        failure_class="none",
        duration_s=1.5,
    )
    obs.record_training_operation(
        base_model="Qwen/Qwen3-0.6B",
        backend="peft",
        op="forward_backward",
        status="error",
        failure_class="cuda_fatal",
        duration_s=0.25,
    )
    obs.record_dense_actor_bind_decision(
        base_model="Qwen/Qwen3-0.6B",
        decision="rebind_refused_poisoned",
    )
    obs.record_dense_actor_fatal(
        base_model="Qwen/Qwen3-0.6B",
        op="forward_backward",
        failure_class="cuda_fatal",
    )
    obs.record_dense_actor_retire(
        base_model="Qwen/Qwen3-0.6B",
        outcome="ok",
    )

    snap = obs.snapshot()
    assert snap["training_operation_latency"] == [
        {
            "base_model": "Qwen/Qwen3-0.6B",
            "backend": "peft",
            "op": "forward_backward",
            "status": "error",
            "failure_class": "cuda_fatal",
            "count": 1,
            "duration_s_total": 0.25,
            "duration_s_max": 0.25,
        },
        {
            "base_model": "Qwen/Qwen3-0.6B",
            "backend": "peft",
            "op": "forward_backward",
            "status": "ok",
            "failure_class": "none",
            "count": 1,
            "duration_s_total": 1.5,
            "duration_s_max": 1.5,
        },
    ]
    assert snap["dense_actor_bind_decision"] == [
        {
            "base_model": "Qwen/Qwen3-0.6B",
            "decision": "rebind_refused_poisoned",
            "count": 1,
        }
    ]
    assert snap["dense_actor_fatal"] == [
        {
            "base_model": "Qwen/Qwen3-0.6B",
            "op": "forward_backward",
            "failure_class": "cuda_fatal",
            "count": 1,
        }
    ]
    assert snap["dense_actor_retire"] == [
        {
            "base_model": "Qwen/Qwen3-0.6B",
            "outcome": "ok",
            "count": 1,
        }
    ]


def test_issue_432_runtime_observability_keeps_recent_training_incidents() -> None:
    obs = RuntimeObservability()

    obs.record_training_incident(
        kind="contract_violation",
        base_model="Qwen/Qwen3-0.6B",
        backend="peft",
        op="forward_backward",
        status="error",
        failure_class="input_contract",
        request_id="req-561",
        session_id="session-561",
        detail="input_ids_out_of_range",
        context={"bad_input_positions": "[2]"},
    )

    snap = obs.snapshot()
    assert len(snap["recent_training_incidents"]) == 1
    incident = snap["recent_training_incidents"][0]
    assert incident["kind"] == "contract_violation"
    assert incident["base_model"] == "Qwen/Qwen3-0.6B"
    assert incident["backend"] == "peft"
    assert incident["op"] == "forward_backward"
    assert incident["status"] == "error"
    assert incident["failure_class"] == "input_contract"
    assert incident["request_id"] == "req-561"
    assert incident["session_id"] == "session-561"
    assert incident["detail"] == "input_ids_out_of_range"
    assert incident["context"] == {"bad_input_positions": "[2]"}
    assert isinstance(incident["ts"], float)


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
    assert snap["training_operation_latency"] == []
    assert snap["dense_actor_bind_decision"] == []
    assert snap["dense_actor_fatal"] == []
    assert snap["dense_actor_retire"] == []
    assert snap["recent_training_incidents"] == []


class _Recorder:
    def __init__(self) -> None:
        self.calls = []

    def add(self, value, attributes=None):
        self.calls.append(("add", value, attributes))

    def record(self, value, attributes=None):
        self.calls.append(("record", value, attributes))


def test_issue_432_otel_latency_metrics_include_failure_labels(monkeypatch) -> None:
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
        failure_class="none",
        duration_s=6.0,
    )
    logging_context.record_training_operation_latency_otel(
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
        op="forward_backward",
        status="canceled",
        failure_class="canceled",
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
                "status": "ok",
                "failure_class": "none",
            },
        ),
        (
            "add",
            1,
            {
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "backend": "megatron",
                "op": "forward_backward",
                "status": "canceled",
                "failure_class": "canceled",
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
                "status": "ok",
                "failure_class": "none",
            },
        ),
        (
            "record",
            3.0,
            {
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "backend": "megatron",
                "op": "forward_backward",
                "status": "canceled",
                "failure_class": "canceled",
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
                "failure_class": "canceled",
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
        mint_total_scheduled_tokens=96,
        mint_scheduled_new_requests=3,
        mint_scheduled_cached_requests=5,
        mint_executor_execute_model_s=0.42,
        mint_worker_execute_model_s=0.31,
    )
    iteration_stats = SimpleNamespace(
        num_preempted_reqs=3,
        num_prompt_tokens=80,
        num_generation_tokens=16,
        mint_prefill_requests=3,
        mint_decode_requests=5,
        time_to_first_tokens_iter=[0.9, 1.2],
        inter_token_latencies_iter=[0.05, 0.07, 0.08],
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

    assert snap["scheduler_waiting_requests"] == 4
    assert snap["scheduler_running_requests"] == 2
    assert snap["scheduler_kv_cache_usage_ratio"] == 0.75
    assert snap["prefix_cache_queries_total"] == 100
    assert snap["prefix_cache_hits_total"] == 60
    assert snap["prefix_cache_hit_ratio"] == 0.6
    assert snap["preemptions_total"] == 3
    assert snap["queue_time_s_total"] == pytest.approx(4.0)
    assert snap["queue_time_s_count"] == 2
    assert snap["queue_time_s_max"] == pytest.approx(2.5)
    assert snap["queue_time_s_p50_recent"] == pytest.approx(2.0)
    assert snap["queue_time_s_p95_recent"] == pytest.approx(2.45)
    assert snap["prefill_time_s_total"] == pytest.approx(5.0)
    assert snap["prefill_time_s_count"] == 2
    assert snap["prefill_time_s_max"] == pytest.approx(3.0)
    assert snap["decode_time_s_total"] == pytest.approx(12.0)
    assert snap["decode_time_s_count"] == 2
    assert snap["decode_time_s_max"] == pytest.approx(7.0)
    assert snap["time_per_output_token_s_total"] == pytest.approx(0.2)
    assert snap["time_per_output_token_s_count"] == 2
    assert snap["time_per_output_token_s_max"] == pytest.approx(0.12)
    assert snap["scheduled_tokens_iter_total"] == pytest.approx(96)
    assert snap["scheduled_new_requests_iter_total"] == pytest.approx(3)
    assert snap["scheduled_cached_requests_iter_total"] == pytest.approx(5)
    assert snap["prefill_requests_iter_total"] == pytest.approx(3)
    assert snap["decode_requests_iter_total"] == pytest.approx(5)
    assert snap["prompt_tokens_iter_total"] == pytest.approx(80)
    assert snap["generation_tokens_iter_total"] == pytest.approx(16)
    assert snap["time_to_first_token_s_total"] == pytest.approx(2.1)
    assert snap["inter_token_latency_s_total"] == pytest.approx(0.2)
    assert snap["executor_execute_model_s_total"] == pytest.approx(0.42)
    assert snap["worker_execute_model_s_total"] == pytest.approx(0.31)
    assert snap["seq_slot_wait_s_count"] == 0
    assert snap["add_request_wait_s_count"] == 0


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
