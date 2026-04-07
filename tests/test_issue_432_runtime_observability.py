from types import SimpleNamespace

from tinker_server.backend.runtime_observability import RuntimeObservability
from tinker_server.backend.verl_training import VerlTrainingEngine


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


def test_issue_432_runtime_observability_tracks_vllm_workload_and_active_requests() -> None:
    obs = RuntimeObservability()

    obs.begin_vllm_request(base_model="Qwen/Qwen3-0.6B", op="asample")
    obs.begin_vllm_request(base_model="Qwen/Qwen3-0.6B", op="asample")
    obs.finish_vllm_request(
        base_model="Qwen/Qwen3-0.6B",
        op="asample",
        status="ok",
        prompt_tokens=128,
        generated_tokens=32,
        duration_s=1.25,
    )
    obs.finish_vllm_request(
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
            "base_model": "Qwen/Qwen3-0.6B",
            "op": "asample",
            "active_requests": 0,
        }
    ]
    assert snap["vllm_workload"] == [
        {
            "base_model": "Qwen/Qwen3-0.6B",
            "op": "asample",
            "status": "error",
            "requests_total": 1,
            "prompt_tokens_total": 64,
            "generated_tokens_total": 0,
            "duration_s_total": 0.5,
            "duration_s_max": 0.5,
        },
        {
            "base_model": "Qwen/Qwen3-0.6B",
            "op": "asample",
            "status": "ok",
            "requests_total": 1,
            "prompt_tokens_total": 128,
            "generated_tokens_total": 32,
            "duration_s_total": 1.25,
            "duration_s_max": 1.25,
        },
    ]
