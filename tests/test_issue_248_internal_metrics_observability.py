from __future__ import annotations

import asyncio

import tinker_server.routes.internal as internal_routes


async def _fake_admission_stats() -> dict:
    return {
        "capacity": {"capacity": 16, "inflight": 3},
        "work_queue": {
            "depth": 2,
            "enqueued": 10,
            "dequeued": 8,
            "by_executor": {"sampling.asample": 1, "weights.save_weights": 1},
            "age_stats": {"oldest_queued_s": 12.5, "avg_queued_s": 6.25},
            "scheduler_arbitration_total": 9,
            "scheduler_arbitration_by_winner": {"legacy": 4, "scheduled": 5},
            "scheduler_arbitration_by_reason": {"legacy_head_older": 4, "scheduled_starvation": 2},
            "scheduled_dequeue_stats": [
                {
                    "scheduler_domain": "vllm:Qwen/Qwen3-4B-Instruct-2507::replica::0",
                    "reason": "starvation",
                    "op": "sampling.asample",
                    "total": 6,
                }
            ],
            "legacy_dequeue_stats": [
                {"reason": "fifo", "op": "sampling.asample", "total": 4},
            ],
            "scheduler_domains": {
                "vllm:Qwen/Qwen3-4B-Instruct-2507::replica::0": {
                    "backend": "vllm",
                    "pending_requests": 3,
                    "active_sessions": 2,
                    "oldest_queued_s": 11.0,
                    "inflight_workers": 1,
                    "capacity_owner": "vllm_replica_single_worker",
                    "capacity_workers": 2,
                    "admissible": True,
                    "service_gap_s": 5.5,
                    "stats": {"picks": 6, "starvation_picks": 1},
                }
            },
        },
        "future_store": {
            "pending": 1,
            "results": 4,
            "errors": 0,
            "refs": 5,
            "meta": 5,
            "expired": 1,
            "retrieved": 8,
            "execution_timeout_s": 3600,
            "queue_timeout_s": 100,
            "result_ttl_s": 100,
            "tombstone_ttl_s": 10,
            "by_op": {"asample": {"pending": 1, "results": 4, "errors": 0}},
            "age_stats": {
                "oldest_pending_s": 8.0,
                "oldest_done_s": 3.0,
                "avg_pending_s": 4.0,
                "avg_done_s": 1.5,
            },
            "payload_stats": {"result_refs_count": 4, "errors_count": 0, "refs_count": 5},
        },
        "actors": {
            "capacity_manager": {"rss_bytes": 1000},
            "api_work_queue": {"rss_bytes": 2000},
            "future_store": {"rss_bytes": 3000},
            "resource_pool": [
                {
                    "actor_type": "vllm",
                    "base_model": "Qwen/Qwen3-4B-Instruct-2507",
                    "actor_name": "vllm-1",
                    "idle_time": 2,
                    "age": 50,
                    "rss_bytes": 100,
                }
            ],
        },
        "process": {"rss_bytes": 12345, "pid": 999},
        "runtime_observability": {
            "megatron_session_switch": [
                {
                    "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                    "session_state": "existing",
                    "count": 3,
                    "save_s_total": 9.0,
                    "save_s_max": 4.0,
                    "swap_s_total": 6.0,
                    "swap_s_max": 2.5,
                    "load_s_total": 12.0,
                    "load_s_max": 5.0,
                    "reset_bias_s_total": 1.5,
                    "reset_bias_s_max": 0.8,
                    "total_s_total": 28.5,
                    "total_s_max": 10.5,
                }
            ],
            "vllm_workload": [
                {
                    "base_model": "Qwen/Qwen3-4B-Instruct-2507",
                    "op": "asample",
                    "status": "ok",
                    "requests_total": 8,
                    "prompt_tokens_total": 4096,
                    "generated_tokens_total": 512,
                    "duration_s_total": 24.0,
                    "duration_s_max": 5.5,
                }
            ],
            "vllm_active_requests": [
                {
                    "base_model": "Qwen/Qwen3-4B-Instruct-2507",
                    "op": "asample",
                    "active_requests": 2,
                }
            ],
        },
    }


def test_issue_248_internal_metrics_exposes_phase1_and_phase2_fields(monkeypatch) -> None:
    monkeypatch.setattr(internal_routes, "admission_stats", _fake_admission_stats)
    resp = asyncio.run(internal_routes.metrics())
    text = resp.body.decode("utf-8")

    expected_lines = (
        "mint_capacity_capacity 16",
        "mint_capacity_inflight 3",
        "mint_work_queue_depth 2",
        'mint_work_queue_depth{executor="sampling.asample"} 1',
        "mint_work_queue_oldest_queued_s 12.5",
        "tinker_work_queue_scheduler_arbitration_total 9",
        'tinker_work_queue_scheduler_arbitration_total{winner_bucket="scheduled"} 5',
        'tinker_work_queue_scheduler_domain_dequeue_total{backend="vllm",execution_scope="local",op="sampling.asample",reason="starvation",scheduler_domain="vllm:Qwen/Qwen3-4B-Instruct-2507::replica::0"} 6',
        'tinker_work_queue_legacy_dequeue_total{execution_scope="local",op="sampling.asample",reason="fifo"} 4',
        'tinker_work_queue_scheduler_domain_pending_requests{backend="vllm",execution_scope="local",scheduler_domain="vllm:Qwen/Qwen3-4B-Instruct-2507::replica::0"} 3',
        'tinker_work_queue_scheduler_domain_inflight_workers{backend="vllm",execution_scope="local",scheduler_domain="vllm:Qwen/Qwen3-4B-Instruct-2507::replica::0"} 1',
        'tinker_work_queue_scheduler_domain_dequeue_picks_total{backend="vllm",execution_scope="local",scheduler_domain="vllm:Qwen/Qwen3-4B-Instruct-2507::replica::0"} 6',
        "mint_future_store_pending 1",
        'mint_future_store_pending{op="asample"} 1',
        "mint_future_store_oldest_pending_s 8",
        "mint_future_store_result_refs_count 4",
        'mint_actor_rss_bytes{actor="future_store"} 3000',
        'mint_resource_pool_actor_rss_bytes{actor_name="vllm-1",actor_type="vllm",model="Qwen/Qwen3-4B-Instruct-2507"} 100',
        "mint_api_server_process_rss_bytes 12345",
        "mint_metrics_up 1",
    )
    for line in expected_lines:
        assert line in text, f"missing metric line: {line}"
