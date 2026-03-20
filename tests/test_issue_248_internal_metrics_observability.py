from __future__ import annotations

import asyncio

import tinker_server.routes.internal as internal_routes


async def _fake_admission_stats() -> dict:
    return {
        "capacity": {"capacity": 16, "inflight": 3},
        "work_queue": {
            "depth": 2,
            "depth_legacy": 1,
            "depth_scheduled": 1,
            "enqueued": 10,
            "dequeued": 8,
            "by_executor": {"sampling.asample": 1, "weights.save_weights": 1},
            "age_stats": {"oldest_queued_s": 12.5, "avg_queued_s": 6.25},
            "execution_time_s_by_op": {
                "training.train_step": {
                    "last": 4.5,
                    "ema": 4.0,
                    "sum": 20.0,
                    "count": 5,
                    "max": 6.0,
                }
            },
            "scheduler_enabled": True,
            "scheduler_picks_total": 9,
            "scheduler_switches_total": 3,
            "scheduler_starvation_picks_total": 1,
            "scheduler_wait_s_sum": 12.0,
            "scheduler_domains_total": 2,
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
            "timeout_counts": {
                "queue": 2,
                "execution": 1,
                "total": 3,
                "by_op": {
                    "sampling.asample": {"queue": 2, "execution": 1, "total": 3},
                },
            },
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
        "driver_state": {
            "sdk_sessions_fallback": 7,
            "session_heartbeat_entries": 11,
            "lora_load_locks": 3,
            "sampling_sessions_total": 5,
            "sampling_sessions_multi_lora": 4,
            "sampling_sessions_base_model": 1,
            "sampling_sessions_lora_loaded": 2,
            "sampling_sessions_inflight": 1,
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
        "mint_work_queue_depth_legacy 1",
        "mint_work_queue_depth_scheduled 1",
        "mint_work_queue_scheduler_enabled 1",
        "mint_work_queue_scheduler_picks_total 9",
        "mint_work_queue_scheduler_switches_total 3",
        "mint_work_queue_scheduler_starvation_picks_total 1",
        "mint_work_queue_scheduler_wait_s_sum 12",
        "mint_work_queue_scheduler_domains_total 2",
        'mint_work_queue_depth{executor="sampling.asample"} 1',
        "mint_work_queue_oldest_queued_s 12.5",
        'mint_work_queue_execution_last_s{op="training.train_step"} 4.5',
        'mint_work_queue_execution_ema_s{op="training.train_step"} 4',
        'mint_work_queue_execution_sum_s{op="training.train_step"} 20',
        'mint_work_queue_execution_count{op="training.train_step"} 5',
        'mint_work_queue_execution_max_s{op="training.train_step"} 6',
        "mint_future_store_pending 1",
        'mint_future_store_pending{op="asample"} 1',
        "mint_future_store_oldest_pending_s 8",
        "mint_future_store_result_refs_count 4",
        'mint_future_store_timeouts_total{kind="queue"} 2',
        'mint_future_store_timeouts_total{kind="execution"} 1',
        'mint_future_store_timeouts_total{kind="total"} 3',
        'mint_future_store_timeouts_total{kind="queue",op="sampling.asample"} 2',
        'mint_actor_rss_bytes{actor="future_store"} 3000',
        'mint_resource_pool_actor_rss_bytes{actor_name="vllm-1",actor_type="vllm",model="Qwen/Qwen3-4B-Instruct-2507"} 100',
        "mint_api_server_process_rss_bytes 12345",
        "mint_driver_process_rss_bytes 12345",
        "mint_driver_sdk_sessions_fallback 7",
        "mint_driver_session_heartbeat_entries 11",
        "mint_driver_lora_load_locks 3",
        "mint_driver_sampling_sessions_total 5",
        "mint_driver_sampling_sessions_multi_lora 4",
        "mint_driver_sampling_sessions_base_model 1",
        "mint_driver_sampling_sessions_lora_loaded 2",
        "mint_driver_sampling_sessions_inflight 1",
        "mint_metrics_up 1",
    )
    for line in expected_lines:
        assert line in text, f"missing metric line: {line}"
