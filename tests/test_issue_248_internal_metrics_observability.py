from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass

import tinker_server.routes.internal as internal_routes


async def _fake_admission_stats(*, include_actor_rss: bool = True) -> dict:
    actors = {
        "resource_pool": [
            {
                "actor_type": "vllm",
                "base_model": "Qwen/Qwen3-4B-Instruct-2507",
                "actor_name": "vllm-1",
                "idle_time": 2,
                "age": 50,
                "rss_bytes": 100,
                "node_id": "node-a",
                "metadata": {
                    "hostname": "host-a",
                    "gpu_indices": [0],
                    "scheduler_waiting_requests": 4,
                    "scheduler_running_requests": 2,
                    "scheduler_kv_cache_usage_ratio": 0.75,
                    "prefix_cache_queries_total": 100,
                    "prefix_cache_hits_total": 60,
                    "prefix_cache_hit_ratio": 0.6,
                    "preemptions_total": 3,
                    "queue_time_s_total": 12.0,
                    "queue_time_s_count": 8,
                    "queue_time_s_max": 3.5,
                    "queue_time_s_p50_recent": 1.4,
                    "queue_time_s_p95_recent": 3.2,
                    "prefill_time_s_total": 16.0,
                    "prefill_time_s_count": 8,
                    "prefill_time_s_max": 4.0,
                    "prefill_time_s_p50_recent": 1.8,
                    "prefill_time_s_p95_recent": 3.8,
                    "decode_time_s_total": 40.0,
                    "decode_time_s_count": 8,
                    "decode_time_s_max": 9.0,
                    "decode_time_s_p50_recent": 4.1,
                    "decode_time_s_p95_recent": 8.7,
                    "time_per_output_token_s_total": 0.96,
                    "time_per_output_token_s_count": 8,
                    "time_per_output_token_s_max": 0.2,
                    "time_per_output_token_s_p50_recent": 0.09,
                    "time_per_output_token_s_p95_recent": 0.19,
                    "scheduled_tokens_iter_total": 512.0,
                    "scheduled_tokens_iter_count": 8,
                    "scheduled_tokens_iter_max": 96.0,
                    "scheduled_tokens_iter_p50_recent": 64.0,
                    "scheduled_tokens_iter_p95_recent": 92.8,
                    "scheduled_new_requests_iter_total": 24.0,
                    "scheduled_new_requests_iter_count": 8,
                    "scheduled_new_requests_iter_max": 5.0,
                    "scheduled_new_requests_iter_p50_recent": 3.0,
                    "scheduled_new_requests_iter_p95_recent": 4.8,
                    "scheduled_cached_requests_iter_total": 40.0,
                    "scheduled_cached_requests_iter_count": 8,
                    "scheduled_cached_requests_iter_max": 7.0,
                    "scheduled_cached_requests_iter_p50_recent": 5.0,
                    "scheduled_cached_requests_iter_p95_recent": 6.8,
                    "prefill_requests_iter_total": 20.0,
                    "prefill_requests_iter_count": 8,
                    "prefill_requests_iter_max": 4.0,
                    "prefill_requests_iter_p50_recent": 2.5,
                    "prefill_requests_iter_p95_recent": 3.8,
                    "decode_requests_iter_total": 44.0,
                    "decode_requests_iter_count": 8,
                    "decode_requests_iter_max": 8.0,
                    "decode_requests_iter_p50_recent": 5.5,
                    "decode_requests_iter_p95_recent": 7.7,
                    "prompt_tokens_iter_total": 4096.0,
                    "prompt_tokens_iter_count": 8,
                    "prompt_tokens_iter_max": 768.0,
                    "prompt_tokens_iter_p50_recent": 512.0,
                    "prompt_tokens_iter_p95_recent": 742.4,
                    "generation_tokens_iter_total": 320.0,
                    "generation_tokens_iter_count": 8,
                    "generation_tokens_iter_max": 64.0,
                    "generation_tokens_iter_p50_recent": 40.0,
                    "generation_tokens_iter_p95_recent": 60.8,
                    "time_to_first_token_s_total": 9.6,
                    "time_to_first_token_s_count": 8,
                    "time_to_first_token_s_max": 1.8,
                    "time_to_first_token_s_p50_recent": 1.1,
                    "time_to_first_token_s_p95_recent": 1.7,
                    "inter_token_latency_s_total": 0.88,
                    "inter_token_latency_s_count": 8,
                    "inter_token_latency_s_max": 0.16,
                    "inter_token_latency_s_p50_recent": 0.1,
                    "inter_token_latency_s_p95_recent": 0.15,
                    "executor_execute_model_s_total": 3.2,
                    "executor_execute_model_s_count": 8,
                    "executor_execute_model_s_max": 0.52,
                    "executor_execute_model_s_p50_recent": 0.39,
                    "executor_execute_model_s_p95_recent": 0.5,
                    "worker_execute_model_s_total": 2.56,
                    "worker_execute_model_s_count": 8,
                    "worker_execute_model_s_max": 0.41,
                    "worker_execute_model_s_p50_recent": 0.31,
                    "worker_execute_model_s_p95_recent": 0.4,
                    "seq_slot_wait_s_total": 5.0,
                    "seq_slot_wait_s_count": 8,
                    "seq_slot_wait_s_max": 1.4,
                    "seq_slot_wait_s_p50_recent": 0.5,
                    "seq_slot_wait_s_p95_recent": 1.2,
                    "generate_lock_wait_s_total": 1.2,
                    "generate_lock_wait_s_count": 8,
                    "generate_lock_wait_s_max": 0.4,
                    "generate_lock_wait_s_p50_recent": 0.1,
                    "generate_lock_wait_s_p95_recent": 0.35,
                    "engine_read_lock_wait_s_total": 0.8,
                    "engine_read_lock_wait_s_count": 8,
                    "engine_read_lock_wait_s_max": 0.3,
                    "engine_read_lock_wait_s_p50_recent": 0.08,
                    "engine_read_lock_wait_s_p95_recent": 0.28,
                    "add_request_wait_s_total": 2.4,
                    "add_request_wait_s_count": 8,
                    "add_request_wait_s_max": 0.7,
                    "add_request_wait_s_p50_recent": 0.2,
                    "add_request_wait_s_p95_recent": 0.6,
                    "add_request_exec_s_total": 0.64,
                    "add_request_exec_s_count": 8,
                    "add_request_exec_s_max": 0.16,
                    "add_request_exec_s_p50_recent": 0.07,
                    "add_request_exec_s_p95_recent": 0.15,
                    "first_token_observed_s_total": 10.4,
                    "first_token_observed_s_count": 8,
                    "first_token_observed_s_max": 1.9,
                    "first_token_observed_s_p50_recent": 1.1,
                    "first_token_observed_s_p95_recent": 1.8,
                },
            },
            {
                "actor_type": "megatron",
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "actor_name": "megatron-1",
                "idle_time": 7,
                "age": 120,
                "rss_bytes": 200,
                "node_id": "node-b",
                "metadata": {
                    "hostname": "host-b",
                    "gpu_bindings": [
                        {"hostname": "host-b", "node_id": "node-b", "gpu_index": 0, "rank": 0},
                        {"hostname": "host-b", "node_id": "node-b", "gpu_index": 1, "rank": 1},
                    ],
                    "active_sessions": 1,
                    "session_unknown": 0,
                    "session_step": 17,
                    "learning_rate": 5e-5,
                    "gpu_memory_allocated_bytes": 48000000000,
                    "gpu_memory_reserved_bytes": 52000000000,
                    "gpu_memory_fragmentation_bytes": 4000000000,
                },
            },
        ],
        "resource_pool_metadata_cache": [
            {
                "actor_type": "megatron",
                "cache_hits_total": 9,
                "cache_stale_total": 3,
                "refresh_success_total": 2,
                "refresh_failures_total": 1,
            },
            {
                "actor_type": "vllm",
                "cache_hits_total": 12,
                "cache_stale_total": 4,
                "refresh_success_total": 3,
                "refresh_failures_total": 0,
            },
        ],
    }
    if include_actor_rss:
        actors.update(
            {
                "capacity_manager": {"rss_bytes": 1000},
                "api_work_queue": {"rss_bytes": 2000},
                "future_store": {"rss_bytes": 3000},
            }
        )
    else:
        actors["resource_pool"][0].pop("rss_bytes", None)
        actors["resource_pool"][0]["rss_cache_state"] = "unknown"
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
        "actors": actors,
        "process": {"rss_bytes": 12345, "pid": 999},
        "queue_execution_runtime": {
            "sampling_sessions": {
                "sampling_sessions_total": 3,
                "sampling_sessions_inflight": 2,
                "sampling_sessions_lora_loaded": 1,
                "sampling_sessions_by_model": [
                    {
                        "base_model": "Qwen/Qwen3-4B-Instruct-2507",
                        "total": 2,
                        "inflight": 1,
                        "lora_loaded": 1,
                    }
                ],
            },
            "training_sessions": {
                "training_sessions_total": 2,
                "training_sessions_active": 1,
                "training_sessions_inflight": 1,
                "training_sessions_by_model": [
                    {
                        "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                        "backend": "megatron",
                        "total": 2,
                        "active": 1,
                        "inflight": 1,
                    }
                ],
            },
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
                "megatron_session_switch_failures": [
                    {
                        "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                        "reason": "partial_swap",
                        "count": 2,
                    }
                ],
                "megatron_actor_lifecycle": [
                    {
                        "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                        "event": "evicted",
                        "count": 1,
                    },
                    {
                        "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                        "event": "startup_timeout",
                        "count": 1,
                    }
                ],
                "vllm_workload": [
                    {
                        "actor_name": "vllm-1",
                        "base_model": "Qwen/Qwen3-4B-Instruct-2507",
                        "op": "asample",
                        "status": "ok",
                        "requests_total": 8,
                        "prompt_tokens_total": 4096,
                        "generated_tokens_total": 512,
                        "duration_s_total": 24.0,
                        "duration_s_max": 5.5,
                        "ttft_s_total": 8.0,
                        "ttft_s_max": 1.5,
                        "ttft_s_count": 8,
                        "tpot_s_total": 0.96,
                        "tpot_s_max": 0.2,
                        "tpot_s_count": 8,
                    }
                ],
                "vllm_active_requests": [
                    {
                        "actor_name": "vllm-1",
                        "base_model": "Qwen/Qwen3-4B-Instruct-2507",
                        "op": "asample",
                        "active_requests": 2,
                    }
                ],
                "vllm_actor_latency": [
                    {
                        "actor_name": "vllm-1",
                        "base_model": "Qwen/Qwen3-4B-Instruct-2507",
                        "op": "asample",
                        "status": "ok",
                        "count": 8,
                        "duration_s_total": 24.0,
                        "duration_s_max": 5.5,
                    }
                ],
                "training_operation_latency": [
                    {
                        "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                        "backend": "megatron",
                        "op": "forward_backward",
                        "status": "ok",
                        "count": 4,
                        "duration_s_total": 18.0,
                        "duration_s_max": 6.0,
                    }
                ],
            },
        },
    }


async def _fake_admission_stats_with_cached_rss(*, include_actor_rss: bool = True) -> dict:
    stats = await _fake_admission_stats(include_actor_rss=include_actor_rss)
    rec = stats["actors"]["resource_pool"][0]
    rec["rss_bytes"] = 4096
    rec["rss_cache_state"] = "fresh"
    rec["rss_sample_age_s"] = 3.0
    rec["rss_sample_source"] = "register"
    return stats


async def _fake_admission_stats_with_stale_cached_rss(*, include_actor_rss: bool = True) -> dict:
    stats = await _fake_admission_stats(include_actor_rss=include_actor_rss)
    rec = stats["actors"]["resource_pool"][0]
    rec.pop("rss_bytes", None)
    rec["rss_cache_state"] = "stale"
    rec["rss_sample_age_s"] = 120.0
    rec["rss_sample_source"] = "touch"
    return stats


def test_issue_248_internal_metrics_omits_unknown_resource_pool_rss(monkeypatch) -> None:
    monkeypatch.setattr(internal_routes, "admission_stats", _fake_admission_stats)
    resp = asyncio.run(internal_routes.metrics())
    text = resp.body.decode("utf-8")

    expected_lines = (
        "mint_capacity_capacity 16",
        "mint_capacity_inflight 3",
        "mint_work_queue_depth 2",
        'mint_work_queue_depth{executor="sampling.asample"} 1',
        "mint_work_queue_oldest_queued_s 12.5",
        "mint_work_queue_scheduler_arbitration_total 9",
        'mint_work_queue_scheduler_arbitration_total{winner_bucket="scheduled"} 5',
        'mint_work_queue_scheduler_domain_dequeue_total{backend="vllm",execution_scope="local",op="sampling.asample",reason="starvation",scheduler_domain="vllm:Qwen/Qwen3-4B-Instruct-2507::replica::0"} 6',
        'mint_work_queue_legacy_dequeue_total{execution_scope="local",op="sampling.asample",reason="fifo"} 4',
        'mint_work_queue_scheduler_domain_pending_requests{backend="vllm",execution_scope="local",scheduler_domain="vllm:Qwen/Qwen3-4B-Instruct-2507::replica::0"} 3',
        'mint_work_queue_scheduler_domain_inflight_workers{backend="vllm",execution_scope="local",scheduler_domain="vllm:Qwen/Qwen3-4B-Instruct-2507::replica::0"} 1',
        'mint_work_queue_scheduler_domain_capacity_workers{backend="vllm",execution_scope="local",scheduler_domain="vllm:Qwen/Qwen3-4B-Instruct-2507::replica::0"} 2',
        'mint_work_queue_scheduler_domain_admissible{backend="vllm",execution_scope="local",scheduler_domain="vllm:Qwen/Qwen3-4B-Instruct-2507::replica::0"} 1',
        'mint_work_queue_scheduler_domain_dequeue_picks_total{backend="vllm",execution_scope="local",scheduler_domain="vllm:Qwen/Qwen3-4B-Instruct-2507::replica::0"} 6',
        "mint_future_store_pending 1",
        'mint_future_store_pending{op="asample"} 1',
        "mint_future_store_oldest_pending_s 8",
        "mint_future_store_result_refs_count 4",
        'mint_resource_pool_actor_idle_time_s{actor_name="vllm-1",actor_type="vllm",model="Qwen/Qwen3-4B-Instruct-2507"} 2',
        'mint_resource_pool_actor_age_s{actor_name="vllm-1",actor_type="vllm",model="Qwen/Qwen3-4B-Instruct-2507"} 50',
        'mint_resource_pool_actors{actor_type="vllm",model="Qwen/Qwen3-4B-Instruct-2507"} 1',
        'mint_resource_pool_actor_rss_cache_state{actor_name="vllm-1",actor_type="vllm",model="Qwen/Qwen3-4B-Instruct-2507",state="unknown"} 1',
        'mint_resource_pool_group_rss_cache_samples{actor_type="vllm",model="Qwen/Qwen3-4B-Instruct-2507",state="unknown"} 1',
        "mint_api_server_process_rss_bytes 12345",
        "mint_metrics_up 1",
    )
    for line in expected_lines:
        assert line in text, f"missing metric line: {line}"

    extra_lines = (
        'mint_model_load_pct{base_model="Qwen/Qwen3-4B-Instruct-2507",workload="sample"} 50',
        'mint_model_pending_requests{base_model="Qwen/Qwen3-4B-Instruct-2507",workload="sample"} 3',
        'mint_sampling_sessions_total 3',
        'mint_sampling_sessions_by_model{base_model="Qwen/Qwen3-4B-Instruct-2507"} 2',
        'mint_training_sessions_total 2',
        'mint_training_sessions_by_model{backend="megatron",base_model="Qwen/Qwen3-30B-A3B-Instruct-2507"} 2',
        'mint_vllm_workload_requests_total{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507",op="asample",status="ok"} 8',
        'mint_vllm_workload_ttft_s_sum{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507",op="asample",status="ok"} 8',
        'mint_vllm_workload_tpot_s_sum{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507",op="asample",status="ok"} 0.96',
        'mint_vllm_workload_active_requests{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507",op="asample"} 2',
        'mint_resource_pool_actor_gpu_binding{actor_name="vllm-1",gpu_index="0",hostname="host-a",workload="sample"} 1',
        'mint_vllm_scheduler_waiting_requests{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 4',
        'mint_vllm_scheduler_running_requests{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 2',
        'mint_vllm_scheduler_kv_cache_usage_ratio{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 0.75',
        'mint_vllm_prefix_cache_queries_total{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 100',
        'mint_vllm_prefix_cache_hits_total{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 60',
        'mint_vllm_prefix_cache_hit_ratio{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 0.6',
        'mint_vllm_preemptions_total{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 3',
        'mint_vllm_queue_time_s_sum{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 12',
        'mint_vllm_queue_time_s_p50_recent{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 1.4',
        'mint_vllm_queue_time_s_p95_recent{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 3.2',
        'mint_vllm_prefill_time_s_sum{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 16',
        'mint_vllm_decode_time_s_sum{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 40',
        'mint_vllm_time_per_output_token_s_sum{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 0.96',
        'mint_vllm_scheduled_tokens_iter_sum{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 512',
        'mint_vllm_scheduled_new_requests_iter_p95_recent{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 4.8',
        'mint_vllm_prefill_requests_iter_sum{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 20',
        'mint_vllm_decode_requests_iter_sum{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 44',
        'mint_vllm_prompt_tokens_iter_p50_recent{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 512',
        'mint_vllm_generation_tokens_iter_max{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 64',
        'mint_vllm_time_to_first_token_s_sum{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 9.6',
        'mint_vllm_inter_token_latency_s_p95_recent{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 0.15',
        'mint_vllm_executor_execute_model_s_sum{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 3.2',
        'mint_vllm_worker_execute_model_s_p50_recent{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 0.31',
        'mint_vllm_seq_slot_wait_s_sum{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 5',
        'mint_vllm_seq_slot_wait_s_p95_recent{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 1.2',
        'mint_vllm_generate_lock_wait_s_sum{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 1.2',
        'mint_vllm_generate_lock_wait_s_p95_recent{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 0.35',
        'mint_vllm_engine_read_lock_wait_s_sum{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 0.8',
        'mint_vllm_engine_read_lock_wait_s_p95_recent{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 0.28',
        'mint_vllm_add_request_wait_s_sum{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 2.4',
        'mint_vllm_add_request_exec_s_p50_recent{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 0.07',
        'mint_vllm_first_token_observed_s_max{actor_name="vllm-1",base_model="Qwen/Qwen3-4B-Instruct-2507"} 1.9',
        'mint_megatron_active_sessions{actor_name="megatron-1",base_model="Qwen/Qwen3-30B-A3B-Instruct-2507"} 1',
        'mint_megatron_session_unknown{actor_name="megatron-1",base_model="Qwen/Qwen3-30B-A3B-Instruct-2507"} 0',
        'mint_megatron_session_step{actor_name="megatron-1",base_model="Qwen/Qwen3-30B-A3B-Instruct-2507"} 17',
        'mint_megatron_learning_rate{actor_name="megatron-1",base_model="Qwen/Qwen3-30B-A3B-Instruct-2507"} 5e-05',
        'mint_megatron_gpu_memory_allocated_bytes{actor_name="megatron-1",base_model="Qwen/Qwen3-30B-A3B-Instruct-2507"} 48000000000',
        'mint_megatron_gpu_memory_reserved_bytes{actor_name="megatron-1",base_model="Qwen/Qwen3-30B-A3B-Instruct-2507"} 52000000000',
        'mint_megatron_gpu_memory_fragmentation_bytes{actor_name="megatron-1",base_model="Qwen/Qwen3-30B-A3B-Instruct-2507"} 4000000000',
        'mint_resource_pool_actor_gpu_binding{actor_name="megatron-1",gpu_index="0",hostname="host-b",workload="train"} 1',
        'mint_resource_pool_actor_gpu_binding{actor_name="megatron-1",gpu_index="1",hostname="host-b",workload="train"} 1',
        'mint_resource_pool_observability_cache_hits_total{actor_type="megatron"} 9',
        'mint_resource_pool_observability_cache_stale_total{actor_type="megatron"} 3',
        'mint_resource_pool_observability_refresh_success_total{actor_type="megatron"} 2',
        'mint_resource_pool_observability_refresh_failures_total{actor_type="megatron"} 1',
        'mint_resource_pool_observability_cache_hits_total{actor_type="vllm"} 12',
        'mint_resource_pool_observability_refresh_success_total{actor_type="vllm"} 3',
        'mint_megatron_session_switch_failures_total{base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",reason="partial_swap"} 2',
        'mint_megatron_actor_lifecycle_events_total{base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",event="evicted"} 1',
        'mint_megatron_actor_lifecycle_events_total{base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",event="startup_timeout"} 1',
    )
    for line in extra_lines:
        assert line in text, f"missing metric line: {line}"

    assert 'mint_actor_rss_bytes{actor="future_store"}' not in text
    assert 'mint_resource_pool_actor_rss_bytes{actor_name="vllm-1"' not in text
    assert 'mint_resource_pool_group_rss_bytes{actor_type="vllm",model="Qwen/Qwen3-4B-Instruct-2507"}' not in text


def test_issue_248_internal_metrics_emits_group_rss_when_cached_sample_exists(monkeypatch) -> None:
    monkeypatch.setattr(internal_routes, "admission_stats", _fake_admission_stats_with_cached_rss)
    resp = asyncio.run(internal_routes.metrics())
    text = resp.body.decode("utf-8")

    assert (
        'mint_resource_pool_group_rss_bytes{actor_type="vllm",model="Qwen/Qwen3-4B-Instruct-2507"} 4096'
        in text
    )
    assert (
        'mint_resource_pool_actor_rss_bytes{actor_name="vllm-1",actor_type="vllm",model="Qwen/Qwen3-4B-Instruct-2507"} 4096'
        in text
    )
    assert (
        'mint_resource_pool_actor_rss_cache_state{actor_name="vllm-1",actor_type="vllm",model="Qwen/Qwen3-4B-Instruct-2507",state="fresh"} 1'
        in text
    )
    assert (
        'mint_resource_pool_group_rss_cache_samples{actor_type="vllm",model="Qwen/Qwen3-4B-Instruct-2507",state="fresh"} 1'
        in text
    )


def test_issue_248_internal_metrics_marks_stale_cached_rss_without_emitting_value(monkeypatch) -> None:
    monkeypatch.setattr(internal_routes, "admission_stats", _fake_admission_stats_with_stale_cached_rss)
    resp = asyncio.run(internal_routes.metrics())
    text = resp.body.decode("utf-8")

    assert (
        'mint_resource_pool_actor_rss_cache_state{actor_name="vllm-1",actor_type="vllm",model="Qwen/Qwen3-4B-Instruct-2507",state="stale"} 1'
        in text
    )
    assert (
        'mint_resource_pool_group_rss_cache_samples{actor_type="vllm",model="Qwen/Qwen3-4B-Instruct-2507",state="stale"} 1'
        in text
    )
    assert (
        'mint_resource_pool_actor_rss_sample_age_s{actor_name="vllm-1",actor_type="vllm",model="Qwen/Qwen3-4B-Instruct-2507"} 120'
        in text
    )
    assert 'mint_resource_pool_actor_rss_bytes{actor_name="vllm-1"' not in text
    assert 'mint_resource_pool_group_rss_bytes{actor_type="vllm",model="Qwen/Qwen3-4B-Instruct-2507"}' not in text


def test_issue_248_admission_stats_metrics_path_uses_cached_pool_snapshot(monkeypatch) -> None:
    import importlib

    api_work_queue_module = importlib.import_module("tinker_server.backend.api_work_queue")
    capacity_manager_module = importlib.import_module("tinker_server.backend.capacity_manager")
    future_store_module = importlib.import_module("tinker_server.backend.future_store")
    resource_pool_module = importlib.import_module("tinker_server.backend.resource_pool")

    @dataclass
    class _CapSnapshot:
        capacity: int
        inflight: int

    class _FakeCapacityManager:
        async def async_snapshot(self, *, timeout_s: float = 10.0) -> _CapSnapshot:
            return _CapSnapshot(capacity=16, inflight=1)

    class _FakeApiWorkQueue:
        def metrics_snapshot(self) -> dict:
            return {"depth": 0, "enqueued": 0, "dequeued": 0}

        async def stats(self, *, timeout_s: float = 10.0) -> dict:
            raise AssertionError("metrics scrape must not call api_work_queue.stats")

    class _FakeFutureStore:
        def metrics_snapshot(self) -> dict:
            return {"pending": 0, "results": 0, "errors": 0}

        def ensure_ready(self, *, timeout_s: float = 10.0) -> dict:
            raise AssertionError("metrics scrape must not call future_store.ensure_ready")

    calls = {"cached_snapshot": 0}

    class _FakePool:
        def cached_snapshot(self) -> list[dict]:
            calls["cached_snapshot"] += 1
            return [{"actor_name": "a", "actor_type": "vllm", "base_model": "m", "idle_time": 1, "age": 2}]

        def metadata_cache_metrics_snapshot(self) -> list[dict]:
            return [{"actor_type": "vllm", "cache_hits_total": 1, "cache_stale_total": 0, "refresh_success_total": 0, "refresh_failures_total": 0}]

        def lifecycle_metrics_snapshot(self) -> list[dict]:
            return []

        def rss_snapshot(self, *, timeout_s: float = 10.0) -> list[dict]:
            raise AssertionError("metrics scrape must not call resource_pool.rss_snapshot")

    monkeypatch.setattr(capacity_manager_module, "capacity_manager", _FakeCapacityManager())
    monkeypatch.setattr(api_work_queue_module, "api_work_queue", _FakeApiWorkQueue())
    monkeypatch.setattr(future_store_module, "future_store", _FakeFutureStore())
    monkeypatch.setattr(resource_pool_module, "get_resource_pool", lambda: _FakePool())

    stats = asyncio.run(internal_routes.admission_stats(include_actor_rss=False))

    assert calls["cached_snapshot"] == 1
    assert isinstance(stats.get("actors", {}).get("resource_pool"), list)


def test_issue_248_metrics_path_exports_cached_scheduler_model_load(monkeypatch) -> None:
    api_work_queue_module = importlib.import_module("tinker_server.backend.api_work_queue")
    capacity_manager_module = importlib.import_module("tinker_server.backend.capacity_manager")
    future_store_module = importlib.import_module("tinker_server.backend.future_store")
    resource_pool_module = importlib.import_module("tinker_server.backend.resource_pool")

    @dataclass
    class _CapSnapshot:
        capacity: int
        inflight: int

    class _FakeCapacityManager:
        async def async_snapshot(self, *, timeout_s: float = 10.0) -> _CapSnapshot:
            return _CapSnapshot(capacity=16, inflight=1)

    class _FakeApiWorkQueue:
        def metrics_snapshot(self) -> dict:
            return {
                "depth": 5,
                "depth_legacy": 3,
                "depth_scheduled": 2,
                "enqueued": 9,
                "dequeued": 4,
                "scheduler_metrics_ready": True,
                "scheduler_enabled": True,
                "scheduler_domains_total": 1,
                "scheduler_domains": {
                    "vllm:Qwen/Qwen3-4B-Instruct-2507::replica::0": {
                        "backend": "vllm",
                        "pending_requests": 3,
                        "active_sessions": 2,
                        "oldest_queued_s": 11.0,
                        "inflight_workers": 1,
                        "capacity_workers": 2,
                        "admissible": True,
                        "service_gap_s": 5.5,
                        "stats": {"picks": 6, "starvation_picks": 1},
                    }
                },
            }

        async def stats(self, *, timeout_s: float = 10.0) -> dict:
            raise AssertionError("metrics scrape must not call api_work_queue.stats")

    class _FakeFutureStore:
        def metrics_snapshot(self) -> dict:
            return {"pending": 0, "results": 0, "errors": 0}

        def ensure_ready(self, *, timeout_s: float = 10.0) -> dict:
            raise AssertionError("metrics scrape must not call future_store.ensure_ready")

    class _FakePool:
        def cached_snapshot(self) -> list[dict]:
            return []

        def metadata_cache_metrics_snapshot(self) -> list[dict]:
            return []

        def lifecycle_metrics_snapshot(self) -> list[dict]:
            return []

        def rss_snapshot(self, *, timeout_s: float = 10.0) -> list[dict]:
            raise AssertionError("metrics scrape must not call resource_pool.rss_snapshot")

    monkeypatch.setattr(capacity_manager_module, "capacity_manager", _FakeCapacityManager())
    monkeypatch.setattr(api_work_queue_module, "api_work_queue", _FakeApiWorkQueue())
    monkeypatch.setattr(future_store_module, "future_store", _FakeFutureStore())
    monkeypatch.setattr(resource_pool_module, "get_resource_pool", lambda: _FakePool())

    resp = asyncio.run(internal_routes.metrics())
    text = resp.body.decode("utf-8")

    assert "mint_work_queue_depth_scheduled 2" in text
    assert "mint_work_queue_scheduler_enabled 1" in text
    assert "mint_work_queue_scheduler_domains_total 1" in text
    assert (
        'mint_work_queue_scheduler_domain_pending_requests{backend="vllm",execution_scope="local",scheduler_domain="vllm:Qwen/Qwen3-4B-Instruct-2507::replica::0"} 3'
        in text
    )
    assert (
        'mint_model_load_pct{base_model="Qwen/Qwen3-4B-Instruct-2507",workload="sample"} 50'
        in text
    )
    assert (
        'mint_model_pending_requests{base_model="Qwen/Qwen3-4B-Instruct-2507",workload="sample"} 3'
        in text
    )


def test_issue_248_scheduler_decisions_debug_route_proxies_filters(monkeypatch) -> None:
    api_work_queue_module = importlib.import_module("tinker_server.backend.api_work_queue")
    captured: dict[str, object] = {}

    class _FakeApiWorkQueue:
        async def scheduler_decisions(self, **kwargs) -> dict:
            captured.update(kwargs)
            return {
                "actor_name": "tinker_api_work_queue",
                "last_seq": 12,
                "items": [],
                "scheduler": {"enabled": True},
            }

    monkeypatch.setattr(api_work_queue_module, "api_work_queue", _FakeApiWorkQueue())

    payload = asyncio.run(
        internal_routes.scheduler_decisions_debug(
            limit=25,
            scheduler_domain="vllm:Qwen/Qwen3-4B-Instruct-2507::replica::0",
            reason="sticky",
            since_seq=7,
        )
    )

    assert captured == {
        "limit": 25,
        "scheduler_domain": "vllm:Qwen/Qwen3-4B-Instruct-2507::replica::0",
        "reason": "sticky",
        "since_seq": 7,
        "timeout_s": 10.0,
    }
    assert payload["last_seq"] == 12


def test_issue_248_internal_metrics_exports_ray_control_plane_cache_timestamps(monkeypatch) -> None:
    async def _fake_with_ray(*, include_actor_rss: bool = True) -> dict:
        stats = await _fake_admission_stats(include_actor_rss=include_actor_rss)
        stats["ray_cluster"] = {
            "up": True,
            "cache_age_s": 7.0,
            "last_success_unixtime": 1700000000.0,
            "last_success_age_s": 12.0,
            "warning_count": 0,
            "probe_error_count": 0,
            "slow_probe_count": 0,
            "total_probe_latency_ms": 15.0,
            "nodes": {"alive": 2, "dead": 0, "dead_missing_heartbeats": 0},
            "resources": {"cpu_total": 8, "cpu_available": 4, "gpu_total": 2, "gpu_available": 1},
            "placement_groups": {"total": 1, "created": 1, "removed": 0, "pending": 0, "pending_gpu": 0},
            "named_actors": {"total": 5, "namespace": 5},
            "probes": {"nodes": {"ok": True, "latency_ms": 1.5}},
        }
        stats["ray_gcs_metrics"] = {
            "up": True,
            "cache_age_s": 5.0,
            "last_success_unixtime": 1700000005.0,
            "last_success_age_s": 9.0,
            "scrape_error_count": 0,
            "sample_count": 1,
            "scrape_latency_ms": 20.0,
            "derived": {"gcs_task_manager_task_events_drop_ratio": 0.1},
            "samples": [
                {"name": "gcs_actors_count", "labels": {"State": "ALIVE"}, "value": 6.0},
            ],
        }
        return stats

    monkeypatch.setattr(internal_routes, "admission_stats", _fake_with_ray)

    resp = asyncio.run(internal_routes.metrics())
    text = resp.body.decode("utf-8")

    assert "mint_ray_cluster_last_success_unixtime 1700000000" in text
    assert "mint_ray_cluster_last_success_age_s 12" in text
    assert "mint_ray_gcs_metrics_bridge_last_success_unixtime 1700000005" in text
    assert "mint_ray_gcs_metrics_bridge_last_success_age_s 9" in text
    assert 'gcs_actors_count{State="ALIVE"} 6' in text
