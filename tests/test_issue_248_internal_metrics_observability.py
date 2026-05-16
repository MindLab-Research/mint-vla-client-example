from __future__ import annotations

import asyncio
import importlib
import time
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
                        {"hostname": "host-b", "node_id": "node-b", "gpu_index": 0, "gpu_uuid": "GPU-host-b-0", "rank": 0},
                        {"hostname": "host-b", "node_id": "node-b", "gpu_index": 1, "gpu_uuid": "GPU-host-b-1", "rank": 1},
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
            {
                "actor_type": "dense",
                "base_model": "Qwen/Qwen3-4B-Instruct-2507",
                "actor_name": "peft_trainer_qwen__qwen3_4b_instruct_2507_maxr64",
                "idle_time": 1,
                "age": 33,
                "rss_bytes": 150,
                "node_id": "node-c",
                "metadata": {
                    "hostname": "host-c",
                    "gpu_indices": [2],
                    "poisoned": True,
                    "poisoned_at": time.time() - 12.0,
                    "last_fatal_op": "reinit_lora_weights",
                    "poison_reason": "reinit_lora_weights:CUDA error: device-side assert triggered",
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
        actors.update({"task_state_futures": {"rss_bytes": 3000}})
    else:
        actors["resource_pool"][0].pop("rss_bytes", None)
        actors["resource_pool"][0]["rss_cache_state"] = "unknown"
    return {
        "model_work_scheduler": {
            "depth": 4,
            "backlog_depth": 2,
            "backlog_depth_by_domain": {"vllm:Qwen/Qwen3-4B-Instruct-2507": 2},
            "replica_queues": {
                "vllm:Qwen/Qwen3-4B-Instruct-2507::replica-0": {
                    "domain_key": "vllm:Qwen/Qwen3-4B-Instruct-2507",
                    "replica_id": "replica-0",
                    "depth": 3,
                    "status": "healthy",
                },
                "megatron:Qwen/Qwen3-30B-A3B-Instruct-2507::replica-0": {
                    "domain_key": "megatron:Qwen/Qwen3-30B-A3B-Instruct-2507",
                    "replica_id": "replica-0",
                    "depth": 1,
                    "status": "healthy",
                },
            },
            "leases": [
                {"item": {"domain_key": "vllm:Qwen/Qwen3-4B-Instruct-2507"}},
            ],
            "counters": {
                "appended": 10,
                "assigned": 8,
                "claimed": 6,
                "completed": 5,
                "failed": 1,
                "requeued": 2,
            },
        },
        "task_state_futures": {
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
        "mint_model_work_scheduler_depth 4",
        "mint_model_work_scheduler_backlog_depth 2",
        "mint_model_work_scheduler_appended_total 10",
        "mint_model_work_scheduler_assigned_total 8",
        'mint_model_work_scheduler_domain_backlog_depth{domain_key="vllm:Qwen/Qwen3-4B-Instruct-2507"} 2',
        'mint_model_work_scheduler_replica_queue_depth{domain_key="vllm:Qwen/Qwen3-4B-Instruct-2507",queue_id="vllm:Qwen/Qwen3-4B-Instruct-2507::replica-0",replica_id="replica-0",status="healthy"} 3',
        "mint_model_work_scheduler_leases 1",
        "mint_task_state_futures_pending 1",
        'mint_task_state_futures_pending{op="asample"} 1',
        "mint_task_state_futures_oldest_pending_s 8",
        "mint_task_state_futures_result_refs_count 4",
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
        'mint_model_load_pct{base_model="Qwen/Qwen3-4B-Instruct-2507",workload="sample"} 100',
        'mint_model_pending_requests{base_model="Qwen/Qwen3-4B-Instruct-2507",workload="sample"} 3',
        'mint_model_inflight_workers{base_model="Qwen/Qwen3-4B-Instruct-2507",workload="sample"} 1',
        'mint_model_capacity_workers{base_model="Qwen/Qwen3-4B-Instruct-2507",workload="sample"} 1',
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
        'mint_resource_pool_actor_gpu_binding{actor_name="megatron-1",gpu_index="0",gpu_uuid="GPU-host-b-0",hostname="host-b",workload="train"} 1',
        'mint_resource_pool_actor_gpu_binding{actor_name="megatron-1",gpu_index="1",gpu_uuid="GPU-host-b-1",hostname="host-b",workload="train"} 1',
        'mint_resource_pool_observability_cache_hits_total{actor_type="megatron"} 9',
        'mint_resource_pool_observability_cache_stale_total{actor_type="megatron"} 3',
        'mint_resource_pool_observability_refresh_success_total{actor_type="megatron"} 2',
        'mint_resource_pool_observability_refresh_failures_total{actor_type="megatron"} 1',
        'mint_resource_pool_observability_cache_hits_total{actor_type="vllm"} 12',
        'mint_resource_pool_observability_refresh_success_total{actor_type="vllm"} 3',
        'mint_dense_actor_poisoned{actor_name="peft_trainer_qwen__qwen3_4b_instruct_2507_maxr64",base_model="Qwen/Qwen3-4B-Instruct-2507",last_fatal_op="reinit_lora_weights"} 1',
        'mint_dense_poisoned_actors{base_model="Qwen/Qwen3-4B-Instruct-2507",last_fatal_op="reinit_lora_weights"} 1',
    )
    for line in extra_lines:
        assert line in text, f"missing metric line: {line}"
    assert 'mint_dense_actor_poisoned_age_s{actor_name="peft_trainer_qwen__qwen3_4b_instruct_2507_maxr64",base_model="Qwen/Qwen3-4B-Instruct-2507",last_fatal_op="reinit_lora_weights"}' in text

    assert 'mint_actor_rss_bytes{actor="task_state_futures"}' not in text
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


def test_issue_588_admission_stats_rss_path_preserves_resource_pool_metadata(monkeypatch) -> None:
    task_state_store_module = importlib.import_module("tinker_server.backend.task_state_store")
    model_actor_supervisor_module = importlib.import_module("tinker_server.backend.model_actor_supervisor")
    model_work_scheduler_module = importlib.import_module("tinker_server.backend.model_work_scheduler")
    maintenance_cron_actor_module = importlib.import_module("tinker_server.backend.maintenance_cron_actor")
    resource_pool_module = importlib.import_module("tinker_server.backend.resource_pool")
    session_heartbeat_store_module = importlib.import_module("tinker_server.backend.session_heartbeat_store")
    sampling_route = importlib.import_module("tinker_server.routes.sampling")
    service_route = importlib.import_module("tinker_server.routes.service")
    dense_session_state_module = importlib.import_module("tinker_server.backend.dense_session_state")
    ray_cluster_health_module = importlib.import_module("tinker_server.ray_cluster_health")
    ray_gcs_metrics_module = importlib.import_module("tinker_server.ray_gcs_metrics")

    class _FakeModelWorkScheduler:
        async def stats(self, *, timeout_s: float = 10.0) -> dict:
            return {"depth": 0, "backlog_depth": 0, "replica_queues": {}, "leases": [], "counters": {}}

    class _FakeModelActorSupervisor:
        async def async_snapshot(self, *, timeout_s: float = 10.0) -> dict:
            return {}

    class _FakeSupervisor:
        async def async_health_snapshot(self, *, timeout_s: float = 10.0) -> dict:
            return {}

    class _FakeSessionHeartbeatStore:
        async def async_size(self) -> int:
            return 0

    class _FakeFutureStore:
        async def async_ensure_ready(self, *, timeout_s: float = 10.0) -> dict:
            return {"pending": 0, "results": 0, "errors": 0}

        async def async_rss_bytes(self, *, timeout_s: float = 10.0) -> int:
            return 3000

    class _FakePool:
        def rss_snapshot(self, *, timeout_s: float = 10.0) -> list[dict]:
            return [
                {
                    "actor_name": "vllm-1",
                    "actor_type": "vllm",
                    "base_model": "Qwen/Qwen3-4B-Instruct-2507",
                    "metadata": {
                        "scheduler_waiting_requests": 4,
                        "scheduler_running_requests": 2,
                    },
                    "metadata_sample_source": "cached_snapshot",
                    "metadata_cache_state": "fresh",
                    "rss_bytes": 4096,
                }
            ]

        def cached_snapshot(self) -> list[dict]:
            raise AssertionError("include_actor_rss=True must use rss_snapshot")

        def metadata_cache_metrics_snapshot(self) -> list[dict]:
            return []

        def lifecycle_metrics_snapshot(self) -> list[dict]:
            return []

    monkeypatch.setattr(task_state_store_module, "task_state_futures", _FakeFutureStore())
    monkeypatch.setattr(model_work_scheduler_module, "model_work_scheduler", _FakeModelWorkScheduler())
    monkeypatch.setattr(model_actor_supervisor_module, "model_actor_supervisor", _FakeModelActorSupervisor())
    monkeypatch.setattr(maintenance_cron_actor_module, "maintenance_cron_actor", _FakeSupervisor())
    monkeypatch.setattr(resource_pool_module, "get_resource_pool", lambda: _FakePool())
    monkeypatch.setattr(session_heartbeat_store_module, "session_heartbeat_store", _FakeSessionHeartbeatStore())
    monkeypatch.setattr(sampling_route, "_lora_load_lock_count", lambda: 0)
    monkeypatch.setattr(service_route, "session_manager", None)
    monkeypatch.setattr(dense_session_state_module, "collect_dense_session_state_stats", lambda: {})
    monkeypatch.setattr(ray_cluster_health_module, "get_ray_cluster_health_snapshot", lambda: {})
    monkeypatch.setattr(ray_gcs_metrics_module, "get_ray_gcs_metrics_snapshot", lambda: {})

    stats = asyncio.run(internal_routes.admission_stats(include_actor_rss=True))
    rec = stats["actors"]["resource_pool"][0]

    assert rec["metadata"]["scheduler_waiting_requests"] == 4
    assert rec["metadata"]["scheduler_running_requests"] == 2
    assert rec["metadata_sample_source"] == "cached_snapshot"
    assert rec["metadata_cache_state"] == "fresh"
    assert rec["rss_bytes"] == 4096


def test_issue_248_admission_stats_metrics_path_uses_cached_pool_snapshot(monkeypatch) -> None:
    task_state_store_module = importlib.import_module("tinker_server.backend.task_state_store")
    model_actor_supervisor_module = importlib.import_module("tinker_server.backend.model_actor_supervisor")
    model_work_scheduler_module = importlib.import_module("tinker_server.backend.model_work_scheduler")
    maintenance_cron_actor_module = importlib.import_module("tinker_server.backend.maintenance_cron_actor")
    resource_pool_module = importlib.import_module("tinker_server.backend.resource_pool")
    session_heartbeat_store_module = importlib.import_module("tinker_server.backend.session_heartbeat_store")
    sampling_route = importlib.import_module("tinker_server.routes.sampling")
    service_route = importlib.import_module("tinker_server.routes.service")
    dense_session_state_module = importlib.import_module("tinker_server.backend.dense_session_state")
    ray_cluster_health_module = importlib.import_module("tinker_server.ray_cluster_health")
    ray_gcs_metrics_module = importlib.import_module("tinker_server.ray_gcs_metrics")

    class _FakeModelWorkScheduler:
        async def stats(self, *, timeout_s: float = 10.0) -> dict:
            return {"depth": 0, "backlog_depth": 0, "replica_queues": {}, "leases": [], "counters": {}}

    class _FakeModelActorSupervisor:
        async def async_snapshot(self, *, timeout_s: float = 10.0) -> dict:
            return {}

    class _FakeSupervisor:
        async def async_health_snapshot(self, *, timeout_s: float = 10.0) -> dict:
            return {}

    class _FakeSessionHeartbeatStore:
        async def async_size(self) -> int:
            return 0

    class _FakeFutureStore:
        def metrics_snapshot(self) -> dict:
            return {"pending": 0, "results": 0, "errors": 0}

        def ensure_ready(self, *, timeout_s: float = 10.0) -> dict:
            raise AssertionError("metrics scrape must not call task_state_futures.ensure_ready")

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

    monkeypatch.setattr(task_state_store_module, "task_state_futures", _FakeFutureStore())
    monkeypatch.setattr(model_work_scheduler_module, "model_work_scheduler", _FakeModelWorkScheduler())
    monkeypatch.setattr(model_actor_supervisor_module, "model_actor_supervisor", _FakeModelActorSupervisor())
    monkeypatch.setattr(maintenance_cron_actor_module, "maintenance_cron_actor", _FakeSupervisor())
    monkeypatch.setattr(resource_pool_module, "get_resource_pool", lambda: _FakePool())
    monkeypatch.setattr(session_heartbeat_store_module, "session_heartbeat_store", _FakeSessionHeartbeatStore())
    monkeypatch.setattr(sampling_route, "_lora_load_lock_count", lambda: 0)
    monkeypatch.setattr(service_route, "session_manager", None)
    monkeypatch.setattr(dense_session_state_module, "collect_dense_session_state_stats", lambda: {})
    monkeypatch.setattr(ray_cluster_health_module, "get_ray_cluster_health_snapshot", lambda: {})
    monkeypatch.setattr(ray_gcs_metrics_module, "get_ray_gcs_metrics_snapshot", lambda: {})

    stats = asyncio.run(internal_routes.admission_stats(include_actor_rss=False))

    assert calls["cached_snapshot"] == 1
    assert isinstance(stats.get("actors", {}).get("resource_pool"), list)


def test_issue_248_metrics_path_exports_cached_scheduler_model_load(monkeypatch) -> None:
    async def _stats(*, include_actor_rss: bool = True) -> dict:
        return {
            "model_work_scheduler": {
                "depth": 5,
                "backlog_depth": 2,
                "backlog_depth_by_domain": {"vllm:Qwen/Qwen3-4B-Instruct-2507": 2},
                "replica_queues": {
                    "vllm:Qwen/Qwen3-4B-Instruct-2507::replica-0": {
                        "domain_key": "vllm:Qwen/Qwen3-4B-Instruct-2507",
                        "replica_id": "replica-0",
                        "depth": 3,
                        "status": "healthy",
                    },
                    "vllm:Qwen/Qwen3-4B-Instruct-2507::replica-1": {
                        "domain_key": "vllm:Qwen/Qwen3-4B-Instruct-2507",
                        "replica_id": "replica-1",
                        "depth": 0,
                        "status": "healthy",
                    },
                },
                "leases": [
                    {"item": {"domain_key": "vllm:Qwen/Qwen3-4B-Instruct-2507"}},
                ],
                "counters": {},
            },
            "task_state_futures": {},
            "actors": {"resource_pool": []},
            "process": {},
        }

    monkeypatch.setattr(internal_routes, "admission_stats", _stats)

    resp = asyncio.run(internal_routes.metrics())
    text = resp.body.decode("utf-8")

    assert "mint_model_work_scheduler_depth 5" in text
    assert "mint_model_work_scheduler_backlog_depth 2" in text
    assert (
        'mint_model_load_pct{base_model="Qwen/Qwen3-4B-Instruct-2507",workload="sample"} 50'
        in text
    )
    assert (
        'mint_model_pending_requests{base_model="Qwen/Qwen3-4B-Instruct-2507",workload="sample"} 3'
        in text
    )


def test_issue_248_scheduler_decisions_debug_route_proxies_filters(monkeypatch) -> None:
    model_work_scheduler_module = importlib.import_module("tinker_server.backend.model_work_scheduler")

    class _FakeModelWorkScheduler:
        async def stats(self, *, timeout_s: float = 10.0) -> dict:
            return {
                "depth": 2,
                "backlog_depth_by_domain": {
                    "vllm:Qwen/Qwen3-4B-Instruct-2507": 2,
                    "megatron:Qwen/Qwen3-30B-A3B-Instruct-2507": 1,
                },
                "replica_queues": {
                    "vllm:Qwen/Qwen3-4B-Instruct-2507::replica-0": {
                        "domain_key": "vllm:Qwen/Qwen3-4B-Instruct-2507",
                        "depth": 2,
                    },
                    "megatron:Qwen/Qwen3-30B-A3B-Instruct-2507::replica-0": {
                        "domain_key": "megatron:Qwen/Qwen3-30B-A3B-Instruct-2507",
                        "depth": 1,
                    },
                },
                "leases": [
                    {"item": {"domain_key": "vllm:Qwen/Qwen3-4B-Instruct-2507"}},
                    {"item": {"domain_key": "megatron:Qwen/Qwen3-30B-A3B-Instruct-2507"}},
                ],
            }

    monkeypatch.setattr(model_work_scheduler_module, "model_work_scheduler", _FakeModelWorkScheduler())

    payload = asyncio.run(
        internal_routes.scheduler_decisions_debug(
            limit=25,
            scheduler_domain="vllm:Qwen/Qwen3-4B-Instruct-2507",
            reason="sticky",
            since_seq=7,
        )
    )

    assert payload["decision_log_removed"] is True
    assert list(payload["backlog_depth_by_domain"]) == ["vllm:Qwen/Qwen3-4B-Instruct-2507"]
    assert list(payload["replica_queues"]) == ["vllm:Qwen/Qwen3-4B-Instruct-2507::replica-0"]
    assert payload["leases"] == [{"item": {"domain_key": "vllm:Qwen/Qwen3-4B-Instruct-2507"}}]


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
