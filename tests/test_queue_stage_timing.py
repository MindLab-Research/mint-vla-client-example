from mint_server.backend.queue_stage_timing import (
    attach_queue_stage_timing,
    build_queue_stage_timing,
)


def test_queue_stage_timing_normalizes_stable_buckets() -> None:
    timing = build_queue_stage_timing(
        {
            "queued_at": 10.0,
            "dequeue_at": 15.0,
            "executor_started_at": 17.0,
            "lora_load_s": 2.5,
            "generate_s": 4.0,
            "executor_done_at": 30.0,
            "done_at": 33.0,
        }
    )

    assert timing["schema_version"] == 1
    assert timing["scheduler_wait_s"] == 5.0
    assert timing["executor_wait_s"] == 2.0
    assert timing["lora_s"] == 2.5
    assert timing["vllm_generate_s"] == 4.0
    assert timing["finalization_s"] == 3.0
    assert timing["total_observed_s"] == 23.0


def test_queue_stage_timing_uses_now_for_pending_queued_request() -> None:
    timing = build_queue_stage_timing(
        {
            "queue_state": "queued",
            "queued_at": 10.0,
        },
        now=13.25,
    )

    assert timing["scheduler_wait_s"] == 3.25
    assert timing["executor_wait_s"] is None


def test_attach_queue_stage_timing_only_changes_dict_payloads() -> None:
    timing = {"schema_version": 1, "scheduler_wait_s": 1.0}

    assert attach_queue_stage_timing({"ok": True}, timing) == {
        "ok": True,
        "queue_stage_timing": timing,
    }
    assert attach_queue_stage_timing(["not", "dict"], timing) == ["not", "dict"]
