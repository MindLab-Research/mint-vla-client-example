"""Admission control and trainer queuing tests (Phase 3).

These tests aim to catch regressions in:
1) Admission control: overload returns 429 without leaking reserved capacity.
2) Trainer multitenancy: competing requests wait/queue instead of killing actors due to timeouts.
"""

from __future__ import annotations

import concurrent.futures
import time
import uuid

import requests

from .conftest import (
    BASE_URL,
    DEFAULT_POLL_TIMEOUT_S,
    DENSE_MODEL,
    MOE_MODEL,
    create_session,
    get_admission_stats,
    get_headers,
    list_actors,
    poll_future,
    sample,
)


def _asample_raw(*, payload: dict, timeout_s: float = 30.0) -> requests.Response:
    url = f"{BASE_URL}/api/v1/asample"
    return requests.post(url, json=payload, headers=get_headers(), timeout=timeout_s)


def _submit_create_model(*, base_model: str, lora_rank: int, lr: float) -> str:
    url = f"{BASE_URL}/api/v1/create_model"
    payload = {
        "session_id": f"merge_gate_queue_{uuid.uuid4().hex[:10]}",
        "model_seq_id": 1,
        "base_model": base_model,
        "lora_config": {"rank": int(lora_rank)},
        "learning_rate": float(lr),
    }
    resp = requests.post(url, json=payload, headers=get_headers(), timeout=60)
    resp.raise_for_status()
    request_id = resp.json().get("request_id")
    assert isinstance(request_id, str) and request_id, "create_model missing request_id"
    return request_id


def _find_trainer_actor(
    *,
    actors_payload: dict,
    backend: str,
    current_session: str | None = None,
    base_model: str | None = None,
) -> dict | None:
    actors = actors_payload.get("actors")
    if not isinstance(actors, list):
        return None
    for a in actors:
        if not isinstance(a, dict):
            continue
        if a.get("role") != "trainer":
            continue
        if a.get("backend") != backend:
            continue
        if current_session is not None and a.get("current_session") != current_session:
            continue
        if base_model is not None and a.get("base_model") != base_model:
            continue
        return a
    return None


def _wait_for_trainer_actor(*, backend: str, base_model: str, timeout_s: float) -> dict:
    deadline = time.time() + float(timeout_s)
    last_payload: dict | None = None
    while time.time() < deadline:
        payload = list_actors()
        last_payload = payload
        a = _find_trainer_actor(
            actors_payload=payload,
            backend=backend,
            base_model=base_model,
        )
        if a is not None:
            return a
        time.sleep(2.0)
    raise AssertionError(f"trainer actor not found (backend={backend} base_model={base_model}): {last_payload!r}")


class TestAdmissionControl:
    def test_flood_rejected_429_no_capacity_leak_and_retry_works(self, tokenizer):
        """Flood oversized sampling requests and ensure server stays healthy.

        - Oversized requests must be rejected with 429 (admission control).
        - Capacity reservations must not increase (bounded memory signal).
        - After flood, a normal request must still succeed.
        """

        stats_before = get_admission_stats()
        cap_before = stats_before.get("capacity") or {}
        if not isinstance(cap_before, dict) or "error" in cap_before:
            raise AssertionError(f"capacity snapshot unavailable: {cap_before!r}")
        queue_before = int(cap_before.get("queue_bytes_reserved", -1))
        obj_before = int(cap_before.get("object_store_bytes_reserved", -1))
        if queue_before < 0 or obj_before < 0:
            raise AssertionError(f"invalid capacity snapshot: {cap_before!r}")

        proc_before = stats_before.get("process") or {}
        if not isinstance(proc_before, dict):
            raise AssertionError(f"invalid process stats: {proc_before!r}")
        rss_before = int(proc_before.get("rss_bytes", -1))
        if rss_before < 0:
            raise AssertionError(f"missing process rss_bytes: {proc_before!r}")

        actors_before = list_actors().get("actors", [])
        actor_names_before = {
            a.get("actor_name") for a in actors_before if isinstance(a, dict) and a.get("actor_name")
        }

        # Make the estimated result size absurdly large to force admission rejection (429)
        # without starting any Ray work.
        overload_payload = {
            "model_id": "merge_gate_dummy_model_id",
            "prompt": {"chunks": [{"tokens": [1], "type": "encoded_text"}]},
            "sampling_params": {"max_tokens": 1, "temperature": 0.0},
            "num_samples": 1,
            "include_prompt_logprobs": True,
            "topk_prompt_logprobs": 1_000_000_000_000,
        }

        statuses: list[int] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
            futs = [ex.submit(_asample_raw, payload=overload_payload) for _ in range(64)]
            for f in concurrent.futures.as_completed(futs):
                resp = f.result()
                statuses.append(int(resp.status_code))

        assert statuses, "no responses collected"
        if any(s != 429 for s in statuses):
            counts: dict[int, int] = {}
            for s in statuses:
                counts[s] = counts.get(s, 0) + 1
            raise AssertionError(f"expected all 429; got status counts: {counts}")

        stats_after = get_admission_stats()
        cap_after = stats_after.get("capacity") or {}
        if not isinstance(cap_after, dict) or "error" in cap_after:
            raise AssertionError(f"capacity snapshot unavailable after flood: {cap_after!r}")
        queue_after = int(cap_after.get("queue_bytes_reserved", -1))
        obj_after = int(cap_after.get("object_store_bytes_reserved", -1))
        assert queue_after == queue_before, f"queue_bytes_reserved leaked: {queue_before} -> {queue_after}"
        assert obj_after == obj_before, f"object_store_bytes_reserved leaked: {obj_before} -> {obj_after}"

        proc_after = stats_after.get("process") or {}
        if not isinstance(proc_after, dict):
            raise AssertionError(f"invalid process stats after flood: {proc_after!r}")
        rss_after = int(proc_after.get("rss_bytes", -1))
        if rss_after < 0:
            raise AssertionError(f"missing process rss_bytes after flood: {proc_after!r}")
        assert (rss_after - rss_before) < 512 * 1024 * 1024, f"API RSS grew too much: {rss_before} -> {rss_after}"

        actors_after = list_actors().get("actors", [])
        actor_names_after = {
            a.get("actor_name") for a in actors_after if isinstance(a, dict) and a.get("actor_name")
        }
        assert actor_names_before.issubset(actor_names_after), "rejected flood should not kill actors"

        # Recovery: a normal request (valid session + small sample) must still succeed.
        _session_id, model_id = create_session(DENSE_MODEL, lora_rank=16, lr=1e-4)
        prompt_tokens = tokenizer.encode("The capital of France is", add_special_tokens=True)
        result = sample(model_id, prompt_tokens, max_tokens=5, temperature=0.0)
        assert "error" not in result, f"post-flood sampling failed: {result.get('error')}"


class TestTrainerQueuing:
    def test_competing_create_model_waits_dense_trainer(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            futs = [
                ex.submit(_submit_create_model, base_model=DENSE_MODEL, lora_rank=32, lr=1e-4),
                ex.submit(_submit_create_model, base_model=DENSE_MODEL, lora_rank=32, lr=1e-4),
            ]
            request_ids = [f.result() for f in futs]
        assert len(set(request_ids)) == len(request_ids), "expected unique request_ids"

        trainer0 = _wait_for_trainer_actor(backend="peft", base_model=DENSE_MODEL, timeout_s=240.0)
        actor_name0 = trainer0.get("actor_name")
        assert isinstance(actor_name0, str) and actor_name0, f"invalid actor_name: {trainer0!r}"
        age0 = float(trainer0.get("age", 0.0) or 0.0)

        results = [poll_future(rid, timeout=int(max(DEFAULT_POLL_TIMEOUT_S, 3600))) for rid in request_ids]
        for r in results:
            assert "error" not in r, f"create_model failed: {r.get('error')}"
            mid = r.get("model_id")
            assert isinstance(mid, str) and mid, f"create_model missing model_id: {r!r}"

        trainer1 = _wait_for_trainer_actor(backend="peft", base_model=DENSE_MODEL, timeout_s=60.0)
        assert trainer1.get("actor_name") == actor_name0, "dense trainer actor_name changed"
        age1 = float(trainer1.get("age", 0.0) or 0.0)
        assert age1 >= age0, "dense trainer age decreased (actor likely restarted)"

    def test_competing_create_model_waits_megatron_trainer(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            futs = [
                ex.submit(_submit_create_model, base_model=MOE_MODEL, lora_rank=32, lr=1e-4),
                ex.submit(_submit_create_model, base_model=MOE_MODEL, lora_rank=32, lr=1e-4),
            ]
            request_ids = [f.result() for f in futs]
        assert len(set(request_ids)) == len(request_ids), "expected unique request_ids"

        trainer0 = _wait_for_trainer_actor(backend="megatron", base_model=MOE_MODEL, timeout_s=240.0)
        actor_name0 = trainer0.get("actor_name")
        assert isinstance(actor_name0, str) and actor_name0, f"invalid actor_name: {trainer0!r}"
        age0 = float(trainer0.get("age", 0.0) or 0.0)

        results = [poll_future(rid, timeout=int(max(DEFAULT_POLL_TIMEOUT_S, 7200))) for rid in request_ids]
        for r in results:
            assert "error" not in r, f"create_model failed: {r.get('error')}"
            mid = r.get("model_id")
            assert isinstance(mid, str) and mid, f"create_model missing model_id: {r!r}"

        trainer1 = _wait_for_trainer_actor(backend="megatron", base_model=MOE_MODEL, timeout_s=60.0)
        assert trainer1.get("actor_name") == actor_name0, "megatron trainer actor_name changed"
        age1 = float(trainer1.get("age", 0.0) or 0.0)
        assert age1 >= age0, "megatron trainer age decreased (actor likely restarted)"
