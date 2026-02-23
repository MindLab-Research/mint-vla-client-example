"""Admission control and trainer queuing tests (Phase 3).

These tests aim to catch regressions in:
1) Admission control: overload returns 429 without leaking reserved capacity.
2) Trainer multitenancy: competing requests wait/queue instead of killing actors due to timeouts.
"""

from __future__ import annotations

import concurrent.futures

import requests

from .conftest import (
    BASE_URL,
    DEFAULT_POLL_TIMEOUT_S,
    DENSE_MODEL,
    create_session,
    get_admission_stats,
    get_headers,
    list_actors,
    make_sft_datum,
    poll_future,
    sample,
)


def _asample_raw(*, payload: dict, timeout_s: float = 30.0) -> requests.Response:
    url = f"{BASE_URL}/api/v1/asample"
    return requests.post(url, json=payload, headers=get_headers(), timeout=timeout_s)


def _submit_forward_backward(*, model_id: str, data: list[dict]) -> str:
    url = f"{BASE_URL}/api/v1/forward_backward"
    payload = {
        "model_id": model_id,
        "forward_backward_input": {"data": data, "loss_fn": "cross_entropy"},
    }
    resp = requests.post(url, json=payload, headers=get_headers(), timeout=120)
    resp.raise_for_status()
    request_id = resp.json().get("request_id")
    assert isinstance(request_id, str) and request_id, "forward_backward missing request_id"
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


class TestAdmissionControl:
    def test_flood_rejected_429_no_capacity_leak_and_retry_works(self, tokenizer):
        """Flood oversized sampling requests and ensure server stays healthy.

        - Oversized requests must be rejected with 429 (admission control).
        - Capacity reservations must not increase (bounded memory signal).
        - After flood, a normal request must still succeed.
        """

        cap_before = get_admission_stats().get("capacity") or {}
        if not isinstance(cap_before, dict) or "error" in cap_before:
            raise AssertionError(f"capacity snapshot unavailable: {cap_before!r}")
        queue_before = int(cap_before.get("queue_bytes_reserved", -1))
        obj_before = int(cap_before.get("object_store_bytes_reserved", -1))
        if queue_before < 0 or obj_before < 0:
            raise AssertionError(f"invalid capacity snapshot: {cap_before!r}")

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

        cap_after = get_admission_stats().get("capacity") or {}
        if not isinstance(cap_after, dict) or "error" in cap_after:
            raise AssertionError(f"capacity snapshot unavailable after flood: {cap_after!r}")
        queue_after = int(cap_after.get("queue_bytes_reserved", -1))
        obj_after = int(cap_after.get("object_store_bytes_reserved", -1))
        assert queue_after == queue_before, f"queue_bytes_reserved leaked: {queue_before} -> {queue_after}"
        assert obj_after == obj_before, f"object_store_bytes_reserved leaked: {obj_before} -> {obj_after}"

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
    def test_competing_requests_wait_dense_trainer(self, dense_session, tokenizer):
        _session_id, model_id = dense_session

        actors0 = list_actors().get("actors", [])
        trainer0 = _find_trainer_actor(
            actors_payload={"actors": actors0}, backend="peft", current_session=model_id
        )
        assert trainer0 is not None, "dense trainer actor not found"
        actor_name0 = trainer0.get("actor_name")
        age0 = float(trainer0.get("age", 0.0) or 0.0)

        prompt_tokens = tokenizer.encode("Translate to Pig Latin: hello", add_special_tokens=True)
        datum = make_sft_datum(
            input_tokens=prompt_tokens,
            target_tokens=prompt_tokens,
            loss_mask=[1] * len(prompt_tokens),
        )
        data = [datum]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            request_ids = list(
                ex.map(lambda _: _submit_forward_backward(model_id=model_id, data=data), range(4))
            )

        assert len(set(request_ids)) == len(request_ids), "expected unique request_ids"

        results = [poll_future(rid, timeout=DEFAULT_POLL_TIMEOUT_S) for rid in request_ids]
        for r in results:
            assert "error" not in r, f"forward_backward failed: {r.get('error')}"

        actors1 = list_actors().get("actors", [])
        trainer1 = _find_trainer_actor(
            actors_payload={"actors": actors1}, backend="peft", current_session=model_id
        )
        assert trainer1 is not None, "dense trainer actor missing after competing requests"
        assert trainer1.get("actor_name") == actor_name0, "dense trainer actor_name changed"
        age1 = float(trainer1.get("age", 0.0) or 0.0)
        assert age1 >= age0, "dense trainer age decreased (actor likely restarted)"

    def test_competing_requests_wait_megatron_trainer(self, moe_session, moe_tokenizer):
        _session_id, model_id = moe_session

        actors0 = list_actors().get("actors", [])
        trainer0 = _find_trainer_actor(
            actors_payload={"actors": actors0}, backend="megatron", current_session=model_id
        )
        assert trainer0 is not None, "megatron trainer actor not found"
        actor_name0 = trainer0.get("actor_name")
        age0 = float(trainer0.get("age", 0.0) or 0.0)

        prompt_tokens = moe_tokenizer.encode("2+2=", add_special_tokens=True)
        datum = make_sft_datum(
            input_tokens=prompt_tokens,
            target_tokens=prompt_tokens,
            loss_mask=[1] * len(prompt_tokens),
        )
        data = [datum]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            request_ids = list(
                ex.map(lambda _: _submit_forward_backward(model_id=model_id, data=data), range(4))
            )

        assert len(set(request_ids)) == len(request_ids), "expected unique request_ids"

        results = [poll_future(rid, timeout=DEFAULT_POLL_TIMEOUT_S) for rid in request_ids]
        for r in results:
            assert "error" not in r, f"forward_backward failed: {r.get('error')}"

        actors1 = list_actors().get("actors", [])
        trainer1 = _find_trainer_actor(
            actors_payload={"actors": actors1}, backend="megatron", current_session=model_id
        )
        assert trainer1 is not None, "megatron trainer actor missing after competing requests"
        assert trainer1.get("actor_name") == actor_name0, "megatron trainer actor_name changed"
        age1 = float(trainer1.get("age", 0.0) or 0.0)
        assert age1 >= age0, "megatron trainer age decreased (actor likely restarted)"
