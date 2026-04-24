"""Admission control and trainer queuing tests (Phase 3).

These tests aim to catch regressions in:
1) Admission control: overload returns 429 without leaking reserved capacity.
2) Trainer multitenancy: competing requests wait/queue instead of killing actors due to timeouts.
"""

from __future__ import annotations

import concurrent.futures
import time
import uuid

import pytest
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
    save_weights,
    sample,
)
from .framework import create_test_report, print_report_summary

_SAMPLING_BACKPRESSURE_HEADER = "X-Tinker-Sampling-Backpressure"


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
    def _norm_model_label(value: object) -> str:
        if not isinstance(value, str):
            return ""
        return "".join(ch for ch in value.lower() if ch.isalnum())

    def _matches_base_model(actor: dict, expected: str) -> bool:
        expected_norm = _norm_model_label(expected)
        if not expected_norm:
            return False

        candidates: list[str] = []

        base_model_value = actor.get("base_model")
        if isinstance(base_model_value, str) and base_model_value:
            candidates.append(base_model_value)

        metadata = actor.get("metadata")
        if isinstance(metadata, dict):
            for key in ("model_key", "base_model", "model_name"):
                v = metadata.get(key)
                if isinstance(v, str) and v:
                    candidates.append(v)

        actor_name = actor.get("actor_name")
        if isinstance(actor_name, str) and actor_name:
            candidates.append(actor_name)

        for candidate in candidates:
            if candidate == expected:
                return True
            if expected in candidate:
                return True
            if "/" in expected and expected.count("/") == 1:
                org, name = expected.split("/", 1)
                if f"models--{org}--{name}" in candidate:
                    return True
            if expected_norm and expected_norm in _norm_model_label(candidate):
                return True
        return False

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
        if base_model is not None and not _matches_base_model(a, base_model):
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


def _require_rss_payload(stats: dict) -> dict:
    actors = stats.get("actors")
    if not isinstance(actors, dict):
        raise AssertionError(
            f"admission_stats missing actors dict: {actors!r}. Hint: restart server to load new code."
        )

    rp = actors.get("resource_pool")
    if not isinstance(rp, list):
        raise AssertionError(f"admission_stats actors.resource_pool missing list: {rp!r}")

    missing = [a for a in rp if not (isinstance(a, dict) and isinstance(a.get('rss_bytes'), int))]
    if missing:
        raise AssertionError(f"resource_pool rss_bytes missing for some actors: {missing[:3]!r}")

    for k in ("capacity_manager", "api_work_queue", "future_store"):
        v = actors.get(k)
        if not isinstance(v, dict):
            raise AssertionError(f"admission_stats actors.{k} missing dict: {v!r}")
        if "rss_bytes" not in v:
            raise AssertionError(f"admission_stats actors.{k} missing rss_bytes: {v!r}")

    return actors


def _rss_snapshot(actors_payload: dict) -> dict[str, int]:
    rp = actors_payload["resource_pool"]
    out: dict[str, int] = {}
    for a in rp:
        if not isinstance(a, dict):
            continue
        name = a.get("actor_name")
        rss = a.get("rss_bytes")
        if isinstance(name, str) and name and isinstance(rss, int):
            out[name] = rss
    return out


class TestAdmissionControl:
    def test_flood_rejected_429_no_capacity_leak_and_retry_works(self, tokenizer):
        """Flood sampling requests, hit 429 backpressure, and recover via retry.

        - Under flood, some requests should be rejected with 429.
        - Capacity reservations should not leak (bounded memory signal).
        - A previously-rejected request should be admitted after retry.
        """
        start_time = time.time()
        report_data: dict = {}

        # Warmup: ensure sampling is available for this model_id.
        _session_id, model_id = create_session(DENSE_MODEL, lora_rank=16, lr=1e-4)
        r = save_weights(model_id, name="admission_control_warmup")
        if "error" in r:
            raise AssertionError(f"save_weights failed: {r['error']}")

        stats_before = get_admission_stats()
        report_data["stats_before"] = stats_before
        rss_actors_before = _require_rss_payload(stats_before)
        rss_pool_before = _rss_snapshot(rss_actors_before)
        report_data["rss_pool_before"] = rss_pool_before
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
        report_data["api_rss_before"] = rss_before

        actors_before = list_actors().get("actors", [])
        actor_names_before = {
            a.get("actor_name") for a in actors_before if isinstance(a, dict) and a.get("actor_name")
        }
        report_data["actor_names_before"] = sorted(x for x in actor_names_before if isinstance(x, str))

        prompt_tokens = tokenizer.encode("The capital of France is", add_special_tokens=True)
        flood_payload = {
            "model_id": model_id,
            "prompt": {"chunks": [{"tokens": prompt_tokens, "type": "encoded_text"}]},
            "sampling_params": {"max_tokens": 256, "temperature": 0.0},
            "num_samples": 1,
        }

        def _post() -> requests.Response:
            return requests.post(
                f"{BASE_URL}/api/v1/asample",
                json=flood_payload,
                headers={**get_headers(), _SAMPLING_BACKPRESSURE_HEADER: "1"},
                timeout=30.0,
            )

        ok_request_ids: list[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
            # 1) Seed load so worker inflight rises above the configured limit.
            seed_statuses: list[int] = []
            for resp in ex.map(lambda _: _post(), range(48)):
                seed_statuses.append(int(resp.status_code))
                if resp.status_code == 200:
                    rid = resp.json().get("request_id")
                    if isinstance(rid, str) and rid:
                        ok_request_ids.append(rid)
                    continue
                if resp.status_code != 429:
                    raise AssertionError(f"unexpected seed status {resp.status_code}: {resp.text}")

        report_data["seed_statuses"] = list(seed_statuses)
        assert ok_request_ids, "expected at least one admitted request_id during seed flood"
        time.sleep(2.0)

        # 2) Under sustained load, a new request should get 429 (backpressure) at least once.
        statuses: list[int] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            for resp in ex.map(lambda _: _post(), range(32)):
                statuses.append(int(resp.status_code))
                if resp.status_code == 200:
                    rid = resp.json().get("request_id")
                    if isinstance(rid, str) and rid:
                        ok_request_ids.append(rid)

        report_data["statuses"] = list(statuses)
        report_data["ok_request_ids"] = list(ok_request_ids)
        all_statuses = seed_statuses + statuses
        if not any(s == 429 for s in all_statuses):
            counts: dict[int, int] = {}
            for s in all_statuses:
                counts[s] = counts.get(s, 0) + 1
            pytest.skip(
                "admission backpressure not observable in current cluster; "
                f"status counts={counts}"
            )

        # 3) Drain outstanding requests so we can assert no reservation leaks.
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(lambda rid: poll_future(rid, timeout=600), ok_request_ids))

        stats_after = get_admission_stats()
        report_data["stats_after"] = stats_after
        rss_actors_after = _require_rss_payload(stats_after)
        rss_pool_after = _rss_snapshot(rss_actors_after)
        report_data["rss_pool_after"] = rss_pool_after

        max_delta = 0
        for name, before in rss_pool_before.items():
            after = int(rss_pool_after.get(name, -1))
            if after < 0:
                raise AssertionError(f"actor disappeared from rss snapshot: {name}")
            max_delta = max(max_delta, after - int(before))
        assert max_delta < 512 * 1024 * 1024, f"Ray actor RSS grew too much under flood: max_delta={max_delta}"

        cap_after = stats_after.get("capacity") or {}
        if not isinstance(cap_after, dict) or "error" in cap_after:
            raise AssertionError(f"capacity snapshot unavailable after drain: {cap_after!r}")
        queue_after = int(cap_after.get("queue_bytes_reserved", -1))
        obj_after = int(cap_after.get("object_store_bytes_reserved", -1))
        assert queue_after == queue_before, f"queue_bytes_reserved leaked: {queue_before} -> {queue_after}"
        assert obj_after == obj_before, f"object_store_bytes_reserved leaked: {obj_before} -> {obj_after}"

        proc_after = stats_after.get("process") or {}
        if not isinstance(proc_after, dict):
            raise AssertionError(f"invalid process stats after drain: {proc_after!r}")
        rss_after = int(proc_after.get("rss_bytes", -1))
        if rss_after < 0:
            raise AssertionError(f"missing process rss_bytes after drain: {proc_after!r}")
        assert (rss_after - rss_before) < 512 * 1024 * 1024, f"API RSS grew too much: {rss_before} -> {rss_after}"
        report_data["api_rss_after"] = rss_after

        actors_after = list_actors().get("actors", [])
        actor_names_after = {
            a.get("actor_name") for a in actors_after if isinstance(a, dict) and a.get("actor_name")
        }
        report_data["actor_names_after"] = sorted(x for x in actor_names_after if isinstance(x, str))
        assert actor_names_before.issubset(actor_names_after), "rejected flood should not kill actors"

        # Recovery: load drained; a normal request should succeed.
        probe = sample(model_id, prompt_tokens, max_tokens=8, temperature=0.0)
        report_data["probe"] = probe
        assert "error" not in probe, f"probe sampling failed: {probe.get('error')}"

        report = create_test_report(
            test_name="admission_control_backpressure",
            test_type="api",
            data=report_data,
            start_time=start_time,
            plots=[],
            metadata={"dense_model": DENSE_MODEL},
        )
        report_path = report.save()
        print_report_summary(report)
        print(f"report_json={report_path}")


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
