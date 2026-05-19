import asyncio
from types import SimpleNamespace

import anyio

from mint_server.routes import sampling as sampling_route


class _StubCoalesceEngine:
    def __init__(self):
        self.generate_calls: list[dict] = []
        self.generate_many_calls: list[dict] = []

    async def generate(self, **kwargs):
        self.generate_calls.append(dict(kwargs))
        await asyncio.sleep(0)
        return SimpleNamespace(token_ids=[101, 102], logprobs=[-0.1, -0.2], stop_reason="stop")

    async def generate_many(self, **kwargs):
        self.generate_many_calls.append(dict(kwargs))
        await asyncio.sleep(0)
        num_samples = int(kwargs["num_samples"])
        return [
            SimpleNamespace(
                token_ids=[200 + idx],
                logprobs=[-0.3],
                stop_reason="length",
            )
            for idx in range(num_samples)
        ]


def _reset_coalesce_state(monkeypatch) -> None:
    monkeypatch.setattr(sampling_route, "_sample_coalesce_lock", asyncio.Lock())
    monkeypatch.setattr(sampling_route, "_sample_coalesce_groups", {})
    monkeypatch.setattr(sampling_route, "_coalesced_abort_aliases_guard", asyncio.Lock())
    monkeypatch.setattr(sampling_route, "_coalesced_abort_aliases", {})
    monkeypatch.setattr(sampling_route, "_SAMPLE_COALESCE_WINDOW_MS", 0.0)
    monkeypatch.setattr(sampling_route, "_SAMPLE_COALESCE_MAX_BATCH", 8)
    monkeypatch.setattr(sampling_route, "_SAMPLE_COALESCE_MAX_SAMPLES", 8)


def test_coalesced_generate_fans_out_greedy_identical_requests(monkeypatch):
    _reset_coalesce_state(monkeypatch)
    engine = _StubCoalesceEngine()

    async def _run():
        return await asyncio.gather(
            sampling_route._coalesced_generate(
                engine=engine,
                sampling_session_id="sess",
                prompt_ids=[1, 2, 3],
                request_id="req-1",
                num_samples=1,
                max_tokens=8,
                stop=None,
                temperature=0.0,
                top_k=-1,
                top_p=1.0,
            ),
            sampling_route._coalesced_generate(
                engine=engine,
                sampling_session_id="sess",
                prompt_ids=[1, 2, 3],
                request_id="req-2",
                num_samples=1,
                max_tokens=8,
                stop=None,
                temperature=0.0,
                top_k=-1,
                top_p=1.0,
            ),
        )

    first, second = anyio.run(_run)

    assert len(engine.generate_calls) == 1
    assert engine.generate_many_calls == []
    assert engine.generate_calls[0]["request_id"] == "req-1_coalesced"
    assert len(first) == 1
    assert len(second) == 1
    assert first[0].token_ids == [101, 102]
    assert second[0].token_ids == [101, 102]
    assert first[0] is not second[0]
    assert sampling_route._coalesced_abort_aliases == {}


def test_coalesced_generate_keeps_generate_many_for_non_greedy_requests(monkeypatch):
    _reset_coalesce_state(monkeypatch)
    engine = _StubCoalesceEngine()

    async def _run():
        return await asyncio.gather(
            sampling_route._coalesced_generate(
                engine=engine,
                sampling_session_id="sess",
                prompt_ids=[4, 5, 6],
                request_id="req-a",
                num_samples=1,
                max_tokens=8,
                stop=None,
                temperature=0.7,
                top_k=-1,
                top_p=1.0,
            ),
            sampling_route._coalesced_generate(
                engine=engine,
                sampling_session_id="sess",
                prompt_ids=[4, 5, 6],
                request_id="req-b",
                num_samples=1,
                max_tokens=8,
                stop=None,
                temperature=0.7,
                top_k=-1,
                top_p=1.0,
            ),
        )

    first, second = anyio.run(_run)

    assert engine.generate_calls == []
    assert len(engine.generate_many_calls) == 1
    assert engine.generate_many_calls[0]["num_samples"] == 2
    assert [sample.token_ids for sample in first] == [[200]]
    assert [sample.token_ids for sample in second] == [[201]]
