from types import SimpleNamespace

from mint_server.routes import sampling


def _request(headers: dict[str, str]):
    return SimpleNamespace(headers=headers)


def test_sampling_backpressure_is_opt_in_by_header(monkeypatch) -> None:
    monkeypatch.setattr(sampling, "_MAX_INFLIGHT_SAMPLE_TASKS", 1)
    monkeypatch.setattr(sampling, "_inflight_sample_tasks", 1)

    assert sampling._should_backpressure(_request({})) is False


def test_sampling_backpressure_rejects_when_opted_in_and_saturated(monkeypatch) -> None:
    monkeypatch.setattr(sampling, "_MAX_INFLIGHT_SAMPLE_TASKS", 1)
    monkeypatch.setattr(sampling, "_inflight_sample_tasks", 1)

    assert sampling._should_backpressure(
        _request({"X-Tinker-Sampling-Backpressure": "1"})
    ) is True


def test_sampling_backpressure_allows_opted_in_when_below_limit(monkeypatch) -> None:
    monkeypatch.setattr(sampling, "_MAX_INFLIGHT_SAMPLE_TASKS", 2)
    monkeypatch.setattr(sampling, "_inflight_sample_tasks", 1)

    assert sampling._should_backpressure(
        _request({"X-Tinker-Sampling-Backpressure": "1"})
    ) is False
