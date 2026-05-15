from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_fast_rl_module():
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts" / "wip"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    sys.modules.setdefault(
        "openpi_libero_sft",
        types.SimpleNamespace(
            _build_transform=lambda *_args, **_kwargs: None,
            _collect_transformed_items=lambda *_args, **_kwargs: None,
            _decode_image=lambda *_args, **_kwargs: None,
            _encode_png_base64=lambda *_args, **_kwargs: None,
            _episode_path=lambda *_args, **_kwargs: None,
            _iter_windows_for_task=lambda *_args, **_kwargs: (),
            _load_tasks=lambda *_args, **_kwargs: [],
            _plot_curve=lambda *_args, **_kwargs: None,
            CONFIG_NAME_BY_BASE_MODEL={},
        ),
    )

    module_name = "_test_issue_567_openpi_libero_fast_rl"
    path = scripts_dir / "openpi_libero_fast_rl.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(self, *, payload: dict[str, object], status_code: int = 200) -> None:
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


def test_issue_567_poll_future_retries_single_request_timeout(monkeypatch) -> None:
    module = _load_fast_rl_module()
    calls = {"count": 0}
    sleeps: list[float] = []

    def _fake_post(_url: str, *, json: dict[str, object], timeout: float, headers: dict[str, str]):
        calls["count"] += 1
        if calls["count"] == 1:
            raise module.requests.Timeout("read timed out")
        assert headers["Connection"] == "close"
        return _FakeResponse(payload={"status": "done", "request_id": json["request_id"]})

    monkeypatch.setattr(module.requests, "post", _fake_post)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = module._poll_future("http://example.test", "req-1", timeout_s=5.0)

    assert result == {"status": "done", "request_id": "req-1"}
    assert calls["count"] == 2
    assert sleeps == [1.0]
