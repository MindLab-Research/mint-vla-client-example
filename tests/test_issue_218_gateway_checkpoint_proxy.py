import asyncio
from pathlib import Path


class _DummyResponse:
    def __init__(self, *, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("response has no json payload")
        return self._payload


class _DummyState:
    def __init__(self, user_data: dict | None) -> None:
        self.user_data = user_data


class _DummyURL:
    def __init__(self, path: str) -> None:
        self.path = path


class _DummyRequest:
    def __init__(self, *, user_data: dict | None = None, headers: dict[str, str] | None = None, url_path: str) -> None:
        self.state = _DummyState(user_data=user_data)
        self.headers = headers or {}
        self.url = _DummyURL(url_path)


def test_issue_218_async_checkpoint_archive_delegates_to_ray_helper(tmp_path, monkeypatch):
    import tinker_server.checkpoints as checkpoints

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    archive_path = tmp_path / "out.tar.gz"
    seen: dict[str, object] = {}

    class _FakeRemote:
        def remote(self, checkpoint_dir: str, archive: str):
            seen["remote_args"] = (checkpoint_dir, archive)
            return "fake-ref"

    monkeypatch.setattr(checkpoints, "_create_checkpoint_archive_remote", lambda: _FakeRemote())
    monkeypatch.setattr("tinker_server.backend.async_ray_control._ensure_ray_initialized", lambda: seen.setdefault("init", True))

    async def _fake_await_ray_ref(ref):
        seen["ref"] = ref
        Path(archive_path).write_bytes(b"archive")
        return str(archive_path)

    monkeypatch.setattr("tinker_server.backend.async_ray_control._await_ray_ref", _fake_await_ray_ref)

    asyncio.run(checkpoints.async_create_checkpoint_archive(str(ckpt_dir), str(archive_path), timeout_s=12.5))

    assert seen["remote_args"] == (str(ckpt_dir), str(archive_path))
    assert seen["ref"] == "fake-ref"
    assert archive_path.exists()


def test_issue_218_async_checkpoint_archive_cancels_ray_task_on_timeout(tmp_path, monkeypatch):
    import pytest

    import tinker_server.checkpoints as checkpoints

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    archive_path = tmp_path / "out-timeout.tar.gz"
    cancelled: list[tuple[object, bool]] = []

    class _FakeRemote:
        def remote(self, checkpoint_dir: str, archive: str):
            _ = (checkpoint_dir, archive)
            return "timeout-ref"

    monkeypatch.setattr(checkpoints, "_create_checkpoint_archive_remote", lambda: _FakeRemote())
    monkeypatch.setattr("tinker_server.backend.async_ray_control._ensure_ray_initialized", lambda: None)

    async def _fake_await_ray_ref(_ref):
        await asyncio.sleep(3600)

    monkeypatch.setattr("tinker_server.backend.async_ray_control._await_ray_ref", _fake_await_ray_ref)
    monkeypatch.setattr("ray.cancel", lambda ref, force=False: cancelled.append((ref, force)))

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(checkpoints.async_create_checkpoint_archive(str(ckpt_dir), str(archive_path), timeout_s=0.01))

    assert cancelled == [("timeout-ref", True)]


def test_issue_218_gateway_create_model_from_state_proxies_local_checkpoint_dir(tmp_path, monkeypatch):
    import tinker_server.gateway as gw
    from tinker_server.gateway import Upstream
    from tinker_server.models.types import CreateModelFromStateRequest, LoRAConfig
    from tinker_server.routes import training as tr

    ckpt_dir = tmp_path / "ckpt_local"
    ckpt_dir.mkdir()

    monkeypatch.setattr(tr, "can_access_model", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        tr, "resolve_checkpoint_path", lambda *_args, **_kwargs: str(ckpt_dir)
    )

    archive_called = []

    async def _fake_async_create_archive(checkpoint_dir: str, archive_path: str, *, timeout_s: float = 600.0) -> None:
        archive_called.append((checkpoint_dir, archive_path, timeout_s))
        Path(archive_path).write_bytes(b"fake")

    monkeypatch.setattr(tr, "async_create_checkpoint_archive", _fake_async_create_archive)

    upstream = Upstream(alias="up", base_url="http://upstream.example", auth_mode="none")
    monkeypatch.setattr(gw, "upstream_for_model", lambda _model: upstream)

    async def _fake_forward_file(*, upstream, path, incoming_headers, file_path, **_kwargs):
        assert upstream.alias == "up"
        assert path == "/api/v1/checkpoints/upload"
        assert Path(file_path).exists()
        assert Path(file_path).stat().st_size > 0
        return _DummyResponse(status_code=200, payload={"checkpoint_id": "ckpt_up"})

    async def _fake_forward_json(*, upstream, method, path, incoming_headers, json_body, **_kwargs):
        assert upstream.alias == "up"
        assert method == "POST"
        assert path == "/api/v1/create_model_from_state"
        assert isinstance(json_body, dict)
        assert json_body["state_path"] == "ckpt_up"
        return _DummyResponse(status_code=200, payload={"request_id": "rid"})

    monkeypatch.setattr(gw, "forward_file", _fake_forward_file)
    monkeypatch.setattr(gw, "forward_json", _fake_forward_json)
    monkeypatch.setattr(gw, "register_remote_training_model", lambda **_kwargs: None)
    monkeypatch.setattr(
        tr, "enforce_base_model_allowed", lambda model_name: model_name, raising=False
    )

    req = CreateModelFromStateRequest(
        session_id="sess",
        model_seq_id=1,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        state_path="ckpt_local",
        lora_config=LoRAConfig(rank=8),
        load_optimizer=True,
        user_metadata={},
    )
    out = asyncio.run(
        tr.create_model_from_state(
            request=req,
            http_request=_DummyRequest(
                url_path="/api/v1/create_model_from_state",
                user_data={"user_id": "admin"},
            ),
        )
    )

    assert gw.decode_request_id(out.request_id) == ("up", "rid")
    assert archive_called and archive_called[0][0] == str(ckpt_dir)
    assert archive_called[0][2] == 600.0


def test_issue_218_gateway_load_state_proxies_local_checkpoint_dir(tmp_path, monkeypatch):
    import tinker_server.gateway as gw
    from tinker_server.gateway import Upstream
    from tinker_server.models.types import LoadStateRequest
    from tinker_server.routes import weights as wt

    ckpt_dir = tmp_path / "ckpt_local"
    ckpt_dir.mkdir()

    monkeypatch.setattr(wt, "can_access_model", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(wt, "resolve_checkpoint_path", lambda *_args, **_kwargs: str(ckpt_dir))

    archive_called = []

    async def _fake_async_create_archive(checkpoint_dir: str, archive_path: str, *, timeout_s: float = 600.0) -> None:
        archive_called.append((checkpoint_dir, archive_path, timeout_s))
        Path(archive_path).write_bytes(b"fake")

    monkeypatch.setattr(wt, "async_create_checkpoint_archive", _fake_async_create_archive)

    upstream = Upstream(alias="up", base_url="http://upstream.example", auth_mode="none")
    monkeypatch.setattr(gw, "remote_training_model", lambda _model_id: ("up", "Qwen/Qwen3-30B-A3B-Instruct-2507"))
    async def _fake_async_remote_training_model(_model_id: str):
        return ("up", "Qwen/Qwen3-30B-A3B-Instruct-2507")
    monkeypatch.setattr(gw, "async_remote_training_model", _fake_async_remote_training_model)
    monkeypatch.setattr(gw, "upstream_for_alias", lambda _alias: upstream)

    async def _fake_forward_file(*, upstream, path, incoming_headers, file_path, **_kwargs):
        assert upstream.alias == "up"
        assert path == "/api/v1/checkpoints/upload"
        assert Path(file_path).exists()
        assert Path(file_path).stat().st_size > 0
        return _DummyResponse(status_code=200, payload={"checkpoint_id": "ckpt_up"})

    async def _fake_forward_json(*, upstream, method, path, incoming_headers, json_body, **_kwargs):
        assert upstream.alias == "up"
        assert method == "POST"
        assert path == "/api/v1/load_state"
        assert isinstance(json_body, dict)
        assert json_body["path"] == "ckpt_up"
        return _DummyResponse(status_code=200, payload={"request_id": "rid"})

    monkeypatch.setattr(gw, "forward_file", _fake_forward_file)
    monkeypatch.setattr(gw, "forward_json", _fake_forward_json)

    req = LoadStateRequest(model_id="m1", path="ckpt_local", optimizer=True)
    out = asyncio.run(
        wt.load_state(
            request=req,
            http_request=_DummyRequest(
                url_path="/api/v1/load_state",
                user_data={"user_id": "admin"},
            ),
        )
    )

    assert gw.decode_request_id(out.request_id) == ("up", "rid")
    assert archive_called and archive_called[0][0] == str(ckpt_dir)
    assert archive_called[0][2] == 600.0
