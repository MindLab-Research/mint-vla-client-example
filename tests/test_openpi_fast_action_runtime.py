from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tinker_server.backend.training_session_manager import TrainingSession
from tinker_server.models.types import EncodedTextChunk, ImageChunk, ModelInput, TensorData


OPENPI_FAST_MODEL = "openpi/pi0-fast-libero-low-mem-finetune"


def _fake_openpi_actor_env() -> dict[str, str]:
    return {
        "PYTHONPATH": "/runtime/site-packages:/repo:/hf",
        "PFS_RUNTIME_ENV_ROOT": "/runtime",
        "PFS_TINKER_PATH": "/repo",
        "PFS_HF_MODULES_PATH": "/hf",
    }


def _make_session() -> TrainingSession:
    return TrainingSession(
        model_id="model-1",
        session_id="session-1",
        model_seq_id=0,
        base_model=OPENPI_FAST_MODEL,
    )


def _make_observation() -> ModelInput:
    return ModelInput(
        chunks=[
            ImageChunk(data=b"img-0", format="png", expected_tokens=256),
            ImageChunk(data=b"img-1", format="png", expected_tokens=256),
            ImageChunk(data=b"img-2", format="png", expected_tokens=256),
            EncodedTextChunk(tokens=[11, 12, 13]),
        ]
    )


def test_openpi_fast_action_worker_prefers_env_tokenizer_override(monkeypatch, tmp_path: Path) -> None:
    from tinker_server.backend.openpi_fast_action_worker import _resolve_fast_tokenizer_path

    override = tmp_path / "fast-override"
    override.mkdir()
    monkeypatch.setenv("MINT_OPENPI_FAST_TOKENIZER_PATH", str(override))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    assert _resolve_fast_tokenizer_path("physical-intelligence/fast") == str(override)


def test_openpi_fast_action_worker_prefers_local_fast_snapshot(monkeypatch, tmp_path: Path) -> None:
    from tinker_server.backend.openpi_fast_action_worker import _resolve_fast_tokenizer_path

    hf_home = tmp_path / "hf-home"
    snapshot_dir = hf_home / "hub" / "models--physical-intelligence--fast" / "snapshots" / "rev-123"
    snapshot_dir.mkdir(parents=True)
    refs_dir = hf_home / "hub" / "models--physical-intelligence--fast" / "refs"
    refs_dir.mkdir(parents=True)
    (refs_dir / "main").write_text("rev-123", encoding="utf-8")

    monkeypatch.delenv("MINT_OPENPI_FAST_TOKENIZER_PATH", raising=False)
    monkeypatch.setenv("HF_HOME", str(hf_home))

    assert _resolve_fast_tokenizer_path("physical-intelligence/fast") == str(snapshot_dir)


def test_openpi_fast_action_worker_captures_non_protocol_stdout(monkeypatch) -> None:
    import tinker_server.backend.openpi_fast_action_worker as worker_module

    def _fake_dispatch(session, op, payload):
        _ = session, op, payload
        print("Tokens: [1, 2, 3]")
        return {"ok": True}, None

    warnings: list[str] = []

    monkeypatch.setattr(worker_module, "_dispatch", _fake_dispatch)
    monkeypatch.setattr(worker_module.logger, "warning", lambda msg, *args: warnings.append(msg % args if args else msg))

    payload, session = worker_module._dispatch_with_protocol_stdout(None, "act", {})

    assert payload == {"ok": True}
    assert session is None
    assert warnings == ["Suppressed non-protocol stdout from OpenPI FAST action worker: Tokens: [1, 2, 3]"]


def test_openpi_fast_action_worker_reply_prefers_protocol_stream(monkeypatch) -> None:
    import io
    import tinker_server.backend.openpi_fast_action_worker as worker_module

    protocol_stream = io.StringIO()
    monkeypatch.setattr(worker_module, "_PROTOCOL_STDOUT", protocol_stream)

    worker_module._reply({"ok": True})

    assert protocol_stream.getvalue() == '{"ok": true}\n'


def test_openpi_fast_action_worker_dispatch_supports_session_state_ops() -> None:
    import tinker_server.backend.openpi_fast_action_worker as worker_module

    class _FakeSession:
        def save_session_state(self, payload):
            return {"save": payload["session_id"]}

        def load_session_state(self, payload):
            return {"load": payload["session_id"]}

    session = _FakeSession()
    assert worker_module._dispatch(session, "save_session_state", {"session_id": "sess-a"}) == ({"save": "sess-a"}, session)
    assert worker_module._dispatch(session, "load_session_state", {"session_id": "sess-a"}) == ({"load": "sess-a"}, session)


def test_openpi_fast_action_worker_dispatch_cleans_up_replaced_session(monkeypatch) -> None:
    import tinker_server.backend.openpi_fast_action_worker as worker_module

    events = []

    class _ExistingSession:
        def shutdown(self):
            events.append("shutdown-old")
            return {"stopped": True}

    class _NewSession:
        def __init__(self, payload):
            events.append(("create-new", payload))

    monkeypatch.setattr(worker_module, "OpenPIFastActionSession", _NewSession)

    result, session = worker_module._dispatch(_ExistingSession(), "create_session", {"foo": "bar"})
    assert result == {"ready": True}
    assert isinstance(session, _NewSession)
    assert events == ["shutdown-old", ("create-new", {"foo": "bar"})]

    result, session = worker_module._dispatch(_ExistingSession(), "shutdown", {})
    assert result == {"stopped": True}
    assert session is None


def test_openpi_fast_action_worker_act_falls_back_on_missing_action_prefix() -> None:
    from tinker_server.backend.openpi_fast_action_worker import OpenPIFastActionSession

    fallback_calls: list[tuple[list[int], int, int]] = []

    session = OpenPIFastActionSession.__new__(OpenPIFastActionSession)
    session._observation_from_payload = lambda payload: {"payload": payload}
    session._jax = SimpleNamespace(
        random=SimpleNamespace(
            split=lambda rng: ("next-rng", "sample-rng"),
            key_data=lambda rng: np.asarray([0, 0], dtype=np.uint32),
        )
    )
    session._rng = "seed-rng"
    session._sample_counter = 0
    session._action_token_budget = 18
    session._model = SimpleNamespace(
        sample_actions=lambda rng, observation, max_decoding_steps=None, temperature=0.0: np.asarray([[1, 2, 3]], dtype=np.int32)
    )

    class _FakeTokenizer:
        _paligemma_tokenizer = SimpleNamespace(decode=lambda tokens: "bad output without action prefix")

        def extract_actions(self, tokens, action_horizon, action_dim):
            fallback_calls.append((np.asarray(tokens, dtype=np.int32).tolist(), action_horizon, action_dim))
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

    session._tokenizer = _FakeTokenizer()
    session._action_horizon = 10
    session._action_dim = 7

    result = session.act({"observation": {"chunks": []}, "extra_inputs": {"state": {"data": [0.0], "shape": [1], "dtype": "float32"}}})

    assert fallback_calls == [([1], 10, 7)]
    assert result["actions"]["shape"] == [10, 7]
    assert np.count_nonzero(np.asarray(result["actions"]["data"], dtype=np.float32)) == 0


def test_openpi_fast_action_worker_act_fails_on_strict_decode_short_payload() -> None:
    from tinker_server.backend.openpi_fast_action_worker import OpenPIFastActionSession

    session = OpenPIFastActionSession.__new__(OpenPIFastActionSession)
    session._observation_from_payload = lambda payload: {"payload": payload}
    session._jax = SimpleNamespace(
        random=SimpleNamespace(
            split=lambda rng: ("next-rng", "sample-rng"),
            key_data=lambda rng: np.asarray([0, 0], dtype=np.uint32),
        )
    )
    session._rng = "seed-rng"
    session._sample_counter = 0
    session._action_token_budget = 18
    session._model = SimpleNamespace(
        sample_actions=lambda rng, observation, max_decoding_steps=None, temperature=0.0: np.asarray([[1, 2, 3]], dtype=np.int32)
    )
    session._action_horizon = 10
    session._action_dim = 7
    session._scipy_idct = lambda arr, axis=0, norm="ortho": arr

    class _FakePaligemmaTokenizer:
        @staticmethod
        def decode(tokens):
            _ = tokens
            return "Action: abc|"

        @staticmethod
        def encode(text):
            _ = text
            return [1, 2, 3]

    class _FakeBpeTokenizer:
        @staticmethod
        def decode(tokens):
            _ = tokens
            return "abc"

    session._tokenizer = SimpleNamespace(
        _paligemma_tokenizer=_FakePaligemmaTokenizer(),
        _fast_tokenizer=SimpleNamespace(bpe_tokenizer=_FakeBpeTokenizer(), min_token=0, scale=1),
        _act_tokens_to_paligemma_tokens=lambda tokens: tokens,
    )

    with pytest.raises(RuntimeError, match="decoded action token count is smaller than the configured action shape"):
        session.act({"observation": {"chunks": []}, "extra_inputs": {"state": {"data": [0.0], "shape": [1], "dtype": "float32"}}})


def test_openpi_fast_action_worker_extracts_action_tokens_without_text_roundtrip() -> None:
    from tinker_server.backend.openpi_fast_action_worker import OpenPIFastActionSession

    session = OpenPIFastActionSession.__new__(OpenPIFastActionSession)
    session._action_horizon = 1
    session._action_dim = 7
    session._scipy_idct = lambda arr, axis=0, norm="ortho": arr
    session._action_prefix_tokens = np.asarray([10, 11], dtype=np.int32)
    session._pipe_tokens = np.asarray([99], dtype=np.int32)

    class _FakePaligemmaTokenizer:
        @staticmethod
        def decode(tokens):
            _ = tokens
            return "Action: <loc0001><loc0002>|"

        @staticmethod
        def encode(text):
            raise AssertionError("strict decode should not re-encode action text")

    class _FakeBpeTokenizer:
        @staticmethod
        def decode(tokens):
            assert tokens == [1, 2, 3, 4, 5, 6, 7]
            return "ABCDEFG"

    session._tokenizer = SimpleNamespace(
        _paligemma_tokenizer=_FakePaligemmaTokenizer(),
        _fast_tokenizer=SimpleNamespace(bpe_tokenizer=_FakeBpeTokenizer(), min_token=0, scale=1),
        _act_tokens_to_paligemma_tokens=lambda tokens: np.asarray([1, 2, 3, 4, 5, 6, 7], dtype=np.int32),
    )

    actions = session._extract_actions_strict(np.asarray([10, 11, 101, 102, 99], dtype=np.int32))

    assert actions.shape == (1, 7)


def test_openpi_fast_action_worker_truncates_excess_decoded_dct_tokens(monkeypatch) -> None:
    from tinker_server.backend.openpi_fast_action_worker import OpenPIFastActionSession

    warnings: list[str] = []

    session = OpenPIFastActionSession.__new__(OpenPIFastActionSession)
    session._action_horizon = 1
    session._action_dim = 7
    session._scipy_idct = lambda arr, axis=0, norm="ortho": arr
    session._action_prefix_tokens = np.asarray([10, 11], dtype=np.int32)
    session._pipe_tokens = np.asarray([99], dtype=np.int32)

    class _FakePaligemmaTokenizer:
        @staticmethod
        def decode(tokens):
            _ = tokens
            return "Action: <loc0001><loc0002>|"

    class _FakeBpeTokenizer:
        @staticmethod
        def decode(tokens):
            assert tokens == [1, 2, 3]
            return "ABCDEFGH"

    session._tokenizer = SimpleNamespace(
        _paligemma_tokenizer=_FakePaligemmaTokenizer(),
        _fast_tokenizer=SimpleNamespace(bpe_tokenizer=_FakeBpeTokenizer(), min_token=0, scale=1),
        _act_tokens_to_paligemma_tokens=lambda tokens: np.asarray([1, 2, 3], dtype=np.int32),
    )
    monkeypatch.setattr(
        "tinker_server.backend.openpi_fast_action_worker.logger.warning",
        lambda msg, *args: warnings.append(msg % args if args else msg),
    )

    actions = session._extract_actions_strict(np.asarray([10, 11, 101, 102, 99], dtype=np.int32))

    assert actions.shape == (1, 7)
    assert actions.reshape(-1).tolist() == [65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0]
    assert warnings == [
        "OpenPI FAST decoded action token count exceeds expected action shape; truncating decoded DCT prefix "
        "(count=8 expected=7 action_dim=7 sampled_token_count=5 raw_action_token_count=2 has_pipe=True)"
    ]


def test_openpi_fast_action_worker_act_bounds_decoding_to_expected_suffix_len() -> None:
    from tinker_server.backend.openpi_fast_action_worker import OpenPIFastActionSession

    calls: list[dict[str, object]] = []

    session = OpenPIFastActionSession.__new__(OpenPIFastActionSession)
    session._observation_from_payload = lambda payload: {"payload": payload}
    session._jax = SimpleNamespace(
        random=SimpleNamespace(
            split=lambda rng: ("next-rng", "sample-rng"),
            key_data=lambda rng: np.asarray([0, 0], dtype=np.uint32),
        )
    )
    session._rng = "seed-rng"
    session._sample_counter = 0
    session._action_token_budget = 18
    session._action_horizon = 10
    session._action_dim = 7
    session._extract_actions_strict = lambda action_tokens: np.ones((10, 7), dtype=np.float32)
    session._model = SimpleNamespace(
        sample_actions=lambda rng, observation, max_decoding_steps, temperature=0.0: (
            calls.append(
                {
                    "rng": rng,
                    "observation": observation,
                    "max_decoding_steps": max_decoding_steps,
                    "temperature": temperature,
                }
            )
            or np.asarray([[1, 2, 3]], dtype=np.int32)
        )
    )

    result = session.act(
        {"observation": {"chunks": []}, "extra_inputs": {"state": {"data": [0.0], "shape": [1], "dtype": "float32"}}, "temperature": 0.3}
    )

    assert calls == [
        {
            "rng": "sample-rng",
            "observation": {"payload": {"observation": {"chunks": []}, "extra_inputs": {"state": {"data": [0.0], "shape": [1], "dtype": "float32"}}, "temperature": 0.3}},
            "max_decoding_steps": 18,
            "temperature": 0.3,
        }
    ]
    assert result["actions"]["shape"] == [10, 7]


def test_openpi_fast_action_worker_trims_sampled_tokens_at_first_eos() -> None:
    from tinker_server.backend.openpi_fast_action_worker import OpenPIFastActionSession

    seen_tokens: list[np.ndarray] = []

    session = OpenPIFastActionSession.__new__(OpenPIFastActionSession)
    session._observation_from_payload = lambda payload: {"payload": payload}
    session._jax = SimpleNamespace(
        random=SimpleNamespace(
            split=lambda rng: ("next-rng", "sample-rng"),
            key_data=lambda rng: np.asarray([0, 0], dtype=np.uint32),
        )
    )
    session._rng = "seed-rng"
    session._sample_counter = 0
    session._action_token_budget = 8
    session._action_horizon = 10
    session._action_dim = 7
    session._paligemma_eos_id = 1
    session._model = SimpleNamespace(
        sample_actions=lambda rng, observation, max_decoding_steps=None, temperature=0.0: np.asarray(
            [[9, 8, 1, 0, 0, 0]], dtype=np.int32
        )
    )
    session._extract_actions_strict = lambda action_tokens: (
        seen_tokens.append(np.asarray(action_tokens, dtype=np.int32).copy()) or np.ones((10, 7), dtype=np.float32)
    )

    result = session.act(
        {"observation": {"chunks": []}, "extra_inputs": {"state": {"data": [0.0], "shape": [1], "dtype": "float32"}}}
    )

    assert [token.tolist() for token in seen_tokens] == [[9, 8, 1]]
    assert result["actions"]["shape"] == [10, 7]


def test_openpi_pi05_action_worker_dispatch_supports_session_state_ops() -> None:
    import tinker_server.backend.openpi_pi05_action_worker as worker_module

    class _FakeSession:
        def save_session_state(self, payload):
            return {"save": payload["session_id"]}

        def load_session_state(self, payload):
            return {"load": payload["session_id"]}

    session = _FakeSession()
    assert worker_module._dispatch(session, "save_session_state", {"session_id": "sess-b"}) == ({"save": "sess-b"}, session)
    assert worker_module._dispatch(session, "load_session_state", {"session_id": "sess-b"}) == ({"load": "sess-b"}, session)


def test_openpi_pi05_action_worker_dispatch_cleans_up_replaced_session(monkeypatch) -> None:
    import tinker_server.backend.openpi_pi05_action_worker as worker_module

    events = []

    class _ExistingSession:
        def shutdown(self):
            events.append("shutdown-old")
            return {"stopped": True}

    class _NewSession:
        def __init__(self, payload):
            events.append(("create-new", payload))

    monkeypatch.setattr(worker_module, "OpenPIPi05ActionSession", _NewSession)

    result, session = worker_module._dispatch(_ExistingSession(), "create_session", {"foo": "bar"})
    assert result == {"ready": True}
    assert isinstance(session, _NewSession)
    assert events == ["shutdown-old", ("create-new", {"foo": "bar"})]

    result, session = worker_module._dispatch(_ExistingSession(), "shutdown", {})
    assert result == {"stopped": True}
    assert session is None


class _FakeTrainingRuntimeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict | None]] = []

    async def request(self, op: str, payload: dict | None = None, *, timeout_s: float | None = None) -> dict:
        self.calls.append((op, payload))
        _ = timeout_s
        if op == "create_session":
            return {"session": "created"}
        if op in {"save_weights", "save_sampler_weights"}:
            save_path = Path(payload["save_path"])
            (save_path / "1" / "params").mkdir(parents=True, exist_ok=True)
            (save_path / "1" / "assets").mkdir(parents=True, exist_ok=True)
            (save_path / "1" / "train_state").mkdir(parents=True, exist_ok=True)
            (save_path / "1" / "params" / "_METADATA").write_text("params", encoding="utf-8")
            (save_path / "1" / "assets" / "asset.json").write_text("{}", encoding="utf-8")
            (save_path / "1" / "train_state" / "_METADATA").write_text("train", encoding="utf-8")
            return {"path": str(save_path)}
        if op == "shutdown":
            return {"stopped": True}
        raise AssertionError(f"unexpected training op {op}")

    async def close(self) -> None:
        return None


class _FakeTrainingRuntimeFactory:
    def __init__(self) -> None:
        self.clients: list[_FakeTrainingRuntimeClient] = []

    async def __call__(self, *, session: TrainingSession, model_config, config_name: str):
        _ = session, model_config, config_name
        client = _FakeTrainingRuntimeClient()
        self.clients.append(client)
        return client


class _FakeActionRuntimeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict | None]] = []

    async def request(self, op: str, payload: dict | None = None, *, timeout_s: float | None = None) -> dict:
        self.calls.append((op, payload))
        _ = timeout_s
        if op == "create_session":
            return {"ready": True}
        if op == "act":
            return {
                "actions": {
                    "data": [0.0] * 28,
                    "shape": [4, 7],
                    "dtype": "float32",
                },
                "policy_timing": {"infer_ms": 4.0},
            }
        if op == "shutdown":
            return {"stopped": True}
        raise AssertionError(f"unexpected action op {op}")

    async def close(self) -> None:
        return None


class _FakeActionRuntimeFactory:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.clients: list[_FakeActionRuntimeClient] = []

    async def __call__(self, *, action_session_id: str, base_model: str, checkpoint_path: str, model_config, config_name: str):
        self.calls.append(
            {
                "action_session_id": action_session_id,
                "base_model": base_model,
                "checkpoint_path": checkpoint_path,
                "config_name": config_name,
                "camera_layout": model_config.camera_layout,
                "action_token_budget": model_config.action_token_budget,
            }
        )
        client = _FakeActionRuntimeClient()
        self.clients.append(client)
        return client


def test_openpi_fast_default_runtime_factory_uses_shared_runtime(
    monkeypatch,
    configure_runtime_env,
) -> None:
    from tinker_server.backend.action_session_manager import _default_runtime_factory

    runtime_env = configure_runtime_env()
    calls: list[dict[str, object]] = []

    async def _fake_start_openpi_shared_ray_runtime(*, session, spec, config_name, model_config, template_reusable):
        calls.append(
            {
                "session_id": session.model_id,
                "base_model": session.base_model,
                "worker_module": spec.worker_module,
                "python_executable": spec.python_executable,
                "pythonpath": spec.pythonpath,
                "config_name": config_name,
                "action_dim": model_config.action_dim,
                "action_horizon": model_config.action_horizon,
                "action_token_budget": model_config.action_token_budget,
                "max_model_len": model_config.max_model_len,
                "template_reusable": template_reusable,
            }
        )
        return "openpi-shared-runtime-client"

    async def _unexpected_local_fast_start(spec=None):
        raise AssertionError(f"local fast action worker path must not run: {spec}")

    async def _unexpected_local_worker_start(spec):
        raise AssertionError(f"local worker path must not run: {spec.worker_module}")

    monkeypatch.setattr(
        "tinker_server.backend.action_session_manager.start_openpi_shared_ray_runtime",
        _fake_start_openpi_shared_ray_runtime,
        raising=False,
    )
    monkeypatch.setattr(
        "tinker_server.backend.openpi_fast_action_runtime.OpenPIFastActionWorkerClient.start",
        _unexpected_local_fast_start,
    )
    monkeypatch.setattr(
        "tinker_server.backend.openpi_fast_runtime.OpenPIFastWorkerClient.start",
        _unexpected_local_worker_start,
    )

    runtime = asyncio.run(
        _default_runtime_factory(
            action_session_id="session-1:action:3",
            base_model=OPENPI_FAST_MODEL,
            checkpoint_path="/tmp/export-1",
            model_config=SimpleNamespace(action_dim=7, action_horizon=10, action_token_budget=21, max_model_len=180),
            config_name="pi0_fast_libero_low_mem_finetune",
        )
    )

    assert runtime == "openpi-shared-runtime-client"
    assert calls == [
        {
            "session_id": "session-1:action:3",
            "base_model": OPENPI_FAST_MODEL,
            "worker_module": "tinker_server.backend.openpi_fast_action_worker",
            "python_executable": str(runtime_env["layout"].host_python),
            "pythonpath": runtime_env["pythonpath"],
            "config_name": "pi0_fast_libero_low_mem_finetune",
            "action_dim": 7,
            "action_horizon": 10,
            "action_token_budget": 21,
            "max_model_len": 180,
            "template_reusable": False,
        }
    ]


def test_openpi_pi05_default_runtime_factory_uses_shared_runtime(
    monkeypatch,
    configure_runtime_env,
) -> None:
    from tinker_server.backend.action_session_manager import _default_pi05_runtime_factory

    runtime_env = configure_runtime_env()
    calls: list[dict[str, object]] = []

    async def _fake_start_openpi_shared_ray_runtime(*, session, spec, config_name, model_config, template_reusable):
        calls.append(
            {
                "session_id": session.model_id,
                "base_model": session.base_model,
                "worker_module": spec.worker_module,
                "python_executable": spec.python_executable,
                "pythonpath": spec.pythonpath,
                "config_name": config_name,
                "action_dim": model_config.action_dim,
                "action_horizon": model_config.action_horizon,
                "max_model_len": model_config.max_model_len,
                "template_reusable": template_reusable,
            }
        )
        return "openpi-shared-runtime-client"

    async def _unexpected_local_fast_start(spec=None):
        raise AssertionError(f"local pi0.5 action worker path must not run: {spec}")

    async def _unexpected_local_worker_start(spec):
        raise AssertionError(f"local worker path must not run: {spec.worker_module}")

    monkeypatch.setattr(
        "tinker_server.backend.action_session_manager.start_openpi_shared_ray_runtime",
        _fake_start_openpi_shared_ray_runtime,
        raising=False,
    )
    monkeypatch.setattr(
        "tinker_server.backend.openpi_fast_action_runtime.OpenPIFastActionWorkerClient.start",
        _unexpected_local_fast_start,
    )
    monkeypatch.setattr(
        "tinker_server.backend.openpi_fast_runtime.OpenPIFastWorkerClient.start",
        _unexpected_local_worker_start,
    )

    runtime = asyncio.run(
        _default_pi05_runtime_factory(
            action_session_id="session-1:action:9",
            base_model="openpi/pi05-libero-low-mem-finetune",
            checkpoint_path="/tmp/export-9",
            model_config=SimpleNamespace(action_dim=7, action_horizon=10, max_model_len=180),
            config_name="pi05_libero",
        )
    )

    assert runtime == "openpi-shared-runtime-client"
    assert calls == [
        {
            "session_id": "session-1:action:9",
            "base_model": "openpi/pi05-libero-low-mem-finetune",
            "worker_module": "tinker_server.backend.openpi_pi05_action_worker",
            "python_executable": str(runtime_env["layout"].host_python),
            "pythonpath": runtime_env["pythonpath"],
            "config_name": "pi05_libero",
            "action_dim": 7,
            "action_horizon": 10,
            "max_model_len": 180,
            "template_reusable": False,
        }
    ]


def test_recover_detached_action_runtime_client_uses_shared_client_for_shared_actor(monkeypatch) -> None:
    from tinker_server.backend import action_session_manager
    from tinker_server.backend.resource_pool import ActorType

    class _FakeActorHandle:
        class _Describe:
            @staticmethod
            def remote():
                return "describe-ref"

        describe = _Describe()

    actor_handle_obj = _FakeActorHandle()

    class _FakeEntry:
        actor_type = ActorType.OPENPI
        actor_name = "openpi_shared_runtime_deadbeef"
        current_session = None
        base_model = OPENPI_FAST_MODEL
        actor_handle = actor_handle_obj
        metadata = {
            "worker_module": "tinker_server.backend.openpi_fast_action_worker",
            "pool_key": {"base_model": OPENPI_FAST_MODEL},
        }

    class _FakePool:
        def iter_entries(self, prune_stale: bool = False):
            assert prune_stale is True
            return [_FakeEntry()]

        def get(self, actor_name):
            assert actor_name == "openpi_shared_runtime_deadbeef"
            return None

    init: dict[str, object] = {}

    class _FakeSharedClient:
        def __init__(self, *, actor, actor_name, spec, session_id, ready_timeout_s):
            init.update(
                {
                    "actor": actor,
                    "actor_name": actor_name,
                    "worker_module": spec.worker_module,
                    "session_id": session_id,
                    "ready_timeout_s": ready_timeout_s,
                }
            )

    class _UnexpectedActionClient:
        def __init__(self, **kwargs):
            raise AssertionError(f"legacy action client recovery must not run: {kwargs}")

    monkeypatch.setattr(action_session_manager, "get_resource_pool", lambda: _FakePool())
    monkeypatch.setattr(action_session_manager, "OpenPISharedRayRuntimeClient", _FakeSharedClient)
    monkeypatch.setattr(action_session_manager, "OpenPIActionRayRuntimeClient", _UnexpectedActionClient)
    monkeypatch.setattr(action_session_manager, "_actor_ready_timeout_s", lambda spec: 123.0)
    monkeypatch.setattr(
        action_session_manager,
        "_runtime_spec_for_worker_module",
        lambda worker_module: SimpleNamespace(worker_module=worker_module),
    )
    monkeypatch.setattr(
        action_session_manager.ray,
        "get",
        lambda ref, timeout=None: {"known_session_ids": ["session-1:action:3"]},
    )

    client = action_session_manager._recover_detached_action_runtime_client(
        action_session_id="session-1:action:3",
        supports_base_model=lambda base_model: base_model == OPENPI_FAST_MODEL,
        supports_worker_module=lambda worker_module: worker_module == "tinker_server.backend.openpi_fast_action_worker",
    )

    assert isinstance(client, _FakeSharedClient)
    assert init == {
        "actor": actor_handle_obj,
        "actor_name": "openpi_shared_runtime_deadbeef",
        "worker_module": "tinker_server.backend.openpi_fast_action_worker",
        "session_id": "session-1:action:3",
        "ready_timeout_s": 123.0,
    }


def test_action_session_router_recovers_shared_session_from_known_session_ids(monkeypatch) -> None:
    from tinker_server.backend import action_session_manager
    from tinker_server.backend.resource_pool import ActorType

    class _FakeActorHandle:
        class _Describe:
            @staticmethod
            def remote():
                return "describe-ref"

        describe = _Describe()

    class _FakeEntry:
        actor_type = ActorType.OPENPI
        actor_name = "openpi_shared_runtime_deadbeef"
        current_session = None
        base_model = OPENPI_FAST_MODEL
        actor_handle = _FakeActorHandle()
        metadata = {
            "worker_module": "tinker_server.backend.openpi_fast_action_worker",
            "pool_key": {"base_model": OPENPI_FAST_MODEL},
        }

    class _FakePool:
        def iter_entries(self, prune_stale: bool = False):
            assert prune_stale is True
            return [_FakeEntry()]

    fake_fast_manager = object()

    monkeypatch.setattr(action_session_manager, "get_resource_pool", lambda: _FakePool())
    monkeypatch.setattr(
        action_session_manager.ray,
        "get",
        lambda ref, timeout=None: {"known_session_ids": ["session-1:action:3"]},
    )

    router = action_session_manager.ActionSessionRouter(
        openpi_fast_manager=fake_fast_manager,
        openpi_pi05_manager=object(),
    )

    recovered = router._recover_manager_for_session("session-1:action:3")

    assert recovered is fake_fast_manager
    assert router._manager_for_session["session-1:action:3"] is fake_fast_manager


def test_start_openpi_action_ray_runtime_registers_actor_metadata_in_resource_pool(monkeypatch) -> None:
    from tinker_server.backend import openpi_action_ray_runtime
    from tinker_server.backend.openpi_fast_action_runtime import OpenPIFastActionRuntimeSpec

    state: dict[str, object] = {}

    class _FakeActorBuilder:
        def options(self, **kwargs):
            state["options"] = kwargs
            return self

        def remote(self, **kwargs):
            state["remote"] = kwargs
            return "actor-1"

    class _FakeClient:
        def __init__(self, *, actor, actor_name, spec, action_session_id, ready_timeout_s):
            state["client_init"] = {
                "actor": actor,
                "actor_name": actor_name,
                "action_session_id": action_session_id,
                "ready_timeout_s": ready_timeout_s,
                "worker_module": spec.worker_module,
            }

        async def ready(self):
            return {
                "actor_id": "actor-123",
                "node_id": "node-456",
                "node_ip": "192.168.0.8",
                "pid": 999,
                "cuda_visible_devices": "0",
                "action_session_id": "session-1:action:3",
            }

        async def close(self):
            return None

    class _FakePool:
        def register(self, **kwargs):
            state["register"] = kwargs

        def mark_ready(self, actor_name):
            state["mark_ready"] = actor_name

        def touch(self, actor_name):
            state["touch"] = actor_name

    monkeypatch.setattr(openpi_action_ray_runtime, "ensure_openpi_ray_initialized", lambda: None)
    monkeypatch.setattr(openpi_action_ray_runtime, "OpenPIActionRayRuntimeActor", _FakeActorBuilder())
    monkeypatch.setattr(openpi_action_ray_runtime, "OpenPIActionRayRuntimeClient", _FakeClient)
    monkeypatch.setattr(openpi_action_ray_runtime, "get_resource_pool", lambda: _FakePool())
    monkeypatch.setattr(openpi_action_ray_runtime, "_openpi_runtime_env_vars", _fake_openpi_actor_env)
    monkeypatch.setenv("PFS_TINKER_PATH", "/repo")

    client = asyncio.run(
        openpi_action_ray_runtime.start_openpi_action_ray_runtime(
            action_session_id="session-1:action:3",
            base_model=OPENPI_FAST_MODEL,
            spec=OpenPIFastActionRuntimeSpec(
                python_executable="/tmp/runtime/host-venv/bin/python",
                worker_module="tinker_server.backend.openpi_fast_action_worker",
                pythonpath=("/tmp/runtime/site-packages", "/tmp/runtime/src/openpi/src"),
            ),
        )
    )

    assert isinstance(client, _FakeClient)
    expected_actor_name = state["client_init"]["actor_name"]
    assert state["options"]["runtime_env"]["env_vars"]["MINT_OPENPI_FAST_ACTION_SESSION_STATE_ROOT"] == (
        f"/repo/checkpoints/openpi_action_session_state/tinker/{expected_actor_name}"
    )
    register = state["register"]
    assert register["actor_type"].value == "openpi"
    assert register["base_model"] == OPENPI_FAST_MODEL
    assert register["session_id"] == "session-1:action:3"
    assert register["node_id"] == "node-456"
    assert register["metadata"]["worker_module"] == "tinker_server.backend.openpi_fast_action_worker"
    assert register["metadata"]["actor_id"] == "actor-123"
    assert register["metadata"]["node_ip"] == "192.168.0.8"
    assert register["metadata"]["pid"] == 999
    assert register["metadata"]["cuda_visible_devices"] == "0"


def test_start_openpi_action_ray_runtime_applies_single_node_pin(monkeypatch) -> None:
    from tinker_server.backend import openpi_action_ray_runtime
    from tinker_server.backend.openpi_fast_action_runtime import OpenPIFastActionRuntimeSpec

    state: dict[str, object] = {}
    node_id = "b" * 56

    class _FakeActorBuilder:
        def options(self, **kwargs):
            state["options"] = kwargs
            return self

        def remote(self, **kwargs):
            state["remote"] = kwargs
            return "actor-1"

    class _FakeClient:
        def __init__(self, *, actor, actor_name, spec, action_session_id, ready_timeout_s):
            _ = actor, actor_name, spec, action_session_id, ready_timeout_s

        async def ready(self):
            return {"actor_id": "actor-123", "node_id": node_id, "node_ip": "192.168.38.176"}

        async def close(self):
            return None

    class _FakePool:
        def register(self, **kwargs):
            state["register"] = kwargs

        def mark_ready(self, actor_name):
            state["mark_ready"] = actor_name

        def touch(self, actor_name):
            state["touch"] = actor_name

    monkeypatch.setattr(openpi_action_ray_runtime, "ensure_openpi_ray_initialized", lambda: None)
    monkeypatch.setattr(openpi_action_ray_runtime, "OpenPIActionRayRuntimeActor", _FakeActorBuilder())
    monkeypatch.setattr(openpi_action_ray_runtime, "OpenPIActionRayRuntimeClient", _FakeClient)
    monkeypatch.setattr(openpi_action_ray_runtime, "get_resource_pool", lambda: _FakePool())
    monkeypatch.setattr(openpi_action_ray_runtime, "_openpi_runtime_env_vars", _fake_openpi_actor_env)
    monkeypatch.setenv("PFS_TINKER_PATH", "/repo")
    monkeypatch.setattr(
        openpi_action_ray_runtime,
        "parse_model_node_ip_list",
        lambda **_kwargs: ["192.168.38.176"],
    )
    capacity_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        openpi_action_ray_runtime,
        "assert_node_ip_capacity",
        lambda **kwargs: capacity_calls.append(kwargs),
    )
    monkeypatch.setattr(
        openpi_action_ray_runtime.ray,
        "nodes",
        lambda: [{"Alive": True, "NodeManagerAddress": "192.168.38.176", "NodeID": node_id}],
    )

    asyncio.run(
        openpi_action_ray_runtime.start_openpi_action_ray_runtime(
            action_session_id="session-1:action:3",
            base_model=OPENPI_FAST_MODEL,
            spec=OpenPIFastActionRuntimeSpec(),
        )
    )

    options = state["options"]
    assert options["runtime_env"]["env_vars"]["MINT_OPENPI_FAST_ACTION_SESSION_STATE_ROOT"] == (
        f"/repo/checkpoints/openpi_action_session_state/tinker/{state['register']['actor_name']}"
    )
    assert options["resources"] == {"node:192.168.38.176": 0.001}
    assert options["scheduling_strategy"].node_id == node_id
    assert options["scheduling_strategy"].soft is False
    assert capacity_calls == [
        {
            "required_gpus_by_node_ip": {"192.168.38.176": 1},
            "context": "[OpenPIActionRuntime] node pinning model='openpi/pi0-fast-libero-low-mem-finetune' actor="
            f"{state['register']['actor_name']!r}",
        }
    ]


def test_openpi_fast_save_weights_for_sampler_exports_policy_loadable_checkpoint(tmp_path: Path) -> None:
    from tinker_server.backend.openpi_fast_training import OpenPIFastTrainingEngine

    factory = _FakeTrainingRuntimeFactory()
    engine = OpenPIFastTrainingEngine(runtime_factory=factory)
    session = _make_session()
    asyncio.run(engine.create_training_session(session))

    export_path = asyncio.run(
        engine.save_weights_for_sampler(
            session=session,
            checkpoint_name="export-1",
            checkpoint_base_dir=str(tmp_path),
        )
    )

    export_dir = Path(export_path)
    assert export_dir == tmp_path / "model-1" / "export-1"
    assert (export_dir / "params" / "_METADATA").exists()
    assert (export_dir / "assets" / "asset.json").exists()
    assert not (export_dir / "train_state").exists()


def test_action_session_manager_create_session_starts_runtime_from_checkpoint(tmp_path: Path) -> None:
    from tinker_server.backend.action_session_manager import OpenPIFastActionSessionManager

    checkpoint_dir = tmp_path / "model-1" / "export-1"
    (checkpoint_dir / "params").mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "assets").mkdir(parents=True, exist_ok=True)

    factory = _FakeActionRuntimeFactory()
    manager = OpenPIFastActionSessionManager(runtime_factory=factory)

    action_session_id = asyncio.run(
        manager.create_session(
            session_id="session-1",
            action_session_seq_id=3,
            base_model=OPENPI_FAST_MODEL,
            model_path=f"file://{checkpoint_dir}",
            user_id="admin",
        )
    )

    assert action_session_id == "session-1:action:3"
    assert factory.calls == [
        {
            "action_session_id": "session-1:action:3",
            "base_model": OPENPI_FAST_MODEL,
            "checkpoint_path": str(checkpoint_dir.resolve()),
            "config_name": "pi0_fast_libero_low_mem_finetune",
            "camera_layout": ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
            "action_token_budget": 64,
        }
    ]
    assert factory.clients[0].calls[0] == (
        "create_session",
        {
            "action_session_id": "session-1:action:3",
            "base_model": OPENPI_FAST_MODEL,
            "checkpoint_path": str(checkpoint_dir.resolve()),
                "config_name": "pi0_fast_libero_low_mem_finetune",
                "action_dim": 7,
                "action_horizon": 10,
                "action_token_budget": 64,
                "max_token_len": 180,
                "camera_layout": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
            },
        )


def test_action_session_manager_act_returns_actions(tmp_path: Path) -> None:
    from tinker_server.backend.action_session_manager import OpenPIFastActionSessionManager

    checkpoint_dir = tmp_path / "model-1" / "export-1"
    (checkpoint_dir / "params").mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "assets").mkdir(parents=True, exist_ok=True)

    factory = _FakeActionRuntimeFactory()
    manager = OpenPIFastActionSessionManager(runtime_factory=factory)
    action_session_id = asyncio.run(
        manager.create_session(
            session_id="session-1",
            action_session_seq_id=None,
            base_model=OPENPI_FAST_MODEL,
            model_path=f"file://{checkpoint_dir}",
            user_id="admin",
        )
    )

    result = asyncio.run(
        manager.act(
            action_session_id=action_session_id,
            observation=_make_observation(),
            extra_inputs={"state": TensorData(data=[0.0] * 8, shape=[8], dtype="float32")},
        )
    )

    assert result == {
        "actions": {
            "data": [0.0] * 28,
            "shape": [4, 7],
            "dtype": "float32",
        },
        "policy_timing": {"infer_ms": 4.0},
    }
    assert factory.clients[0].calls[-1][0] == "act"


def test_action_session_manager_rejects_unsupported_model_family(tmp_path: Path) -> None:
    from tinker_server.backend.action_session_manager import OpenPIFastActionSessionManager

    checkpoint_dir = tmp_path / "model-1" / "export-1"
    (checkpoint_dir / "params").mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "assets").mkdir(parents=True, exist_ok=True)

    manager = OpenPIFastActionSessionManager(runtime_factory=_FakeActionRuntimeFactory())

    with pytest.raises(ValueError, match="OpenPI FAST"):
        asyncio.run(
            manager.create_session(
                session_id="session-1",
                action_session_seq_id=None,
                base_model="Qwen/Qwen2.5-0.5B-Instruct",
                model_path=f"file://{checkpoint_dir}",
                user_id="admin",
            )
        )
