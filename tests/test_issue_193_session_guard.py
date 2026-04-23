# ruff: noqa: F403,F405
from tests.issue193_common import *


def test_issue_193_actor_only_state_marker_corruption_fails_closed(tmp_path: Path):
    manager = MegatronSessionStateManager(base_path=str(tmp_path))
    marker_path = Path(manager._actor_only_state_path("session_issue_193_corrupt_marker"))
    marker_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Failed to read actor_only_state marker"):
        manager.list_actor_only_state_sessions("megatron_qwen3_30b_a3b_instruct_2507")


def test_issue_193_session_metadata_cache_does_not_mask_disk_corruption(tmp_path: Path):
    manager = MegatronSessionStateManager(base_path=str(tmp_path))
    session_id = "session_issue_193_corrupt_metadata"
    manager.save_metadata(session_id, step=3, lr=1e-4, actual_rank=8)
    metadata_path = Path(manager._metadata_path(session_id))
    metadata_path.write_text("{not-json", encoding="utf-8")

    assert manager.get_metadata(session_id) is None


@pytest.mark.parametrize(
    "lr_value",
    [True, float("nan"), float("inf")],
    ids=["bool", "nan", "inf"],
)
def test_issue_193_session_metadata_rejects_nonfinite_or_bool_lr(tmp_path: Path, lr_value):
    manager = MegatronSessionStateManager(base_path=str(tmp_path))
    session_id = "session_issue_193_bad_lr_metadata"
    metadata_path = Path(manager._metadata_path(session_id))
    metadata_path.write_text(
        json.dumps({"step": 1, "lr": lr_value, "actual_rank": 8}),
        encoding="utf-8",
    )

    assert manager.get_metadata(session_id) is None


def test_issue_193_prime_session_uses_sidecars_and_detaches_on_dirty(tmp_path: Path):
    manager = MegatronSessionStateManager(base_path=str(tmp_path / "sessions"))
    checkpoint_dir = tmp_path / "checkpoint_source"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "mp_rank_00_adapter.pt").write_text("adapter", encoding="utf-8")

    session_id = "session_issue_193_sidecar_detach"
    session_path = Path(manager.prime_session(session_id, str(checkpoint_dir), step=3, lr=1e-4, actual_rank=8, optimizer_restored=False))
    metadata_path = Path(manager._metadata_path(session_id))
    marker_path = Path(manager._actor_only_state_path(session_id))

    assert session_path.is_dir()
    assert not session_path.is_symlink()
    assert metadata_path.exists()
    assert not (checkpoint_dir / "session_metadata.json").exists()
    assert (checkpoint_dir / "mp_rank_00_adapter.pt").read_text(encoding="utf-8") == "adapter"
    assert (session_path / "mp_rank_00_adapter.pt").read_text(encoding="utf-8") == "adapter"

    manager.mark_actor_only_state(
        session_id,
        reason="forward_backward",
        actor_name="megatron_qwen3_30b_a3b_instruct_2507",
    )

    assert session_path.exists()
    assert session_path.is_dir()
    assert not session_path.is_symlink()
    assert marker_path.exists()
    assert not (checkpoint_dir / "actor_only_state.json").exists()
    assert manager.get_metadata(session_id)["checkpoint_path"] == os.path.realpath(checkpoint_dir)


def test_issue_193_megatron_dirty_noncurrent_session_fails_before_swap(monkeypatch):
    group_cls = MegatronWorkerGroup.__ray_actor_class__
    group = group_cls.__new__(group_cls)
    group._current_session = "session_current"
    group.base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    group.learning_rate = 1e-4
    group._actual_rank = 8
    group.lora_rank = 8

    class _FakeHasSessionStateCachedRemoteMethod:
        def remote(self, session_id):
            return False

    group.workers = [type("W", (), {"has_session_state_cached": _FakeHasSessionStateCachedRemoteMethod()})()]
    group._session_manager = SimpleNamespace(
        session_exists=lambda session_id: session_id == "session_target",
        has_actor_only_state=lambda session_id: session_id == "session_target",
    )
    group._bind_traceparent = lambda traceparent: None
    group._resolve_required_session_id = lambda session_id, op: session_id
    group._get_lora_weight_norm = lambda: 0.0
    group._get_lora_weight_checksum = lambda: "0"
    group._get_base_weight_checksum = lambda: "0"
    group._get_buffer_checksum = lambda: "0"
    group._get_optimizer_param_counts = lambda: {}
    group._swap_session_on_workers = lambda session_id: (_ for _ in ()).throw(
        AssertionError("dirty target session must fail before swap")
    )
    group.save_adapter_state = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("dirty target session must fail before saving outgoing session")
    )
    group.reinit_lora_weights = lambda *args, **kwargs: None
    group.reset_expert_bias = lambda *args, **kwargs: None

    with pytest.raises(RuntimeError, match="actor-only training state"):
        group._ensure_session_loaded("session_target")


def test_issue_193_megatron_dirty_noncurrent_session_without_adapter_cache_fails_before_swap():
    group_cls = MegatronWorkerGroup.__ray_actor_class__
    group = group_cls.__new__(group_cls)
    group._current_session = "session_current"
    group.base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    group.learning_rate = 1e-4
    group._actual_rank = 8
    group.lora_rank = 8

    class _FakeHasSessionStateCachedRemoteMethod:
        def remote(self, session_id):
            return False

    group.workers = [type("W", (), {"has_session_state_cached": _FakeHasSessionStateCachedRemoteMethod()})()]
    group._session_manager = SimpleNamespace(
        session_exists=lambda session_id: False,
        has_actor_only_state=lambda session_id: session_id == "session_target",
    )
    group._bind_traceparent = lambda traceparent: None
    group._get_lora_weight_norm = lambda: 0.0
    group._get_lora_weight_checksum = lambda: "0"
    group._get_base_weight_checksum = lambda: "0"
    group._get_buffer_checksum = lambda: "0"
    group._get_optimizer_param_counts = lambda: {}
    group._swap_session_on_workers = lambda session_id: (_ for _ in ()).throw(
        AssertionError("dirty target session must fail before swap")
    )
    group.save_adapter_state = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("dirty target session must fail before saving outgoing session")
    )
    group.reinit_lora_weights = lambda *args, **kwargs: None
    group.reset_expert_bias = lambda *args, **kwargs: None

    with pytest.raises(RuntimeError, match="actor-only training state"):
        group._ensure_session_loaded("session_target")


def test_issue_193_megatron_invalid_noncurrent_metadata_fails_before_swap():
    group_cls = MegatronWorkerGroup.__ray_actor_class__
    group = group_cls.__new__(group_cls)
    group._current_session = "session_current"
    group.base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    group.learning_rate = 1e-4
    group._actual_rank = 8
    group.lora_rank = 8
    group.workers = []
    group._session_manager = SimpleNamespace(
        session_exists=lambda session_id: session_id == "session_target",
        has_actor_only_state=lambda session_id: False,
        get_metadata=lambda session_id: None,
    )
    group._bind_traceparent = lambda traceparent: None
    group._get_lora_weight_norm = lambda: 0.0
    group._get_lora_weight_checksum = lambda: "0"
    group._get_base_weight_checksum = lambda: "0"
    group._get_buffer_checksum = lambda: "0"
    group._get_optimizer_param_counts = lambda: {}
    group._swap_session_on_workers = lambda session_id: (_ for _ in ()).throw(
        AssertionError("invalid metadata must fail before swap")
    )
    group.save_adapter_state = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("invalid metadata must fail before saving outgoing session")
    )
    group.reinit_lora_weights = lambda *args, **kwargs: None
    group.reset_expert_bias = lambda *args, **kwargs: None

    with pytest.raises(RuntimeError, match="missing session_metadata.json"):
        group._ensure_session_loaded("session_target")


def test_issue_193_megatron_current_session_corruption_fails_closed():
    group_cls = MegatronWorkerGroup.__ray_actor_class__
    group = group_cls.__new__(group_cls)
    group._current_session = "session_current"
    group.base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    group.learning_rate = 1e-4
    group._actual_rank = 8
    group.lora_rank = 8
    group.workers = []
    group._session_manager = SimpleNamespace(
        has_actor_only_state=lambda session_id: (_ for _ in ()).throw(
            RuntimeError("Failed to read actor_only_state marker")
        ),
        session_exists=lambda session_id: True,
        get_metadata=lambda session_id: {"step": 1, "lr": 1e-4, "actual_rank": 8},
    )
    group._bind_traceparent = lambda traceparent: None

    with pytest.raises(RuntimeError, match="Failed to read actor_only_state marker"):
        group._ensure_session_loaded("session_current")


def test_issue_193_megatron_explicit_load_prepare_converges_outgoing_actor_only_state(monkeypatch):
    group_cls = MegatronWorkerGroup.__ray_actor_class__
    group = group_cls.__new__(group_cls)
    group._current_session = "session_outgoing"
    group.base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    group.learning_rate = 1e-4
    group._step_count = 0
    group._actual_rank = 8
    group.lora_rank = 8
    group._session_unknown_due_to_partial_swap = False

    swap_calls: list[str] = []
    mark_calls: list[str] = []
    clear_calls: list[str] = []
    clear_persisted_calls: list[str] = []
    save_metadata_calls: list[tuple] = []
    persisted_manifest_calls: list[dict] = []

    class _FakeWorker:
        class clear_session_state:
            @staticmethod
            def remote(session_id, traceparent=None):
                clear_calls.append(session_id)
                return object()

        class mark_session_loaded:
            @staticmethod
            def remote(session_id):
                mark_calls.append(session_id)
                return object()

    group.workers = [_FakeWorker()]
    group._session_manager = SimpleNamespace(
        get_session_path=lambda session_id: f"/tmp/{session_id}",
        has_actor_only_state=lambda session_id: session_id == "session_outgoing",
        session_exists=lambda session_id: session_id == "session_target",
        save_metadata=lambda *args: save_metadata_calls.append(args),
        clear_persisted_actor_only_state=lambda session_id: clear_persisted_calls.append(session_id),
        clear_actor_only_state=lambda session_id: clear_calls.append(session_id),
        save_persisted_actor_only_state=lambda session_id, actor_name, worker_entries: persisted_manifest_calls.append(
            {
                "session_id": session_id,
                "actor_name": actor_name,
                "worker_entries": worker_entries,
            }
        ),
    )
    group._bind_traceparent = lambda traceparent: None
    group.save_adapter_state = lambda *args, **kwargs: None
    group._swap_session_on_workers = lambda session_id: (
        swap_calls.append(session_id)
        or [{"outgoing_persisted": {"rank": 0, "path": "/tmp/r0.pt", "bytes": 123}}]
    )
    monkeypatch.setattr(ray, "get", lambda refs, timeout=None: None)

    group._prepare_session_for_explicit_load("session_target")

    assert swap_calls == ["session_target"]
    assert mark_calls == []
    assert save_metadata_calls and save_metadata_calls[0][0] == "session_outgoing"
    assert clear_persisted_calls == ["session_target"]
    assert persisted_manifest_calls == [
        {
            "session_id": "session_outgoing",
            "actor_name": "megatron_qwen3_30b_a3b_instruct_2507",
            "worker_entries": [{"rank": 0, "path": "/tmp/r0.pt", "bytes": 123}],
        }
    ]
    # clear_session_state(target) + clear_actor_only_state(target) + clear_actor_only_state(outgoing)
    assert clear_calls.count("session_target") == 2
    assert clear_calls.count("session_outgoing") == 1
    assert group._current_session == "session_target"


def test_issue_193_megatron_explicit_load_prepare_allows_dirty_target_on_fresh_actor(monkeypatch):
    group_cls = MegatronWorkerGroup.__ray_actor_class__
    group = group_cls.__new__(group_cls)
    group._current_session = None
    group.base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    group.learning_rate = 1e-4
    group._actual_rank = 8
    group.lora_rank = 8
    group._session_unknown_due_to_partial_swap = False
    swap_calls: list[str] = []
    clear_calls: list[str] = []

    class _FakeWorker:
        class clear_session_state:
            @staticmethod
            def remote(session_id, traceparent=None):
                clear_calls.append(session_id)
                return object()

    group.workers = [_FakeWorker()]
    group._session_manager = SimpleNamespace(
        get_session_path=lambda session_id: f"/tmp/{session_id}",
        has_actor_only_state=lambda session_id: (_ for _ in ()).throw(
            AssertionError("explicit checkpoint prepare must not consult target dirty marker")
        ),
    )
    group._bind_traceparent = lambda traceparent: None
    group._swap_session_on_workers = lambda session_id: swap_calls.append(session_id)
    monkeypatch.setattr(ray, "get", lambda refs, timeout=None: None)

    group._prepare_session_for_explicit_load("session_target")

    assert clear_calls == ["session_target"]
    assert swap_calls == ["session_target"]
    assert group._current_session == "session_target"


def test_issue_193_megatron_forward_uses_ensure_session_loaded(monkeypatch):
    group_cls = MegatronWorkerGroup.__ray_actor_class__
    group = group_cls.__new__(group_cls)
    ensure_calls: list[str] = []

    class _FakeWorker:
        class forward:
            @staticmethod
            def remote(data_items, reset_bias, traceparent=None):
                return {"loss_fn_outputs": [{"loss": {"data": [0.0]}}], "loss_value": 0.0, "num_tokens": 0}

    group.workers = [_FakeWorker()]
    group._bind_traceparent = lambda traceparent: None
    group._resolve_required_session_id = lambda session_id, op: session_id
    group._ensure_session_loaded = lambda session_id, **kwargs: ensure_calls.append(session_id)
    group._prepare_session_for_explicit_load = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("forward must not use explicit-load preparation")
    )
    monkeypatch.setattr(ray, "get", lambda futures, timeout=None: futures)

    result = group.forward([{"x": 1}], session_id="session_target")

    assert ensure_calls == ["session_target"]
    assert result["loss_fn_outputs"][0]["loss"]["data"] == [0.0]


def test_issue_193_megatron_forward_backward_uses_ensure_session_loaded(monkeypatch):
    group_cls = MegatronWorkerGroup.__ray_actor_class__
    group = group_cls.__new__(group_cls)
    ensure_calls: list[str] = []

    class _FakeWorker:
        class forward_backward:
            @staticmethod
            def remote(
                data_items,
                loss_fn,
                loss_fn_config,
                rollout_correction_config,
                session_id,
                reset_bias,
                traceparent=None,
            ):
                return {
                    "loss_fn_outputs": [{"loss": {"data": [0.0]}}],
                    "loss_value": 0.0,
                    "loss_sum_value": 0.0,
                    "num_tokens": 0,
                    "valid_count": len(data_items),
                }

    group.workers = [_FakeWorker()]
    group.base_model = "Qwen/Qwen3-0.6B"
    group._bind_traceparent = lambda traceparent: None
    group._resolve_required_session_id = lambda session_id, op: session_id
    group._ensure_session_loaded = lambda session_id, **kwargs: ensure_calls.append(session_id)
    group._prepare_session_for_explicit_load = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("forward_backward must not use explicit-load preparation")
    )
    group._start_slow_group_watchdog = lambda **kwargs: None
    group._stop_slow_group_watchdog = lambda watchdog: None
    group._session_manager = SimpleNamespace(mark_actor_only_state=lambda *args, **kwargs: None)
    monkeypatch.setattr(ray, "get", lambda futures, timeout=None: futures)

    result = group.forward_backward([{"x": 1}], session_id="session_target")

    assert ensure_calls == ["session_target"]
    assert result["loss_fn_outputs"][0]["loss"]["data"] == [0.0]


def test_issue_193_megatron_optim_step_uses_ensure_session_loaded(monkeypatch):
    group_cls = MegatronWorkerGroup.__ray_actor_class__
    group = group_cls.__new__(group_cls)
    ensure_calls: list[str] = []

    class _FakeWorker:
        class optim_step:
            @staticmethod
            def remote(learning_rate, session_id, **kwargs):
                return {"grad_norm": 1.25, "lr": learning_rate}

    group.workers = [_FakeWorker()]
    group.base_model = "Qwen/Qwen3-0.6B"
    group._step_count = 0
    group._bind_traceparent = lambda traceparent: None
    group._resolve_required_session_id = lambda session_id, op: session_id
    group._ensure_session_loaded = lambda session_id, **kwargs: ensure_calls.append(session_id)
    group._prepare_session_for_explicit_load = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("optim_step must not use explicit-load preparation")
    )
    group._session_manager = SimpleNamespace(mark_actor_only_state=lambda *args, **kwargs: None)
    monkeypatch.setattr(ray, "get", lambda futures, timeout=None: futures)

    result = group.optim_step(2e-4, session_id="session_target")

    assert ensure_calls == ["session_target"]
    assert result["metrics"]["grad_norm"] == 1.25


def test_issue_193_megatron_ab_interleave_reloads_session_before_optim_step(monkeypatch):
    group_cls = MegatronWorkerGroup.__ray_actor_class__
    group = group_cls.__new__(group_cls)
    ensure_calls: list[str] = []
    call_log: list[tuple[str, str, str | None]] = []

    def _ensure_session_loaded(session_id, **kwargs):
        ensure_calls.append(session_id)
        group._current_session = session_id

    class _FakeWorker:
        class forward_backward:
            @staticmethod
            def remote(
                data_items,
                loss_fn,
                loss_fn_config,
                rollout_correction_config,
                session_id,
                reset_bias,
                traceparent=None,
            ):
                call_log.append(("forward_backward", session_id, group._current_session))
                return {
                    "loss_fn_outputs": [],
                    "loss_value": 0.0,
                    "loss_sum_value": 0.0,
                    "num_tokens": 0,
                    "valid_count": 0,
                }

        class forward:
            @staticmethod
            def remote(data_items, reset_bias, traceparent=None):
                call_log.append(("forward", group._current_session, group._current_session))
                return {
                    "loss_fn_outputs": [],
                    "loss_value": 0.0,
                    "loss_sum_value": 0.0,
                    "num_tokens": 0,
                    "valid_count": 0,
                    "log_probs": None,
                }

        class optim_step:
            @staticmethod
            def remote(learning_rate, session_id, **kwargs):
                call_log.append(("optim_step", session_id, group._current_session))
                return {"grad_norm": 0.5, "lr": learning_rate}

    group.workers = [_FakeWorker()]
    group.base_model = "Qwen/Qwen3-0.6B"
    group._step_count = 0
    group._current_session = None
    group._bind_traceparent = lambda traceparent: None
    group._resolve_required_session_id = lambda session_id, op: session_id
    group._ensure_session_loaded = _ensure_session_loaded
    group._prepare_session_for_explicit_load = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("forward/forward_backward/optim_step must not use explicit-load preparation")
    )
    group._start_slow_group_watchdog = lambda **kwargs: None
    group._stop_slow_group_watchdog = lambda watchdog: None
    group._session_manager = SimpleNamespace(mark_actor_only_state=lambda *args, **kwargs: None)
    monkeypatch.setattr(ray, "get", lambda futures, timeout=None: futures)

    group.forward_backward([], session_id="session_a")
    group.forward([], session_id="session_b")
    result = group.optim_step(3e-4, session_id="session_a")

    assert ensure_calls == ["session_a", "session_b", "session_a"]
    assert call_log == [
        ("forward_backward", "session_a", "session_a"),
        ("forward", "session_b", "session_b"),
        ("optim_step", "session_a", "session_a"),
    ]
    assert result["metrics"]["step"] == 1


def test_issue_193_megatron_load_checkpoint_uses_explicit_load_prepare(tmp_path: Path):
    group_cls = MegatronWorkerGroup.__ray_actor_class__
    group = group_cls.__new__(group_cls)
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "mp_rank_00_000_000_adapter.pt").write_bytes(b"adapter")
    (ckpt_dir / "adapter_config.json").write_text(json.dumps({"r": 8}), encoding="utf-8")
    (ckpt_dir / "training_meta.json").write_text(
        json.dumps({"current_step": 3, "learning_rate": 2e-4}),
        encoding="utf-8",
    )

    prepare_calls: list[str] = []
    load_adapter_calls: list[tuple[str, dict]] = []
    reset_optimizer_calls: list[tuple[tuple, dict]] = []
    group.workers = []
    group._bind_traceparent = lambda traceparent: None
    group._resolve_required_session_id = lambda session_id, op: session_id
    group._prepare_session_for_explicit_load = lambda session_id, traceparent=None: prepare_calls.append(session_id)
    group._ensure_session_loaded = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("load_checkpoint must not use ordinary ensure path")
    )
    group.load_adapter_state = lambda load_path, **kwargs: load_adapter_calls.append((load_path, kwargs)) or {}
    group.reset_optimizer = lambda *args, **kwargs: reset_optimizer_calls.append((args, kwargs)) or None
    group._step_count = 0
    group.learning_rate = 1e-4
    group._current_session = None
    group._session_unknown_due_to_partial_swap = False
    group._actual_rank = 8
    group.lora_rank = 8
    group._session_manager = SimpleNamespace(
        checkpoint_identity=lambda checkpoint_path: f"identity:{checkpoint_path}",
        has_actor_only_state=lambda session_id: False,
        session_exists=lambda session_id: False,
        get_metadata=lambda session_id: None,
        prime_session=lambda *args, **kwargs: None,
        clear_actor_only_state=lambda session_id: None,
        mark_actor_only_state=lambda *args, **kwargs: None,
    )

    result = group.load_checkpoint(str(ckpt_dir), load_optimizer=False, session_id="session_target")

    assert prepare_calls == ["session_target"]
    assert len(load_adapter_calls) == 1
    call_path, call_kwargs = load_adapter_calls[0]
    assert call_path == str(ckpt_dir)
    assert call_kwargs["actual_rank"] == 8
    assert call_kwargs["traceparent"] is None
    assert call_kwargs["train_attn"] is None
    assert call_kwargs["train_mlp"] is None
    assert call_kwargs["train_unembed"] is None
    assert reset_optimizer_calls == [
        (
            (2e-4,),
            {"traceparent": None, "zero_grad_buffers": False},
        )
    ]
    assert result["optimizer_reset"] is True


def test_issue_193_megatron_resolution_never_falls_back_to_default_base_model():
    engine = VerlTrainingEngine()
    engine.default_base_model = "/tmp/wrong-default"
    session = TrainingSession(
        model_id="model_issue_193_megatron_resolution_strict",
        session_id="session_issue_193_megatron_resolution_strict",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )
    engine._resolve_hf_model_path = lambda requested_model: None

    with pytest.raises(RuntimeError, match="could not resolve Megatron base model"):
        engine._resolve_megatron_base_model(session)


def test_issue_193_megatron_midcall_mutating_op_fails_closed_even_when_actor_was_clean(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_midcall_mutating"
    dead_worker = object()
    recovered_worker = object()
    actor_name = "shared-megatron-actor"
    engine._workers[model_id] = dead_worker
    engine._resource_pool_actor_names[model_id] = actor_name

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_midcall_mutating",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    async def fake_get_live_worker(*args, **kwargs):
        return dead_worker

    async def fake_keepalive(awaitable, *_args, **_kwargs):
        raise ray.exceptions.ActorDiedError()

    async def fake_recycle(recycle_session, *, op, cause):
        assert recycle_session is session
        assert op == "forward_backward"
        return recovered_worker

    monkeypatch.setattr(engine, "_get_live_worker", fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_recycle_megatron_actor", fake_recycle)
    monkeypatch.setattr(engine, "_touch_actor", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    async def _run():
        await engine._run_worker_call_with_actor_recycle(
            session,
            op="forward_backward",
            submit_fn=lambda worker: worker,
        )

    with pytest.raises(RuntimeError, match="operation may have partially executed before the crash"):
        asyncio.run(_run())

    assert engine._poisoned_sessions[model_id].startswith(
        f"[{model_id}] megatron actor died during op=forward_backward"
    )


def test_issue_193_dense_recycle_fails_loud_after_dead_worker_during_forward(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_dense_dead_midcall"
    dead_worker = object()
    recovered_worker = object()
    engine._workers[model_id] = dead_worker
    engine._resource_pool_actor_names[model_id] = "dense-actor"

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_dense_dead_midcall",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="peft",
    )

    async def fake_get_live_worker(*args, **kwargs):
        return dead_worker

    async def fake_keepalive(awaitable, *_args, **_kwargs):
        raise ray.exceptions.ActorDiedError()

    async def fake_recycle(recycle_session, *, op, cause):
        assert recycle_session is session
        assert op == "forward"
        return recovered_worker

    monkeypatch.setattr(engine, "_get_live_worker", fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_recycle_dense_actor", fake_recycle)
    monkeypatch.setattr(engine, "_touch_actor", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    async def _run():
        await engine._run_worker_call_with_actor_recycle(
            session,
            op="forward",
            submit_fn=lambda worker: worker,
        )

    with pytest.raises(RuntimeError, match="dense actor recycle detected after op=forward"):
        asyncio.run(_run())

    assert model_id in engine._poisoned_sessions
    assert model_id in engine._poisoned_sessions


def test_issue_527_contaminated_session_blocks_request_before_restore():
    group_cls = MegatronWorkerGroup.__ray_actor_class__
    group = group_cls.__new__(group_cls)
    group._resolve_required_session_id = lambda session_id, op: session_id
    group._contaminated_sessions = {"session_target": "forward:group_timeout:600s"}

    with pytest.raises(RuntimeError, match="contaminated"):
        group._ensure_session_for_request(
            op="forward",
            session_id="session_target",
            traceparent=None,
            train_attn=None,
            train_mlp=None,
            train_unembed=None,
        )


def test_issue_527_ensure_failure_marks_session_contaminated():
    group_cls = MegatronWorkerGroup.__ray_actor_class__
    group = group_cls.__new__(group_cls)
    group._resolve_required_session_id = lambda session_id, op: session_id
    group._ensure_session_loaded = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("swap failed"))

    with pytest.raises(ValueError, match="swap failed"):
        group._ensure_session_for_request(
            op="optim_step",
            session_id="session_target",
            traceparent=None,
            train_attn=None,
            train_mlp=None,
            train_unembed=None,
        )

    assert "session_target" in group._contaminated_sessions
    assert "optim_step:ensure_session_loaded:ValueError" in group._contaminated_sessions["session_target"]


def test_issue_527_trusted_pair_mismatch_blocks_request(monkeypatch):
    group_cls = MegatronWorkerGroup.__ray_actor_class__
    group = group_cls.__new__(group_cls)
    group._resolve_required_session_id = lambda session_id, op: session_id
    group._ensure_session_loaded = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("trusted pair mismatch must fail before restore")
    )
    group._session_manager = SimpleNamespace(
        get_external_checkpoint=lambda session_id: {
            "checkpoint_path": "/tmp/external",
            "checkpoint_identity": "identity-a",
            "is_fresh": True,
            "invalidated_at": None,
            "invalidated_reason": None,
        },
        get_trusted_recovery_baseline=lambda session_id: {
            "checkpoint_path": "/tmp/baseline",
            "checkpoint_identity": "identity-b",
            "is_fresh": True,
            "invalidated_at": None,
            "invalidated_reason": None,
        },
    )

    with pytest.raises(RuntimeError, match="trusted pair checkpoint_identity mismatch"):
        group._ensure_session_for_request(
            op="forward_backward",
            session_id="session_target",
            traceparent=None,
            train_attn=None,
            train_mlp=None,
            train_unembed=None,
        )

    assert group._blocked_sessions["session_target"].startswith("trusted_pair_mismatch")


def test_issue_527_trusted_pair_strict_mode_requires_counterpart_marker(monkeypatch):
    monkeypatch.setenv("MINT_MEGATRON_ENFORCE_TRUSTED_PAIR", "1")
    group_cls = MegatronWorkerGroup.__ray_actor_class__
    group = group_cls.__new__(group_cls)
    group._resolve_required_session_id = lambda session_id, op: session_id
    group._ensure_session_loaded = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("strict trusted pair should fail before restore")
    )
    group._session_manager = SimpleNamespace(
        get_external_checkpoint=lambda session_id: {
            "checkpoint_path": "/tmp/external",
            "checkpoint_identity": "identity-a",
            "is_fresh": True,
            "invalidated_at": None,
            "invalidated_reason": None,
        },
        get_trusted_recovery_baseline=lambda session_id: None,
    )

    with pytest.raises(RuntimeError, match="strict trusted-pair enforcement"):
        group._ensure_session_for_request(
            op="forward",
            session_id="session_target",
            traceparent=None,
            train_attn=None,
            train_mlp=None,
            train_unembed=None,
        )


def test_issue_527_trusted_recovery_baseline_marker_roundtrip(tmp_path: Path):
    manager = MegatronSessionStateManager(base_path=str(tmp_path / "sessions"))
    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir()
    (ckpt_dir / "mp_rank_00_adapter.pt").write_text("adapter", encoding="utf-8")

    identity = manager.checkpoint_identity(str(ckpt_dir))
    session_id = "session_issue_527_baseline"
    manager.mark_trusted_recovery_baseline(
        session_id,
        checkpoint_path=str(ckpt_dir),
        checkpoint_identity=identity,
        reason="unit_test",
        actor_name="megatron_qwen3_30b_a3b_instruct_2507",
    )

    fresh = manager.get_trusted_recovery_baseline(session_id)
    assert fresh is not None
    assert fresh["checkpoint_identity"] == identity
    assert fresh["is_fresh"] is True

    stale = manager.invalidate_trusted_recovery_baseline(session_id, reason="after_train_step")
    assert stale is not None
    assert stale["is_fresh"] is False
    assert stale["invalidated_reason"] == "after_train_step"

    manager.clear_trusted_recovery_baseline(session_id)
    assert manager.get_trusted_recovery_baseline(session_id) is None


def test_issue_527_group_timeout_marks_session_contaminated(monkeypatch):
    group_cls = MegatronWorkerGroup.__ray_actor_class__
    group = group_cls.__new__(group_cls)
    group.workers = [object(), object()]

    def _raise_timeout(*_args, **_kwargs):
        raise ray.exceptions.GetTimeoutError("timeout")

    monkeypatch.setattr(ray, "get", _raise_timeout)

    with pytest.raises(RuntimeError, match="timed out"):
        group._ray_get_group_results(
            [object()],
            op="forward_backward",
            session_id="session_target",
            timeout_s=10,
        )

    assert "session_target" in group._contaminated_sessions
    assert "forward_backward:group_timeout:10s" in group._contaminated_sessions["session_target"]


def test_issue_527_invalidate_session_durability_invalidates_external_and_baseline_markers():
    group_cls = MegatronWorkerGroup.__ray_actor_class__
    group = group_cls.__new__(group_cls)
    calls: list[tuple[str, str, str]] = []

    group._session_manager = SimpleNamespace(
        invalidate_external_checkpoint=lambda session_id, reason: calls.append(("external", session_id, reason))
        or {"is_fresh": False, "invalidated_reason": reason},
        invalidate_trusted_recovery_baseline=lambda session_id, reason: calls.append(("baseline", session_id, reason)),
    )

    marker = group._invalidate_session_durability("session_target", reason="test_reason")

    assert marker is not None
    assert ("external", "session_target", "test_reason") in calls
    assert ("baseline", "session_target", "test_reason") in calls


def test_issue_527_get_session_guard_state_reports_markers_and_latches():
    group_cls = MegatronWorkerGroup.__ray_actor_class__
    group = group_cls.__new__(group_cls)
    group._contaminated_sessions = {"session_target": "forward_backward:group_timeout:600s"}
    group._blocked_sessions = {"session_target": "trusted_pair_mismatch:deadbeef!=cafebabe"}
    group._session_manager = SimpleNamespace(
        get_external_checkpoint=lambda session_id: {
            "checkpoint_path": "/tmp/external",
            "checkpoint_identity": "identity-a",
            "is_fresh": True,
        },
        get_trusted_recovery_baseline=lambda session_id: {
            "checkpoint_path": "/tmp/baseline",
            "checkpoint_identity": "identity-a",
            "is_fresh": True,
        },
    )

    state = group.get_session_guard_state("session_target")

    assert state["session_id"] == "session_target"
    assert state["contaminated"] is True
    assert state["blocked"] is True
    assert state["external_checkpoint"]["checkpoint_identity"] == "identity-a"
    assert state["trusted_recovery_baseline"]["checkpoint_identity"] == "identity-a"


