from types import MethodType, SimpleNamespace

from tinker_server.backend.megatron_distributed import MegatronWorkerGroup


def test_train_step_keeps_durability_fresh_until_optim_step() -> None:
    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)

    group._bind_traceparent = MethodType(lambda self, traceparent: None, group)
    group._resolve_required_session_id = MethodType(
        lambda self, session_id, op: "session-a",
        group,
    )

    calls: dict[str, object] = {}

    def _forward_backward(
        self,
        data_items,
        loss_fn="cross_entropy",
        loss_fn_config=None,
        rollout_correction_config=None,
        session_id=None,
        reset_bias=None,
        traceparent=None,
        *,
        train_attn=None,
        train_mlp=None,
        train_unembed=None,
        invalidate_durability=True,
    ):
        calls["invalidate_durability"] = invalidate_durability
        calls["forward_session_id"] = session_id
        return {"metrics": {"fb": 1.0}}

    def _optim_step(
        self,
        learning_rate,
        session_id=None,
        traceparent=None,
        *,
        train_attn=None,
        train_mlp=None,
        train_unembed=None,
    ):
        calls["optim_session_id"] = session_id
        return {"metrics": {"opt": 2.0}}

    group.forward_backward = MethodType(_forward_backward, group)
    group.optim_step = MethodType(_optim_step, group)
    group.learning_rate = 1e-4

    out = group.train_step(
        data_items=[{"dummy": True}],
        session_id="session-a",
    )

    assert calls["invalidate_durability"] is False
    assert calls["forward_session_id"] == "session-a"
    assert calls["optim_session_id"] == "session-a"
    assert out["metrics"]["fb"] == 1.0
    assert out["metrics"]["opt"] == 2.0


def test_save_checkpoint_does_not_require_fresh_trusted_pair() -> None:
    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)

    group._bind_traceparent = MethodType(lambda self, traceparent: None, group)
    group._resolve_required_session_id = MethodType(lambda self, session_id, op: "session-a", group)
    group._assert_session_request_allowed = MethodType(lambda self, session_id, op: None, group)
    group._validate_trusted_pair_for_request = MethodType(
        lambda self, session_id, op: (_ for _ in ()).throw(AssertionError("should not be called")),
        group,
    )
    group._ensure_session_loaded = MethodType(
        lambda self, session_id, traceparent=None, train_attn=None, train_mlp=None, train_unembed=None: {
            "switched": False
        },
        group,
    )
    group._ray_get_group_results = MethodType(
        lambda self, futures, op, session_id=None, timeout_s=None: [{"current_step": 3}],
        group,
    )
    group._step_count = 3
    group._actual_rank = 16
    group.base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    class _RemoteCall:
        def remote(self, *args, **kwargs):
            return "fake-ref"

    group.workers = [SimpleNamespace(save_checkpoint=_RemoteCall())]
    group._session_manager = SimpleNamespace(
        checkpoint_identity=lambda path: "ckpt-id",
        mark_external_checkpoint=lambda *args, **kwargs: None,
        mark_trusted_recovery_baseline=lambda *args, **kwargs: None,
    )

    out = group.save_checkpoint("/tmp/ckpt", session_id="session-a")

    assert out["current_step"] == 3
