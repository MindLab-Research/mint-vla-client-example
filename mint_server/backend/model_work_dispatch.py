from __future__ import annotations

import json
import logging
import time
from typing import Any

from ..logging_context import run_async_with_otel_span

logger = logging.getLogger(__name__)

KNOWN_MODEL_WORK_OPS = (
    "sampling.asample",
    "sampling.compute_logprobs",
    "training.create_model",
    "training.create_model_from_state",
    "training.train_step",
    "training.forward",
    "training.forward_backward",
    "training.save_weights_for_sampler",
    "training.optim_step",
    "training.reset_expert_bias",
    "training.delete_model",
    "weights.save_weights",
    "weights.save_state",
    "weights.load_state",
    "internal.noop",
    "mint.interpolate_checkpoints",
    "mint.forward_backward_reverse_kl",
    "mint.vla.train_step",
    "mint.action.act",
)


def _ensure_ray_context_for_dispatch() -> None:
    import ray

    if ray.is_initialized():
        return

    raise RuntimeError("Ray is not initialized")


async def execute_model_work_item(item: Any, *, component: str = "model_work_dispatch") -> None:
    _ensure_ray_context_for_dispatch()
    from ..models.mint_types import (
        ForwardBackwardReverseKLRequest,
        InterpolateCheckpointsRequest,
        VLATrainStepRequest,
    )
    from ..models.types import (
        ActRequest,
        ComputeLogprobsRequest,
        CreateModelFromStateRequest,
        CreateModelRequest,
        ForwardRequest,
        ForwardBackwardRequest,
        LoadStateRequest,
        OptimStepRequest,
        ResetExpertBiasRequest,
        SampleRequest,
        SaveStateRequest,
        SaveWeightsForSamplerRequest,
        TrainStepRequest,
    )
    from ..routes import action_sampling, mint, sampling, training, weights

    op = str(item.op)

    if op == "sampling.asample":
        async def _run():
            logger.info(
                "[%s] sampling.asample request_id=%s stage=before_model_validate",
                component,
                str(item.request_id),
            )
            req = SampleRequest.model_validate_json(item.request_json)
            logger.info(
                "[%s] sampling.asample request_id=%s stage=after_model_validate",
                component,
                str(item.request_id),
            )
            await sampling._do_sample(
                item.request_id,
                req,
                item.user_id,
                (item.extra or {}).get("gateway_auth"),
            )

        return await run_async_with_otel_span(
            "queue.stage.sampling.asample",
            _run,
            component=component,
            op=op,
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.sampling.asample"},
        )

    if op == "sampling.compute_logprobs":
        async def _run():
            req = ComputeLogprobsRequest.model_validate_json(item.request_json)
            await sampling._do_compute_logprobs(
                item.request_id,
                req,
                item.user_id,
                (item.extra or {}).get("gateway_auth"),
            )

        return await run_async_with_otel_span(
            "queue.stage.sampling.compute_logprobs",
            _run,
            component=component,
            op=op,
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.sampling.compute_logprobs"},
        )

    if op == "training.create_model":
        async def _run():
            req = CreateModelRequest.model_validate_json(item.request_json)
            await training._do_create_model(item.request_id, req, item.user_id, item.webhook_url)

        return await run_async_with_otel_span(
            "queue.stage.training.create_model",
            _run,
            component=component,
            op=op,
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.training.create_model"},
        )

    if op == "training.create_model_from_state":
        async def _run():
            req = CreateModelFromStateRequest.model_validate_json(item.request_json)
            await training._do_create_model_from_state(item.request_id, req, item.user_id)

        return await run_async_with_otel_span(
            "queue.stage.training.create_model_from_state",
            _run,
            component=component,
            op=op,
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.training.create_model_from_state"},
        )

    if op == "training.train_step":
        async def _run():
            req = TrainStepRequest.model_validate_json(item.request_json)
            await training._do_train_step(
                item.request_id,
                req,
                item.user_id,
                (item.extra or {}).get("gateway_auth"),
            )

        return await run_async_with_otel_span(
            "queue.stage.training.train_step",
            _run,
            component=component,
            op=op,
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.training.train_step"},
        )

    if op == "training.forward":
        async def _run():
            req = ForwardRequest.model_validate_json(item.request_json)
            await training._do_forward(
                item.request_id,
                req,
                (item.extra or {}).get("gateway_auth"),
            )

        return await run_async_with_otel_span(
            "queue.stage.training.forward",
            _run,
            component=component,
            op=op,
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.training.forward"},
        )

    if op == "training.forward_backward":
        async def _run():
            req = ForwardBackwardRequest.model_validate_json(item.request_json)
            await training._do_forward_backward(
                item.request_id,
                req,
                item.user_id,
                (item.extra or {}).get("gateway_auth"),
            )

        return await run_async_with_otel_span(
            "queue.stage.training.forward_backward",
            _run,
            component=component,
            op=op,
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.training.forward_backward"},
        )

    if op == "training.save_weights_for_sampler":
        async def _run():
            req = SaveWeightsForSamplerRequest.model_validate_json(item.request_json)
            prefer_tinker = bool((item.extra or {}).get("prefer_tinker"))
            is_admin = bool((item.extra or {}).get("is_admin"))
            await training._do_save_weights_for_sampler(
                item.request_id,
                req,
                item.user_id,
                prefer_tinker,
                is_admin,
            )

        return await run_async_with_otel_span(
            "queue.stage.training.save_weights_for_sampler",
            _run,
            component=component,
            op=op,
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.training.save_weights_for_sampler"},
        )

    if op == "training.optim_step":
        async def _run():
            req = OptimStepRequest.model_validate_json(item.request_json)
            await training._do_optim_step(item.request_id, req, item.user_id)

        return await run_async_with_otel_span(
            "queue.stage.training.optim_step",
            _run,
            component=component,
            op=op,
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.training.optim_step"},
        )

    if op == "training.reset_expert_bias":
        async def _run():
            req = ResetExpertBiasRequest.model_validate_json(item.request_json)
            await training._do_reset_expert_bias(item.request_id, req)

        return await run_async_with_otel_span(
            "queue.stage.training.reset_expert_bias",
            _run,
            component=component,
            op=op,
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.training.reset_expert_bias"},
        )

    if op == "training.delete_model":
        async def _run():
            payload = json.loads(item.request_json.decode("utf-8"))
            model_id = payload.get("model_id")
            if not isinstance(model_id, str) or not model_id:
                raise ValueError("training.delete_model missing model_id")
            await training._do_delete_model(item.request_id, model_id)

        return await run_async_with_otel_span(
            "queue.stage.training.delete_model",
            _run,
            component=component,
            op=op,
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.training.delete_model"},
        )

    if op == "weights.save_weights":
        async def _run():
            req = SaveStateRequest.model_validate_json(item.request_json)
            prefer_tinker = bool((item.extra or {}).get("prefer_tinker"))
            await weights._do_save_state(
                item.request_id,
                req,
                user_id=item.user_id,
                webhook_url=item.webhook_url,
                prefer_tinker=prefer_tinker,
            )

        return await run_async_with_otel_span(
            "queue.stage.weights.save_weights",
            _run,
            component=component,
            op=op,
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.weights.save_weights"},
        )

    if op == "weights.save_state":
        async def _run():
            req = SaveStateRequest.model_validate_json(item.request_json)
            prefer_tinker = bool((item.extra or {}).get("prefer_tinker"))
            await weights._do_save_state(
                item.request_id,
                req,
                user_id=item.user_id,
                webhook_url=item.webhook_url,
                prefer_tinker=prefer_tinker,
            )

        return await run_async_with_otel_span(
            "queue.stage.weights.save_state",
            _run,
            component=component,
            op=op,
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.weights.save_state"},
        )

    if op == "weights.load_state":
        async def _run():
            req = LoadStateRequest.model_validate_json(item.request_json)
            await weights._do_load_state(item.request_id, req, item.user_id)

        return await run_async_with_otel_span(
            "queue.stage.weights.load_state",
            _run,
            component=component,
            op=op,
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.weights.load_state"},
        )

    if op == "internal.noop":
        async def _run():
            from .task_state_store import task_futures

            await task_futures.async_resolve(
                str(item.request_id),
                {"ok": True, "op": "internal.noop", "ts": time.time()},
            )

        return await run_async_with_otel_span(
            "queue.stage.internal.noop",
            _run,
            component=component,
            op=op,
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.internal.noop"},
        )

    if op == "mint.interpolate_checkpoints":
        async def _run():
            req = InterpolateCheckpointsRequest.model_validate_json(item.request_json)
            await mint._do_interpolate_checkpoints(
                item.request_id,
                req,
                item.user_id,
                (item.extra or {}).get("gateway_auth"),
                (item.extra or {}).get("billing_observation_input"),
            )

        return await run_async_with_otel_span(
            "queue.stage.mint.interpolate_checkpoints",
            _run,
            component=component,
            op=op,
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.mint.interpolate_checkpoints"},
        )

    if op == "mint.forward_backward_reverse_kl":
        async def _run():
            req = ForwardBackwardReverseKLRequest.model_validate_json(item.request_json)
            await mint._do_forward_backward_reverse_kl(item.request_id, req, item.user_id)

        return await run_async_with_otel_span(
            "queue.stage.mint.forward_backward_reverse_kl",
            _run,
            component=component,
            op=op,
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.mint.forward_backward_reverse_kl"},
        )

    if op == "mint.vla.train_step":
        async def _run():
            req = VLATrainStepRequest.model_validate_json(item.request_json)
            await mint._do_vla_train_step(
                item.request_id,
                req,
                item.user_id,
                (item.extra or {}).get("gateway_auth"),
                (item.extra or {}).get("billing_observation_input"),
            )

        return await run_async_with_otel_span(
            "queue.stage.mint.vla.train_step",
            _run,
            component=component,
            op=op,
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.mint.vla.train_step"},
        )

    if op == "mint.action.act":
        async def _run():
            req = ActRequest.model_validate_json(item.request_json)
            await action_sampling._do_act(
                item.request_id,
                req,
                gateway_auth=(item.extra or {}).get("gateway_auth"),
                billing_observation_input=(item.extra or {}).get("billing_observation_input"),
            )

        return await run_async_with_otel_span(
            "queue.stage.mint.action.act",
            _run,
            component=component,
            op=op,
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.mint.action.act"},
        )

    raise KeyError(f"unknown queue op: {op}")


def register_model_work_executors(target: Any) -> None:
    _ensure_ray_context_for_dispatch()
    for op in KNOWN_MODEL_WORK_OPS:
        target.set_executor(op, execute_model_work_item)
