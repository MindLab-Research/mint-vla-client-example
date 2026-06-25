"""Qwen3.6-27B TrainingWorker — single-GPU PEFT LoRA trainer.

This worker uses an **isolated** transformers v5 + peft >=0.19.0 PYTHONPATH
to load the Qwen3Next (model_type=qwen3_5) architecture that transformers 4.x
does not recognize.  All public method signatures are identical to
:class:`TrainingWorker` so that ``VerlTrainingEngine`` can route Qwen3.6
sessions transparently.

The model (27B bf16 ≈ 54 GB) fits on a single A800 80 GB GPU with
gradient checkpointing, so no distributed training framework (FSDP2, DDP)
is needed.  FSDP2 can be added later by wrapping ``self.model`` with
``torch.distributed.fsdp._fully_shard`` without changing the interface.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import ray
import structlog

from mint_server.observability.logging_context import (
    get_request_id,
    init_actor_observability,
    record_span_event_otel,
    restore_trace_id_from_traceparent,
)
from mint_server.backend.training.verl.training_worker_contract import TrainingWorkerInputContract

logger = structlog.get_logger(__name__)

# Default idle timeout — 0 means disabled (lifecycle managed by session/admin).
DEFAULT_IDLE_TIMEOUT = 0

# LoRA target modules for Qwen3Next architecture.
# Full-attention layers have: q_proj, k_proj, v_proj, o_proj (with q_norm/k_norm)
# Linear-attention layers (GatedDeltaNet) have: in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj
# MLP layers have: gate_proj, up_proj, down_proj (same as standard Qwen)
QWEN36_LORA_TARGET_MODULES = [
    # standard attention
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    # linear attention (GatedDeltaNet)
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    # MLP
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _get_torch():
    import torch
    return torch


@ray.remote(num_gpus=2)
class Qwen36TrainingWorker:
    """Single-GPU PEFT LoRA trainer for Qwen3.6-27B.

    Identical public interface to :class:`TrainingWorker`:
        forward_backward, optim_step, save_lora_weights, reinit_lora_weights,
        heartbeat, get_session_info, shutdown.

    Uses isolated transformers v5 + peft >=0.19.0 via PYTHONPATH so that
    the ``qwen3_5`` model type is recognized.
    """

    def __init__(
        self,
        base_model: str,
        lora_rank: int,
        learning_rate: float,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        session_state_root: str = "/tmp/mint_sessions",
    ):
        init_actor_observability()
        torch = _get_torch()
        self.device = torch.device("cuda")
        self._base_model = base_model

        self._last_activity = time.time()
        self._idle_timeout = idle_timeout
        self._shutdown_requested = False
        self._idle_watchdog_task: asyncio.Task | None = None

        logger.info("qwen36_loading_model", model=base_model, lora_rank=lora_rank)

        # --- Model loading (transformers v5 path) ---
        # Qwen3_5ForCausalLM is the text-only causal LM wrapper.
        # It avoids pulling in vision components.
        #
        # Weight key remapping: checkpoints saved from Qwen3_5ForConditionalGeneration
        # use prefix "model.language_model.layers.X", but Qwen3_5ForCausalLM (which wraps
        # Qwen3_5TextModel) expects "model.layers.X".  We pass the text_config directly so
        # AutoModelForCausalLM picks Qwen3_5ForCausalLM, and install a _load_pretrained_model
        # hook to strip the "language_model." prefix from checkpoint keys.
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        raw_cfg = AutoConfig.from_pretrained(
            base_model,
            trust_remote_code=True,
            local_files_only=True,
        )
        text_cfg = getattr(raw_cfg, "text_config", raw_cfg)

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            config=text_cfg,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
            device_map="auto",
        )

        # Disable KV cache for training
        if hasattr(self.model, "config") and getattr(self.model.config, "use_cache", None):
            self.model.config.use_cache = False

        # Enable gradient checkpointing (required for 27B on 80 GB)
        self.model.gradient_checkpointing_enable()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        logger.info("qwen36_gradient_checkpointing_enabled")

        # --- PEFT LoRA ---
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=QWEN36_LORA_TARGET_MODULES,
            lora_dropout=0.0,
            bias="none",
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        # --- Optimizer ---
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
        )

        self._step_count = 0
        self.max_lora_rank = int(lora_rank)
        self._current_actual_rank: int | None = None
        self._current_session_id: str | None = None
        self._input_contract = self._make_input_contract()

        # Session state manager (same as TrainingWorker)
        from mint_server.backend.training.verl.verl_training import SessionStateManager
        self._state_manager = SessionStateManager(base_path=session_state_root)

        logger.info("qwen36_worker_ready", model=base_model, rank=lora_rank)

    # ------------------------------------------------------------------
    # Public interface (identical to TrainingWorker)
    # ------------------------------------------------------------------

    def _touch(self) -> None:
        self._last_activity = time.time()

    def _bind_traceparent(self, traceparent: str | None) -> None:
        if isinstance(traceparent, str) and traceparent:
            restore_trace_id_from_traceparent(traceparent)

    def _resolve_actual_rank(self, actual_rank: int | None = None) -> int:
        rank = self._current_actual_rank if actual_rank is None else int(actual_rank)
        if rank is None:
            rank = self.max_lora_rank
        if rank <= 0 or rank > self.max_lora_rank:
            raise ValueError(f"actual_rank {rank} must be in [1, {self.max_lora_rank}]")
        return int(rank)

    def _zero_lora_rank_tail(self, actual_rank: int | None = None, *, zero_grads: bool = True) -> dict[str, int]:
        from mint_server.backend.inference.lora_utils import zero_lora_rank_tail_named_parameters

        effective_rank = self._resolve_actual_rank(actual_rank)
        stats = zero_lora_rank_tail_named_parameters(
            self.model.named_parameters(),
            actual_rank=effective_rank,
            trainer_rank=self.max_lora_rank,
            zero_grads=zero_grads,
        )
        self._current_actual_rank = effective_rank
        return stats

    def _ensure_session_loaded(self, session_id: str, actual_rank: int | None = None) -> None:
        if self._current_session_id == session_id:
            self._zero_lora_rank_tail(actual_rank, zero_grads=True)
            return

        # Save outgoing session
        if self._current_session_id is not None:
            lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 1e-4
            self._state_manager.save_state(
                self._current_session_id, self.model, self.optimizer,
                self._step_count, lr, self.device, save_gradients=True,
                actual_rank=self._current_actual_rank,
            )

        # Load new session
        if self._state_manager.session_exists(session_id):
            meta = self._state_manager.load_state(
                session_id, self.model, self.optimizer, self.device, load_gradients=True
            )
            self._step_count = meta.get("current_step", 0)
            actual_rank = actual_rank if actual_rank is not None else meta.get("actual_rank")
            if "learning_rate" in meta:
                for pg in self.optimizer.param_groups:
                    pg["lr"] = meta["learning_rate"]
            self._zero_lora_rank_tail(actual_rank, zero_grads=True)
        else:
            self.reinit_lora_weights(learning_rate=None, actual_rank=actual_rank)
            self.optimizer.zero_grad()

        self._current_session_id = session_id

    def _save_session_state(self, session_id: str, actual_rank: int | None = None) -> None:
        lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 1e-4
        actual_rank = self._resolve_actual_rank(actual_rank)
        self._zero_lora_rank_tail(actual_rank, zero_grads=True)
        self._state_manager.save_state(
            session_id, self.model, self.optimizer, self._step_count, lr, self.device,
            save_gradients=False,
            actual_rank=actual_rank,
        )

    def forward_backward(
        self,
        data_items: list[dict],
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict | None = None,
        session_id: str | None = None,
        actual_rank: int | None = None,
        traceparent: str | None = None,
    ) -> dict:
        """Forward + backward pass. Same contract as TrainingWorker.forward_backward."""
        self._bind_traceparent(traceparent)
        torch = _get_torch()
        self._touch()

        if session_id:
            self._ensure_session_loaded(session_id, actual_rank=actual_rank)
        elif actual_rank is not None:
            self._zero_lora_rank_tail(actual_rank, zero_grads=True)

        loss_fn_config = loss_fn_config or {}
        self.model.train()

        total_loss = 0.0
        total_tokens = 0
        loss_fn_outputs = []
        total_ratio = 0.0
        total_clipfrac = 0.0
        num_rl_samples = 0

        for item in data_items:
            model_input = item.get("model_input", {})
            loss_fn_inputs = item.get("loss_fn_inputs", {})

            chunks = model_input.get("chunks", [])
            if chunks:
                input_ids = []
                for chunk in chunks:
                    if chunk.get("type") == "encoded_text" and "tokens" in chunk:
                        input_ids.extend(chunk["tokens"])
                if not input_ids:
                    continue
            else:
                continue

            target_data = loss_fn_inputs.get("target_tokens", {})
            weights_data = loss_fn_inputs.get("weights") or loss_fn_inputs.get("loss_mask") or loss_fn_inputs.get("mask", {})

            target_tokens = target_data.get("data", [])
            weights = weights_data.get("data", []) if weights_data else []

            if not weights and loss_fn in ("importance_sampling", "ppo"):
                advantages_data = loss_fn_inputs.get("advantages", {})
                advantages = advantages_data.get("data", [])
                if advantages:
                    weights = [1.0 if a != 0 else 0.0 for a in advantages]

            if not target_tokens or not weights:
                continue

            input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            target_ids_t = torch.tensor([target_tokens], dtype=torch.long, device=self.device)
            weights_t = torch.tensor([weights], dtype=torch.float32, device=self.device)

            outputs = self.model(input_ids=input_ids_t)
            logits = outputs.logits

            logits_flat = logits.squeeze(0)
            targets_flat = target_ids_t.squeeze(0)
            weights_flat = weights_t.squeeze(0)

            log_probs = torch.nn.functional.log_softmax(logits_flat, dim=-1)
            target_logprobs = torch.gather(
                log_probs, dim=-1, index=targets_flat.unsqueeze(-1)
            ).squeeze(-1)

            token_count = float((weights_flat != 0).sum().item())

            if loss_fn == "cross_entropy":
                loss = -(target_logprobs * weights_flat).sum()
                loss.backward()
                item_loss = loss.item()
                logprobs_list = target_logprobs.detach().tolist()
                loss_fn_outputs.append({
                    "loss": {"data": [item_loss], "shape": [1], "dtype": "float32"},
                    "logprobs": {"data": logprobs_list, "shape": [len(logprobs_list)], "dtype": "float32"},
                })

            elif loss_fn in ("importance_sampling", "ppo"):
                old_logprobs_data = loss_fn_inputs.get("logprobs", {})
                advantages_data = loss_fn_inputs.get("advantages", {})
                old_logprobs = old_logprobs_data.get("data", [])
                advantages = advantages_data.get("data", [])

                if not old_logprobs or not advantages:
                    continue

                old_logprobs_t = torch.tensor([old_logprobs], dtype=torch.float32, device=self.device).squeeze(0)
                advantages_t = torch.tensor([advantages], dtype=torch.float32, device=self.device).squeeze(0)

                epsilon = float(loss_fn_config.get("epsilon", 0.2))
                clip_low = float(loss_fn_config.get("clip_low", 1.0 - epsilon))
                clip_high = float(loss_fn_config.get("clip_high", 1.0 + epsilon))

                # Clamp log ratio for numerical stability (matches TrainingWorker)
                log_ratio = target_logprobs - old_logprobs_t
                log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
                ratio = torch.exp(log_ratio)

                if loss_fn == "importance_sampling":
                    loss = (-ratio * advantages_t * weights_flat).sum()
                else:  # ppo
                    # PPO clipped objective: max(-A*r, -A*clip(r))
                    pg_loss1 = -advantages_t * ratio
                    clipped_ratio = torch.clamp(ratio, clip_low, clip_high)
                    pg_loss2 = -advantages_t * clipped_ratio
                    loss = (torch.maximum(pg_loss1, pg_loss2) * weights_flat).sum()

                    # Masked clip fraction
                    clipped = ((ratio < clip_low) | (ratio > clip_high)).float() * weights_flat
                    clipfrac = clipped.sum() / max(token_count, 1.0)

                loss.backward()
                item_loss = loss.item()
                logprobs_list = target_logprobs.detach().tolist()

                loss_fn_outputs.append({
                    "loss": {"data": [item_loss], "shape": [1], "dtype": "float32"},
                    "logprobs": {"data": logprobs_list, "shape": [len(logprobs_list)], "dtype": "float32"},
                })

                # Masked ratio mean
                denom = max(token_count, 1.0)
                masked_ratio = (ratio * weights_flat).sum() / denom
                total_ratio += masked_ratio.item()
                if loss_fn == "ppo":
                    total_clipfrac += clipfrac.item()
                num_rl_samples += 1

            else:
                raise ValueError(f"Unknown loss_fn: {loss_fn}")

            total_loss += item_loss
            total_tokens += int(token_count)

        # Zero rank tail gradients after all backward passes (matches TrainingWorker)
        self._zero_lora_rank_tail(actual_rank, zero_grads=True)

        avg_loss = total_loss / max(total_tokens, 1)
        metrics: dict[str, Any] = {
            "loss:mean": avg_loss,
            "num_samples:sum": float(len(data_items)),
            "num_tokens:sum": float(total_tokens),
        }
        if num_rl_samples > 0:
            metrics["ratio:mean"] = total_ratio / num_rl_samples
            if loss_fn == "ppo":
                metrics["clipfrac:mean"] = total_clipfrac / num_rl_samples

        return {
            "loss_fn_output_type": f"{loss_fn}_loss",
            "loss_fn_outputs": loss_fn_outputs,
            "metrics": metrics,
        }

    def optim_step(
        self,
        learning_rate: float | None,
        session_id: str | None = None,
        actual_rank: int | None = None,
        traceparent: str | None = None,
    ) -> dict:
        self._bind_traceparent(traceparent)
        torch = _get_torch()
        self._touch()

        if session_id:
            self._ensure_session_loaded(session_id, actual_rank=actual_rank)
        elif actual_rank is not None:
            self._zero_lora_rank_tail(actual_rank, zero_grads=True)

        if learning_rate is not None:
            for pg in self.optimizer.param_groups:
                pg["lr"] = learning_rate

        self._zero_lora_rank_tail(actual_rank, zero_grads=True)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self._zero_lora_rank_tail(actual_rank, zero_grads=True)
        self.optimizer.zero_grad()

        self._step_count += 1

        if session_id:
            self._save_session_state(session_id, actual_rank=actual_rank)

        logger.info("qwen36_optim_step", grad_norm=float(grad_norm), step=self._step_count)

        return {
            "metrics": {"grad_norm:last": float(grad_norm)},
            "type": "optim_step",
        }


    def forward(
        self,
        data_items: list[dict],
        session_id: str | None = None,
        traceparent: str | None = None,
    ) -> dict:
        """Forward pass only (no backward). Returns logprobs.

        Mirrors :meth:`TrainingWorker.forward` so that the ``/api/v1/forward``
        route works for Qwen3.6 sessions.
        """
        self._bind_traceparent(traceparent)
        torch = _get_torch()
        self._touch()

        if session_id:
            self._ensure_session_loaded(session_id)

        prev_training = self.model.training
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0
        loss_fn_outputs = []

        try:
            with torch.no_grad():
                for item in data_items:
                    model_input = item.get("model_input", {})
                    loss_fn_inputs = item.get("loss_fn_inputs", {})

                    chunks = model_input.get("chunks", [])
                    if chunks:
                        input_ids = []
                        for chunk in chunks:
                            if chunk.get("type") == "encoded_text" and "tokens" in chunk:
                                input_ids.extend(chunk["tokens"])
                        if not input_ids:
                            continue
                    else:
                        continue

                    target_data = loss_fn_inputs.get("target_tokens", {})
                    weights_data = (
                        loss_fn_inputs.get("weights")
                        or loss_fn_inputs.get("loss_mask")
                        or loss_fn_inputs.get("mask", {})
                    )

                    target_tokens = target_data.get("data", [])
                    weights = weights_data.get("data", []) if weights_data else []

                    if not target_tokens:
                        # For noop_loss / forward-only without target tokens,
                        # compute logprobs of the input tokens themselves.
                        target_tokens = input_ids[1:]
                        input_ids = input_ids[:-1]
                        if not target_tokens:
                            continue
                        weights = [1.0] * len(target_tokens)

                    if not weights:
                        weights = [1.0] * len(target_tokens)

                    input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)
                    target_ids_t = torch.tensor([target_tokens], dtype=torch.long, device=self.device)
                    weights_t = torch.tensor([weights], dtype=torch.float32, device=self.device)

                    outputs = self.model(input_ids=input_ids_t)
                    logits = outputs.logits

                    logits_flat = logits.squeeze(0)
                    targets_flat = target_ids_t.squeeze(0)
                    weights_flat = weights_t.squeeze(0)

                    log_probs = torch.nn.functional.log_softmax(logits_flat, dim=-1)
                    target_logprobs = torch.gather(
                        log_probs, dim=-1, index=targets_flat.unsqueeze(-1)
                    ).squeeze(-1)

                    ce_loss = torch.nn.functional.cross_entropy(
                        logits_flat, targets_flat, reduction="none"
                    )
                    weighted_loss = ce_loss * weights_flat
                    num_weighted = weights_flat.sum()
                    if num_weighted > 0:
                        loss = weighted_loss.sum() / num_weighted
                    else:
                        loss = weighted_loss.sum()

                    item_loss = loss.item()
                    total_loss += item_loss * num_weighted.item()
                    total_tokens += num_weighted.item()

                    logprobs_list = target_logprobs.tolist()
                    loss_fn_outputs.append({
                        "loss": {"data": [item_loss], "shape": [1], "dtype": "float32"},
                        "logprobs": {"data": logprobs_list, "shape": [len(logprobs_list)], "dtype": "float32"},
                    })
        finally:
            if prev_training:
                self.model.train()
            else:
                self.model.eval()

        avg_loss = total_loss / max(total_tokens, 1)

        logger.info("forward", loss=avg_loss, tokens=total_tokens)

        return {
            "loss_fn_output_type": "sft_loss",
            "loss_fn_outputs": loss_fn_outputs,
            "metrics": {
                "loss:mean": avg_loss,
                "num_samples:sum": float(len(data_items)),
                "num_tokens:sum": float(total_tokens),
            },
        }

    def get_lora_state_dict(self) -> dict[str, Any]:
        """Extract LoRA adapter weights as state dict (on CPU)."""
        self._touch()
        from peft.utils.save_and_load import get_peft_model_state_dict
        state_dict = get_peft_model_state_dict(self.model)
        return {k: v.cpu() for k, v in state_dict.items()}

    def get_lora_config(self, actual_rank: int | None = None) -> dict:
        """Get LoRA configuration as dict compatible with vLLM's PEFTHelper."""
        peft_config = self.model.peft_config.get("default")
        effective_rank = self._resolve_actual_rank(actual_rank)
        assert peft_config is not None
        alpha_per_rank = peft_config.lora_alpha / peft_config.r
        return {
            "r": effective_rank,
            "lora_alpha": int(alpha_per_rank * effective_rank),
            "lora_dropout": peft_config.lora_dropout,
            "target_modules": list(peft_config.target_modules),
            "bias": peft_config.bias,
            "task_type": peft_config.task_type.value if peft_config.task_type else None,
            "peft_type": "LORA",
            "base_model_name_or_path": self._base_model,
        }

    def save_lora_weights(
        self,
        save_path: str,
        traceparent: str | None = None,
        *,
        session_id: str | None = None,
        actual_rank: int | None = None,
    ) -> dict:
        """Save LoRA adapter to directory. Same format as TrainingWorker."""
        self._bind_traceparent(traceparent)
        self._touch()
        if session_id is not None:
            self._ensure_session_loaded(session_id, actual_rank=actual_rank)

        from safetensors.torch import save_file
        from mint_server.backend.inference.lora_utils import truncate_lora_state_dict

        os.makedirs(save_path, exist_ok=True)
        effective_rank = self._resolve_actual_rank(actual_rank)
        self._zero_lora_rank_tail(effective_rank, zero_grads=True)

        state_dict = self.get_lora_state_dict()
        state_dict = truncate_lora_state_dict(state_dict, self.max_lora_rank, effective_rank)
        save_file(state_dict, os.path.join(save_path, "adapter_model.safetensors"))

        config = self.get_lora_config(effective_rank)
        with open(os.path.join(save_path, "adapter_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        abs_path = os.path.abspath(save_path)
        logger.info("qwen36_saved_lora_weights", path=abs_path)
        return {
            "path": abs_path,
            "state_dict": state_dict,
            "peft_config": config,
            "current_step": self._step_count,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def reinit_lora_weights(
        self,
        learning_rate: float | None = None,
        actual_rank: int | None = None,
        traceparent: str | None = None,
    ) -> dict:
        """Reinitialize LoRA weights AND optimizer state for fresh session."""
        self._bind_traceparent(traceparent)
        import torch.nn.init as init

        reinit_count = 0
        lr_updated = False

        for name, param in self.model.named_parameters():
            name_lower = name.lower()
            if "lora" not in name_lower:
                continue
            if not param.requires_grad:
                continue

            is_lora_a = "lora_a" in name_lower
            is_lora_b = "lora_b" in name_lower

            if is_lora_a:
                init.xavier_uniform_(param.data)
                reinit_count += 1
            elif is_lora_b:
                init.zeros_(param.data)
                reinit_count += 1

        if learning_rate is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
            lr_updated = True
            logger.info("qwen36_set_lr", lr=learning_rate)

        self.optimizer.zero_grad()
        tail_zeroed = self._zero_lora_rank_tail(actual_rank, zero_grads=True)

        opt_state_reset = len(self.optimizer.state)
        self.optimizer.state.clear()

        self._step_count = 0

        logger.info("qwen36_reinit_lora", count=reinit_count)
        return {
            "status": "ok",
            "reinit_count": reinit_count,
            "tail_zeroed": tail_zeroed,
            "opt_state_reset": opt_state_reset,
            "lr_updated": lr_updated,
            "learning_rate": learning_rate,
            "actual_rank": self._resolve_actual_rank(actual_rank),
        }

    async def heartbeat(self) -> dict:
        self._touch()
        if self._idle_watchdog_task is None and self._idle_timeout > 0:
            self._idle_watchdog_task = asyncio.create_task(self._idle_watchdog_async())
        return {
            "idle_timeout": self._idle_timeout,
            "time_until_timeout": max(0, self._idle_timeout - (time.time() - self._last_activity)),
            "max_lora_rank": self.max_lora_rank,
            "current_actual_rank": self._current_actual_rank,
        }

    async def _idle_watchdog_async(self) -> None:
        check_interval = 30
        while not self._shutdown_requested:
            await asyncio.sleep(check_interval)
            if self._shutdown_requested:
                break
            idle_time = time.time() - self._last_activity
            if idle_time >= self._idle_timeout:
                logger.warning("qwen36_idle_timeout", idle_time=idle_time)
                try:
                    self.shutdown()
                except Exception:
                    logger.error("qwen36_shutdown_error")
                ray.actor.exit_actor()
                return

    def get_tokenizer_info(self) -> dict:
        """Return tokenizer configuration for client use.

        Mirrors :meth:`TrainingWorker.get_tokenizer_info` so that
        ``VerlTrainingEngine.get_tokenizer_info`` works for Qwen3.6 sessions.
        """
        return {
            "vocab_size": self.tokenizer.vocab_size,
            "model_max_length": self.tokenizer.model_max_length,
            "pad_token": self.tokenizer.pad_token,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token": self.tokenizer.eos_token,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token": self.tokenizer.bos_token,
            "bos_token_id": self.tokenizer.bos_token_id,
            "unk_token": self.tokenizer.unk_token,
            "unk_token_id": self.tokenizer.unk_token_id,
        }

    def get_session_info(self) -> dict:
        lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else None
        peft_config = self.model.peft_config.get("default")
        lora_rank = peft_config.r if peft_config else None
        return {
            "learning_rate": lr,
            "lora_rank": lora_rank,
            "step_count": self._step_count,
            "device": str(self.device),
        }

    def get_rss_bytes(self) -> int:
        with open("/proc/self/statm", encoding="utf-8") as f:
            parts = f.read().strip().split()
        if len(parts) < 2:
            raise ValueError(f"unexpected /proc/self/statm format: {parts!r}")
        rss_pages = int(parts[1])
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return rss_pages * page_size

    def shutdown(self) -> None:
        torch = _get_torch()
        logger.info("qwen36_shutting_down")
        self._shutdown_requested = True
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "optimizer"):
            del self.optimizer
        torch.cuda.empty_cache()
        try:
            ray.actor.exit_actor()
        except Exception:
            pass

    def _make_input_contract(self) -> TrainingWorkerInputContract:
        from mint_server.backend.observability.runtime_observability import runtime_observability
        return TrainingWorkerInputContract(
            base_model=lambda: str(getattr(self, "_base_model", "unknown") or "unknown"),
            vocab_size=self._resolve_vocab_size,
            request_id=get_request_id,
            record_span_event=record_span_event_otel,
            record_training_incident=runtime_observability.record_training_incident,
        )

    def _resolve_vocab_size(self) -> int | None:
        config = getattr(self.model, "config", None)
        vocab_size = getattr(config, "vocab_size", None)
        if isinstance(vocab_size, int) and vocab_size > 0:
            return int(vocab_size)
        tokenizer_vocab = getattr(self.tokenizer, "vocab_size", None)
        if isinstance(tokenizer_vocab, int) and tokenizer_vocab > 0:
            return int(tokenizer_vocab)
        return None
