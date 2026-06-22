"""VerlTrainingEngine - SFT training with LoRA using Ray actors.

Each training session gets a dedicated TrainingWorker Ray actor with its own GPU.
"""

from __future__ import annotations

import asyncio
import json
import structlog
import os
import time
from typing import TYPE_CHECKING, Any, cast

import ray

import mint_server.backend.ray_cluster.ray_kill as ray_kill
from mint_server.backend.ray_cluster.async_ray_control import _ray_ref_to_future, _silence_late_result, async_get_ray_ref
from mint_server.observability.logging_context import (
    classify_failure_reason,
    get_current_traceparent,
    get_request_id,
    init_actor_observability,
    record_span_event_otel,
    restore_trace_id_from_traceparent,
    run_async_with_otel_span,
    start_as_current_span,
)
from mint_server.config import RAY_NAMESPACE
from mint_server.config import config as server_config
from mint_server.ray.ray_utils import init_ray
from mint_server.backend.core.training_backend_selection import (
    _DISTRIBUTED_MOE_BACKENDS,
    _is_qwen3_30b_model,
    _is_qwen35_model,
    _select_moe_training_backend,
    _uses_distributed_training_backend,
)
from mint_server.backend.training.verl.verl_actor_recycler import VerlActorRecycler
from mint_server.backend.training.verl.verl_checkpoint_ops import CheckpointOps
from mint_server.backend.training.verl.verl_engine_config import VerlEngineConfig
from mint_server.backend.training.verl.training_worker_contract import TrainingWorkerInputContract

torch = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from mint_server.backend.training.training_session_manager import TrainingSession

logger = structlog.get_logger(__name__)

# Default idle timeout for TrainingWorker (seconds). Lifecycle is controlled by
# session shutdown, explicit admin actions, and supervisor reconciliation.
DEFAULT_IDLE_TIMEOUT = 0

VERL_FSDP2_LORA_BACKEND = "verl_fsdp2_lora"


def _is_qwen36_model(model: str | None) -> bool:
    try:
        from mint_server.backend.qwen36_verl_fsdp2_lora import is_qwen36_model

        return is_qwen36_model(model)
    except Exception:
        return "qwen3.6-27b" in str(model or "").lower()


def _uses_verl_fsdp2_lora_backend(requested_model: str | None) -> bool:
    if _is_qwen36_model(requested_model):
        return True
    try:
        from mint_server.backend.core.model_registry import get_model_config

        return (
            str(getattr(get_model_config(requested_model or ""), "training_backend", "") or "")
            == VERL_FSDP2_LORA_BACKEND
        )
    except Exception:
        logger.debug("verl_fsdp2_lora_model_config_lookup_failed_for__s", exc_info=True)
    return False


# =====================================================================
# Session State Manager - Per-iteration state persistence for stateless trainers
# =====================================================================

def _get_torch():
    global torch
    if torch is None:
        import torch as _torch  # type: ignore

        torch = _torch
    return torch


class SessionStateManager:
    """Manages session state (LoRA weights + optimizer + gradients) for stateless trainers.

    Enables multiple sessions to share a single trainer by loading/saving
    state per iteration. Each session has its own checkpoint directory.

    Full session state tuple:
        - LoRA weights (theta): trainable adapter parameters
        - Gradients (grad theta): accumulated gradients for gradient accumulation
        - Optimizer state: Adam momentum (exp_avg) and variance (exp_avg_sq)

    Storage layout:
        {base_path}/{session_id}_checkpoint/
            adapter_model.safetensors  # LoRA weights
            gradients.pt               # Accumulated gradients (optional)
            optimizer.pt               # Adam state (exp_avg, exp_avg_sq)
            training_meta.json         # step count, learning_rate
    """

    def __init__(self, base_path: str = "/tmp/mint_sessions"):
        """Initialize the session state manager.

        Args:
            base_path: Root directory for all session checkpoints.
        """
        import os
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        logger.info("initialized_with", base_path=base_path)

    def get_session_path(self, session_id: str) -> str:
        """Get checkpoint directory path for a session."""
        import os
        return os.path.join(self.base_path, f"{session_id}_checkpoint")

    def session_exists(self, session_id: str) -> bool:
        """Check if a session has saved state."""
        import os
        session_path = self.get_session_path(session_id)
        adapter_path = os.path.join(session_path, "adapter_model.safetensors")
        return os.path.exists(adapter_path)

    def save_state(
        self,
        session_id: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        lr: float,
        device: torch.device,
        save_gradients: bool = True,
        actual_rank: int | None = None,
    ) -> str:
        """Save session state (LoRA weights + optimizer + gradients + metadata).

        Args:
            session_id: Unique session identifier.
            model: PEFT model with LoRA adapters.
            optimizer: Optimizer with state to save.
            step: Current training step.
            lr: Current learning rate.
            device: Device for tensor operations.
            save_gradients: If True, save accumulated gradients for later restoration.

        Returns:
            Absolute path to saved checkpoint.
        """
        import json
        import os

        torch = _get_torch()
        from peft.utils.save_and_load import get_peft_model_state_dict
        from safetensors.torch import save_file

        session_path = self.get_session_path(session_id)
        os.makedirs(session_path, exist_ok=True)

        # 1. Save LoRA weights
        state_dict = get_peft_model_state_dict(model)
        # Move to CPU for serialization
        state_dict = {k: v.cpu() for k, v in state_dict.items()}
        save_file(state_dict, os.path.join(session_path, "adapter_model.safetensors"))

        # 2. Save optimizer state
        torch.save(optimizer.state_dict(), os.path.join(session_path, "optimizer.pt"))

        # 3. Save gradients (for gradient accumulation across session switches)
        if save_gradients:
            grads = {}
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grads[name] = param.grad.cpu().clone()
            if grads:
                torch.save(grads, os.path.join(session_path, "gradients.pt"))
                logger.debug("saved__s_gradient_tensors_for__s")
            else:
                # Remove old gradients file if no gradients to save
                grads_path = os.path.join(session_path, "gradients.pt")
                if os.path.exists(grads_path):
                    os.remove(grads_path)

        # 4. Save metadata
        meta = {"current_step": step, "learning_rate": lr}
        if actual_rank is not None:
            meta["actual_rank"] = int(actual_rank)
        with open(os.path.join(session_path, "training_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        logger.debug("saved_state_for__s", step=session_id)
        return os.path.abspath(session_path)

    def load_state(
        self,
        session_id: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        load_gradients: bool = True,
    ) -> dict:
        """Load session state into model/optimizer.

        Args:
            session_id: Unique session identifier.
            model: PEFT model to load weights into.
            optimizer: Optimizer to load state into.
            device: Device for tensor operations.
            load_gradients: If True, restore accumulated gradients.

        Returns:
            Dict with metadata (current_step, learning_rate, has_gradients).

        Raises:
            FileNotFoundError: If session checkpoint doesn't exist.
        """
        import json
        import os

        torch = _get_torch()
        from peft.utils.save_and_load import set_peft_model_state_dict
        from safetensors.torch import load_file

        session_path = self.get_session_path(session_id)
        adapter_path = os.path.join(session_path, "adapter_model.safetensors")

        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Session {session_id} has no saved state")

        # 1. Load LoRA weights
        state_dict = load_file(adapter_path, device=str(device))
        set_peft_model_state_dict(model, state_dict)

        # 2. Load optimizer state
        optimizer_path = os.path.join(session_path, "optimizer.pt")
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(
                torch.load(optimizer_path, map_location=device, weights_only=True)
            )

        # 3. Load gradients (for gradient accumulation across session switches)
        has_gradients = False
        if load_gradients:
            grads_path = os.path.join(session_path, "gradients.pt")
            if os.path.exists(grads_path):
                grads = torch.load(grads_path, map_location=device, weights_only=True)
                grad_count = 0
                for name, param in model.named_parameters():
                    if name in grads:
                        if param.grad is None:
                            param.grad = grads[name].to(device)
                        else:
                            param.grad.copy_(grads[name])
                        grad_count += 1
                has_gradients = grad_count > 0
                logger.debug("restored__s_gradient_tensors_for__s")

        # 4. Load metadata
        meta = {"current_step": 0, "learning_rate": optimizer.param_groups[0]["lr"], "has_gradients": has_gradients}
        meta_path = os.path.join(session_path, "training_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                loaded_meta = json.load(f)
                meta.update(loaded_meta)
                meta["has_gradients"] = has_gradients

        logger.debug("loaded_state_for__s", step=session_id, has_gradients=meta.get('current_step', 0))
        return meta

    def delete_session(self, session_id: str) -> bool:
        """Delete session checkpoint.

        Args:
            session_id: Unique session identifier.

        Returns:
            True if deleted, False if not found.
        """
        import os
        import shutil

        session_path = self.get_session_path(session_id)
        if os.path.exists(session_path):
            shutil.rmtree(session_path)
            logger.info("deleted_session__s")
            return True
        return False


@ray.remote(num_gpus=1)
class TrainingWorker:
    """Ray actor holding model + optimizer on dedicated GPU.

    Each instance runs in its own process with exclusive GPU access.
    Auto-terminates after idle_timeout seconds of inactivity.
    """

    def __init__(
        self,
        base_model: str,
        lora_rank: int,
        learning_rate: float,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        session_state_root: str = "/tmp/mint_sessions",
    ):
        """Initialize model and optimizer on this worker's GPU.

        Args:
            base_model: HuggingFace model path.
            lora_rank: LoRA adapter rank.
            learning_rate: Initial learning rate for optimizer.
            idle_timeout: Seconds of inactivity before self-termination.
                          Set to 0 to disable auto-termination.
            session_state_root: Root directory for per-session dense trainer state.
        """
        init_actor_observability()
        torch = _get_torch()
        self.device = torch.device("cuda")
        self._base_model = base_model  # Store for get_lora_config()

        # Idle timeout tracking
        self._last_activity = time.time()
        self._idle_timeout = idle_timeout
        self._shutdown_requested = False
        self._idle_watchdog_task: asyncio.Task | None = None

        logger.info("loading__s_with_lora", rank=base_model)

        self._configure_attention_backends()

        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True, local_files_only=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model on this worker's GPU
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
            device_map="cuda",  # Use this worker's GPU
        )

        # Enable gradient checkpointing for large dense models (trades compute for memory)
        # Must be done before PEFT wrapping to properly set up the model
        from mint_server.backend.core.model_registry import get_model_config
        force_grad_ckpt = bool(server_config.training_force_grad_checkpointing)
        try:
            use_grad_ckpt = get_model_config(base_model).gradient_checkpointing or force_grad_ckpt
        except ValueError:
            use_grad_ckpt = force_grad_ckpt

        # Disable KV cache for training to reduce memory
        if hasattr(self.model, "config") and getattr(self.model.config, "use_cache", None):
            self.model.config.use_cache = False

        if use_grad_ckpt:
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            logger.info("gradient_checkpointing_enabled_for__s")

        # Apply LoRA
        # Per Tinker docs: "LoRA performs better when applied to all weight matrices,
        # especially MLP and MoE layers. Attention-only LoRA underperforms."
        # For dense models, vLLM supports MLP LoRA (gate_proj, up_proj, down_proj).
        # Note: MoE models use FusedMoE kernel which doesn't support LoRA on expert layers.
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0,
            bias="none",
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
        )

        # Track training step count
        self._step_count = 0

        # Session state management for stateless trainer pattern
        self._state_manager = SessionStateManager(base_path=session_state_root)
        self._current_session_id: str | None = None
        self.max_lora_rank = int(lora_rank)
        self._current_actual_rank: int | None = None
        self._input_contract = self._make_input_contract()

        logger.info("Ready")

    @staticmethod
    def _configure_attention_backends() -> None:
        """Enable memory-efficient SDP where available."""
        if not bool(server_config.training_enable_sdp):
            return
        torch = _get_torch()
        try:
            if hasattr(torch.backends, "cuda"):
                if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                    torch.backends.cuda.enable_flash_sdp(True)
                if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                if hasattr(torch.backends.cuda, "enable_math_sdp"):
                    torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            logger.warning("failed_to_configure_sdp_backends___s")

    def _touch(self) -> None:
        """Update last activity timestamp. Call at start of every method."""
        self._last_activity = time.time()

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
        """Ensure the specified session's state is loaded.

        If a different session is currently loaded, this saves its state first,
        then loads the requested session's state.

        Args:
            session_id: Session ID to load state for.
        """
        print(f"[DEBUG] _ensure_session_loaded: current={self._current_session_id}, target={session_id}", flush=True)

        if self._current_session_id == session_id:
            # Already loaded
            print(f"[DEBUG] Session {session_id} already loaded, no switch needed", flush=True)
            self._zero_lora_rank_tail(actual_rank, zero_grads=True)
            return

        # Save outgoing session's state INCLUDING gradients
        # This is critical for gradient accumulation across session switches
        if self._current_session_id is not None:
            # Check current gradient state before saving
            grad_count = sum(1 for _, p in self.model.named_parameters() if p.requires_grad and p.grad is not None)
            grad_norm = sum((p.grad.norm().item() ** 2) for _, p in self.model.named_parameters() if p.requires_grad and p.grad is not None) ** 0.5
            print(f"[DEBUG] Saving session {self._current_session_id}: {grad_count} grads, norm={grad_norm:.4f}", flush=True)

            lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 1e-4
            self._state_manager.save_state(
                self._current_session_id, self.model, self.optimizer,
                self._step_count, lr, self.device, save_gradients=True,
                actual_rank=self._current_actual_rank,
            )
            print(f"[DEBUG] Saved outgoing session {self._current_session_id} before switch", flush=True)

        # Load new session's state
        if self._state_manager.session_exists(session_id):
            meta = self._state_manager.load_state(
                session_id, self.model, self.optimizer, self.device, load_gradients=True
            )
            self._step_count = meta.get("current_step", 0)
            actual_rank = actual_rank if actual_rank is not None else meta.get("actual_rank")
            # Update learning rate if saved
            if "learning_rate" in meta:
                for pg in self.optimizer.param_groups:
                    pg["lr"] = meta["learning_rate"]
            self._zero_lora_rank_tail(actual_rank, zero_grads=True)
            # Check actual gradient state after loading
            grad_count = sum(1 for _, p in self.model.named_parameters() if p.requires_grad and p.grad is not None)
            grad_norm = sum((p.grad.norm().item() ** 2) for _, p in self.model.named_parameters() if p.requires_grad and p.grad is not None) ** 0.5
            print(f"[DEBUG] Loaded session {session_id}: step={self._step_count}, actual_grads={grad_count}, norm={grad_norm:.4f}", flush=True)
        else:
            # New session: reinitialize weights and zero gradients
            self.reinit_lora_weights(actual_rank=actual_rank)
            self.optimizer.zero_grad()
            self._step_count = 0
            print(f"[DEBUG] New session {session_id}, initialized fresh weights", flush=True)

        self._current_session_id = session_id

    def _save_session_state(self, session_id: str, actual_rank: int | None = None) -> None:
        """Save current session state to disk.

        Called at the end of optim_step to persist state.
        Note: Gradients are NOT saved here because optim_step zeros them.

        Args:
            session_id: Session ID to save state for.
        """
        lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 1e-4
        actual_rank = self._resolve_actual_rank(actual_rank)
        self._zero_lora_rank_tail(actual_rank, zero_grads=True)
        self._state_manager.save_state(
            session_id, self.model, self.optimizer, self._step_count, lr, self.device,
            save_gradients=False,  # Gradients already applied and zeroed
            actual_rank=actual_rank,
        )
        logger.debug("saved_session__s_state", step=session_id)

    async def _idle_watchdog_async(self) -> None:
        """Async task that monitors for idle timeout.

        Runs on the actor's main asyncio event loop so Ray API calls
        (ray.actor.exit_actor) have proper thread-local connection state.
        """
        check_interval = 30  # seconds between checks
        while not self._shutdown_requested:
            await asyncio.sleep(check_interval)
            if self._shutdown_requested:
                break

            idle_time = time.time() - self._last_activity
            if idle_time >= self._idle_timeout:
                logger.warning(
                    f"[TrainingWorker] Idle for {idle_time:.0f}s (timeout={self._idle_timeout}s), "
                    "self-terminating to release GPU"
                )
                # Clean shutdown
                try:
                    self.shutdown()
                except Exception:
                    logger.error("shutdown_error___s")
                # Exit the Ray actor
                ray.actor.exit_actor()
                return

    async def heartbeat(self) -> dict:
        """Keep worker alive and return status. Call periodically to prevent idle timeout.

        Also lazily starts the idle watchdog task on the actor's main event loop
        on first call. This avoids creating a daemon thread in __init__ (which
        runs in Ray's thread pool without a running event loop).

        Returns:
            Dict with idle_time and timeout info.
        """
        self._touch()
        if self._idle_watchdog_task is None and self._idle_timeout > 0:
            self._idle_watchdog_task = asyncio.create_task(self._idle_watchdog_async())
            logger.info("idle_watchdog_started___s", timeout=self._idle_timeout)
        return {
            "idle_timeout": self._idle_timeout,
            "time_until_timeout": max(0, self._idle_timeout - (time.time() - self._last_activity)),
            "max_lora_rank": self.max_lora_rank,
            "current_actual_rank": self._current_actual_rank,
        }

    def _bind_traceparent(self, traceparent: str | None) -> None:
        if isinstance(traceparent, str) and traceparent:
            restore_trace_id_from_traceparent(traceparent)

    def get_rss_bytes(self) -> int:
        with open("/proc/self/statm", encoding="utf-8") as f:
            parts = f.read().strip().split()
        if len(parts) < 2:
            raise ValueError(f"unexpected /proc/self/statm format: {parts!r}")
        rss_pages = int(parts[1])
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return rss_pages * page_size

    def _resolve_vocab_size(self) -> int | None:
        config = getattr(self.model, "config", None)
        vocab_size = getattr(config, "vocab_size", None)
        if isinstance(vocab_size, int) and vocab_size > 0:
            return int(vocab_size)
        tokenizer_vocab = getattr(self.tokenizer, "vocab_size", None)
        if isinstance(tokenizer_vocab, int) and tokenizer_vocab > 0:
            return int(tokenizer_vocab)
        return None

    def _make_input_contract(self) -> TrainingWorkerInputContract:
        from mint_server.backend.observability.runtime_observability import runtime_observability

        return TrainingWorkerInputContract(
            base_model=lambda: str(getattr(self, "_base_model", "unknown") or "unknown"),
            vocab_size=self._resolve_vocab_size,
            request_id=get_request_id,
            record_span_event=record_span_event_otel,
            record_training_incident=runtime_observability.record_training_incident,
        )

    def _input_contract_validator(self) -> TrainingWorkerInputContract:
        validator = getattr(self, "_input_contract", None)
        if not isinstance(validator, TrainingWorkerInputContract):
            validator = self._make_input_contract()
            self._input_contract = validator
        return validator

    def _validate_forward_backward_contract(
        self,
        *,
        session_id: str | None,
        loss_fn: str,
        input_ids: list[Any],
        target_tokens: list[Any],
        weights: list[Any],
        old_logprobs: list[Any] | None = None,
        advantages: list[Any] | None = None,
    ) -> None:
        self._input_contract_validator().validate_forward_backward(
            session_id=session_id,
            loss_fn=loss_fn,
            input_ids=input_ids,
            target_tokens=target_tokens,
            weights=weights,
            old_logprobs=old_logprobs,
            advantages=advantages,
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
        """Forward + backward pass using tinker Datum format.

        Args:
            data_items: List of serialized Datum dicts with:
                - model_input.chunks[0].tokens: input token IDs
                - loss_fn_inputs.target_tokens: target token IDs (shifted by 1)
                - loss_fn_inputs.weights: per-token loss coefficients (float, can be positive, negative, or zero)
                  Interpreted as token-level coefficients applied to -logp (typically a 0/1 mask for SFT).
                For RL losses (importance_sampling, ppo), also needs:
                - loss_fn_inputs.logprobs: old policy logprobs
                - loss_fn_inputs.advantages: advantage estimates
            loss_fn: Loss function type ("cross_entropy", "importance_sampling", "ppo")
            loss_fn_config: Optional config (e.g., {"epsilon": 0.2} for PPO)
            session_id: Optional session ID for stateless trainer pattern.
                       If provided, loads session state before forward pass.

        Returns:
            Dict with loss_fn_outputs and metrics.
        """
        self._bind_traceparent(traceparent)
        torch = _get_torch()
        self._touch()

        # Stateless trainer: load session state if session_id provided
        if session_id:
            self._ensure_session_loaded(session_id, actual_rank=actual_rank)
        elif actual_rank is not None:
            self._zero_lora_rank_tail(actual_rank, zero_grads=True)

        loss_fn_config = loss_fn_config or {}

        # Training path always runs in train() mode.
        # forward() temporarily switches to eval() for deterministic logprobs and restores state.
        self.model.train()

        total_loss = 0.0
        total_tokens = 0
        loss_fn_outputs = []

        # RL-specific metrics
        total_ratio = 0.0
        total_clipfrac = 0.0
        num_rl_samples = 0

        for item in data_items:
            # Parse tinker Datum format
            model_input = item.get("model_input", {})
            loss_fn_inputs = item.get("loss_fn_inputs", {})

            # Extract input token IDs from ALL chunks (not just chunks[0])
            chunks = model_input.get("chunks", [])
            if chunks:
                # Flatten all chunks into a single token list (like ModelInput.to_token_ids())
                input_ids = []
                for chunk in chunks:
                    if chunk.get("type") == "encoded_text" and "tokens" in chunk:
                        input_ids.extend(chunk["tokens"])

                if not input_ids:
                    logger.warning("No tokens in model_input chunks, skipping item")
                    continue
            else:
                logger.warning("No chunks in model_input, skipping item")
                continue

            # Extract target tokens and weights/mask
            # Accept "weights", "mask", or "loss_mask" field names (tinker API uses "loss_mask")
            target_data = loss_fn_inputs.get("target_tokens", {})
            weights_data = loss_fn_inputs.get("weights") or loss_fn_inputs.get("loss_mask") or loss_fn_inputs.get("mask", {})

            target_tokens = target_data.get("data", [])
            weights = weights_data.get("data", []) if weights_data else []

            # For RL losses, derive weights from advantages if not provided
            # (tinker-cookbook removes mask before sending, but advantages implicitly encode it)
            if not weights and loss_fn in ("importance_sampling", "ppo"):
                advantages_data = loss_fn_inputs.get("advantages", {})
                advantages = advantages_data.get("data", [])
                if advantages:
                    # Binary mask: 1 where advantage != 0 (action tokens), 0 elsewhere
                    weights = [1.0 if a != 0 else 0.0 for a in advantages]

            if not target_tokens:
                logger.warning(
                    f"[TrainingWorker] Missing target_tokens, skipping item. "
                    f"loss_fn_inputs keys: {list(loss_fn_inputs.keys())}"
                )
                continue

            if not weights:
                logger.warning(
                    f"[TrainingWorker] Missing weights/mask and cannot derive from advantages, skipping item. "
                    f"loss_fn_inputs keys: {list(loss_fn_inputs.keys())}, "
                    f"target_tokens len: {len(target_tokens)}"
                )
                continue

            self._validate_forward_backward_contract(
                session_id=session_id,
                loss_fn=loss_fn,
                input_ids=input_ids,
                target_tokens=target_tokens,
                weights=weights,
            )

            # Convert to tensors
            input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            target_ids_t = torch.tensor([target_tokens], dtype=torch.long, device=self.device)
            weights_t = torch.tensor([weights], dtype=torch.float32, device=self.device)

            # Forward pass - get logits
            outputs = self.model(input_ids=input_ids_t)
            logits = outputs.logits  # [1, seq_len, vocab_size]

            # DEBUG: Inspect raw logits for anomalies
            logits_for_debug = logits.squeeze(0)  # [seq_len, vocab]
            max_logit = logits_for_debug.max().item()
            min_logit = logits_for_debug.min().item()
            has_nan = torch.isnan(logits_for_debug).any().item()
            has_inf = torch.isinf(logits_for_debug).any().item()
            # Get logits at target positions
            target_logits = logits_for_debug[torch.arange(len(target_tokens)), target_ids_t.squeeze(0)]
            target_max = target_logits.max().item()
            target_min = target_logits.min().item()
            # Store debug info for return
            _debug_logits_info = {
                "max_logit": max_logit,
                "min_logit": min_logit,
                "has_nan": has_nan,
                "has_inf": has_inf,
                "target_max": target_max,
                "target_min": target_min,
            }
            # Check for extreme logits (> 50 or < -50)
            extreme_mask = (logits_for_debug.abs() > 50).any(dim=-1)
            if extreme_mask.any():
                extreme_positions = extreme_mask.nonzero().squeeze(-1).tolist()
                _debug_logits_info["extreme_positions"] = extreme_positions[:10]

            # Flatten for loss computation
            logits_flat = logits.squeeze(0)  # [seq_len, vocab]
            targets_flat = target_ids_t.squeeze(0)  # [seq_len]
            weights_flat = weights_t.squeeze(0)  # [seq_len]

            # Target logprobs are needed for all supported loss_fns.
            log_probs = torch.nn.functional.log_softmax(logits_flat, dim=-1)  # [seq_len, vocab]
            target_logprobs = torch.gather(
                log_probs, dim=-1, index=targets_flat.unsqueeze(-1)
            ).squeeze(-1)  # [seq_len]

            token_count = float((weights_flat != 0).sum().item())

            if loss_fn == "cross_entropy":
                # Contract: loss = sum(-logp * weight). Weights may be real-valued (including negative).
                loss = -(target_logprobs * weights_flat).sum()
                loss.backward()

                item_loss = loss.item()
                logprobs_list = target_logprobs.detach().tolist()
                loss_fn_outputs.append({
                    "loss": {"data": [item_loss], "shape": [1], "dtype": "float32"},
                    "logprobs": {"data": logprobs_list, "shape": [len(logprobs_list)], "dtype": "float32"},
                })

            elif loss_fn in ("importance_sampling", "ppo"):
                # RL losses require old logprobs and advantages.
                old_logprobs_data = loss_fn_inputs.get("logprobs", {})
                advantages_data = loss_fn_inputs.get("advantages", {})

                old_logprobs = old_logprobs_data.get("data", [])
                advantages = advantages_data.get("data", [])

                if not old_logprobs or not advantages:
                    logger.warning(
                        f"[TrainingWorker] Missing logprobs or advantages for {loss_fn}, skipping item"
                    )
                    continue

                self._validate_forward_backward_contract(
                    session_id=session_id,
                    loss_fn=loss_fn,
                    input_ids=input_ids,
                    target_tokens=target_tokens,
                    weights=weights,
                    old_logprobs=old_logprobs,
                    advantages=advantages,
                )

                old_logprobs_t = torch.tensor(old_logprobs, dtype=torch.float32, device=self.device)
                advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)

                new_logprobs = target_logprobs

                # Compute importance ratio: exp(new_logprobs - old_logprobs).
                # Clamp for numerical stability.
                log_ratio = new_logprobs - old_logprobs_t
                log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
                ratio = torch.exp(log_ratio)

                if loss_fn == "importance_sampling":
                    # loss = -sum(ratio * advantages * weights)
                    loss = (-ratio * advantages_t * weights_flat).sum()

                else:  # ppo
                    # PPO with clipping
                    epsilon = loss_fn_config.get("epsilon", 0.2)
                    clip_low = loss_fn_config.get("clip_low", 1.0 - epsilon)
                    clip_high = loss_fn_config.get("clip_high", 1.0 + epsilon)

                    # Unclipped objective (negated for minimization)
                    pg_loss1 = -advantages_t * ratio

                    # Clipped objective (negated for minimization)
                    clipped_ratio = torch.clamp(ratio, clip_low, clip_high)
                    pg_loss2 = -advantages_t * clipped_ratio

                    # PPO loss is negative of objective: max(negated_unclipped, negated_clipped)
                    loss = (torch.maximum(pg_loss1, pg_loss2) * weights_flat).sum()

                    # Track clip fraction (fraction of masked tokens that were clipped).
                    clipped = ((ratio < clip_low) | (ratio > clip_high)).float() * weights_flat
                    denom = max(token_count, 1.0)
                    clipfrac = clipped.sum() / denom
                    total_clipfrac += clipfrac.item()

                loss.backward()
                item_loss = loss.item()

                # Track RL metrics
                denom = max(token_count, 1.0)
                masked_ratio = (ratio * weights_flat).sum() / denom
                total_ratio += masked_ratio.item()
                num_rl_samples += 1

                rl_logprobs_list = new_logprobs.detach().tolist()
                loss_fn_outputs.append({
                    "loss": {"data": [item_loss], "shape": [1], "dtype": "float32"},
                    "logprobs": {"data": rl_logprobs_list, "shape": [len(rl_logprobs_list)], "dtype": "float32"},
                })

            else:
                raise ValueError(f"Unknown loss_fn: {loss_fn}")

            total_loss += item_loss
            total_tokens += token_count

        avg_loss = total_loss / max(total_tokens, 1)
        self._zero_lora_rank_tail(actual_rank, zero_grads=True)

        metrics = {
            "loss:mean": avg_loss,
            "num_samples:sum": float(len(data_items)),
            "num_tokens:sum": float(total_tokens),
        }

        # Debug logits info removed - was causing type mismatch with client
        # (client expects Dict[str, float], not nested dicts)

        # Add RL-specific metrics
        if num_rl_samples > 0:
            metrics["ratio:mean"] = total_ratio / num_rl_samples
            if loss_fn == "ppo":
                metrics["clipfrac:mean"] = total_clipfrac / num_rl_samples

        logger.info(
            f"[TrainingWorker] forward_backward ({loss_fn}): loss={avg_loss:.4f}, tokens={total_tokens:.0f}, "
            f"request_id={get_request_id() or '-'}"
        )

        return {
            "loss_fn_output_type": f"{loss_fn}_loss",
            "loss_fn_outputs": loss_fn_outputs,
            "metrics": metrics,
        }

    def forward(
        self,
        data_items: list[dict],
        session_id: str | None = None,
        traceparent: str | None = None,
    ) -> dict:
        """Forward pass only (no backward). Returns logprobs.

        Same input format as forward_backward but skips gradient computation.
        Useful for inference-time logprob computation with training model.

        Args:
            data_items: List of serialized Datum dicts with:
                - model_input.chunks[0].tokens: input token IDs
                - loss_fn_inputs.target_tokens: target token IDs (shifted by 1)
                - loss_fn_inputs.weights: per-token weights (float)
            session_id: Optional session ID for stateless trainer pattern.
                       If provided, loads session state before forward pass.

        Returns:
            Dict with loss_fn_outputs (including logprobs) and metrics.
        """
        self._bind_traceparent(traceparent)
        torch = _get_torch()
        self._touch()

        # Stateless trainer: load session state if session_id provided
        if session_id:
            self._ensure_session_loaded(session_id)

        # Use eval() mode for deterministic logprobs, but do not leak mode to later calls.
        prev_training = self.model.training
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0
        loss_fn_outputs = []

        try:
            with torch.no_grad():
                for item in data_items:
                    # Parse tinker Datum format
                    model_input = item.get("model_input", {})
                    loss_fn_inputs = item.get("loss_fn_inputs", {})

                    # Extract input token IDs from ALL chunks (not just chunks[0])
                    chunks = model_input.get("chunks", [])
                    if chunks:
                        # Flatten all chunks into a single token list (like ModelInput.to_token_ids())
                        input_ids = []
                        for chunk in chunks:
                            if chunk.get("type") == "encoded_text" and "tokens" in chunk:
                                input_ids.extend(chunk["tokens"])

                        if not input_ids:
                            logger.warning("No tokens in model_input chunks, skipping item")
                            continue
                    else:
                        logger.warning("No chunks in model_input, skipping item")
                        continue

                    # Extract target tokens and weights/mask
                    # Accept "weights", "mask", or "loss_mask" field names (tinker API uses "loss_mask")
                    target_data = loss_fn_inputs.get("target_tokens", {})
                    weights_data = loss_fn_inputs.get("weights") or loss_fn_inputs.get("loss_mask") or loss_fn_inputs.get("mask", {})

                    target_tokens = target_data.get("data", [])
                    weights = weights_data.get("data", []) if weights_data else []

                    if not target_tokens:
                        logger.warning("Missing target_tokens, skipping item")
                        continue

                    # For forward-only, weights are optional - default to all 1s
                    if not weights:
                        weights = [1.0] * len(target_tokens)

                    # Convert to tensors
                    input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)
                    target_ids_t = torch.tensor([target_tokens], dtype=torch.long, device=self.device)
                    weights_t = torch.tensor([weights], dtype=torch.float32, device=self.device)

                    # Forward pass - get logits
                    outputs = self.model(input_ids=input_ids_t)
                    logits = outputs.logits  # [1, seq_len, vocab_size]

                    # Compute log probabilities
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [1, seq_len, vocab]

                    # Gather logprobs at target token indices
                    target_logprobs = torch.gather(
                        log_probs.squeeze(0),  # [seq_len, vocab]
                        dim=-1,
                        index=target_ids_t.squeeze(0).unsqueeze(-1),  # [seq_len, 1]
                    ).squeeze(-1)  # [seq_len]

                    # Compute cross-entropy loss (for metrics)
                    weights_flat = weights_t.squeeze(0)  # [seq_len]
                    logits_flat = logits.squeeze(0)
                    targets_flat = target_ids_t.squeeze(0)
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

    def _infer_adapter_checkpoint_rank(self, checkpoint_path: str) -> int:
        """Infer adapter rank from checkpoint metadata and tensor shapes."""
        from safetensors import safe_open

        rank_sources: list[tuple[str, int]] = []

        meta_path = os.path.join(checkpoint_path, "training_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            if not isinstance(meta, dict):
                raise ValueError(f"training_meta.json must contain a JSON object, got {type(meta).__name__}")
            if "actual_rank" in meta:
                rank = meta["actual_rank"]
                if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
                    raise ValueError(f"training_meta.actual_rank must be a positive int, got {rank!r}")
                rank_sources.append(("training_meta.actual_rank", int(rank)))

        config_path = os.path.join(checkpoint_path, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                adapter_config = json.load(f)
            if not isinstance(adapter_config, dict):
                raise ValueError(f"adapter_config.json must contain a JSON object, got {type(adapter_config).__name__}")
            rank = adapter_config.get("r")
            if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
                raise ValueError(f"adapter_config.r must be a positive int, got {rank!r}")
            rank_sources.append(("adapter_config.r", int(rank)))

        weights_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Adapter not found: {weights_path}")
        tensor_ranks: list[tuple[str, int]] = []
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                lowered = str(key).lower()
                is_lora_a = "lora_a" in lowered or ("adapter" in lowered and "linear_in" in lowered)
                is_lora_b = "lora_b" in lowered or ("adapter" in lowered and "linear_out" in lowered)
                if not (is_lora_a or is_lora_b):
                    continue
                shape = tuple(int(dim) for dim in handle.get_slice(key).get_shape())
                if len(shape) < 2:
                    raise ValueError(f"{key}: expected rank-2+ LoRA tensor, got shape={shape}")
                tensor_ranks.append((str(key), int(shape[0] if is_lora_a else shape[-1])))
        if tensor_ranks:
            values = {rank for _, rank in tensor_ranks}
            if len(values) != 1:
                raise ValueError(f"LoRA tensor rank mismatch: {tensor_ranks}")
            rank_sources.append(("adapter_model.safetensors", next(iter(values))))

        if not rank_sources:
            raise ValueError(f"No LoRA rank metadata or tensors found in checkpoint: {checkpoint_path}")
        values = {rank for _, rank in rank_sources}
        if len(values) != 1:
            raise ValueError(f"Checkpoint LoRA rank metadata mismatch: {rank_sources}")
        return self._resolve_actual_rank(next(iter(values)))

    def forward_backward_reverse_kl(
        self,
        data_items: list[dict],
        reference_checkpoint_path: str,
        temperature: float,
        session_id: str | None = None,
        actual_rank: int | None = None,
        traceparent: str | None = None,
    ) -> dict:
        """Forward/backward for Mint reverse-KL distillation against a fixed reference checkpoint."""
        self._bind_traceparent(traceparent)
        torch = _get_torch()
        self._touch()

        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature!r}")
        if session_id:
            self._ensure_session_loaded(session_id, actual_rank=actual_rank)
        current_actual_rank = self._resolve_actual_rank(actual_rank)

        from mint_server.backend.core.mintx_ops import (
            build_scoring_sequence,
            compute_teacher_log_probs_cpu,
            parse_reverse_kl_item,
            reverse_kl_from_teacher_log_probs,
            temporary_adapter_snapshot_dir,
        )

        block_size = int(os.environ.get("MINT_REVERSE_KL_VOCAB_BLOCK", "4096"))
        student_batches = [parse_reverse_kl_item(item, input_key="student_input") for item in data_items]
        reference_batches = [parse_reverse_kl_item(item, input_key="reference_input") for item in data_items]

        teacher_log_probs_cpu: list[torch.Tensor] = []
        prev_training = self.model.training
        try:
            with temporary_adapter_snapshot_dir("mintx_dense_ref_") as snapshot_dir:
                self.save_adapter_state(
                    snapshot_dir,
                    actual_rank=current_actual_rank,
                    trainer_rank=self.max_lora_rank,
                )
                try:
                    self.model.eval()
                    reference_actual_rank = self._infer_adapter_checkpoint_rank(reference_checkpoint_path)
                    self.load_adapter_state(
                        reference_checkpoint_path,
                        actual_rank=reference_actual_rank,
                        trainer_rank=self.max_lora_rank,
                    )
                    with torch.no_grad():
                        for batch in reference_batches:
                            scoring_input, completion_start = build_scoring_sequence(
                                batch.prefix_tokens,
                                batch.completion_tokens,
                            )
                            input_ids_t = torch.tensor(
                                [scoring_input], dtype=torch.long, device=self.device
                            )
                            logits = self.model(input_ids=input_ids_t).logits.squeeze(0)
                            completion_logits = logits[
                                completion_start: completion_start + len(batch.completion_tokens)
                            ]
                            teacher_log_probs_cpu.append(
                                compute_teacher_log_probs_cpu(
                                    completion_logits,
                                    temperature=temperature,
                                    block_size=block_size,
                                )
                            )
                finally:
                    self.load_adapter_state(
                        snapshot_dir,
                        actual_rank=current_actual_rank,
                        trainer_rank=self.max_lora_rank,
                    )

            self.model.train()
            outputs = []
            total_loss = 0.0
            total_tokens = 0.0
            for batch, teacher_log_probs in zip(student_batches, teacher_log_probs_cpu, strict=True):
                scoring_input, completion_start = build_scoring_sequence(
                    batch.prefix_tokens,
                    batch.completion_tokens,
                )
                input_ids_t = torch.tensor([scoring_input], dtype=torch.long, device=self.device)
                logits = self.model(input_ids=input_ids_t).logits.squeeze(0)
                completion_logits = logits[
                    completion_start: completion_start + len(batch.completion_tokens)
                ]
                token_kl = reverse_kl_from_teacher_log_probs(
                    completion_logits,
                    teacher_log_probs,
                    temperature=temperature,
                    block_size=block_size,
                )
                weights_t = torch.tensor(batch.weights, dtype=torch.float32, device=self.device)
                loss = (token_kl * weights_t).sum()
                loss.backward()
                outputs.append(
                    {
                        "loss": {
                            "data": [float(loss.detach().item())],
                            "shape": [1],
                            "dtype": "float32",
                        }
                    }
                )
                total_loss += float(loss.detach().item())
                total_tokens += float((weights_t != 0).sum().item())

            avg_loss = total_loss / max(total_tokens, 1.0)
            return {
                "outputs": outputs,
                "metrics": {
                    "loss:mean": float(avg_loss),
                    "reverse_kl:mean": float(avg_loss),
                    "num_samples:sum": float(len(outputs)),
                    "num_tokens:sum": float(total_tokens),
                },
                "type": "mint_forward_backward_reverse_kl",
            }
        finally:
            if prev_training:
                self.model.train()
            else:
                self.model.eval()

    def get_tokenizer_info(self) -> dict:
        """Return tokenizer configuration for client use.

        Returns:
            Dict with tokenizer info (vocab_size, special tokens, etc.)
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

    def optim_step(
        self,
        learning_rate: float | None,
        session_id: str | None = None,
        actual_rank: int | None = None,
        traceparent: str | None = None,
    ) -> dict:
        """Optimizer update step.

        Args:
            learning_rate: Optional new learning rate.
            session_id: Optional session ID for stateless trainer pattern.
                       If provided, saves session state after optimizer step.

        Returns:
            Dict with metrics.
        """
        self._bind_traceparent(traceparent)
        torch = _get_torch()
        self._touch()

        # Stateless trainer: ensure session state is loaded
        if session_id:
            self._ensure_session_loaded(session_id, actual_rank=actual_rank)
        elif actual_rank is not None:
            self._zero_lora_rank_tail(actual_rank, zero_grads=True)

        # Update learning rate if provided
        if learning_rate is not None:
            for pg in self.optimizer.param_groups:
                pg["lr"] = learning_rate

        self._zero_lora_rank_tail(actual_rank, zero_grads=True)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self._zero_lora_rank_tail(actual_rank, zero_grads=True)
        self.optimizer.zero_grad()

        self._step_count += 1

        # Stateless trainer: save session state after update
        if session_id:
            self._save_session_state(session_id, actual_rank=actual_rank)

        logger.info("optim_step", grad_norm=grad_norm, step=self._step_count)

        return {
            "metrics": {"grad_norm:last": float(grad_norm)},
            "type": "optim_step",
        }

    def get_lora_state_dict(self) -> dict[str, torch.Tensor]:
        """Extract LoRA adapter weights as state dict.

        Returns:
            Dict mapping parameter names to tensors (on CPU).
        """
        self._touch()
        from peft.utils.save_and_load import get_peft_model_state_dict

        state_dict = get_peft_model_state_dict(self.model)
        # Move to CPU for serialization
        return {k: v.cpu() for k, v in state_dict.items()}

    def get_lora_config(self, actual_rank: int | None = None) -> dict:
        """Get LoRA configuration as dictionary.

        Returns:
            PEFT config dict compatible with vLLM's PEFTHelper.
        """
        peft_config = self.model.peft_config.get("default")
        effective_rank = self._resolve_actual_rank(actual_rank)
        assert peft_config is not None
        assert peft_config is not None
        assert peft_config is not None
        assert peft_config is not None
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
    ) -> str:
        """Save LoRA adapter to directory.

        Args:
            save_path: Directory path to save adapter files.
            session_id: Optional session to materialize before export.

        Returns:
            Absolute path where weights were saved.
        """
        self._bind_traceparent(traceparent)
        self._touch()
        if session_id is not None:
            self._ensure_session_loaded(session_id, actual_rank=actual_rank)
        import json
        import os

        from safetensors.torch import save_file
        from mint_server.backend.inference.lora_utils import truncate_lora_state_dict

        os.makedirs(save_path, exist_ok=True)
        effective_rank = self._resolve_actual_rank(actual_rank)
        self._zero_lora_rank_tail(effective_rank, zero_grads=True)

        # Save adapter weights
        state_dict = self.get_lora_state_dict()
        state_dict = truncate_lora_state_dict(state_dict, self.max_lora_rank, effective_rank)
        save_file(state_dict, os.path.join(save_path, "adapter_model.safetensors"))

        # Save adapter config
        config = self.get_lora_config(effective_rank)
        with open(os.path.join(save_path, "adapter_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        abs_path = os.path.abspath(save_path)
        logger.info("saved_lora_weights_to__s")
        return {
            "path": abs_path,
            "state_dict": state_dict,
            "peft_config": config,
            "current_step": self._step_count,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def save_checkpoint(
        self,
        save_path: str,
        traceparent: str | None = None,
        *,
        session_id: str | None = None,
        actual_rank: int | None = None,
    ) -> dict:
        """Save full checkpoint: LoRA weights + optimizer state + training metadata.

        Args:
            save_path: Directory path to save checkpoint files.
            session_id: Optional session to materialize before export.

        Returns:
            Dict with training metadata, state_dict, and peft_config for registration.
        """
        self._bind_traceparent(traceparent)
        self._touch()
        if session_id is not None:
            self._ensure_session_loaded(session_id, actual_rank=actual_rank)
        import json
        import os

        torch = _get_torch()
        from safetensors.torch import save_file
        from mint_server.backend.inference.lora_utils import truncate_lora_state_dict

        os.makedirs(save_path, exist_ok=True)
        effective_rank = self._resolve_actual_rank(actual_rank)
        self._zero_lora_rank_tail(effective_rank, zero_grads=True)

        # 1. LoRA weights
        state_dict = self.get_lora_state_dict()
        state_dict = truncate_lora_state_dict(state_dict, self.max_lora_rank, effective_rank)
        save_file(state_dict, os.path.join(save_path, "adapter_model.safetensors"))

        # 2. LoRA config
        peft_config = self.get_lora_config(effective_rank)
        with open(os.path.join(save_path, "adapter_config.json"), "w") as f:
            json.dump(peft_config, f, indent=2)

        # 3. Optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))

        # 4. Training metadata
        meta = {
            "current_step": self._step_count,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "actual_rank": effective_rank,
            # Include state_dict and config for multi-LoRA registration
            # (API server can't read files from Ray worker filesystem)
            "state_dict": state_dict,
            "peft_config": peft_config,
        }
        with open(os.path.join(save_path, "training_meta.json"), "w") as f:
            # Don't write state_dict/peft_config to file (already saved separately)
            json.dump({k: v for k, v in meta.items() if k not in ("state_dict", "peft_config")}, f, indent=2)

        abs_path = os.path.abspath(save_path)
        logger.info("saved_checkpoint_to__s", step=abs_path)
        return meta

    def load_checkpoint(
        self,
        load_path: str,
        load_optimizer: bool = True,
        traceparent: str | None = None,
        *,
        session_id: str | None = None,
    ) -> dict:
        """Load checkpoint, optionally restoring optimizer state.

        Args:
            load_path: Directory path to load checkpoint from.
            load_optimizer: Whether to restore optimizer state.
            session_id: Optional session to materialize before loading.

        Returns:
            Dict with training metadata.
        """
        self._bind_traceparent(traceparent)
        self._touch()
        import json
        import os

        torch = _get_torch()
        from safetensors.torch import load_file

        # 1. Load LoRA weights and metadata
        adapter_path = os.path.join(load_path, "adapter_model.safetensors")
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")

        meta: dict[str, object] = {}
        meta_path = os.path.join(load_path, "training_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                loaded_meta = json.load(f)
            if isinstance(loaded_meta, dict):
                meta = loaded_meta
            else:
                logger.warning(
                    "[TrainingWorker] Invalid checkpoint metadata type %s in %s; "
                    "preserving existing step/lr state",
                    type(loaded_meta).__name__,
                    meta_path,
                )

        config_rank: int | None = None
        adapter_config_path = os.path.join(load_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)
            if not isinstance(adapter_config, dict):
                raise ValueError(
                    f"adapter_config.json must contain a JSON object, got {type(adapter_config).__name__}"
                )
            raw_config_rank = adapter_config.get("r")
            if not isinstance(raw_config_rank, int) or isinstance(raw_config_rank, bool):
                raise ValueError(f"adapter_config.r must be an int, got {raw_config_rank!r}")
            config_rank = int(raw_config_rank)

        # 2. Optionally load optimizer state
        optimizer_path = os.path.join(load_path, "optimizer.pt")
        if load_optimizer:
            if not os.path.exists(optimizer_path):
                raise FileNotFoundError(
                    f"Optimizer restore requested, but optimizer state not found: {optimizer_path}"
                )

        state_dict = load_file(adapter_path, device=str(self.device))
        from mint_server.backend.inference.lora_utils import get_lora_rank_from_state_dict, pad_lora_state_dict

        meta_rank: int | None = None
        if "actual_rank" in meta:
            raw_meta_rank = meta["actual_rank"]
            if not isinstance(raw_meta_rank, int) or isinstance(raw_meta_rank, bool):
                raise ValueError(f"training_meta.actual_rank must be an int, got {raw_meta_rank!r}")
            meta_rank = int(raw_meta_rank)
        inferred_rank = get_lora_rank_from_state_dict(state_dict)
        rank_sources = [
            ("training_meta.actual_rank", meta_rank),
            ("adapter_config.r", config_rank),
            ("adapter_model.safetensors", inferred_rank),
        ]
        present_ranks = [(name, rank) for name, rank in rank_sources if rank is not None]
        if not present_ranks:
            raise ValueError(f"No LoRA rank metadata or tensors found in checkpoint: {load_path}")
        rank_values = {int(rank) for _, rank in present_ranks}
        if len(rank_values) != 1:
            raise ValueError(f"Checkpoint LoRA rank metadata mismatch: {present_ranks}")
        actual_rank = next(iter(rank_values))
        actual_rank = self._resolve_actual_rank(actual_rank)

        if session_id is not None:
            self._ensure_session_loaded(session_id, actual_rank=actual_rank)

        state_dict = pad_lora_state_dict(state_dict, actual_rank, self.max_lora_rank)
        # Load into PEFT model
        from peft.utils.save_and_load import set_peft_model_state_dict
        set_peft_model_state_dict(self.model, state_dict)
        self._zero_lora_rank_tail(actual_rank=actual_rank, zero_grads=True)
        logger.info("loaded_lora_weights_from__s")

        # 3. Apply metadata and optimizer state
        if "current_step" in meta:
            meta_step = meta["current_step"]
            if isinstance(meta_step, int) and not isinstance(meta_step, bool):
                self._step_count = meta_step
                logger.info("loaded_metadata", step=self._step_count)
            else:
                logger.warning(
                    "[TrainingWorker] Invalid current_step type=%s value=%r in %s; "
                    "preserving existing step=%s",
                    type(meta_step).__name__,
                    meta_step,
                    meta_path,
                    self._step_count,
                )

        checkpoint_lr = meta.get("learning_rate")
        try:
            checkpoint_lr = float(checkpoint_lr) if checkpoint_lr is not None else None
        except Exception:
            logger.warning(
                "[TrainingWorker] Invalid learning_rate value=%r in %s; preserving optimizer lr",
                checkpoint_lr,
                meta_path,
            )
            checkpoint_lr = None

        if load_optimizer:
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            logger.info("loaded_optimizer_state_from__s")
        else:
            # Non-resume loads must drop any session-local momentum/gradients that
            # _ensure_session_loaded() may have materialized from a previous session incarnation.
            self.reset_optimizer(checkpoint_lr)
            logger.info(
                "[TrainingWorker] Reset optimizer state after non-resume checkpoint load "
                f"(lr={checkpoint_lr})"
            )

        return meta


    # =====================================================================
    # Phase 8: Session Management Methods (backported from MegatronRankWorker)
    # =====================================================================

    def load_adapter_state(
        self,
        checkpoint_path: str,
        actual_rank: int | None = None,
        trainer_rank: int | None = None,
    ) -> dict:
        """Load LoRA adapter weights from checkpoint.

        Phase 8: Supports padding for unified rank training.

        Args:
            checkpoint_path: Directory containing adapter checkpoint.
            actual_rank: The rank of the checkpoint being loaded.
            trainer_rank: The trainer's max rank.

        Returns:
            Dict with status info.
        """
        import os

        from safetensors.torch import load_file

        from mint_server.backend.inference.lora_utils import pad_lora_state_dict

        adapter_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")

        state_dict = load_file(adapter_path, device=str(self.device))

        # Phase 7: Apply padding if actual_rank < trainer_rank
        if actual_rank is not None and trainer_rank is not None and actual_rank < trainer_rank:
            logger.info("padding_adapter_from_rank__s_to__s")
            state_dict = pad_lora_state_dict(state_dict, actual_rank, trainer_rank)

        # Load into PEFT model
        from peft.utils.save_and_load import set_peft_model_state_dict

        set_peft_model_state_dict(self.model, state_dict)
        self._zero_lora_rank_tail(
            actual_rank if actual_rank is not None else trainer_rank,
            zero_grads=True,
        )
        logger.info("loaded_adapter_state_from__s")

        return {"status": "ok", "path": checkpoint_path, "actual_rank": actual_rank}

    def save_adapter_state(
        self,
        checkpoint_path: str,
        actual_rank: int | None = None,
        trainer_rank: int | None = None,
    ) -> dict:
        """Save LoRA adapter weights to checkpoint.

        Phase 8: Supports truncation for unified rank training.

        Args:
            checkpoint_path: Directory to save adapter checkpoint.
            actual_rank: The rank to save as (truncate to).
            trainer_rank: The trainer's max rank.

        Returns:
            Dict with status info.
        """
        import os

        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file

        from mint_server.backend.inference.lora_utils import truncate_lora_state_dict

        os.makedirs(checkpoint_path, exist_ok=True)
        if actual_rank is not None:
            self._zero_lora_rank_tail(actual_rank, zero_grads=True)

        # Get LoRA state dict
        state_dict = get_peft_model_state_dict(self.model)

        # Phase 7: Apply truncation if actual_rank < trainer_rank
        if actual_rank is not None and trainer_rank is not None and actual_rank < trainer_rank:
            logger.info("truncating_adapter_from_rank__s_to__s")
            state_dict = truncate_lora_state_dict(state_dict, trainer_rank, actual_rank)

        # Save to safetensors format
        adapter_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
        save_file(state_dict, adapter_path)
        logger.info("saved_adapter_state_to__s")

        return {"status": "ok", "path": checkpoint_path, "actual_rank": actual_rank}

    def reset_optimizer(self, learning_rate: float | None = None) -> dict:
        """Reset optimizer state for a new session.

        Phase 8: Resets learning rate and zeros gradients for session swap.

        Args:
            learning_rate: New learning rate. If None, keeps current rate.

        Returns:
            Dict with status info.
        """
        # Update learning rate if provided
        if learning_rate is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
            logger.info("set_learning_rate_to__s")

        # Zero gradients
        self.optimizer.zero_grad()

        # Reset optimizer state (momentum/variance)
        # For AdamW, this resets exp_avg and exp_avg_sq
        self.optimizer.state.clear()
        logger.info("Reset optimizer state (momentum cleared)")

        return {"status": "ok", "learning_rate": learning_rate}

    def reinit_lora_weights(
        self,
        learning_rate: float | None = None,
        actual_rank: int | None = None,
        traceparent: str | None = None,
    ) -> dict:
        """Reinitialize LoRA weights AND optimizer state for fresh session.

        Uses standard initialization:
        - lora_A: xavier_uniform
        - lora_B: zeros

        Also resets Adam optimizer state to prevent momentum from previous
        sessions affecting new training.

        Args:
            learning_rate: New learning rate. If provided, updates param_groups.

        Returns:
            dict with reinit_count, opt_state_reset, lr_updated.
        """
        self._bind_traceparent(traceparent)
        import torch.nn.init as init

        reinit_count = 0
        opt_state_reset = 0
        lr_updated = False

        # Find and reinitialize all LoRA parameters
        for name, param in self.model.named_parameters():
            name_lower = name.lower()
            if 'lora' not in name_lower:
                continue
            if not param.requires_grad:
                continue

            is_lora_a = 'lora_a' in name_lower
            is_lora_b = 'lora_b' in name_lower

            if is_lora_a:
                init.xavier_uniform_(param.data)
                reinit_count += 1
            elif is_lora_b:
                init.zeros_(param.data)
                reinit_count += 1

        # Update learning rate
        if learning_rate is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
            lr_updated = True
            logger.info("set_learning_rate_to__s")

        # Zero gradients
        self.optimizer.zero_grad()
        tail_zeroed = self._zero_lora_rank_tail(actual_rank, zero_grads=True)

        # Reset optimizer state (momentum/variance)
        opt_state_reset = len(self.optimizer.state)
        self.optimizer.state.clear()
        logger.info("reset_optimizer_state___s_entries")

        # Reset step count
        self._step_count = 0

        logger.info("reinitialized__s_lora_params")
        return {
            "status": "ok",
            "reinit_count": reinit_count,
            "tail_zeroed": tail_zeroed,
            "opt_state_reset": opt_state_reset,
            "lr_updated": lr_updated,
            "learning_rate": learning_rate,
            "actual_rank": self._resolve_actual_rank(actual_rank),
        }

    def get_session_info(self) -> dict:
        """Get current session info for diagnostics.

        Returns:
            Dict with session and worker info.
        """
        # Get current learning rate
        lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else None

        # Get model info
        peft_config = self.model.peft_config.get("default")
        lora_rank = peft_config.r if peft_config else None

        return {
            "learning_rate": lr,
            "lora_rank": lora_rank,
            "step_count": self._step_count,
            "device": str(self.device),
        }

    def swap_session(
        self,
        old_session_id: str | None,
        new_session_id: str,
        old_checkpoint_path: str | None,
        new_checkpoint_path: str | None,
        new_learning_rate: float,
        new_actual_rank: int | None = None,
    ) -> dict:
        """Atomically swap from old session to new session.

        Phase 8: Session swap for dense models.

        Args:
            old_session_id: ID of session being swapped out (None if first).
            new_session_id: ID of session being swapped in.
            old_checkpoint_path: Where to save old session state (None to skip).
            new_checkpoint_path: Where to load new session state (None to reset).
            new_learning_rate: Learning rate for new session.
            new_actual_rank: Actual LoRA rank for new session.

        Returns:
            Dict with swap status.
        """
        import os

        logger.info("swapping_session___s_____s")

        # Get trainer rank from current model
        peft_config = self.model.peft_config.get("default")
        trainer_rank = peft_config.r if peft_config else None

        # 1. Save old session state (if applicable)
        if old_session_id and old_checkpoint_path:
            logger.info("saving_old_session__s")
            self.save_adapter_state(old_checkpoint_path)

        # 2. Load new session state or reset
        if new_checkpoint_path and os.path.exists(new_checkpoint_path):
            logger.info("loading_new_session__s")
            self.load_adapter_state(
                new_checkpoint_path,
                actual_rank=new_actual_rank,
                trainer_rank=trainer_rank,
            )
        else:
            logger.info("resetting_for_new_session__s")
            self.reset_optimizer(new_learning_rate)

        # 3. Update step count
        self._step_count = 0

        logger.info("session_swap_complete__now_on__s")
        return {
            "status": "ok",
            "old_session": old_session_id,
            "new_session": new_session_id,
            "actual_rank": new_actual_rank,
        }

    def shutdown(self) -> None:
        """Release GPU resources and stop watchdog thread."""
        torch = _get_torch()
        logger.info("Shutting down")
        # Stop watchdog thread
        self._shutdown_requested = True
        # Release GPU resources
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "optimizer"):
            del self.optimizer
        torch.cuda.empty_cache()
        # Ensure this actor cannot remain alive in a partially-shutdown state.
        # A detached actor that survives after deleting model/optimizer will
        # later fail get_session_info()/training calls and wedge create_model.
        try:
            ray.actor.exit_actor()
        except Exception:
            pass


class VerlTrainingEngine:
    """Manages per-session TrainingWorker Ray actors."""

    def __init__(
        self,
        default_base_model: str | None = None,
        default_lora_rank: int = 32,
    ):
        # No default model - clients specify per-request
        self.default_base_model = default_base_model
        self.default_lora_rank = default_lora_rank
        self._workers: dict[str, ray.actor.ActorHandle] = {}
        self._actor_recycler = VerlActorRecycler()
        self._engine_config = VerlEngineConfig()
        self._checkpoint_ops = CheckpointOps()

    def _megatron_guard_preflight_enabled(self) -> bool:
        return self._engine_config.megatron_guard_preflight_enabled()

    def _megatron_guard_query_timeout_s(self) -> float:
        return self._engine_config.megatron_guard_query_timeout_s()

    async def _ensure_megatron_session_guard_clean(
        self,
        session: "TrainingSession",
        *,
        op: str,
        worker: ray.actor.ActorHandle,
    ) -> None:
        if session.backend != "megatron" or not self._megatron_guard_preflight_enabled():
            return
        get_guard_state = getattr(worker, "get_session_guard_state", None)
        if get_guard_state is None:
            return
        guard_timeout_s = self._megatron_guard_query_timeout_s()
        try:
            guard_state = await self._await_with_keepalive(
                get_guard_state.remote(session.model_id),
                session,
                interval_s=30.0,
                timeout_s=guard_timeout_s,
            )
        except Exception as e:
            raise RuntimeError(
                f"[{session.model_id}] failed to query megatron session guard before op={op}: "
                f"{type(e).__name__}: {e}"
            ) from e
        if not isinstance(guard_state, dict):
            raise RuntimeError(
                f"[{session.model_id}] invalid megatron session guard payload type "
                f"{type(guard_state).__name__} before op={op}"
            )
        contaminated = bool(guard_state.get("contaminated", False))
        blocked = bool(guard_state.get("blocked", False))
        if not contaminated and not blocked:
            return
        contamination_reason = guard_state.get("contamination_reason")
        block_reason = guard_state.get("block_reason")
        raise RuntimeError(
            f"[{session.model_id}] megatron session guard denied op={op}: "
            f"contaminated={contaminated} blocked={blocked} "
            f"contamination_reason={contamination_reason!r} block_reason={block_reason!r}; "
            "requires clean reload"
        )

    async def initialize(self) -> None:
        """Initialize Ray connection."""
        if not ray.is_initialized():
            # Use fixed namespace for persistent vLLM actor support
            init_ray(
                namespace=RAY_NAMESPACE,
                ignore_reinit_error=True,
            )
        logger.info("VerlTrainingEngine ready (Ray actors)")

    def _touch_actor(self, session: "TrainingSession") -> None:
        """Update last_accessed timestamp and session for the session's actor.

        Supervisor inventory idleness is time-based; a 32k forward/backward can easily run
        longer than the idle timeout. We keep actors marked as active while a
        request is in-flight to prevent eviction of busy actors.
        """
        actor_name = self._actor_name_for_session(session)
        if not actor_name:
            return
        try:
            from mint_server.backend.actors.model_actor_supervisor import get_model_actor_supervisor

            model_actor_supervisor = get_model_actor_supervisor()
            model_actor_supervisor.touch(actor_name)
            model_actor_supervisor.set_session(actor_name, session.model_id)
        except Exception:
            logger.debug("skip_touch_actor_for__s", model_id=session.model_id, exc_info=True)

    def _actor_name_for_session(self, session: "TrainingSession") -> str | None:
        return self._actor_recycler.actor_name_for_session(session) or str(getattr(session, "actor_name", "") or "") or None

    def _raise_if_session_poisoned(self, session: "TrainingSession", *, op: str) -> None:
        self._actor_recycler.raise_if_session_poisoned(session, op=op)

    def _note_successful_worker_call(self, session: "TrainingSession", *, op: str) -> None:
        self._actor_recycler.note_successful_worker_call(session, op=op)

    def _megatron_op_requires_fail_closed_after_actor_death(self, op: str) -> bool:
        return op in {
            "forward_backward",
            "optim_step",
            "train_step",
            "save_weights",
            "save_lora_weights_for_sampler",
            "load_weights",
        }

    def _megatron_missing_actor_recovery_error(
        self,
        session: "TrainingSession",
        *,
        op: str,
        cause: BaseException,
        explicit_checkpoint_path: str | None = None,
    ) -> str | None:
        if not isinstance(cause, RuntimeError) or "missing worker" not in str(cause):
            return None

        from mint_server.backend.ray_cluster.model_actor_names import megatron_actor_name as _make_megatron_actor_name
        from mint_server.backend.training.megatron.megatron_distributed import MegatronSessionStateManager

        session_manager = MegatronSessionStateManager()
        actor_name = self._actor_name_for_session(session)
        if not actor_name:
            actor_name = _make_megatron_actor_name(session.base_model or "")

        list_actor_only_state_sessions = getattr(session_manager, "list_actor_only_state_sessions", None)
        dirty_sessions: list[str] = []
        if callable(list_actor_only_state_sessions):
            raw_dirty = list_actor_only_state_sessions(actor_name)
            if isinstance(raw_dirty, (list, tuple, set)):
                dirty_sessions = [str(s) for s in raw_dirty if isinstance(s, str)]

        sibling_dirty = sorted([sid for sid in dirty_sessions if sid != session.model_id])
        if sibling_dirty:
            siblings = ", ".join(sibling_dirty)
            return (
                f"[{session.model_id}] megatron actor was missing before op={op}; "
                f"dirty_sibling session(s) detected on shared actor cache: {siblings}. "
                "Reload all dirty sessions from explicit checkpoints before continuing."
            )
        if (
            op == "load_weights"
            and isinstance(explicit_checkpoint_path, str)
            and os.path.isdir(explicit_checkpoint_path)
        ):
            if session.model_id in dirty_sessions:
                logger.warning(
                    "[%s] megatron actor was missing before explicit load_weights; "
                    "deferring same-session actor-only marker cleanup until checkpoint load succeeds: %s",
                    session.model_id,
                    explicit_checkpoint_path,
                )
            return None
        if session.model_id in dirty_sessions:
            return (
                f"[{session.model_id}] megatron actor was missing before op={op}, and this session has "
                "actor-only training state that was never fully persisted. Reload from an explicit "
                "checkpoint before continuing."
            )

        if not session_manager.session_exists(session.model_id):
            return (
                f"[{session.model_id}] megatron actor was missing before op={op}, but the session "
                "has no persisted Megatron session cache. The request was not retried because "
                "that would recreate blank in-memory state. Recreate or reload the session from "
                "a checkpoint before continuing."
            )
        meta = session_manager.get_metadata(session.model_id)
        if not isinstance(meta, dict):
            return (
                f"[{session.model_id}] megatron actor was missing before op={op}, and the session "
                "cache is missing session_metadata.json. Reload the session from an explicit "
                "checkpoint before continuing."
            )
        return None

    async def _recycle_worker_after_failure(
        self,
        session: "TrainingSession",
        *,
        op: str,
        cause: BaseException,
        request_started: bool = False,
        explicit_checkpoint_path: str | None = None,
    ) -> ray.actor.ActorHandle:
        actor_name = self._actor_name_for_session(session)
        lost_session_ids: list[str] = []
        if actor_name is not None:
            lost_session_ids = self._actor_recycler.volatile_sessions_for_actor(actor_name)

        explicit_same_session_reload = (
            session.backend == "megatron"
            and op == "load_weights"
            and not request_started
            and lost_session_ids == [session.model_id]
            and isinstance(explicit_checkpoint_path, str)
            and os.path.isdir(explicit_checkpoint_path)
        )
        if session.backend == "megatron" and lost_session_ids and not explicit_same_session_reload:
            joined = ", ".join(lost_session_ids)
            err = (
                f"[{session.model_id}] megatron actor recycle detected after op={op}, but "
                f"session(s) {joined} had live in-memory state that was never persisted. "
                "The request was not retried because that would hide rollback. "
                "Reload the lost session from a checkpoint before continuing."
            )
            for sid in lost_session_ids:
                self._actor_recycler.mark_poisoned(sid, err)
            if actor_name:
                self._actor_recycler.clear_actor_runtime_state(actor_name)
            raise RuntimeError(err) from cause

        if explicit_same_session_reload and actor_name:
            logger.warning(
                "[%s] megatron actor recycle before explicit load_weights; "
                "deferring same-session volatile marker cleanup until checkpoint load succeeds: %s",
                session.model_id,
                explicit_checkpoint_path,
            )

        if session.backend == "megatron":
            missing_actor_error = self._megatron_missing_actor_recovery_error(
                session,
                op=op,
                cause=cause,
                explicit_checkpoint_path=explicit_checkpoint_path,
            )
            if missing_actor_error is not None:
                self._actor_recycler.mark_poisoned(session.model_id, missing_actor_error)
                if actor_name:
                    self._actor_recycler.clear_actor_runtime_state(actor_name)
                raise RuntimeError(missing_actor_error) from cause
            if request_started and self._megatron_op_requires_fail_closed_after_actor_death(op):
                worker = await self._recycle_megatron_actor(session, op=op, cause=cause)
                err = (
                    f"[{session.model_id}] megatron actor died during op={op}; "
                    "operation may have partially executed before the crash; "
                    "reload from checkpoint before retrying."
                )
                self._actor_recycler.mark_poisoned(session.model_id, err)
                if actor_name:
                    self._actor_recycler.clear_actor_runtime_state(actor_name)
                raise RuntimeError(err) from cause
            worker = await self._recycle_megatron_actor(session, op=op, cause=cause)
            if actor_name and not explicit_same_session_reload:
                self._actor_recycler.clear_actor_runtime_state(actor_name)
            return worker

        if session.backend == "bumblebee":
            worker = await self._recycle_bumblebee_actor(session, op=op, cause=cause)
            missing_before_request = (
                not request_started
                and isinstance(cause, RuntimeError)
                and "missing worker" in str(cause)
            )
            if missing_before_request and not lost_session_ids:
                if actor_name:
                    self._actor_recycler.clear_actor_runtime_state(actor_name)
                return worker
            if lost_session_ids:
                joined = ", ".join(lost_session_ids)
                err = (
                    f"[{session.model_id}] bumblebee actor recycle detected before op={op}, but "
                    f"session(s) {joined} had live in-memory state that was never persisted. "
                    "The request was not retried because that would hide rollback. "
                    "Reload the lost session from a checkpoint before continuing."
                )
                for sid in lost_session_ids:
                    self._actor_recycler.mark_poisoned(sid, err)
                if actor_name:
                    self._actor_recycler.clear_actor_runtime_state(actor_name)
                raise RuntimeError(err) from cause
            explicit_checkpoint_reload = (
                op == "load_weights"
                and isinstance(explicit_checkpoint_path, str)
                and os.path.isdir(explicit_checkpoint_path)
            )
            if explicit_checkpoint_reload:
                self._actor_recycler.mark_poisoned(
                    session.model_id,
                    f"[{session.model_id}] bumblebee actor recycled before explicit load_weights; "
                    "checkpoint reload must complete successfully before training can continue."
                )
                return worker
            err = (
                f"[{session.model_id}] bumblebee actor recycle detected after op={op}; "
                "operation may have partially executed before the crash; "
                "reload from checkpoint before retrying."
            )
            self._actor_recycler.mark_poisoned(session.model_id, err)
            raise RuntimeError(err) from cause

        worker = await self._recycle_dense_actor(session, op=op, cause=cause)
        if op != "load_weights":
            err = (
                f"[{session.model_id}] dense actor recycle detected after op={op}; "
                "operation may have partially executed before the crash; "
                "reload from checkpoint before retrying."
            )
            self._actor_recycler.mark_poisoned(session.model_id, err)
            raise RuntimeError(err) from cause
        return worker

    def _resolve_megatron_base_model(self, session: "TrainingSession") -> tuple[str, str]:
        """Resolve Megatron base model path strictly; never fallback to unrelated defaults."""
        requested_model = session.base_model or self.default_base_model
        if not requested_model:
            raise RuntimeError(f"[{session.model_id}] could not resolve Megatron base model: empty model id")
        if requested_model.startswith("/"):
            return requested_model, requested_model
        resolved = self._resolve_hf_model_path(requested_model)
        if not resolved:
            raise RuntimeError(
                f"[{session.model_id}] could not resolve Megatron base model {requested_model!r} to local path"
            )
        return resolved, requested_model

    def _resolve_bumblebee_base_model(self, session: "TrainingSession") -> tuple[str, str]:
        """Resolve Bumblebee base model path strictly; never fallback to unrelated defaults."""
        requested_model = session.base_model or self.default_base_model
        if not requested_model:
            raise RuntimeError(f"[{session.model_id}] could not resolve Bumblebee base model: empty model id")
        if requested_model.startswith("/"):
            return requested_model, requested_model
        resolved = self._resolve_hf_model_path(requested_model)
        if not resolved:
            raise RuntimeError(
                f"[{session.model_id}] could not resolve Bumblebee base model {requested_model!r} to local path"
            )
        return resolved, requested_model

    def _resolve_session_base_model(self, session: "TrainingSession") -> tuple[str | None, str | None]:
        requested_model = session.base_model or self.default_base_model
        if requested_model and not requested_model.startswith("/"):
            base_model = self._resolve_hf_model_path(requested_model)
            if base_model:
                return base_model, requested_model
            return self.default_base_model, requested_model
        return requested_model, requested_model

    def _build_megatron_distributed_config(
        self,
        *,
        requested_model: str | None,
        base_model: str | None,
    ):
        from mint_server.backend.training.megatron.megatron_distributed import DistributedConfig
        from mint_server.backend.core.model_registry import get_model_config, get_training_parallelism

        cfg = get_model_config(requested_model or base_model or "")
        train_tp, train_pp, train_ep, train_cp, train_etp = get_training_parallelism(
            requested_model or base_model or ""
        )
        return DistributedConfig(
            tensor_parallel_size=train_tp,
            pipeline_parallel_size=train_pp,
            expert_parallel_size=train_ep,
            context_parallel_size=train_cp,
            expert_tensor_parallel_size=train_etp,
            use_fp8=bool(getattr(cfg, "train_use_fp8", False)),
            router_replay_mode=server_config.router_replay_mode,
        )

    def _build_bumblebee_distributed_config(
        self,
        *,
        requested_model: str | None,
        base_model: str | None,
    ):
        from mint_server.backend.training.megatron.megatron_distributed import DistributedConfig
        from mint_server.backend.core.model_registry import get_model_config, get_training_parallelism

        model_key = requested_model or base_model or ""
        cfg = get_model_config(model_key)
        if _is_qwen3_30b_model(model_key):
            train_tp, train_pp, train_ep, train_cp, train_etp = 4, 1, 4, 1, 1
        elif _is_qwen35_model(model_key):
            train_tp, train_pp, train_ep, train_cp, train_etp = 4, 1, 2, 1, 1
        else:
            train_tp, train_pp, train_ep, train_cp, train_etp = get_training_parallelism(model_key)
        return DistributedConfig(
            tensor_parallel_size=train_tp,
            pipeline_parallel_size=train_pp,
            expert_parallel_size=train_ep,
            context_parallel_size=train_cp,
            expert_tensor_parallel_size=train_etp,
            use_fp8=bool(getattr(cfg, "train_use_fp8", False)),
            router_replay_mode=server_config.router_replay_mode,
        )

    async def _rebind_megatron_worker(
        self,
        session: "TrainingSession",
        *,
        reason: str,
        allow_create: bool = True,
    ) -> ray.actor.ActorHandle:
        from mint_server.backend.training.megatron.megatron_distributed import (
            PERSISTENT_NAMESPACE,
            _make_megatron_actor_name,
            async_get_or_create_megatron_worker_group,
        )
        from mint_server.backend.core.model_registry import is_topology_desired_model
        from mint_server.backend.actors.model_actor_inventory import ActorType
        from mint_server.backend.actors.model_actor_publication import BackendModelActorLaunch, publish_backend_model_actor

        base_model, requested_model = self._resolve_megatron_base_model(session)
        actor_name = _make_megatron_actor_name(base_model or requested_model or session.base_model or "")
        actual_rank = session.lora_config.rank if session.lora_config else self.default_lora_rank
        trainer_lora_rank = int(server_config.max_lora_rank)
        distributed_config = self._build_megatron_distributed_config(
            requested_model=requested_model,
            base_model=base_model,
        )

        if allow_create:
            worker = await async_get_or_create_megatron_worker_group(
                base_model=base_model,
                lora_rank=trainer_lora_rank,
                learning_rate=session.learning_rate,
                distributed_config=distributed_config,
                session_id=session.model_id,
                actual_rank=actual_rank,
            )
            ready_timeout_s = (
                float(server_config.training_actor_ready_timeout_s)
                if server_config.training_actor_ready_timeout_s is not None
                else 3600.0
            )
            try:
                await self._await_with_keepalive(
                    worker.__ray_ready__.remote(),
                    session,
                    interval_s=30.0,
                    timeout_s=ready_timeout_s,
                )
            except Exception as e:
                if self._is_dead_actor_error(e):
                    raise RuntimeError(f"[{session.model_id}] missing worker for backend=megatron") from e
                raise
        else:
            try:
                worker = await asyncio.to_thread(
                    ray.get_actor,
                    actor_name,
                    namespace=PERSISTENT_NAMESPACE,
                )
            except ValueError as e:
                raise RuntimeError(f"[{session.model_id}] missing worker for backend=megatron") from e

        publish_backend_model_actor(BackendModelActorLaunch(
            actor_name=actor_name,
            actor_type=ActorType.MEGATRON,
            num_gpus=distributed_config.world_size,
            actor_handle=worker,
            namespace=PERSISTENT_NAMESPACE,
            base_model=base_model or requested_model or "",
            session_id=session.model_id,
            protected=is_topology_desired_model(base_model or requested_model or ""),
        ),
            refresh_observability=False,
        )
        self._actor_recycler.bind_session_actor(session.model_id, actor_name)
        self._workers[session.model_id] = worker
        self._touch_actor(session)
        session.backend = "megatron"
        session.is_active = True
        logger.warning(
            "[%s] megatron worker rebound without recycle actor=%s reason=%s",
            session.model_id,
            actor_name,
            reason,
        )
        return worker

    async def _rebind_bumblebee_worker(
        self,
        session: "TrainingSession",
        *,
        reason: str,
        allow_create: bool = True,
    ) -> ray.actor.ActorHandle:
        from mint_server.backend.training.bumblebee.bumblebee_distributed import (
            PERSISTENT_NAMESPACE,
            _make_bumblebee_actor_name,
            async_get_or_create_bumblebee_worker_group,
        )
        from mint_server.backend.actors.model_actor_inventory import ActorType
        from mint_server.backend.actors.model_actor_publication import BackendModelActorLaunch, publish_backend_model_actor
        from mint_server.backend.core.model_registry import is_topology_desired_model

        base_model, requested_model = self._resolve_bumblebee_base_model(session)
        actor_name = _make_bumblebee_actor_name(base_model or requested_model or session.base_model or "")
        actual_rank = session.lora_config.rank if session.lora_config else self.default_lora_rank
        trainer_lora_rank = int(server_config.max_lora_rank)
        distributed_config = self._build_bumblebee_distributed_config(
            requested_model=requested_model,
            base_model=base_model,
        )

        if allow_create:
            worker = await async_get_or_create_bumblebee_worker_group(
                base_model=base_model,
                lora_rank=trainer_lora_rank,
                learning_rate=session.learning_rate,
                distributed_config=distributed_config,
                session_id=session.model_id,
                actual_rank=actual_rank,
                observability_base_model=requested_model or base_model,
            )
            ready_timeout_s = (
                float(server_config.training_actor_ready_timeout_s)
                if server_config.training_actor_ready_timeout_s is not None
                else 3600.0
            )
            try:
                await self._await_with_keepalive(
                    worker.__ray_ready__.remote(),
                    session,
                    interval_s=30.0,
                    timeout_s=ready_timeout_s,
                )
            except Exception as e:
                if self._is_dead_actor_error(e):
                    raise RuntimeError(f"[{session.model_id}] missing worker for backend=bumblebee") from e
                raise
        else:
            try:
                worker = await asyncio.to_thread(ray.get_actor, actor_name, namespace=PERSISTENT_NAMESPACE)
            except ValueError as e:
                raise RuntimeError(f"[{session.model_id}] missing worker for backend=bumblebee") from e

        publish_backend_model_actor(
            BackendModelActorLaunch(
                actor_name=actor_name,
                actor_type=ActorType.MEGATRON,
                num_gpus=distributed_config.world_size,
                actor_handle=worker,
                namespace=PERSISTENT_NAMESPACE,
                base_model=base_model or requested_model or "",
                session_id=session.model_id,
                protected=is_topology_desired_model(base_model or requested_model or ""),
                metadata={
                    "backend": "bumblebee",
                    "max_lora_rank": trainer_lora_rank,
                    "actual_rank": actual_rank,
                },
            ),
            refresh_observability=False,
        )
        self._actor_recycler.bind_session_actor(session.model_id, actor_name)
        self._workers[session.model_id] = worker
        self._touch_actor(session)
        session.backend = "bumblebee"
        session.actor_name = actor_name
        session.namespace = PERSISTENT_NAMESPACE
        session.is_active = True
        logger.warning(
            "[%s] bumblebee worker rebound actor=%s reason=%s",
            session.model_id,
            actor_name,
            reason,
        )
        return worker

    async def _get_actor_recycle_lock(self, actor_name: str) -> asyncio.Lock:
        return await self._actor_recycler.recycle_lock_for_actor(actor_name)

    @staticmethod
    def _walk_exception_chain(error: BaseException) -> list[BaseException]:
        pending = [error]
        seen: set[int] = set()
        ordered: list[BaseException] = []
        while pending:
            exc = pending.pop()
            exc_id = id(exc)
            if exc_id in seen:
                continue
            seen.add(exc_id)
            ordered.append(exc)
            for attr in ("cause", "__cause__", "__context__"):
                child = getattr(exc, attr, None)
                if isinstance(child, BaseException):
                    pending.append(child)
        return ordered

    @classmethod
    def _is_dead_actor_error(cls, error: BaseException) -> bool:
        dead_types = (
            ray.exceptions.ActorDiedError,
            ray.exceptions.RayActorError,
        )
        for exc in cls._walk_exception_chain(error):
            if isinstance(exc, dead_types):
                return True
        text = " | ".join(f"{type(exc).__name__}: {exc}" for exc in cls._walk_exception_chain(error)).lower()
        return (
            "actordiederror" in text
            or "rayactorerror" in text
            or "the actor died unexpectedly" in text
            or "worker process has died" in text
        )

    @staticmethod
    def _actor_state_name_by_handle(worker: ray.actor.ActorHandle) -> str | None:
        actor_id_obj = getattr(worker, "_actor_id", None)
        actor_id_hex = None
        if actor_id_obj is not None:
            try:
                actor_id_hex = actor_id_obj.hex()
            except Exception:
                actor_id_hex = None
        try:
            info = ray._private.state.actors(actor_id=actor_id_hex) if actor_id_hex else None
        except Exception:
            return None
        if isinstance(info, dict):
            state = info.get("State")
            if isinstance(state, str):
                return state
        return None

    async def _classify_actor_timeout(
        self,
        *,
        worker: ray.actor.ActorHandle,
        actor_name: str,
        namespace: str,
    ) -> str:
        def _inspect() -> str:
            state_name = self._actor_state_name_by_handle(worker)
            if isinstance(state_name, str) and state_name:
                return state_name
            try:
                ray.get_actor(actor_name, namespace=namespace)
            except ValueError:
                return "DEAD"
            except Exception:
                return "UNKNOWN"
            return "ALIVE"

        try:
            return await asyncio.to_thread(_inspect)
        except Exception:
            return "UNKNOWN"

    async def _maybe_collect_worker_cuda_memory_summary(
        self,
        worker: ray.actor.ActorHandle,
    ) -> dict[str, Any] | None:
        timeout_s = float(os.environ.get("MINT_WORKER_CUDA_SUMMARY_TIMEOUT_S", "2.0"))
        try:
            try:
                return await async_get_ray_ref(worker.get_cuda_memory_summary.remote(), timeout_s=timeout_s)
            except AttributeError:
                return await async_get_ray_ref(worker.get_cuda_memory_stats.remote(), timeout_s=timeout_s)
        except Exception as e:
            return {
                "cuda_available": False,
                "error_type": type(e).__name__,
                "error": str(e),
            }

    async def _log_worker_request_context(
        self,
        session: "TrainingSession",
        worker: ray.actor.ActorHandle,
        *,
        op: str,
        stage: str,
        batch_stats: dict[str, int] | None = None,
    ) -> None:
        cuda_summary = await self._maybe_collect_worker_cuda_memory_summary(worker)
        logger.info(
            "[%s] worker_request_context backend=%s op=%s stage=%s batch_size=%s total_tokens=%s max_seq_len=%s cuda_summary=%s",
            session.model_id,
            session.backend,
            op,
            stage,
            None if batch_stats is None else batch_stats.get("batch_size"),
            None if batch_stats is None else batch_stats.get("total_tokens"),
            None if batch_stats is None else batch_stats.get("max_seq_len"),
            cuda_summary,
        )

    async def _recycle_megatron_actor(
        self,
        session: "TrainingSession",
        *,
        op: str,
        cause: BaseException,
    ) -> ray.actor.ActorHandle:
        actor_name = self._actor_name_for_session(session)
        if not actor_name:
            from mint_server.backend.ray_cluster.model_actor_names import megatron_actor_name as _make_megatron_actor_name

            actor_name = _make_megatron_actor_name(session.base_model or "")
        lock = await self._get_actor_recycle_lock(actor_name)
        async with lock:
            self._workers.pop(session.model_id, None)
            return await self._rebind_megatron_worker(
                session,
                reason=f"{op}:{type(cause).__name__}",
                allow_create=True,
            )

    async def _recycle_bumblebee_actor(
        self,
        session: "TrainingSession",
        *,
        op: str,
        cause: BaseException,
    ) -> ray.actor.ActorHandle:
        actor_name = self._actor_name_for_session(session)
        if not actor_name:
            from mint_server.backend.ray_cluster.model_actor_names import bumblebee_actor_name as _make_bumblebee_actor_name

            actor_name = _make_bumblebee_actor_name(session.base_model or "")
        lock = await self._get_actor_recycle_lock(actor_name)
        async with lock:
            self._workers.pop(session.model_id, None)
            return await self._rebind_bumblebee_worker(
                session,
                reason=f"{op}:{type(cause).__name__}",
                allow_create=True,
            )

    async def _recycle_dense_actor(
        self,
        session: "TrainingSession",
        *,
        op: str,
        cause: BaseException,
    ) -> ray.actor.ActorHandle:
        return await self._recover_dense_worker(session, reason=f"{op}:{type(cause).__name__}")

    async def _run_worker_call_with_actor_recycle(
        self,
        session: "TrainingSession",
        *,
        op: str,
        submit_fn,
        batch_stats: dict[str, int] | None = None,
        interval_s: float = 30.0,
        timeout_s: float | None = None,
        allow_recover: bool = False,
        explicit_checkpoint_path: str | None = None,
    ):
        self._raise_if_session_poisoned(session, op=op)
        try:
            worker = await self._get_live_worker(session, op=op, allow_recover=allow_recover)
        except RuntimeError as e:
            if "missing worker" not in str(e):
                raise
            worker = await self._recycle_worker_after_failure(
                session,
                op=op,
                cause=e,
                explicit_checkpoint_path=explicit_checkpoint_path,
            )

        self._touch_actor(session)
        await self._log_worker_request_context(
            session,
            worker,
            op=op,
            stage="before_submit",
            batch_stats=batch_stats,
        )

        attempts = 0
        while True:
            try:
                pending = submit_fn(worker)
                result = await self._await_worker_call(
                    pending,
                    session,
                    op=op,
                    worker=worker,
                    interval_s=interval_s,
                    timeout_s=timeout_s,
                )
                self._note_successful_worker_call(session, op=op)
                return result
            except Exception as e:
                if not self._is_dead_actor_error(e):
                    raise
                if attempts >= 1:
                    raise
                attempts += 1
                logger.warning(
                    "[%s] actor_recycle_retry op=%s attempt=%s error_type=%s",
                    session.model_id,
                    op,
                    attempts,
                    type(e).__name__,
                )
                worker = await self._recycle_worker_after_failure(
                    session,
                    op=op,
                    cause=e,
                    request_started=True,
                    explicit_checkpoint_path=explicit_checkpoint_path,
                )
                await self._log_worker_request_context(
                    session,
                    worker,
                    op=op,
                    stage="after_recycle",
                    batch_stats=batch_stats,
                )

    @staticmethod
    def _iter_exception_chain(error: BaseException) -> list[BaseException]:
        chain: list[BaseException] = []
        seen: set[int] = set()
        current: BaseException | None = error
        while current is not None and id(current) not in seen:
            chain.append(current)
            seen.add(id(current))
            next_error = getattr(current, "cause", None)
            if not isinstance(next_error, BaseException):
                next_error = getattr(current, "__cause__", None)
            if not isinstance(next_error, BaseException):
                next_error = getattr(current, "__context__", None)
            current = next_error if isinstance(next_error, BaseException) else None
        return chain

    @classmethod
    def _format_exception_summary(cls, error: BaseException) -> str:
        parts: list[str] = []
        for exc in cls._iter_exception_chain(error)[:3]:
            msg = str(exc).strip()
            if msg:
                parts.append(f"{type(exc).__name__}: {msg}")
            else:
                parts.append(type(exc).__name__)
        return " | ".join(parts)

    @classmethod
    def _dense_fatal_error_reason(cls, error: BaseException) -> str | None:
        keywords = (
            "acceleratorerror",
            "cuda error",
            "device-side assert",
            "device side assert",
            "illegal memory access",
        )
        for exc in cls._iter_exception_chain(error):
            haystack = f"{type(exc).__name__}: {exc}".lower()
            if any(keyword in haystack for keyword in keywords):
                return cls._format_exception_summary(error)
        return None

    @classmethod
    def _classify_training_failure(cls, error: BaseException) -> tuple[str, str]:
        if isinstance(error, asyncio.TimeoutError):
            return "timeout", "timeout"
        if isinstance(error, getattr(ray.exceptions, "GetTimeoutError", tuple())):
            return "timeout", "timeout"
        if cls._dense_fatal_error_reason(error) is not None:
            return "error", "cuda_fatal"
        if isinstance(error, Exception):
            failure_class = "internal_error"
            for exc in cls._iter_exception_chain(error):
                if not isinstance(exc, Exception):
                    continue
                candidate = classify_failure_reason(exc)
                if candidate == "canceled":
                    return "canceled", candidate
                if candidate == "timeout":
                    return "timeout", candidate
                if candidate != "internal_error":
                    failure_class = candidate
                    break
            return "error", failure_class
        return "error", "internal_error"

    @staticmethod
    def _resolve_actor_node_id(actor_name: str | None) -> str | None:
        if not actor_name:
            return None
        try:
            from mint_server.backend.actors.model_actor_supervisor import get_model_actor_supervisor

            entry = get_model_actor_supervisor().get(str(actor_name))
        except Exception:
            return None
        node_id = getattr(entry, "node_id", None) if entry is not None else None
        if isinstance(node_id, str) and node_id.strip():
            return node_id.strip()
        return None

    async def _handle_dense_worker_failure(
        self,
        session: "TrainingSession",
        *,
        op: str,
        error: BaseException,
        worker: ray.actor.ActorHandle | None = None,
    ) -> None:
        if session.backend != "peft":
            return
        if isinstance(error, asyncio.TimeoutError):
            return
        if isinstance(error, getattr(ray.exceptions, "GetTimeoutError", tuple())):
            return

        fatal_reason = self._dense_fatal_error_reason(error)
        if fatal_reason is None:
            return

        actor_name = str(self._actor_name_for_session(session) or getattr(session, "actor_name", "") or "")
        if not actor_name:
            return

        from mint_server.backend.observability.runtime_observability import runtime_observability

        runtime_observability.record_dense_actor_fatal(
            base_model=session.base_model,
            op=op,
            failure_class="cuda_fatal",
        )
        runtime_observability.record_training_incident(
            kind="dense_actor_fatal",
            base_model=session.base_model,
            backend=session.backend,
            op=op,
            status="error",
            failure_class="cuda_fatal",
            actor_name=actor_name,
            node_id=self._resolve_actor_node_id(actor_name),
            request_id=str(get_request_id() or "") or None,
            session_id=str(getattr(session, "session_id", "") or session.model_id),
            detail=fatal_reason,
        )
        record_span_event_otel(
            "mint.dense_actor.fatal_gpu_error",
            attributes={
                "model_id": str(session.model_id),
                "session_id": str(getattr(session, "session_id", "") or session.model_id),
                "base_model": str(session.base_model or "unknown"),
                "backend": str(session.backend or "unknown"),
                "op": str(op),
                "actor_name": actor_name,
                "node_id": str(self._resolve_actor_node_id(actor_name) or "unknown"),
                "request_id": str(get_request_id() or "-"),
                "error": fatal_reason,
            },
        )

        logger.warning(
            "[%s] retiring dense trainer after fatal op=%s actor_name=%s node_id=%s request_id=%s session_id=%s error=%s",
            session.model_id,
            op,
            actor_name,
            self._resolve_actor_node_id(actor_name) or "unknown",
            get_request_id() or "-",
            getattr(session, "session_id", "") or session.model_id,
            fatal_reason,
        )

        from mint_server.backend.training.dense.dense_trainer import retire_dense_trainer

        retire_outcome = await asyncio.to_thread(
            retire_dense_trainer,
            actor_name=actor_name,
            actor=worker,
            reason=f"{op}:{fatal_reason}",
            base_model=session.base_model,
            session_id=session.model_id,
            fatal_op=op,
            request_id=str(get_request_id() or "") or None,
            namespace=str(getattr(session, "namespace", "") or RAY_NAMESPACE),
        )
        if retire_outcome != "ok":
            retire_failure = (
                f"[{session.model_id}] dense actor retirement failed after fatal op={op}; "
                f"actor_name={actor_name} outcome={retire_outcome}. "
                "Operator must recycle the actor before this session can continue."
            )
            self._actor_recycler.mark_hard_poisoned(session.model_id, retire_failure)
            runtime_observability.record_training_incident(
                kind="dense_actor_retire_failed",
                base_model=session.base_model,
                backend=session.backend,
                op=op,
                status="error",
                failure_class=str(retire_outcome or "unknown"),
                actor_name=actor_name,
                node_id=self._resolve_actor_node_id(actor_name),
                request_id=str(get_request_id() or "") or None,
                session_id=str(getattr(session, "session_id", "") or session.model_id),
                detail=retire_failure,
            )
            logger.error(retire_failure)

        self._workers.pop(session.model_id, None)
        self._actor_recycler.unbind_session_actor(session.model_id)
        if str(getattr(session, "actor_name", "") or "") == actor_name:
            session.actor_name = None
            session.namespace = None

    async def _await_worker_call(
        self,
        awaitable,
        session: "TrainingSession",
        *,
        op: str,
        worker: ray.actor.ActorHandle | None = None,
        interval_s: float = 30.0,
        timeout_s: float | None = None,
    ):
        from mint_server.backend.observability.runtime_observability import runtime_observability

        started = time.monotonic()
        effective_timeout_s = timeout_s
        if effective_timeout_s is None:
            configured_timeout_s = server_config.training_remote_call_timeout_s
            if configured_timeout_s is not None:
                configured_timeout_s = float(configured_timeout_s)
                effective_timeout_s = configured_timeout_s if configured_timeout_s > 0 else None
        try:
            result = await self._await_with_keepalive(
                awaitable,
                session,
                interval_s=interval_s,
                timeout_s=effective_timeout_s,
            )
            runtime_observability.record_training_operation(
                base_model=session.base_model,
                backend=session.backend,
                op=op,
                status="ok",
                failure_class="none",
                duration_s=time.monotonic() - started,
            )
            return result
        except Exception as e:
            status, failure_class = self._classify_training_failure(e)
            actor_name = str(
                self._actor_name_for_session(session)
                or getattr(session, "actor_name", "")
                or ""
            )
            node_id = self._resolve_actor_node_id(actor_name)
            duration_s = time.monotonic() - started
            runtime_observability.record_training_operation(
                base_model=session.base_model,
                backend=session.backend,
                op=op,
                status=status,
                failure_class=failure_class,
                duration_s=duration_s,
            )
            runtime_observability.record_training_incident(
                kind="training_failure",
                base_model=session.base_model,
                backend=session.backend,
                op=op,
                status=status,
                failure_class=failure_class,
                actor_name=actor_name or None,
                node_id=node_id,
                request_id=str(get_request_id() or "") or None,
                session_id=str(getattr(session, "session_id", "") or session.model_id),
                detail=self._format_exception_summary(e),
            )
            logger.warning(
                "[%s] training op failed op=%s backend=%s status=%s failure_class=%s actor_name=%s node_id=%s request_id=%s session_id=%s error=%s",
                session.model_id,
                op,
                session.backend,
                status,
                failure_class,
                actor_name or "-",
                node_id or "unknown",
                get_request_id() or "-",
                getattr(session, "session_id", "") or session.model_id,
                self._format_exception_summary(e),
            )
            await self._handle_dense_worker_failure(
                session,
                op=op,
                error=e,
                worker=worker,
            )
            raise

    def _record_megatron_result_metrics(self, session: "TrainingSession", result: dict | None) -> None:
        if session.backend != "megatron" or not isinstance(result, dict):
            return
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            return
        switch_count = float(metrics.get("session_switch_total:sum", 0.0) or 0.0)
        if switch_count <= 0:
            return
        from mint_server.backend.observability.runtime_observability import runtime_observability

        session_state = "existing" if float(metrics.get("session_switch_existing_session:mean", 0.0) or 0.0) >= 0.5 else "new"
        runtime_observability.record_megatron_session_switch(
            base_model=session.base_model,
            session_state=session_state,
            save_s=float(metrics.get("session_switch_save_s:sum", 0.0) or 0.0),
            swap_s=float(metrics.get("session_switch_swap_s:sum", 0.0) or 0.0),
            load_s=float(metrics.get("session_switch_load_s:sum", 0.0) or 0.0),
            reset_bias_s=float(metrics.get("session_switch_reset_bias_s:sum", 0.0) or 0.0),
            total_s=float(metrics.get("session_switch_total_s:sum", 0.0) or 0.0),
        )

    async def _recover_dense_worker(self, session: "TrainingSession", *, reason: str) -> ray.actor.ActorHandle:
        """Rebind a dense trainer actor after eviction/death."""
        from mint_server.backend.training.dense.dense_trainer import get_or_create_dense_trainer
        from mint_server.backend.observability.runtime_observability import runtime_observability

        model_id = session.model_id
        base_model, name_key = self._resolve_session_base_model(session)
        lora_rank = (
            session.lora_config.rank if session.lora_config else self.default_lora_rank
        )
        dense = await asyncio.to_thread(
            get_or_create_dense_trainer,
            training_worker_cls=TrainingWorker,
            base_model=base_model,
            model_key=name_key,
            lora_rank=lora_rank,
            learning_rate=session.learning_rate,
            session_id=session.model_id,
        )
        self._workers[model_id] = dense.actor
        self._actor_recycler.bind_session_actor(model_id, dense.actor_name)
        session.actor_name = dense.actor_name
        session.namespace = RAY_NAMESPACE
        self._touch_actor(session)
        try:
            from mint_server.backend.stores.training_session_store import upsert_training_session

            upsert_training_session(
                {
                    "model_id": session.model_id,
                    "actor_name": dense.actor_name,
                    "namespace": RAY_NAMESPACE,
                    "backend": session.backend,
                    "current_step": session.current_step,
                    "last_activity": session.last_activity,
                    "metadata_version": getattr(session, "metadata_version", 1),
                }
            )
        except Exception:
            pass
        runtime_observability.record_dense_actor_bind_decision(
            base_model=session.base_model,
            decision="recover",
        )
        runtime_observability.record_training_incident(
            kind="dense_actor_bind_decision",
            base_model=session.base_model,
            backend=session.backend,
            op="bind",
            status="ok",
            failure_class="none",
            actor_name=dense.actor_name,
            request_id=str(get_request_id() or "") or None,
            session_id=str(getattr(session, "session_id", "") or session.model_id),
            detail=f"recover:{reason}",
        )
        logger.warning(
            "[%s] rebound dense trainer actor=%s reason=%s base_model=%s",
            model_id,
            dense.actor_name,
            reason,
            base_model,
        )
        return dense.actor

    async def _rebind_worker_from_session_metadata(
        self,
        session: "TrainingSession",
        *,
        reason: str,
    ) -> ray.actor.ActorHandle | None:
        actor_name = str(getattr(session, "actor_name", "") or "")
        if not actor_name:
            return None
        namespace = str(getattr(session, "namespace", "") or RAY_NAMESPACE)
        from mint_server.backend.observability.runtime_observability import runtime_observability

        if session.backend == "peft":
            from mint_server.backend.training.dense.dense_trainer import dense_trainer_reuse_block_reason

            reuse_block_reason = dense_trainer_reuse_block_reason(actor_name)
            if reuse_block_reason is not None:
                runtime_observability.record_dense_actor_bind_decision(
                    base_model=session.base_model,
                    decision="rebind_refused_poisoned",
                )
                runtime_observability.record_training_incident(
                    kind="dense_actor_bind_decision",
                    base_model=session.base_model,
                    backend=session.backend,
                    op="bind",
                    status="error",
                    failure_class="poisoned_actor",
                    actor_name=actor_name,
                    request_id=str(get_request_id() or "") or None,
                    session_id=str(getattr(session, "session_id", "") or session.model_id),
                    detail=f"rebind_refused_poisoned:{reuse_block_reason}",
                )
                logger.warning(
                    "[%s] refusing to rebind poisoned dense trainer actor=%s reason=%s",
                    str(getattr(session, "model_id", "")),
                    actor_name,
                    reuse_block_reason,
                )
                return None
        try:
            worker = await asyncio.to_thread(ray.get_actor, actor_name, namespace=namespace)
        except Exception as e:
            logger.warning(
                "[%s] failed to rebind actor=%s reason=%s error_type=%s error=%s",
                str(getattr(session, "model_id", "")),
                actor_name,
                reason,
                type(e).__name__,
                e,
            )
            return None
        self._workers[session.model_id] = worker
        self._actor_recycler.bind_session_actor(session.model_id, actor_name)
        self._touch_actor(session)
        if session.backend == "peft":
            runtime_observability.record_dense_actor_bind_decision(
                base_model=session.base_model,
                decision="reuse",
            )
            runtime_observability.record_training_incident(
                kind="dense_actor_bind_decision",
                base_model=session.base_model,
                backend=session.backend,
                op="bind",
                status="ok",
                failure_class="none",
                actor_name=actor_name,
                request_id=str(get_request_id() or "") or None,
                session_id=str(getattr(session, "session_id", "") or session.model_id),
                detail=f"reuse:{reason}",
            )
        logger.warning(
            "[%s] rebound worker actor=%s reason=%s backend=%s",
            session.model_id,
            actor_name,
            reason,
            session.backend,
        )
        return worker

    async def _get_live_worker(
        self,
        session: "TrainingSession",
        *,
        op: str,
        allow_recover: bool = False,
    ) -> ray.actor.ActorHandle:
        """Return a live worker handle, rebinding dense trainers when evicted."""
        self._raise_if_session_poisoned(session, op=op)
        model_id = session.model_id
        worker = self._workers.get(model_id)
        authoritative_actor_name = str(getattr(session, "actor_name", "") or "")
        bound_actor_name = str(self._actor_recycler.actor_name_for_model(model_id) or "")
        if authoritative_actor_name and bound_actor_name and bound_actor_name != authoritative_actor_name:
            self._workers.pop(model_id, None)
            self._actor_recycler.bind_session_actor(model_id, authoritative_actor_name)
            worker = None
        if worker is None:
            worker = await self._rebind_worker_from_session_metadata(session, reason=f"{op}:metadata_rebind")
        if worker is None:
            if session.backend == "megatron":
                return await self._rebind_megatron_worker(
                    session,
                    reason=f"{op}:missing_worker",
                    allow_create=allow_recover,
                )
            if session.backend == "bumblebee":
                return await self._rebind_bumblebee_worker(
                    session,
                    reason=f"{op}:missing_worker",
                    allow_create=allow_recover,
                )
            if session.backend == "peft" and allow_recover:
                return await self._recover_dense_worker(session, reason=f"{op}:missing_worker")
            raise RuntimeError(f"[{model_id}] missing worker for backend={session.backend}")

        if session.backend == "bumblebee":
            try:
                await async_get_ray_ref(worker.heartbeat.remote(), timeout_s=10)
                return worker
            except ray.exceptions.GetTimeoutError:
                self._touch_actor(session)
                return worker
            except Exception as e:
                missing = RuntimeError(
                    f"[{model_id}] missing worker for backend=bumblebee: "
                    f"rank liveness probe failed before op={op}"
                )
                if op == "load_weights":
                    raise missing from e
                return await self._recycle_worker_after_failure(
                    session,
                    op=op,
                    cause=missing,
                    request_started=False,
                )

        # Megatron workers are managed by a persistent group; keep existing behavior.
        if session.backend != "peft":
            return worker

        # Dense trainer can be evicted by supervisor inventory policy between RL stages.
        try:
            await async_get_ray_ref(worker.heartbeat.remote(), timeout_s=5)
            return worker
        except ray.exceptions.GetTimeoutError:
            # Busy actor is still healthy; proceed with original handle.
            self._touch_actor(session)
            return worker
        except (ray.exceptions.ActorDiedError, ray.exceptions.RayActorError) as e:
            if not allow_recover:
                raise RuntimeError(
                    f"[{model_id}] dense worker became unavailable before op={op}; "
                    "refusing automatic recovery because in-memory session state may be lost. "
                    "Reload from a checkpoint before retrying."
                ) from e
            logger.warning(
                "[%s] dense worker unhealthy before op=%s: %s: %s; attempting rebind",
                model_id,
                op,
                type(e).__name__,
                e,
            )
            return await self._recover_dense_worker(
                session,
                reason=f"{op}:{type(e).__name__}",
            )

    def _strict_megatron_save_meta_enabled(self) -> bool:
        """Whether invalid save metadata should fail the request for megatron."""
        return self._checkpoint_ops.strict_megatron_save_meta_enabled()

    def _apply_megatron_loaded_lora_config(
        self,
        session: "TrainingSession",
        meta: dict[str, object],
    ) -> None:
        self._checkpoint_ops.apply_megatron_loaded_lora_config(session, meta)

    def _validate_megatron_load_meta(self, meta: Any, *, op: str) -> dict[str, object]:
        return self._checkpoint_ops.validate_megatron_load_meta(meta, op=op)

    def _update_session_step_monotonic(
        self,
        session: "TrainingSession",
        meta: Any,
        *,
        op: str,
        strict: bool = False,
    ) -> None:
        self._checkpoint_ops.update_session_step_monotonic(
            session,
            meta,
            op=op,
            strict=strict,
        )

    def _update_session_from_load_meta(
        self,
        session: "TrainingSession",
        meta: Any,
        *,
        op: str,
    ) -> None:
        self._checkpoint_ops.update_session_from_load_meta(session, meta, op=op)

    async def _await_with_keepalive(
        self,
        awaitable,
        session: "TrainingSession",
        interval_s: float = 30.0,
        timeout_s: float | None = None,
    ):
        """Await a Ray call while periodically touching supervisor inventory.

        Uses one Ray ObjectRef future so polling-slice timeouts do not cancel
        or restart the underlying Ray wait.
        """
        start = time.time()
        ref_future = _ray_ref_to_future(awaitable)
        try:
            while True:
                self._touch_actor(session)

                wait_s = interval_s
                if timeout_s is not None and timeout_s > 0:
                    remaining = timeout_s - (time.time() - start)
                    if remaining <= 0:
                        logger.warning("ray_call_timed_out_after__ss", model_id=session.model_id)
                        raise asyncio.TimeoutError(f"Ray call timed out after {timeout_s}s")
                    wait_s = min(wait_s, remaining)

                try:
                    return await asyncio.wait_for(asyncio.shield(ref_future), timeout=wait_s)
                except asyncio.TimeoutError:
                    continue
        finally:
            _silence_late_result(ref_future)

    def _resolve_hf_model_path(self, hf_model_id: str) -> str | None:
        """Resolve HuggingFace model ID to local cache path.

        HF cache structure: HF_HOME/hub/models--{org}--{name}/snapshots/{hash}/

        Args:
            hf_model_id: Model ID like "Qwen/Qwen3-30B-A3B-Instruct-2507"

        Returns:
            Local path to model snapshot, or None if not found.
        """
        import os
        from pathlib import Path

        # Explicit path overrides for models with non-standard cache locations
        MODEL_PATH_OVERRIDES = {
            "moonshotai/Kimi-K2-Instruct": "/vePFS-Mindverse/share/huggingface/hub/models--unsloth--Kimi-K2-Instruct-0905-BF16/snapshots/fbaf30b3baf5fdc2b2170ae04f4ff4948b0487cb",
            "moonshotai/Kimi-K2-Thinking": "/vePFS-Mindverse/share/huggingface/hub/models--moonshotai--Kimi-K2-Thinking/snapshots/612681931a8c906ddb349f8ad0f582cb552189cd",
        }

        try:
            from mint_server.backend.qwen36_verl_fsdp2_lora import qwen36_model_path_override

            qwen36_path = qwen36_model_path_override(hf_model_id)
        except Exception:
            qwen36_path = None
        if qwen36_path:
            logger.info("using_qwen3_6_path_override_for__s_____s")
            return qwen36_path

        if hf_model_id in MODEL_PATH_OVERRIDES:
            override_path = MODEL_PATH_OVERRIDES[hf_model_id]
            logger.info("using_path_override_for__s_____s")
            return override_path

        hf_home = os.environ.get("HF_HOME", "/vePFS-Mindverse/share/huggingface")
        # Convert "org/model" to "models--org--model"
        cache_name = "models--" + hf_model_id.replace("/", "--")
        model_dir = Path(hf_home) / "hub" / cache_name / "snapshots"

        if not model_dir.exists():
            logger.warning("model_cache_not_found___s")
            return None

        # Get the latest snapshot (usually only one)
        snapshots = list(model_dir.iterdir())
        if not snapshots:
            logger.warning("no_snapshots_in___s")
            return None

        # Return the first snapshot path
        snapshot_path = str(snapshots[0])
        logger.info("resolved__s_____s")
        return snapshot_path

    async def create_training_session(self, session: TrainingSession) -> None:
        """Create Ray actor for session.

        Routes MoE models to MegatronWorkerGroup, dense models to pooled dense trainers.
        Blocks until GPU is available (Ray queuing).

        Args:
            session: TrainingSession with configuration.
        """
        from mint_server.backend.core.model_registry import get_model_config

        model_id = session.model_id

        # Determine base model path
        requested_model = session.base_model or self.default_base_model

        lora_rank = (
            session.lora_config.rank if session.lora_config else self.default_lora_rank
        )

        model_uses_distributed_training = _uses_distributed_training_backend(requested_model)
        moe_backend = _select_moe_training_backend(requested_model) if model_uses_distributed_training else None
        use_megatron = model_uses_distributed_training and moe_backend == "megatron"
        use_bumblebee = model_uses_distributed_training and moe_backend == "bumblebee"
        use_verl_fsdp2_lora = _uses_verl_fsdp2_lora_backend(requested_model)

        # Resolve model path based on backend
        with start_as_current_span(
            "training.create_model.resolve_base_model",
            component="backend.verl_training",
            op="training.create_model.resolve_base_model",
            request_id=str(get_request_id() or "") or None,
            attributes={
                "model_id": str(model_id),
                "requested_model": str(requested_model) if requested_model is not None else None,
                "use_verl_fsdp2_lora": bool(use_verl_fsdp2_lora),
                "use_megatron": bool(use_megatron),
                "use_bumblebee": bool(use_bumblebee),
                "moe_backend": str(moe_backend or ""),
            },
        ):
            if use_verl_fsdp2_lora and requested_model and not requested_model.startswith("/"):
                base_model = self._resolve_hf_model_path(requested_model)
                if base_model:
                    logger.info("resolved_qwen3_6_verl_fsdp2_lora_model_to_local___s", model_id=model_id)
                else:
                    base_model = requested_model
                    logger.info("using_qwen3_6_verl_fsdp2_lora_model_path_as_requested___s", model_id=model_id)
            elif model_uses_distributed_training:
                resolver = self._resolve_megatron_base_model if use_megatron else self._resolve_bumblebee_base_model
                base_model, requested_model = resolver(session)
                logger.info(
                    "[%s] Resolved %s distributed training model to local: %s (requested=%s)",
                    model_id,
                    moe_backend,
                    base_model,
                    requested_model,
                )
            elif requested_model and not requested_model.startswith("/"):
                # HuggingFace model ID - resolve to local cache path
                base_model = self._resolve_hf_model_path(requested_model)
                if base_model:
                    logger.info("resolved_hf_model_to_local___s", model_id=model_id)
                else:
                    # Fall back to default (works for dense models on same architecture)
                    base_model = self.default_base_model
                    logger.info("using_default_model_path___s__requested___s", model_id=model_id)
            else:
                base_model = requested_model

        observability_base_model = str(requested_model or base_model or "unknown")

        print(
            f"[DEBUG {model_id}] create_training_session start: requested_model={requested_model} moe_backend={moe_backend} base_model={base_model}",
            flush=True,
        )
        observability_base_model = str(requested_model or base_model or "")

        if use_megatron:
            import asyncio
            from mint_server.backend.training.megatron.megatron_distributed import async_get_or_create_megatron_worker_group
            # MoE models need tensor/expert parallelism from model registry
            from mint_server.backend.core.model_registry import get_training_parallelism, get_model_config

            # Get model-specific parallelism and FP8 config from registry
            cfg = get_model_config(requested_model or base_model or "")
            train_tp, train_pp, train_ep, train_cp, train_etp = get_training_parallelism(requested_model or base_model or "")
            use_fp8 = bool(getattr(cfg, "train_use_fp8", False))
            trainer_lora_rank = int(server_config.max_lora_rank)
            distributed_config = self._build_megatron_distributed_config(
                requested_model=requested_model,
                base_model=base_model,
            )
            logger.info("creating_megatronworkergroup_for_moe_model", model_id=model_id, base=base_model, actual_rank=lora_rank, trainer_lora_rank=trainer_lora_rank, TP=train_tp, PP=train_pp, EP=train_ep, CP=train_cp, ETP=train_etp, world_size=distributed_config.world_size, fp8=use_fp8)

            # Get or create persistent Megatron worker group
            # Uses detached Ray actor pattern like vLLM for crash resilience
            # Use async version to avoid blocking uvicorn event loop
            # Issue #44: Pass model_id for unique session state isolation
            megatron_timeout_s = float(server_config.training_megatron_create_timeout_s)
            print(
                f"[DEBUG {model_id}] megatron get_or_create start: timeout_s={megatron_timeout_s}",
                flush=True,
            )
            try:
                worker = await run_async_with_otel_span(
                    "training.create_model.get_or_create_megatron_worker_group",
                    lambda: asyncio.wait_for(
                        async_get_or_create_megatron_worker_group(
                            base_model=base_model,
                            lora_rank=trainer_lora_rank,
                            learning_rate=session.learning_rate,
                            distributed_config=distributed_config,
                            session_id=session.model_id,
                            actual_rank=lora_rank,
                            observability_base_model=observability_base_model,
                        ),
                        timeout=megatron_timeout_s,
                    ),
                    component="backend.verl_training",
                    op="training.create_model.get_or_create_megatron_worker_group",
                    request_id=str(get_request_id() or "") or None,
                    attributes={
                        "model_id": str(model_id),
                        "base_model": str(base_model),
                        "requested_model": str(requested_model) if requested_model is not None else None,
                        "world_size": int(distributed_config.world_size),
                        "train_tp": int(train_tp),
                        "train_pp": int(train_pp),
                        "train_ep": int(train_ep),
                        "train_cp": int(train_cp),
                        "megatron_timeout_s": float(megatron_timeout_s),
                    },
                )
            except asyncio.TimeoutError:
                # Best-effort: kill the persistent Megatron actor to unblock retries.
                from mint_server.backend.ray_cluster.model_actor_names import megatron_actor_name as _make_megatron_actor_name

                actor_name = _make_megatron_actor_name(base_model or requested_model or "")
                try:
                    actor = ray.get_actor(actor_name, namespace=RAY_NAMESPACE)
                    ray_kill.kill(
                        actor,
                        reason="megatron_create_timeout",
                        actor_name=actor_name,
                        namespace=RAY_NAMESPACE,
                        no_restart=True,
                        model_id=model_id,
                        timeout_s=megatron_timeout_s,
                    )
                except Exception:
                    pass
                raise
            print(
                f"[DEBUG {model_id}] megatron get_or_create done",
                flush=True,
            )
            session.backend = "megatron"
            from mint_server.backend.ray_cluster.model_actor_names import megatron_actor_name as _make_megatron_actor_name

            actor_name = _make_megatron_actor_name(base_model or "")
            self._actor_recycler.bind_session_actor(model_id, actor_name)
            session.actor_name = actor_name
            session.namespace = RAY_NAMESPACE
            self._touch_actor(session)
            # Note: reinit_lora_weights is now called inside get_or_create_megatron_worker_group
            # with session_id for proper session state management (Issue #44)
        elif use_bumblebee:
            import asyncio
            from mint_server.backend.training.bumblebee.bumblebee_distributed import (
                _make_bumblebee_actor_name,
                async_get_or_create_bumblebee_worker_group,
            )

            trainer_lora_rank = int(server_config.max_lora_rank)
            distributed_config = self._build_bumblebee_distributed_config(
                requested_model=requested_model,
                base_model=base_model,
            )
            logger.info(
                "[%s] Creating BumblebeeWorkerGroup for MoE model "
                "(base=%s, requested=%s, actual_rank=%s, trainer_lora_rank=%s, "
                "TP=%s, PP=%s, EP=%s, CP=%s, ETP=%s, world_size=%s)",
                model_id,
                base_model,
                requested_model,
                lora_rank,
                trainer_lora_rank,
                distributed_config.tensor_parallel_size,
                distributed_config.pipeline_parallel_size,
                distributed_config.expert_parallel_size,
                distributed_config.context_parallel_size,
                distributed_config.expert_tensor_parallel_size,
                distributed_config.world_size,
            )
            bumblebee_timeout_s = float(server_config.training_megatron_create_timeout_s)
            print(
                f"[DEBUG {model_id}] bumblebee get_or_create start: timeout_s={bumblebee_timeout_s}",
                flush=True,
            )
            try:
                worker = await run_async_with_otel_span(
                    "training.create_model.get_or_create_bumblebee_worker_group",
                    lambda: asyncio.wait_for(
                        async_get_or_create_bumblebee_worker_group(
                            base_model=base_model,
                            lora_rank=trainer_lora_rank,
                            learning_rate=session.learning_rate,
                            distributed_config=distributed_config,
                            session_id=session.model_id,
                            actual_rank=lora_rank,
                            observability_base_model=observability_base_model,
                        ),
                        timeout=bumblebee_timeout_s,
                    ),
                    component="backend.verl_training",
                    op="training.create_model.get_or_create_bumblebee_worker_group",
                    request_id=str(get_request_id() or "") or None,
                    attributes={
                        "model_id": str(model_id),
                        "base_model": str(base_model),
                        "requested_model": str(requested_model) if requested_model is not None else None,
                        "world_size": int(distributed_config.world_size),
                        "train_tp": int(distributed_config.tensor_parallel_size),
                        "train_pp": int(distributed_config.pipeline_parallel_size),
                        "train_ep": int(distributed_config.expert_parallel_size),
                        "train_cp": int(distributed_config.context_parallel_size),
                        "bumblebee_timeout_s": float(bumblebee_timeout_s),
                    },
                )
            except asyncio.TimeoutError:
                actor_name = _make_bumblebee_actor_name(base_model or requested_model or "")
                try:
                    actor = ray.get_actor(actor_name, namespace=RAY_NAMESPACE)
                    ray_kill.kill(
                        actor,
                        reason="bumblebee_create_timeout",
                        actor_name=actor_name,
                        namespace=RAY_NAMESPACE,
                        no_restart=True,
                        model_id=model_id,
                        timeout_s=bumblebee_timeout_s,
                    )
                except Exception:
                    pass
                raise
            print(
                f"[DEBUG {model_id}] bumblebee get_or_create done",
                flush=True,
            )
            session.backend = "bumblebee"
            actor_name = _make_bumblebee_actor_name(base_model or "")
            self._actor_recycler.bind_session_actor(model_id, actor_name)
            session.actor_name = actor_name
            session.namespace = RAY_NAMESPACE
            self._touch_actor(session)
        elif use_verl_fsdp2_lora:
            session.backend = VERL_FSDP2_LORA_BACKEND
            raise NotImplementedError(
                "Qwen3.6-27B is configured for veRL FSDP2 + PEFT LoRA. "
                "The MinT per-request TrainingWorker API cannot materialize this backend yet; "
                "launch it through a veRL job with MINT_QWEN36_VERL_FSDP2_LORA_PATCHES=1."
            )
        else:
            logger.info(
                f"[{model_id}] Using pooled PEFT trainer actors for dense model (base={base_model}, lora_rank={lora_rank})"
            )

            import asyncio
            from mint_server.backend.training.dense.dense_trainer import get_or_create_dense_trainer
            dense_get_timeout_s = float(server_config.training_dense_get_or_create_timeout_s)
            print(
                f"[DEBUG {model_id}] dense get_or_create start: timeout_s={dense_get_timeout_s}",
                flush=True,
            )
            dense = await asyncio.wait_for(
                asyncio.to_thread(
                    get_or_create_dense_trainer,
                    training_worker_cls=TrainingWorker,
                    base_model=base_model,
                    model_key=requested_model,
                    lora_rank=lora_rank,
                    max_lora_rank=int(server_config.max_lora_rank),
                    learning_rate=session.learning_rate,
                    session_id=session.model_id,
                ),
                timeout=dense_get_timeout_s,
            )
            print(
                f"[DEBUG {model_id}] dense get_or_create done: max_rank={dense.max_lora_rank}",
                flush=True,
            )
            worker = dense.actor
            self._actor_recycler.bind_session_actor(model_id, dense.actor_name)
            self._touch_actor(session)

            # Reinitialize LoRA weights for fresh session (statelessness)
            # This ensures each new session starts with fresh random weights
            # instead of inheriting trained weights from previous session
            logger.info("reinitializing_lora_weights_for_new_session", model_id=model_id, lr=session.learning_rate)
            reinit_timeout_s = float(server_config.training_reinit_lora_timeout_s)
            effective_reinit_timeout_s = reinit_timeout_s if reinit_timeout_s > 0 else None
            print(
                f"[DEBUG {model_id}] dense reinit_lora_weights start: timeout_s={effective_reinit_timeout_s}",
                flush=True,
            )
            traceparent = get_current_traceparent()
            try:
                result = await self._await_worker_call(
                    worker.reinit_lora_weights.remote(
                        session.learning_rate,
                        actual_rank=lora_rank,
                        traceparent=traceparent,
                    ),
                    session,
                    op="create_training_session.reinit_lora_weights",
                    worker=worker,
                    interval_s=30.0,
                    timeout_s=effective_reinit_timeout_s,
                )
            except Exception:
                self._actor_recycler.unbind_session_actor(model_id)
                self._workers.pop(model_id, None)
                try:
                    from mint_server.backend.actors.model_actor_supervisor import get_model_actor_supervisor

                    get_model_actor_supervisor().clear_session(model_id)
                except Exception:
                    pass
                raise
            print(
                f"[DEBUG {model_id}] dense reinit_lora_weights done",
                flush=True,
            )
            logger.info("lora_weights_reinitialized___s_params", model_id=model_id, lr_updated=result.get('reinit_count', 0))

            session.backend = "peft"
            logger.info("dense_trainer_ready_for__s", model_id=model_id, max_rank=base_model)

        # Optional non-blocking create for Megatron actors.
        # Default remains blocking __ray_ready__ wait for deterministic API
        # semantics and unit-test contract compatibility.
        if session.backend == "megatron":
            skip_ready_wait = os.environ.get("MINT_MEGATRON_SKIP_CREATE_READY_WAIT", "0").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if skip_ready_wait:
                actor_name = self._actor_recycler.actor_name_for_model(model_id)
                if actor_name:
                    from mint_server.backend.actors.model_actor_publication import mark_backend_model_actor_ready

                    mark_backend_model_actor_ready(actor_name)

                self._workers[model_id] = worker
                session.is_active = True
                logger.info(
                    "[%s] TrainingWorker ready (backend=%s, skip_ready_wait=1)",
                    model_id,
                    session.backend,
                )
                return

        # Wait for actor to be ready (model loaded)
        # Use await instead of ray.get() to not block the event loop
        default_ready_timeout_s = 3600.0 if session.backend in _DISTRIBUTED_MOE_BACKENDS else 900.0
        ready_timeout_s = (
            float(server_config.training_actor_ready_timeout_s)
            if server_config.training_actor_ready_timeout_s is not None
            else default_ready_timeout_s
        )
        print(
            f"[DEBUG {model_id}] __ray_ready__ start: timeout_s={ready_timeout_s}",
            flush=True,
        )
        try:
            await self._await_worker_call(
                worker.__ray_ready__.remote(),
                session,
                op="create_training_session.ready",
                worker=worker,
                interval_s=30.0,
                timeout_s=ready_timeout_s,
            )
        except Exception:
            self._actor_recycler.unbind_session_actor(model_id)
            self._workers.pop(model_id, None)
            try:
                from mint_server.backend.actors.model_actor_supervisor import get_model_actor_supervisor

                get_model_actor_supervisor().clear_session(model_id)
            except Exception:
                pass
            raise
        print(
            f"[DEBUG {model_id}] __ray_ready__ done",
            flush=True,
        )
        actor_name = self._actor_name_for_session(session)
        if actor_name:
            from mint_server.backend.actors.model_actor_publication import mark_backend_model_actor_ready

            mark_backend_model_actor_ready(actor_name)

        self._workers[model_id] = worker
        session.is_active = True
        logger.info("trainingworker_ready", model_id=model_id, backend=session.backend)

    async def forward_backward(
        self,
        session: TrainingSession,
        request: Any,
    ) -> dict:
        """Remote call to worker for forward + backward pass.

        Args:
            session: TrainingSession.
            request: ForwardBackwardRequest with training data.

        Returns:
            Dict with loss_fn_outputs and metrics.
        """
        self._raise_if_session_poisoned(session, op="forward_backward")
        model_id = session.model_id
        worker = None
        if session.backend != "bumblebee":
            worker = await self._get_live_worker(session, op="forward_backward")

            # Mark actor as recently used for supervisor inventory and admin visibility.
            self._touch_actor(session)
            await self._ensure_megatron_session_guard_clean(
                session,
                op="forward_backward",
                worker=worker,
            )

        # Serialize data for Ray
        data_items = [item.model_dump() for item in request.forward_backward_input.data]
        loss_fn = request.forward_backward_input.loss_fn
        loss_fn_config = dict(request.forward_backward_input.loss_fn_config or {})
        session_rollout_corr = getattr(session, "rollout_correction_config", None)
        rollout_correction_config = None
        if loss_fn in ("ppo", "importance_sampling") and isinstance(session_rollout_corr, dict):
            if session.backend not in _DISTRIBUTED_MOE_BACKENDS:
                raise ValueError(
                    "session-level rollout_correction_config is only supported on distributed MoE backends "
                    f"(got backend={session.backend!r})"
                )
            rollout_correction_config = dict(session_rollout_corr)
            logger.info(
                "[%s] Applied session rollout_correction_config: loss_fn=%r config=%s",
                session.model_id,
                loss_fn,
                rollout_correction_config,
            )

        lora_cfg = getattr(session, "lora_config", None)
        train_attn = True if lora_cfg is None else bool(getattr(lora_cfg, "train_attn", True))
        train_mlp = True if lora_cfg is None else bool(getattr(lora_cfg, "train_mlp", True))
        train_unembed = True if lora_cfg is None else bool(getattr(lora_cfg, "train_unembed", True))
        traceparent = get_current_traceparent()

        if session.backend == "bumblebee":
            result = await self._run_worker_call_with_actor_recycle(
                session,
                op="forward_backward",
                submit_fn=lambda call_worker: call_worker.forward_backward.remote(
                    data_items,
                    loss_fn,
                    loss_fn_config,
                    rollout_correction_config,
                    session.model_id,
                    session.lora_config.rank if session.lora_config else None,
                    traceparent=traceparent,
                    train_attn=train_attn,
                    train_mlp=train_mlp,
                    train_unembed=train_unembed,
                ),
                interval_s=30.0,
            )
        else:
            assert worker is not None
            # Remote call - pass session_id for stateless trainer pattern
            if session.backend in _DISTRIBUTED_MOE_BACKENDS:
                pending = worker.forward_backward.remote(
                    data_items,
                    loss_fn,
                    loss_fn_config,
                    rollout_correction_config,
                    session.model_id,
                    session.lora_config.rank if session.lora_config else None,
                    traceparent=traceparent,
                    train_attn=train_attn,
                    train_mlp=train_mlp,
                    train_unembed=train_unembed,
                )
            else:
                pending = worker.forward_backward.remote(
                    data_items,
                    loss_fn,
                    loss_fn_config,
                    session.model_id,
                    session.lora_config.rank if session.lora_config else None,
                    traceparent=traceparent,
                )
            result = await self._await_worker_call(
                pending,
                session,
                op="forward_backward",
                worker=worker,
                interval_s=30.0,
            )

        # Update session state
        session.accumulated_gradients += 1
        self._record_megatron_result_metrics(session, result)

        logger.info("forward_backward_completed", model_id=model_id, loss_fn=loss_fn)
        return result

    async def forward_backward_reverse_kl(
        self,
        session: TrainingSession,
        request: Any,
    ) -> dict:
        """Remote call to worker for Mint reverse-KL forward/backward."""
        model_id = session.model_id
        worker = await self._get_live_worker(session, op="forward_backward_reverse_kl")
        self._touch_actor(session)

        data_items = [item.model_dump() for item in request.data]
        from mint_server.backend.core.mintx_ops import build_scoring_sequence, parse_reverse_kl_item

        reference_items = []
        for item in data_items:
            batch = parse_reverse_kl_item(item, input_key="reference_input")
            scoring_input, completion_start = build_scoring_sequence(
                batch.prefix_tokens,
                batch.completion_tokens,
            )
            full_targets = scoring_input[1:] + [int(batch.completion_tokens[-1])]
            full_weights = [0.0] * completion_start + [float(x) for x in batch.weights]
            reference_items.append(
                {
                    "model_input": {"chunks": [{"type": "encoded_text", "tokens": scoring_input}]},
                    "loss_fn_inputs": {
                        "target_tokens": {
                            "data": full_targets,
                            "shape": [len(full_targets)],
                            "dtype": "int64",
                        },
                        "weights": {
                            "data": full_weights,
                            "shape": [len(full_weights)],
                            "dtype": "float32",
                        },
                    },
                }
            )
        lora_cfg = getattr(session, "lora_config", None)
        train_attn = True if lora_cfg is None else bool(getattr(lora_cfg, "train_attn", True))
        train_mlp = True if lora_cfg is None else bool(getattr(lora_cfg, "train_mlp", True))
        train_unembed = True if lora_cfg is None else bool(getattr(lora_cfg, "train_unembed", True))
        traceparent = get_current_traceparent()

        if session.backend == "megatron":
            import hashlib


            ref_session_id = f"mintx_ref_{hashlib.md5(request.reference_model_path.encode('utf-8')).hexdigest()[:16]}"
            reference_actual_rank = None
            try:
                with open(os.path.join(request.reference_model_path, "adapter_config.json"), encoding="utf-8") as f:
                    ref_cfg = json.load(f)
                if isinstance(ref_cfg.get("r"), int):
                    reference_actual_rank = int(ref_cfg["r"])
            except Exception:
                reference_actual_rank = None

            try:
                logger.info(
                    f"[{model_id}] reverse_kl prime reference session start: "
                    f"ref_session_id={ref_session_id} actual_rank={reference_actual_rank}"
                )
                await async_get_ray_ref(
                    worker.prime_session_checkpoint.remote(
                        ref_session_id,
                        request.reference_model_path,
                        step_count=0,
                        learning_rate=0.0,
                        actual_rank=reference_actual_rank,
                    ),
                )
                logger.info("reverse_kl_prime_reference_session_done", model_id=model_id, ref_session_id=ref_session_id)
                logger.info("reverse_kl_reference_forward_start", model_id=model_id, ref_session_id=ref_session_id)
                reference_chunks = await async_get_ray_ref(
                    worker.forward_reference_full_log_probs.remote(
                        data_items=reference_items,
                        temperature=float(request.temperature),
                        session_id=ref_session_id,
                        actual_rank=reference_actual_rank,
                        traceparent=traceparent,
                        train_attn=train_attn,
                        train_mlp=train_mlp,
                        train_unembed=train_unembed,
                    ),
                )
                logger.info(
                    f"[{model_id}] reverse_kl reference forward done: ref_session_id={ref_session_id} "
                    f"chunks={len(reference_chunks) if isinstance(reference_chunks, list) else 'unknown'}"
                )
                logger.info("reverse_kl_student_backward_start", model_id=model_id)
                pending = worker.forward_backward_reverse_kl.remote(
                    data_items,
                    None,
                    float(request.temperature),
                    session.model_id,
                    session.lora_config.rank if session.lora_config else None,
                    traceparent=traceparent,
                    train_attn=train_attn,
                    train_mlp=train_mlp,
                    train_unembed=train_unembed,
                    reference_full_log_prob_chunks=reference_chunks,
                )
                result = await self._await_worker_call(
                    pending,
                    session,
                    op="forward_backward_reverse_kl",
                    worker=worker,
                    interval_s=30.0,
                )
            finally:
                try:
                    await async_get_ray_ref(worker.delete_session.remote(ref_session_id, traceparent=traceparent))
                except Exception:
                    logger.warning(
                        "[%s] reverse_kl reference session cleanup failed: ref_session_id=%s",
                        model_id,
                        ref_session_id,
                        exc_info=True,
                    )
        elif session.backend == "bumblebee":
            pending = worker.forward_backward_reverse_kl.remote(
                data_items,
                request.reference_model_path,
                float(request.temperature),
                session.model_id,
                session.lora_config.rank if session.lora_config else None,
                traceparent=traceparent,
                train_attn=train_attn,
                train_mlp=train_mlp,
                train_unembed=train_unembed,
                preserve_current_gradients=bool(session.accumulated_gradients > 0),
            )
            result = await self._await_worker_call(
                pending,
                session,
                op="forward_backward_reverse_kl",
                worker=worker,
                interval_s=30.0,
            )
        else:
            pending = worker.forward_backward_reverse_kl.remote(
                data_items,
                request.reference_model_path,
                float(request.temperature),
                session.model_id,
                session.lora_config.rank if session.lora_config else None,
                traceparent=traceparent,
            )
            result = await self._await_worker_call(
                pending,
                session,
                op="forward_backward_reverse_kl",
                worker=worker,
                interval_s=30.0,
            )
        session.accumulated_gradients += 1
        logger.info("forward_backward_reverse_kl_completed", model_id=model_id)
        return result

    async def forward(
        self,
        session: TrainingSession,
        request: Any,
    ) -> dict:
        """Remote call to worker for forward pass only (no backward).

        Args:
            session: TrainingSession.
            request: ForwardRequest with forward_input field.

        Returns:
            Dict with loss_fn_outputs (including logprobs) and metrics.
        """
        model_id = session.model_id
        worker = await self._get_live_worker(session, op="forward")

        # Mark actor as recently used for supervisor inventory and admin visibility.
        self._touch_actor(session)
        await self._ensure_megatron_session_guard_clean(
            session,
            op="forward",
            worker=worker,
        )

        # Serialize data for Ray
        # ForwardRequest uses forward_input (not forward_backward_input)
        data_items = [item.model_dump() for item in request.forward_input.data]

        lora_cfg = getattr(session, "lora_config", None)
        train_attn = True if lora_cfg is None else bool(getattr(lora_cfg, "train_attn", True))
        train_mlp = True if lora_cfg is None else bool(getattr(lora_cfg, "train_mlp", True))
        train_unembed = True if lora_cfg is None else bool(getattr(lora_cfg, "train_unembed", True))
        traceparent = get_current_traceparent()

        # Remote call - pass session_id for stateless trainer pattern
        if session.backend in _DISTRIBUTED_MOE_BACKENDS:
            pending = worker.forward.remote(
                data_items,
                session.model_id,
                session.lora_config.rank if session.lora_config else None,
                traceparent=traceparent,
                train_attn=train_attn,
                train_mlp=train_mlp,
                train_unembed=train_unembed,
            )
        else:
            pending = worker.forward.remote(data_items, session.model_id, traceparent=traceparent)
        result = await self._await_worker_call(
            pending,
            session,
            op="forward",
            worker=worker,
            interval_s=30.0,
        )
        self._record_megatron_result_metrics(session, result)

        logger.info("forward_completed", model_id=model_id)
        return result

    async def get_tokenizer_info(
        self,
        session: TrainingSession,
    ) -> dict:
        """Get tokenizer info from worker.

        Args:
            session: TrainingSession.

        Returns:
            Dict with tokenizer configuration.
        """
        worker = await self._get_live_worker(session, op="get_tokenizer_info")

        result = await worker.get_tokenizer_info.remote()

        return result

    async def get_session_guard_state(
        self,
        session: TrainingSession,
    ) -> dict:
        if session.backend != "megatron":
            return {
                "session_id": session.model_id,
                "contaminated": False,
                "blocked": False,
                "contamination_reason": None,
                "block_reason": None,
                "external_checkpoint": None,
                "trusted_recovery_baseline": None,
            }
        worker = await self._get_live_worker(session, op="get_session_guard_state")
        get_guard_state = getattr(worker, "get_session_guard_state", None)
        if get_guard_state is None:
            return {
                "session_id": session.model_id,
                "contaminated": False,
                "blocked": False,
                "contamination_reason": None,
                "block_reason": None,
                "external_checkpoint": None,
                "trusted_recovery_baseline": None,
            }
        self._touch_actor(session)
        state = await self._await_with_keepalive(
            get_guard_state.remote(session.model_id),
            session,
            interval_s=30.0,
            timeout_s=30.0,
        )
        if not isinstance(state, dict):
            raise RuntimeError(
                f"[{session.model_id}] invalid get_session_guard_state payload type={type(state).__name__}"
            )
        return state

    async def optim_step(
        self,
        session: TrainingSession,
        request: Any,
    ) -> dict:
        """Remote call to worker for optimizer step.

        Args:
            session: TrainingSession.
            request: OptimStepRequest with adam_params.

        Returns:
            Dict with metrics.
        """
        self._raise_if_session_poisoned(session, op="optim_step")
        model_id = session.model_id
        worker = None
        if session.backend != "bumblebee":
            worker = await self._get_live_worker(session, op="optim_step")

            # Mark actor as recently used for supervisor inventory and admin visibility.
            self._touch_actor(session)
            await self._ensure_megatron_session_guard_clean(
                session,
                op="optim_step",
                worker=worker,
            )

        # Extract learning rate
        lr = request.adam_params.learning_rate if request.adam_params else None
        if lr is not None:
            session.learning_rate = lr

        lora_cfg = getattr(session, "lora_config", None)
        train_attn = True if lora_cfg is None else bool(getattr(lora_cfg, "train_attn", True))
        train_mlp = True if lora_cfg is None else bool(getattr(lora_cfg, "train_mlp", True))
        train_unembed = True if lora_cfg is None else bool(getattr(lora_cfg, "train_unembed", True))
        traceparent = get_current_traceparent()

        if session.backend == "bumblebee":
            result = await self._run_worker_call_with_actor_recycle(
                session,
                op="optim_step",
                submit_fn=lambda call_worker: call_worker.optim_step.remote(
                    lr,
                    session.model_id,
                    session.lora_config.rank if session.lora_config else None,
                    traceparent=traceparent,
                    train_attn=train_attn,
                    train_mlp=train_mlp,
                    train_unembed=train_unembed,
                ),
                interval_s=30.0,
            )
        else:
            assert worker is not None
            # Remote call - pass session_id for stateless trainer pattern
            if session.backend in _DISTRIBUTED_MOE_BACKENDS:
                pending = worker.optim_step.remote(
                    lr,
                    session.model_id,
                    session.lora_config.rank if session.lora_config else None,
                    traceparent=traceparent,
                    train_attn=train_attn,
                    train_mlp=train_mlp,
                    train_unembed=train_unembed,
                )
            else:
                pending = worker.optim_step.remote(
                    lr,
                    session.model_id,
                    session.lora_config.rank if session.lora_config else None,
                    traceparent=traceparent,
                )
            result = await self._await_worker_call(
                pending,
                session,
                op="optim_step",
                worker=worker,
                interval_s=30.0,
            )

        # Update session state
        session.current_step += 1
        session.accumulated_gradients = 0

        # Add step to result metrics
        result["metrics"]["step"] = session.current_step
        self._record_megatron_result_metrics(session, result)

        logger.info("optim_step", model_id=model_id, step=session.current_step)
        return result

    async def train_step(
        self,
        session: TrainingSession,
        request: Any,
    ) -> dict:
        """Combined forward_backward + optim_step in a single call.

        This is the correct way to train MoE models with param_offload=True.
        Keeping both operations in a single remote call ensures they run in
        the same train_mode context, so gradients survive for the optimizer step.

        Args:
            session: TrainingSession.
            request: ForwardBackwardRequest with training data.

        Returns:
            Dict with loss_fn_outputs, metrics, and optimizer results.
        """
        from mint_server.backend.core.model_registry import get_model_config

        self._raise_if_session_poisoned(session, op="train_step")
        model_id = session.model_id
        worker = None
        if session.backend != "bumblebee":
            worker = await self._get_live_worker(session, op="train_step")

            # Mark actor as recently used for supervisor inventory and admin visibility.
            self._touch_actor(session)
            await self._ensure_megatron_session_guard_clean(
                session,
                op="train_step",
                worker=worker,
            )

        # Serialize data for Ray
        data_items = [item.model_dump() for item in request.forward_backward_input.data]
        loss_fn = request.forward_backward_input.loss_fn
        loss_fn_config = dict(request.forward_backward_input.loss_fn_config or {})
        session_rollout_corr = getattr(session, "rollout_correction_config", None)
        rollout_correction_config = None
        if loss_fn in ("ppo", "importance_sampling") and isinstance(session_rollout_corr, dict):
            if session.backend not in _DISTRIBUTED_MOE_BACKENDS:
                raise ValueError(
                    "session-level rollout_correction_config is only supported on distributed MoE backends "
                    f"(got backend={session.backend!r})"
                )
            rollout_correction_config = dict(session_rollout_corr)
            logger.info(
                "[%s] Applied session rollout_correction_config: loss_fn=%r config=%s",
                session.model_id,
                loss_fn,
                rollout_correction_config,
            )
        lr = request.adam_params.learning_rate if request.adam_params else session.learning_rate
        session.learning_rate = lr

        lora_cfg = getattr(session, "lora_config", None)
        train_attn = True if lora_cfg is None else bool(getattr(lora_cfg, "train_attn", True))
        train_mlp = True if lora_cfg is None else bool(getattr(lora_cfg, "train_mlp", True))
        train_unembed = True if lora_cfg is None else bool(getattr(lora_cfg, "train_unembed", True))
        traceparent = get_current_traceparent()

        # Only MoE models use the combined MegatronWorkerGroup.train_step path.
        # Avoid importing megatron_training on CPU-only API hosts (Aliyun gateway).
        is_moe = False
        try:
            is_moe = bool(get_model_config(session.base_model or "").is_moe)
        except Exception:
            is_moe = False
        use_train_step = session.backend == "bumblebee" or (session.backend in _DISTRIBUTED_MOE_BACKENDS and is_moe)

        if use_train_step:
            if session.backend == "bumblebee":
                result = await self._run_worker_call_with_actor_recycle(
                    session,
                    op="train_step",
                    submit_fn=lambda call_worker: call_worker.train_step.remote(
                        data_items,
                        loss_fn,
                        loss_fn_config,
                        rollout_correction_config,
                        lr,
                        session.model_id,
                        session.lora_config.rank if session.lora_config else None,
                        traceparent=traceparent,
                        train_attn=train_attn,
                        train_mlp=train_mlp,
                        train_unembed=train_unembed,
                    ),
                    interval_s=30.0,
                )
            else:
                assert worker is not None
                # MoE: Use combined train_step to keep gradients in same context
                pending = worker.train_step.remote(
                    data_items,
                    loss_fn,
                    loss_fn_config,
                    rollout_correction_config,
                    lr,
                    session.model_id,
                    session.lora_config.rank if session.lora_config else None,
                    traceparent=traceparent,
                    train_attn=train_attn,
                    train_mlp=train_mlp,
                    train_unembed=train_unembed,
                )
                result = await self._await_worker_call(
                    pending,
                    session,
                    op="train_step",
                    worker=worker,
                    interval_s=30.0,
                )
        else:
            assert worker is not None
            # Dense models: Use separate calls (they don't have param_offload issues)
            # Pass session_id for stateless trainer pattern
            if session.backend in _DISTRIBUTED_MOE_BACKENDS:
                fb_pending = worker.forward_backward.remote(
                    data_items,
                    loss_fn,
                    loss_fn_config,
                    rollout_correction_config,
                    session.model_id,
                    session.lora_config.rank if session.lora_config else None,
                    traceparent=traceparent,
                    train_attn=train_attn,
                    train_mlp=train_mlp,
                    train_unembed=train_unembed,
                )
            else:
                fb_pending = worker.forward_backward.remote(
                    data_items,
                    loss_fn,
                    loss_fn_config,
                    session.model_id,
                    session.lora_config.rank if session.lora_config else None,
                    traceparent=traceparent,
                )
            fb_result = await self._await_worker_call(
                fb_pending,
                session,
                op="forward_backward",
                worker=worker,
                interval_s=30.0,
            )
            if session.backend in _DISTRIBUTED_MOE_BACKENDS:
                opt_pending = worker.optim_step.remote(
                    lr,
                    session.model_id,
                    session.lora_config.rank if session.lora_config else None,
                    traceparent=traceparent,
                    train_attn=train_attn,
                    train_mlp=train_mlp,
                    train_unembed=train_unembed,
                )
            else:
                opt_pending = worker.optim_step.remote(
                    lr,
                    session.model_id,
                    session.lora_config.rank if session.lora_config else None,
                    traceparent=traceparent,
                )
            opt_result = await self._await_worker_call(
                opt_pending,
                session,
                op="optim_step",
                worker=worker,
                interval_s=30.0,
            )

            # Merge results
            result = fb_result.copy()
            if "metrics" not in result:
                result["metrics"] = {}
            result["metrics"].update(opt_result.get("metrics", {}))

        # Update session state
        session.current_step += 1
        session.accumulated_gradients = 0

        # Ensure step is in metrics
        if "metrics" not in result:
            result["metrics"] = {}
        result["metrics"]["step"] = session.current_step
        self._record_megatron_result_metrics(session, result)

        logger.info("train_step", model_id=model_id, step=session.current_step)
        return result

    async def reset_expert_bias(self, session: TrainingSession) -> dict:
        """Reset expert_bias buffers in MoE router modules.

        The expert_bias buffer accumulates during training (via finalize_model_grads)
        to balance token distribution across experts. However, this buffer is NOT
        exported with LoRA weights, causing train-inference mismatch:
        - Megatron (trained): has accumulated expert_bias != 0
        - vLLM (loaded LoRA): has expert_bias = 0

        This causes different routing decisions and thus different logprobs even
        with identical LoRA weights.

        Call this before computing logprobs to ensure consistent behavior with vLLM.

        Args:
            session: Training session with model.

        Returns:
            dict with modules_reset count.
        """
        model_id = session.model_id
        try:
            worker = await self._get_live_worker(session, op="reset_expert_bias")
        except Exception:
            logger.warning("reset_expert_bias__no_live_worker___s___s", model_id=model_id)
            return {"modules_reset": 0}

        # Mark actor as recently used
        self._touch_actor(session)
        await self._ensure_megatron_session_guard_clean(
            session,
            op="reset_expert_bias",
            worker=worker,
        )

        logger.info("reset_expert_bias__calling_worker", model_id=model_id)

        try:
            traceparent = get_current_traceparent()
            result_ref = worker.reset_expert_bias.remote(traceparent=traceparent)
            result = await async_get_ray_ref(result_ref)
            # MegatronWorkerGroup returns 'reset_count', normalize to 'modules_reset'
            modules_reset = result.get("reset_count", result.get("modules_reset", 0))
            logger.info("reset_expert_bias__reset__s_modules", model_id=model_id)
            return {"modules_reset": modules_reset}
        except Exception as e:
            logger.exception(f"[{model_id}] reset_expert_bias failed: {e}")
            return {"modules_reset": 0, "error": str(e)}

    async def save_weights_for_sampler(
        self,
        session: TrainingSession,
        checkpoint_name: str,
        checkpoint_base_dir: str,
        use_per_expert_lora: bool = False,
        checkpoint_type: str | None = None,
    ) -> str:
        """Save LoRA weights for inference use.

        Delegates to save_weights with constructed path.

        Args:
            session: Training session with model.
            checkpoint_name: Name for this checkpoint.
            checkpoint_base_dir: Base directory for checkpoints.

        Returns:
            Absolute path to saved checkpoint directory.
        """
        import os

        if use_per_expert_lora:
            raise ValueError("Dense/megatron backend does not support per-expert LoRA sampler export")

        save_path = os.path.join(checkpoint_base_dir, session.model_id, checkpoint_name)
        if checkpoint_type:
            save_path = os.path.join(save_path, checkpoint_type)
        if session.backend in _DISTRIBUTED_MOE_BACKENDS:
            return await self.save_lora_weights_for_sampler(session, save_path)
        return await self.save_dense_lora_weights_for_sampler(session, save_path)

    async def save_dense_lora_weights_for_sampler(
        self,
        session: TrainingSession,
        save_path: str,
    ) -> str:
        """Save minimal PEFT LoRA artifacts for sampling on the dense backend.

        This intentionally excludes optimizer/resume artifacts.
        """
        import os

        model_id = session.model_id
        worker = await self._get_live_worker(session, op="save_dense_lora_weights_for_sampler")
        abs_path = os.path.abspath(save_path)

        try:
            from mint_server.backend.core.model_registry import get_model_config

            train_gpus = get_model_config(session.base_model).train_gpus
        except Exception:
            train_gpus = 1

        if train_gpus >= 32:
            default_timeout_s = 3600
        elif train_gpus >= 16:
            default_timeout_s = 1800
        elif train_gpus >= 4:
            default_timeout_s = 600
        else:
            default_timeout_s = 300
        timeout_s = int(os.environ.get("MINT_SAVE_LORA_TIMEOUT_S", str(default_timeout_s)))

        traceparent = get_current_traceparent()
        ref = worker.save_lora_weights.remote(
            abs_path,
            traceparent=traceparent,
            session_id=session.model_id,
            actual_rank=session.lora_config.rank if session.lora_config else None,
        )
        result = await run_async_with_otel_span(
            "training.save_weights_for_sampler.remote_save",
            lambda: self._await_worker_call(
                ref,
                session,
                op="save_dense_lora_weights_for_sampler",
                worker=worker,
                interval_s=30.0,
                timeout_s=timeout_s,
            ),
            component="backend.verl_training",
            op="training.save_weights_for_sampler.remote_save",
            request_id=str(get_request_id() or "") or None,
            attributes={
                "model_id": str(model_id),
                "base_model": str(session.base_model),
                "backend": str(session.backend),
                "save_path": str(abs_path),
                "timeout_s": int(timeout_s),
            },
        )

        if isinstance(result, dict) and "state_dict" in result and "peft_config" in result:
            import json

            from safetensors.torch import save_file

            os.makedirs(abs_path, exist_ok=True)
            save_file(result["state_dict"], os.path.join(abs_path, "adapter_model.safetensors"))
            with open(os.path.join(abs_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                json.dump(result["peft_config"], f, indent=2)
            training_meta = {
                "current_step": int(result.get("current_step", session.current_step)),
                "learning_rate": float(result.get("learning_rate", session.learning_rate)),
            }
            with open(os.path.join(abs_path, "training_meta.json"), "w", encoding="utf-8") as f:
                json.dump(training_meta, f, indent=2)

        logger.info("save_dense_lora_weights_for_sampler___s", model_id=model_id)
        return abs_path

    async def save_lora_weights_for_sampler(
        self,
        session: TrainingSession,
        save_path: str,
        use_per_expert_lora: bool = False,
    ) -> str:
        """Save minimal PEFT LoRA artifacts for sampling (no optimizer/resume artifacts)."""
        import os

        model_id = session.model_id
        worker = await self._get_live_worker(session, op="save_lora_weights_for_sampler")
        await self._ensure_megatron_session_guard_clean(
            session,
            op="save_lora_weights_for_sampler",
            worker=worker,
        )
        abs_path = os.path.abspath(save_path)

        try:
            from mint_server.backend.core.model_registry import get_model_config

            train_gpus = get_model_config(session.base_model).train_gpus
        except Exception:
            train_gpus = 1

        if train_gpus >= 32:
            default_timeout_s = 3600
        elif train_gpus >= 16:
            default_timeout_s = 1800
        elif train_gpus >= 4:
            default_timeout_s = 600
        else:
            default_timeout_s = 300
        timeout_s = int(os.environ.get("MINT_SAVE_LORA_TIMEOUT_S", str(default_timeout_s)))

        lora_cfg = getattr(session, "lora_config", None)
        train_attn = True if lora_cfg is None else bool(getattr(lora_cfg, "train_attn", True))
        train_mlp = True if lora_cfg is None else bool(getattr(lora_cfg, "train_mlp", True))
        train_unembed = True if lora_cfg is None else bool(getattr(lora_cfg, "train_unembed", True))
        traceparent = get_current_traceparent()
        meta_ref = worker.save_lora_weights.remote(
            abs_path,
            session_id=session.model_id,
            actual_rank=session.lora_config.rank if session.lora_config else None,
            traceparent=traceparent,
            train_attn=train_attn,
            train_mlp=train_mlp,
            train_unembed=train_unembed,
        )
        meta = await run_async_with_otel_span(
            "training.save_weights_for_sampler.remote_save",
            lambda: self._await_worker_call(
                meta_ref,
                session,
                op="save_lora_weights_for_sampler",
                worker=worker,
                interval_s=30.0,
                timeout_s=timeout_s,
            ),
            component="backend.verl_training",
            op="training.save_weights_for_sampler.remote_save",
            request_id=str(get_request_id() or "") or None,
            attributes={
                "model_id": str(model_id),
                "base_model": str(session.base_model),
                "backend": str(session.backend),
                "save_path": str(abs_path),
                "timeout_s": int(timeout_s),
                "train_attn": bool(train_attn),
                "train_mlp": bool(train_mlp),
                "train_unembed": bool(train_unembed),
            },
        )
        strict_meta = self._strict_megatron_save_meta_enabled()
        self._update_session_step_monotonic(
            session,
            meta,
            op="save_lora_weights_for_sampler",
            strict=strict_meta,
        )

        logger.info("save_lora_weights_for_sampler___s", model_id=model_id)
        return abs_path

    async def save_weights(
        self,
        session: TrainingSession,
        save_path: str,
        use_per_expert_lora: bool = False,
    ) -> str:
        """Save checkpoint via Ray actor.

        Saves LoRA weights directly on worker to shared filesystem.
        Returns path for path-based vLLM loading.

        Args:
            session: Training session.
            save_path: Directory path for checkpoint.
            use_per_expert_lora: Router compatibility flag from backend-agnostic
                save_weights() interface. Megatron checkpoint export currently
                uses train_attn/train_mlp/train_unembed gating and ignores this
                flag.

        Returns:
            Absolute path to saved checkpoint directory.
        """
        if use_per_expert_lora:
            logger.warning(
                "[%s] save_weights called with use_per_expert_lora=True; full training checkpoints ignore this sampler-only flag",
                session.model_id,
            )

        import os

        from mint_server.backend.core.model_registry import get_model_config

        model_id = session.model_id
        worker = await self._get_live_worker(session, op="save_weights")
        await self._ensure_megatron_session_guard_clean(
            session,
            op="save_weights",
            worker=worker,
        )
        abs_path = os.path.abspath(save_path)

        # Save on worker - returns metadata
        try:
            cfg = get_model_config(session.base_model)
            train_gpus = cfg.train_gpus
        except Exception:
            train_gpus = 1

        if train_gpus >= 32:
            default_timeout_s = 3600
        elif train_gpus >= 16:
            default_timeout_s = 1800
        elif train_gpus >= 4:
            default_timeout_s = 600
        else:
            default_timeout_s = 300
        timeout_s = int(os.environ.get("MINT_SAVE_CHECKPOINT_TIMEOUT_S", str(default_timeout_s)))

        if session.backend in _DISTRIBUTED_MOE_BACKENDS:
            traceparent = get_current_traceparent()
            lora_cfg = getattr(session, "lora_config", None)
            train_attn = True if lora_cfg is None else bool(getattr(lora_cfg, "train_attn", True))
            train_mlp = True if lora_cfg is None else bool(getattr(lora_cfg, "train_mlp", True))
            train_unembed = True if lora_cfg is None else bool(getattr(lora_cfg, "train_unembed", True))
            meta_ref = worker.save_checkpoint.remote(
                abs_path,
                traceparent=traceparent,
                session_id=session.model_id,
                actual_rank=getattr(lora_cfg, "rank", None),
                train_attn=train_attn,
                train_mlp=train_mlp,
                train_unembed=train_unembed,
            )
        else:
            traceparent = get_current_traceparent()
            meta_ref = worker.save_checkpoint.remote(
                abs_path,
                traceparent=traceparent,
                session_id=session.model_id,
                actual_rank=session.lora_config.rank if session.lora_config else None,
            )
        meta = await self._await_worker_call(
            meta_ref,
            session,
            op="save_weights",
            worker=worker,
            interval_s=30.0,
            timeout_s=timeout_s,
        )

        # Update session state
        strict_meta = bool(session.backend == "megatron" and self._strict_megatron_save_meta_enabled())
        self._update_session_step_monotonic(
            session,
            meta,
            op="save_weights",
            strict=strict_meta,
        )

        logger.info("save_weights___s", model_id=model_id)
        return abs_path

    async def load_weights(
        self,
        session: TrainingSession,
        load_path: str,
        load_optimizer: bool = True,
    ) -> dict[str, object] | None:
        """Load checkpoint via Ray actor.

        Args:
            session: Training session.
            load_path: Directory path to load from.
            load_optimizer: Whether to restore optimizer state.
        """
        model_id = session.model_id
        try:
            worker = await self._get_live_worker(
                session,
                op="load_weights",
                allow_recover=(session.backend == "peft"),
            )
        except RuntimeError as e:
            if "missing worker" not in str(e):
                raise
            worker = await self._recycle_worker_after_failure(
                session,
                op="load_weights",
                cause=e,
                explicit_checkpoint_path=load_path,
            )

        if session.backend == "megatron":
            ready_timeout_s = (
                float(server_config.training_actor_ready_timeout_s)
                if server_config.training_actor_ready_timeout_s is not None
                else 1800.0
            )
            ready_attempts = 0
            while True:
                try:
                    await self._await_with_keepalive(
                        worker.__ray_ready__.remote(),
                        session,
                        interval_s=30.0,
                        timeout_s=ready_timeout_s,
                    )
                    break
                except Exception as e:
                    if not self._is_dead_actor_error(e) or ready_attempts >= 1:
                        raise
                    ready_attempts += 1
                    worker = await self._recycle_worker_after_failure(
                        session,
                        op="load_weights",
                        cause=e,
                        explicit_checkpoint_path=load_path,
                    )

        default_timeout_s = 1800.0 if session.backend in _DISTRIBUTED_MOE_BACKENDS else 120.0
        load_timeout_s = float(os.environ.get("MINT_LOAD_CHECKPOINT_TIMEOUT_S", str(default_timeout_s)))

        traceparent = get_current_traceparent()
        kwargs: dict[str, object] = {
            "traceparent": traceparent,
            "session_id": session.model_id,
        }
        if session.backend == "bumblebee" and session.lora_config is not None:
            kwargs["actual_rank"] = session.lora_config.rank
        if session.backend == "megatron" and not os.path.isfile(os.path.join(load_path, "adapter_config.json")):
            lora_config = getattr(session, "lora_config", None)
            for key in ("train_attn", "train_mlp", "train_unembed"):
                value = getattr(lora_config, key, None)
                if value is not None:
                    kwargs[key] = bool(value)

        def _submit(call_worker):
            return call_worker.load_checkpoint.remote(load_path, load_optimizer, **kwargs)

        meta = await self._run_worker_call_with_actor_recycle(
            session,
            op="load_weights",
            submit_fn=_submit,
            interval_s=30.0,
            timeout_s=load_timeout_s,
            allow_recover=(session.backend == "peft"),
            explicit_checkpoint_path=load_path,
        )

        if session.backend == "megatron":
            meta = self._validate_megatron_load_meta(meta, op="load_weights")
            session.current_step = int(cast(int, meta["current_step"]))
            session.learning_rate = float(cast(float, meta["learning_rate"]))
            self._apply_megatron_loaded_lora_config(session, meta)
            actor_name = self._actor_name_for_session(session)
            if actor_name and bool(meta["optimizer_restored"]):
                self._actor_recycler.mark_session_volatile(actor_name, session.model_id)

            worker = await self._get_live_worker(session, op="load_weights", allow_recover=False)
            await async_get_ray_ref(
                worker.mark_session_loaded.remote(
                    session.model_id,
                    step_count=session.current_step,
                    learning_rate=session.learning_rate,
                    actual_rank=meta["actual_rank"],
                    actor_only_state_dirty=meta["actor_only_state_dirty"],
                    checkpoint_path=meta["checkpoint_path"],
                    optimizer_restored=meta["optimizer_restored"],
                    train_attn=meta["train_attn"],
                    train_mlp=meta["train_mlp"],
                    train_unembed=meta["train_unembed"],
                ),
                timeout_s=load_timeout_s,
            )
            if not bool(meta["optimizer_restored"]):
                actor_name = self._actor_name_for_session(session)
                if actor_name:
                    self._actor_recycler.discard_session_volatile(actor_name, session.model_id)
        else:
            # Dense workers keep the older lenient metadata path because legacy
            # PEFT checkpoints can omit training_meta.json.
            self._update_session_from_load_meta(
                session,
                meta,
                op="load_weights",
            )
            if not load_optimizer:
                actor_name = self._actor_name_for_session(session)
                if actor_name:
                    self._actor_recycler.discard_session_volatile(actor_name, session.model_id)

        self._actor_recycler.clear_poisoned(session.model_id)

        logger.info("load_weights", model_id=model_id, step=session.current_step)
        return meta if isinstance(meta, dict) else None

    async def shutdown_session(self, session: TrainingSession) -> None:
        """Delete actor-local session state, then release or unbind the worker."""
        model_id = session.model_id

        actor_name = self._actor_name_for_session(session)
        worker = self._workers.get(model_id)
        authoritative_actor_name = str(getattr(session, "actor_name", "") or "")
        if authoritative_actor_name and actor_name and actor_name != authoritative_actor_name:
            self._workers.pop(model_id, None)
            self._actor_recycler.bind_session_actor(model_id, authoritative_actor_name)
            worker = None
            actor_name = authoritative_actor_name
        if worker is None:
            worker = await self._rebind_worker_from_session_metadata(session, reason="shutdown_session")
            actor_name = self._actor_recycler.actor_name_for_model(model_id) or str(getattr(session, "actor_name", "") or "") or None
        traceparent = get_current_traceparent()

        if worker is not None:
            delete_session = getattr(worker, "delete_session", None)
            if delete_session is not None:
                try:
                    await async_get_ray_ref(
                        delete_session.remote(model_id, traceparent=traceparent),
                        timeout_s=30,
                    )
                except TypeError:
                    await async_get_ray_ref(delete_session.remote(model_id), timeout_s=30)
                except Exception:
                    logger.warning(
                        "[%s] delete_session remote cleanup failed",
                        model_id,
                        exc_info=True,
                    )

        actor_protected = False
        other_users: list[str] = []
        if actor_name:
            other_users = self._actor_recycler.bound_model_ids_for_actor(actor_name, exclude_model_id=model_id)
            try:
                from mint_server.backend.stores.training_session_store import list_training_sessions

                for info in list_training_sessions():
                    if not isinstance(info, dict):
                        continue
                    other_model_id = str(info.get("model_id") or "")
                    if not other_model_id or other_model_id == model_id:
                        continue
                    if str(info.get("actor_name") or "") != actor_name:
                        continue
                    if other_model_id not in other_users:
                        other_users.append(other_model_id)
            except Exception:
                pass
        replacement_session = other_users[0] if other_users else None
        should_kill_actor = not other_users
        if actor_name:
            try:
                from mint_server.backend.actors.model_actor_supervisor import get_model_actor_supervisor

                actor_protected = get_model_actor_supervisor().is_protected(actor_name)
                if actor_protected:
                    should_kill_actor = False
            except Exception:
                pass

        self._actor_recycler.unbind_session_actor(model_id)
        self._workers.pop(model_id, None)
        self._actor_recycler.clear_session_runtime_state(model_id, actor_name)

        try:
            if session.backend == "peft":
                from mint_server.backend.training.dense.dense_trainer import clear_dense_trainer_session

                clear_dense_trainer_session(model_id)
        except Exception:
            pass

        try:
            if actor_name:
                from mint_server.backend.actors.model_actor_supervisor import get_model_actor_supervisor

                get_model_actor_supervisor().set_session(actor_name, replacement_session)
        except Exception:
            pass

        actor_namespace = str(getattr(session, "namespace", "") or RAY_NAMESPACE)
        if should_kill_actor:
            if worker:
                try:
                    worker.shutdown.remote()
                except Exception:
                    pass
                try:
                    ray_kill.kill(
                        worker,
                        reason="shutdown_session",
                        actor_name=actor_name,
                        namespace=actor_namespace,
                        no_restart=True,
                        model_id=model_id,
                    )
                except Exception:
                    pass
            elif actor_name:
                try:
                    actor = ray.get_actor(actor_name, namespace=actor_namespace)
                    ray_kill.kill(
                        actor,
                        reason="shutdown_session_race_no_worker",
                        actor_name=actor_name,
                        namespace=actor_namespace,
                        no_restart=True,
                        model_id=model_id,
                    )
                except Exception:
                    pass
        else:
            if actor_protected:
                logger.info("shutdown_session__session_deleted__keeping_protected_actor__", model_id=model_id)
            else:
                logger.info(
                    f"[{model_id}] shutdown_session: session deleted; keeping shared actor {actor_name} "
                    f"(still referenced by {len(other_users)} other model_id(s))"
                )

        session.is_active = False
        logger.info("trainingworker_shutdown", model_id=model_id)

    async def delete_session(self, session: TrainingSession) -> None:
        """Backward-compatible alias for session deletion paths."""
        await self.shutdown_session(session)


# Global engine instance (initialized in app lifespan)
verl_training_engine: VerlTrainingEngine | None = None
