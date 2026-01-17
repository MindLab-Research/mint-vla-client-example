"""VerlTrainingEngine - SFT training with LoRA using Ray actors.

Each training session gets a dedicated TrainingWorker Ray actor with its own GPU.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import ray
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from .training_session_manager import TrainingSession

logger = logging.getLogger(__name__)

# Default idle timeout for TrainingWorker (seconds)
# Set to 0 to disable self-termination (ResourcePool LRU eviction handles lifecycle)
DEFAULT_IDLE_TIMEOUT = 0  # Disabled - LRU eviction manages actor lifecycle

# Import centralized PFS paths from config
from tinker_server.config import PFS_PYTHONPATH


# =====================================================================
# Session State Manager - Per-iteration state persistence for stateless trainers
# =====================================================================

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
        logger.info(f"[SessionStateManager] Initialized with base_path={base_path}")

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
                logger.debug(f"[SessionStateManager] Saved {len(grads)} gradient tensors for {session_id}")
            else:
                # Remove old gradients file if no gradients to save
                grads_path = os.path.join(session_path, "gradients.pt")
                if os.path.exists(grads_path):
                    os.remove(grads_path)

        # 4. Save metadata
        meta = {"current_step": step, "learning_rate": lr}
        with open(os.path.join(session_path, "training_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        logger.debug(f"[SessionStateManager] Saved state for {session_id} (step={step})")
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
                logger.debug(f"[SessionStateManager] Restored {grad_count} gradient tensors for {session_id}")

        # 4. Load metadata
        meta = {"current_step": 0, "learning_rate": optimizer.param_groups[0]["lr"], "has_gradients": has_gradients}
        meta_path = os.path.join(session_path, "training_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                loaded_meta = json.load(f)
                meta.update(loaded_meta)
                meta["has_gradients"] = has_gradients

        logger.debug(f"[SessionStateManager] Loaded state for {session_id} (step={meta.get('current_step', 0)}, has_gradients={has_gradients})")
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
            logger.info(f"[SessionStateManager] Deleted session {session_id}")
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
    ):
        """Initialize model and optimizer on this worker's GPU.

        Args:
            base_model: HuggingFace model path.
            lora_rank: LoRA adapter rank.
            learning_rate: Initial learning rate for optimizer.
            idle_timeout: Seconds of inactivity before self-termination.
                          Set to 0 to disable auto-termination.
        """
        self.device = torch.device("cuda")
        self._base_model = base_model  # Store for get_lora_config()

        # Idle timeout tracking
        self._last_activity = time.time()
        self._idle_timeout = idle_timeout
        self._shutdown_requested = False

        # Start watchdog thread if timeout enabled
        if idle_timeout > 0:
            self._watchdog_thread = threading.Thread(
                target=self._idle_watchdog,
                daemon=True,
                name="TrainingWorker-IdleWatchdog",
            )
            self._watchdog_thread.start()
            logger.info(f"[TrainingWorker] Idle watchdog started (timeout={idle_timeout}s)")

        logger.info(f"[TrainingWorker] Loading {base_model} with LoRA rank={lora_rank}")

        self._configure_attention_backends()

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
        from .model_registry import get_model_config
        force_grad_ckpt = os.getenv("TINKER_FORCE_GRAD_CHECKPOINTING", "1") == "1"
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
            logger.info(f"[TrainingWorker] Gradient checkpointing enabled for {base_model}")

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
        self._state_manager = SessionStateManager()
        self._current_session_id: str | None = None

        logger.info("[TrainingWorker] Ready")

    @staticmethod
    def _configure_attention_backends() -> None:
        """Enable memory-efficient SDP where available."""
        if os.getenv("TINKER_ENABLE_SDP", "1") != "1":
            return
        try:
            if hasattr(torch.backends, "cuda"):
                if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                    torch.backends.cuda.enable_flash_sdp(True)
                if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                if hasattr(torch.backends.cuda, "enable_math_sdp"):
                    torch.backends.cuda.enable_math_sdp(True)
        except Exception as e:
            logger.warning(f"[TrainingWorker] Failed to configure SDP backends: {e}")

    def _touch(self) -> None:
        """Update last activity timestamp. Call at start of every method."""
        self._last_activity = time.time()

    def _ensure_session_loaded(self, session_id: str) -> None:
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
                self._step_count, lr, self.device, save_gradients=True
            )
            print(f"[DEBUG] Saved outgoing session {self._current_session_id} before switch", flush=True)

        # Load new session's state
        if self._state_manager.session_exists(session_id):
            meta = self._state_manager.load_state(
                session_id, self.model, self.optimizer, self.device, load_gradients=True
            )
            self._step_count = meta.get("current_step", 0)
            # Update learning rate if saved
            if "learning_rate" in meta:
                for pg in self.optimizer.param_groups:
                    pg["lr"] = meta["learning_rate"]
            # Check actual gradient state after loading
            grad_count = sum(1 for _, p in self.model.named_parameters() if p.requires_grad and p.grad is not None)
            grad_norm = sum((p.grad.norm().item() ** 2) for _, p in self.model.named_parameters() if p.requires_grad and p.grad is not None) ** 0.5
            print(f"[DEBUG] Loaded session {session_id}: step={self._step_count}, actual_grads={grad_count}, norm={grad_norm:.4f}", flush=True)
        else:
            # New session: reinitialize weights and zero gradients
            self.reinit_lora_weights()
            self.optimizer.zero_grad()
            self._step_count = 0
            print(f"[DEBUG] New session {session_id}, initialized fresh weights", flush=True)

        self._current_session_id = session_id

    def _save_session_state(self, session_id: str) -> None:
        """Save current session state to disk.

        Called at the end of optim_step to persist state.
        Note: Gradients are NOT saved here because optim_step zeros them.

        Args:
            session_id: Session ID to save state for.
        """
        lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 1e-4
        self._state_manager.save_state(
            session_id, self.model, self.optimizer, self._step_count, lr, self.device,
            save_gradients=False  # Gradients already applied and zeroed
        )
        logger.debug(f"[TrainingWorker] Saved session {session_id} state (step={self._step_count})")

    def _idle_watchdog(self) -> None:
        """Background thread that monitors for idle timeout.

        Checks every 30 seconds if the worker has been idle too long.
        If so, terminates the actor to release GPU resources.
        """
        check_interval = 30  # seconds between checks
        while not self._shutdown_requested:
            time.sleep(check_interval)
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
                except Exception as e:
                    logger.error(f"[TrainingWorker] Shutdown error: {e}")
                # Exit the Ray actor
                ray.actor.exit_actor()
                return

    def heartbeat(self) -> dict:
        """Keep worker alive and return status. Call periodically to prevent idle timeout.

        Returns:
            Dict with idle_time and timeout info.
        """
        self._touch()
        return {
            "idle_timeout": self._idle_timeout,
            "time_until_timeout": max(0, self._idle_timeout - (time.time() - self._last_activity)),
        }

    def forward_backward(
        self,
        data_items: list[dict],
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict | None = None,
        session_id: str | None = None,
    ) -> dict:
        """Forward + backward pass using tinker Datum format.

        Args:
            data_items: List of serialized Datum dicts with:
                - model_input.chunks[0].tokens: input token IDs
                - loss_fn_inputs.target_tokens: target token IDs (shifted by 1)
                - loss_fn_inputs.weights: per-token weights (float)
                  For SFT: binary mask (0.0 or 1.0)
                  For custom loss backward: negative gradients from client
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
        self._touch()

        # Stateless trainer: load session state if session_id provided
        if session_id:
            self._ensure_session_loaded(session_id)

        self.model.train()
        loss_fn_config = loss_fn_config or {}

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

            # Extract input token IDs from model_input.chunks[0].tokens
            chunks = model_input.get("chunks", [])
            if chunks and "tokens" in chunks[0]:
                input_ids = chunks[0]["tokens"]
            else:
                logger.warning("[TrainingWorker] No tokens in model_input, skipping item")
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

            # Determine if this is custom loss backward (weights has negative values)
            # or standard SFT/RL (weights are non-negative mask)
            has_negative_weights = (weights_flat < 0).any().item()

            # For standard loss: average over non-zero weights
            # For custom loss backward: sum without averaging (weights encode gradients)
            if has_negative_weights:
                # Custom loss backward: sum(ce * weights) without averaging
                num_weighted = 1.0  # No averaging
            else:
                # Standard SFT/RL: average over weighted tokens
                num_weighted = weights_flat.sum()

            if loss_fn == "cross_entropy":
                # Cross-entropy loss
                ce_loss = torch.nn.functional.cross_entropy(
                    logits_flat, targets_flat, reduction="none"
                )  # [seq_len]
                weighted_loss = ce_loss * weights_flat

                if has_negative_weights:
                    # Custom loss backward: sum without averaging
                    loss = weighted_loss.sum()
                elif num_weighted > 0:
                    # Standard: average over weighted tokens
                    loss = weighted_loss.sum() / num_weighted
                else:
                    loss = weighted_loss.sum()

                loss.backward()
                item_loss = loss.item()

                # Compute logprobs for training metrics (cookbook expects this)
                # logprobs = log_softmax(logits)[targets]
                log_probs = torch.nn.functional.log_softmax(logits_flat, dim=-1)
                target_logprobs = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)

                logprobs_list = target_logprobs.detach().tolist()
                loss_fn_outputs.append({
                    "loss": {"data": [item_loss], "shape": [1], "dtype": "float32"},
                    "logprobs": {"data": logprobs_list, "shape": [len(logprobs_list)], "dtype": "float32"},
                })

            elif loss_fn in ("importance_sampling", "ppo"):
                # RL losses require old logprobs and advantages
                old_logprobs_data = loss_fn_inputs.get("logprobs", {})
                advantages_data = loss_fn_inputs.get("advantages", {})

                old_logprobs = old_logprobs_data.get("data", [])
                advantages = advantages_data.get("data", [])

                if not old_logprobs or not advantages:
                    logger.warning(
                        f"[TrainingWorker] Missing logprobs or advantages for {loss_fn}, skipping item"
                    )
                    continue

                old_logprobs_t = torch.tensor(old_logprobs, dtype=torch.float32, device=self.device)
                advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)

                # Compute new logprobs from current policy
                log_probs = torch.nn.functional.log_softmax(logits_flat, dim=-1)  # [seq_len, vocab]
                new_logprobs = torch.gather(
                    log_probs, dim=-1, index=targets_flat.unsqueeze(-1)
                ).squeeze(-1)  # [seq_len]

                # Compute importance ratio: exp(new_logprobs - old_logprobs)
                # Clamp for numerical stability
                log_ratio = new_logprobs - old_logprobs_t
                log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
                ratio = torch.exp(log_ratio)

                if loss_fn == "importance_sampling":
                    # Policy gradient with importance sampling
                    # loss = -ratio * advantages * weights
                    pg_loss = -ratio * advantages_t * weights_flat
                    if num_weighted > 0:
                        loss = pg_loss.sum() / num_weighted
                    else:
                        loss = pg_loss.sum()

                else:  # ppo
                    # PPO with clipping
                    epsilon = loss_fn_config.get("epsilon", 0.2)
                    clip_low = loss_fn_config.get("clip_low", 1.0 - epsilon)
                    clip_high = loss_fn_config.get("clip_high", 1.0 + epsilon)

                    # Unclipped objective
                    pg_loss1 = -advantages_t * ratio

                    # Clipped objective
                    clipped_ratio = torch.clamp(ratio, clip_low, clip_high)
                    pg_loss2 = -advantages_t * clipped_ratio

                    # PPO objective: max(unclipped, clipped) for positive advantages
                    # This prevents too large policy updates
                    pg_loss = torch.maximum(pg_loss1, pg_loss2) * weights_flat

                    if num_weighted > 0:
                        loss = pg_loss.sum() / num_weighted
                    else:
                        loss = pg_loss.sum()

                    # Track clip fraction
                    clipped = ((ratio < clip_low) | (ratio > clip_high)).float() * weights_flat
                    clipfrac = clipped.sum() / max(num_weighted, 1)
                    total_clipfrac += clipfrac.item()

                loss.backward()
                item_loss = loss.item()

                # Track RL metrics
                masked_ratio = (ratio * weights_flat).sum() / max(num_weighted, 1)
                total_ratio += masked_ratio.item()
                num_rl_samples += 1

                rl_logprobs_list = new_logprobs.detach().tolist()
                loss_fn_outputs.append({
                    "loss": {"data": [item_loss], "shape": [1], "dtype": "float32"},
                    "logprobs": {"data": rl_logprobs_list, "shape": [len(rl_logprobs_list)], "dtype": "float32"},
                })

            else:
                raise ValueError(f"Unknown loss_fn: {loss_fn}")

            # For metrics tracking: use absolute sum of weights for token count
            effective_tokens = weights_flat.abs().sum().item() if has_negative_weights else num_weighted.item()
            total_loss += item_loss * effective_tokens
            total_tokens += effective_tokens

        avg_loss = total_loss / max(total_tokens, 1)

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
            f"[TrainingWorker] forward_backward ({loss_fn}): loss={avg_loss:.4f}, tokens={total_tokens:.0f}"
        )

        return {
            "loss_fn_output_type": f"{loss_fn}_loss",
            "loss_fn_outputs": loss_fn_outputs,
            "metrics": metrics,
        }

    def forward(self, data_items: list[dict], session_id: str | None = None) -> dict:
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
        self._touch()

        # Stateless trainer: load session state if session_id provided
        if session_id:
            self._ensure_session_loaded(session_id)

        self.model.eval()

        total_loss = 0.0
        total_tokens = 0
        loss_fn_outputs = []

        with torch.no_grad():
            for item in data_items:
                # Parse tinker Datum format
                model_input = item.get("model_input", {})
                loss_fn_inputs = item.get("loss_fn_inputs", {})

                # Extract input token IDs from model_input.chunks[0].tokens
                chunks = model_input.get("chunks", [])
                if chunks and "tokens" in chunks[0]:
                    input_ids = chunks[0]["tokens"]
                else:
                    logger.warning("[TrainingWorker] No tokens in model_input, skipping item")
                    continue

                # Extract target tokens and weights/mask
                # Accept "weights", "mask", or "loss_mask" field names (tinker API uses "loss_mask")
                target_data = loss_fn_inputs.get("target_tokens", {})
                weights_data = loss_fn_inputs.get("weights") or loss_fn_inputs.get("loss_mask") or loss_fn_inputs.get("mask", {})

                target_tokens = target_data.get("data", [])
                weights = weights_data.get("data", []) if weights_data else []

                if not target_tokens:
                    logger.warning("[TrainingWorker] Missing target_tokens, skipping item")
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

                # Get logprobs for target tokens
                # log_probs: [1, seq_len, vocab], target_ids_t: [1, seq_len]
                # Gather logprobs at target token indices
                target_logprobs = torch.gather(
                    log_probs.squeeze(0),  # [seq_len, vocab]
                    dim=-1,
                    index=target_ids_t.squeeze(0).unsqueeze(-1),  # [seq_len, 1]
                ).squeeze(-1)  # [seq_len]

                # Apply weights
                weights_flat = weights_t.squeeze(0)  # [seq_len]

                # Compute cross-entropy loss (for metrics)
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

                # Return logprobs in loss_fn_outputs (TensorData format required by Tinker SDK)
                logprobs_list = target_logprobs.tolist()
                loss_fn_outputs.append({
                    "loss": {"data": [item_loss], "shape": [1], "dtype": "float32"},
                    "logprobs": {"data": logprobs_list, "shape": [len(logprobs_list)], "dtype": "float32"},
                })

        avg_loss = total_loss / max(total_tokens, 1)

        logger.info(f"[TrainingWorker] forward: loss={avg_loss:.4f}, tokens={total_tokens:.0f}")

        return {
            "loss_fn_output_type": "sft_loss",
            "loss_fn_outputs": loss_fn_outputs,
            "metrics": {
                "loss:mean": avg_loss,
                "num_samples:sum": float(len(data_items)),
                "num_tokens:sum": float(total_tokens),
            },
        }

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

    def optim_step(self, learning_rate: float | None, session_id: str | None = None) -> dict:
        """Optimizer update step.

        Args:
            learning_rate: Optional new learning rate.
            session_id: Optional session ID for stateless trainer pattern.
                       If provided, saves session state after optimizer step.

        Returns:
            Dict with metrics.
        """
        self._touch()

        # Stateless trainer: ensure session state is loaded
        if session_id:
            self._ensure_session_loaded(session_id)

        # Update learning rate if provided
        if learning_rate is not None:
            for pg in self.optimizer.param_groups:
                pg["lr"] = learning_rate

        # Log gradient state before applying update
        grad_count = sum(1 for p in self.model.parameters() if p.requires_grad and p.grad is not None)
        pre_clip_norm = sum((p.grad.norm().item() ** 2) for p in self.model.parameters() if p.requires_grad and p.grad is not None) ** 0.5
        print(f"[DEBUG] optim_step: {grad_count} grads, pre_clip_norm={pre_clip_norm:.4f}", flush=True)

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        self._step_count += 1

        # Stateless trainer: save session state after update
        if session_id:
            self._save_session_state(session_id)

        logger.info(f"[TrainingWorker] optim_step: grad_norm={grad_norm:.4f}, step={self._step_count}")

        return {
            "metrics": {"grad_norm:last": float(grad_norm)},
            "type": "optim_step",
        }

    def get_lora_state_dict(self, use_per_expert_lora: bool = False) -> dict[str, torch.Tensor]:
        """Extract LoRA adapter weights as state dict.

        Args:
            use_per_expert_lora: Ignored for dense models (TrainingWorker).
                Only applies to MoE models using MegatronWorkerGroup.

        Returns:
            Dict mapping parameter names to tensors (on CPU).
        """
        self._touch()
        from peft.utils.save_and_load import get_peft_model_state_dict

        state_dict = get_peft_model_state_dict(self.model)
        # Move to CPU for serialization
        return {k: v.cpu() for k, v in state_dict.items()}

    def get_lora_config(self) -> dict:
        """Get LoRA configuration as dictionary.

        Returns:
            PEFT config dict compatible with vLLM's PEFTHelper.
        """
        peft_config = self.model.peft_config.get("default")
        return {
            "r": peft_config.r,
            "lora_alpha": peft_config.lora_alpha,
            "lora_dropout": peft_config.lora_dropout,
            "target_modules": list(peft_config.target_modules),
            "bias": peft_config.bias,
            "task_type": peft_config.task_type.value if peft_config.task_type else None,
            "peft_type": "LORA",
            "base_model_name_or_path": self._base_model,
        }

    def save_lora_weights(self, save_path: str) -> str:
        """Save LoRA adapter to directory.

        Args:
            save_path: Directory path to save adapter files.

        Returns:
            Absolute path where weights were saved.
        """
        self._touch()
        import json
        import os

        from safetensors.torch import save_file

        os.makedirs(save_path, exist_ok=True)

        # Save adapter weights
        state_dict = self.get_lora_state_dict()
        save_file(state_dict, os.path.join(save_path, "adapter_model.safetensors"))

        # Save adapter config
        config = self.get_lora_config()
        with open(os.path.join(save_path, "adapter_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        abs_path = os.path.abspath(save_path)
        logger.info(f"[TrainingWorker] Saved LoRA weights to {abs_path}")
        return abs_path

    def save_checkpoint(self, save_path: str) -> dict:
        """Save full checkpoint: LoRA weights + optimizer state + training metadata.

        Args:
            save_path: Directory path to save checkpoint files.

        Returns:
            Dict with training metadata, state_dict, and peft_config for registration.
        """
        self._touch()
        import json
        import os

        from safetensors.torch import save_file

        os.makedirs(save_path, exist_ok=True)

        # 1. LoRA weights
        state_dict = self.get_lora_state_dict()
        save_file(state_dict, os.path.join(save_path, "adapter_model.safetensors"))

        # 2. LoRA config
        peft_config = self.get_lora_config()
        with open(os.path.join(save_path, "adapter_config.json"), "w") as f:
            json.dump(peft_config, f, indent=2)

        # 3. Optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))

        # 4. Training metadata
        meta = {
            "current_step": self._step_count,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            # Include state_dict and config for multi-LoRA registration
            # (API server can't read files from Ray worker filesystem)
            "state_dict": state_dict,
            "peft_config": peft_config,
        }
        with open(os.path.join(save_path, "training_meta.json"), "w") as f:
            # Don't write state_dict/peft_config to file (already saved separately)
            json.dump({k: v for k, v in meta.items() if k not in ("state_dict", "peft_config")}, f, indent=2)

        abs_path = os.path.abspath(save_path)
        logger.info(f"[TrainingWorker] Saved checkpoint to {abs_path} (step={self._step_count})")
        return meta

    def load_checkpoint(self, load_path: str, load_optimizer: bool = True) -> dict:
        """Load checkpoint, optionally restoring optimizer state.

        Args:
            load_path: Directory path to load checkpoint from.
            load_optimizer: Whether to restore optimizer state.

        Returns:
            Dict with training metadata.
        """
        self._touch()
        import json
        import os

        from safetensors.torch import load_file

        # 1. Load LoRA weights
        adapter_path = os.path.join(load_path, "adapter_model.safetensors")
        if os.path.exists(adapter_path):
            state_dict = load_file(adapter_path, device=str(self.device))
            # Load into PEFT model
            from peft.utils.save_and_load import set_peft_model_state_dict
            set_peft_model_state_dict(self.model, state_dict)
            logger.info(f"[TrainingWorker] Loaded LoRA weights from {adapter_path}")
        else:
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")

        # 2. Optionally load optimizer state
        if load_optimizer:
            optimizer_path = os.path.join(load_path, "optimizer.pt")
            if os.path.exists(optimizer_path):
                self.optimizer.load_state_dict(
                    torch.load(optimizer_path, map_location=self.device)
                )
                logger.info(f"[TrainingWorker] Loaded optimizer state from {optimizer_path}")

        # 3. Load and return metadata
        meta = {}
        meta_path = os.path.join(load_path, "training_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self._step_count = meta.get("current_step", 0)
            logger.info(f"[TrainingWorker] Loaded metadata: step={self._step_count}")

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

        from tinker_server.backend.lora_utils import pad_lora_state_dict

        adapter_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")

        state_dict = load_file(adapter_path, device=str(self.device))

        # Phase 7: Apply padding if actual_rank < trainer_rank
        if actual_rank is not None and trainer_rank is not None and actual_rank < trainer_rank:
            logger.info(f"[TrainingWorker] Padding adapter from rank {actual_rank} to {trainer_rank}")
            state_dict = pad_lora_state_dict(state_dict, actual_rank, trainer_rank)

        # Load into PEFT model
        from peft.utils.save_and_load import set_peft_model_state_dict

        set_peft_model_state_dict(self.model, state_dict)
        logger.info(f"[TrainingWorker] Loaded adapter state from {checkpoint_path}")

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

        from tinker_server.backend.lora_utils import truncate_lora_state_dict

        os.makedirs(checkpoint_path, exist_ok=True)

        # Get LoRA state dict
        state_dict = get_peft_model_state_dict(self.model)

        # Phase 7: Apply truncation if actual_rank < trainer_rank
        if actual_rank is not None and trainer_rank is not None and actual_rank < trainer_rank:
            logger.info(f"[TrainingWorker] Truncating adapter from rank {trainer_rank} to {actual_rank}")
            state_dict = truncate_lora_state_dict(state_dict, trainer_rank, actual_rank)

        # Save to safetensors format
        adapter_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
        save_file(state_dict, adapter_path)
        logger.info(f"[TrainingWorker] Saved adapter state to {checkpoint_path}")

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
            logger.info(f"[TrainingWorker] Set learning rate to {learning_rate}")

        # Zero gradients
        self.optimizer.zero_grad()

        # Reset optimizer state (momentum/variance)
        # For AdamW, this resets exp_avg and exp_avg_sq
        self.optimizer.state.clear()
        logger.info("[TrainingWorker] Reset optimizer state (momentum cleared)")

        return {"status": "ok", "learning_rate": learning_rate}

    def reinit_lora_weights(self, learning_rate: float | None = None) -> dict:
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
            logger.info(f"[TrainingWorker] Set learning rate to {learning_rate}")

        # Zero gradients
        self.optimizer.zero_grad()

        # Reset optimizer state (momentum/variance)
        opt_state_reset = len(self.optimizer.state)
        self.optimizer.state.clear()
        logger.info(f"[TrainingWorker] Reset optimizer state ({opt_state_reset} entries)")

        # Reset step count
        self._step_count = 0

        logger.info(f"[TrainingWorker] Reinitialized {reinit_count} LoRA params")
        return {
            "status": "ok",
            "reinit_count": reinit_count,
            "opt_state_reset": opt_state_reset,
            "lr_updated": lr_updated,
            "learning_rate": learning_rate,
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

        logger.info(f"[TrainingWorker] Swapping session: {old_session_id} -> {new_session_id}")

        # Get trainer rank from current model
        peft_config = self.model.peft_config.get("default")
        trainer_rank = peft_config.r if peft_config else None

        # 1. Save old session state (if applicable)
        if old_session_id and old_checkpoint_path:
            logger.info(f"[TrainingWorker] Saving old session {old_session_id}")
            self.save_adapter_state(old_checkpoint_path)

        # 2. Load new session state or reset
        if new_checkpoint_path and os.path.exists(new_checkpoint_path):
            logger.info(f"[TrainingWorker] Loading new session {new_session_id}")
            self.load_adapter_state(
                new_checkpoint_path,
                actual_rank=new_actual_rank,
                trainer_rank=trainer_rank,
            )
        else:
            logger.info(f"[TrainingWorker] Resetting for new session {new_session_id}")
            self.reset_optimizer(new_learning_rate)

        # 3. Update step count
        self._step_count = 0

        logger.info(f"[TrainingWorker] Session swap complete: now on {new_session_id}")
        return {
            "status": "ok",
            "old_session": old_session_id,
            "new_session": new_session_id,
            "actual_rank": new_actual_rank,
        }

    def shutdown(self) -> None:
        """Release GPU resources and stop watchdog thread."""
        logger.info("[TrainingWorker] Shutting down")
        # Stop watchdog thread
        self._shutdown_requested = True
        # Release GPU resources
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "optimizer"):
            del self.optimizer
        torch.cuda.empty_cache()


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

    async def initialize(self) -> None:
        """Initialize Ray connection."""
        if not ray.is_initialized():
            # Use fixed namespace for persistent vLLM actor support
            ray.init(address="auto", namespace="tinker", ignore_reinit_error=True)
        logger.info("VerlTrainingEngine ready (Ray actors)")

    def _touch_actor(self, session: "TrainingSession") -> None:
        """Update last_accessed timestamp and session for the session's actor.

        Called during training operations to:
        1. Mark actor as recently used (prevents LRU eviction)
        2. Associate current session with actor (prevents idle eviction)
        """
        from .resource_pool import get_resource_pool
        from .megatron_distributed import _make_megatron_actor_name

        # Only Megatron actors are tracked in the unified resource pool
        # Dense trainers use their own DenseTrainerPool with touch() calls
        if session.backend == "megatron":
            base_model = session.base_model or self.default_base_model
            if base_model:
                actor_name = _make_megatron_actor_name(base_model)
                resource_pool = get_resource_pool()
                resource_pool.touch(actor_name)
                # Associate session with actor to prevent idle eviction
                resource_pool.set_session(actor_name, session.model_id)

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

        if hf_model_id in MODEL_PATH_OVERRIDES:
            override_path = MODEL_PATH_OVERRIDES[hf_model_id]
            logger.info(f"Using path override for {hf_model_id} -> {override_path}")
            return override_path

        hf_home = os.environ.get("HF_HOME", "/vePFS-Mindverse/share/huggingface")
        # Convert "org/model" to "models--org--model"
        cache_name = "models--" + hf_model_id.replace("/", "--")
        model_dir = Path(hf_home) / "hub" / cache_name / "snapshots"

        if not model_dir.exists():
            logger.warning(f"Model cache not found: {model_dir}")
            return None

        # Get the latest snapshot (usually only one)
        snapshots = list(model_dir.iterdir())
        if not snapshots:
            logger.warning(f"No snapshots in: {model_dir}")
            return None

        # Return the first snapshot path
        snapshot_path = str(snapshots[0])
        logger.info(f"Resolved {hf_model_id} -> {snapshot_path}")
        return snapshot_path

    async def create_training_session(self, session: TrainingSession) -> None:
        """Create Ray actor for session.

        Routes MoE models to MegatronTrainingWorker, dense models to TrainingWorker.
        Blocks until GPU is available (Ray queuing).

        Args:
            session: TrainingSession with configuration.
        """
        from .megatron_training import is_moe_model
        from .megatron_distributed import async_get_or_create_megatron_worker_group, DistributedConfig

        model_id = session.model_id

        # Determine base model path
        requested_model = session.base_model or self.default_base_model

        lora_rank = (
            session.lora_config.rank if session.lora_config else self.default_lora_rank
        )

        # Check if this is an MoE model requiring Megatron backend
        use_megatron = is_moe_model(requested_model or "")

        # Resolve model path based on backend
        if requested_model and not requested_model.startswith("/"):
            # HuggingFace model ID - resolve to local cache path
            base_model = self._resolve_hf_model_path(requested_model)
            if base_model:
                logger.info(f"[{model_id}] Resolved HF model to local: {base_model}")
            else:
                # Fall back to default (works for dense models on same architecture)
                base_model = self.default_base_model
                logger.info(f"[{model_id}] Using default model path: {base_model} (requested: {requested_model})")
        else:
            base_model = requested_model

        print(f"[DEBUG create_training_session] use_megatron={use_megatron}, base_model={base_model}", flush=True)

        if use_megatron:
            # MoE models need tensor/expert parallelism from model registry
            from .model_registry import get_training_parallelism, requires_fp8

            # Get model-specific parallelism and FP8 config from registry
            train_tp, train_ep, train_cp, train_etp = get_training_parallelism(requested_model or "")
            use_fp8 = requires_fp8(requested_model or "")
            distributed_config = DistributedConfig(
                tensor_parallel_size=train_tp,
                expert_parallel_size=train_ep,
                context_parallel_size=train_cp,
                expert_tensor_parallel_size=train_etp,
                use_fp8=use_fp8,
            )
            logger.info(f"[{model_id}] Creating MegatronWorkerGroup for MoE model (base={base_model}, lora_rank={lora_rank}, TP={train_tp}, EP={train_ep}, CP={train_cp}, ETP={train_etp}, world_size={distributed_config.world_size}, fp8={use_fp8})")

            # Get or create persistent Megatron worker group
            # Uses detached Ray actor pattern like vLLM for crash resilience
            # Use async version to avoid blocking uvicorn event loop
            # Issue #44: Pass model_id for unique session state isolation
            worker = await async_get_or_create_megatron_worker_group(
                base_model=base_model,
                lora_rank=lora_rank,
                learning_rate=session.learning_rate,
                distributed_config=distributed_config,
                session_id=session.model_id,
            )
            session.backend = "megatron"
            # Associate session with actor in resource pool immediately
            self._touch_actor(session)
            # Note: reinit_lora_weights is now called inside get_or_create_megatron_worker_group
            # with session_id for proper session state management (Issue #44)
        else:
            # Phase 8: Use DenseTrainerPool for actor reuse
            logger.info(f"[{model_id}] Using DenseTrainerPool for dense model (base={base_model}, lora_rank={lora_rank})")

            pool = get_dense_trainer_pool()
            entry = pool.get_or_create(
                base_model=base_model,
                lora_rank=lora_rank,
                learning_rate=session.learning_rate,
                session_id=session.model_id,
            )
            worker = entry.actor

            # Reinitialize LoRA weights for fresh session (statelessness)
            # This ensures each new session starts with fresh random weights
            # instead of inheriting trained weights from previous session
            logger.info(f"[{model_id}] Reinitializing LoRA weights for new session (lr={session.learning_rate})...")
            import ray
            result = ray.get(worker.reinit_lora_weights.remote(session.learning_rate))
            logger.info(f"[{model_id}] LoRA weights reinitialized: {result.get('reinit_count', 0)} params, lr_updated={result.get('lr_updated', False)}")

            # Update pool entry with current session
            entry.current_session = session.model_id
            entry.actual_rank = lora_rank

            session.backend = "peft"
            logger.info(f"[{model_id}] DenseTrainerPool: reusing actor for {base_model} (max_rank={entry.max_lora_rank})")

        # Wait for actor to be ready (model loaded)
        # Use await instead of ray.get() to not block the event loop
        await worker.__ray_ready__.remote()

        self._workers[model_id] = worker
        session.is_active = True
        logger.info(f"[{model_id}] TrainingWorker ready (backend={session.backend})")

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
        model_id = session.model_id
        worker = self._workers[model_id]

        # Mark actor as recently used to prevent LRU eviction during training
        self._touch_actor(session)

        # Serialize data for Ray
        data_items = [item.model_dump() for item in request.forward_backward_input.data]
        loss_fn = request.forward_backward_input.loss_fn
        loss_fn_config = request.forward_backward_input.loss_fn_config or {}

        # Remote call - pass session_id for stateless trainer pattern
        result = await worker.forward_backward.remote(
            data_items, loss_fn, loss_fn_config, session.model_id
        )

        # Update session state
        session.accumulated_gradients += 1

        logger.info(f"[{model_id}] forward_backward completed (loss_fn={loss_fn})")
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
        worker = self._workers[model_id]

        # Mark actor as recently used to prevent LRU eviction during training
        self._touch_actor(session)

        # Serialize data for Ray
        # ForwardRequest uses forward_input (not forward_backward_input)
        data_items = [item.model_dump() for item in request.forward_input.data]

        # Remote call - pass session_id for stateless trainer pattern
        result = await worker.forward.remote(data_items, session.model_id)

        logger.info(f"[{model_id}] forward completed")
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
        model_id = session.model_id
        worker = self._workers[model_id]

        result = await worker.get_tokenizer_info.remote()

        return result

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
        model_id = session.model_id
        worker = self._workers[model_id]

        # Mark actor as recently used to prevent LRU eviction during training
        self._touch_actor(session)

        # Extract learning rate
        lr = request.adam_params.learning_rate if request.adam_params else None

        # Remote call - pass session_id for stateless trainer pattern
        result = await worker.optim_step.remote(lr, session.model_id)

        # Update session state
        session.current_step += 1
        session.accumulated_gradients = 0

        # Add step to result metrics
        result["metrics"]["step"] = session.current_step

        logger.info(f"[{model_id}] optim_step: step={session.current_step}")
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
        from .megatron_training import is_moe_model

        model_id = session.model_id
        worker = self._workers[model_id]

        # Mark actor as recently used to prevent LRU eviction during training
        self._touch_actor(session)

        # Serialize data for Ray
        data_items = [item.model_dump() for item in request.forward_backward_input.data]
        loss_fn = request.forward_backward_input.loss_fn
        loss_fn_config = request.forward_backward_input.loss_fn_config or {}
        lr = request.adam_params.learning_rate if request.adam_params else session.learning_rate

        # Check if this is an MoE model (uses MegatronWorkerGroup with train_step)
        use_train_step = session.backend == "megatron" and is_moe_model(session.base_model or "")

        if use_train_step:
            # MoE: Use combined train_step to keep gradients in same context
            result = await worker.train_step.remote(
                data_items,
                loss_fn,
                loss_fn_config,
                lr,
                session.model_id,
            )
        else:
            # Dense models: Use separate calls (they don't have param_offload issues)
            # Pass session_id for stateless trainer pattern
            fb_result = await worker.forward_backward.remote(
                data_items, loss_fn, loss_fn_config, session.model_id
            )
            opt_result = await worker.optim_step.remote(lr, session.model_id)

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

        logger.info(f"[{model_id}] train_step: step={session.current_step}")
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
        import asyncio
        import ray

        model_id = session.model_id
        worker = self._workers.get(model_id)
        if worker is None:
            logger.warning(f"[{model_id}] reset_expert_bias: No worker found")
            return {"modules_reset": 0}

        # Mark actor as recently used
        self._touch_actor(session)

        logger.info(f"[{model_id}] reset_expert_bias: calling worker...")

        loop = asyncio.get_running_loop()
        try:
            result_ref = worker.reset_expert_bias.remote()
            result = await loop.run_in_executor(None, ray.get, result_ref)
            # MegatronWorkerGroup returns 'reset_count', normalize to 'modules_reset'
            modules_reset = result.get("reset_count", result.get("modules_reset", 0))
            logger.info(f"[{model_id}] reset_expert_bias: reset {modules_reset} modules")
            return {"modules_reset": modules_reset}
        except Exception as e:
            logger.exception(f"[{model_id}] reset_expert_bias failed: {e}")
            return {"modules_reset": 0, "error": str(e)}

    async def save_weights_for_sampler(
        self,
        session: TrainingSession,
        checkpoint_name: str,
        checkpoint_base_dir: str,
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

        save_path = os.path.join(checkpoint_base_dir, session.model_id, checkpoint_name)
        return await self.save_weights(session, save_path)

    async def save_weights(
        self,
        session: TrainingSession,
        save_path: str,
    ) -> str:
        """Save checkpoint via Ray actor.

        Saves LoRA weights directly on worker to shared filesystem.
        Returns path for path-based vLLM loading.

        Args:
            session: Training session.
            save_path: Directory path for checkpoint.

        Returns:
            Absolute path to saved checkpoint directory.
        """
        import asyncio
        import os

        import ray

        model_id = session.model_id
        worker = self._workers[model_id]
        abs_path = os.path.abspath(save_path)

        # Save on worker - returns metadata
        loop = asyncio.get_running_loop()
        meta_ref = worker.save_checkpoint.remote(abs_path)
        meta = await loop.run_in_executor(None, lambda: ray.get(meta_ref, timeout=300))

        # Update session state
        session.current_step = meta.get("current_step", session.current_step)

        logger.info(f"[{model_id}] save_weights: {abs_path}")
        return abs_path

    async def load_weights(
        self,
        session: TrainingSession,
        load_path: str,
        load_optimizer: bool = True,
    ) -> None:
        """Load checkpoint via Ray actor.

        Args:
            session: Training session.
            load_path: Directory path to load from.
            load_optimizer: Whether to restore optimizer state.
        """
        import asyncio

        import ray

        model_id = session.model_id
        worker = self._workers[model_id]

        # Remote call to load checkpoint
        # Must use ray.get() in executor since await on ObjectRef doesn't await completion
        loop = asyncio.get_running_loop()
        meta_ref = worker.load_checkpoint.remote(load_path, load_optimizer)
        meta = await loop.run_in_executor(None, lambda: ray.get(meta_ref, timeout=120))

        # Update session state from loaded metadata
        session.current_step = meta.get("current_step", 0)

        logger.info(f"[{model_id}] load_weights: step={session.current_step}")

    async def shutdown_session(self, session: TrainingSession) -> None:
        """Kill Ray actor to release GPU.

        Args:
            session: TrainingSession to shutdown.
        """
        model_id = session.model_id
        worker = self._workers.pop(model_id, None)
        if worker:
            # Call shutdown method first for clean cleanup
            try:
                await worker.shutdown.remote()
            except Exception:
                pass  # Actor may already be dead
            # Kill the actor to release resources
            ray.kill(worker)
        session.is_active = False
        logger.info(f"[{model_id}] TrainingWorker shutdown")


# =====================================================================
# Phase 8: Dense Trainer Pool (mirrors MegatronActorPool)
# =====================================================================

# Persistent actor naming
PERSISTENT_DENSE_NAMESPACE = "tinker"
PERSISTENT_DENSE_ACTOR_PREFIX = "dense_trainer_pool_"

# PFS PYTHONPATH for worker processes
PFS_PYTHONPATH_DENSE = "/vePFS-Mindverse/share/code/tinker-server:/vePFS-Mindverse/share/code/verl:/vePFS-Mindverse/share/code/vllm"


@dataclass
class DenseTrainerEntry:
    """Entry in the dense trainer pool with session tracking.

    Phase 9: Adds LRU tracking for adaptive resource management.
    """

    actor: ray.actor.ActorHandle
    base_model: str
    max_lora_rank: int  # Trainer's max rank
    num_gpus: int = 1  # GPUs used by this actor
    current_session: str | None = None
    actual_rank: int | None = None  # Current session's actual rank
    created_at: float = 0.0
    last_accessed: float = 0.0  # Phase 9: LRU tracking

    def __post_init__(self):
        import time

        now = time.time()
        if self.created_at == 0.0:
            self.created_at = now
        if self.last_accessed == 0.0:
            self.last_accessed = now

    def touch(self):
        """Update last_accessed timestamp (call on each access)."""
        import time
        self.last_accessed = time.time()

    def is_idle(self) -> bool:
        """Check if actor has no active session."""
        return self.current_session is None

    def age(self) -> float:
        """Seconds since creation."""
        import time
        return time.time() - self.created_at

    def idle_time(self) -> float:
        """Seconds since last access."""
        import time
        return time.time() - self.last_accessed


class DenseTrainerPool:
    """Pool of dense TrainingWorker actors keyed by base_model.

    Phase 8: Unified rank support - actors are initialized with max_lora_rank
    and can train adapters with any rank <= max_lora_rank via padding/truncation.

    Phase 9: LRU-based eviction for adaptive resource management.

    Thread-safe for concurrent access from multiple API requests.
    """

    # Default max rank for pooled actors
    DEFAULT_MAX_LORA_RANK = 64
    # Default GPUs per dense trainer actor
    DEFAULT_NUM_GPUS = 1
    # Minimum actor age before eligible for eviction (seconds)
    MIN_ACTOR_AGE = 300  # 5 minutes

    def __init__(self):
        import threading

        self._actors: dict[str, DenseTrainerEntry] = {}  # key: base_model
        self._lock = threading.Lock()

    def _make_key(self, base_model: str) -> str:
        """Create pool key from model path."""
        return base_model

    def _make_actor_name(self, base_model: str, max_rank: int) -> str:
        """Create unique actor name for this config."""
        model_name = base_model.split("/")[-1].replace("-", "_").lower()
        return f"{PERSISTENT_DENSE_ACTOR_PREFIX}{model_name}_maxr{max_rank}"

    def get(self, base_model: str) -> DenseTrainerEntry | None:
        """Get existing actor entry if it exists.

        Args:
            base_model: HuggingFace model path.

        Returns:
            DenseTrainerEntry if actor exists, None otherwise.
        """
        key = self._make_key(base_model)
        with self._lock:
            entry = self._actors.get(key)
            if entry:
                entry.touch()  # Phase 9: Update LRU
            return entry

    def _get_idle_actors_lru(self) -> list[DenseTrainerEntry]:
        """Get idle actors sorted by last access time (LRU first).

        Must be called with lock held.

        An actor is evictable if:
        1. It is idle (no active session)
        2. It has been idle longer than MIN_ACTOR_AGE

        Note: We use idle_time() (time since last access) rather than age()
        (time since creation) to protect recently-active actors from eviction.
        """
        idle = [e for e in self._actors.values() if e.is_idle() and e.idle_time() > self.MIN_ACTOR_AGE]
        return sorted(idle, key=lambda e: e.last_accessed)

    def _evict_for_gpus(self, needed_gpus: int, save_state: bool = True) -> int:
        """Evict idle actors to free up GPUs.

        Must be called with lock held.

        Args:
            needed_gpus: Number of GPUs needed.
            save_state: If True, save session state before eviction.

        Returns:
            Number of GPUs freed.
        """
        freed_gpus = 0
        idle_actors = self._get_idle_actors_lru()

        for entry in idle_actors:
            if freed_gpus >= needed_gpus:
                break

            logger.info(
                f"[DenseTrainerPool] Evicting idle actor: {entry.base_model} "
                f"(idle {entry.idle_time():.1f}s, frees {entry.num_gpus} GPUs)"
            )

            # Remove from pool
            key = self._make_key(entry.base_model)
            self._actors.pop(key, None)

            # Kill actor (no state to save since it's idle)
            try:
                ray.get(entry.actor.shutdown.remote(), timeout=30)
                ray.kill(entry.actor)
            except Exception as e:
                logger.warning(f"[DenseTrainerPool] Error killing evicted actor: {e}")

            freed_gpus += entry.num_gpus

        return freed_gpus

    def get_or_create(
        self,
        base_model: str,
        lora_rank: int,
        learning_rate: float,
        session_id: str | None = None,
        max_lora_rank: int | None = None,
    ) -> DenseTrainerEntry:
        """Get existing or create new dense trainer for this base model.

        Phase 8: If actor exists, it will be reused regardless of lora_rank
        (padding/truncation happens at session swap time).

        Phase 9: If resources exhausted, tries LRU eviction of idle actors.

        Args:
            base_model: HuggingFace model path.
            lora_rank: Requested LoRA rank for this session.
            learning_rate: Initial learning rate (only used if creating new).
            session_id: Optional session ID to register with actor.
            max_lora_rank: Override max rank for new actor creation.

        Returns:
            DenseTrainerEntry with actor handle.

        Raises:
            ValueError: If rank exceeds max or insufficient resources.
        """
        key = self._make_key(base_model)

        with self._lock:
            # Return existing actor if available
            if key in self._actors:
                entry = self._actors[key]

                # Check if actor is still alive (may have self-terminated due to idle timeout)
                try:
                    ray.get(entry.actor.heartbeat.remote(), timeout=5)
                except (ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError) as e:
                    logger.warning(
                        f"[DenseTrainerPool] Actor for {base_model} is dead (idle timeout?), removing from pool"
                    )
                    self._actors.pop(key, None)
                    # Also remove from unified resource pool
                    from tinker_server.backend.resource_pool import get_resource_pool
                    actor_name = self._make_actor_name(base_model, entry.max_lora_rank)
                    resource_pool = get_resource_pool()
                    resource_pool.unregister(actor_name)
                    # Fall through to create new actor
                else:
                    if lora_rank > entry.max_lora_rank:
                        raise ValueError(
                            f"Requested rank {lora_rank} exceeds actor's max_lora_rank {entry.max_lora_rank}. "
                            f"Kill the actor and restart with higher max_lora_rank."
                        )
                    entry.touch()  # Phase 9: Update LRU
                    # Also update unified resource pool
                    from tinker_server.backend.resource_pool import get_resource_pool
                    actor_name = self._make_actor_name(base_model, entry.max_lora_rank)
                    resource_pool = get_resource_pool()
                    pool_entry = resource_pool.get(actor_name)  # This calls touch()
                    if pool_entry:
                        pool_entry.current_session = session_id

                    logger.info(
                        f"[DenseTrainerPool] Reusing existing actor for {base_model} "
                        f"(max_rank={entry.max_lora_rank}, session_rank={lora_rank})"
                    )
                    return entry

            # Create new actor with max rank
            effective_max_rank = max(
                lora_rank,
                max_lora_rank or 0,
                self.DEFAULT_MAX_LORA_RANK,
            )
            actor_name = self._make_actor_name(base_model, effective_max_rank)
            num_gpus = self.DEFAULT_NUM_GPUS  # 1 GPU for dense models
            logger.info(
                f"[DenseTrainerPool] Creating new actor: {actor_name} for {base_model}, "
                f"max_rank={effective_max_rank}, gpus={num_gpus}"
            )

            # Phase 9: Check available GPUs and evict LRU actors if necessary
            from tinker_server.backend.resource_pool import get_resource_pool, ActorType
            resource_pool = get_resource_pool()
            resource_pool.ensure_gpus_available(num_gpus)

            # Check if detached actor already exists
            actor = None
            need_create = True
            try:
                existing_actor = ray.get_actor(actor_name, namespace=PERSISTENT_DENSE_NAMESPACE)
                logger.info(f"[DenseTrainerPool] Found existing detached actor: {actor_name}")
                # Verify it's alive
                ray.get(existing_actor.get_session_info.remote(), timeout=10)
                actor = existing_actor  # Use existing actor
                need_create = False
            except ValueError:
                # Actor doesn't exist, will create new one
                pass
            except (ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError):
                # Actor is dead or unresponsive, kill to free name before creating new one
                logger.warning(f"[DenseTrainerPool] Actor {actor_name} is dead/unresponsive, killing to free name")
                try:
                    ray.kill(existing_actor, no_restart=True)
                except Exception as kill_err:
                    logger.warning(f"[DenseTrainerPool] Could not kill dead actor: {kill_err}")

            if need_create:
                # Create new detached actor
                runtime_env = {
                    "env_vars": {
                        "PYTHONPATH": PFS_PYTHONPATH_DENSE,
                        "HF_HOME": "/vePFS-Mindverse/share/huggingface",
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    }
                }

                actor = TrainingWorker.options(
                    name=actor_name,
                    namespace=PERSISTENT_DENSE_NAMESPACE,
                    lifetime="detached",
                    num_gpus=1,
                    runtime_env=runtime_env,
                ).remote(
                    base_model=base_model,
                    lora_rank=effective_max_rank,  # Initialize with max rank
                    learning_rate=learning_rate,
                )

                # Wait for initialization
                ray.get(actor.__ray_ready__.remote())
                logger.info(f"[DenseTrainerPool] Actor {actor_name} initialized")

            entry = DenseTrainerEntry(
                actor=actor,
                base_model=base_model,
                max_lora_rank=effective_max_rank,
                num_gpus=self.DEFAULT_NUM_GPUS,
                current_session=session_id,
                actual_rank=lora_rank,
            )
            self._actors[key] = entry

            # Register with unified resource pool for LRU tracking
            resource_pool.register(
                actor_name=actor_name,
                actor_type=ActorType.DENSE,
                num_gpus=self.DEFAULT_NUM_GPUS,
                actor_handle=actor,
                namespace=PERSISTENT_DENSE_NAMESPACE,
                base_model=base_model,
                session_id=session_id,
            )

            return entry

    def remove(self, base_model: str, kill_actor: bool = True) -> bool:
        """Remove actor from pool.

        Args:
            base_model: HuggingFace model path.
            kill_actor: If True, also kill the Ray actor.

        Returns:
            True if actor was removed, False if not found.
        """
        key = self._make_key(base_model)

        with self._lock:
            if key not in self._actors:
                return False

            entry = self._actors.pop(key)
            if kill_actor:
                try:
                    ray.get(entry.actor.shutdown.remote(), timeout=30)
                    ray.kill(entry.actor)
                except Exception as e:
                    logger.warning(f"[DenseTrainerPool] Error killing actor: {e}")

            logger.info(f"[DenseTrainerPool] Removed actor for {base_model}")
            return True

    def list_actors(self) -> list[dict]:
        """List all actors in the pool.

        Returns:
            List of actor info dicts.
        """
        with self._lock:
            return [
                {
                    "base_model": entry.base_model,
                    "max_lora_rank": entry.max_lora_rank,
                    "actual_rank": entry.actual_rank,
                    "current_session": entry.current_session,
                    "num_gpus": entry.num_gpus,
                    "created_at": entry.created_at,
                    "last_accessed": entry.last_accessed,
                    "idle_time": entry.idle_time(),
                }
                for entry in self._actors.values()
            ]

    def total_gpus(self) -> int:
        """Total GPUs used by all actors in pool."""
        with self._lock:
            return sum(e.num_gpus for e in self._actors.values())

    def evict_idle(self, min_idle_seconds: float = 600) -> int:
        """Evict all idle actors that have been idle longer than threshold.

        Args:
            min_idle_seconds: Minimum idle time before eviction.

        Returns:
            Number of actors evicted.
        """
        with self._lock:
            to_evict = [
                e for e in self._actors.values()
                if e.is_idle() and e.idle_time() > min_idle_seconds
            ]

            for entry in to_evict:
                key = self._make_key(entry.base_model)
                self._actors.pop(key, None)
                try:
                    ray.get(entry.actor.shutdown.remote(), timeout=30)
                    ray.kill(entry.actor)
                except Exception as e:
                    logger.warning(f"[DenseTrainerPool] Error killing idle actor: {e}")
                logger.info(f"[DenseTrainerPool] Evicted idle actor: {entry.base_model}")

            return len(to_evict)

    def clear(self, kill_actors: bool = True) -> int:
        """Remove all actors from pool.

        Args:
            kill_actors: If True, also kill the Ray actors.

        Returns:
            Number of actors removed.
        """
        with self._lock:
            count = len(self._actors)
            if kill_actors:
                for entry in self._actors.values():
                    try:
                        ray.get(entry.actor.shutdown.remote(), timeout=30)
                        ray.kill(entry.actor)
                    except Exception as e:
                        logger.warning(f"[DenseTrainerPool] Error killing actor: {e}")
            self._actors.clear()
            logger.info(f"[DenseTrainerPool] Cleared {count} actors")
            return count


# Global pool instance
_dense_trainer_pool: DenseTrainerPool | None = None


def get_dense_trainer_pool() -> DenseTrainerPool:
    """Get or create the global DenseTrainerPool instance."""
    global _dense_trainer_pool
    if _dense_trainer_pool is None:
        _dense_trainer_pool = DenseTrainerPool()
    return _dense_trainer_pool


# Global engine instance (initialized in app lifespan)
verl_training_engine: VerlTrainingEngine | None = None
