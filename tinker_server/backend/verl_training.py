"""VerlTrainingEngine - SFT training with LoRA using Ray actors.

Each training session gets a dedicated TrainingWorker Ray actor with its own GPU.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import ray
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from .training_session_manager import TrainingSession

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class TrainingWorker:
    """Ray actor holding model + optimizer on dedicated GPU.

    Each instance runs in its own process with exclusive GPU access.
    """

    def __init__(
        self,
        base_model: str,
        lora_rank: int,
        learning_rate: float,
    ):
        """Initialize model and optimizer on this worker's GPU.

        Args:
            base_model: HuggingFace model path.
            lora_rank: LoRA adapter rank.
            learning_rate: Initial learning rate for optimizer.
        """
        self.device = torch.device("cuda")

        logger.info(f"[TrainingWorker] Loading {base_model} with LoRA rank={lora_rank}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model on this worker's GPU
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda",  # Use this worker's GPU
        )

        # Apply LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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

        logger.info("[TrainingWorker] Ready")

    def forward_backward(
        self,
        data_items: list[dict],
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict | None = None,
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

        Returns:
            Dict with loss_fn_outputs and metrics.
        """
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
            # Accept both "weights" and "mask" field names (tinker spec uses "mask")
            target_data = loss_fn_inputs.get("target_tokens", {})
            weights_data = loss_fn_inputs.get("weights") or loss_fn_inputs.get("mask", {})

            target_tokens = target_data.get("data", [])
            weights = weights_data.get("data", []) if weights_data else []

            if not target_tokens or not weights:
                logger.warning(
                    f"[TrainingWorker] Missing target_tokens or weights/mask, skipping item. "
                    f"loss_fn_inputs keys: {list(loss_fn_inputs.keys())}, "
                    f"target_tokens len: {len(target_tokens)}, weights len: {len(weights)}"
                )
                continue

            # Convert to tensors
            input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            target_ids_t = torch.tensor([target_tokens], dtype=torch.long, device=self.device)
            weights_t = torch.tensor([weights], dtype=torch.float32, device=self.device)

            # Forward pass - get logits
            outputs = self.model(input_ids=input_ids_t)
            logits = outputs.logits  # [1, seq_len, vocab_size]

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

                loss_fn_outputs.append(
                    {"loss": {"data": [item_loss], "shape": [1], "dtype": "float32"}}
                )

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

                loss_fn_outputs.append(
                    {"loss": {"data": [item_loss], "shape": [1], "dtype": "float32"}}
                )

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

    def forward(self, data_items: list[dict]) -> dict:
        """Forward pass only (no backward). Returns logprobs.

        Same input format as forward_backward but skips gradient computation.
        Useful for inference-time logprob computation with training model.

        Args:
            data_items: List of serialized Datum dicts with:
                - model_input.chunks[0].tokens: input token IDs
                - loss_fn_inputs.target_tokens: target token IDs (shifted by 1)
                - loss_fn_inputs.weights: per-token weights (float)

        Returns:
            Dict with loss_fn_outputs (including logprobs) and metrics.
        """
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
                # Accept both "weights" and "mask" field names (tinker spec uses "mask")
                target_data = loss_fn_inputs.get("target_tokens", {})
                weights_data = loss_fn_inputs.get("weights") or loss_fn_inputs.get("mask", {})

                target_tokens = target_data.get("data", [])
                weights = weights_data.get("data", []) if weights_data else []

                if not target_tokens or not weights:
                    logger.warning("[TrainingWorker] Missing target_tokens or weights/mask, skipping item")
                    continue

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

                # Return logprobs in loss_fn_outputs
                loss_fn_outputs.append({
                    "loss": {"data": [item_loss], "shape": [1], "dtype": "float32"},
                    "logprobs": {
                        "data": target_logprobs.tolist(),
                        "shape": list(target_logprobs.shape),
                        "dtype": "float32",
                    },
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

    def optim_step(self, learning_rate: float | None) -> dict:
        """Optimizer update step.

        Args:
            learning_rate: Optional new learning rate.

        Returns:
            Dict with metrics.
        """
        # Update learning rate if provided
        if learning_rate is not None:
            for pg in self.optimizer.param_groups:
                pg["lr"] = learning_rate

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        self._step_count += 1

        logger.info(f"[TrainingWorker] optim_step: grad_norm={grad_norm:.4f}, step={self._step_count}")

        return {
            "metrics": {"grad_norm": float(grad_norm)},
            "type": "optim_step",
        }

    def get_lora_state_dict(self) -> dict[str, torch.Tensor]:
        """Extract LoRA adapter weights as state dict.

        Returns:
            Dict mapping parameter names to tensors (on CPU).
        """
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
        }

    def save_lora_weights(self, save_path: str) -> str:
        """Save LoRA adapter to directory.

        Args:
            save_path: Directory path to save adapter files.

        Returns:
            Absolute path where weights were saved.
        """
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
            Dict with training metadata.
        """
        import json
        import os

        from safetensors.torch import save_file

        os.makedirs(save_path, exist_ok=True)

        # 1. LoRA weights
        state_dict = self.get_lora_state_dict()
        save_file(state_dict, os.path.join(save_path, "adapter_model.safetensors"))

        # 2. LoRA config
        config = self.get_lora_config()
        with open(os.path.join(save_path, "adapter_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # 3. Optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))

        # 4. Training metadata
        meta = {
            "current_step": self._step_count,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }
        with open(os.path.join(save_path, "training_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

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

    def shutdown(self) -> None:
        """Release GPU resources."""
        logger.info("[TrainingWorker] Shutting down")
        del self.model
        del self.optimizer
        torch.cuda.empty_cache()


class VerlTrainingEngine:
    """Manages per-session TrainingWorker Ray actors."""

    def __init__(
        self,
        default_base_model: str | None = None,
        default_lora_rank: int = 32,
    ):
        # Use config model_path if not specified (supports local paths)
        from ..config import config
        self.default_base_model = default_base_model or config.model_path
        self.default_lora_rank = default_lora_rank
        self._workers: dict[str, ray.actor.ActorHandle] = {}

    async def initialize(self) -> None:
        """Initialize Ray connection."""
        if not ray.is_initialized():
            # Use fixed namespace for persistent vLLM actor support
            ray.init(address="auto", namespace="tinker", ignore_reinit_error=True)
        logger.info("VerlTrainingEngine ready (Ray actors)")

    async def create_training_session(self, session: TrainingSession) -> None:
        """Create Ray actor for session.

        Blocks until GPU is available (Ray queuing).

        Args:
            session: TrainingSession with configuration.
        """
        model_id = session.model_id

        # Determine base model path
        # If request specifies a HuggingFace ID but we have a local path configured,
        # prefer the local path (worker nodes may not have network access)
        requested_model = session.base_model or self.default_base_model
        if requested_model and not requested_model.startswith("/"):
            # Not a local path - use configured default which should be local
            base_model = self.default_base_model
            logger.info(f"[{model_id}] Using local model path: {base_model} (requested: {requested_model})")
        else:
            base_model = requested_model

        lora_rank = (
            session.lora_config.rank if session.lora_config else self.default_lora_rank
        )

        logger.info(f"[{model_id}] Creating TrainingWorker (base={base_model}, lora_rank={lora_rank})")

        # Create Ray actor - queues if no GPU available
        worker = TrainingWorker.remote(
            base_model=base_model,
            lora_rank=lora_rank,
            learning_rate=session.learning_rate,
        )

        # Wait for actor to be ready (model loaded)
        # Use await instead of ray.get() to not block the event loop
        await worker.__ray_ready__.remote()

        self._workers[model_id] = worker
        session.is_active = True
        logger.info(f"[{model_id}] TrainingWorker ready")

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

        # Serialize data for Ray
        data_items = [item.model_dump() for item in request.forward_backward_input.data]
        loss_fn = request.forward_backward_input.loss_fn
        loss_fn_config = request.forward_backward_input.loss_fn_config or {}

        # Debug: log first item's loss_fn_inputs keys
        if data_items:
            first_item = data_items[0]
            lfi = first_item.get("loss_fn_inputs", {})
            print(f"[DEBUG] First datum loss_fn_inputs keys: {list(lfi.keys())}", flush=True)

        # Remote call
        result = await worker.forward_backward.remote(data_items, loss_fn, loss_fn_config)

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
            request: ForwardBackwardRequest with data.

        Returns:
            Dict with loss_fn_outputs (including logprobs) and metrics.
        """
        model_id = session.model_id
        worker = self._workers[model_id]

        # Serialize data for Ray
        data_items = [item.model_dump() for item in request.forward_backward_input.data]

        # Remote call
        result = await worker.forward.remote(data_items)

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

        # Extract learning rate
        lr = request.adam_params.learning_rate if request.adam_params else None

        # Remote call
        result = await worker.optim_step.remote(lr)

        # Update session state
        session.current_step += 1
        session.accumulated_gradients = 0

        # Add step to result metrics
        result["metrics"]["step"] = session.current_step

        logger.info(f"[{model_id}] optim_step: step={session.current_step}")
        return result

    async def save_weights_for_sampler(
        self,
        session: TrainingSession,
        checkpoint_name: str,
        checkpoint_base_dir: str,
    ) -> str:
        """Save LoRA weights for inference use.

        Fetches weights from remote Ray worker via object store, then saves
        locally on API server. This handles distributed deployments where
        training worker and API server are on different machines.

        Args:
            session: Training session with model.
            checkpoint_name: Name for this checkpoint.
            checkpoint_base_dir: Base directory for checkpoints.

        Returns:
            Absolute path to saved checkpoint directory.
        """
        import json
        import os

        from safetensors.torch import save_file

        model_id = session.model_id
        worker = self._workers[model_id]

        # Fetch weights and config from remote worker via Ray object store
        state_dict, config = await asyncio.gather(
            worker.get_lora_state_dict.remote(),
            worker.get_lora_config.remote(),
        )

        # Save locally on API server
        save_path = os.path.join(checkpoint_base_dir, model_id, checkpoint_name)
        os.makedirs(save_path, exist_ok=True)

        save_file(state_dict, os.path.join(save_path, "adapter_model.safetensors"))
        with open(os.path.join(save_path, "adapter_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        abs_path = os.path.abspath(save_path)
        logger.info(f"[{model_id}] Saved weights for sampler to {abs_path}")
        return abs_path

    async def save_weights(
        self,
        session: TrainingSession,
        save_path: str,
    ) -> str:
        """Save full checkpoint via Ray actor.

        Saves LoRA weights, optimizer state, and training metadata.

        Args:
            session: Training session.
            save_path: Directory path for checkpoint.

        Returns:
            Absolute path to saved checkpoint.
        """
        import os

        model_id = session.model_id
        worker = self._workers[model_id]

        # Remote call to save checkpoint
        meta = await worker.save_checkpoint.remote(save_path)

        # Update session state from worker metadata
        session.current_step = meta.get("current_step", session.current_step)

        abs_path = os.path.abspath(save_path)
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
        model_id = session.model_id
        worker = self._workers[model_id]

        # Remote call to load checkpoint
        meta = await worker.load_checkpoint.remote(load_path, load_optimizer)

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


# Global engine instance (initialized in app lifespan)
verl_training_engine: VerlTrainingEngine | None = None
