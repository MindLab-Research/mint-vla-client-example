"""VerlTrainingEngine - SFT training with LoRA using Ray actors.

Each training session gets a dedicated TrainingWorker Ray actor with its own GPU.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import ray
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from .training_session_manager import TrainingSession

logger = logging.getLogger(__name__)

# Import centralized PFS paths from config
from tinker_server.config import PFS_PYTHONPATH


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

                loss_fn_outputs.append({
                    "loss": {"data": [item_loss], "shape": [1], "dtype": "float32"},
                    "logprobs": {
                        "data": target_logprobs.detach().tolist(),
                        "shape": list(target_logprobs.shape),
                        "dtype": "float32",
                    },
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

                loss_fn_outputs.append({
                    "loss": {"data": [item_loss], "shape": [1], "dtype": "float32"},
                    "logprobs": {
                        "data": new_logprobs.detach().tolist(),
                        "shape": list(new_logprobs.shape),
                        "dtype": "float32",
                    },
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

        if use_megatron:
            # MoE models (30B+) need tensor + expert parallelism to fit model+optimizer
            # 30B MoE with TP=4: ~76 GiB/GPU for model+gradients, no room for optimizer
            # TP=4, EP=2 = 8 GPUs: distributes optimizer state, ~38 GiB/GPU
            distributed_config = DistributedConfig(
                tensor_parallel_size=4,
                expert_parallel_size=2,
            )
            logger.info(f"[{model_id}] Creating MegatronWorkerGroup for MoE model (base={base_model}, lora_rank={lora_rank}, TP={distributed_config.tensor_parallel_size}, EP={distributed_config.expert_parallel_size})")

            # Get or create persistent Megatron worker group
            # Uses detached Ray actor pattern like vLLM for crash resilience
            # Use async version to avoid blocking uvicorn event loop
            worker = await async_get_or_create_megatron_worker_group(
                base_model=base_model,
                lora_rank=lora_rank,
                learning_rate=session.learning_rate,
                distributed_config=distributed_config,
            )
            session.backend = "megatron"
        else:
            logger.info(f"[{model_id}] Creating TrainingWorker for dense model (base={base_model}, lora_rank={lora_rank})")

            # Create single-GPU worker
            # Set PYTHONPATH and HF_HOME so workers can find code and cached models
            # HF_HUB_OFFLINE prevents network calls for model metadata
            dense_runtime_env = {
                "env_vars": {
                    "PYTHONPATH": PFS_PYTHONPATH,
                    "HF_HOME": "/vePFS-Mindverse/share/huggingface",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                }
            }
            worker = TrainingWorker.options(
                runtime_env=dense_runtime_env
            ).remote(
                base_model=base_model,
                lora_rank=lora_rank,
                learning_rate=session.learning_rate,
            )
            session.backend = "peft"

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

        # Serialize data for Ray
        data_items = [item.model_dump() for item in request.forward_backward_input.data]
        loss_fn = request.forward_backward_input.loss_fn
        loss_fn_config = request.forward_backward_input.loss_fn_config or {}

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
            request: ForwardRequest with forward_input field.

        Returns:
            Dict with loss_fn_outputs (including logprobs) and metrics.
        """
        model_id = session.model_id
        worker = self._workers[model_id]

        # Serialize data for Ray
        # ForwardRequest uses forward_input (not forward_backward_input)
        data_items = [item.model_dump() for item in request.forward_input.data]

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

        logger.info(f"[{model_id}] save_weights_for_sampler: ENTRY")
        logger.info(f"[{model_id}] save_weights_for_sampler: worker type = {type(worker)}")

        # Fetch weights and config from remote worker via Ray object store
        # Use ray.get() with timeout in thread executor for reliable async handling
        logger.info(f"[{model_id}] save_weights_for_sampler: calling get_lora_state_dict.remote()...")
        import ray
        loop = asyncio.get_running_loop()

        # Schedule remote calls
        state_dict_ref = worker.get_lora_state_dict.remote()
        config_ref = worker.get_lora_config.remote()
        logger.info(f"[{model_id}] save_weights_for_sampler: remote calls scheduled, waiting for results...")

        # Use ray.get() with timeout in executor to avoid blocking event loop
        def get_with_timeout():
            return ray.get([state_dict_ref, config_ref], timeout=120)

        state_dict, config = await loop.run_in_executor(None, get_with_timeout)
        logger.info(f"[{model_id}] save_weights_for_sampler: got {len(state_dict)} state_dict keys")

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
    """Entry in the dense trainer pool with session tracking."""

    actor: ray.actor.ActorHandle
    base_model: str
    max_lora_rank: int  # Trainer's max rank
    current_session: str | None = None
    actual_rank: int | None = None  # Current session's actual rank
    created_at: float = 0.0

    def __post_init__(self):
        import time

        if self.created_at == 0.0:
            self.created_at = time.time()


class DenseTrainerPool:
    """Pool of dense TrainingWorker actors keyed by base_model.

    Phase 8: Unified rank support - actors are initialized with max_lora_rank
    and can train adapters with any rank <= max_lora_rank via padding/truncation.

    Thread-safe for concurrent access from multiple API requests.
    """

    # Default max rank for pooled actors
    DEFAULT_MAX_LORA_RANK = 64

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
            return self._actors.get(key)

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

        Args:
            base_model: HuggingFace model path.
            lora_rank: Requested LoRA rank for this session.
            learning_rate: Initial learning rate (only used if creating new).
            session_id: Optional session ID to register with actor.
            max_lora_rank: Override max rank for new actor creation.

        Returns:
            DenseTrainerEntry with actor handle.
        """
        key = self._make_key(base_model)

        with self._lock:
            # Return existing actor if available
            if key in self._actors:
                entry = self._actors[key]
                if lora_rank > entry.max_lora_rank:
                    raise ValueError(
                        f"Requested rank {lora_rank} exceeds actor's max_lora_rank {entry.max_lora_rank}. "
                        f"Kill the actor and restart with higher max_lora_rank."
                    )
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
            logger.info(
                f"[DenseTrainerPool] Creating new actor: {actor_name} for {base_model}, "
                f"max_rank={effective_max_rank}"
            )

            # Check if detached actor already exists
            try:
                actor = ray.get_actor(actor_name, namespace=PERSISTENT_DENSE_NAMESPACE)
                logger.info(f"[DenseTrainerPool] Found existing detached actor: {actor_name}")
                # Verify it's alive
                ray.get(actor.get_session_info.remote(), timeout=10)
            except (ValueError, ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError):
                # Create new detached actor
                runtime_env = {
                    "env_vars": {
                        "PYTHONPATH": PFS_PYTHONPATH_DENSE,
                        "HF_HOME": "/vePFS-Mindverse/share/huggingface",
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        "PYTHONDONTWRITEBYTECODE": "1",
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
                current_session=session_id,
                actual_rank=lora_rank,
            )
            self._actors[key] = entry
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
                    "created_at": entry.created_at,
                }
                for entry in self._actors.values()
            ]

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
