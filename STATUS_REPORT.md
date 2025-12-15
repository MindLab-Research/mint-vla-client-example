# Tinker-Server Status Report

Generated: 2025-12-08

## Background Experiments (Current)

| Test | Status | Progress |
|------|--------|----------|
| Phase 3: MATH RL | Running | 188 batches, initial eval complete (50.9s) |
| Phase 5: RLHF Pipeline | Running | SFT stage step 10/37, NLL 2.31 → 1.84 |

## Paradigms Supported

| Paradigm | Loss Function | Status |
|----------|---------------|--------|
| Supervised Fine-Tuning (SFT) | `cross_entropy` | Verified |
| Policy Gradient RL | `importance_sampling` | Verified |
| PPO | `ppo` (with clipping) | Verified |
| Custom Losses (DPO, etc.) | via `weights` in `cross_entropy` | Verified |

All loss functions support configurable parameters via `loss_fn_config`.

## Tinker API Compatibility

### Inference

- `POST /asample` - async sampling with logprobs
- `POST /compute_logprobs` - sequence log probabilities
- `POST /create_sampling_session` - session management

### Training

- `POST /create_model` - LoRA initialization
- `POST /forward_backward` - gradient computation
- `POST /forward` - inference-only logprobs
- `POST /optim_step` - Adam optimizer step
- `GET /models/{model_id}/tokenizer` - tokenizer config

### Checkpointing

- `POST /save_weights` / `POST /load_weights` - full state
- `POST /save_weights_for_sampler` - ephemeral weight sync
- `POST /create_model_from_state` - resume from checkpoint

### Validation

Unmodified Tinker Cookbook recipes run against our server with:
```bash
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy
```

## Performance

| Operation | Latency |
|-----------|---------|
| Hot LoRA reload (ephemeral flow) | ~0.7s |
| Cold vLLM init | ~60s |
| SFT step (batch 256) | ~30s |
| Eval (SFT) | ~13s |

Weight sync improved from ~60s to ~0.7s (88x speedup) via hot LoRA reload pattern.

## Test Results Summary

| Phase | Recipe | Metrics |
|-------|--------|---------|
| 1 | Arithmetic RL | reward 0.66 → 1.0 |
| 2 | Chat SL (NoRobots) | test/nll → 1.78 |
| 4 | DPO (HHH) | accuracy 0.41 → 0.61 |
| 3 | MATH RL | In progress |
| 5 | RLHF Pipeline | SFT stage in progress |
