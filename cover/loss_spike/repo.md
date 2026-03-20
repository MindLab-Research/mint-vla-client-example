# Issue 193/194 Loss-Spike Evidence

## Experiment

- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Worker topology: single shared 30B Megatron trainer
- Target run:
  - `train_limit=400`
  - `num_epochs=4`
  - planned `184` steps
- Perturbation windows:
  - `W1: step 60-90`, total sessions `2`
  - `W2: step 100-145`, total sessions `3`
  - `W3: step 155-180`, total sessions `2`

The target was allowed to settle first, then short background sessions were injected to force session switches under load.

## Evidence

- Figure: [issue193194_30b_134steps_windows.svg](/vePFS-Mindverse/user/intern/nolanho/code/mint-issue193-194-session-serialization/cover/loss_spike/issue193194_30b_134steps_windows.svg)
- Summary: [issue193194_30b_134steps_windows_summary.json](/vePFS-Mindverse/user/intern/nolanho/code/mint-issue193-194-session-serialization/cover/loss_spike/issue193194_30b_134steps_windows_summary.json)

## Key result

The run was interrupted externally at step `134`, but by then it had already covered:

- a stable pre-window convergence segment
- a full `2-session` perturbation window
- recovery after that window
- a long `3-session` perturbation window up to step `134`

Window statistics:

- `step 50-59`
  - `loss_mean = 0.162232`
  - `loss_max = 0.217234`
  - `step_time_mean = 10.902s`
- `step 60-90`
  - `loss_mean = 0.185092`
  - `loss_max = 0.297272`
  - `step_time_mean = 84.659s`
- `step 93-100`
  - `loss_mean = 0.133245`
  - `loss_max = 0.183652`
  - `step_time_mean = 10.591s`
- `step 101-134`
  - `loss_mean = 0.143319`
  - `loss_max = 0.221203`
  - `step_time_mean = 141.559s`

Representative high-load steps in the `3-session` window:

| step | loss:mean | step_time_sec |
|---|---:|---:|
| 101 | 0.160838 | 88.101 |
| 102 | 0.129632 | 130.959 |
| 103 | 0.167317 | 132.081 |
| 107 | 0.076638 | 212.494 |
| 121 | 0.184231 | 151.177 |
| 123 | 0.194991 | 140.256 |
| 134 | 0.221203 | 150.342 |

## Interpretation

The experiment still produces severe `step_time_sec` spikes under session-switch contention.

It does **not** reproduce the old signature where `loss:mean` spikes together with those latency spikes and permanently diverges from the target curve.

Up to step `134`, the evidence supports:

- latency perturbation remains
- loss corruption is no longer reproduced

## Interruption note

This run stopped because the shared Megatron placement group was removed externally during the second window. That interruption is operational noise, not a loss-spike signal from the target curve itself.
