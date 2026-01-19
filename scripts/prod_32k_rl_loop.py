#!/usr/bin/env python3
import datetime
import os
import random
import re
import sys
import time

import mint
from mint import types


def _ts() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _wait_future(fut, label: str, heartbeat_s: float) -> object:
    start = time.time()
    while True:
        try:
            return fut.result(timeout=heartbeat_s)
        except TimeoutError:
            elapsed = time.time() - start
            print(f"[{_ts()}] waiting {label} elapsed_s={elapsed:.0f}", flush=True)


def _first_int(s: str) -> int | None:
    m = re.search(r"(-?\\d+)", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _pad_to(tokens: list[int], length: int, pad_id: int) -> list[int]:
    if len(tokens) >= length:
        return tokens[:length]
    return tokens + [pad_id] * (length - len(tokens))


def main() -> int:
    base_model = os.environ.get("BASE_MODEL") or (sys.argv[1] if len(sys.argv) > 1 else "")
    if not base_model:
        print("BASE_MODEL env var or argv[1] required", file=sys.stderr)
        return 2

    lora_rank = int(os.environ.get("LORA_RANK", "16"))
    lr = float(os.environ.get("RL_LEARNING_RATE", "5e-5"))
    steps = int(os.environ.get("RL_STEPS", "2"))
    prompts_per_step = int(os.environ.get("RL_PROMPTS_PER_STEP", "4"))
    samples_per_prompt = int(os.environ.get("RL_SAMPLES_PER_PROMPT", "2"))
    max_seq_len = int(os.environ.get("RL_MAX_SEQ_LEN", "32000"))
    gen_max_tokens = int(os.environ.get("RL_GEN_MAX_TOKENS", "32"))
    temperature = float(os.environ.get("RL_TEMPERATURE", "0.7"))
    heartbeat_s = float(os.environ.get("MINT_FUTURE_HEARTBEAT_S", "60"))

    print(f"[{_ts()}] RL start base_model={base_model} lora_rank={lora_rank} lr={lr}", flush=True)
    print(
        f"[{_ts()}] config steps={steps} prompts_per_step={prompts_per_step} "
        f"samples_per_prompt={samples_per_prompt} max_seq_len={max_seq_len} "
        f"gen_max_tokens={gen_max_tokens} temperature={temperature}",
        flush=True,
    )

    random.seed(42)

    print(f"[{_ts()}] create ServiceClient", flush=True)
    service_client = mint.ServiceClient()
    print(f"[{_ts()}] create_lora_training_client start", flush=True)
    training_client = service_client.create_lora_training_client(
        base_model=base_model,
        rank=lora_rank,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )
    print(f"[{_ts()}] get_tokenizer start", flush=True)
    tokenizer = training_client.get_tokenizer()
    print(f"[{_ts()}] get_tokenizer done vocab_size={tokenizer.vocab_size}", flush=True)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    for step in range(steps):
        ckpt_name = f"rl_step_{step:04d}"
        print(f"[{_ts()}] step {step+1}/{steps}: save_weights_and_get_sampling_client start name={ckpt_name}", flush=True)
        t0 = time.time()
        sampling_client = training_client.save_weights_and_get_sampling_client(name=ckpt_name)
        print(f"[{_ts()}] step {step+1}/{steps}: save_weights_and_get_sampling_client done elapsed_s={time.time()-t0:.1f}", flush=True)

        datums: list[types.Datum] = []
        step_rewards: list[float] = []

        for p in range(prompts_per_step):
            a = random.randint(10, 99)
            b = random.randint(10, 99)
            expected = a * b

            prompt_text = f"Question: What is {a} * {b}?\\nAnswer:"
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
            prompt = types.ModelInput.from_ints(tokens=prompt_tokens)

            print(
                f"[{_ts()}] step {step+1}/{steps}: rollout prompt {p+1}/{prompts_per_step} "
                f"num_samples={samples_per_prompt} prompt_len={len(prompt_tokens)}",
                flush=True,
            )

            sample_future = sampling_client.sample(
                prompt=prompt,
                num_samples=samples_per_prompt,
                sampling_params=types.SamplingParams(
                    max_tokens=gen_max_tokens,
                    temperature=temperature,
                    top_k=-1,
                    top_p=1.0,
                ),
            )
            sample_res = _wait_future(
                sample_future,
                label=f"sample prompt {p+1}/{prompts_per_step} step {step+1}/{steps}",
                heartbeat_s=heartbeat_s,
            )

            rewards = []
            token_payloads: list[tuple[list[int], int, list[int]]] = []
            # Each payload: (full_tokens, prompt_len, completion_tokens)

            for seq in sample_res.sequences:
                completion_tokens = list(seq.tokens)
                # Some backends may return full sequence tokens; detect and normalize to full_tokens.
                if completion_tokens[: len(prompt_tokens)] == prompt_tokens:
                    full_tokens = completion_tokens
                    completion_tokens = completion_tokens[len(prompt_tokens) :]
                else:
                    full_tokens = prompt_tokens + completion_tokens

                if not full_tokens or full_tokens[-1] != eos_id:
                    full_tokens = full_tokens + [eos_id]

                full_tokens = _pad_to(full_tokens, max_seq_len, pad_id)
                completion_text = tokenizer.decode(completion_tokens)
                pred = _first_int(completion_text)
                reward = 1.0 if pred == expected else 0.0
                rewards.append(reward)
                token_payloads.append((full_tokens, len(prompt_tokens), completion_tokens))

            mean_reward = sum(rewards) / max(1, len(rewards))

            for reward, (full_tokens, prompt_len, completion_tokens) in zip(rewards, token_payloads):
                adv_scalar = float(reward - mean_reward)
                # Use training model forward() to compute per-token logprobs for PPO.
                # vLLM prompt_logprobs at 32k can OOM; forward() stays within the training worker.
                lp_datum = types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=full_tokens[:-1]),
                    loss_fn_inputs={"target_tokens": full_tokens[1:]},
                )
                fwd_future = training_client.forward([lp_datum], loss_fn="cross_entropy")
                fwd_res = _wait_future(
                    fwd_future,
                    label=f"forward(logprobs) prompt {p+1}/{prompts_per_step} step {step+1}/{steps}",
                    heartbeat_s=heartbeat_s,
                )
                logprobs_td = fwd_res.loss_fn_outputs[0]["logprobs"]
                sampling_logprobs = [float(x) for x in list(getattr(logprobs_td, "data", logprobs_td))]

                # Build advantage vector aligned with target_tokens (len = max_seq_len-1).
                n = max_seq_len - 1
                adv = [0.0] * n
                start_i = max(0, prompt_len - 1)
                end_i = min(n, start_i + len(completion_tokens) + 1)  # include EOS
                for i in range(start_i, end_i):
                    adv[i] = adv_scalar

                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=full_tokens[:-1]),
                    loss_fn_inputs={
                        "target_tokens": full_tokens[1:],
                        "logprobs": sampling_logprobs,
                        "advantages": adv,
                    },
                )
                datums.append(datum)
                step_rewards.append(reward)

        print(
            f"[{_ts()}] step {step+1}/{steps}: forward_backward loss_fn=ppo batch={len(datums)}",
            flush=True,
        )
        fb_future = training_client.forward_backward(datums, loss_fn="ppo")
        _wait_future(fb_future, label=f"forward_backward step {step+1}/{steps}", heartbeat_s=heartbeat_s)

        print(f"[{_ts()}] step {step+1}/{steps}: optim_step lr={lr}", flush=True)
        opt_future = training_client.optim_step(types.AdamParams(learning_rate=lr))
        _wait_future(opt_future, label=f"optim_step step {step+1}/{steps}", heartbeat_s=heartbeat_s)

        avg_reward = sum(step_rewards) / max(1, len(step_rewards))
        print(
            f"[{_ts()}] step {step+1}/{steps}: avg_reward={avg_reward:.4f} rewards={step_rewards}",
            flush=True,
        )

    print(f"[{_ts()}] RL finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
