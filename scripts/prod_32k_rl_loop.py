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
    models_csv = (
        os.environ.get("BASE_MODELS")
        or os.environ.get("BASE_MODEL")
        or (sys.argv[1] if len(sys.argv) > 1 else "")
    ).strip()
    if not models_csv:
        print("BASE_MODELS or BASE_MODEL env var (or argv[1]) required", file=sys.stderr)
        return 2
    base_models = [m.strip() for m in models_csv.split(",") if m.strip()]

    lora_rank = int(os.environ.get("LORA_RANK", "16"))
    lr = float(os.environ.get("RL_LEARNING_RATE", "5e-5"))
    steps = int(os.environ.get("RL_STEPS", "2"))
    prompts_per_step = int(os.environ.get("RL_PROMPTS_PER_STEP", "4"))
    samples_per_prompt = int(os.environ.get("RL_SAMPLES_PER_PROMPT", "2"))
    max_seq_len = int(os.environ.get("RL_MAX_SEQ_LEN", "32000"))
    gen_max_tokens = int(os.environ.get("RL_GEN_MAX_TOKENS", "32"))
    temperature = float(os.environ.get("RL_TEMPERATURE", "0.7"))
    heartbeat_s = float(os.environ.get("MINT_FUTURE_HEARTBEAT_S", "60"))

    print(f"[{_ts()}] RL start base_models={base_models} lora_rank={lora_rank} lr={lr}", flush=True)
    print(
        f"[{_ts()}] config steps={steps} prompts_per_step={prompts_per_step} "
        f"samples_per_prompt={samples_per_prompt} max_seq_len={max_seq_len} "
        f"gen_max_tokens={gen_max_tokens} temperature={temperature}",
        flush=True,
    )

    random.seed(42)

    print(f"[{_ts()}] create ServiceClient", flush=True)
    service_client = mint.ServiceClient()

    for base_model in base_models:
        print(f"[{_ts()}] model start base_model={base_model}", flush=True)

        print(f"[{_ts()}] create_lora_training_client start base_model={base_model}", flush=True)
        training_client = service_client.create_lora_training_client(
            base_model=base_model,
            rank=lora_rank,
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
        )
        print(f"[{_ts()}] get_tokenizer start base_model={base_model}", flush=True)
        tokenizer = training_client.get_tokenizer()
        print(f"[{_ts()}] get_tokenizer done base_model={base_model} vocab_size={tokenizer.vocab_size}", flush=True)
        eos_id = tokenizer.eos_token_id

        filler_ids = tokenizer.encode(" a", add_special_tokens=False) or tokenizer.encode("0", add_special_tokens=False)
        if not filler_ids:
            raise RuntimeError("Failed to get filler token id from tokenizer")
        filler_id = int(filler_ids[0])

        for step in range(steps):
            ckpt_name = f"{base_model.replace('/', '_')}_rl_step_{step:04d}"
            print(
                f"[{_ts()}] step {step+1}/{steps}: save_weights_and_get_sampling_client start base_model={base_model} name={ckpt_name}",
                flush=True,
            )
            t0 = time.time()
            sampling_client = training_client.save_weights_and_get_sampling_client(name=ckpt_name)
            print(
                f"[{_ts()}] step {step+1}/{steps}: save_weights_and_get_sampling_client done base_model={base_model} "
                f"elapsed_s={time.time()-t0:.1f}",
                flush=True,
            )

            datums: list[types.Datum] = []
            step_rewards: list[float] = []

            for p in range(prompts_per_step):
                a = random.randint(10, 99)
                b = random.randint(10, 99)
                expected = a * b

                prompt_text = f"Question: What is {a} * {b}?\\nAnswer:"
                base_prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)

                # Stress vLLM prefill at ~32K by padding prompt tokens up to max_seq_len.
                # Leave room for generation + EOS.
                desired_prompt_len = max(2, max_seq_len - gen_max_tokens - 1)
                prompt_tokens = _pad_to(base_prompt_tokens, desired_prompt_len, filler_id)
                prompt = types.ModelInput.from_ints(tokens=prompt_tokens)

                print(
                    f"[{_ts()}] step {step+1}/{steps}: rollout prompt {p+1}/{prompts_per_step} "
                    f"base_model={base_model} num_samples={samples_per_prompt} prompt_len={len(prompt_tokens)}",
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
                    label=f"sample base_model={base_model} prompt {p+1}/{prompts_per_step} step {step+1}/{steps}",
                    heartbeat_s=heartbeat_s,
                )

                rewards: list[float] = []
                token_payloads: list[tuple[list[int], int, list[int]]] = []
                # Each payload: (full_tokens, prompt_len, completion_tokens)

                for seq in sample_res.sequences:
                    completion_tokens = list(seq.tokens)
                    # Some backends return full sequence tokens; normalize.
                    if completion_tokens[: len(prompt_tokens)] == prompt_tokens:
                        full_tokens = completion_tokens
                        completion_tokens = completion_tokens[len(prompt_tokens) :]
                    else:
                        full_tokens = prompt_tokens + completion_tokens

                    if not full_tokens or full_tokens[-1] != eos_id:
                        full_tokens = full_tokens + [eos_id]

                    full_tokens = _pad_to(full_tokens, max_seq_len, filler_id)
                    completion_text = tokenizer.decode(completion_tokens)
                    pred = _first_int(completion_text)
                    reward = 1.0 if pred == expected else 0.0
                    rewards.append(reward)
                    token_payloads.append((full_tokens, len(prompt_tokens), completion_tokens))

                mean_reward = sum(rewards) / max(1, len(rewards))

                for reward, (full_tokens, prompt_len, completion_tokens) in zip(rewards, token_payloads):
                    adv_scalar = float(reward - mean_reward)

                    # vLLM prompt_logprobs must support ~32K context for 32K RL.
                    lp_future = sampling_client.compute_logprobs(types.ModelInput.from_ints(tokens=full_tokens))
                    lp_res = _wait_future(
                        lp_future,
                        label=f"compute_logprobs base_model={base_model} prompt {p+1}/{prompts_per_step} step {step+1}/{steps}",
                        heartbeat_s=heartbeat_s,
                    )
                    # API returns len(full_tokens) values with index 0 as placeholder; align to target_tokens.
                    lp = [(-100.0 if x is None else float(x)) for x in list(lp_res)]
                    sampling_logprobs = lp[1:]

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
                f"[{_ts()}] step {step+1}/{steps}: forward_backward loss_fn=ppo base_model={base_model} batch={len(datums)}",
                flush=True,
            )
            fb_future = training_client.forward_backward(datums, loss_fn="ppo")
            _wait_future(
                fb_future,
                label=f"forward_backward base_model={base_model} step {step+1}/{steps}",
                heartbeat_s=heartbeat_s,
            )

            print(f"[{_ts()}] step {step+1}/{steps}: optim_step base_model={base_model} lr={lr}", flush=True)
            opt_future = training_client.optim_step(types.AdamParams(learning_rate=lr))
            _wait_future(
                opt_future,
                label=f"optim_step base_model={base_model} step {step+1}/{steps}",
                heartbeat_s=heartbeat_s,
            )

            avg_reward = sum(step_rewards) / max(1, len(step_rewards))
            print(
                f"[{_ts()}] step {step+1}/{steps}: base_model={base_model} avg_reward={avg_reward:.4f} rewards={step_rewards}",
                flush=True,
            )

        print(f"[{_ts()}] model finished base_model={base_model}", flush=True)

    print(f"[{_ts()}] RL finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
