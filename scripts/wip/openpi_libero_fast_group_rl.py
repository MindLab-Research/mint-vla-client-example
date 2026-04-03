#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import requests


def _request_headers() -> dict[str, str]:
    api_key = (os.environ.get("TINKER_API_KEY") or os.environ.get("MINT_API_KEY") or "").strip()
    if not api_key:
        return {}
    return {"X-API-Key": api_key}


_orig_post = requests.post
_orig_delete = requests.delete

def _post(*args, **kwargs):
    headers = dict(kwargs.pop("headers", {}) or {})
    headers.update(_request_headers())
    kwargs["headers"] = headers
    return _orig_post(*args, **kwargs)

def _delete(*args, **kwargs):
    headers = dict(kwargs.pop("headers", {}) or {})
    headers.update(_request_headers())
    kwargs["headers"] = headers
    return _orig_delete(*args, **kwargs)

requests.post = _post
requests.delete = _delete

from openpi_libero_fast_rl import (
    _create_action_session,
    _create_model,
    _delete_action_session,
    _delete_model,
    _forward_logprobs,
    _make_rl_datum,
    _poll_future,
    _resolve_fast_tokenizer_path,
    _sample_actions,
    _save_weights_for_sampler,
    _tokenize_sampled_actions,
)
from openpi_libero_sft import _build_transform, _collect_transformed_items, _load_tasks, _plot_curve


def _ppo_train_step_batch(base_url: str, model_id: str, datums: list[dict]) -> dict:
    resp = requests.post(
        f'{base_url}/api/v1/train_step',
        json={
            'model_id': model_id,
            'forward_backward_input': {'loss_fn': 'ppo', 'loss_fn_config': {'epsilon': 0.2}, 'data': datums},
            'adam_params': {'learning_rate': 1e-4},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return _poll_future(base_url, resp.json()['request_id'], timeout_s=3600)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default=os.environ.get('TINKER_BASE_URL', 'http://localhost:8000'))
    parser.add_argument('--task-index', type=int, default=16)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--groups-per-step', type=int, default=4)
    parser.add_argument('--group-size', type=int, default=4)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--max-episodes', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--initial-action-ckpt', default=os.environ.get('OPENPI_VLA_INITIAL_ACTION_CKPT', ''))
    args = parser.parse_args()

    base_model = 'openpi/pi0-fast-libero-low-mem-finetune'
    base_url = args.base_url.rstrip('/')
    tasks = _load_tasks()
    task_text = tasks[args.task_index]
    cfg, tx = _build_transform(base_model)
    items, pool_meta = _collect_transformed_items(base_model, tx, task_text, int(cfg.model.action_horizon), max_episodes=args.max_episodes, stride=args.stride)
    rng = random.Random(args.seed)

    from openpi.models.tokenizer import FASTTokenizer
    tokenizer = FASTTokenizer(cfg.model.max_token_len, fast_tokenizer_path=_resolve_fast_tokenizer_path())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / 'metrics.jsonl'
    run_log = out_dir / 'run.log'
    reward_steps, rewards, losses = [], [], []

    model_id = _create_model(base_url, base_model)
    action_session_id = None
    try:
        for step in range(1, args.steps + 1):
            with run_log.open('a', encoding='utf-8') as handle:
                handle.write(f'STEP {step} save_weights\n')
            if step == 1 and args.initial_action_ckpt:
                ckpt = args.initial_action_ckpt
            else:
                ckpt = _save_weights_for_sampler(base_url, model_id, f'group-rl-{step}')
            if action_session_id:
                _delete_action_session(base_url, action_session_id)
            with run_log.open('a', encoding='utf-8') as handle:
                handle.write(f'STEP {step} create_action_session\n')
            action_session_id = _create_action_session(base_url, base_model, ckpt, timeout_s=3600.0)

            candidate_datums = []
            raw_rewards = []
            for _ in range(args.groups_per_step):
                item = items[rng.randrange(len(items))]
                group = []
                group_rewards = []
                for _ in range(args.group_size):
                    sampled_actions = _sample_actions(base_url, action_session_id, item)
                    expert_actions = np.asarray(item['actions'], dtype=np.float32)
                    mse = float(np.mean((sampled_actions - expert_actions) ** 2))
                    reward = -mse
                    prefix_tokens, target_tokens, suffix_token_ar_mask = _tokenize_sampled_actions(tokenizer, task_text, item, sampled_actions)
                    if not target_tokens:
                        continue
                    group.append((item, prefix_tokens, target_tokens, suffix_token_ar_mask, reward))
                    group_rewards.append(reward)
                if not group_rewards:
                    continue
                r = np.asarray(group_rewards, dtype=np.float32)
                adv = r - r.mean()
                std = float(r.std())
                if std > 1e-6:
                    adv = adv / std
                for (item, prefix_tokens, target_tokens, suffix_token_ar_mask, reward), adv_val in zip(group, adv.tolist()):
                    datum = _make_rl_datum(item, prefix_tokens, target_tokens, suffix_token_ar_mask, logprobs=[0.0] * len(target_tokens), advantages=[adv_val] * len(target_tokens))
                    candidate_datums.append(datum)
                    raw_rewards.append(reward)

            if not candidate_datums:
                continue

            with run_log.open('a', encoding='utf-8') as handle:
                handle.write(f'STEP {step} forward_logprobs {len(candidate_datums)} datums\n')
            train_datums = json.loads(json.dumps(candidate_datums))
            for datum in train_datums:
                datum['loss_fn_inputs']['logprobs']['data'] = _forward_logprobs(base_url, model_id, datum)

            with run_log.open('a', encoding='utf-8') as handle:
                handle.write(f'STEP {step} ppo_train {len(train_datums)} datums\n')
            result = _ppo_train_step_batch(base_url, model_id, train_datums)
            mean_reward = float(sum(raw_rewards) / len(raw_rewards))
            mean_loss = float(result['metrics']['loss:mean'])
            record = {'step': step, 'reward': mean_reward, 'loss': mean_loss, 'num_samples': len(raw_rewards), 'groups_per_step': args.groups_per_step, 'group_size': args.group_size}
            with metrics_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
            reward_steps.append(step)
            rewards.append(mean_reward)
            losses.append(mean_loss)

        _plot_curve(reward_steps, rewards, out_dir / 'reward_curve.png', f'{base_model} | grouped reward task={args.task_index}')
        _plot_curve(reward_steps, losses, out_dir / 'loss_curve.png', f'{base_model} | grouped ppo loss task={args.task_index}')
        summary = {
            'base_model': base_model,
            'task_index': args.task_index,
            'task': task_text,
            'steps': args.steps,
            'groups_per_step': args.groups_per_step,
            'group_size': args.group_size,
            'initial_reward': rewards[0] if rewards else None,
            'final_reward': rewards[-1] if rewards else None,
            'max_reward': max(rewards) if rewards else None,
            'initial_loss': losses[0] if losses else None,
            'final_loss': losses[-1] if losses else None,
            'reward_curve_path': str(out_dir / 'reward_curve.png'),
            'loss_curve_path': str(out_dir / 'loss_curve.png'),
            **pool_meta,
        }
        (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps({'event': 'done', **summary}), flush=True)
    except Exception as exc:
        with run_log.open('a', encoding='utf-8') as handle:
            handle.write('ERROR\n' + repr(exc) + '\n')
        raise
    finally:
        if action_session_id:
            _delete_action_session(base_url, action_session_id)
        _delete_model(base_url, model_id)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
