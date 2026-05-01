#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

from openpi_libero_fast_rl import (
    _create_action_session,
    _create_model,
    _create_model_from_state,
    _delete_action_session,
    _delete_model,
    _forward_logprobs,
    _http_post,
    _make_rl_datum,
    _poll_future,
    _resolve_fast_tokenizer_path,
    _sample_actions,
    _save_weights_for_sampler,
    _tokenize_sampled_actions,
)
from openpi_libero_sft import _build_transform, _collect_transformed_items, _load_tasks, _plot_curve


def _save_training_state(base_url: str, model_id: str, checkpoint_name: str) -> str:
    resp = _http_post(
        f'{base_url}/api/v1/save_weights',
        payload={'model_id': model_id, 'path': checkpoint_name},
        timeout=120,
    )
    resp.raise_for_status()
    result = _poll_future(base_url, resp.json()['request_id'], timeout_s=3600)
    path = result.get('path')
    if not isinstance(path, str) or not path:
        raise RuntimeError(f'save_weights missing path: {result!r}')
    return path


def _ppo_train_step_batch(base_url: str, model_id: str, datums: list[dict], *, learning_rate: float) -> dict:
    resp = _http_post(
        f'{base_url}/api/v1/train_step',
        payload={
            'model_id': model_id,
            'forward_backward_input': {'loss_fn': 'ppo', 'loss_fn_config': {'epsilon': 0.2}, 'data': datums},
            'adam_params': {'learning_rate': float(learning_rate)},
        },
        timeout=120,
    )
    resp.raise_for_status()
    result = _poll_future(base_url, resp.json()['request_id'], timeout_s=3600)
    if not isinstance(result.get('metrics'), dict):
        raise RuntimeError(f'ppo_train failed: {result!r}')
    return result


def _forward_logprobs_batch(base_url: str, model_id: str, datums: list[dict]) -> list[list[float]]:
    payload_datums = json.loads(json.dumps(datums))
    for datum in payload_datums:
        zero_len = len(datum['loss_fn_inputs']['target_tokens']['data'])
        datum['loss_fn_inputs']['logprobs']['data'] = [0.0] * zero_len
        datum['loss_fn_inputs']['advantages']['data'] = [0.0] * zero_len
    resp = _http_post(
        f'{base_url}/api/v1/forward_backward',
        payload={
            'model_id': model_id,
            'forward_backward_input': {
                'loss_fn': 'importance_sampling',
                'data': payload_datums,
            },
        },
        timeout=120,
    )
    resp.raise_for_status()
    result = _poll_future(base_url, resp.json()['request_id'], timeout_s=3600)
    outputs = result.get('loss_fn_outputs')
    if not isinstance(outputs, list):
        raise RuntimeError(f'forward_logprobs batch failed: {result!r}')
    if len(outputs) != len(payload_datums):
        raise RuntimeError(
            f'forward_logprobs batch size mismatch: expected {len(payload_datums)} outputs, got {len(outputs)}'
        )
    return [[float(x) for x in output['logprobs']['data']] for output in outputs]


def _forward_logprobs_chunked(
    base_url: str,
    model_id: str,
    datums: list[dict],
    *,
    batch_size: int,
) -> list[list[float]]:
    if batch_size <= 0:
        raise ValueError(f'logprob batch size must be positive, got {batch_size}')
    outputs: list[list[float]] = []
    for start in range(0, len(datums), batch_size):
        outputs.extend(_forward_logprobs_batch(base_url, model_id, datums[start : start + batch_size]))
    return outputs


def _logprob_ratio_stats(
    old_logprobs: list[list[float]],
    new_logprobs: list[list[float]],
    *,
    clip_low: float,
    clip_high: float,
) -> tuple[float, float]:
    if len(old_logprobs) != len(new_logprobs):
        raise ValueError(
            f'post-update logprob batch length mismatch: old={len(old_logprobs)} new={len(new_logprobs)}'
        )
    ratio_means: list[float] = []
    clipfracs: list[float] = []
    for old_item, new_item in zip(old_logprobs, new_logprobs, strict=True):
        old_arr = np.asarray(old_item, dtype=np.float32).reshape(-1)
        new_arr = np.asarray(new_item, dtype=np.float32).reshape(-1)
        if old_arr.shape != new_arr.shape:
            raise ValueError(
                f'post-update logprob shape mismatch: old={old_arr.shape} new={new_arr.shape}'
            )
        if old_arr.size == 0:
            raise ValueError('post-update logprob stats require at least one token')
        log_ratio = np.clip(new_arr - old_arr, a_min=-20.0, a_max=20.0)
        ratio = np.exp(log_ratio)
        ratio_means.append(float(ratio.mean()))
        clipfracs.append(float(np.mean((ratio < clip_low) | (ratio > clip_high))))
    return float(sum(ratio_means) / len(ratio_means)), float(sum(clipfracs) / len(clipfracs))


def _same_state_diversity_probe(
    base_url: str,
    action_session_id: str,
    *,
    items: list[dict[str, Any]],
    item_indices: list[int],
    temperature: float,
    repeats: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item_index in item_indices:
        item = items[item_index]
        expert_actions = np.asarray(item["actions"], dtype=np.float32)
        samples = []
        rewards = []
        for _ in range(repeats):
            sampled_actions = _sample_actions(
                base_url,
                action_session_id,
                item,
                temperature=temperature,
            )
            samples.append(sampled_actions)
            rewards.append(float(-np.mean((sampled_actions - expert_actions) ** 2)))
        stacked = np.stack(samples, axis=0)
        rows.append(
            {
                "item_index": item_index,
                "reward_mean": float(np.mean(rewards)),
                "reward_std": float(np.std(rewards)),
                "unique_exact_actions": int(len({stacked[i].tobytes() for i in range(repeats)})),
                "max_pairwise_action_diff": float(
                    max(
                        np.max(np.abs(stacked[i] - stacked[j]))
                        for i in range(repeats)
                        for j in range(i + 1, repeats)
                    )
                    if repeats > 1
                    else 0.0
                ),
            }
        )
    return rows


def _action_hash(actions: np.ndarray) -> str:
    arr = np.asarray(actions, dtype=np.float32)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


def _token_hash(tokens: list[int]) -> str:
    return hashlib.sha1(json.dumps(tokens, separators=(",", ":")).encode("utf-8")).hexdigest()[:12]


def _evaluate_checkpoint_reward(
    base_url: str,
    *,
    base_model: str,
    checkpoint_path: str,
    eval_items: list[dict[str, Any]],
    temperature: float,
) -> tuple[float, float]:
    action_session_id = _create_action_session(
        base_url,
        base_model,
        checkpoint_path,
        timeout_s=3600.0,
    )
    try:
        rewards: list[float] = []
        for item in eval_items:
            sampled_actions = _sample_actions(
                base_url,
                action_session_id,
                item,
                temperature=temperature,
            )
            expert_actions = np.asarray(item['actions'], dtype=np.float32)
            rewards.append(float(-np.mean((sampled_actions - expert_actions) ** 2)))
        reward_arr = np.asarray(rewards, dtype=np.float32)
        return float(reward_arr.mean()), float(reward_arr.std())
    finally:
        _delete_action_session(base_url, action_session_id)


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
    parser.add_argument('--training-state-path', default=os.environ.get('OPENPI_VLA_TRAINING_STATE_PATH', ''))
    parser.add_argument('--train-epochs', type=int, default=int(os.environ.get('OPENPI_VLA_TRAIN_EPOCHS', '1')))
    parser.add_argument('--logprob-batch-size', type=int, default=int(os.environ.get('OPENPI_VLA_LOGPROB_BATCH_SIZE', '4')))
    parser.add_argument('--learning-rate', type=float, default=float(os.environ.get('OPENPI_VLA_LEARNING_RATE', '1e-5')))
    parser.add_argument('--temperature', type=float, default=float(os.environ.get('OPENPI_VLA_ACTION_TEMPERATURE', '0.3')))
    parser.add_argument('--eval-count', type=int, default=int(os.environ.get('OPENPI_VLA_EVAL_COUNT', '4')))
    parser.add_argument('--eval-temperature', type=float, default=float(os.environ.get('OPENPI_VLA_EVAL_TEMPERATURE', '0.0')))
    parser.add_argument(
        '--serialize-runtime-roles',
        action=argparse.BooleanOptionalAction,
        default=(os.environ.get('OPENPI_VLA_SERIALIZE_RUNTIME_ROLES', '1').strip().lower() in {'1', 'true', 'yes', 'on'}),
    )
    parser.add_argument('--eval-item-indices', default=os.environ.get('OPENPI_VLA_EVAL_ITEM_INDICES', ''))
    parser.add_argument('--train-item-indices', default=os.environ.get('OPENPI_VLA_TRAIN_ITEM_INDICES', ''))
    parser.add_argument('--diversity-probe-count', type=int, default=3)
    parser.add_argument('--diversity-probe-repeats', type=int, default=4)
    parser.add_argument(
        '--min-group-reward-std',
        type=float,
        default=float(os.environ.get('OPENPI_VLA_MIN_GROUP_REWARD_STD', '1e-5')),
    )
    parser.add_argument(
        '--max-group-resample-attempts',
        type=int,
        default=int(os.environ.get('OPENPI_VLA_MAX_GROUP_RESAMPLE_ATTEMPTS', '2')),
    )
    parser.add_argument(
        '--resample-temperature-step',
        type=float,
        default=float(os.environ.get('OPENPI_VLA_RESAMPLE_TEMPERATURE_STEP', '0.025')),
    )
    parser.add_argument(
        '--min-accepted-groups-per-step',
        type=int,
        default=int(os.environ.get('OPENPI_VLA_MIN_ACCEPTED_GROUPS_PER_STEP', '2')),
    )
    args = parser.parse_args()

    base_model = 'openpi/pi0-fast-libero-low-mem-finetune'
    base_url = args.base_url.rstrip('/')
    tasks = _load_tasks()
    task_text = tasks[args.task_index]
    cfg, tx = _build_transform(base_model)
    items, pool_meta = _collect_transformed_items(base_model, tx, task_text, int(cfg.model.action_horizon), max_episodes=args.max_episodes, stride=args.stride)
    rng = random.Random(args.seed)
    if len(items) < 2:
        raise RuntimeError(f'grouped RL requires at least 2 transformed items, got {len(items)}')

    if args.eval_item_indices or args.train_item_indices:
        eval_indices = [int(x) for x in args.eval_item_indices.split(',') if x.strip()]
        train_indices = [int(x) for x in args.train_item_indices.split(',') if x.strip()]
        if not eval_indices:
            raise ValueError('--eval-item-indices must be non-empty when explicit split is provided')
        if not train_indices:
            raise ValueError('--train-item-indices must be non-empty when explicit split is provided')
        overlap = set(eval_indices) & set(train_indices)
        if overlap:
            raise ValueError(f'explicit eval/train split overlaps on indices: {sorted(overlap)}')
        all_indices = set(range(len(items)))
        if not set(eval_indices).issubset(all_indices):
            raise ValueError(f'explicit eval indices out of range: {sorted(set(eval_indices) - all_indices)}')
        if not set(train_indices).issubset(all_indices):
            raise ValueError(f'explicit train indices out of range: {sorted(set(train_indices) - all_indices)}')
    else:
        split_order = list(range(len(items)))
        rng.shuffle(split_order)
        eval_count = min(max(int(args.eval_count), 1), len(items) - 1)
        eval_indices = split_order[:eval_count]
        train_indices = split_order[eval_count:]
    eval_items = [items[idx] for idx in eval_indices]

    from openpi.models.tokenizer import FASTTokenizer
    tokenizer = FASTTokenizer(cfg.model.max_token_len, fast_tokenizer_path=_resolve_fast_tokenizer_path())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / 'metrics.jsonl'
    run_log = out_dir / 'run.log'
    reward_steps, rewards, losses, train_rewards, eval_rewards = [], [], [], [], []
    with run_log.open('a', encoding='utf-8') as handle:
        handle.write(
            json.dumps(
                {
                    'event': 'split',
                    'eval_indices': eval_indices,
                    'train_indices': train_indices,
                }
            )
            + '\n'
        )

    if args.initial_action_ckpt:
        raise ValueError(
            '--initial-action-ckpt is invalid for grouped RL: sampler-only checkpoints cannot seed the '
            'training model, so PPO would compare against the wrong behavior policy'
        )

    if args.training_state_path:
        model_id = _create_model_from_state(
            base_url,
            base_model=base_model,
            state_path=args.training_state_path,
            load_optimizer=False,
        )
    else:
        model_id = _create_model(base_url, base_model)
    action_session_id = None
    baseline_eval_reward = None
    baseline_eval_reward_std = None
    baseline_train_reward = None
    baseline_train_reward_std = None
    try:
        baseline_state_ckpt = None
        if args.serialize_runtime_roles:
            with run_log.open('a', encoding='utf-8') as handle:
                handle.write('BASELINE save_train_state\n')
            baseline_state_ckpt = _save_training_state(base_url, model_id, 'group-rl-state-baseline')
            with run_log.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'event': 'baseline_state_ckpt', 'path': baseline_state_ckpt}) + '\n')
        with run_log.open('a', encoding='utf-8') as handle:
            handle.write('BASELINE save_eval_weights\n')
        baseline_eval_ckpt = _save_weights_for_sampler(base_url, model_id, 'group-rl-eval-baseline')
        if args.serialize_runtime_roles:
            _delete_model(base_url, model_id)
            model_id = None
        with run_log.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'event': 'baseline_eval_ckpt', 'path': baseline_eval_ckpt}) + '\n')
            handle.write('BASELINE train_action_session\n')
        baseline_train_reward, baseline_train_reward_std = _evaluate_checkpoint_reward(
            base_url,
            base_model=base_model,
            checkpoint_path=baseline_eval_ckpt,
            eval_items=[items[idx] for idx in train_indices],
            temperature=args.eval_temperature,
        )
        with run_log.open('a', encoding='utf-8') as handle:
            handle.write('BASELINE eval_action_session\n')
        baseline_eval_reward, baseline_eval_reward_std = _evaluate_checkpoint_reward(
            base_url,
            base_model=base_model,
            checkpoint_path=baseline_eval_ckpt,
            eval_items=eval_items,
            temperature=args.eval_temperature,
        )
        with run_log.open('a', encoding='utf-8') as handle:
            handle.write(
                json.dumps(
                    {
                        'event': 'baseline_eval',
                        'baseline_train_reward': baseline_train_reward,
                        'baseline_train_reward_std': baseline_train_reward_std,
                        'baseline_train_count': len(train_indices),
                        'eval_reward': baseline_eval_reward,
                        'eval_reward_std': baseline_eval_reward_std,
                        'eval_count': len(eval_items),
                    }
                )
                + '\n'
            )
        if args.serialize_runtime_roles:
            if not baseline_state_ckpt:
                raise RuntimeError('serialize_runtime_roles requires a baseline training checkpoint')
            with run_log.open('a', encoding='utf-8') as handle:
                handle.write('BASELINE reload_train_state\n')
            model_id = _create_model_from_state(
                base_url,
                base_model=base_model,
                state_path=baseline_state_ckpt,
                load_optimizer=True,
            )
            with run_log.open('a', encoding='utf-8') as handle:
                handle.write(
                    json.dumps(
                        {
                            'event': 'baseline_state_reloaded',
                            'path': baseline_state_ckpt,
                            'model_id': model_id,
                        }
                    )
                    + '\n'
                )
        for step in range(1, args.steps + 1):
            presample_state_ckpt = None
            if args.serialize_runtime_roles:
                with run_log.open('a', encoding='utf-8') as handle:
                    handle.write(f'STEP {step} save_train_state_presample\n')
                presample_state_ckpt = _save_training_state(base_url, model_id, f'group-rl-state-{step}-presample')
                with run_log.open('a', encoding='utf-8') as handle:
                    handle.write(
                        json.dumps(
                            {
                                'event': 'train_state_ckpt',
                                'step': step,
                                'phase': 'presample',
                                'path': presample_state_ckpt,
                            }
                        )
                        + '\n'
                    )
            with run_log.open('a', encoding='utf-8') as handle:
                handle.write(f'STEP {step} save_weights\n')
            ckpt = _save_weights_for_sampler(base_url, model_id, f'group-rl-{step}')
            if args.serialize_runtime_roles:
                _delete_model(base_url, model_id)
                model_id = None
            if action_session_id:
                _delete_action_session(base_url, action_session_id)
            with run_log.open('a', encoding='utf-8') as handle:
                handle.write(f'STEP {step} create_action_session\n')
            action_session_id = _create_action_session(base_url, base_model, ckpt, timeout_s=3600.0)
            diversity_probe_indices = train_indices[: max(1, min(args.diversity_probe_count, len(train_indices)))]
            diversity_rows = _same_state_diversity_probe(
                base_url,
                action_session_id,
                items=items,
                item_indices=diversity_probe_indices,
                temperature=args.temperature,
                repeats=args.diversity_probe_repeats,
            )
            with run_log.open('a', encoding='utf-8') as handle:
                handle.write(
                    json.dumps(
                        {
                            'event': 'same_state_diversity_probe',
                            'step': step,
                            'temperature': args.temperature,
                            'repeats': args.diversity_probe_repeats,
                            'rows': diversity_rows,
                        }
                    )
                    + '\n'
                )
            if all(row['unique_exact_actions'] == 1 for row in diversity_rows):
                raise RuntimeError(
                    'grouped RL is not meaningful for this checkpoint at the configured action temperature: '
                    f'same-state diversity probe found identical actions for all probe items {diversity_probe_indices}. '
                    f'rows={json.dumps(diversity_rows)}'
                )

            candidate_datums = []
            candidate_group_rows = []
            raw_rewards = []
            centered_rewards = []
            skipped_group_rows = []
            order = list(train_indices)
            rng.shuffle(order)
            candidate_item_attempt_count = max(args.groups_per_step, len(order))
            selected_item_indices: list[int] = []
            for item_attempt_idx in range(candidate_item_attempt_count):
                if len(selected_item_indices) >= args.groups_per_step:
                    break
                group_idx = len(selected_item_indices)
                item_index = order[item_attempt_idx % len(order)]
                item = items[item_index]
                accepted_group = False
                for group_attempt_idx in range(args.max_group_resample_attempts + 1):
                    sample_temperature = args.temperature + (
                        float(group_attempt_idx) * args.resample_temperature_step
                    )
                    group_datums = []
                    group_rewards = []
                    group_action_hashes = []
                    group_token_hashes = []
                    group_samples = []
                    for sample_in_group in range(args.group_size):
                        with run_log.open('a', encoding='utf-8') as handle:
                            handle.write(
                                json.dumps(
                                    {
                                        'event': 'candidate_sample',
                                        'step': step,
                                        'group_idx': group_idx,
                                        'item_attempt_idx': item_attempt_idx,
                                        'group_attempt_idx': group_attempt_idx,
                                        'sample_in_group': sample_in_group,
                                        'item_index': item_index,
                                        'temperature': sample_temperature,
                                    }
                                )
                                + '\n'
                            )
                        sampled_actions = _sample_actions(
                            base_url,
                            action_session_id,
                            item,
                            temperature=sample_temperature,
                        )
                        expert_actions = np.asarray(item['actions'], dtype=np.float32)
                        mse = float(np.mean((sampled_actions - expert_actions) ** 2))
                        reward = -mse
                        prefix_tokens, target_tokens, suffix_token_ar_mask = _tokenize_sampled_actions(
                            tokenizer,
                            task_text,
                            item,
                            sampled_actions,
                        )
                        if not target_tokens:
                            raise RuntimeError(
                                f'grouped RL sample produced empty target tokens at step {step}, '
                                f'group_idx={group_idx}, item_index={item_index}'
                            )
                        datum = _make_rl_datum(
                            item,
                            prefix_tokens,
                            target_tokens,
                            suffix_token_ar_mask,
                            logprobs=[0.0] * len(target_tokens),
                            advantages=[0.0] * len(target_tokens),
                        )
                        group_datums.append(datum)
                        group_rewards.append(reward)
                        group_action_hashes.append(_action_hash(sampled_actions))
                        group_token_hashes.append(_token_hash(target_tokens))
                        group_samples.append(np.asarray(sampled_actions, dtype=np.float32))
                    reward_arr = np.asarray(group_rewards, dtype=np.float32)
                    reward_std = float(reward_arr.std())
                    group_mean = float(reward_arr.mean())
                    group_centered_rewards = (reward_arr - group_mean).astype(np.float32)
                    unique_exact_actions = len(set(group_action_hashes))
                    unique_target_tokenizations = len(set(group_token_hashes))
                    stacked_samples = np.stack(group_samples, axis=0)
                    max_pairwise_action_diff = float(
                        max(
                            np.max(np.abs(stacked_samples[i] - stacked_samples[j]))
                            for i in range(len(group_samples))
                            for j in range(i + 1, len(group_samples))
                        )
                    ) if len(group_samples) > 1 else 0.0
                    group_event = {
                        'event': 'candidate_group',
                        'step': step,
                        'group_idx': group_idx,
                        'item_attempt_idx': item_attempt_idx,
                        'group_attempt_idx': group_attempt_idx,
                        'item_index': item_index,
                        'temperature': sample_temperature,
                        'rewards': group_rewards,
                        'reward_mean': group_mean,
                        'reward_std': reward_std,
                        'centered_rewards': group_centered_rewards.tolist(),
                        'unique_exact_actions': unique_exact_actions,
                        'unique_target_tokenizations': unique_target_tokenizations,
                        'action_hashes': group_action_hashes,
                        'token_hashes': group_token_hashes,
                        'max_pairwise_action_diff': max_pairwise_action_diff,
                    }
                    with run_log.open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(group_event) + '\n')
                    rejection_reasons = []
                    if reward_std <= args.min_group_reward_std:
                        rejection_reasons.append('reward_std_below_threshold')
                    if unique_exact_actions <= 1:
                        rejection_reasons.append('identical_actions')
                    if rejection_reasons:
                        with run_log.open('a', encoding='utf-8') as handle:
                            handle.write(
                                json.dumps(
                                    {
                                        'event': 'candidate_group_rejected',
                                        **group_event,
                                        'rejection_reasons': rejection_reasons,
                                        'min_group_reward_std': args.min_group_reward_std,
                                    }
                                )
                                + '\n'
                            )
                        if group_attempt_idx == args.max_group_resample_attempts:
                            skipped_group_rows.append(
                                {
                                    'group_idx': group_idx,
                                    'item_attempt_idx': item_attempt_idx,
                                    'item_index': item_index,
                                    'temperature': sample_temperature,
                                    'reward_std': reward_std,
                                    'unique_exact_actions': unique_exact_actions,
                                    'unique_target_tokenizations': unique_target_tokenizations,
                                    'max_pairwise_action_diff': max_pairwise_action_diff,
                                    'rejection_reasons': rejection_reasons,
                                }
                            )
                        continue
                    accepted_group = True
                    selected_item_indices.append(item_index)
                    for datum, centered_reward in zip(group_datums, group_centered_rewards.tolist(), strict=True):
                        candidate_datums.append(datum)
                        candidate_group_rows.append(
                            {
                                'step': step,
                                'group_idx': group_idx,
                                'item_index': item_index,
                                'temperature': sample_temperature,
                                'centered_reward': float(centered_reward),
                                'reward_mean': group_mean,
                                'reward_std': reward_std,
                                'unique_exact_actions': unique_exact_actions,
                                'unique_target_tokenizations': unique_target_tokenizations,
                                'max_pairwise_action_diff': max_pairwise_action_diff,
                            }
                        )
                        centered_rewards.append(float(centered_reward))
                    raw_rewards.extend(group_rewards)
                    break
                if not accepted_group:
                    continue

            accepted_group_count = len(selected_item_indices)
            with run_log.open('a', encoding='utf-8') as handle:
                handle.write(
                    json.dumps(
                        {
                            'event': 'candidate_group_selection_summary',
                            'step': step,
                            'requested_groups': args.groups_per_step,
                            'accepted_groups': accepted_group_count,
                            'min_accepted_groups_per_step': args.min_accepted_groups_per_step,
                            'selected_item_indices': selected_item_indices,
                            'skipped_groups': skipped_group_rows,
                        }
                    )
                    + '\n'
                )
            if accepted_group_count < args.min_accepted_groups_per_step:
                raise RuntimeError(
                    f'grouped RL step {step} accepted too few same-state groups: '
                    f'accepted={accepted_group_count} requested={args.groups_per_step} '
                    f'min_required={args.min_accepted_groups_per_step} '
                    f'skipped_groups={json.dumps(skipped_group_rows)}'
                )

            if args.serialize_runtime_roles:
                if action_session_id:
                    _delete_action_session(base_url, action_session_id)
                    action_session_id = None
                if not presample_state_ckpt:
                    raise RuntimeError(f'serialize_runtime_roles requires presample state checkpoint at step {step}')
                with run_log.open('a', encoding='utf-8') as handle:
                    handle.write(f'STEP {step} reload_train_state_presample\n')
                model_id = _create_model_from_state(
                    base_url,
                    base_model=base_model,
                    state_path=presample_state_ckpt,
                    load_optimizer=True,
                )
                with run_log.open('a', encoding='utf-8') as handle:
                    handle.write(
                        json.dumps(
                            {
                                'event': 'train_state_reloaded',
                                'step': step,
                                'phase': 'presample',
                                'path': presample_state_ckpt,
                                'model_id': model_id,
                            }
                        )
                        + '\n'
                    )

            if not candidate_datums:
                continue

            reward_arr = np.asarray(raw_rewards, dtype=np.float32)
            reward_std = float(reward_arr.std())
            centered_reward_arr = np.asarray(centered_rewards, dtype=np.float32)
            centered_reward_scale = float(centered_reward_arr.std())
            if centered_reward_scale <= 1e-6:
                raise RuntimeError(
                    f'grouped RL step {step} has zero centered-reward variance across same-state groups: '
                    f'accepted_groups={accepted_group_count} centered_rewards={centered_rewards} '
                    f'skipped_groups={json.dumps(skipped_group_rows)}'
                )
            for datum, row in zip(candidate_datums, candidate_group_rows, strict=True):
                target_len = len(datum['loss_fn_inputs']['target_tokens']['data'])
                if target_len <= 0:
                    raise RuntimeError(f'grouped RL datum has empty target token slice at step {step}')
                per_token_adv = float(row['centered_reward']) / centered_reward_scale / float(target_len)
                datum['loss_fn_inputs']['advantages']['data'] = [per_token_adv] * target_len
                row['target_len'] = target_len
                row['per_token_advantage'] = per_token_adv
            with run_log.open('a', encoding='utf-8') as handle:
                handle.write(
                    json.dumps(
                        {
                            'event': 'candidate_batch_advantages',
                            'step': step,
                            'centered_reward_scale': centered_reward_scale,
                            'rows': candidate_group_rows,
                        }
                    )
                    + '\n'
                )

            with run_log.open('a', encoding='utf-8') as handle:
                handle.write(f'STEP {step} forward_logprobs {len(candidate_datums)} datums\n')
            train_datums = json.loads(json.dumps(candidate_datums))
            batched_logprobs = _forward_logprobs_chunked(
                base_url,
                model_id,
                train_datums,
                batch_size=args.logprob_batch_size,
            )
            for datum, old_logprobs in zip(train_datums, batched_logprobs, strict=True):
                datum['loss_fn_inputs']['logprobs']['data'] = old_logprobs

            with run_log.open('a', encoding='utf-8') as handle:
                handle.write(f'STEP {step} ppo_train {len(train_datums)} datums epochs={args.train_epochs}\n')
            epoch_losses = []
            epoch_loss_abs = []
            epoch_ratios = []
            epoch_clipfracs = []
            post_update_ratio_mean = None
            post_update_clipfrac_mean = None
            result = None
            for epoch_idx in range(args.train_epochs):
                result = _ppo_train_step_batch(base_url, model_id, train_datums, learning_rate=args.learning_rate)
                epoch_losses.append(float(result['metrics']['loss:mean']))
                outputs = list(result.get('loss_fn_outputs') or [])
                loss_abs = float(sum(abs(float(output['loss']['data'][0])) for output in outputs) / max(len(outputs), 1))
                epoch_loss_abs.append(loss_abs)
                if 'ratio:mean' in result['metrics']:
                    epoch_ratios.append(float(result['metrics']['ratio:mean']))
                if 'clipfrac:mean' in result['metrics']:
                    epoch_clipfracs.append(float(result['metrics']['clipfrac:mean']))
                epoch_record = {
                    'event': 'epoch',
                    'step': step,
                    'epoch': epoch_idx + 1,
                    'loss': epoch_losses[-1],
                    'loss_abs_mean': loss_abs,
                    'ratio_mean': epoch_ratios[-1] if epoch_ratios else None,
                    'clipfrac_mean': epoch_clipfracs[-1] if epoch_clipfracs else None,
                    'learning_rate': args.learning_rate,
                    'temperature': args.temperature,
                }
                with run_log.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(epoch_record) + '\n')
                print(json.dumps(epoch_record), flush=True)
            post_update_logprobs = _forward_logprobs_chunked(
                base_url,
                model_id,
                train_datums,
                batch_size=args.logprob_batch_size,
            )
            post_update_ratio_mean, post_update_clipfrac_mean = _logprob_ratio_stats(
                batched_logprobs,
                post_update_logprobs,
                clip_low=0.8,
                clip_high=1.2,
            )
            mean_reward = float(sum(raw_rewards) / len(raw_rewards))
            mean_loss = float(sum(epoch_losses) / len(epoch_losses))
            mean_loss_abs = float(sum(epoch_loss_abs) / len(epoch_loss_abs))
            record = {'step': step, 'reward': mean_reward, 'reward_std': reward_std, 'loss': mean_loss, 'loss_last': epoch_losses[-1], 'loss_abs_mean': mean_loss_abs, 'loss_abs_last': epoch_loss_abs[-1], 'num_samples': len(raw_rewards), 'groups_per_step': args.groups_per_step, 'accepted_groups': accepted_group_count, 'skipped_group_count': len(skipped_group_rows), 'group_size': args.group_size, 'train_epochs': args.train_epochs, 'logprob_batch_size': args.logprob_batch_size, 'learning_rate': args.learning_rate, 'temperature': args.temperature, 'advantage_mode': 'same_state_group_centered_batchstd_per_token_normalized', 'centered_reward_scale': centered_reward_scale}
            if epoch_ratios:
                record['ratio_mean'] = float(sum(epoch_ratios) / len(epoch_ratios))
            if epoch_clipfracs:
                record['clipfrac_mean'] = float(sum(epoch_clipfracs) / len(epoch_clipfracs))
            record['post_update_ratio_mean'] = post_update_ratio_mean
            record['post_update_clipfrac_mean'] = post_update_clipfrac_mean
            posttrain_state_ckpt = None
            if args.serialize_runtime_roles:
                with run_log.open('a', encoding='utf-8') as handle:
                    handle.write(f'STEP {step} save_train_state_posttrain\n')
                posttrain_state_ckpt = _save_training_state(base_url, model_id, f'group-rl-state-{step}-posttrain')
                with run_log.open('a', encoding='utf-8') as handle:
                    handle.write(
                        json.dumps(
                            {
                                'event': 'train_state_ckpt',
                                'step': step,
                                'phase': 'posttrain',
                                'path': posttrain_state_ckpt,
                            }
                        )
                        + '\n'
                    )
            with run_log.open('a', encoding='utf-8') as handle:
                handle.write(f'STEP {step} save_eval_weights\n')
            eval_ckpt = _save_weights_for_sampler(base_url, model_id, f'group-rl-eval-{step}')
            if args.serialize_runtime_roles:
                _delete_model(base_url, model_id)
                model_id = None
            with run_log.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'event': 'eval_ckpt', 'step': step, 'path': eval_ckpt}) + '\n')
                handle.write(f'STEP {step} train_action_session\n')
            train_reward, train_reward_std = _evaluate_checkpoint_reward(
                base_url,
                base_model=base_model,
                checkpoint_path=eval_ckpt,
                eval_items=[items[idx] for idx in train_indices],
                temperature=args.eval_temperature,
            )
            with run_log.open('a', encoding='utf-8') as handle:
                handle.write(f'STEP {step} eval_action_session\n')
            eval_reward, eval_reward_std = _evaluate_checkpoint_reward(
                base_url,
                base_model=base_model,
                checkpoint_path=eval_ckpt,
                eval_items=eval_items,
                temperature=args.eval_temperature,
            )
            with run_log.open('a', encoding='utf-8') as handle:
                handle.write(
                    json.dumps(
                        {
                            'event': 'eval',
                            'step': step,
                            'train_reward': train_reward,
                            'train_reward_std': train_reward_std,
                            'train_count': len(train_indices),
                            'eval_reward': eval_reward,
                            'eval_reward_std': eval_reward_std,
                            'eval_count': len(eval_items),
                        }
                    )
                    + '\n'
                )
            if args.serialize_runtime_roles:
                if not posttrain_state_ckpt:
                    raise RuntimeError(f'serialize_runtime_roles requires posttrain state checkpoint at step {step}')
                with run_log.open('a', encoding='utf-8') as handle:
                    handle.write(f'STEP {step} reload_train_state_posttrain\n')
                model_id = _create_model_from_state(
                    base_url,
                    base_model=base_model,
                    state_path=posttrain_state_ckpt,
                    load_optimizer=True,
                )
                with run_log.open('a', encoding='utf-8') as handle:
                    handle.write(
                        json.dumps(
                            {
                                'event': 'train_state_reloaded',
                                'step': step,
                                'phase': 'posttrain',
                                'path': posttrain_state_ckpt,
                                'model_id': model_id,
                            }
                        )
                        + '\n'
                    )
            record['eval_reward'] = eval_reward
            record['eval_reward_std'] = eval_reward_std
            record['eval_count'] = len(eval_items)
            record['train_reward'] = train_reward
            record['train_reward_std'] = train_reward_std
            record['train_count'] = len(train_indices)
            with metrics_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
            reward_steps.append(step)
            rewards.append(mean_reward)
            losses.append(mean_loss)
            train_rewards.append(train_reward)
            eval_rewards.append(eval_reward)

        _plot_curve(reward_steps, rewards, out_dir / 'reward_curve.png', f'{base_model} | grouped reward task={args.task_index}')
        _plot_curve(reward_steps, losses, out_dir / 'loss_curve.png', f'{base_model} | grouped ppo loss task={args.task_index}')
        _plot_curve(
            reward_steps,
            train_rewards,
            out_dir / 'train_reward_curve.png',
            f'{base_model} | grouped train reward task={args.task_index}',
        )
        _plot_curve(reward_steps, eval_rewards, out_dir / 'eval_reward_curve.png', f'{base_model} | grouped eval reward task={args.task_index}')
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
            'initial_train_reward': train_rewards[0] if train_rewards else None,
            'final_train_reward': train_rewards[-1] if train_rewards else None,
            'max_train_reward': max(train_rewards) if train_rewards else None,
            'initial_eval_reward': eval_rewards[0] if eval_rewards else None,
            'final_eval_reward': eval_rewards[-1] if eval_rewards else None,
            'max_eval_reward': max(eval_rewards) if eval_rewards else None,
            'baseline_train_reward': baseline_train_reward,
            'baseline_train_reward_std': baseline_train_reward_std,
            'baseline_eval_reward': baseline_eval_reward,
            'baseline_eval_reward_std': baseline_eval_reward_std,
            'reward_curve_path': str(out_dir / 'reward_curve.png'),
            'loss_curve_path': str(out_dir / 'loss_curve.png'),
            'train_reward_curve_path': str(out_dir / 'train_reward_curve.png'),
            'eval_reward_curve_path': str(out_dir / 'eval_reward_curve.png'),
            'eval_indices': eval_indices,
            'train_indices': train_indices,
            'serialize_runtime_roles': bool(args.serialize_runtime_roles),
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
        if model_id:
            _delete_model(base_url, model_id)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
