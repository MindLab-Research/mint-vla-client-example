from __future__ import annotations

import collections
import json
import math
import uuid
import os
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw
from libero.libero import benchmark
from openpi_libero_fast_real_eval_report import (
    FASTTokenizerLite,
    LIBERO_DUMMY_ACTION,
    _create_action_session,
    _create_model,
    _delete_action_session,
    _delete_model,
    _get_env,
    _make_action_observation,
    _make_state,
    _poll_future,
    _request_headers,
    _resolve_fast_tokenizer_path,
    _save_weights_for_sampler,
)

BASE_URL = os.environ.get('MINT_BASE_URL', 'http://localhost:18080')
BASE_MODEL = 'openpi/pi0-fast-libero-low-mem-finetune'
OUT_DIR = Path(os.environ.get('OPENPI_VLA_OUT_DIR', '/vePFS-Mindverse/share/code/root/mint-server-pr422-vla-20260402/results/rl_pi0fast_fixed_eval_object0_v9'))
OUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH = OUT_DIR / 'metrics.jsonl'
TASK_SUITE = 'libero_object'
TASK_INDEX = 0
TRAIN_STATE_INDEX = 0
EVAL_STATE_INDICES = [0, 0]
UPDATES = int(os.environ.get('OPENPI_VLA_UPDATES', '4'))
REPLAN_STEPS = int(os.environ.get('OPENPI_VLA_REPLAN_STEPS', '5'))
NUM_STEPS_WAIT = 10
MAX_STEPS = int(os.environ.get('OPENPI_VLA_MAX_STEPS', '20'))
LR = 1e-3
TRAIN_EPOCHS = int(os.environ.get('OPENPI_VLA_TRAIN_EPOCHS', '8'))
GAMMA = 0.99
OBJ_BODY = 'alphabet_soup_1_main'
TARGET_SITE = 'basket_1_contain_region'
SUCCESS_BONUS = 5.0
TRAIN_TEMPERATURE = 1.0
EVAL_TEMPERATURE = 0.0
INITIAL_ACTION_CKPT = os.environ.get('OPENPI_VLA_INITIAL_ACTION_CKPT', '').strip()
TASK_OBJ = None


def plot_curve(xs, ys, out_path, title, ylabel):
    width, height = 1200, 700
    margin_left, margin_top, margin_right, margin_bottom = 90, 40, 40, 80
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    if not xs:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        return
    min_y = min(ys)
    max_y = max(ys)
    if max_y == min_y:
        max_y = min_y + 1.0
    min_x = min(xs)
    max_x = max(xs)
    if max_x == min_x:
        max_x = min_x + 1
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill='black', width=2)
    draw.line((margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h), fill='black', width=2)
    pts = []
    for x_val, y_val in zip(xs, ys):
        px = margin_left + (x_val - min_x) / (max_x - min_x) * plot_w
        py = margin_top + plot_h - (y_val - min_y) / (max_y - min_y) * plot_h
        pts.append((px, py))
    if len(pts) >= 2:
        draw.line(pts, fill='blue', width=3)
    for px, py in pts:
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill='red')
    draw.text((margin_left, 10), title, fill='black')
    draw.text((margin_left, height - 30), 'Update', fill='black')
    draw.text((10, margin_top), f'{ylabel} [{min_y:.3f}, {max_y:.3f}]', fill='black')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def act_safe(base_url, action_session_id, obs, task_text, tokenizer, *, temperature=0.0):
    resp = requests.post(
        f'{base_url}/api/v1/mint/action_sessions/{action_session_id}/act',
        json={**_make_action_observation(obs, task_text, tokenizer), 'temperature': temperature},
        timeout=120,
        headers=_request_headers(),
    )
    resp.raise_for_status()
    result = _poll_future(base_url, resp.json()['request_id'], timeout_s=3600.0)
    if 'actions' not in result:
        raise RuntimeError(json.dumps(result))
    arr = np.asarray(result['actions']['data'], dtype=np.float32)
    return arr.reshape(result['actions']['shape'])


def tokenize_executed(tokenizer, task_text, obs, actions):
    state = _make_state(obs)
    tokens, token_mask, token_ar_mask, loss_mask = tokenizer.tokenize(task_text, state, np.asarray(actions, dtype=np.float32))
    prefix_mask = np.asarray(token_mask).astype(bool) & ~np.asarray(loss_mask).astype(bool)
    target_mask = np.asarray(loss_mask).astype(bool)
    prefix_tokens = np.asarray(tokens)[prefix_mask].astype(int).tolist()
    target_tokens = np.asarray(tokens)[target_mask].astype(int).tolist()
    suffix_token_ar_mask = np.asarray(token_ar_mask)[target_mask].astype(int).tolist()
    return prefix_tokens, target_tokens, suffix_token_ar_mask


def build_model_input(obs, task_text, tokenizer):
    return _make_action_observation(obs, task_text, tokenizer)['observation']['model_input']


def make_rl_datum(obs, task_text, tokenizer, actions, logprobs, advantages):
    prefix_tokens, target_tokens, suffix_token_ar_mask = tokenize_executed(tokenizer, task_text, obs, actions)
    if not (len(target_tokens) == len(logprobs) == len(advantages) == len(suffix_token_ar_mask)):
        raise ValueError('token/logprob/advantage length mismatch')
    state = _make_state(obs)
    return {
        'model_input': build_model_input(obs, task_text, tokenizer),
        'loss_fn_inputs': {
            'state': {'data': state.tolist(), 'shape': list(state.shape), 'dtype': 'float32'},
            'target_tokens': {'data': target_tokens, 'shape': [len(target_tokens)], 'dtype': 'int64'},
            'weights': {'data': [1.0] * len(target_tokens), 'shape': [len(target_tokens)], 'dtype': 'float32'},
            'token_ar_mask': {'data': suffix_token_ar_mask, 'shape': [len(suffix_token_ar_mask)], 'dtype': 'int64'},
            'logprobs': {'data': logprobs, 'shape': [len(logprobs)], 'dtype': 'float32'},
            'advantages': {'data': advantages, 'shape': [len(advantages)], 'dtype': 'float32'},
        },
    }


def forward_logprobs(base_url, model_id, datum):
    zero_len = len(datum['loss_fn_inputs']['target_tokens']['data'])
    payload = json.loads(json.dumps(datum))
    payload['loss_fn_inputs']['logprobs']['data'] = [0.0] * zero_len
    payload['loss_fn_inputs']['advantages']['data'] = [0.0] * zero_len
    resp = requests.post(
        f'{base_url}/api/v1/forward_backward',
        json={'model_id': model_id, 'forward_backward_input': {'loss_fn': 'importance_sampling', 'data': [payload]}},
        timeout=120,
        headers=_request_headers(),
    )
    resp.raise_for_status()
    result = _poll_future(base_url, resp.json()['request_id'], timeout_s=3600.0)
    return [float(x) for x in result['loss_fn_outputs'][0]['logprobs']['data']]


def discounted_returns(rewards, gamma):
    out = [0.0] * len(rewards)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = float(rewards[i]) + gamma * running
        out[i] = running
    return out


def normalize(vals):
    arr = np.asarray(vals, dtype=np.float32)
    if arr.size == 0:
        return []
    centered = arr - float(arr.mean())
    scale = float(arr.std())
    if scale < 1e-6:
        return centered.tolist()
    return (centered / scale).tolist()


def ppo_train_step(base_url, model_id, datums):
    resp = requests.post(
        f'{base_url}/api/v1/train_step',
        json={
            'model_id': model_id,
            'forward_backward_input': {'loss_fn': 'ppo', 'loss_fn_config': {'epsilon': 0.2}, 'data': datums},
            'adam_params': {'learning_rate': LR},
        },
        timeout=120,
        headers=_request_headers(),
    )
    resp.raise_for_status()
    return _poll_future(base_url, resp.json()['request_id'], timeout_s=3600.0)


def snapshot_obs(obs):
    keys = ('agentview_image', 'robot0_eye_in_hand_image', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos')
    return {k: np.asarray(obs[k]).copy() for k in keys}


def shaped_reward(env, obs, obj_body_id, target_site_id, obj_start_z, done):
    obj_pos = np.asarray(env.sim.data.body_xpos[obj_body_id], dtype=np.float32)
    target_pos = np.asarray(env.sim.data.site_xpos[target_site_id], dtype=np.float32)
    ee_pos = np.asarray(obs['robot0_eef_pos'], dtype=np.float32)
    obj_target = float(np.linalg.norm(obj_pos - target_pos))
    ee_obj = float(np.linalg.norm(ee_pos - obj_pos))
    lift = max(0.0, float(obj_pos[2] - obj_start_z))
    return 1.25 * math.exp(-6.0 * obj_target) + 0.35 * math.exp(-8.0 * ee_obj) + 4.0 * lift + (SUCCESS_BONUS if done else 0.0)


def rollout_episode(action_session_id, init_state_index, *, train_mode, temperature):
    env, _ = _get_env(TASK_OBJ, seed=7 + init_state_index)
    model = env.sim.model
    obj_body_id = next(i for i in range(model.nbody) if model.body(i).name == OBJ_BODY)
    target_site_id = next(i for i in range(model.nsite) if model.site(i).name == TARGET_SITE)
    try:
        env.reset()
        obs = env.set_init_state(init_states[init_state_index])
        obj_start_z = float(env.sim.data.body_xpos[obj_body_id][2])
        action_plan = collections.deque()
        done = False
        t = 0
        episode_reward = 0.0
        chunk_rewards = []
        chunk_records = []
        while t < MAX_STEPS + NUM_STEPS_WAIT:
            if t < NUM_STEPS_WAIT:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue
            if not action_plan:
                obs_before = snapshot_obs(obs)
                sampled_chunk = act_safe(BASE_URL, action_session_id, obs_before, task_text, tokenizer, temperature=temperature)
                executed = np.asarray(sampled_chunk[:REPLAN_STEPS], dtype=np.float32)
                action_plan.extend(executed)
                if train_mode:
                    chunk_records.append({'obs': obs_before, 'actions': executed})
            chunk_reward = 0.0
            while action_plan:
                action = np.asarray(action_plan.popleft(), dtype=np.float32)
                obs, reward, done, info = env.step(action.tolist())
                step_reward = shaped_reward(env, obs, obj_body_id, target_site_id, obj_start_z, done)
                chunk_reward += step_reward
                episode_reward += step_reward
                t += 1
                if done:
                    action_plan.clear()
                    break
            if train_mode:
                chunk_rewards.append(chunk_reward)
            if done:
                break
        return {
            'episode_reward': float(episode_reward),
            'success': 1.0 if done else 0.0,
            'chunk_rewards': chunk_rewards,
            'chunk_records': chunk_records,
        }
    finally:
        env.close()


def main():
    bench = benchmark.get_benchmark_dict()[TASK_SUITE]()
    global init_states, task_text, tokenizer, TASK_OBJ
    TASK_OBJ = bench.get_task(TASK_INDEX)
    init_states = bench.get_task_init_states(TASK_INDEX)
    task_text = TASK_OBJ.language
    tokenizer = FASTTokenizerLite(180, fast_tokenizer_path=_resolve_fast_tokenizer_path())
    update_steps = []
    train_rewards = []
    eval_rewards = []
    eval_success = []
    loss_curve = []
    model_id = _create_model(BASE_URL, BASE_MODEL)
    action_session_id = None
    try:
        for update_idx in range(1, UPDATES + 1):
            if update_idx == 1 and INITIAL_ACTION_CKPT:
                ckpt = INITIAL_ACTION_CKPT
            else:
                ckpt = _save_weights_for_sampler(BASE_URL, model_id, f'fixed-train-{uuid.uuid4().hex[:8]}')
            if action_session_id:
                _delete_action_session(BASE_URL, action_session_id)
            action_session_id = _create_action_session(BASE_URL, BASE_MODEL, ckpt, timeout_s=3600.0)
            train_rollout = rollout_episode(action_session_id, TRAIN_STATE_INDEX, train_mode=True, temperature=TRAIN_TEMPERATURE)
            returns = discounted_returns(train_rollout['chunk_rewards'], GAMMA)
            advantages = normalize(returns)
            datums = []
            for rec, adv in zip(train_rollout['chunk_records'], advantages):
                _, target_tokens, _ = tokenize_executed(tokenizer, task_text, rec['obs'], rec['actions'])
                zeroes = [0.0] * len(target_tokens)
                probe = make_rl_datum(rec['obs'], task_text, tokenizer, rec['actions'], zeroes, zeroes)
                old_logprobs = forward_logprobs(BASE_URL, model_id, probe)
                datums.append(make_rl_datum(rec['obs'], task_text, tokenizer, rec['actions'], old_logprobs, [float(adv)] * len(old_logprobs)))
            if datums:
                train_result = None
                for _ in range(TRAIN_EPOCHS):
                    train_result = ppo_train_step(BASE_URL, model_id, datums)
            else:
                train_result = {'metrics': {'loss:mean': 0.0}}
            assert train_result is not None
            loss_value = float(train_result['metrics']['loss:mean'])
            ckpt_eval = _save_weights_for_sampler(BASE_URL, model_id, f'fixed-eval-{uuid.uuid4().hex[:8]}')
            _delete_action_session(BASE_URL, action_session_id)
            action_session_id = _create_action_session(BASE_URL, BASE_MODEL, ckpt_eval, timeout_s=3600.0)
            eval_rollouts = [rollout_episode(action_session_id, idx, train_mode=False, temperature=EVAL_TEMPERATURE) for idx in EVAL_STATE_INDICES]
            eval_reward_mean = float(sum(x['episode_reward'] for x in eval_rollouts) / len(eval_rollouts))
            eval_success_mean = float(sum(x['success'] for x in eval_rollouts) / len(eval_rollouts))
            record = {
                'update': update_idx,
                'train_episode_reward': float(train_rollout['episode_reward']),
                'eval_mean_reward': eval_reward_mean,
                'eval_success_rate': eval_success_mean,
                'ppo_loss': loss_value,
            }
            with METRICS_PATH.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
            update_steps.append(update_idx)
            train_rewards.append(float(train_rollout['episode_reward']))
            eval_rewards.append(eval_reward_mean)
            eval_success.append(eval_success_mean)
            loss_curve.append(loss_value)
        plot_curve(update_steps, train_rewards, OUT_DIR / 'train_reward_curve.png', f'{BASE_MODEL} | fixed-state train reward', 'train_reward')
        plot_curve(update_steps, eval_rewards, OUT_DIR / 'eval_reward_curve.png', f'{BASE_MODEL} | fixed-state eval reward', 'eval_reward')
        plot_curve(update_steps, eval_success, OUT_DIR / 'eval_success_curve.png', f'{BASE_MODEL} | fixed-state eval success', 'eval_success')
        plot_curve(update_steps, loss_curve, OUT_DIR / 'loss_curve.png', f'{BASE_MODEL} | fixed-state ppo loss', 'loss')
        summary = {
            'base_model': BASE_MODEL,
            'task_suite': TASK_SUITE,
            'task_index': TASK_INDEX,
            'task': task_text,
            'updates': UPDATES,
            'train_state_index': TRAIN_STATE_INDEX,
            'eval_state_indices': EVAL_STATE_INDICES,
            'max_steps': MAX_STEPS,
            'replan_steps': REPLAN_STEPS,
            'train_reward_curve_path': str(OUT_DIR / 'train_reward_curve.png'),
            'eval_reward_curve_path': str(OUT_DIR / 'eval_reward_curve.png'),
            'eval_success_curve_path': str(OUT_DIR / 'eval_success_curve.png'),
            'loss_curve_path': str(OUT_DIR / 'loss_curve.png'),
            'initial_eval_mean_reward': eval_rewards[0] if eval_rewards else None,
            'final_eval_mean_reward': eval_rewards[-1] if eval_rewards else None,
            'max_eval_mean_reward': max(eval_rewards) if eval_rewards else None,
            'initial_eval_success_rate': eval_success[0] if eval_success else None,
            'final_eval_success_rate': eval_success[-1] if eval_success else None,
        }
        (OUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps({'event': 'done', **summary}), flush=True)
    finally:
        if action_session_id:
            _delete_action_session(BASE_URL, action_session_id)
        _delete_model(BASE_URL, model_id)


if __name__ == '__main__':
    main()
