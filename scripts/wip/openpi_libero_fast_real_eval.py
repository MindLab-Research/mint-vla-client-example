#!/usr/bin/env python3
from __future__ import annotations

import base64
import collections
import io
import json
import math
import time
import uuid
from pathlib import Path

import numpy as np
import requests
import sentencepiece
import torch
from PIL import Image
from transformers import AutoProcessor

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

_real_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _real_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
PALIGEMMA_TOKENIZER_PATH = Path('/vePFS-Mindverse/share/code/root/.openpi-data-vla-pr422/big_vision/paligemma_tokenizer.model')


def _poll_future(base_url: str, request_id: str, *, timeout_s: float = 3600.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = requests.post(f'{base_url}/api/v1/retrieve_future', json={'request_id': request_id}, timeout=120)
        if resp.status_code in {408, 503}:
            time.sleep(1.0)
            continue
        resp.raise_for_status()
        return resp.json()
    raise TimeoutError(f'timed out waiting for {request_id}')


def _create_model(base_url: str, base_model: str) -> str:
    payload = {
        'session_id': f'rl-{uuid.uuid4().hex[:12]}',
        'model_seq_id': 0,
        'base_model': base_model,
        'lora_config': {'rank': 16, 'train_attn': True, 'train_mlp': True, 'train_unembed': True},
        'user_metadata': {'script': 'scripts/wip/openpi_libero_fast_real_eval.py'},
    }
    resp = requests.post(f'{base_url}/api/v1/create_model', json=payload, timeout=120)
    resp.raise_for_status()
    result = _poll_future(base_url, resp.json()['request_id'])
    model_id = result.get('model_id')
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f'create_model missing model_id: {result!r}')
    return model_id


def _delete_model(base_url: str, model_id: str) -> None:
    try:
        requests.delete(f'{base_url}/api/v1/models/{model_id}', timeout=300)
    except Exception:
        pass


def _save_weights_for_sampler(base_url: str, model_id: str, checkpoint_name: str) -> str:
    resp = requests.post(f'{base_url}/api/v1/save_weights_for_sampler', json={'model_id': model_id, 'path': checkpoint_name}, timeout=120)
    resp.raise_for_status()
    result = _poll_future(base_url, resp.json()['request_id'])
    path = result.get('path')
    if not isinstance(path, str) or not path:
        raise RuntimeError(f'save_weights_for_sampler missing path: {result!r}')
    return path


def _create_action_session(base_url: str, base_model: str, model_path: str, *, timeout_s: float = 3600.0) -> str:
    deadline = time.time() + timeout_s
    while True:
        resp = requests.post(
            f'{base_url}/api/v1/mint/action_sessions',
            json={'session_id': f'act-{uuid.uuid4().hex[:12]}', 'base_model': base_model, 'model_path': model_path},
            timeout=3600,
        )
        if resp.status_code in {429, 503} and time.time() < deadline:
            time.sleep(2.0)
            continue
        resp.raise_for_status()
        result = resp.json()
        session_id = result.get('action_session_id')
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError(f'create_action_session missing action_session_id: {result!r}')
        return session_id


def _delete_action_session(base_url: str, action_session_id: str) -> None:
    try:
        requests.delete(f'{base_url}/api/v1/mint/action_sessions/{action_session_id}', timeout=120)
    except Exception:
        pass


def _resolve_fast_tokenizer_path() -> str:
    hf_home = Path('/vePFS-Mindverse/share/huggingface')
    repo_root = hf_home / 'hub' / 'models--physical-intelligence--fast'
    refs_main = repo_root / 'refs' / 'main'
    if refs_main.exists():
        revision = refs_main.read_text(encoding='utf-8').strip()
        if revision:
            snapshot_dir = repo_root / 'snapshots' / revision
            if snapshot_dir.exists():
                return str(snapshot_dir)
    snapshots_dir = repo_root / 'snapshots'
    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    return str(snapshots[-1])


class FASTTokenizerLite:
    def __init__(self, max_len: int = 256, fast_tokenizer_path: str = 'physical-intelligence/fast'):
        self._max_len = max_len
        with PALIGEMMA_TOKENIZER_PATH.open('rb') as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        self._fast_skip_tokens = 128

    def tokenize(self, prompt: str, state: np.ndarray, actions: np.ndarray | None):
        cleaned_text = prompt.lower().strip().replace('_', ' ')
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        state_str = ' '.join(map(str, discretized_state))
        prefix = f'Task: {cleaned_text}, State: {state_str};\n'
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)
        if actions is not None:
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)
            postfix_tokens = self._paligemma_tokenizer.encode('Action: ') + action_tokens_in_pg.tolist() + self._paligemma_tokenizer.encode('|', add_eos=True)
        else:
            postfix_tokens = []
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)
        if len(tokens) < self._max_len:
            padding = [False] * (self._max_len - len(tokens))
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]
        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def _act_tokens_to_paligemma_tokens(self, tokens):
        arr = np.asarray(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - arr


def _quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def _get_env(task, seed: int):
    task_description = task.language
    task_bddl_file = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=task_bddl_file, camera_heights=256, camera_widths=256)
    env.seed(seed)
    return env, task_description


def _make_state(obs):
    return np.concatenate((obs['robot0_eef_pos'], _quat2axisangle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos'])).astype(np.float32)


def _png_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _prompt_tokens(tokenizer: FASTTokenizerLite, task_text: str, state: np.ndarray):
    tokens, token_mask, _, _ = tokenizer.tokenize(task_text, state, None)
    return np.asarray(tokens)[np.asarray(token_mask).astype(bool)].astype(int).tolist()


def _make_action_observation(obs, task_text: str, tokenizer: FASTTokenizerLite):
    img = np.ascontiguousarray(obs['agentview_image'][::-1, ::-1])
    wrist = np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
    state = _make_state(obs)
    prompt_tokens = _prompt_tokens(tokenizer, task_text, state)
    return {
        'observation': {
            'state': {'data': state.tolist(), 'shape': list(state.shape), 'dtype': 'float32'},
            'model_input': {
                'chunks': [
                    {'type': 'image', 'data': _png_b64(img), 'format': 'png', 'expected_tokens': 256},
                    {'type': 'image', 'data': _png_b64(wrist), 'format': 'png', 'expected_tokens': 256},
                    {'type': 'image', 'data': _png_b64(wrist), 'format': 'png', 'expected_tokens': 256},
                    {'type': 'encoded_text', 'tokens': prompt_tokens},
                ]
            },
        }
    }


def _act(base_url: str, action_session_id: str, obs, task_text: str, tokenizer: FASTTokenizerLite):
    resp = requests.post(
        f'{base_url}/api/v1/mint/action_sessions/{action_session_id}/act',
        json=_make_action_observation(obs, task_text, tokenizer),
        timeout=120,
    )
    resp.raise_for_status()
    result = _poll_future(base_url, resp.json()['request_id'])
    arr = np.asarray(result['actions']['data'], dtype=np.float32)
    return arr.reshape(result['actions']['shape'])


def main():
    base_url = 'http://localhost:8000'
    base_model = 'openpi/pi0-fast-libero-low-mem-finetune'
    bench = benchmark.get_benchmark_dict()['libero_spatial']()
    task = bench.get_task(0)
    init_states = bench.get_task_init_states(0)
    env, task_text = _get_env(task, seed=7)
    tokenizer = FASTTokenizerLite(180, fast_tokenizer_path=_resolve_fast_tokenizer_path())
    model_id = _create_model(base_url, base_model)
    action_session_id = None
    try:
        ckpt = _save_weights_for_sampler(base_url, model_id, f'eval-{uuid.uuid4().hex[:8]}')
        action_session_id = _create_action_session(base_url, base_model, ckpt, timeout_s=3600.0)
        successes = 0
        for ep in range(3):
            env.reset()
            obs = env.set_init_state(init_states[ep])
            action_plan = collections.deque()
            done = False
            t = 0
            while t < 230:
                if t < 10:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue
                if not action_plan:
                    chunk = _act(base_url, action_session_id, obs, task_text, tokenizer)
                    action_plan.extend(chunk[:5])
                action = action_plan.popleft()
                obs, reward, done, info = env.step(action.tolist())
                t += 1
                if done:
                    successes += 1
                    break
            print(json.dumps({'episode': ep, 'done': bool(done), 'steps': t}), flush=True)
        print(json.dumps({'successes': successes, 'episodes': 3, 'replan_steps': 5, 'task': task_text}), flush=True)
    finally:
        if action_session_id:
            _delete_action_session(base_url, action_session_id)
        _delete_model(base_url, model_id)


if __name__ == '__main__':
    main()
