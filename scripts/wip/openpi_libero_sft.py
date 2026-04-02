#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import collections
import io
import json
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw

import openpi.training.config as config_mod
from openpi import transforms as T
from tinker_server.backend.model_registry import MODEL_CONFIGS

DATASET_ROOT = Path('/vePFS-Mindverse/share/code/conley/.hf-lerobot/physical-intelligence/libero')
ASSETS_DIR_BY_BASE_MODEL = {
    'openpi/pi0-fast-libero-low-mem-finetune': Path('/vePFS-Mindverse/share/code/conley/openpi/assets/pi0_fast_libero_low_mem_finetune'),
    'openpi/pi05-libero-low-mem-finetune': Path('/vePFS-Mindverse/share/code/conley/openpi/assets/pi05_libero'),
}
CONFIG_NAME_BY_BASE_MODEL = {
    'openpi/pi0-fast-libero-low-mem-finetune': 'pi0_fast_libero_low_mem_finetune',
    'openpi/pi05-libero-low-mem-finetune': 'pi05_libero',
}
DEFAULT_OPENPI_DATA_HOME = '/vePFS-Mindverse/share/code/conley/.openpi_cache'
DEFAULT_HF_HOME = '/vePFS-Mindverse/share/huggingface'


def _load_tasks() -> dict[int, str]:
    tasks_path = DATASET_ROOT / 'meta/tasks.jsonl'
    tasks: dict[int, str] = {}
    for line in tasks_path.read_text().splitlines():
        row = json.loads(line)
        tasks[int(row['task_index'])] = str(row['task'])
    return tasks


def _load_episode_task_map() -> list[tuple[int, str, int]]:
    episodes_path = DATASET_ROOT / 'meta/episodes.jsonl'
    rows: list[tuple[int, str, int]] = []
    for line in episodes_path.read_text().splitlines():
        row = json.loads(line)
        rows.append((int(row['episode_index']), str(row['tasks'][0]), int(row['length'])))
    return rows


def _decode_image(cell: dict[str, Any]) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(cell['bytes'])).convert('RGB'))


def _encode_png_base64(image: np.ndarray) -> str:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    bio = io.BytesIO()
    Image.fromarray(arr).save(bio, format='PNG')
    return base64.b64encode(bio.getvalue()).decode('utf-8')


def _episode_path(episode_index: int) -> Path:
    chunk = episode_index // 1000
    return DATASET_ROOT / f'data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet'


def _build_transform(base_model: str):
    config_name = CONFIG_NAME_BY_BASE_MODEL[base_model]
    assets_dir = ASSETS_DIR_BY_BASE_MODEL[base_model]
    cfg = config_mod.get_config(config_name)
    data_cfg = cfg.data.create(assets_dir, cfg.model)
    tx = T.compose([
        *data_cfg.repack_transforms.inputs,
        *data_cfg.data_transforms.inputs,
        T.Normalize(data_cfg.norm_stats, use_quantiles=data_cfg.use_quantile_norm),
        *data_cfg.model_transforms.inputs,
    ])
    return cfg, tx


def _iter_windows_for_task(task_text: str, action_horizon: int, *, max_episodes: int, stride: int):
    selected = [row for row in _load_episode_task_map() if row[1] == task_text][:max_episodes]
    if not selected:
        raise ValueError(f'No episodes found for task: {task_text!r}')
    for episode_index, _task, length in selected:
        ep = pd.read_parquet(_episode_path(episode_index))
        upper = max(0, min(len(ep), length) - action_horizon + 1)
        for start in range(0, upper, stride):
            row0 = ep.iloc[start]
            actions = np.stack([np.asarray(x, dtype=np.float32) for x in ep['actions'].iloc[start:start + action_horizon]])
            yield {
                'image': _decode_image(row0['image']),
                'wrist_image': _decode_image(row0['wrist_image']),
                'state': np.asarray(row0['state'], dtype=np.float32),
                'actions': actions,
                'prompt': task_text,
                'episode_index': episode_index,
                'start_index': start,
            }


def _fast_datum_from_transformed(base_model: str, item: dict[str, Any]) -> dict[str, Any]:
    model_cfg = MODEL_CONFIGS[base_model]
    tokens = np.asarray(item['tokenized_prompt'])
    token_mask = np.asarray(item['tokenized_prompt_mask']).astype(bool)
    loss_mask = np.asarray(item['token_loss_mask']).astype(bool) & token_mask
    prefix_mask = token_mask & ~loss_mask
    token_ar_mask = np.asarray(item['token_ar_mask'])
    prefix_tokens = tokens[prefix_mask].astype(int).tolist()
    target_tokens = tokens[loss_mask].astype(int).tolist()
    if not target_tokens:
        raise ValueError('FAST transformed sample produced empty target token slice')
    suffix_ar_mask = token_ar_mask[loss_mask].astype(int).tolist()
    image_chunks = []
    for camera_name in model_cfg.camera_layout:
        image_chunks.append({
            'type': 'image',
            'data': _encode_png_base64(np.asarray(item['image'][camera_name])),
            'format': 'png',
            'expected_tokens': 256,
        })
    return {
        'observation': {
            'state': {
                'data': np.asarray(item['state'], dtype=np.float32).reshape(-1).tolist(),
                'shape': list(np.asarray(item['state']).shape),
                'dtype': 'float32',
            },
            'model_input': {
                'chunks': [*image_chunks, {'type': 'encoded_text', 'tokens': prefix_tokens}],
            },
        },
        'supervision': {
            'target_tokens': {'data': target_tokens, 'shape': [len(target_tokens)], 'dtype': 'int64'},
            'weights': {'data': [1.0] * len(target_tokens), 'shape': [len(target_tokens)], 'dtype': 'float32'},
            'token_ar_mask': {'data': suffix_ar_mask, 'shape': [len(suffix_ar_mask)], 'dtype': 'int64'},
        },
    }


def _pi05_datum_from_transformed(base_model: str, item: dict[str, Any]) -> dict[str, Any]:
    model_cfg = MODEL_CONFIGS[base_model]
    prompt_tokens = np.asarray(item['tokenized_prompt'])[np.asarray(item['tokenized_prompt_mask']).astype(bool)].astype(int).tolist()
    actions = np.asarray(item['actions'], dtype=np.float32)
    image_chunks = []
    for camera_name in model_cfg.camera_layout:
        image_chunks.append({
            'type': 'image',
            'data': _encode_png_base64(np.asarray(item['image'][camera_name])),
            'format': 'png',
            'expected_tokens': 256,
        })
    return {
        'observation': {
            'state': {
                'data': np.asarray(item['state'], dtype=np.float32).reshape(-1).tolist(),
                'shape': list(np.asarray(item['state']).shape),
                'dtype': 'float32',
            },
            'model_input': {'chunks': [*image_chunks, {'type': 'encoded_text', 'tokens': prompt_tokens}]},
        },
        'supervision': {
            'actions': {'data': actions.reshape(-1).tolist(), 'shape': list(actions.shape), 'dtype': 'float32'},
        },
    }


def _collect_transformed_items(base_model: str, tx, task_text: str, action_horizon: int, *, max_episodes: int, stride: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items: list[dict[str, Any]] = []
    token_lengths: list[int] = []
    for raw in _iter_windows_for_task(task_text, action_horizon, max_episodes=max_episodes, stride=stride):
        transformed = tx(dict(raw))
        items.append(transformed)
        if 'pi0-fast' in base_model:
            token_lengths.append(int(np.asarray(transformed['tokenized_prompt_mask']).astype(bool).sum()))
    meta: dict[str, Any] = {'raw_pool_size': len(items)}
    if 'pi0-fast' in base_model and token_lengths:
        common_len, common_count = collections.Counter(token_lengths).most_common(1)[0]
        items = [item for item, length in zip(items, token_lengths) if length == common_len]
        meta['fixed_token_len'] = common_len
        meta['fixed_token_count'] = common_count
    if not items:
        raise RuntimeError('No transformed items collected')
    meta['final_pool_size'] = len(items)
    return items, meta


def _poll_future(base_url: str, request_id: str, *, timeout_s: float = 3600.0) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = requests.post(f'{base_url}/api/v1/retrieve_future', json={'request_id': request_id}, timeout=120)
        if resp.status_code == 408:
            time.sleep(1.0)
            continue
        resp.raise_for_status()
        return resp.json()
    raise TimeoutError(f'timed out waiting for {request_id}')


def _create_model(base_url: str, base_model: str) -> tuple[str, str]:
    session_id = f'sft-{uuid.uuid4().hex[:12]}'
    payload = {
        'session_id': session_id,
        'model_seq_id': 0,
        'base_model': base_model,
        'lora_config': {'rank': 16, 'train_attn': True, 'train_mlp': True, 'train_unembed': True},
        'user_metadata': {'script': 'scripts/wip/openpi_libero_sft.py'},
    }
    resp = requests.post(f'{base_url}/api/v1/create_model', json=payload, timeout=120)
    resp.raise_for_status()
    rid = resp.json()['request_id']
    result = _poll_future(base_url, rid, timeout_s=1800)
    return result['model_id'], session_id


def _delete_model(base_url: str, model_id: str) -> None:
    try:
        requests.delete(f'{base_url}/api/v1/models/{model_id}', timeout=120)
    except Exception:
        pass


def _plot_curve(steps: list[int], losses: list[float], output_path: Path, title: str) -> None:
    width, height = 1200, 700
    margin_left, margin_top, margin_right, margin_bottom = 90, 40, 40, 80
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    if not steps:
        img.save(output_path)
        return
    min_loss = min(losses)
    max_loss = max(losses)
    if max_loss == min_loss:
        max_loss = min_loss + 1.0
    min_step = min(steps)
    max_step = max(steps)
    if max_step == min_step:
        max_step = min_step + 1
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill='black', width=2)
    draw.line((margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h), fill='black', width=2)
    pts = []
    for step, loss in zip(steps, losses):
        x = margin_left + (step - min_step) / (max_step - min_step) * plot_w
        y = margin_top + plot_h - (loss - min_loss) / (max_loss - min_loss) * plot_h
        pts.append((x, y))
    if len(pts) >= 2:
        draw.line(pts, fill='blue', width=3)
    for x, y in pts:
        r = 4
        draw.ellipse((x - r, y - r, x + r, y + r), fill='red')
    draw.text((margin_left, 10), title, fill='black')
    draw.text((margin_left, height - 30), 'Step', fill='black')
    draw.text((10, margin_top), f'loss [{min_loss:.3f}, {max_loss:.3f}]', fill='black')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://localhost:8000')
    parser.add_argument('--base-model', required=True, choices=sorted(CONFIG_NAME_BY_BASE_MODEL))
    parser.add_argument('--task-index', type=int, required=True)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--max-episodes', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    os.environ.setdefault('OPENPI_DATA_HOME', DEFAULT_OPENPI_DATA_HOME)
    os.environ.setdefault('HF_HOME', DEFAULT_HF_HOME)

    rng = random.Random(args.seed)
    tasks = _load_tasks()
    task_text = tasks[args.task_index]
    cfg, tx = _build_transform(args.base_model)
    items, pool_meta = _collect_transformed_items(args.base_model, tx, task_text, int(cfg.model.action_horizon), max_episodes=args.max_episodes, stride=args.stride)

    base_url = args.base_url.rstrip('/')
    model_id, session_id = _create_model(base_url, args.base_model)
    print(json.dumps({'event': 'model_created', 'model_id': model_id, 'session_id': session_id, 'task': task_text, **pool_meta}), flush=True)

    loss_fn = 'cross_entropy' if 'pi0-fast' in args.base_model else 'flow_matching'
    datum_builder = _fast_datum_from_transformed if 'pi0-fast' in args.base_model else _pi05_datum_from_transformed
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / 'metrics.jsonl'

    steps: list[int] = []
    losses: list[float] = []
    try:
        for step in range(1, args.steps + 1):
            batch = [datum_builder(args.base_model, rng.choice(items)) for _ in range(args.batch_size)]
            payload = {'model_id': model_id, 'loss_fn': loss_fn, 'data': batch}
            resp = requests.post(f'{base_url}/api/v1/mint/vla/train_step', json=payload, timeout=120)
            resp.raise_for_status()
            request_id = resp.json()['request_id']
            result = _poll_future(base_url, request_id, timeout_s=1800)
            loss = float(result['metrics']['loss:mean'])
            steps.append(step)
            losses.append(loss)
            record = {'step': step, 'loss': loss, 'metrics': result['metrics']}
            with metrics_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
        _plot_curve(steps, losses, output_dir / 'loss_curve.png', f"{args.base_model} | task={args.task_index}")
        summary = {
            'base_model': args.base_model,
            'task_index': args.task_index,
            'task': task_text,
            'model_id': model_id,
            'steps': args.steps,
            'batch_size': args.batch_size,
            'initial_loss': losses[0],
            'final_loss': losses[-1],
            'min_loss': min(losses),
            'curve_path': str(output_dir / 'loss_curve.png'),
            **pool_meta,
        }
        (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps({'event': 'done', **summary}), flush=True)
    finally:
        _delete_model(base_url, model_id)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
