#!/usr/bin/env python3
"""Pure-HTTP multi-producer training client for OpenPI pi0.5 (LoRA).

Does NOT import mint_server and does NOT subprocess the mint driver -- it
talks to the mint-server over HTTP only. Multi-producer prefetch is built in
(no BatchPrefetcher single-producer limitation), so bs=128 saturates the GPU
(~52% sm%, ~80% busy on 8 GPUs) when --num-producers is left at auto.

Layering: this script needs the openpi package (data transforms) on PYTHONPATH
(it lives in MINT_GRB_ROOT/src/openpi/src, added by scripts/remote/run_client.sh).
It does NOT need mint_server code -- camera_layout comes from the client's own
openpi_profiles.py.

Usage (via the repo's launcher, which sets PYTHONPATH/env):
  bash scripts/remote/run_client.sh scripts/train/train_http_multiprod.py \
    --lance-dataset <path> --steps 2000
Recommended (GPU-saturating, auto producer count):
  --batch-size 128            # 8-GPU data-parallel (must be a multiple of 8)
  --num-producers auto        # default; auto-selects 8 at bs=128
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np

import openpi_vla_smoke_lance_base as L

BASE_MODEL = L.LEGACY_L_LORA_MODEL  # openpi/pi05-libero-low-mem-finetune


class MultiProducerPrefetcher:
    """N-producer batch builder over one shared bounded output queue.

    Single producer (pid 0) reuses the same build callable -> identical batch
    order + RNG draws to the serial loop (reproducible). Producers pid>=1 each
    own their own dataset/data_config/rng (thread-safety). A Semaphore(steps)
    work permit ensures exactly `steps` batches total.
    """

    def __init__(
        self,
        prefetch_depth: int,
        build_callables: list[Callable[[], list[dict[str, Any]]]],
        *,
        max_batches: int | None = None,
    ) -> None:
        if prefetch_depth <= 0:
            raise ValueError("prefetch_depth must be positive")
        if not build_callables:
            raise ValueError("build_callables must be non-empty")
        self._queue: queue.Queue[list[dict[str, Any]]] = queue.Queue(maxsize=prefetch_depth)
        self._slots = threading.Semaphore(prefetch_depth)
        self._build_callables = build_callables
        self._max_batches = max_batches
        self._permits = (
            threading.Semaphore(max_batches) if max_batches is not None else None
        )
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._stats_lock = threading.Lock()
        self.build_seconds = 0.0
        self.batches_built = 0
        self._threads = [
            threading.Thread(target=self._produce, args=(pid,),
                             name=f"prefetch-{pid}", daemon=True)
            for pid in range(len(build_callables))
        ]
        for t in self._threads:
            t.start()

    def _produce(self, pid: int) -> None:
        try:
            build_next = self._build_callables[pid]
            while not self._stop.is_set():
                if self._permits is not None and not self._permits.acquire(timeout=0.1):
                    return
                if self._stop.is_set():
                    if self._permits is not None:
                        self._permits.release()
                    return
                started = time.perf_counter()
                batch = build_next()
                elapsed = time.perf_counter() - started
                with self._stats_lock:
                    self.build_seconds += elapsed
                    self.batches_built += 1
                while not self._stop.is_set():
                    try:
                        self._queue.put(batch, timeout=1.0)
                        break
                    except queue.Full:
                        continue
                else:
                    if self._permits is not None:
                        self._permits.release()
        except BaseException as exc:  # noqa: BLE001
            self._error = exc

    def next_batch(self) -> list[dict[str, Any]]:
        while True:
            if self._error is not None:
                raise self._error
            try:
                batch = self._queue.get(timeout=0.1)
                self._slots.release()
                return batch
            except queue.Empty:
                if not any(t.is_alive() for t in self._threads):
                    if self._error is not None:
                        raise self._error
                    raise RuntimeError("producers stopped unexpectedly")

    def close(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5.0)


def auto_num_producers(batch_size: int) -> int:
    """Calibrated (2026-07-28, 8x A800, mano): bs<=8->1, 64->4, 128->8.
    Rule: num_producers >= ceil(data_load/gpu_compute). 8 @ bs=128 -> sm% 55.7%.
    """
    if batch_size <= 8:
        return 1
    if batch_size <= 64:
        return 4
    return 8


def _compute_norm_stats_fast(dataset) -> dict:
    """Vectorized norm stats (PyArrow .take().flatten()), ~12s on 6.8M frames.

    Replaces L._compute_norm_stats, which loops over dataset._index in pure
    Python (6.8M iterations -> minutes). This mirrors the mint smoke impl but
    is self-contained (Lance + openpi normalize). frame_window="full" means the
    training population IS every raw frame, so raw per-frame stats are correct.
    """
    import openpi.shared.normalize as normalize
    state_col = dataset._dataset.to_table(columns=["state"]).column("state")
    state_flat = np.asarray(state_col.combine_chunks().flatten().flatten(), dtype=np.float32)
    state_dim = int(np.asarray(state_col[0])[0].shape[0])
    state_arr = state_flat.reshape(-1, state_dim)
    actions_col = dataset._dataset.to_table(columns=["actions"]).column("actions")
    actions_flat = np.asarray(actions_col.combine_chunks().flatten().flatten(), dtype=np.float32)
    actions_dim = int(np.asarray(actions_col[0])[0].shape[0])
    actions_arr = actions_flat.reshape(-1, actions_dim)
    ss = normalize.RunningStats(); ss.update(state_arr)
    astat = normalize.RunningStats(); astat.update(actions_arr)
    return {"state": ss.get_statistics(), "actions": astat.get_statistics()}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=os.environ.get("MINT_BASE_URL", "http://127.0.0.1:30530"))
    p.add_argument("--api-key", default=os.environ.get("MINT_API_KEY", "tml-dummy"))
    p.add_argument("--lance-dataset", type=Path, required=True)
    p.add_argument("--model", default=BASE_MODEL, choices=L.MODEL_CHOICES)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-producers", default="auto", help="'auto' (default) or explicit int")
    p.add_argument("--prefetch-depth", type=int, default=0, help="0=max(2,num_producers)")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-checkpoint-name", default=None, help="omit to skip save (probe only)")
    p.add_argument("--output-json", default="")
    args = p.parse_args()

    # action_dim hard check (mirrors the mint skill invariant).
    # frame_window="full" avoids the contact-window manifest (this script trains
    # on the raw libero schema, not the MANO contact-window contract).
    dataset = L.LanceViewpi05Dataset(args.lance_dataset, action_horizon=10, frame_window="full")
    sample = dataset[0]
    action_dim = int(sample["observation/state"].shape[0])
    if action_dim != 32:
        print(f"error: action_dim {action_dim} != 32 (hard invariant). See ActionHeadSummary.md.",
              file=sys.stderr)
        return 2

    # num_producers
    if args.num_producers == "auto":
        num_producers = auto_num_producers(args.batch_size)
        print(json.dumps({"num_producers_auto": num_producers,
                          "batch_size": args.batch_size,
                          "note": "pass --num-producers <N> to override"}), flush=True)
    else:
        num_producers = max(1, int(args.num_producers))
    prefetch_depth = args.prefetch_depth or max(2, num_producers)
    if num_producers > 1:
        os.environ["MINT_PI05_NO_BATCH_CACHE"] = "1"

    # data config (shared, read-only norm_stats; per-producer dataset/data_config below)
    model_cfg = L._build_model_config(10, action_dim=32)
    norm_stats = _compute_norm_stats_fast(dataset)
    data_config = L._make_data_config(model_cfg, norm_stats)
    sample_rng = np.random.default_rng(args.seed)
    augmentation_rng = np.random.default_rng(args.seed + 1)

    base_url = args.base_url.rstrip("/")
    headers = L._headers(args.api_key)
    model_id, _ = L._create_model(base_url, headers, base_model=args.model)
    print(f"model created: {model_id}", flush=True)

    # Build per-producer callables. pid 0 reuses the main dataset/rng (reproducible
    # at num_producers==1). pid>=1 get their own dataset/data_config/rng.
    def _make_fn(_ds, _dc, _srng, _arng):
        def _fn():
            indices = _ds.sample_indices(args.batch_size, _srng)
            return L._build_batch(_ds, _dc, base_model=args.model, indices=indices)
        return _fn

    build_callables = [_make_fn(dataset, data_config, sample_rng, augmentation_rng)]
    for pid in range(1, num_producers):
        p_ds = L.LanceViewpi05Dataset(args.lance_dataset, action_horizon=10, frame_window="full")
        p_dc = L._make_data_config(model_cfg, norm_stats)
        p_srng = np.random.default_rng(args.seed + 1 + pid)
        p_arng = np.random.default_rng(args.seed + 1 + pid)
        build_callables.append(_make_fn(p_ds, p_dc, p_srng, p_arng))

    prefetcher = MultiProducerPrefetcher(prefetch_depth, build_callables, max_batches=args.steps)
    steps_log: list[dict[str, Any]] = []
    started = time.time()
    print(f"{'step':>4} {'loss':>8}  (bs={args.batch_size} producers={num_producers}, sync HTTP + multi-producer prefetch)", flush=True)

    try:
        for step in range(1, args.steps + 1):
            t0 = time.perf_counter()
            batch = prefetcher.next_batch()  # built by producer threads; queue_wait~0 when fed
            t1 = time.perf_counter()
            result = L._await_result(base_url, headers, L._post_json(
                base_url, "/api/v1/mint/vla/train_step", headers,
                {"model_id": model_id, "loss_fn": "flow_matching", "data": batch},
            ))
            t2 = time.perf_counter()
            metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
            loss = metrics.get("loss:mean")
            steps_log.append({"step": step, "loss": loss, "metrics": metrics})
            print(json.dumps({"step": step, "loss": loss,
                               "queue_wait": round(t1 - t0, 3),
                               "step_time": round(t2 - t0, 3)}), flush=True)

        save_result = {}
        if args.save_checkpoint_name:
            save_result = L._await_result(base_url, headers, L._post_json(
                base_url, "/api/v1/save_weights_for_sampler", headers,
                {"model_id": model_id, "path": args.save_checkpoint_name},
            ))
            print(f"checkpoint: {save_result.get('path')}", flush=True)
    finally:
        prefetcher.close()
        L._delete_model(base_url, headers, model_id)

    elapsed = time.time() - started
    losses = [s["loss"] for s in steps_log if s.get("loss") is not None]
    print(f"\nfinal loss: {losses[-1]:.4f}  (first: {losses[0]:.4f})  "
          f"steps={len(steps_log)} elapsed={elapsed:.0f}s "
          f"throughput={args.batch_size*len(steps_log)/max(1,elapsed):.1f} samples/s", flush=True)
    print(f"done: model_id={model_id}  checkpoint_saved={bool(args.save_checkpoint_name)}", flush=True)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps({
            "model_id": model_id, "base_model": args.model,
            "steps": steps_log, "save_result": save_result,
            "num_producers": num_producers, "batch_size": args.batch_size,
            "elapsed_seconds": elapsed,
        }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
