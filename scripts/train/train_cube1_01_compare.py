#!/usr/bin/env python3
"""Train reproducible pi0.5 LoRA experiments from explicit or full Lance rows.

The client owns row scheduling and training-only augmentation. State noise is
injected before prompt tokenization; PD-target noise is injected into normalized
supervision actions, leaving the observed state and padding dimensions unchanged.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import sys
import threading
import time
import uuid
from typing import Any, Callable, Iterable

import lance
import numpy as np

import openpi_vla_smoke_lance_base as L
from scripts.gesture_language import (
    DEFAULT_GESTURE_INDEX_PATH,
    GestureIndex,
    format_gesture_prompt,
)
from scripts.mano_state_contract import (
    CONTACT_RULE,
    CONTACT_SEMANTICS,
    STATE_CONTRACT_ID,
    verify_locked_norm_stats,
)
from scripts.openpi_profiles import resolve_profile
from scripts.target_actions import (
    ACTION_SOURCES,
    MEASURED_DELTA,
    PD_TARGET_DELTA,
    URDF_TARGET_ABSOLUTE,
    project_row_actions,
)


# Historical 12-row cube1 comparison population. The canonical gesture index
# shows four semantic actions: gesture02/04/09/10; it is not one cube1_01 action.
DEFAULT_ROW_INDICES = (
    656, 657, 658, 659,
    995, 996, 997, 998,
    1155, 1156,
    1303, 1304,
)
OBJECT_ONLY_LANGUAGE = "object_only"
GESTURE_LANGUAGE = "gesture"
# Historical checkpoints used raw_data_info.id. Keep this choice only for
# explicit reproduction; new runs default to the canonical gesture index.
MOTION_VARIANT_LANGUAGE = "motion_variant"
LANGUAGE_CONDITIONING_CHOICES = (
    GESTURE_LANGUAGE,
    OBJECT_ONLY_LANGUAGE,
    MOTION_VARIANT_LANGUAGE,
)


def motion_variant_identifier(trajectory_metadata: dict[str, Any]) -> str:
    """Build the object-scoped identifier and reject malformed source metadata."""
    if not isinstance(trajectory_metadata, dict):
        raise ValueError("motion-variant language requires trajectory_metadata")
    object_names = trajectory_metadata.get("object_names") or []
    if not object_names or not isinstance(object_names[0], str) or not object_names[0].strip():
        raise ValueError(
            "motion-variant language requires a non-empty string "
            "trajectory_metadata.object_names[0]"
        )
    object_slug = re.sub(r"[^a-z0-9]+", "_", object_names[0].strip().lower()).strip("_")
    if not object_slug:
        raise ValueError("motion-variant language requires a usable object identifier")
    raw_info = trajectory_metadata.get("raw_data_info")
    if not isinstance(raw_info, dict):
        raise ValueError("motion-variant language requires trajectory_metadata.raw_data_info")
    raw_id = raw_info.get("id")
    if isinstance(raw_id, bool) or not isinstance(raw_id, (int, np.integer)) or int(raw_id) < 0:
        raise ValueError(
            "motion-variant language requires a non-negative integer "
            "trajectory_metadata.raw_data_info.id"
        )
    return f"{object_slug}_{int(raw_id):02d}"


def validate_motion_variant_metadata(metadata_rows: Iterable[dict[str, Any]]) -> None:
    """Prevent one language identifier from silently aliasing distinct raw sources."""
    source_by_variant: dict[str, tuple[Any, Any, Any, int]] = {}
    for metadata in metadata_rows:
        variant = motion_variant_identifier(metadata)
        raw_info = metadata["raw_data_info"]
        source = (
            raw_info.get("capMachine"),
            raw_info.get("operator"),
            raw_info.get("scene"),
            int(raw_info["id"]),
        )
        existing = source_by_variant.setdefault(variant, source)
        if existing != source:
            raise ValueError(
                f"motion variant {variant!r} aliases distinct raw sources: "
                f"{existing!r} and {source!r}"
            )


def format_language_prompt(
    base_prompt: str,
    trajectory_metadata: dict[str, Any],
    language_conditioning: str,
    *,
    gesture: str | None = None,
) -> str:
    """Format the canonical gesture prompt or an explicit legacy language."""
    if language_conditioning not in LANGUAGE_CONDITIONING_CHOICES:
        raise ValueError(
            f"unsupported language conditioning {language_conditioning!r}; "
            f"expected one of {LANGUAGE_CONDITIONING_CHOICES}"
        )
    if not isinstance(base_prompt, str) or not base_prompt.strip():
        raise ValueError("language prompt must be a non-empty string")
    if language_conditioning == OBJECT_ONLY_LANGUAGE:
        return base_prompt
    if language_conditioning == GESTURE_LANGUAGE:
        if gesture is None:
            raise ValueError("gesture language requires a canonical gesture label")
        return format_gesture_prompt(base_prompt, gesture)
    return (
        f"{base_prompt.strip()} using motion variant "
        f"{motion_variant_identifier(trajectory_metadata)}"
    )


class SelectedLanceDataset(L.LanceViewpi05Dataset):
    """The base Lance frame-window view restricted to explicit episode rows."""

    def __init__(
        self,
        lance_dataset: Path,
        *,
        row_indices: list[int],
        action_horizon: int,
        frame_window: str = "contact",
        contact_context_frames: int = L.contact_windows_lib.DEFAULT_CONTACT_CONTEXT_FRAMES,
        contact_window_manifest: Path | None = None,
        missing_contact_policy: str = "full",
        action_source: str = MEASURED_DELTA,
        language_conditioning: str = GESTURE_LANGUAGE,
        gesture_index: Path = DEFAULT_GESTURE_INDEX_PATH,
        target_lance_dataset: Path | None = None,
        extended_state: bool = False,
    ) -> None:
        if action_source not in ACTION_SOURCES:
            raise ValueError(f"unsupported action_source {action_source!r}; expected one of {ACTION_SOURCES}")
        if action_source != MEASURED_DELTA and target_lance_dataset is None:
            raise ValueError(
                f"{action_source} requires --target-lance-dataset"
            )
        if language_conditioning not in LANGUAGE_CONDITIONING_CHOICES:
            raise ValueError(
                f"unsupported language conditioning {language_conditioning!r}; "
                f"expected one of {LANGUAGE_CONDITIONING_CHOICES}"
            )
        self._action_source = action_source
        self._extended_state = bool(extended_state)
        self._language_conditioning = language_conditioning
        self._gesture_index = (
            GestureIndex.load(gesture_index)
            if language_conditioning == GESTURE_LANGUAGE
            else None
        )
        self._gesture_index_path = (
            self._gesture_index.path if self._gesture_index is not None else None
        )
        self._gesture_index_sha256 = (
            self._gesture_index.sha256 if self._gesture_index is not None else None
        )
        self._target_dataset_path = target_lance_dataset
        self._target_dataset = (
            lance.dataset(str(target_lance_dataset)) if target_lance_dataset is not None else None
        )
        self._dataset = lance.dataset(str(lance_dataset))
        self._dataset_path = Path(lance_dataset)
        all_rows = self._dataset.to_table(
            columns=["episode_metadata", "trajectory_metadata", "index"]
        ).to_pylist()
        if not row_indices:
            raise ValueError("row_indices must not be empty")
        if min(row_indices) < 0 or max(row_indices) >= len(all_rows):
            raise IndexError(f"row index outside dataset: {row_indices!r}, rows={len(all_rows)}")
        self._source_row_indices = [int(x) for x in row_indices]
        self._rows = [all_rows[i] for i in self._source_row_indices]
        self._gesture_records: dict[int, Any] = {}
        if self._gesture_index is not None:
            if len(self._gesture_index) != len(all_rows):
                raise ValueError(
                    "gesture/Lance row-count mismatch: "
                    f"{len(self._gesture_index)} != {len(all_rows)}"
                )
            for local_row, (source_row, row) in enumerate(
                zip(self._source_row_indices, self._rows, strict=True)
            ):
                metadata = row["trajectory_metadata"]
                object_names = metadata.get("object_names") or []
                if len(object_names) != 1 or not isinstance(object_names[0], str):
                    raise ValueError(
                        f"gesture language requires exactly one object at row {source_row}"
                    )
                index = row["index"]
                self._gesture_records[local_row] = self._gesture_index.record_for(
                    source_row,
                    uuid=index["uuid"],
                    seed_uuid=index["seed_uuid"],
                    object_type=object_names[0],
                    total_frames=int(row["episode_metadata"]["total_frames"]),
                )
        elif self._language_conditioning == MOTION_VARIANT_LANGUAGE:
            validate_motion_variant_metadata(
                row["trajectory_metadata"] for row in self._rows
            )
        if self._target_dataset is not None:
            if self._target_dataset.count_rows() != len(all_rows):
                raise ValueError(
                    "target/image dataset row-count mismatch: "
                    f"{self._target_dataset.count_rows()} != {len(all_rows)}"
                )
            image_ids = self._dataset.take(
                self._source_row_indices, columns=["index"]
            ).to_pylist()
            target_ids = self._target_dataset.take(
                self._source_row_indices, columns=["index"]
            ).to_pylist()
            mismatched = [
                source_row
                for source_row, image_id, target_id in zip(
                    self._source_row_indices, image_ids, target_ids, strict=True
                )
                if image_id["index"]["uuid"] != target_id["index"]["uuid"]
            ]
            if mismatched:
                raise ValueError(f"target/image row UUID mismatch: {mismatched[:8]}")
        self._action_horizon = int(action_horizon)
        self._frame_window = str(frame_window)
        self._contact_context_frames = int(contact_context_frames)
        self._contact_window_manifest = (
            Path(contact_window_manifest)
            if contact_window_manifest is not None
            else L.contact_windows_lib.default_manifest_path(lance_dataset)
        )
        self._missing_contact_policy = str(missing_contact_policy)
        manifest_entries: dict[int, dict[str, Any]] = {}
        if self._frame_window == "contact":
            manifest_entries = L.contact_windows_lib.load_or_build_windows(
                self._dataset,
                lance_dataset,
                self._source_row_indices,
                manifest_path=self._contact_window_manifest,
                context_frames=self._contact_context_frames,
                missing_policy=self._missing_contact_policy,
            )
        elif self._frame_window != "full":
            raise ValueError(f"frame_window must be contact or full, got {self._frame_window!r}")

        self._row_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._row_cache_lock = threading.RLock()
        self._index: list[tuple[int, int]] = []
        self._row_start_offset: dict[int, int] = {}
        self._row_window_start: dict[int, int] = {}
        self._row_windows: dict[int, L.contact_windows_lib.ContactWindow] = {}
        for local_row, row in enumerate(self._rows):
            source_row = self._source_row_indices[local_row]
            total_frames = int(row["episode_metadata"]["total_frames"])
            window = L.contact_windows_lib.select_window(
                row,
                row_index=source_row,
                total_frames=total_frames,
                mode=self._frame_window,
                manifest_entry=manifest_entries.get(source_row),
                context_frames=self._contact_context_frames,
                missing_policy=self._missing_contact_policy,
            )
            if window is None or window.frame_count <= 0:
                continue
            self._row_windows[local_row] = window
            self._row_start_offset[local_row] = len(self._index)
            self._row_window_start[local_row] = window.start_frame
            self._index.extend(
                (local_row, frame)
                for frame in range(window.start_frame, window.end_frame + 1)
            )
        if not self._index:
            raise ValueError(f"No samples available in selected rows from {lance_dataset}")
        self._slate_size = min(16, len(self._row_start_offset))
        self._slate_rotate_every = 250
        self._slate_row_indices: list[int] = []
        self._slate_calls_since_rotate = 0

    def _get_row(self, row_index: int) -> dict[str, Any]:
        # Lance rows are immutable after loading. Protect the small row-level LRU
        # so batch datum workers can safely decode different frames in parallel.
        # Holding the lock during a miss also prevents duplicate full-row reads;
        # misses occur only when the coverage slate rotates.
        with self._row_cache_lock:
            cached = self._row_cache.get(row_index)
            if cached is not None:
                self._row_cache.move_to_end(row_index)
                return cached
            source_row = self._source_row_indices[row_index]
            columns = ["state", "actions", "prompt", "image", "wrist_image"]
            if self._extended_state:
                columns += ["contact", "objects"]
            row = self._dataset.take([source_row], columns=columns).to_pylist()[0]
            if self._action_source != MEASURED_DELTA:
                assert self._target_dataset is not None
                target_row = self._target_dataset.take(
                    [source_row], columns=["hands"]
                ).to_pylist()[0]
                target_q = np.asarray(target_row["hands"][0]["urdf_dof"], dtype=np.float32)
                image_q = np.asarray(row["state"], dtype=np.float32)[:, :26]
                if target_q.shape != image_q.shape or not np.array_equal(target_q, image_q):
                    raise ValueError(
                        f"target/image q mismatch at source row {source_row}: "
                        f"target={target_q.shape} image={image_q.shape}"
                    )
                row = {**row, "hands": target_row["hands"]}
            row = project_row_actions(row, self._action_source)
            row = {
                **row,
                "prompt": format_language_prompt(
                    row["prompt"],
                    self._rows[row_index]["trajectory_metadata"],
                    self._language_conditioning,
                    gesture=(
                        self._gesture_records[row_index].gesture
                        if self._language_conditioning == GESTURE_LANGUAGE
                        else None
                    ),
                ),
            }
            self._row_cache[row_index] = row
            self._row_cache.move_to_end(row_index)
            while len(self._row_cache) > max(4, self._slate_size):
                self._row_cache.popitem(last=False)
            return row


def selected_norm_stats(dataset: SelectedLanceDataset) -> dict[str, Any]:
    """Compute normalization from the selected rows' active frame windows."""
    if dataset._action_source == MEASURED_DELTA:
        return L._compute_norm_stats(dataset)

    assert dataset._target_dataset is not None
    frames_by_row: dict[int, tuple[int, int]] = {}
    for local_row, frame in dataset._index:
        previous = frames_by_row.get(local_row)
        frames_by_row[local_row] = (
            (frame, frame)
            if previous is None
            else (min(previous[0], frame), max(previous[1], frame))
        )
    state_stats = L.normalize.RunningStats()
    action_stats = L.normalize.RunningStats()
    for local_row in sorted(frames_by_row):
        source_row = dataset._source_row_indices[local_row]
        image_row = dataset._dataset.take(
            [source_row], columns=["state", "actions"]
        ).to_pylist()[0]
        target_row = dataset._target_dataset.take([source_row], columns=["hands"]).to_pylist()[0]
        target_q = np.asarray(target_row["hands"][0]["urdf_dof"], dtype=np.float32)
        image_q = np.asarray(image_row["state"], dtype=np.float32)[:, :26]
        if target_q.shape != image_q.shape or not np.array_equal(target_q, image_q):
            raise ValueError(f"target/image q mismatch while normalizing source row {source_row}")
        projected = project_row_actions(
            {**image_row, "hands": target_row["hands"]}, dataset._action_source
        )
        start, end = frames_by_row[local_row]
        states = np.asarray(image_row["state"][start : end + 1], dtype=np.float32)
        if dataset._action_source == URDF_TARGET_ABSOLUTE:
            # Match the exact model supervision population. Dataset item t emits
            # target[t:t+H], repeat-padded at the selected window end, and
            # DeltaActions anchors every horizon element to the same query q[t].
            query_frames = np.arange(start, end + 1, dtype=np.int64)
            horizon_offsets = np.arange(dataset._action_horizon, dtype=np.int64)
            target_frames = np.minimum(
                query_frames[:, None] + horizon_offsets[None, :], end
            )
            absolute_targets = np.asarray(projected["actions"], dtype=np.float32)
            actions = absolute_targets[target_frames].copy()
            actions[:, :, :3] -= states[:, None, :3]
            actions[:, :, 6:26] -= states[:, None, 6:26]
            action_values = actions.reshape(-1, actions.shape[-1])
        else:
            action_values = np.asarray(
                projected["actions"][start : end + 1], dtype=np.float32
            )
        state_stats.update(states)
        action_stats.update(action_values)
    state_result = state_stats.get_statistics()
    if getattr(dataset, "_extended_state", False):
        # Extended state: contact dims [26:31] use fixed q01=0/q99=1 so the
        # standard Normalize maps 0→-1, 1→+1 (not quantile-dependent).
        # Lift height [31] uses quantile stats from the actual distribution.
        _m = np.asarray(state_result.mean, dtype=np.float32).copy()
        _s = np.asarray(state_result.std, dtype=np.float32).copy()
        _q01 = np.asarray(state_result.q01, dtype=np.float32).copy()
        _q99 = np.asarray(state_result.q99, dtype=np.float32).copy()
        _m[26:31] = 0.5; _s[26:31] = 0.5
        _q01[26:31] = 0.0; _q99[26:31] = 1.0
        # Compute lift height stats from objects data
        lift_values = []
        for local_row in sorted(frames_by_row):
            source_row = dataset._source_row_indices[local_row]
            obj_row = dataset._dataset.take([source_row], columns=["objects"]).to_pylist()[0]
            obj_pos = np.asarray(obj_row["objects"][0]["pos"], dtype=np.float64)
            start, end = frames_by_row[local_row]
            lift = obj_pos[start:end+1, 2] - obj_pos[0, 2]
            lift_values.append(lift)
        lift_all = np.concatenate(lift_values)
        lift_stats = L.normalize.RunningStats()
        lift_stats.update(lift_all.reshape(-1, 1))
        lift_result = lift_stats.get_statistics()
        _m[31] = float(np.asarray(lift_result.mean).flat[0])
        _s[31] = float(np.asarray(lift_result.std).flat[0])
        _q01[31] = float(np.asarray(lift_result.q01).flat[0])
        _q99[31] = float(np.asarray(lift_result.q99).flat[0])
        state_result = L.normalize.NormStats(mean=_m, std=_s, q01=_q01, q99=_q99)
    return {
        "state": state_result,
        "actions": action_stats.get_statistics(),
    }


def load_or_compute_norm_stats(
    dataset: SelectedLanceDataset, norm_stats_dir: Path | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    extended_state = bool(getattr(dataset, "_extended_state", False))
    if norm_stats_dir is None:
        if extended_state:
            raise ValueError(
                "v1 extended-state requires --norm-stats-dir; "
                "computed fallback is not allowed"
            )
        return selected_norm_stats(dataset), {
            "source": "computed",
            "directory": None,
            "sha256": None,
        }
    path = norm_stats_dir / "norm_stats.json"
    if extended_state:
        path, actual_sha = verify_locked_norm_stats(norm_stats_dir)
    else:
        if not path.is_file():
            raise ValueError(f"normalization cache is missing norm_stats.json: {path}")
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    stats = L.normalize.load(norm_stats_dir)
    for key in ("state", "actions"):
        if key not in stats:
            raise ValueError(f"normalization cache is missing {key!r}: {path}")
        if tuple(np.asarray(stats[key].mean).shape) != (32,):
            raise ValueError(
                f"normalization cache {key!r} mean must have shape (32,), "
                f"got {np.asarray(stats[key].mean).shape}"
            )
        if stats[key].q01 is None or stats[key].q99 is None:
            raise ValueError(f"normalization cache {key!r} is missing quantiles: {path}")
    if extended_state:
        # Structural validation after the exact v1 norm bytes are authenticated.
        state_q01 = np.asarray(stats["state"].q01, dtype=np.float32)
        state_q99 = np.asarray(stats["state"].q99, dtype=np.float32)
        if not np.allclose(state_q01[26:31], 0.0):
            raise ValueError(
                f"extended-state norm cache q01[26:31] must be 0, got {state_q01[26:31]}: {path}"
            )
        if not np.allclose(state_q99[26:31], 1.0):
            raise ValueError(
                f"extended-state norm cache q99[26:31] must be 1, got {state_q99[26:31]}: {path}"
            )
        lift_range = state_q99[31] - state_q01[31]
        if lift_range < 1e-4:
            raise ValueError(
                f"extended-state norm cache lift range must be > 1e-4, got {lift_range}: {path}"
            )
    return stats, {
        "source": "loaded",
        "directory": str(norm_stats_dir.resolve()),
        "sha256": actual_sha,
    }


def make_rngs(
    sample_seed: int, augmentation_seed: int | None = None
) -> tuple[np.random.Generator, np.random.Generator, int]:
    """Create independent streams for frame selection and state perturbations.

    The derived augmentation seed is stable and distinct from the sample seed,
    so toggling state noise cannot advance the stream that selects frames.
    """
    resolved_augmentation_seed = sample_seed + 1 if augmentation_seed is None else augmentation_seed
    return (
        np.random.default_rng(sample_seed),
        np.random.default_rng(resolved_augmentation_seed),
        resolved_augmentation_seed,
    )


def advance_coverage_rngs(
    sampler: "CoverageSampler",
    augmentation_rng: np.random.Generator,
    *,
    completed_steps: int,
    batch_size: int,
    action_horizon: int,
    state_noise_std: float,
    target_noise_std: float,
) -> None:
    """Advance deterministic data/noise streams without materializing prior batches."""
    for _ in range(completed_steps):
        sampler.sample_indices(batch_size)
        for _ in range(batch_size):
            if state_noise_std > 0:
                augmentation_rng.normal(0.0, state_noise_std, size=(32,))
            if target_noise_std > 0:
                augmentation_rng.normal(
                    0.0, target_noise_std, size=(action_horizon, 32)
                )


class DatumCache:
    """Run-owned bounded LRU cache safe for concurrent datum builders."""

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("datum cache size must be non-negative")
        self.capacity = capacity
        self._items: OrderedDict[int, Any] = OrderedDict()
        self._inflight: dict[int, Future[Any]] = {}
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get_or_create(self, key: int, create: Callable[[], Any]) -> Any:
        if self.capacity == 0:
            with self._lock:
                self.misses += 1
            return create()

        with self._lock:
            datum = self._items.get(key)
            if datum is not None:
                self.hits += 1
                self._items.move_to_end(key)
                return datum
            pending = self._inflight.get(key)
            if pending is None:
                pending = Future()
                self._inflight[key] = pending
                self.misses += 1
                creator = True
            else:
                # Under serial execution this lookup would occur after the first
                # creation and count as a hit. Preserve that accounting while
                # coalescing duplicate work within a parallel batch.
                self.hits += 1
                creator = False

        if not creator:
            datum = pending.result()
            with self._lock:
                if self._items.get(key) is datum:
                    self._items.move_to_end(key)
            return datum

        try:
            datum = create()
        except BaseException as error:
            with self._lock:
                self._inflight.pop(key, None)
                pending.set_exception(error)
            raise

        with self._lock:
            self._items[key] = datum
            self._items.move_to_end(key)
            if len(self._items) > self.capacity:
                self._items.popitem(last=False)
                self.evictions += 1
            self._inflight.pop(key, None)
            pending.set_result(datum)
        return datum

    def summary(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "current_size": len(self._items),
                "capacity": self.capacity,
            }


class AugmentationDiagnostics:
    def __init__(self) -> None:
        self.samples = 0
        self.token_changed_samples = 0
        self.valid_coordinates = 0
        self.changed_bins = 0
        self.clean_out_of_range_coordinates = 0
        self.augmented_out_of_range_coordinates = 0
        self.realized_noise_squares = 0.0
        self.valid_by_dimension = np.zeros(32, dtype=np.int64)
        self.changed_bins_by_dimension = np.zeros(32, dtype=np.int64)
        self.clean_out_of_range_by_dimension = np.zeros(32, dtype=np.int64)
        self.augmented_out_of_range_by_dimension = np.zeros(32, dtype=np.int64)
        self.clean_token_lengths: list[int] = []
        self.augmented_token_lengths: list[int] = []

    def record(
        self,
        clean_state: np.ndarray,
        augmented_state: np.ndarray,
        valid: np.ndarray,
        clean_tokens: list[int],
        augmented_tokens: list[int],
    ) -> None:
        self.samples += 1
        self.token_changed_samples += clean_tokens != augmented_tokens
        bins = np.linspace(-1, 1, 257)[:-1]
        clean_bins = np.digitize(clean_state, bins) - 1
        augmented_bins = np.digitize(augmented_state, bins) - 1
        changed = (clean_bins != augmented_bins) & valid
        clean_out_of_range = ((clean_state < -1.0) | (clean_state > 1.0)) & valid
        augmented_out_of_range = ((augmented_state < -1.0) | (augmented_state > 1.0)) & valid
        self.valid_coordinates += int(valid.sum())
        self.changed_bins += int(np.count_nonzero(changed))
        self.clean_out_of_range_coordinates += int(np.count_nonzero(clean_out_of_range))
        self.augmented_out_of_range_coordinates += int(np.count_nonzero(augmented_out_of_range))
        self.valid_by_dimension += valid.astype(np.int64)
        self.changed_bins_by_dimension += changed.astype(np.int64)
        self.clean_out_of_range_by_dimension += clean_out_of_range.astype(np.int64)
        self.augmented_out_of_range_by_dimension += augmented_out_of_range.astype(np.int64)
        self.clean_token_lengths.append(len(clean_tokens))
        self.augmented_token_lengths.append(len(augmented_tokens))
        delta = augmented_state - clean_state
        self.realized_noise_squares += float(np.square(delta[valid]).sum())

    def summary(self, requested_sigma: float, *, token_budget: int = 200) -> dict[str, Any]:
        valid = max(1, self.valid_coordinates)
        valid_dims = self.valid_by_dimension > 0
        bin_rates = np.divide(
            self.changed_bins_by_dimension,
            self.valid_by_dimension,
            out=np.zeros(32, dtype=np.float64),
            where=valid_dims,
        )
        clean_out_of_range_rates = np.divide(
            self.clean_out_of_range_by_dimension,
            self.valid_by_dimension,
            out=np.zeros(32, dtype=np.float64),
            where=valid_dims,
        )
        augmented_out_of_range_rates = np.divide(
            self.augmented_out_of_range_by_dimension,
            self.valid_by_dimension,
            out=np.zeros(32, dtype=np.float64),
            where=valid_dims,
        )
        active_bin_rates = bin_rates[valid_dims]
        lengths = self.augmented_token_lengths
        return {
            "samples": self.samples,
            "valid_coordinates": self.valid_coordinates,
            "token_changed_fraction": self.token_changed_samples / max(1, self.samples),
            "valid_coordinate_bin_changed_fraction": self.changed_bins / valid,
            "median_valid_dimension_bin_changed_fraction": (
                float(np.median(active_bin_rates)) if active_bin_rates.size else 0.0
            ),
            "clean_out_of_range_fraction_valid_coordinates": (
                self.clean_out_of_range_coordinates / valid
            ),
            "augmented_out_of_range_fraction_valid_coordinates": (
                self.augmented_out_of_range_coordinates / valid
            ),
            "max_valid_dimension_augmented_out_of_range_fraction": (
                float(augmented_out_of_range_rates[valid_dims].max())
                if np.any(valid_dims)
                else 0.0
            ),
            "requested_sigma": requested_sigma,
            "realized_sigma": math.sqrt(self.realized_noise_squares / valid),
            "clean_token_length_range": (
                [min(self.clean_token_lengths), max(self.clean_token_lengths)]
                if self.clean_token_lengths
                else [0, 0]
            ),
            "augmented_token_length_range": [min(lengths), max(lengths)] if lengths else [0, 0],
            "token_budget_reached_fraction": (
                sum(length >= token_budget for length in lengths) / max(1, len(lengths))
            ),
            "bin_changed_fraction_by_dimension": bin_rates.tolist(),
            "clean_out_of_range_fraction_by_dimension": clean_out_of_range_rates.tolist(),
            "augmented_out_of_range_fraction_by_dimension": augmented_out_of_range_rates.tolist(),
        }


class TargetAugmentationDiagnostics:
    """Population diagnostics for normalized PD-target supervision noise."""

    def __init__(self) -> None:
        self.samples = 0
        self.valid_coordinates = 0
        self.changed_coordinates = 0
        self.clean_out_of_range_coordinates = 0
        self.augmented_out_of_range_coordinates = 0
        self.realized_noise_squares = 0.0

    def record(
        self,
        clean_actions: np.ndarray,
        augmented_actions: np.ndarray,
        valid_dimensions: np.ndarray,
    ) -> None:
        clean = np.asarray(clean_actions, dtype=np.float32)
        augmented = np.asarray(augmented_actions, dtype=np.float32)
        if clean.shape != augmented.shape or clean.ndim != 2:
            raise ValueError(
                f"target augmentation diagnostics require matching [T,D] actions, "
                f"got {clean.shape} and {augmented.shape}"
            )
        valid_dimensions = np.asarray(valid_dimensions, dtype=bool)
        if valid_dimensions.shape != (clean.shape[1],):
            raise ValueError(
                f"target valid-dimension mask must have shape {(clean.shape[1],)}, "
                f"got {valid_dimensions.shape}"
            )
        valid = np.broadcast_to(valid_dimensions, clean.shape)
        delta = augmented - clean
        self.samples += 1
        self.valid_coordinates += int(valid.sum())
        self.changed_coordinates += int((np.abs(delta[valid]) > 0).sum())
        self.clean_out_of_range_coordinates += int((np.abs(clean[valid]) > 1.0).sum())
        self.augmented_out_of_range_coordinates += int((np.abs(augmented[valid]) > 1.0).sum())
        self.realized_noise_squares += float(np.square(delta[valid]).sum())

    def summary(self, requested_sigma: float) -> dict[str, Any]:
        valid = max(1, self.valid_coordinates)
        return {
            "samples": self.samples,
            "valid_coordinates": self.valid_coordinates,
            "changed_fraction": self.changed_coordinates / valid,
            "clean_out_of_range_fraction": self.clean_out_of_range_coordinates / valid,
            "augmented_out_of_range_fraction": (
                self.augmented_out_of_range_coordinates / valid
            ),
            "requested_sigma": requested_sigma,
            "realized_sigma": math.sqrt(self.realized_noise_squares / valid),
        }


class PreparedDatum:
    def __init__(self, prefix: dict[str, Any], token_transform: Any, suffix_transforms: tuple[Any, ...], clean_datum: dict[str, Any]) -> None:
        self.prefix = prefix
        self.token_transform = token_transform
        self.suffix_transforms = suffix_transforms
        self.clean_datum = clean_datum


def _split_model_transforms(data_config: Any) -> tuple[tuple[Any, ...], Any, tuple[Any, ...]]:
    transforms = tuple(data_config.model_transforms.inputs)
    for index, transform in enumerate(transforms):
        if type(transform).__name__ == "TokenizePrompt":
            return transforms[:index], transform, transforms[index + 1 :]
    raise ValueError("discrete-state augmentation requires a TokenizePrompt transform")


def _prepare_discrete_datum(sample: dict[str, Any], data_config: Any, base_model: str) -> PreparedDatum:
    """Cache the deterministic normalized prefix, never a noisy token sequence."""
    before_token, token_transform, suffix = _split_model_transforms(data_config)
    data = sample
    for transform in (*data_config.data_transforms.inputs, L.transforms.Normalize(
        data_config.norm_stats, use_quantiles=data_config.use_quantile_norm
    ), *before_token):
        data = transform(data)
    prefix = data
    clean = _tokenize_prepared(prefix, token_transform, suffix, np.asarray(prefix["state"], dtype=np.float32))
    return PreparedDatum(prefix, token_transform, suffix, L._pi05_datum_from_transformed(base_model, clean))


def _tokenize_prepared(prefix: dict[str, Any], token_transform: Any, suffix: tuple[Any, ...], state: np.ndarray) -> dict[str, Any]:
    data = dict(prefix)
    data["state"] = np.asarray(state, dtype=np.float32)
    data = token_transform(data)
    for transform in suffix:
        data = transform(data)
    return data


def _prompt_tokens(item: dict[str, Any]) -> list[int]:
    mask = np.asarray(item["tokenized_prompt_mask"], dtype=bool)
    return np.asarray(item["tokenized_prompt"], dtype=np.int32)[mask].tolist()


def _replace_wire_state_and_tokens(clean: dict[str, Any], state: np.ndarray, tokens: list[int]) -> dict[str, Any]:
    """Reuse cached images/actions; copy only the observation/text/state path."""
    result = clean.copy()
    observation = clean["observation"].copy()
    observation["state"] = {**clean["observation"]["state"], "data": state.astype(np.float32).tolist()}
    model_input = clean["observation"]["model_input"].copy()
    chunks = list(model_input["chunks"])
    text_index = next(index for index, chunk in enumerate(chunks) if chunk["type"] == "encoded_text")
    chunks[text_index] = {"type": "encoded_text", "tokens": tokens}
    model_input["chunks"] = chunks
    observation["model_input"] = model_input
    result["observation"] = observation
    return result


def _replace_wire_actions(clean: dict[str, Any], actions: np.ndarray) -> dict[str, Any]:
    """Reuse cached observation/text and replace only normalized supervision."""
    result = clean.copy()
    supervision = clean["supervision"].copy()
    supervision["actions"] = {
        **clean["supervision"]["actions"],
        "data": np.asarray(actions, dtype=np.float32).tolist(),
    }
    result["supervision"] = supervision
    return result


def _quantile_valid_dimensions(
    norm_stats: dict[str, Any], key: str, width: int
) -> np.ndarray:
    stats = norm_stats[key]
    q01 = getattr(stats, "q01", None)
    q99 = getattr(stats, "q99", None)
    if q01 is None or q99 is None:
        raise ValueError(f"{key} augmentation requires quantile normalization statistics")
    lo = np.asarray(q01, dtype=np.float32)[:width]
    hi = np.asarray(q99, dtype=np.float32)[:width]
    if lo.shape != (width,) or hi.shape != (width,):
        raise ValueError(
            f"{key} quantile statistics must cover {width} dimensions, "
            f"got {lo.shape} and {hi.shape}"
        )
    return np.isfinite(lo) & np.isfinite(hi) & ((hi - lo) > 1e-6)


def build_batch(
    dataset: SelectedLanceDataset,
    data_config: Any,
    *,
    base_model: str,
    indices: list[int],
    norm_stats: dict[str, Any],
    state_noise_std: float,
    rng: np.random.Generator,
    target_noise_std: float = 0.0,
    datum_cache: DatumCache | None = None,
    augmentation_diagnostics: AugmentationDiagnostics | None = None,
    target_augmentation_diagnostics: TargetAugmentationDiagnostics | None = None,
    executor: ThreadPoolExecutor | None = None,
) -> list[dict[str, Any]]:
    """Build a batch in order with deterministic state or PD-target augmentation.

    Sampling and augmentation RNG stay on the producer thread. Worker completion
    order therefore cannot change frame selection, noise values, or batch order.
    """
    profile = resolve_profile(base_model)
    cache = datum_cache if datum_cache is not None else DatumCache(0)
    keys = [int(index % len(dataset)) for index in indices]

    def map_ordered(function: Callable[[Any], Any], values: list[Any]) -> list[Any]:
        if executor is None:
            return [function(value) for value in values]
        return list(executor.map(function, values))

    if state_noise_std == 0 and target_noise_std == 0:
        # Preserve the historical clean transform/lowering path and do not touch rng.
        def build_clean(key: int) -> dict[str, Any]:
            return cache.get_or_create(
                key, lambda: L._pi05_datum_from_transformed(
                    base_model, L._transform_sample(dataset[key], data_config)
                )
            )

        return map_ordered(build_clean, keys)

    if state_noise_std > 0 and not profile.discrete_state_input:
        raise ValueError("state augmentation requires a profile with discrete state computation")
    if target_noise_std > 0 and dataset._action_source != PD_TARGET_DELTA:
        raise ValueError("target augmentation requires action_source=pd_target_delta")

    # Generate all noise before dispatch so thread scheduling cannot reorder RNG.
    requests: list[tuple[int, np.ndarray | None, np.ndarray | None]] = []
    for key in keys:
        state_noise = (
            rng.normal(0.0, state_noise_std, size=32).astype(np.float32)
            if state_noise_std > 0
            else None
        )
        target_noise = (
            rng.normal(
                0.0,
                target_noise_std,
                size=(dataset._action_horizon, 32),
            ).astype(np.float32)
            if target_noise_std > 0
            else None
        )
        requests.append((key, state_noise, target_noise))

    def build_augmented(
        request: tuple[int, np.ndarray | None, np.ndarray | None]
    ) -> tuple[dict[str, Any], tuple[Any, ...] | None, tuple[Any, ...] | None]:
        key, state_noise, target_noise = request
        prepared = cache.get_or_create(
            key, lambda: _prepare_discrete_datum(dataset[key], data_config, base_model)
        )
        if not isinstance(prepared, PreparedDatum):
            raise TypeError("augmented datum cache contains an incompatible clean datum")
        datum = prepared.clean_datum
        state_diagnostic: tuple[Any, ...] | None = None
        target_diagnostic: tuple[Any, ...] | None = None

        if state_noise is not None:
            clean_state = np.asarray(prepared.prefix["state"], dtype=np.float32)
            if clean_state.shape != (32,):
                raise ValueError(
                    "discrete-state augmentation requires normalized state shape (32,), "
                    f"got {clean_state.shape}"
                )
            valid_state = _quantile_valid_dimensions(norm_stats, "state", 32)
            if getattr(dataset, "_extended_state", False):
                # Extended state: only noise hand qpos [0:26]; do NOT pollute
                # finger contacts [26:31] or lift height [31].
                valid_state[26:] = False
            augmented_state = clean_state.copy()
            augmented_state[valid_state] = (
                clean_state[valid_state] + state_noise[valid_state]
            ).astype(np.float32)
            transformed = _tokenize_prepared(
                prepared.prefix,
                prepared.token_transform,
                prepared.suffix_transforms,
                augmented_state,
            )
            augmented_tokens = _prompt_tokens(transformed)
            clean_tokens = next(
                chunk["tokens"]
                for chunk in prepared.clean_datum["observation"]["model_input"]["chunks"]
                if chunk["type"] == "encoded_text"
            )
            datum = _replace_wire_state_and_tokens(
                datum, augmented_state, augmented_tokens
            )
            # B scheme: the action residual was computed with the CLEAN state
            # (DeltaActions: target - q_clean). After augmenting the query state,
            # the residual must be recomputed for q_noisy so that
            # q_noisy + residual_noisy == absolute_target.
            # State and action have different q01/q99 scales, so the normalized
            # noise cannot be subtracted directly. Under quantile normalization:
            #   δa_norm = -δq_norm * (state_range / action_range)
            # Only delta-eligible dims (xyz[0:3], fingers[6:26]) are adjusted;
            # euler[3:6] and padding[26:32] are state-independent.
            if dataset._action_source == URDF_TARGET_ABSOLUTE:
                actions_meta = prepared.clean_datum["supervision"]["actions"]
                clean_acts = np.asarray(
                    actions_meta["data"], dtype=np.float32
                ).reshape(actions_meta["shape"])
                # Per-dim scale ratio between state and action quantile ranges.
                _sq01 = np.asarray(norm_stats["state"].q01, dtype=np.float32)[:32]
                _sq99 = np.asarray(norm_stats["state"].q99, dtype=np.float32)[:32]
                _aq01 = np.asarray(norm_stats["actions"].q01, dtype=np.float32)[:32]
                _aq99 = np.asarray(norm_stats["actions"].q99, dtype=np.float32)[:32]
                _s_range = _sq99 - _sq01
                _a_range = _aq99 - _aq01
                _valid = (_s_range > 1e-6) & (_a_range > 1e-6)
                _scale = np.where(_valid, _s_range / np.maximum(_a_range, 1e-8), 0.0)
                augmented_acts = clean_acts.copy()
                augmented_acts[:, :3] -= state_noise[:3] * _scale[:3]
                augmented_acts[:, 6:26] -= state_noise[6:26] * _scale[6:26]
                # wire format stores actions as reshape(-1).tolist()
                datum = _replace_wire_actions(datum, augmented_acts.reshape(-1))
            state_diagnostic = (
                clean_state,
                augmented_state,
                valid_state,
                clean_tokens,
                augmented_tokens,
            )

        if target_noise is not None:
            clean_actions = np.asarray(prepared.prefix["actions"], dtype=np.float32)
            expected_shape = (dataset._action_horizon, 32)
            if clean_actions.shape != expected_shape:
                raise ValueError(
                    "target augmentation requires normalized actions shape "
                    f"{expected_shape}, got {clean_actions.shape}"
                )
            valid_actions = _quantile_valid_dimensions(norm_stats, "actions", 32)
            augmented_actions = clean_actions.copy()
            augmented_actions[:, valid_actions] = (
                clean_actions[:, valid_actions] + target_noise[:, valid_actions]
            ).astype(np.float32)
            datum = _replace_wire_actions(datum, augmented_actions)
            target_diagnostic = (clean_actions, augmented_actions, valid_actions)

        return datum, state_diagnostic, target_diagnostic

    built = map_ordered(build_augmented, requests)
    if augmentation_diagnostics is not None:
        for _, diagnostic, _ in built:
            if diagnostic is not None:
                augmentation_diagnostics.record(*diagnostic)
    if target_augmentation_diagnostics is not None:
        for _, _, diagnostic in built:
            if diagnostic is not None:
                target_augmentation_diagnostics.record(*diagnostic)
    return [datum for datum, _, _ in built]


class BatchPrefetcher:
    """Ordered, single-producer batch builder with cancellable bounded output."""

    def __init__(
        self,
        prefetch_batches: int,
        build_next: Callable[[], list[dict[str, Any]]],
        *,
        max_batches: int | None = None,
    ) -> None:
        if prefetch_batches <= 0:
            raise ValueError("prefetch_batches must be positive")
        if max_batches is not None and max_batches < 0:
            raise ValueError("max_batches must be non-negative")
        self._queue: queue.Queue[list[dict[str, Any]]] = queue.Queue(maxsize=prefetch_batches)
        self._slots = threading.Semaphore(prefetch_batches)
        self._build_next = build_next
        self._max_batches = max_batches
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._closed = False
        self._close_warning_emitted = False
        self.build_seconds = 0.0
        self.batches_built = 0
        self._thread = threading.Thread(target=self._produce, name="batch-prefetch", daemon=True)
        self._thread.start()

    def _produce(self) -> None:
        try:
            while not self._stop.is_set() and (
                self._max_batches is None or self.batches_built < self._max_batches
            ):
                # Reserve capacity before building so this includes a batch
                # currently being constructed, not only queued items.
                if not self._slots.acquire(timeout=0.1):
                    continue
                if self._stop.is_set():
                    self._slots.release()
                    break
                started_at = time.perf_counter()
                batch = self._build_next()
                self.build_seconds += time.perf_counter() - started_at
                self.batches_built += 1
                self._queue.put(batch)
        except BaseException as error:
            self._error = error

    def next_batch(self) -> list[dict[str, Any]]:
        while True:
            if self._error is not None:
                raise self._error
            try:
                batch = self._queue.get(timeout=0.1)
                self._slots.release()
                return batch
            except queue.Empty:
                if not self._thread.is_alive():
                    if self._error is not None:
                        raise self._error
                    raise RuntimeError("batch prefetch producer stopped unexpectedly")

    def close(self) -> None:
        """Request producer shutdown without masking training/model-cleanup errors."""
        if self._closed and not self._thread.is_alive():
            return
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._closed = not self._thread.is_alive()
        if not self._closed and not self._close_warning_emitted:
            print(
                "warning: batch prefetch producer did not stop within 5 seconds; "
                "continuing cleanup because the producer thread is daemonized",
                file=sys.stderr,
                flush=True,
            )
            self._close_warning_emitted = True


class CoverageSampler:
    """Seeded row coverage grouped into cache-friendly, balanced slates."""

    def __init__(
        self,
        dataset: SelectedLanceDataset,
        rng: np.random.Generator,
        *,
        slate_size: int,
        anchors_per_row: int,
    ) -> None:
        if slate_size <= 0 or anchors_per_row <= 0:
            raise ValueError("coverage slate size and anchors per row must be positive")
        self.dataset, self.rng = dataset, rng
        self.slate_size, self.anchors_per_row = slate_size, anchors_per_row
        self.epoch = 0
        self._rows: list[int] = []
        self._position = 0
        self._slate: list[int] = []
        self._round = 0
        self._row_cursor = 0
        self.current_epoch_visited_rows: set[int] = set()
        self.all_visited_rows: set[int] = set()
        self.anchor_counts: dict[int, int] = {}
        self.schedule_digest = hashlib.sha256()

    def _new_epoch(self) -> None:
        self._rows = [int(row) for row in self.dataset._row_start_offset]
        self.rng.shuffle(self._rows)
        self._position = 0
        self._slate = []
        self._round = 0
        self._row_cursor = 0
        self.current_epoch_visited_rows = set()
        self.epoch += 1

    def _next_row(self) -> int:
        if not self._rows:
            self._new_epoch()
        if not self._slate:
            if self._position >= len(self._rows):
                self._new_epoch()
            self._slate = self._rows[self._position : self._position + self.slate_size]
            self._position += len(self._slate)
            self._round = 0
            self._row_cursor = 0
        row = self._slate[self._row_cursor]
        self.current_epoch_visited_rows.add(row)
        self.all_visited_rows.add(row)
        self._row_cursor += 1
        if self._row_cursor == len(self._slate):
            self._row_cursor = 0
            self._round += 1
            if self._round == self.anchors_per_row:
                self._slate = []
        return row

    def _source_row(self, local_row: int) -> int:
        source_rows = getattr(self.dataset, "_source_row_indices", None)
        return int(source_rows[local_row]) if source_rows is not None else int(local_row)

    def sample_indices(self, n: int) -> list[int]:
        result: list[int] = []
        for _ in range(n):
            row = self._next_row()
            window = self.dataset._row_windows[row]
            frame = int(self.rng.integers(window.start_frame, window.end_frame + 1))
            flat = self.dataset.flat_index(row, frame)
            result.append(flat)
            self.anchor_counts[row] = self.anchor_counts.get(row, 0) + 1
            source_row = self._source_row(row)
            self.schedule_digest.update(f"{self.epoch}:{source_row}:{frame};".encode())
        return result

    def summary(self) -> dict[str, Any]:
        valid_rows = list(self.dataset._row_start_offset)
        counts = [self.anchor_counts.get(row, 0) for row in valid_rows] or [0]
        return {
            "strategy": "coverage",
            "epoch": self.epoch,
            "current_epoch_visited_rows": len(self.current_epoch_visited_rows),
            "cumulative_visited_rows": len(self.all_visited_rows),
            "valid_rows": len(valid_rows),
            "anchors_per_row_per_epoch": self.anchors_per_row,
            "anchor_min": min(counts),
            "anchor_max": max(counts),
            "schedule_hash": self.schedule_digest.hexdigest(),
        }


def parse_row_indices(value: str, row_count: int) -> tuple[list[int], dict[str, Any]]:
    if value.strip().lower() == "all":
        rows = list(range(row_count))
        mode = "all"
    else:
        rows = list(dict.fromkeys(int(part.strip()) for part in value.split(",") if part.strip()))
        if not rows:
            raise ValueError("--row-indices must be an explicit non-empty list or 'all'")
        mode = "explicit"
    if min(rows) < 0 or max(rows) >= row_count:
        raise IndexError(f"row index outside dataset: {rows[:5]!r}, rows={row_count}")
    digest = hashlib.sha256(",".join(map(str, rows)).encode()).hexdigest()
    return rows, {"mode": mode, "count": len(rows), "sha256": digest}


def validate_state_noise(base_model: str, state_noise_std: float) -> None:
    if not math.isfinite(state_noise_std) or state_noise_std < 0:
        raise ValueError("--state-noise-std must be a finite non-negative value")
    if state_noise_std > 0 and not resolve_profile(base_model).discrete_state_input:
        raise ValueError("state augmentation requires a profile with discrete state computation")


def vla_train_step_payload(
    *, model_id: str, batch: list[dict[str, Any]], learning_rate: float
) -> dict[str, Any]:
    """Build one explicit optimizer-step contract for the MINT VLA endpoint."""
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be a finite positive value")
    return {
        "model_id": model_id,
        "loss_fn": "flow_matching",
        "data": batch,
        "adam_params": {
            "learning_rate": float(learning_rate),
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1e-12,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("MINT_BASE_URL", "http://127.0.0.1:30531"))
    parser.add_argument("--api-key", default=os.environ.get("MINT_API_KEY", "tml-dummy"))
    parser.add_argument("--model", default=L.PI05_MODEL, choices=L.MODEL_CHOICES)
    parser.add_argument("--lance-dataset", type=Path, required=True)
    parser.add_argument(
        "--target-lance-dataset",
        type=Path,
        default=None,
        help="row-aligned raw Lance source for hands[].urdf_dof_target",
    )
    parser.add_argument("--row-indices", default=",".join(map(str, DEFAULT_ROW_INDICES)), help="comma-separated source rows or 'all'")
    parser.add_argument(
        "--action-source",
        choices=ACTION_SOURCES,
        default=MEASURED_DELTA,
        help="measured next-frame delta or PD setpoint residual supervision",
    )
    parser.add_argument(
        "--language-conditioning",
        choices=LANGUAGE_CONDITIONING_CHOICES,
        default=GESTURE_LANGUAGE,
        help=(
            "gesture appends the canonical index.json gesture; object_only preserves "
            "the dataset prompt; motion_variant is legacy raw_data_info.id reproduction"
        ),
    )
    parser.add_argument(
        "--gesture-index",
        type=Path,
        default=DEFAULT_GESTURE_INDEX_PATH,
        help="canonical generated-MANO index.json containing row-aligned gesture labels",
    )
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument(
        "--frame-window", choices=("contact", "full"), default="contact",
        help="contact is the default; full is an explicit diagnostic override",
    )
    parser.add_argument(
        "--contact-context-frames", type=int,
        default=L.contact_windows_lib.DEFAULT_CONTACT_CONTEXT_FRAMES,
    )
    parser.add_argument(
        "--contact-window-manifest",
        default=os.environ.get("MINT_CONTACT_WINDOW_MANIFEST", ""),
    )
    parser.add_argument(
        "--norm-stats-dir",
        type=Path,
        default=None,
        help="optional directory containing a locked norm_stats.json for this exact population",
    )
    parser.add_argument(
        "--missing-contact-policy", choices=("full", "skip", "error"), default="full",
    )
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="constant Adam learning rate sent explicitly with every VLA train step",
    )
    parser.add_argument(
        "--global-step-offset",
        type=int,
        default=0,
        help="completed coverage steps whose sample/noise RNG streams are replayed before this phase",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--augmentation-seed",
        type=int,
        default=None,
        help="state-noise RNG seed; defaults deterministically to --seed + 1",
    )
    parser.add_argument("--state-noise-std", type=float, default=0.0, help="normalized pre-tokenization Gaussian sigma for discrete state")
    parser.add_argument(
        "--extended-state",
        action="store_true",
        help="use MANO extended 32-dim state: finger contacts at [26:31], lift height at [31]. "
        "Requires recomputed norm stats. Old checkpoints are incompatible.",
    )
    parser.add_argument(
        "--target-noise-std",
        type=float,
        default=0.0,
        help="Gaussian sigma on quantile-normalized pd_target_delta supervision",
    )
    parser.add_argument("--sampling-strategy", choices=("legacy", "coverage"), default="legacy")
    parser.add_argument(
        "--stop-at",
        default=None,
        help="ISO 8601 deadline with timezone (e.g. 2026-07-27T08:00:00+08:00). "
        "Training checks the deadline after each completed step and periodic checkpoint; "
        "when reached, it saves final weights and exits with stop_reason=deadline.",
    )
    parser.add_argument("--slate-size", type=int, default=16)
    parser.add_argument(
        "--coverage-anchors-per-row",
        type=int,
        default=8,
        help="equal anchors per row before advancing to the next coverage slate",
    )
    parser.add_argument(
        "--datum-cache-size", type=int, default=4096,
        help="maximum transformed frame datums retained per training run; 0 disables caching",
    )
    parser.add_argument(
        "--prefetch-batches", type=int, default=2,
        help="ordered batches to build ahead during blocking train_step requests; 0 is synchronous",
    )
    parser.add_argument(
        "--batch-build-workers",
        type=int,
        default=4,
        help="CPU worker threads for independent datum decode/transform/encode; use 1 for serial A/B",
    )
    parser.add_argument("--save-path", required=True)
    parser.add_argument(
        "--skip-final-save",
        action="store_true",
        help="skip save_weights_for_sampler after training; intended for bounded performance probes",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=0,
        help="also save an intermediate checkpoint after this training step",
    )
    parser.add_argument(
        "--checkpoint-save-path",
        default="",
        help="sampler checkpoint path for --checkpoint-step",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="save non-final sampler checkpoints every N global steps; 0 disables",
    )
    parser.add_argument(
        "--checkpoint-save-path-template",
        default="",
        help="sampler path template for --checkpoint-every; must contain {step}",
    )
    parser.add_argument(
        "--checkpoint-state-path",
        default="",
        help="optional full training-state checkpoint path for --checkpoint-step",
    )
    parser.add_argument(
        "--checkpoint-ready-marker",
        type=Path,
        default=None,
        help="write this marker after the intermediate checkpoint is saved",
    )
    parser.add_argument(
        "--checkpoint-resume-marker",
        type=Path,
        default=None,
        help="when set, pause after the intermediate checkpoint until this marker exists",
    )
    parser.add_argument(
        "--checkpoint-poll-seconds", type=float, default=5.0,
    )
    parser.add_argument(
        "--checkpoint-resume-timeout-seconds",
        type=float,
        default=0.0,
        help="maximum checkpoint pause wait; 0 keeps the existing unbounded barrier",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--metrics-jsonl",
        type=Path,
        default=None,
        help="exclusive per-step JSONL metrics stream for long-running monitoring",
    )
    parser.add_argument(
        "--augmentation-audit-samples",
        type=int,
        default=0,
        help="with --dry-run, build at least this many samples and report augmentation population metrics",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def periodic_checkpoint_path(
    *,
    phase_step: int,
    global_step_offset: int,
    phase_steps: int,
    checkpoint_every: int,
    path_template: str,
) -> tuple[int, str] | None:
    """Return a non-final periodic checkpoint's global step and formatted path."""
    global_step = global_step_offset + phase_step
    if (
        checkpoint_every <= 0
        or phase_step >= phase_steps
        or global_step % checkpoint_every != 0
    ):
        return None
    return global_step, path_template.format(step=global_step)


from scripts.deadline import parse_stop_at


def main() -> int:
    args = parse_args()
    validate_state_noise(args.model, args.state_noise_std)
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning-rate must be a finite positive value")
    if not math.isfinite(args.target_noise_std) or args.target_noise_std < 0:
        raise ValueError("--target-noise-std must be a finite non-negative value")
    if args.target_noise_std > 0 and args.action_source != PD_TARGET_DELTA:
        raise ValueError("--target-noise-std requires --action-source pd_target_delta")
    if args.state_noise_std > 0 and args.target_noise_std > 0:
        raise ValueError("state and target augmentation are mutually exclusive")
    if args.global_step_offset < 0:
        raise ValueError("global_step_offset must be non-negative")
    stop_at_ts = None
    if args.stop_at:
        stop_at_ts = parse_stop_at(args.stop_at)
    if args.checkpoint_step < 0 or args.checkpoint_step > args.steps:
        raise ValueError(
            f"checkpoint_step must be between 0 and steps ({args.steps}), got {args.checkpoint_step}"
        )
    if args.checkpoint_step and not args.checkpoint_save_path:
        raise ValueError("--checkpoint-save-path is required with --checkpoint-step")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be non-negative")
    if args.checkpoint_every:
        if args.checkpoint_step:
            raise ValueError("--checkpoint-every and --checkpoint-step are mutually exclusive")
        if not args.checkpoint_save_path_template:
            raise ValueError(
                "--checkpoint-save-path-template is required with --checkpoint-every"
            )
        if "{step}" not in args.checkpoint_save_path_template:
            raise ValueError("--checkpoint-save-path-template must contain {step}")
        try:
            args.checkpoint_save_path_template.format(step=args.checkpoint_every)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"invalid checkpoint path template: {exc}") from exc
    if args.checkpoint_resume_marker and not args.checkpoint_ready_marker:
        raise ValueError("--checkpoint-ready-marker is required when pausing for resume")
    if args.checkpoint_poll_seconds <= 0:
        raise ValueError("checkpoint_poll_seconds must be positive")
    if args.checkpoint_resume_timeout_seconds < 0:
        raise ValueError("checkpoint_resume_timeout_seconds must be non-negative")
    if args.datum_cache_size < 0:
        raise ValueError("datum_cache_size must be non-negative")
    if args.prefetch_batches < 0:
        raise ValueError("prefetch_batches must be non-negative")
    if args.batch_build_workers <= 0:
        raise ValueError("batch_build_workers must be positive")
    if args.slate_size <= 0 or args.coverage_anchors_per_row <= 0:
        raise ValueError("slate_size and coverage_anchors_per_row must be positive")
    if args.augmentation_audit_samples < 0:
        raise ValueError("augmentation_audit_samples must be non-negative")
    if args.augmentation_audit_samples and not args.dry_run:
        raise ValueError("augmentation_audit_samples requires --dry-run")
    row_count = int(lance.dataset(str(args.lance_dataset)).count_rows())
    row_indices, row_selection = parse_row_indices(args.row_indices, row_count)
    dataset = SelectedLanceDataset(
        args.lance_dataset,
        row_indices=row_indices,
        action_horizon=args.action_horizon,
        frame_window=args.frame_window,
        contact_context_frames=args.contact_context_frames,
        contact_window_manifest=(
            Path(args.contact_window_manifest) if args.contact_window_manifest else None
        ),
        missing_contact_policy=args.missing_contact_policy,
        action_source=args.action_source,
        language_conditioning=args.language_conditioning,
        gesture_index=args.gesture_index,
        target_lance_dataset=args.target_lance_dataset,
        extended_state=args.extended_state,
    )
    sample = dataset[0]
    action_dim = int(sample["observation/state"].shape[0])
    model_config = L._build_model_config(
        args.action_horizon, action_dim=action_dim, base_model=args.model
    )
    norm_stats, norm_stats_provenance = load_or_compute_norm_stats(dataset, args.norm_stats_dir)
    data_config = L._make_data_config(
        model_config, norm_stats, action_source=dataset._action_source
    )
    sample_rng, augmentation_rng, augmentation_seed = make_rngs(
        args.seed, args.augmentation_seed
    )
    datum_cache = DatumCache(args.datum_cache_size)
    augmentation_diagnostics = AugmentationDiagnostics()
    target_augmentation_diagnostics = TargetAugmentationDiagnostics()
    batch_executor = (
        ThreadPoolExecutor(
            max_workers=args.batch_build_workers,
            thread_name_prefix="datum-build",
        )
        if args.batch_build_workers > 1
        else None
    )
    coverage_sampler = (
        CoverageSampler(
            dataset,
            sample_rng,
            slate_size=args.slate_size,
            anchors_per_row=args.coverage_anchors_per_row,
        )
        if args.sampling_strategy == "coverage"
        else None
    )
    if args.global_step_offset:
        if coverage_sampler is None:
            raise ValueError("global_step_offset currently requires coverage sampling")
        advance_coverage_rngs(
            coverage_sampler,
            augmentation_rng,
            completed_steps=args.global_step_offset,
            batch_size=args.batch_size,
            action_horizon=args.action_horizon,
            state_noise_std=args.state_noise_std,
            target_noise_std=args.target_noise_std,
        )

    print(json.dumps({
        "dataset": str(args.lance_dataset),
        "target_lance_dataset": (
            str(args.target_lance_dataset) if args.target_lance_dataset else None
        ),
        "model": args.model,
        "profile_id": resolve_profile(args.model).profile_id,
        "action_source": args.action_source,
        "language_conditioning": args.language_conditioning,
        "extended_state": bool(args.extended_state),
        "state_contract": (
            STATE_CONTRACT_ID if args.extended_state else None
        ),
        "contact_semantics": (
            CONTACT_SEMANTICS if args.extended_state else None
        ),
        "contact_rule": (
            CONTACT_RULE if args.extended_state else None
        ),
        "norm_sha_expected": (
            norm_stats_provenance.get("sha256") if args.extended_state else None
        ),
        "norm_sha_actual": norm_stats_provenance.get("sha256"),
        "gesture_index": (
            str(dataset._gesture_index_path) if dataset._gesture_index_path else None
        ),
        "gesture_index_sha256": dataset._gesture_index_sha256,
        "prompt_example": str(sample["prompt"]),
        "discrete_state_input": resolve_profile(args.model).discrete_state_input,
        "row_selection": row_selection,
        "selected_frames": len(dataset),
        "action_horizon": args.action_horizon,
        "frame_window": args.frame_window,
        "contact_context_frames": args.contact_context_frames,
        "contact_window_manifest": str(dataset._contact_window_manifest),
        "missing_contact_policy": args.missing_contact_policy,
        "window_summary": dataset.window_summary(),
        "norm_stats": norm_stats_provenance,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "global_step_offset": args.global_step_offset,
        "state_noise_std_normalized": args.state_noise_std,
        "target_noise_std_normalized": args.target_noise_std,
        "seed": args.seed,
        "sample_seed": args.seed,
        "augmentation_seed": augmentation_seed,
        "datum_cache_size": args.datum_cache_size,
        "prefetch_batches": args.prefetch_batches,
        "batch_build_workers": args.batch_build_workers,
        "sampling_strategy": args.sampling_strategy,
        "slate_size": args.slate_size,
        "coverage_anchors_per_row": args.coverage_anchors_per_row,
    }, sort_keys=True), flush=True)

    def build_next_batch() -> list[dict[str, Any]]:
        return build_batch(
            dataset,
            data_config,
            base_model=args.model,
            indices=(coverage_sampler.sample_indices(args.batch_size) if coverage_sampler else dataset.sample_indices(args.batch_size, sample_rng)),
            norm_stats=norm_stats,
            state_noise_std=args.state_noise_std,
            target_noise_std=args.target_noise_std,
            rng=augmentation_rng,
            datum_cache=datum_cache,
            augmentation_diagnostics=augmentation_diagnostics,
            target_augmentation_diagnostics=target_augmentation_diagnostics,
            executor=batch_executor,
        )

    if args.dry_run:
        audit_target = max(args.batch_size, args.augmentation_audit_samples)
        batches_to_build = math.ceil(audit_target / args.batch_size)
        first_batch: list[dict[str, Any]] | None = None
        built_samples = 0
        for _ in range(batches_to_build):
            batch = build_next_batch()
            first_batch = batch if first_batch is None else first_batch
            built_samples += len(batch)
        assert first_batch is not None
        if batch_executor is not None:
            batch_executor.shutdown(wait=True)
        print(json.dumps({
            "dry_run": True,
            "language_conditioning": args.language_conditioning,
            "gesture_index": (
                str(dataset._gesture_index_path) if dataset._gesture_index_path else None
            ),
            "gesture_index_sha256": dataset._gesture_index_sha256,
            "prompt_example": str(sample["prompt"]),
            "batch_size": len(first_batch),
            "audit_samples": built_samples,
            "state_shape": first_batch[0]["observation"]["state"]["shape"],
            "actions_shape": first_batch[0]["supervision"]["actions"]["shape"],
            "sample_seed": args.seed,
            "augmentation_seed": augmentation_seed,
            "datum_cache": datum_cache.summary(),
            "prefetch_batches": args.prefetch_batches,
            "batch_build_workers": args.batch_build_workers,
            "augmentation": augmentation_diagnostics.summary(
                args.state_noise_std,
                token_budget=resolve_profile(args.model).max_tokens,
            ),
            "target_augmentation": target_augmentation_diagnostics.summary(
                args.target_noise_std
            ),
            "sampling": coverage_sampler.summary() if coverage_sampler else {"strategy": "legacy"},
        }), flush=True)
        return 0

    base_url = args.base_url.rstrip("/")
    headers = L._headers(args.api_key)
    model_id = ""
    steps_log: list[dict[str, Any]] = []
    save_result: dict[str, Any] = {}
    intermediate_checkpoint: dict[str, Any] | None = None
    periodic_checkpoints: list[dict[str, Any]] = []
    prefetcher: BatchPrefetcher | None = None
    metrics_stream = None
    batch_build_seconds = 0.0
    batch_ready_wait_seconds = 0.0
    request_seconds = 0.0
    started_at = time.time()
    stop_reason = None
    completed_step = 0
    if args.metrics_jsonl is not None:
        args.metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
        metrics_stream = args.metrics_jsonl.open("x", encoding="utf-8", buffering=1)
    try:
        model_id, create_result = L._create_model(base_url, headers, base_model=args.model)
        if args.prefetch_batches:
            # This sole producer owns both RNG streams, so queue timing cannot
            # change sample selection or augmentation noise.
            prefetcher = BatchPrefetcher(
                args.prefetch_batches, build_next_batch, max_batches=args.steps
            )
        for step in range(1, args.steps + 1):
            step_started_at = time.perf_counter()
            global_step = args.global_step_offset + step
            # Cooperative deadline: check before entering the next step.
            if stop_at_ts is not None and time.time() >= stop_at_ts:
                stop_reason = "deadline"
                completed_step = global_step - 1
                print(json.dumps({"stop_reason": stop_reason, "deadline": args.stop_at,
                                  "completed_step": completed_step}), flush=True)
                break
            batch_wait_started_at = time.perf_counter()
            if prefetcher is not None:
                batch = prefetcher.next_batch()
            else:
                batch_started_at = time.perf_counter()
                batch = build_next_batch()
                batch_build_seconds += time.perf_counter() - batch_started_at
            batch_ready_wait = time.perf_counter() - batch_wait_started_at
            batch_ready_wait_seconds += batch_ready_wait
            request_started_at = time.perf_counter()
            result = L._await_result(base_url, headers, L._post_json(
                base_url,
                "/api/v1/mint/vla/train_step",
                headers,
                vla_train_step_payload(
                    model_id=model_id,
                    batch=batch,
                    learning_rate=args.learning_rate,
                ),
            ))
            train_request_seconds = time.perf_counter() - request_started_at
            request_seconds += train_request_seconds
            metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
            entry = {
                "step": global_step,
                "phase_step": step,
                "loss": metrics.get("loss:mean"),
                "metrics": metrics,
                "timing_seconds": {
                    "batch_ready_wait": batch_ready_wait,
                    "train_request": train_request_seconds,
                    "step_total": time.perf_counter() - step_started_at,
                },
            }
            steps_log.append(entry)
            if metrics_stream is not None:
                metrics_stream.write(json.dumps(entry, separators=(",", ":")) + "\n")
            if step == 1 or step % 100 == 0 or step == args.steps:
                print(json.dumps(entry), flush=True)

            periodic_target = periodic_checkpoint_path(
                phase_step=step,
                global_step_offset=args.global_step_offset,
                phase_steps=args.steps,
                checkpoint_every=args.checkpoint_every,
                path_template=args.checkpoint_save_path_template,
            )
            if periodic_target is not None:
                checkpoint_global_step, checkpoint_path = periodic_target
                periodic_checkpoint = {
                    "step": checkpoint_global_step,
                    "sampler": L._await_result(base_url, headers, L._post_json(
                        base_url,
                        "/api/v1/save_weights_for_sampler",
                        headers,
                        {"model_id": model_id, "path": checkpoint_path},
                    )),
                }
                periodic_checkpoints.append(periodic_checkpoint)
                print(
                    json.dumps({"periodic_checkpoint": periodic_checkpoint}, indent=2),
                    flush=True,
                )
                # Check deadline after periodic checkpoint save.
                if stop_at_ts is not None and time.time() >= stop_at_ts:
                    stop_reason = "deadline"
                    completed_step = global_step
                    print(json.dumps({"stop_reason": stop_reason, "deadline": args.stop_at,
                                      "completed_step": completed_step}), flush=True)
                    break

            # Check deadline after each completed train_step.
            if stop_at_ts is not None and time.time() >= stop_at_ts:
                stop_reason = "deadline"
                completed_step = global_step
                print(json.dumps({"stop_reason": stop_reason, "deadline": args.stop_at,
                                  "completed_step": completed_step}), flush=True)
                break

            if args.checkpoint_step and step == args.checkpoint_step:
                intermediate_checkpoint = {
                    "step": step,
                    "sampler": L._await_result(base_url, headers, L._post_json(
                        base_url,
                        "/api/v1/save_weights_for_sampler",
                        headers,
                        {"model_id": model_id, "path": args.checkpoint_save_path},
                    )),
                }
                if args.checkpoint_state_path:
                    intermediate_checkpoint["training_state"] = L._await_result(
                        base_url,
                        headers,
                        L._post_json(
                            base_url,
                            "/api/v1/save_state",
                            headers,
                            {"model_id": model_id, "path": args.checkpoint_state_path},
                        ),
                    )
                print(json.dumps({"intermediate_checkpoint": intermediate_checkpoint}, indent=2), flush=True)
                if args.checkpoint_ready_marker:
                    args.checkpoint_ready_marker.parent.mkdir(parents=True, exist_ok=True)
                    args.checkpoint_ready_marker.write_text(
                        json.dumps(intermediate_checkpoint, indent=2), encoding="utf-8"
                    )
                if args.checkpoint_resume_marker:
                    print(
                        f"waiting for checkpoint resume marker: {args.checkpoint_resume_marker}",
                        flush=True,
                    )
                    resume_deadline = (
                        time.monotonic() + args.checkpoint_resume_timeout_seconds
                        if args.checkpoint_resume_timeout_seconds > 0
                        else None
                    )
                    while not args.checkpoint_resume_marker.exists():
                        if resume_deadline is not None and time.monotonic() >= resume_deadline:
                            raise TimeoutError(
                                "checkpoint resume marker wait exceeded "
                                f"{args.checkpoint_resume_timeout_seconds}s"
                            )
                        sleep_seconds = args.checkpoint_poll_seconds
                        if resume_deadline is not None:
                            sleep_seconds = min(
                                sleep_seconds,
                                max(0.01, resume_deadline - time.monotonic()),
                            )
                        time.sleep(sleep_seconds)
                    print("checkpoint resume marker observed; continuing training", flush=True)

        if prefetcher is not None:
            batch_build_seconds = prefetcher.build_seconds
            prefetcher.close()

        if args.skip_final_save:
            save_result = {"skipped": True, "reason": "--skip-final-save"}
        else:
            save_result = L._await_result(base_url, headers, L._post_json(
                base_url, "/api/v1/save_weights_for_sampler", headers,
                {"model_id": model_id, "path": args.save_path},
            ))
        payload = {
            "experiment": (
                f"all_rows_{len(row_indices)}_state_aug"
                if row_selection["mode"] == "all" and args.state_noise_std > 0
                else (f"all_rows_{len(row_indices)}_clean" if row_selection["mode"] == "all"
                      else (f"selected_rows_{len(row_indices)}_state_aug" if args.state_noise_std > 0
                            else f"selected_rows_{len(row_indices)}_clean"))
            ),
            "client_git_commit": os.environ.get("VLA_CLIENT_GIT_COMMIT", "unknown"),
            "base_url": base_url,
            "model": args.model,
            "model_id": model_id,
            "action_source": args.action_source,
            "language_conditioning": args.language_conditioning,
            "extended_state": bool(args.extended_state),
            "state_contract": (
                STATE_CONTRACT_ID if args.extended_state else None
            ),
            "contact_semantics": (
                CONTACT_SEMANTICS if args.extended_state else None
            ),
            "contact_rule": (
                CONTACT_RULE if args.extended_state else None
            ),
            "norm_sha_expected": (
                norm_stats_provenance.get("sha256") if args.extended_state else None
            ),
            "norm_sha_actual": norm_stats_provenance.get("sha256"),
            "gesture_index": (
                str(dataset._gesture_index_path) if dataset._gesture_index_path else None
            ),
            "gesture_index_sha256": dataset._gesture_index_sha256,
            "prompt_example": str(sample["prompt"]),
            "create_result": create_result,
            "save_result": save_result,
            "intermediate_checkpoint": intermediate_checkpoint,
            "periodic_checkpoints": periodic_checkpoints,
            "global_step_offset": args.global_step_offset,
            "checkpoint_step": args.checkpoint_step,
            "checkpoint_save_path": args.checkpoint_save_path,
            "checkpoint_every": args.checkpoint_every,
            "checkpoint_save_path_template": args.checkpoint_save_path_template,
            "checkpoint_state_path": args.checkpoint_state_path,
            "checkpoint_resume_timeout_seconds": args.checkpoint_resume_timeout_seconds,
            "lance_dataset": str(args.lance_dataset),
            "target_lance_dataset": (
                str(args.target_lance_dataset) if args.target_lance_dataset else None
            ),
            "row_selection": row_selection,
            "frame_window": args.frame_window,
            "contact_context_frames": args.contact_context_frames,
            "contact_window_manifest": str(dataset._contact_window_manifest),
            "missing_contact_policy": args.missing_contact_policy,
            "window_summary": dataset.window_summary(),
            "norm_stats": norm_stats_provenance,
            "steps": steps_log,
            "metrics_jsonl": str(args.metrics_jsonl) if args.metrics_jsonl else None,
            "state_noise_std_normalized": args.state_noise_std,
            "target_noise_std_normalized": args.target_noise_std,
            "augmentation": augmentation_diagnostics.summary(args.state_noise_std),
            "target_augmentation": target_augmentation_diagnostics.summary(
                args.target_noise_std
            ),
            "sampling": coverage_sampler.summary() if coverage_sampler else {"strategy": "legacy"},
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "skip_final_save": bool(args.skip_final_save),
            "seed": args.seed,
            "sample_seed": args.seed,
            "augmentation_seed": augmentation_seed,
            "datum_cache": datum_cache.summary(),
            "prefetch_batches": args.prefetch_batches,
            "batch_build_workers": args.batch_build_workers,
            "timing_seconds": {
                "batch_build_total": batch_build_seconds,
                "batch_ready_wait_total": batch_ready_wait_seconds,
                "train_request_total": request_seconds,
                "train_requests": args.steps,
            },
            "elapsed_seconds": time.time() - started_at,
            "inference_run": False,
            "stop_reason": stop_reason,
            "completed_step": completed_step if stop_reason else args.steps,
            "deadline": args.stop_at if stop_reason else None,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"completed": True, "output_json": str(args.output_json), "save_result": save_result}, indent=2), flush=True)
        return 0
    finally:
        if metrics_stream is not None:
            metrics_stream.close()
        if prefetcher is not None:
            prefetcher.close()
        if batch_executor is not None:
            batch_executor.shutdown(wait=True, cancel_futures=True)
        if model_id:
            L._delete_model(base_url, headers, model_id)


if __name__ == "__main__":
    raise SystemExit(main())
