#!/usr/bin/env python3
"""OpenPI pi0.5 VLA smoke driver backed by a real Lance dataset.

This mirrors ``scripts/wip/openpi_vla_smoke.py`` (create_model -> train_step ->
save_weights_for_sampler -> action_session -> act -> cleanup, all via future
polling) but replaces the synthetic 1x1 PNG / zero-action payload with data read
from a Lance dataset produced by the MuJoCo/MANO replay pipeline
(``pi-finetune/data_source/lance/*.lance``).

Preprocessing is FAITHFUL: each Lance frame goes through the exact OpenPI
transform pipeline used by real fine-tuning
(``LiberoInputs`` -> ``Normalize`` -> ``TokenizePrompt`` (PaliGemma) ->
``PadStatesAndActions``), then is lowered to the mint-server VLA wire format via
the same ``_pi05_datum_from_transformed`` shape used by
``scripts/wip/openpi_libero_sft.py``. Normalization stats are computed from the
Lance dataset itself (the MuJoCo distribution differs from the libero assets),
matching ``pi-finetune/case/finetune_pi05_lance/pi05_lance_smoke.py``.

This is a pi0.5 (flow-matching, continuous action) driver only; the discrete
FAST path is intentionally out of scope here.

Environment / interpreter:
  Needs a python that can import BOTH ``openpi`` (+ jax + sentencepiece) AND
  ``lance``. On this box:
    GRB=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl
    export OPENPI_DATA_HOME=/vePFS-Mindverse/share/code/conley/.openpi_cache
    export HF_HOME=/vePFS-Mindverse/share/huggingface
    PYTHONPATH="/vePFS-Mindverse/user/intern/wenxi/mint\
:/vePFS-Mindverse/user/intern/wenxi/mint_env/extra-pydeps\
:$GRB/site-packages:$GRB/src/openpi/src:$GRB/src/openpi/packages/openpi-client/src" \
      "$GRB/host-venv/bin/python" scripts/wip/openpi_vla_smoke_lance.py \
        --lance-dataset /vePFS-Mindverse/user/intern/wenxi/pi-finetune/data_source/lance/pi_video_streams_lance_smoke.lance \
        --steps 4 --batch-size 1 --dry-run
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import replace
import io
import json
import os
import sys
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

import lance

import openpi.policies.libero_policy as libero_policy
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.shared.normalize as normalize
import openpi.training.config as openpi_config
import openpi.transforms as transforms

from mint_server.backend.core.model_registry import MODEL_CONFIGS
from scripts import contact_windows as contact_windows_lib
from scripts import mano_dataset_release
from scripts.openpi_profiles import (
    ACTION_LORA_R16_MODEL,
    ACTION_LORA_R16_STATE44_MODEL,
    LEGACY_L_LORA_MODEL,
    MODEL_CHOICES,
    OpenPIClientProfile,
    resolve_profile,
)
from scripts.target_actions import MANO_DELTA_MASK_SEGMENTS, URDF_TARGET_ABSOLUTE

# Backwards-compatible name for the default L-LoRA server identity.
PI05_MODEL = LEGACY_L_LORA_MODEL
PI05_ACTION_LORA_R16_MODEL = ACTION_LORA_R16_MODEL
PI05_ACTION_LORA_R16_STATE44_MODEL = ACTION_LORA_R16_STATE44_MODEL


# --------------------------------------------------------------------------- #
# HTTP helpers (same protocol as openpi_vla_smoke.py)                         #
# --------------------------------------------------------------------------- #
def _headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"X-API-Key": api_key}


def _get_json(base_url: str, path: str, headers: dict[str, str], *, timeout_s: float = 30.0) -> dict[str, Any]:
    resp = requests.get(f"{base_url}{path}", headers=headers, timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise TypeError(f"GET {path} returned non-dict JSON: {type(payload)}")
    return payload


def _post_json(base_url: str, path: str, headers: dict[str, str], payload: dict[str, Any], *, timeout_s: float = 900.0) -> dict[str, Any]:
    # OpenPI ops run inline in the request handler (Ray-free), so a cold-start
    # train_step POST blocks through weight load + first XLA compile — can take
    # minutes. Keep this generous so the first step doesn't read-timeout.
    resp = requests.post(f"{base_url}{path}", headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    body = resp.json()
    if not isinstance(body, dict):
        raise TypeError(f"POST {path} returned non-dict JSON: {type(body)}")
    return body


def _poll_future(
    base_url: str,
    headers: dict[str, str],
    request_id: str,
    *,
    timeout_s: float = 1800.0,
    poll_interval_s: float = 1.0,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = requests.post(f"{base_url}/api/v1/retrieve_future", headers=headers, json={"request_id": request_id}, timeout=60.0)
        if resp.status_code in {408, 503}:
            time.sleep(max(0.01, float(poll_interval_s)))
            continue
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict):
            raise TypeError(f"retrieve_future returned non-dict JSON: {type(payload)}")
        return payload
    raise TimeoutError(f"retrieve_future timed out after {timeout_s}s for request_id={request_id}")


def _await_result(
    base_url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    timeout_s: float = 1800.0,
    poll_interval_s: float = 1.0,
) -> dict[str, Any]:
    request_id = payload.get("request_id")
    if isinstance(request_id, str) and request_id:
        result = _poll_future(
            base_url, headers, request_id,
            timeout_s=timeout_s, poll_interval_s=poll_interval_s,
        )
    else:
        result = payload
    # A failed future comes back as {"error": ..., "category": ...} with HTTP 200
    # (see routes/futures.py _failed_payload). Surface it instead of silently
    # returning empty metrics — otherwise a mid-run OOM looks like a clean run.
    if isinstance(result, dict) and result.get("error") and "metrics" not in result:
        raise RuntimeError(f"future {request_id} failed: {result.get('error')}")
    return result


def _request_action_batch(
    base_url: str,
    headers: dict[str, str],
    action_session_id: str,
    observations: list[dict[str, Any]],
    *,
    fixed_batch_size: int,
    timeout_s: float = 1800.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Infer observations through ``act_batch`` and repeat-pad the tail.

    Padding keeps the batch shape stable across a session and makes a batch of
    four divisible across the project's four visible server GPUs.  Only the
    first ``len(observations)`` action payloads are returned.  The timing wraps
    HTTP submission, server execution, result materialization, and future
    retrieval; it therefore cannot report an asynchronous JAX dispatch as a
    completed inference.
    """
    actual_count = len(observations)
    batch_size = int(fixed_batch_size)
    if actual_count <= 0:
        raise ValueError("observations must not be empty")
    if batch_size <= 0:
        raise ValueError("fixed_batch_size must be positive")
    if actual_count > batch_size:
        raise ValueError(
            f"{actual_count} observations exceed fixed_batch_size={batch_size}"
        )
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict) or not {"state", "model_input"} <= set(observation):
            raise ValueError(
                f"observation {index} must contain state and model_input"
            )
    padded = list(observations)
    padded.extend([observations[-1]] * (batch_size - actual_count))
    started = time.monotonic()
    result = _await_result(
        base_url,
        headers,
        _post_json(
            base_url,
            f"/api/v1/mint/action_sessions/{action_session_id}/act_batch",
            headers,
            {"observations": padded},
            timeout_s=timeout_s,
        ),
        timeout_s=timeout_s,
        poll_interval_s=0.05,
    )
    wall_seconds = time.monotonic() - started
    action_payloads = result.get("actions")
    if not isinstance(action_payloads, list) or len(action_payloads) != batch_size:
        raise RuntimeError(
            f"act_batch returned {type(action_payloads).__name__} with "
            f"length={len(action_payloads) if isinstance(action_payloads, list) else 'n/a'}; "
            f"expected {batch_size}: {result!r}"
        )
    timing = {
        "wall_seconds": wall_seconds,
        "actual_observation_count": actual_count,
        "request_batch_size": batch_size,
        "padding_count": batch_size - actual_count,
        "server_elapsed_ms": result.get("elapsed_ms"),
        "used_data_sharding": result.get("used_data_sharding"),
        "response_batch_size": result.get("batch_size"),
    }
    return action_payloads[:actual_count], timing


# --------------------------------------------------------------------------- #
# Lance dataset -> OpenPI raw sample                                          #
# --------------------------------------------------------------------------- #
def _decode_jpeg(blob: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))


def _encode_png_base64(image: np.ndarray) -> str:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    bio = io.BytesIO()
    Image.fromarray(arr).save(bio, format="PNG")
    return base64.b64encode(bio.getvalue()).decode("utf-8")


class LanceViewpi05Dataset:
    """Frame-window view over a Lance episode table, matching the schema used by
    ``pi-finetune/case/finetune_pi05_lance/pi05_lance_smoke.py``.

    Each row is one episode with per-frame lists ``image``/``wrist_image``
    (JPEG bytes), ``state``/``actions`` (float vectors) and a scalar ``prompt``.
    A sample is an ``action_horizon``-length window of actions anchored at a
    frame. When an episode has fewer than ``action_horizon`` frames after the
    anchor (e.g. the tiny smoke export with 2 frames), the trailing actions are
    padded by repeating the last available action so the window always has the
    horizon the model expects.

    Lazy row loading: ``__init__`` only reads ``episode_metadata`` (a small
    per-episode struct) to build the frame index -- it does NOT call
    ``to_table().to_pylist()`` on the full table. That used to be fine on the
    small smoke datasets (~100MB total), but on a real-scale dataset (hundreds
    of GB of JPEG image bytes across thousands of episodes) it means
    materializing every image/state/action array into Python objects up front,
    which can take tens of minutes or hang the caller regardless of
    ``max_samples`` (confirmed: 50 rows' `image` column alone took >90s to
    `to_pylist()` on `new_all_generated_mano_with_images.lance`, while
    `episode_metadata` for all 7539 rows took 0.06s). ``__getitem__`` instead
    calls ``lance.Dataset.take([row_index], columns=[...])`` on demand, with a
    small LRU cache keyed by row_index so consecutive frames within the same
    episode (the common access pattern here) don't re-fetch the whole episode
    every call.
    """

    _ROW_CACHE_SIZE = 4

    def __init__(
        self,
        lance_dataset: Path,
        *,
        action_horizon: int,
        max_samples: int | None = None,
        slate_size: int = 16,
        slate_rotate_every: int = 250,
        frame_window: str = "contact",
        contact_context_frames: int = contact_windows_lib.DEFAULT_CONTACT_CONTEXT_FRAMES,
        contact_window_manifest: Path | None = None,
        missing_contact_policy: str = "full",
        extended_state: bool = False,
        state_contract: str | None = None,
    ) -> None:
        """`slate_size`/`slate_rotate_every` control `sample_indices()`'s
        episode-slate rotation (see its docstring) -- irrelevant if callers
        keep indexing via `dataset[i]`/`__getitem__` directly (e.g.
        `_compute_norm_stats`, `probe_lance_dataset`), only used by
        `sample_indices()`.

        `extended_state`: legacy alias for state_contract="state32".
        `state_contract`: None for legacy raw state, "state32" for contact/lift,
        "state44" for derived 26D geometry/dynamics, or "state41" for the
        persisted native-simulated 28D release.
        """
        if state_contract not in {None, "state32", "state44", "state41"}:
            raise ValueError(f"unsupported state_contract {state_contract!r}")
        if extended_state and state_contract not in {None, "state32"}:
            raise ValueError(
                "--extended-state cannot be combined with an explicit state44/state41 contract"
            )
        self._state_contract = state_contract or ("state32" if extended_state else None)
        self._extended_state = self._state_contract is not None
        self._state_dim = {"state44": 44, "state41": 41}.get(self._state_contract, 32)
        self._dataset = lance.dataset(str(lance_dataset))
        self._dataset_path = Path(lance_dataset)
        # Metadata stays small; contact records are scanned separately and
        # cached in a sidecar manifest so image columns are never materialized.
        self._rows: list[dict[str, Any]] = self._dataset.to_table(
            columns=["episode_metadata", "trajectory_metadata"]
        ).to_pylist()
        self._source_row_indices = list(range(len(self._rows)))
        self._action_horizon = int(action_horizon)
        self._frame_window = str(frame_window)
        self._contact_context_frames = int(contact_context_frames)
        self._contact_window_manifest = (
            Path(contact_window_manifest)
            if contact_window_manifest is not None
            else contact_windows_lib.default_manifest_path(lance_dataset)
        )
        self._missing_contact_policy = str(missing_contact_policy)
        manifest_entries: dict[int, dict[str, Any]] = {}
        if self._frame_window == "contact":
            manifest_entries = contact_windows_lib.load_or_build_windows(
                self._dataset,
                lance_dataset,
                range(len(self._rows)),
                manifest_path=self._contact_window_manifest,
                context_frames=self._contact_context_frames,
                missing_policy=self._missing_contact_policy,
            )
        elif self._frame_window != "full":
            raise ValueError(
                f"frame_window must be contact or full, got {self._frame_window!r}"
            )

        self._row_cache: "OrderedDict[int, dict[str, Any]]" = OrderedDict()
        self._index: list[tuple[int, int]] = []
        self._row_start_offset: dict[int, int] = {}
        self._row_window_start: dict[int, int] = {}
        self._row_windows: dict[int, contact_windows_lib.ContactWindow] = {}
        reached_limit = False
        for row_index, row in enumerate(self._rows):
            total_frames = int(row["episode_metadata"]["total_frames"])
            window = contact_windows_lib.select_window(
                row,
                row_index=row_index,
                total_frames=total_frames,
                mode=self._frame_window,
                manifest_entry=manifest_entries.get(row_index),
                context_frames=self._contact_context_frames,
                missing_policy=self._missing_contact_policy,
            )
            if window is None or window.frame_count <= 0:
                continue
            self._row_windows[row_index] = window
            self._row_start_offset[row_index] = len(self._index)
            self._row_window_start[row_index] = window.start_frame
            # Frames are absolute source indices. Action targets stop at the
            # selected end frame and repeat-pad there, so the discarded suffix
            # cannot leak back into a selected training sample.
            for frame_index in range(window.start_frame, window.end_frame + 1):
                self._index.append((row_index, frame_index))
                if max_samples is not None and len(self._index) >= max_samples:
                    if frame_index < window.end_frame:
                        self._row_windows[row_index] = replace(
                            window, end_frame=frame_index, status=f"{window.status}:max_samples"
                        )
                    reached_limit = True
                    break
            if reached_limit:
                break
        if not self._index:
            raise ValueError(f"No samples available in {lance_dataset}")

        self._slate_size = max(1, int(slate_size))
        self._slate_rotate_every = max(1, int(slate_rotate_every))
        self._slate_row_indices: list[int] = []
        self._slate_calls_since_rotate = 0

    def __len__(self) -> int:
        return len(self._index)

    def row_window(self, row_index: int) -> contact_windows_lib.ContactWindow:
        return self._row_windows[row_index]

    def flat_index(self, row_index: int, absolute_frame: int) -> int:
        window = self.row_window(row_index)
        frame = int(absolute_frame)
        if not window.start_frame <= frame <= window.end_frame:
            raise IndexError(
                f"frame {frame} is outside row {row_index} window "
                f"[{window.start_frame}, {window.end_frame}]"
            )
        return self._row_start_offset[row_index] + frame - window.start_frame

    def window_summary(self) -> dict[str, Any]:
        source_frames = sum(window.total_frames for window in self._row_windows.values())
        selected_frames = sum(window.frame_count for window in self._row_windows.values())
        status_counts: dict[str, int] = {}
        for window in self._row_windows.values():
            status_counts[window.status] = status_counts.get(window.status, 0) + 1
        return {
            "frame_window": self._frame_window,
            "contact_context_frames": self._contact_context_frames,
            "contact_window_manifest": str(self._contact_window_manifest),
            "trajectory_count": len(self._row_windows),
            "source_frame_count": source_frames,
            "selected_frame_count": selected_frames,
            "retained_fraction": selected_frames / source_frames if source_frames else None,
            "status_counts": dict(sorted(status_counts.items())),
        }

    def _row_cache_capacity(self) -> int:
        return max(self._ROW_CACHE_SIZE, self._slate_size)

    def _get_row(self, row_index: int) -> dict[str, Any]:
        """Fetch (and LRU-cache) the per-frame `state`/`actions`/`prompt`/
        `image`/`wrist_image` columns for one episode row, on demand."""
        cached = self._row_cache.get(row_index)
        if cached is not None:
            self._row_cache.move_to_end(row_index)
            return cached
        columns = ["state", "actions", "prompt", "image", "wrist_image"]
        if self._extended_state and self._state_contract != "state41":
            columns += ["contact", "objects"]
        if self._state_contract == "state44":
            columns.append("timestamp")
        table = self._dataset.take([row_index], columns=columns)
        row = table.to_pylist()[0]
        if self._state_contract == "state44":
            from scripts.mano_state44_contract import compute_source_state44_sequence

            metadata = self._rows[row_index].get("trajectory_metadata") or {}
            object_names = metadata.get("object_names") or []
            if len(object_names) != 1:
                raise ValueError(f"state44 requires exactly one object name, got {object_names!r}")
            row["_state44_sequence"] = compute_source_state44_sequence(
                {**row, "trajectory_metadata": metadata}, object_names[0]
            )
        self._row_cache[row_index] = row
        self._row_cache.move_to_end(row_index)
        while len(self._row_cache) > self._row_cache_capacity():
            self._row_cache.popitem(last=False)
        return row

    def _rotate_slate(self, rng: np.random.Generator) -> None:
        valid_rows = list(self._row_start_offset.keys())
        n = min(self._slate_size, len(valid_rows))
        self._slate_row_indices = [int(x) for x in rng.choice(valid_rows, size=n, replace=False)]
        for row_index in self._slate_row_indices:
            self._get_row(row_index)  # pre-warm: pays the "read whole episode" cost once per slate
        self._slate_calls_since_rotate = 0

    def sample_indices(self, n: int, rng: np.random.Generator) -> list[int]:
        """Draw `n` flat indices (valid for `dataset[i]`) using episode-slate
        rotation instead of pure uniform sampling over `len(self)`.

        Why: each `dataset[i]` call that misses the row cache pays the cost of
        reading the WHOLE episode row (Lance's `list<binary>` image columns
        can't be randomly accessed per-frame -- see `_get_row`'s docstring and
        `ExperimentLog_MultiGPU.md`'s "Step 6" for the measured numbers). On a
        dataset with many more frames than episodes (e.g. 6.8M frames across
        7539 episodes here), pure uniform sampling over frames almost always
        lands on a fresh episode every draw, so the per-batch data-loading cost
        (~2.7s/batch observed) ends up dwarfing the actual JAX train_step
        (~0.5s/batch) -- the GPU sits idle waiting on IO, not on compute.

        This keeps a small rotating "slate" of `slate_size` episodes resident
        in the row cache, and samples `n` indices by picking a random episode
        from the current slate + a random frame within it. The per-episode
        "read whole episode" cost is paid once per `slate_rotate_every` calls
        instead of on every draw, amortizing it across many steps. This is the
        standard "sample trajectories, then sample frames within them" access
        pattern used by most robot-learning/RL data loaders -- it does not
        change what data the model sees over the course of training (the
        slate itself is drawn uniformly at random, same as before), only the
        temporal correlation of which frames are grouped into nearby batches.
        """
        if not self._slate_row_indices or self._slate_calls_since_rotate >= self._slate_rotate_every:
            self._rotate_slate(rng)
        self._slate_calls_since_rotate += 1

        result: list[int] = []
        for _ in range(n):
            row_index = int(rng.choice(self._slate_row_indices))
            window = self._row_windows[row_index]
            frame = int(rng.integers(window.start_frame, window.end_frame + 1))
            result.append(self.flat_index(row_index, frame))
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index, frame = self._index[index]
        window = self._row_windows[row_index]
        row = self._get_row(row_index)
        end = min(frame + self._action_horizon, window.end_frame + 1)
        # `row["actions"]` may be a resident numpy array. DeltaActions mutates
        # its input in place, so returning a view would subtract the query state
        # from the cached absolute targets again on every reuse/coverage epoch.
        actions = np.asarray(row["actions"][frame:end], dtype=np.float32).copy()
        if actions.shape[0] < self._action_horizon:
            pad = np.repeat(actions[-1:], self._action_horizon - actions.shape[0], axis=0)
            actions = np.concatenate([actions, pad], axis=0)
        if getattr(self, "_state_contract", None) == "state44":
            state = np.asarray(row["_state44_sequence"][frame], dtype=np.float32).copy()
        elif getattr(self, "_state_contract", None) == "state41":
            state = np.asarray(row["state"][frame], dtype=np.float32).copy()
            if state.shape != (41,):
                raise ValueError(f"persisted state41 frame has shape {state.shape}")
        elif self._extended_state:
            from scripts.mano_state_contract import build_extended_state
            objects = row["objects"]
            traj_meta = self._rows[row_index].get("trajectory_metadata", {})
            obj_names = traj_meta.get("object_names", []) if isinstance(traj_meta, dict) else []
            if len(obj_names) != 1:
                raise ValueError(
                    f"extended state requires exactly one object_name in trajectory_metadata, "
                    f"got {obj_names!r} at row {row_index}"
                )
            if len(objects) != 1:
                raise ValueError(
                    f"extended state requires exactly one object in objects, "
                    f"got {len(objects)} at row {row_index}"
                )
            obj_name = obj_names[0]
            obj_pos = objects[0]["pos"]
            state = build_extended_state(
                row["state"][frame],
                row["contact"][frame],
                obj_name,
                obj_pos[frame][2],
                obj_pos[0][2],
            )
        else:
            state = np.asarray(row["state"][frame], dtype=np.float32)
        return {
            "observation/image": _decode_jpeg(row["image"][frame]),
            "observation/wrist_image": _decode_jpeg(row["wrist_image"][frame]),
            "observation/state": state,
            "actions": actions,
            "prompt": str(row["prompt"]),
        }


# --------------------------------------------------------------------------- #
# OpenPI transform pipeline (faithful) + norm stats from the Lance dataset    #
# --------------------------------------------------------------------------- #
def _build_model_config(
    action_horizon: int,
    action_dim: int = 32,
    *,
    state_dim: int | None = None,
    base_model: str | None = None,
    profile: OpenPIClientProfile | str | None = None,
) -> pi0_config.Pi0Config:
    """Build client transforms for a supported server model identity.

    The profile governs tokenization only. Adapter construction and placement are
    server responsibilities selected by ``base_model`` in HTTP payloads.
    """
    resolved = resolve_profile(base_model, profile=profile)
    resolved_state_dim = (
        resolved.state_dim if resolved.discrete_state_input else action_dim
    ) if state_dim is None else int(state_dim)
    if resolved.discrete_state_input and (
        resolved_state_dim != resolved.state_dim
        or action_dim != resolved.action_dim
        or action_horizon != resolved.action_horizon
    ):
        raise ValueError(
            f"{resolved.profile_id} requires state_dim={resolved.state_dim}, "
            f"action_dim={resolved.action_dim}, and action_horizon={resolved.action_horizon}; got "
            f"{resolved_state_dim}, {action_dim}, and {action_horizon}"
        )
    return pi0_config.Pi0Config(
        pi05=True,
        state_dim=resolved_state_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        max_token_len=resolved.max_tokens,
        fail_on_token_truncation=resolved.fail_on_token_truncation,
        discrete_state_input=resolved.discrete_state_input,
        # Adapter ownership remains server-side, but matching variants prevent
        # the client transform declaration from drifting from that contract.
        paligemma_variant=resolved.paligemma_variant,
        action_expert_variant=resolved.action_expert_variant,
    )


def _make_data_config(
    model_config: pi0_config.Pi0Config,
    norm_stats: dict[str, normalize.NormStats] | None,
    *,
    action_source: str | None = None,
    delta_mask_segments: tuple[int, ...] | None = None,
) -> openpi_config.DataConfig:
    data_transforms = transforms.Group(
        inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
        outputs=[libero_policy.LiberoOutputs()],
    )
    # B scheme: for the absolute urdf_dof_target label, let the framework convert
    # the delta-eligible dims (xyz + fingers) to `target - q[t]` before Normalize
    # and invert with `+ q[t]` on outputs. Euler dims and padding stay absolute.
    # data_transforms.inputs run BEFORE Normalize in _transform_sample, so the
    # subtraction is on raw physical values (required for correct delta).
    if action_source == URDF_TARGET_ABSOLUTE:
        segments = MANO_DELTA_MASK_SEGMENTS if delta_mask_segments is None else delta_mask_segments
        delta_action_mask = transforms.make_bool_mask(*segments)
        if len(delta_action_mask) != model_config.action_dim:
            raise ValueError(
                f"delta mask {segments} has width {len(delta_action_mask)}, "
                f"expected action_dim={model_config.action_dim}"
            )
        data_transforms = data_transforms.push(
            inputs=[transforms.DeltaActions(delta_action_mask)],
            outputs=[transforms.AbsoluteActions(delta_action_mask)],
        )
    model_transforms = openpi_config.ModelTransformFactory()(model_config)
    return openpi_config.DataConfig(
        repo_id="pi0_mujoco_lance",
        asset_id="pi0_mujoco_lance",
        norm_stats=norm_stats,
        data_transforms=data_transforms,
        model_transforms=model_transforms,
        use_quantile_norm=model_config.model_type != _model.ModelType.PI0,
    )


def _compute_norm_stats(
    dataset: LanceViewpi05Dataset, *, batch_rows: int = 32
) -> dict[str, normalize.NormStats]:
    """Compute stats from exactly the source frames exposed for training.

    Contact-window selection changes the training population, so normalization
    must use the same selected frames. Reading only ``state`` and ``actions`` in
    row batches avoids image decoding and bounds memory. Raw selected actions
    are counted once each; repeat-padding at a window boundary is excluded from
    the population statistics.
    """
    frames_by_row: dict[int, tuple[int, int]] = {}
    for row_index, frame in dataset._index:
        previous = frames_by_row.get(row_index)
        if previous is None:
            frames_by_row[row_index] = (frame, frame)
        else:
            frames_by_row[row_index] = (min(previous[0], frame), max(previous[1], frame))
    if not frames_by_row:
        raise ValueError("cannot compute normalization statistics for an empty dataset")

    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    local_rows = sorted(frames_by_row)
    source_rows = getattr(dataset, "_source_row_indices", list(range(len(dataset._rows))))
    step = max(1, int(batch_rows))
    for offset in range(0, len(local_rows), step):
        local_batch = local_rows[offset : offset + step]
        source_batch = [int(source_rows[local_row]) for local_row in local_batch]
        rows = dataset._dataset.take(
            source_batch, columns=["state", "actions"]
        ).to_pylist()
        for local_row, row in zip(local_batch, rows, strict=True):
            start, end = frames_by_row[local_row]
            states = np.asarray(row["state"][start : end + 1], dtype=np.float32)
            actions = np.asarray(row["actions"][start : end + 1], dtype=np.float32)
            if states.shape[0] != actions.shape[0] or states.shape[0] == 0:
                raise ValueError(
                    f"row {source_rows[local_row]} has inconsistent selected "
                    f"state/action lengths: {states.shape[0]} vs {actions.shape[0]}"
                )
            state_stats.update(states)
            action_stats.update(actions)
    return {
        "state": state_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
    }


def _transform_sample(sample: dict[str, Any], data_config: openpi_config.DataConfig) -> dict[str, Any]:
    data = sample
    for transform in (
        *data_config.data_transforms.inputs,
        transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ):
        data = transform(data)
    return data


def _pi05_datum_from_transformed(base_model: str, item: dict[str, Any]) -> dict[str, Any]:
    """Lower a transformed OpenPI sample to the mint-server VLA wire format.

    Identical shape to scripts/wip/openpi_libero_sft.py:_pi05_datum_from_transformed:
      - one image chunk per camera in ModelConfig.camera_layout (positional),
      - exactly one pre-tokenized encoded_text prompt chunk,
      - rank-1 state, rank-2 actions [action_horizon, action_dim].
    """
    model_cfg = MODEL_CONFIGS[base_model]
    profile = resolve_profile(base_model)
    prompt_mask = np.asarray(item["tokenized_prompt_mask"]).astype(bool)
    prompt_tokens = np.asarray(item["tokenized_prompt"])[prompt_mask].astype(int).tolist()
    state = np.asarray(item["state"], dtype=np.float32)
    actions = np.asarray(item["actions"], dtype=np.float32)
    if state.shape != (profile.state_dim,):
        raise ValueError(
            f"{profile.profile_id} requires transformed state shape {(profile.state_dim,)}, "
            f"got {state.shape}"
        )
    expected_actions = (profile.action_horizon, profile.action_dim)
    if actions.shape != expected_actions:
        raise ValueError(
            f"{profile.profile_id} requires transformed actions shape {expected_actions}, "
            f"got {actions.shape}"
        )
    image_chunks = []
    for camera_name in model_cfg.camera_layout:
        image_chunks.append({
            "type": "image",
            "data": _encode_png_base64(np.asarray(item["image"][camera_name])),
            "format": "png",
            "expected_tokens": 256,
        })
    return {
        "observation": {
            "state": {
                "data": state.reshape(-1).tolist(),
                "shape": list(state.shape),
                "dtype": "float32",
            },
            "model_input": {"chunks": [*image_chunks, {"type": "encoded_text", "tokens": prompt_tokens}]},
        },
        "supervision": {
            "actions": {"data": actions.reshape(-1).tolist(), "shape": list(actions.shape), "dtype": "float32"},
        },
    }


# --------------------------------------------------------------------------- #
# mint-server model lifecycle                                                 #
# --------------------------------------------------------------------------- #
def _create_model(base_url: str, headers: dict[str, str], *, base_model: str) -> tuple[str, dict[str, Any]]:
    payload = {
        "session_id": f"lance-smoke-{uuid.uuid4().hex[:12]}",
        "model_seq_id": 0,
        "base_model": base_model,
        "lora_config": {"rank": 16, "train_attn": True, "train_mlp": True, "train_unembed": True},
        "user_metadata": {"script": "scripts/wip/openpi_vla_smoke_lance.py"},
    }
    result = _await_result(base_url, headers, _post_json(base_url, "/api/v1/create_model", headers, payload))
    model_id = result.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"create_model missing model_id: {result!r}")
    return model_id, result


def _delete_model(base_url: str, headers: dict[str, str], model_id: str) -> None:
    try:
        requests.delete(f"{base_url}/api/v1/models/{model_id}", headers=headers, timeout=120.0)
    except Exception:
        pass


# 每个 index 的 transform 结果(图像 resize + PaliGemma 分词 + pad)与训练步无关、
# 对固定数据集是常量;缓存后可把「每步重跑 transform」的开销从 ~40s 降到亚秒级。
# 语义不变:同一 index 仍喂完全相同的 datum。设 MINT_PI05_NO_BATCH_CACHE=1 可关。
_DATUM_CACHE: dict[tuple[str, int], dict[str, Any]] = {}


def _build_batch(dataset: LanceViewpi05Dataset, data_config: openpi_config.DataConfig, *, base_model: str, indices: list[int]) -> list[dict[str, Any]]:
    use_cache = str(os.environ.get("MINT_PI05_NO_BATCH_CACHE") or "").strip() not in ("1", "true", "True")
    batch: list[dict[str, Any]] = []
    for i in indices:
        key = (base_model, i % len(dataset))
        if use_cache and key in _DATUM_CACHE:
            batch.append(_DATUM_CACHE[key])
            continue
        transformed = _transform_sample(dataset[i % len(dataset)], data_config)
        datum = _pi05_datum_from_transformed(base_model, transformed)
        if use_cache:
            _DATUM_CACHE[key] = datum
        batch.append(datum)
    return batch


def main() -> int:
    default_ds = str(mano_dataset_release.resolve_role("training_dataset"))
    parser = argparse.ArgumentParser(description="pi0.5 VLA smoke driver over a real Lance dataset")
    parser.add_argument("--base-url", default=os.environ.get("MINT_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.environ.get("MINT_API_KEY", "dummy"))
    parser.add_argument("--model", default=PI05_MODEL, choices=MODEL_CHOICES)
    parser.add_argument("--lance-dataset", default=os.environ.get("MINT_LANCE_DATASET", default_ds))
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--frame-window", choices=("contact", "full"), default="contact",
        help="contact is the default; full is an explicit diagnostic override",
    )
    parser.add_argument(
        "--contact-context-frames", type=int,
        default=contact_windows_lib.DEFAULT_CONTACT_CONTEXT_FRAMES,
    )
    parser.add_argument(
        "--contact-window-manifest",
        default=os.environ.get("MINT_CONTACT_WINDOW_MANIFEST", ""),
        help="JSON sidecar; canonical data resolves release role contact_windows",
    )
    parser.add_argument(
        "--missing-contact-policy", choices=("full", "skip", "error"), default="full",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-action", action="store_true", help="skip save_weights + action session + act")
    parser.add_argument("--dry-run", action="store_true", help="read Lance + build one batch, print shapes, do NOT contact the server")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    model_cfg = MODEL_CONFIGS[args.model]
    if int(model_cfg.action_horizon or 0) != args.action_horizon:
        print(f"warning: --action-horizon {args.action_horizon} != model action_horizon {model_cfg.action_horizon}; "
              f"the server will reject a mismatch", file=sys.stderr)

    # --- Lance -> transform pipeline (faithful) ------------------------------ #
    dataset = LanceViewpi05Dataset(
        Path(args.lance_dataset),
        action_horizon=args.action_horizon,
        max_samples=args.max_samples,
        frame_window=args.frame_window,
        contact_context_frames=args.contact_context_frames,
        contact_window_manifest=(
            Path(args.contact_window_manifest) if args.contact_window_manifest else None
        ),
        missing_contact_policy=args.missing_contact_policy,
    )

    sample = dataset[0]
    profile = resolve_profile(args.model)
    state_dim = int(sample["observation/state"].shape[0])
    action_dim = int(sample["actions"].shape[-1])
    if profile.discrete_state_input and (
        state_dim != profile.state_dim or action_dim != profile.action_dim
    ):
        raise ValueError(
            f"dataset state/action widths {(state_dim, action_dim)} disagree with "
            f"profile widths {(profile.state_dim, profile.action_dim)}"
        )
    print(f"Resolved state_dim={state_dim}, action_dim={action_dim}")

    model_config = _build_model_config(
        args.action_horizon,
        state_dim=state_dim,
        action_dim=action_dim,
        base_model=args.model,
    )
    norm_stats = _compute_norm_stats(dataset)
    data_config = _make_data_config(model_config, norm_stats)
    rng = np.random.default_rng(args.seed)

    def _sample_indices(n: int) -> list[int]:
        return [int(rng.integers(0, len(dataset))) for _ in range(n)]

    print(f"lance_dataset: {args.lance_dataset}")
    print(f"samples(frame windows): {len(dataset)}  action_horizon={args.action_horizon}")
    print("window_summary=" + json.dumps(dataset.window_summary(), sort_keys=True))
    for key, stats in norm_stats.items():
        q01 = None if stats.q01 is None else tuple(stats.q01.shape)
        print(f"  norm[{key}]: mean{tuple(stats.mean.shape)} std{tuple(stats.std.shape)} q01={q01}")

    # --- Dry run: verify the data path without a server ---------------------- #
    if args.dry_run:
        batch = _build_batch(dataset, data_config, base_model=args.model, indices=_sample_indices(args.batch_size))
        datum = batch[0]
        chunks = datum["observation"]["model_input"]["chunks"]
        img_chunks = [c for c in chunks if c["type"] == "image"]
        txt_chunks = [c for c in chunks if c["type"] == "encoded_text"]
        print("dry-run OK: built batch of", len(batch))
        print(f"  image chunks: {len(img_chunks)} (camera_layout={model_cfg.camera_layout})")
        print(f"  prompt tokens: {len(txt_chunks[0]['tokens'])}  first8={txt_chunks[0]['tokens'][:8]}")
        print(f"  state shape: {datum['observation']['state']['shape']}")
        print(f"  actions shape: {datum['supervision']['actions']['shape']}")
        return 0

    # --- Live run against the mint-server ------------------------------------ #
    base_url = args.base_url.rstrip("/")
    headers = _headers(args.api_key)

    def _actors_snapshot() -> Any:
        # Ray-free (openpi Separate) servers have no /internal/actors endpoint;
        # this is observability only, so tolerate its absence.
        try:
            return _get_json(base_url, "/internal/actors", headers)
        except Exception as e:
            return {"unavailable": f"{type(e).__name__}: {e}"}

    actors_before = _actors_snapshot()
    model_id = ""
    action_session_id = ""
    steps_log: list[dict[str, Any]] = []
    try:
        model_id, create_result = _create_model(base_url, headers, base_model=args.model)
        for step in range(1, args.steps + 1):
            batch = _build_batch(dataset, data_config, base_model=args.model, indices=_sample_indices(args.batch_size))
            train_result = _await_result(base_url, headers, _post_json(
                base_url, "/api/v1/mint/vla/train_step", headers,
                {"model_id": model_id, "loss_fn": "flow_matching", "data": batch}))
            metrics = train_result.get("metrics", {}) if isinstance(train_result, dict) else {}
            loss = metrics.get("loss:mean")
            steps_log.append({"step": step, "loss": loss, "metrics": metrics})
            print(json.dumps({"step": step, "loss": loss}), flush=True)

        save_result: dict[str, Any] = {}
        action_result: dict[str, Any] = {}
        if not args.skip_action:
            save_result = _await_result(base_url, headers, _post_json(
                base_url, "/api/v1/save_weights_for_sampler", headers,
                {"model_id": model_id, "path": f"lance_smoke_sampler_{uuid.uuid4().hex[:8]}"}))
            model_path = save_result.get("path")
            if not isinstance(model_path, str) or not model_path:
                raise RuntimeError(f"save_weights_for_sampler missing path: {save_result!r}")
            action_created = _post_json(base_url, "/api/v1/mint/action_sessions", headers, {
                "session_id": f"lance-smoke-action-{uuid.uuid4().hex[:12]}",
                "base_model": args.model, "model_path": model_path, "owner_id": save_result.get("owner_id")})
            action_session_id = action_created["action_session_id"]
            obs = _build_batch(dataset, data_config, base_model=args.model, indices=[0])[0]["observation"]
            action_result = _await_result(base_url, headers, _post_json(
                base_url, f"/api/v1/mint/action_sessions/{action_session_id}/act", headers, {"observation": obs}))

        actors_after = _actors_snapshot()
        payload = {
            "client_git_commit": os.environ.get("VLA_CLIENT_GIT_COMMIT", "unknown"),
            "base_url": base_url,
            "model": args.model,
            "model_id": model_id,
            "lance_dataset": args.lance_dataset,
            "action_horizon": args.action_horizon,
            "frame_window": args.frame_window,
            "contact_context_frames": args.contact_context_frames,
            "contact_window_manifest": str(dataset._contact_window_manifest),
            "missing_contact_policy": args.missing_contact_policy,
            "window_summary": dataset.window_summary(),
            "batch_size": args.batch_size,
            "seed": args.seed,
            "create_result": create_result,
            "steps": steps_log,
            "save_result": save_result,
            "action_session_id": action_session_id,
            "action_result": action_result,
            "actors_before": actors_before,
            "actors_after": actors_after,
        }
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        print(json.dumps(payload, indent=2))
        return 0
    finally:
        if action_session_id:
            try:
                requests.delete(f"{base_url}/api/v1/mint/action_sessions/{action_session_id}", headers=headers, timeout=120.0)
            except Exception:
                pass
        if model_id:
            _delete_model(base_url, headers, model_id)


if __name__ == "__main__":
    raise SystemExit(main())
