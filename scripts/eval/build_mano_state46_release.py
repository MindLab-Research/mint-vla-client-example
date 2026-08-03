#!/usr/bin/env python3
"""Build the qualified native MANO state46/action32 RGB release.

Each worker restores a qualified native qpos/qvel trace into one visual model,
calls ``mj_forward`` exactly once per frame, and extracts state46/contact/object
plus Head/Wrist JPEG from that same ``MjData``. Workers own independent Lance
shards; a deterministic final pass concatenates contiguous shards in filtered
row order. Source Lance and quality traces are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
import uuid

import numpy as np

from scripts import mano_state46_contract as state46
from scripts.eval import mano_action_support as cameras
from scripts.eval import manorl_native_physics as physics
from scripts.eval import replay_mano_target_physics as replay_quality

RELEASE_CONTRACT = "mano_28d_native_replay_state46_rgb_v1"
DEFAULT_QUALITY_ROOT = Path(
    "/vePFS-Mindverse/user/intern/wenxi/results/datas/28dof_manohand/quality/"
    "native_target_replay_28d_v2"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/vePFS-Mindverse/user/intern/wenxi/results/datas/28dof_manohand/release/"
    "mano_28d_native_replay_state46_rgb_v1"
)
EXPECTED_QUALITY_SUMMARY_SHA256 = (
    "9d293d3bf1f86a682bb61deee686a103f7e77906769b942d5fc43da69d65bf78"
)
EXPECTED_QUALIFIED_ROWS = 5_060
EXPECTED_QUALIFIED_FRAMES = 2_811_006
WIDTH = 640
HEIGHT = 360
JPEG_QUALITY = 90
HEAD_CAMERA_PRESET = "current"
# Measured on the 128-row, 16-env native EGL benchmark at Client 2dcd479.
# Signed-distance cost scales with object collision complexity, so frame count
# alone creates 3.5x shard stragglers. These conservative seconds/frame weights
# drive contiguous, object-aware load balancing; they do not affect row data.
OBJECT_SECONDS_PER_FRAME = {
    "banana": 0.0045,
    "bowl": 0.0200,
    "cube1": 0.0045,
    "cube2": 0.0045,
    "cylinder3": 0.0045,
    "cylinder4": 0.0045,
    "cylinder7": 0.0045,
    "iphone": 0.0055,
    "largeclamp": 0.0175,
    "mayonnaisebottle": 0.0047,
    "powerdrill": 0.0175,
}
SOURCE_COLUMNS = [
    "index", "trajectory_metadata", "hands", "reference", "provenance"
]


@dataclass
class RenderScene:
    model: Any
    data: Any
    renderer: Any
    object_addr: int
    hand_addresses: np.ndarray
    object_body_id: int
    feature_ids: physics.State46FeatureIds


_SCENES: dict[str, RenderScene] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def release_code_identity() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[2]
    return {
        "client_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "release_script_sha256": sha256(Path(__file__).resolve()),
        "state_contract_sha256": sha256(
            repo / "scripts" / "mano_state46_contract.py"
        ),
        "physics_adapter_sha256": sha256(
            repo / "scripts" / "eval" / "manorl_native_physics.py"
        ),
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, suffix=".json.tmp", encoding="utf-8"
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def quality_records(quality_root: Path) -> list[dict[str, Any]]:
    summary = quality_root / "global_grade_summary.json"
    if sha256(summary) != EXPECTED_QUALITY_SUMMARY_SHA256:
        raise ValueError(f"quality summary SHA mismatch: {summary}")
    records = [
        json.loads(path.read_text())
        for path in quality_root.glob("objects/*/records/*.json")
    ]
    records.sort(key=lambda value: int(value["row_index"]))
    if len(records) != replay_quality.EXPECTED_ROWS:
        raise ValueError(f"quality population mismatch: {len(records)}")
    if any(value.get("status") != "ok" for value in records):
        raise ValueError("quality population contains non-ok records")
    qualified = [value for value in records if value.get("grade") in ("A", "B")]
    if len(qualified) != EXPECTED_QUALIFIED_ROWS:
        raise ValueError(f"qualified population mismatch: {len(qualified)}")
    if sum(int(value["frames"]) for value in qualified) != EXPECTED_QUALIFIED_FRAMES:
        raise ValueError("qualified frame population mismatch")
    if len({value["row_uuid"] for value in qualified}) != len(qualified):
        raise ValueError("qualified UUIDs are not unique")
    return qualified


def representative_subset(
    records: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    if count <= 0 or count >= len(records):
        return records
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["object"])].append(record)
    allocation = {name: min(2, len(values)) for name, values in grouped.items()}
    remaining = count - sum(allocation.values())
    if remaining < 0:
        raise ValueError(f"benchmark count {count} cannot cover {len(grouped)} objects")
    weights = {
        name: sum(int(record["frames"]) for record in values)
        for name, values in grouped.items()
    }
    total_weight = sum(weights.values())
    while remaining:
        best = max(
            grouped,
            key=lambda name: (
                weights[name] / total_weight * count - allocation[name],
                -allocation[name],
                name,
            ),
        )
        if allocation[best] >= len(grouped[best]):
            weights[best] = 0
            total_weight = max(1, sum(weights.values()))
            continue
        allocation[best] += 1
        remaining -= 1
    selected: list[dict[str, Any]] = []
    for name, values in sorted(grouped.items()):
        amount = allocation[name]
        indices = np.linspace(0, len(values) - 1, amount).round().astype(int)
        selected.extend(values[index] for index in indices)
    selected.sort(key=lambda value: int(value["row_index"]))
    if len(selected) != count or len({value["row_uuid"] for value in selected}) != count:
        raise RuntimeError("representative benchmark selection is not exact")
    return selected


def estimated_render_seconds(record: dict[str, Any]) -> float:
    object_name = str(record.get("object"))
    try:
        coefficient = OBJECT_SECONDS_PER_FRAME[object_name]
    except KeyError as exc:
        raise ValueError(f"no measured render cost for object {object_name!r}") from exc
    return int(record["frames"]) * coefficient


def contiguous_weighted_shards(
    records: list[dict[str, Any]], shard_count: int
) -> list[list[dict[str, Any]]]:
    if shard_count <= 0 or shard_count > len(records):
        raise ValueError(f"invalid shard count {shard_count}")
    shards: list[list[dict[str, Any]]] = []
    cursor = 0
    remaining_weight = sum(estimated_render_seconds(value) for value in records)
    for shard_index in range(shard_count):
        shards_left = shard_count - shard_index
        rows_left = len(records) - cursor
        if shards_left == 1:
            shard = records[cursor:]
            shards.append(shard)
            break
        target = remaining_weight / shards_left
        shard: list[dict[str, Any]] = []
        weight = 0
        max_take = rows_left - (shards_left - 1)
        while len(shard) < max_take:
            candidate = records[cursor]
            candidate_weight = estimated_render_seconds(candidate)
            if shard and abs(weight - target) <= abs(weight + candidate_weight - target):
                break
            shard.append(candidate)
            cursor += 1
            weight += candidate_weight
        if not shard:
            shard.append(records[cursor])
            cursor += 1
            weight = estimated_render_seconds(shard[0])
        shards.append(shard)
        remaining_weight -= weight
    if [record["row_uuid"] for shard in shards for record in shard] != [
        record["row_uuid"] for record in records
    ]:
        raise RuntimeError("weighted shards changed qualified order")
    return shards


def write_plan(
    *, quality_root: Path, output_root: Path, shard_count: int, benchmark_rows: int
) -> Path:
    repo = Path(__file__).resolve().parents[2]
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    ).strip()
    if status:
        raise RuntimeError(
            f"release plan requires a clean Client checkout: {status.splitlines()[:8]}"
        )
    if output_root.exists():
        raise FileExistsError(f"refusing existing release root: {output_root}")
    records = representative_subset(quality_records(quality_root), benchmark_rows)
    shards = contiguous_weighted_shards(records, shard_count)
    payload: dict[str, Any] = {
        "contract": RELEASE_CONTRACT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quality_root": str(quality_root.resolve()),
        "quality_summary_sha256": EXPECTED_QUALITY_SUMMARY_SHA256,
        "source_dataset": replay_quality.DEFAULT_DATASET,
        "source_dataset_version": replay_quality.EXPECTED_DATASET_VERSION,
        "population": {
            "rows": len(records),
            "frames": sum(int(value["frames"]) for value in records),
            "estimated_render_seconds": sum(
                estimated_render_seconds(value) for value in records
            ),
            "render_cost_contract": "measured_object_seconds_per_frame_128row_16env_v1",
            "benchmark_rows": benchmark_rows,
            "qualified_grades": ["A", "B"],
        },
        "render": {
            "width": WIDTH,
            "height": HEIGHT,
            "jpeg_quality": JPEG_QUALITY,
            "head_camera_preset": HEAD_CAMERA_PRESET,
            "head_camera": cameras.head_camera_config(HEAD_CAMERA_PRESET),
            "wrist_camera": cameras.WRIST_CAMERA,
            "dynamics_steps_during_render": 0,
        },
        "state_contract": state46.STATE46_CONTRACT_ID,
        "action_contract": state46.ACTION32_CONTRACT_ID,
        "code": release_code_identity(),
        "shards": [],
    }
    for shard_index, values in enumerate(shards):
        entries = [
            {
                "filtered_row_index": int(value["row_index"]),
                "original_merged_row_index": int(value["original_merged_row_index"]),
                "row_uuid": str(value["row_uuid"]),
                "seed_uuid": str(value["seed_uuid"]),
                "object": str(value["object"]),
                "gesture": str(value["gesture"]),
                "grade": str(value["grade"]),
                "frames": int(value["frames"]),
                "trace_npz": str(value["trace_npz"]),
                "trace_sha256": str(value["trace_sha256"]),
                "quality_report": str(
                    quality_root
                    / "objects"
                    / str(value["object"])
                    / "records"
                    / f"{int(value['row_index']):05d}.json"
                ),
            }
            for value in values
        ]
        payload["shards"].append(
            {
                "shard_index": shard_index,
                "rows": len(entries),
                "frames": sum(value["frames"] for value in entries),
                "estimated_render_seconds": sum(
                    estimated_render_seconds(value) for value in entries
                ),
                "filtered_row_min": entries[0]["filtered_row_index"],
                "filtered_row_max": entries[-1]["filtered_row_index"],
                "object_counts": dict(Counter(value["object"] for value in entries)),
                "entries": entries,
            }
        )
    payload["population"]["uuid_sha256"] = canonical_sha256(
        [value["row_uuid"] for value in records]
    )
    output_root.mkdir(parents=True)
    path = output_root / "release_plan.json"
    atomic_json(path, payload)
    print(path)
    return path


def _vector32():
    import pyarrow as pa

    return pa.list_(pa.float32(), state46.ACTION_DIM)


def _vector46():
    import pyarrow as pa

    return pa.list_(pa.float32(), state46.STATE_DIM)


def release_schema():
    import pyarrow as pa

    vec3 = pa.list_(pa.float32(), 3)
    vec4 = pa.list_(pa.float32(), 4)
    vec5 = pa.list_(pa.float32(), 5)
    vec10 = pa.list_(pa.float32(), 10)
    vec28 = pa.list_(pa.float32(), 28)
    pair = pa.struct(
        [
            ("finger", pa.string()),
            ("hand_geom", pa.string()),
            ("object_geom", pa.string()),
            ("position_world", vec3),
            ("normal_force_world", vec3),
            ("normal_force_norm", pa.float32()),
            ("signed_distance_m", pa.float32()),
        ]
    )
    return pa.schema(
        [
            (
                "index",
                pa.struct(
                    [
                        ("uuid", pa.string()),
                        ("seed_uuid", pa.string()),
                        ("object", pa.string()),
                        ("gesture", pa.string()),
                        ("filtered_row_index", pa.int64()),
                        ("original_merged_row_index", pa.int64()),
                        ("is_generated", pa.bool_()),
                        ("grade", pa.string()),
                        ("forward_aligned_grade", pa.string()),
                    ]
                ),
            ),
            ("timestamp", pa.list_(pa.float64())),
            ("state", pa.list_(_vector46())),
            ("actions", pa.list_(_vector32())),
            ("image", pa.list_(pa.large_binary())),
            ("wrist_image", pa.list_(pa.large_binary())),
            ("prompt", pa.string()),
            (
                "hands",
                pa.list_(
                    pa.struct(
                        [
                            ("hand_name", pa.string()),
                            ("urdf_dof", pa.list_(vec28)),
                            ("urdf_dof_target", pa.list_(vec28)),
                        ]
                    )
                ),
            ),
            (
                "objects",
                pa.list_(
                    pa.struct(
                        [
                            ("object_name", pa.string()),
                            ("pos", pa.list_(vec3)),
                            ("rot_aa", pa.list_(vec3)),
                            ("quat_wxyz", pa.list_(vec4)),
                        ]
                    )
                ),
            ),
            (
                "contact",
                pa.list_(
                    pa.struct(
                        [
                            ("finger_contacts", vec5),
                            ("object_floor", pa.bool_()),
                            ("pair_count", pa.int32()),
                            ("pairs", pa.list_(pair)),
                        ]
                    )
                ),
            ),
            (
                "reference",
                pa.struct(
                    [
                        ("recorded_hand_urdf_dof", pa.list_(vec28)),
                        ("recorded_object_pos", pa.list_(vec3)),
                        ("recorded_object_quat_wxyz", pa.list_(vec4)),
                        ("quality_sampled_object_pos", pa.list_(vec3)),
                        ("quality_sampled_object_quat_wxyz", pa.list_(vec4)),
                        ("forward_alignment_position_delta_m", pa.list_(pa.float32())),
                        ("forward_alignment_rotation_delta_rad", pa.list_(pa.float32())),
                        ("qpos_error", pa.list_(pa.float32())),
                        ("object_position_error", pa.list_(pa.float32())),
                        ("object_rotation_error", pa.list_(pa.float32())),
                        ("trace_npz", pa.string()),
                        ("trace_sha256", pa.string()),
                        ("quality_report", pa.string()),
                        ("quality_report_sha256", pa.string()),
                    ]
                ),
            ),
            (
                "trajectory_metadata",
                pa.struct(
                    [
                        ("data_fps", pa.int32()),
                        ("total_frames", pa.int32()),
                        ("gesture", pa.string()),
                        ("hand_names", pa.list_(pa.string())),
                        ("hand_slots", pa.list_(pa.string())),
                        ("object_names", pa.list_(pa.string())),
                        ("mano_hand_shapes", pa.list_(vec10)),
                    ]
                ),
            ),
            (
                "episode_metadata",
                pa.struct(
                    [
                        ("fps", pa.int32()),
                        ("frames", pa.int32()),
                        ("transitions", pa.int32()),
                        ("object", pa.string()),
                        ("gesture", pa.string()),
                        ("grade", pa.string()),
                        ("qualified", pa.bool_()),
                        ("forward_aligned_grade", pa.string()),
                    ]
                ),
            ),
            (
                "provenance",
                pa.struct(
                    [
                        ("contract", pa.string()),
                        ("state_contract", pa.string()),
                        ("action_contract", pa.string()),
                        ("replay_contract", pa.string()),
                        ("client_commit", pa.string()),
                        ("replay_client_commit", pa.string()),
                        ("release_script_sha256", pa.string()),
                        ("state_contract_sha256", pa.string()),
                        ("physics_adapter_sha256", pa.string()),
                        ("manorl_commit", pa.string()),
                        ("asset_source_commit", pa.string()),
                        ("source_contract", pa.string()),
                        ("source_identity", pa.string()),
                        ("source_dataset", pa.string()),
                        ("source_dataset_version", pa.int32()),
                        ("quality_summary_sha256", pa.string()),
                        ("release_plan_sha256", pa.string()),
                        ("render_width", pa.int32()),
                        ("render_height", pa.int32()),
                        ("jpeg_quality", pa.int32()),
                        ("head_camera_preset", pa.string()),
                        ("dynamics_steps_during_render", pa.int32()),
                        ("derived_state_alignment", pa.string()),
                        ("source_trajectory_metadata_json", pa.string()),
                        ("source_provenance_json", pa.string()),
                    ]
                ),
            ),
            ("frame_count", pa.int32()),
            ("image_count", pa.int32()),
            ("image_bytes", pa.int64()),
            ("row_payload_sha256", pa.string()),
        ]
    )


def scene(object_name: str) -> RenderScene:
    cached = _SCENES.get(object_name)
    if cached is not None:
        return cached
    import mujoco

    (
        _, model, data, renderer, object_addr, _, hand_addresses, _, _
    ) = physics.make_scene(
        object_name,
        WIDTH,
        HEIGHT,
        physics=True,
        create_renderer=True,
        head_camera_preset=HEAD_CAMERA_PRESET,
    )
    if renderer is None:
        raise RuntimeError("release scene has no renderer")
    result = RenderScene(
        model=model,
        data=data,
        renderer=renderer,
        object_addr=int(object_addr),
        hand_addresses=np.asarray(hand_addresses, dtype=np.int64),
        object_body_id=physics.object_body_id(model, object_name),
        feature_ids=physics.resolve_state46_feature_ids(model, object_name),
    )
    _SCENES[object_name] = result
    return result


def close_scenes() -> None:
    for value in _SCENES.values():
        value.renderer.close()
    _SCENES.clear()


def quaternion_wxyz_to_axis_angle(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"quaternions must have shape [T,4], got {values.shape}")
    values = values / np.linalg.norm(values, axis=1, keepdims=True)
    values = np.where(values[:, :1] < 0.0, -values, values)
    xyz_norm = np.linalg.norm(values[:, 1:], axis=1)
    angles = 2.0 * np.arctan2(xyz_norm, np.clip(values[:, 0], -1.0, 1.0))
    result = np.zeros((len(values), 3), dtype=np.float64)
    active = xyz_norm > 1e-12
    result[active] = values[active, 1:] / xyz_norm[active, None] * angles[active, None]
    return result.astype(np.float32)


def encode_jpeg(frame: np.ndarray) -> bytes:
    from PIL import Image

    buffer = BytesIO()
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
        buffer, format="JPEG", quality=JPEG_QUALITY
    )
    return buffer.getvalue()


def render_release_row(
    entry: dict[str, Any],
    source: dict[str, Any],
    *,
    plan_sha: str,
    release_code: dict[str, str] | None = None,
) -> dict[str, Any]:
    import mujoco

    release_code = release_code or release_code_identity()
    source_index = source.get("index") or {}
    source_meta = source.get("trajectory_metadata") or {}
    source_provenance = source.get("provenance") or {}
    if (
        str(source_index.get("uuid")) != entry["row_uuid"]
        or str(source_index.get("seed_uuid")) != entry["seed_uuid"]
        or str(source_index.get("scene")) != entry["object"]
        or str(source_meta.get("gesture")) != entry["gesture"]
    ):
        raise ValueError(f"release source identity mismatch for {entry['row_uuid']}")
    trace_path = Path(entry["trace_npz"])
    if sha256(trace_path) != entry["trace_sha256"]:
        raise ValueError(f"quality trace SHA mismatch: {trace_path}")
    quality_report = Path(entry["quality_report"])
    quality = json.loads(quality_report.read_text())
    if quality.get("row_uuid") != entry["row_uuid"] or quality.get("grade") != entry["grade"]:
        raise ValueError("quality report identity mismatch")
    with np.load(trace_path) as trace:
        arrays = {name: np.asarray(trace[name]) for name in trace.files}
    frame_count = int(entry["frames"])
    required = {
        "timestamp": (frame_count,),
        "simulated_time": (frame_count,),
        "simulated_full_qpos": (frame_count, 35),
        "simulated_full_qvel": (frame_count, 34),
        "simulated_hand_qpos": (frame_count, 28),
        "source_target_qpos": (frame_count, 28),
        "recorded_hand_qpos": (frame_count, 28),
        "simulated_object_position": (frame_count, 3),
        "simulated_object_quaternion": (frame_count, 4),
        "recorded_object_position": (frame_count, 3),
        "recorded_object_quaternion": (frame_count, 4),
        "qpos_error": (frame_count,),
        "object_position_error": (frame_count,),
        "object_rotation_error": (frame_count,),
    }
    for name, shape in required.items():
        if arrays.get(name) is None or arrays[name].shape != shape:
            raise ValueError(f"trace {name} shape mismatch: {getattr(arrays.get(name), 'shape', None)}")
        if not np.isfinite(arrays[name]).all():
            raise FloatingPointError(f"trace {name} is non-finite")
    state46.validate_timestamps(arrays["timestamp"], frame_count)
    actions = state46.absolute_target_actions32(arrays["source_target_qpos"])
    visual = scene(entry["object"])
    data = visual.data
    contacts = np.empty((frame_count, 5), dtype=np.float32)
    surface = np.empty_like(contacts)
    radial = np.empty_like(contacts)
    floor = np.empty(frame_count, dtype=np.float32)
    sim_position = np.empty((frame_count, 3), dtype=np.float32)
    sim_quaternion = np.empty((frame_count, 4), dtype=np.float32)
    contact_frames: list[dict[str, Any]] = []
    head_images: list[bytes] = []
    wrist_images: list[bytes] = []
    payload = hashlib.sha256()
    for frame in range(frame_count):
        data.qpos[:] = arrays["simulated_full_qpos"][frame]
        data.qvel[:] = arrays["simulated_full_qvel"][frame]
        data.time = float(arrays["simulated_time"][frame])
        data.ctrl[:] = (
            0.0 if frame == 0 else arrays["source_target_qpos"][frame - 1]
        )
        mujoco.mj_forward(visual.model, data)
        if not np.allclose(data.qpos, arrays["simulated_full_qpos"][frame], rtol=0, atol=1e-7):
            raise RuntimeError("mj_forward changed restored qpos")
        sim_position[frame] = data.xpos[visual.object_body_id]
        sim_quaternion[frame] = data.xquat[visual.object_body_id]
        restored_position = data.qpos[
            visual.object_addr : visual.object_addr + 3
        ]
        restored_quaternion = data.qpos[
            visual.object_addr + 3 : visual.object_addr + 7
        ]
        position_delta = float(np.max(np.abs(sim_position[frame] - restored_position)))
        quaternion_dot = float(np.abs(np.dot(sim_quaternion[frame], restored_quaternion)))
        if position_delta > 2e-7 or abs(quaternion_dot - 1.0) > 2e-7:
            raise RuntimeError(
                "mj_forward derived object pose does not match restored freejoint: "
                f"frame={frame} position_delta={position_delta} "
                f"quaternion_dot={quaternion_dot}"
            )
        (
            contacts[frame], surface[frame], radial[frame], floor[frame], pairs
        ) = physics.state46_features_from_mujoco(
            visual.model, data, entry["object"], feature_ids=visual.feature_ids
        )
        contact_frames.append(
            {
                "finger_contacts": contacts[frame].tolist(),
                "object_floor": bool(floor[frame] > 0.5),
                "pair_count": len(pairs),
                "pairs": pairs,
            }
        )
        visual.renderer.update_scene(data, camera=cameras.HEAD_CAMERA_NAME)
        head = encode_jpeg(visual.renderer.render())
        visual.renderer.update_scene(data, camera=cameras.WRIST_CAMERA_NAME)
        wrist = encode_jpeg(visual.renderer.render())
        head_images.append(head)
        wrist_images.append(wrist)
        payload.update(head)
        payload.update(wrist)
    sampled_position = arrays["simulated_object_position"].astype(np.float32)
    sampled_quaternion = arrays["simulated_object_quaternion"].astype(np.float32)
    forward_alignment_position = np.linalg.norm(
        sim_position - sampled_position, axis=1
    ).astype(np.float32)
    forward_alignment_rotation = (
        2.0
        * np.arccos(
            np.clip(
                np.abs(np.sum(sim_quaternion * sampled_quaternion, axis=1)),
                -1.0,
                1.0,
            )
        )
    ).astype(np.float32)
    forward_object_error = np.linalg.norm(
        sim_position - arrays["recorded_object_position"], axis=1
    )
    forward_aligned_grade = replay_quality.grade_from_max_error(
        float(np.max(forward_object_error[1:]))
    )
    lift = sim_position[:, 2] - sim_position[0, 2]
    state = state46.assemble_state46_sequence(
        hand_qpos=arrays["simulated_hand_qpos"],
        contacts=contacts,
        object_lift=lift,
        signed_surface_distances=surface,
        radial_distances=radial,
        floor_support=floor,
    )
    payload.update(state.tobytes(order="C"))
    payload.update(actions.tobytes(order="C"))
    payload.update(sim_position.tobytes(order="C"))
    payload.update(sim_quaternion.tobytes(order="C"))
    image_bytes = sum(map(len, head_images)) + sum(map(len, wrist_images))
    source_shapes = source_meta.get("mano_hand_shapes") or []
    right_shape = source_shapes[:1]
    source_identity = str(source_provenance.get("source_identity"))
    if source_identity != quality.get("source_identity"):
        raise ValueError("release source provenance identity mismatch")
    provenance = quality["provenance"]
    result = {
        "index": {
            "uuid": entry["row_uuid"],
            "seed_uuid": entry["seed_uuid"],
            "object": entry["object"],
            "gesture": entry["gesture"],
            "filtered_row_index": entry["filtered_row_index"],
            "original_merged_row_index": entry["original_merged_row_index"],
            "is_generated": bool(source_index.get("is_generated")),
            "grade": entry["grade"],
            "forward_aligned_grade": forward_aligned_grade,
        },
        "timestamp": arrays["timestamp"].astype(np.float64).tolist(),
        "state": state.tolist(),
        "actions": actions.tolist(),
        "image": head_images,
        "wrist_image": wrist_images,
        "prompt": f"pick up the {entry['object']} using gesture {entry['gesture']}",
        "hands": [
            {
                "hand_name": "right",
                "urdf_dof": arrays["simulated_hand_qpos"].astype(np.float32).tolist(),
                "urdf_dof_target": arrays["source_target_qpos"].astype(np.float32).tolist(),
            }
        ],
        "objects": [
            {
                "object_name": entry["object"],
                "pos": sim_position.tolist(),
                "rot_aa": quaternion_wxyz_to_axis_angle(sim_quaternion).tolist(),
                "quat_wxyz": sim_quaternion.tolist(),
            }
        ],
        "contact": contact_frames,
        "reference": {
            "recorded_hand_urdf_dof": arrays["recorded_hand_qpos"].astype(np.float32).tolist(),
            "recorded_object_pos": arrays["recorded_object_position"].astype(np.float32).tolist(),
            "recorded_object_quat_wxyz": arrays["recorded_object_quaternion"].astype(np.float32).tolist(),
            "quality_sampled_object_pos": sampled_position.tolist(),
            "quality_sampled_object_quat_wxyz": sampled_quaternion.tolist(),
            "forward_alignment_position_delta_m": forward_alignment_position.tolist(),
            "forward_alignment_rotation_delta_rad": forward_alignment_rotation.tolist(),
            "qpos_error": arrays["qpos_error"].astype(np.float32).tolist(),
            "object_position_error": arrays["object_position_error"].astype(np.float32).tolist(),
            "object_rotation_error": arrays["object_rotation_error"].astype(np.float32).tolist(),
            "trace_npz": str(trace_path),
            "trace_sha256": entry["trace_sha256"],
            "quality_report": str(quality_report),
            "quality_report_sha256": sha256(quality_report),
        },
        "trajectory_metadata": {
            "data_fps": 200,
            "total_frames": frame_count,
            "gesture": entry["gesture"],
            "hand_names": ["right"],
            "hand_slots": ["right"],
            "object_names": [entry["object"]],
            "mano_hand_shapes": right_shape,
        },
        "episode_metadata": {
            "fps": 200,
            "frames": frame_count,
            "transitions": frame_count - 1,
            "object": entry["object"],
            "gesture": entry["gesture"],
            "grade": entry["grade"],
            "qualified": True,
            "forward_aligned_grade": forward_aligned_grade,
        },
        "provenance": {
            "contract": RELEASE_CONTRACT,
            "state_contract": state46.STATE46_CONTRACT_ID,
            "action_contract": state46.ACTION32_CONTRACT_ID,
            "replay_contract": replay_quality.CONTRACT,
            "client_commit": release_code["client_commit"],
            "replay_client_commit": provenance["client_commit"],
            "release_script_sha256": release_code["release_script_sha256"],
            "state_contract_sha256": release_code["state_contract_sha256"],
            "physics_adapter_sha256": release_code["physics_adapter_sha256"],
            "manorl_commit": provenance["manorl"]["commit"],
            "asset_source_commit": provenance["manorl"]["all_assets_commit"],
            "source_contract": replay_quality.SOURCE_CONTRACT,
            "source_identity": source_identity,
            "source_dataset": provenance["source_dataset"],
            "source_dataset_version": int(provenance["source_dataset_version"]),
            "quality_summary_sha256": EXPECTED_QUALITY_SUMMARY_SHA256,
            "release_plan_sha256": plan_sha,
            "render_width": WIDTH,
            "render_height": HEIGHT,
            "jpeg_quality": JPEG_QUALITY,
            "head_camera_preset": HEAD_CAMERA_PRESET,
            "dynamics_steps_during_render": 0,
            "derived_state_alignment": "restore_full_qpos_qvel_then_mj_forward_v1",
            "source_trajectory_metadata_json": json.dumps(source_meta, sort_keys=True),
            "source_provenance_json": json.dumps(source_provenance, sort_keys=True),
        },
        "frame_count": frame_count,
        "image_count": 2 * frame_count,
        "image_bytes": image_bytes,
        "row_payload_sha256": payload.hexdigest(),
    }
    if len(head_images) != frame_count or len(wrist_images) != frame_count:
        raise RuntimeError("release JPEG population mismatch")
    return result


def _load_plan(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text())
    if payload.get("contract") != RELEASE_CONTRACT:
        raise ValueError(f"release plan contract mismatch: {path}")
    planned_code = payload.get("code") or {}
    current_code = release_code_identity()
    for field in (
        "release_script_sha256", "state_contract_sha256", "physics_adapter_sha256"
    ):
        if planned_code.get(field) != current_code[field]:
            raise ValueError(
                f"release code identity mismatch for {field}: "
                f"plan={planned_code.get(field)} current={current_code[field]}"
            )
    return payload, sha256(path)


def _shard_path(output_root: Path, shard_index: int) -> Path:
    return output_root / "shards" / f"shard-{shard_index:03d}.lance"


def existing_prefix(path: Path, entries: list[dict[str, Any]], plan_sha: str) -> int:
    if not path.exists():
        return 0
    import lance

    dataset = lance.dataset(str(path))
    rows = dataset.to_table(columns=["index", "provenance", "frame_count"]).to_pylist()
    if len(rows) > len(entries):
        raise ValueError(f"existing shard has excess rows: {path}")
    for offset, row in enumerate(rows):
        if (
            row["index"]["uuid"] != entries[offset]["row_uuid"]
            or row["index"]["filtered_row_index"] != entries[offset]["filtered_row_index"]
            or row["provenance"]["release_plan_sha256"] != plan_sha
            or row["frame_count"] != entries[offset]["frames"]
        ):
            raise ValueError(f"existing shard prefix mismatch at {path}:{offset}")
    return len(rows)


def run_shard(plan_path: Path, shard_index: int, batch_size: int) -> Path:
    import lance
    import pyarrow as pa

    plan, plan_sha = _load_plan(plan_path)
    output_root = plan_path.parent
    try:
        shard = plan["shards"][shard_index]
    except IndexError as exc:
        raise ValueError(f"shard index {shard_index} is outside plan") from exc
    if int(shard["shard_index"]) != shard_index:
        raise ValueError("shard plan index mismatch")
    entries = list(shard["entries"])
    target = _shard_path(output_root, shard_index)
    target.parent.mkdir(parents=True, exist_ok=True)
    completed = existing_prefix(target, entries, plan_sha)
    dataset = lance.dataset(
        str(plan["source_dataset"]), version=int(plan["source_dataset_version"])
    )
    started = time.monotonic()
    try:
        for batch_start in range(completed, len(entries), batch_size):
            batch_entries = entries[batch_start : batch_start + batch_size]
            source_rows = dataset.take(
                [value["filtered_row_index"] for value in batch_entries],
                columns=SOURCE_COLUMNS,
            ).to_pylist()
            rendered = [
                render_release_row(
                    entry, source, plan_sha=plan_sha, release_code=plan["code"]
                )
                for entry, source in zip(batch_entries, source_rows, strict=True)
            ]
            table = pa.Table.from_pylist(rendered, schema=release_schema())
            lance.write_dataset(
                table, str(target), mode="append" if target.exists() else "create"
            )
            completed = existing_prefix(target, entries, plan_sha)
            progress = {
                "contract": RELEASE_CONTRACT,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "shard_index": shard_index,
                "completed_rows": completed,
                "total_rows": len(entries),
                "completed_frames": sum(value["frames"] for value in entries[:completed]),
                "total_frames": shard["frames"],
                "elapsed_seconds": time.monotonic() - started,
                "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "egl_device": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
            }
            atomic_json(output_root / "progress" / f"shard-{shard_index:03d}.json", progress)
            print(json.dumps(progress, sort_keys=True), flush=True)
    except Exception as exc:
        atomic_json(
            output_root / "failures" / f"shard-{shard_index:03d}.json",
            {
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "shard_index": shard_index,
                "completed_rows": completed,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    finally:
        close_scenes()
    verify_shard(plan_path, shard_index)
    return target


def verify_shard(plan_path: Path, shard_index: int) -> dict[str, Any]:
    import lance

    plan, plan_sha = _load_plan(plan_path)
    shard = plan["shards"][shard_index]
    entries = shard["entries"]
    path = _shard_path(plan_path.parent, shard_index)
    dataset = lance.dataset(str(path))
    rows = dataset.to_table(
        columns=["index", "provenance", "frame_count", "image_count", "image_bytes", "row_payload_sha256"]
    ).to_pylist()
    if len(rows) != len(entries):
        raise ValueError(f"shard {shard_index} row count mismatch")
    if [row["index"]["uuid"] for row in rows] != [value["row_uuid"] for value in entries]:
        raise ValueError(f"shard {shard_index} UUID order mismatch")
    if sum(int(row["frame_count"]) for row in rows) != int(shard["frames"]):
        raise ValueError(f"shard {shard_index} frame count mismatch")
    if any(
        row["provenance"]["release_plan_sha256"] != plan_sha
        or row["image_count"] != 2 * row["frame_count"]
        or len(row["row_payload_sha256"]) != 64
        for row in rows
    ):
        raise ValueError(f"shard {shard_index} contract mismatch")
    verification = {
        "contract": RELEASE_CONTRACT,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "shard_index": shard_index,
        "rows": len(rows),
        "frames": sum(int(row["frame_count"]) for row in rows),
        "images": sum(int(row["image_count"]) for row in rows),
        "image_bytes": sum(int(row["image_bytes"]) for row in rows),
        "uuid_order_verified": True,
        "path": str(path),
    }
    atomic_json(
        plan_path.parent / "verifications" / f"shard-{shard_index:03d}.json",
        verification,
    )
    return verification


def aggregate_release(plan_path: Path, overwrite: bool) -> Path:
    import lance

    plan, plan_sha = _load_plan(plan_path)
    output_root = plan_path.parent
    target = output_root / "mano_28d_native_replay_state46_rgb_v1.lance"
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"final release exists: {target}")
        shutil.rmtree(target)
    all_entries = [entry for shard in plan["shards"] for entry in shard["entries"]]
    mode = "create"
    shard_verifications = []
    for shard in plan["shards"]:
        index = int(shard["shard_index"])
        shard_verifications.append(verify_shard(plan_path, index))
        source = lance.dataset(str(_shard_path(output_root, index)))
        lance.write_dataset(source.scanner().to_reader(), str(target), mode=mode)
        mode = "append"
    dataset = lance.dataset(str(target))
    light = dataset.to_table(
        columns=["index", "provenance", "frame_count", "image_count", "image_bytes"]
    ).to_pylist()
    expected_uuid = [entry["row_uuid"] for entry in all_entries]
    if [row["index"]["uuid"] for row in light] != expected_uuid:
        raise ValueError("final release UUID order mismatch")
    if any(row["provenance"]["release_plan_sha256"] != plan_sha for row in light):
        raise ValueError("final release provenance mismatch")
    totals = {
        "rows": len(light),
        "frames": sum(int(row["frame_count"]) for row in light),
        "images": sum(int(row["image_count"]) for row in light),
        "image_bytes": sum(int(row["image_bytes"]) for row in light),
    }
    if totals["rows"] != int(plan["population"]["rows"]) or totals["frames"] != int(plan["population"]["frames"]):
        raise ValueError(f"final release population mismatch: {totals}")
    verification = {
        "contract": RELEASE_CONTRACT,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "release": str(target),
        "release_plan": str(plan_path),
        "release_plan_sha256": plan_sha,
        "uuid_order_verified": True,
        "shards": len(plan["shards"]),
        **totals,
        "client_commit": plan["code"]["client_commit"],
        "state_contract": state46.STATE46_CONTRACT_ID,
        "action_contract": state46.ACTION32_CONTRACT_ID,
    }
    atomic_json(output_root / "release_verification.json", verification)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return target


def run_all(plan_path: Path, gpu_count: int) -> None:
    plan, _ = _load_plan(plan_path)
    output_root = plan_path.parent
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[int, Any, Any]] = []
    for shard in plan["shards"]:
        index = int(shard["shard_index"])
        gpu = index % gpu_count
        environment = os.environ.copy()
        environment.update(
            MUJOCO_GL="egl",
            CUDA_VISIBLE_DEVICES=str(gpu),
            MUJOCO_EGL_DEVICE_ID=str(gpu),
        )
        log = (log_root / f"shard-{index:03d}.log").open("a")
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "run-shard",
                "--plan",
                str(plan_path),
                "--shard-index",
                str(index),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        processes.append((index, process, log))
    failures = []
    for index, process, log in processes:
        return_code = process.wait()
        log.close()
        if return_code:
            failures.append((index, return_code))
    if failures:
        raise RuntimeError(f"release shard failures: {failures}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--quality-root", type=Path, default=DEFAULT_QUALITY_ROOT)
    plan.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    plan.add_argument("--shards", type=int, default=16)
    plan.add_argument("--benchmark-rows", type=int, default=0)
    run = subparsers.add_parser("run-shard")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--shard-index", type=int, required=True)
    run.add_argument("--batch-size", type=int, default=4)
    verify = subparsers.add_parser("verify-shard")
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--shard-index", type=int, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--plan", type=Path, required=True)
    aggregate.add_argument("--overwrite", action="store_true")
    launch = subparsers.add_parser("run-all")
    launch.add_argument("--plan", type=Path, required=True)
    launch.add_argument("--gpu-count", type=int, default=8)
    args = parser.parse_args()
    if args.command == "plan":
        write_plan(
            quality_root=args.quality_root.resolve(),
            output_root=args.output_root.resolve(),
            shard_count=args.shards,
            benchmark_rows=args.benchmark_rows,
        )
    elif args.command == "run-shard":
        run_shard(args.plan.resolve(), args.shard_index, args.batch_size)
    elif args.command == "verify-shard":
        print(json.dumps(verify_shard(args.plan.resolve(), args.shard_index), indent=2))
    elif args.command == "aggregate":
        aggregate_release(args.plan.resolve(), args.overwrite)
    else:
        run_all(args.plan.resolve(), args.gpu_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
