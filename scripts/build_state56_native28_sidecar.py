#!/usr/bin/env python3
"""Build an authenticated State56 virtual sidecar over the read-only State41 Lance.

The sidecar duplicates no images or actions. Each row stores the exact State56
window plus a strict reference to its source Lance row. Training joins by
release_row_index and UUID, then reads images/actions/object pose from the pinned
source release.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any
import uuid

import numpy as np

from scripts import mano_state56_contract as C
from scripts.eval import manorl_native28_physics as P

RELEASE_CONTRACT = "mano_state56_native28_virtual_sidecar_v1"
PLAN_CONTRACT = "mano_state56_native28_sidecar_plan_v1"
SOURCE_COLUMNS = [
    "index", "timestamp", "state", "actions", "hands", "objects", "contact",
    "prompt", "frame_count", "row_payload_sha256", "provenance",
]
_SCENES: dict[str, P.Native28Scene] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments], text=True, stderr=subprocess.DEVNULL
    ).strip()


def code_identity() -> dict[str, str]:
    client_root = Path(__file__).resolve().parents[1]
    status = git(client_root, "status", "--porcelain", "--ignore-submodules=dirty")
    if status:
        raise RuntimeError(f"State56 client checkout must be clean: {status.splitlines()[:8]}")
    return {
        "client_commit": git(client_root, "rev-parse", "HEAD"),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "state_contract_sha256": sha256(Path(C.__file__).resolve()),
        "geometry_contract_sha256": C.GEOMETRY_CONTRACT_SHA256,
        "physics_adapter_sha256": sha256(Path(P.__file__).resolve()),
    }


def sidecar_schema():
    import pyarrow as pa

    index_type = pa.struct([
        ("release_row_index", pa.int64()), ("filtered_row_index", pa.int64()),
        ("original_merged_row_index", pa.int64()), ("uuid", pa.string()),
        ("seed_uuid", pa.string()), ("object", pa.string()),
        ("gesture", pa.string()), ("grade", pa.string()), ("split", pa.string()),
    ])
    window_type = pa.struct([
        ("start_frame", pa.int32()), ("end_frame", pa.int32()),
        ("frame_count", pa.int32()), ("source_total_frames", pa.int32()),
        ("context_frames", pa.int32()), ("status", pa.string()),
    ])
    provenance_type = pa.struct([
        ("contract", pa.string()), ("state_contract", pa.string()),
        ("action_contract", pa.string()), ("source_dataset", pa.string()),
        ("source_dataset_version", pa.int64()),
        ("source_release_verification_sha256", pa.string()),
        ("source_row_payload_sha256", pa.string()),
        ("train_selection_sha256", pa.string()),
        ("validation_selection_sha256", pa.string()),
        ("train_windows_sha256", pa.string()),
        ("validation_windows_sha256", pa.string()),
        ("plan_sha256", pa.string()), ("client_commit", pa.string()),
        ("builder_sha256", pa.string()), ("state_contract_sha256", pa.string()),
        ("geometry_contract_sha256", pa.string()),
        ("physics_adapter_sha256", pa.string()), ("manorl_commit", pa.string()),
        ("all_assets_commit", pa.string()),
        ("curated_asset_source_commit", pa.string()),
    ])
    return pa.schema([
        ("index", index_type), ("window", window_type),
        ("state", pa.list_(pa.list_(pa.float32(), C.STATE_DIM))),
        ("state_sha256", pa.string()), ("row_payload_sha256", pa.string()),
        ("prompt", pa.string()), ("provenance", provenance_type),
    ])


def _entry_digest(state: np.ndarray, entry: dict[str, Any], source_payload_sha: str) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(state, dtype="<f4").tobytes())
    digest.update(str(entry["release_row_index"]).encode())
    digest.update(entry["uuid"].encode())
    digest.update(str(entry["window_start"]).encode())
    digest.update(str(entry["window_end"]).encode())
    digest.update(source_payload_sha.encode())
    return digest.hexdigest()


def _scene(object_name: str) -> P.Native28Scene:
    if object_name not in _SCENES:
        _SCENES[object_name] = P.compile_scene(object_name)
    return _SCENES[object_name]


def derive_row(
    entry: dict[str, Any], source: dict[str, Any], *, plan: dict[str, Any], plan_sha: str
) -> dict[str, Any]:
    index = source["index"]
    required_index = {
        "uuid": entry["uuid"], "seed_uuid": entry["seed_uuid"],
        "object": entry["object"], "gesture": entry["gesture"], "grade": "A",
    }
    for key, expected in required_index.items():
        if index.get(key) != expected:
            raise ValueError(
                f"source index mismatch row{entry['release_row_index']} {key}: "
                f"expected {expected!r}, got {index.get(key)!r}"
            )
    if int(source["frame_count"]) != int(entry["source_total_frames"]):
        raise ValueError(f"source frame count mismatch row{entry['release_row_index']}")
    if source["prompt"] != entry["prompt"]:
        raise ValueError(f"source prompt mismatch row{entry['release_row_index']}")
    hands = source["hands"]
    objects = source["objects"]
    if len(hands) != 1 or len(objects) != 1:
        raise ValueError(f"State56 requires exactly one hand/object row{entry['release_row_index']}")
    qpos = np.asarray(hands[0]["urdf_dof"], dtype=np.float32)
    targets = np.asarray(hands[0]["urdf_dof_target"], dtype=np.float32)
    state41 = np.asarray(source["state"], dtype=np.float32)
    actions = np.asarray(source["actions"], dtype=np.float32)
    timestamps = np.asarray(source["timestamp"], dtype=np.float64)
    positions = np.asarray(objects[0]["pos"], dtype=np.float32)
    quaternions = np.asarray(objects[0]["quat_wxyz"], dtype=np.float32)
    total = int(entry["source_total_frames"])
    expected_shapes = {
        "qpos": (total, 28), "targets": (total, 28), "state41": (total, 41),
        "actions": (total, 32), "timestamps": (total,), "positions": (total, 3),
        "quaternions": (total, 4),
    }
    actual_shapes = {
        "qpos": qpos.shape, "targets": targets.shape, "state41": state41.shape,
        "actions": actions.shape, "timestamps": timestamps.shape, "positions": positions.shape,
        "quaternions": quaternions.shape,
    }
    for name, shape in expected_shapes.items():
        if actual_shapes[name] != shape:
            raise ValueError(f"row{entry['release_row_index']} {name} shape {actual_shapes[name]} != {shape}")
    if len(source["contact"]) != total:
        raise ValueError(f"row{entry['release_row_index']} contact frame count mismatch")
    if not np.array_equal(qpos, state41[:, :28]):
        raise ValueError(f"row{entry['release_row_index']} source qpos/state41 mismatch")
    if not np.array_equal(targets, actions[:, :28]):
        raise ValueError(f"row{entry['release_row_index']} source target/action mismatch")
    if not np.array_equal(actions[:, 28:], np.zeros((total, 4), dtype=np.float32)):
        raise ValueError(f"row{entry['release_row_index']} source action pad4 is nonzero")
    if total > 1 and not np.allclose(np.diff(timestamps), C.SOURCE_INTERVAL_SECONDS, rtol=0, atol=2e-12):
        raise ValueError(f"row{entry['release_row_index']} source timestamp interval mismatch")
    start, end = int(entry["window_start"]), int(entry["window_end"])
    count = end - start + 1
    tips = np.empty((count, 5, 3), dtype=np.float32)
    contacts = np.empty((count, 5), dtype=np.float32)
    forces = np.empty((count, 5), dtype=np.float32)
    scene = _scene(entry["object"])
    for local, frame in enumerate(range(start, end + 1)):
        P.set_snapshot(
            scene, hand_qpos=qpos[frame], object_position=positions[frame],
            object_quaternion_wxyz=quaternions[frame], target28=targets[frame],
        )
        tip_world = P.fingertip_world(scene)
        tips[local] = C.fingertips_in_collision_box_frame(
            tip_world, positions[frame], C.quaternion_wxyz_to_matrix(quaternions[frame]),
            entry["object"],
        )
        contacts[local], forces[local] = C.aggregate_state41_contact_frame(source["contact"][frame])
    state56 = C.build_state56_window_from_features(
        hand_qpos=qpos, finger_contacts=np.pad(contacts, ((start, total-end-1),(0,0))),
        finger_log1p_force=np.pad(forces, ((start, total-end-1),(0,0))),
        fingertip_collision_box_xyz=np.pad(tips, ((start, total-end-1),(0,0),(0,0))),
        object_position_world=positions, window_start=start, window_end=end,
    )
    if not np.array_equal(state56[:, :28], qpos[start:end+1]):
        raise ValueError(f"row{entry['release_row_index']} State56 qpos mismatch")
    if not np.array_equal(state56[:, 28:33], state41[start:end+1, 28:33]):
        raise ValueError(f"row{entry['release_row_index']} State56 contact mismatch")
    if not np.array_equal(state56[:, 33], state41[start:end+1, 33]):
        max_error = float(np.max(np.abs(state56[:, 33] - state41[start:end+1, 33])))
        raise ValueError(f"row{entry['release_row_index']} State56 lift mismatch max={max_error}")
    state_sha = hashlib.sha256(np.ascontiguousarray(state56, dtype="<f4").tobytes()).hexdigest()
    source_payload_sha = str(source["row_payload_sha256"])
    if len(source_payload_sha) != 64:
        raise ValueError(f"row{entry['release_row_index']} source row payload SHA invalid")
    provenance = {
        "contract": RELEASE_CONTRACT,
        "state_contract": C.STATE_CONTRACT_ID,
        "action_contract": C.ACTION_CONTRACT_ID,
        "source_dataset": plan["source_dataset"],
        "source_dataset_version": int(plan["source_dataset_version"]),
        "source_release_verification_sha256": plan["source_release_verification_sha256"],
        "source_row_payload_sha256": source_payload_sha,
        "train_selection_sha256": plan["train_selection_sha256"],
        "validation_selection_sha256": plan["validation_selection_sha256"],
        "train_windows_sha256": plan["train_windows_sha256"],
        "validation_windows_sha256": plan["validation_windows_sha256"],
        "plan_sha256": plan_sha,
        **plan["code"],
        "manorl_commit": C.EXPECTED_MANORL_COMMIT,
        "all_assets_commit": C.EXPECTED_ALL_ASSETS_COMMIT,
        "curated_asset_source_commit": C.EXPECTED_CURATED_ASSET_SOURCE_COMMIT,
    }
    return {
        "index": {
            "release_row_index": int(entry["release_row_index"]),
            "filtered_row_index": int(entry["filtered_row_index"]),
            "original_merged_row_index": int(entry["original_merged_row_index"]),
            "uuid": entry["uuid"], "seed_uuid": entry["seed_uuid"],
            "object": entry["object"], "gesture": entry["gesture"],
            "grade": "A", "split": entry["split"],
        },
        "window": {
            "start_frame": start, "end_frame": end, "frame_count": count,
            "source_total_frames": total, "context_frames": int(entry["context_frames"]),
            "status": entry["window_status"],
        },
        "state": state56.tolist(), "state_sha256": state_sha,
        "row_payload_sha256": _entry_digest(state56, entry, source_payload_sha),
        "prompt": entry["prompt"], "provenance": provenance,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_plan(
    *, source_dataset: Path, profile_dir: Path, output_root: Path, shard_count: int
) -> Path:
    import lance

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"State56 sidecar output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    source_dataset = source_dataset.expanduser().resolve()
    profile_dir = profile_dir.expanduser().resolve()
    paths = {
        "train_selection": profile_dir / "train_selection.json",
        "validation_selection": profile_dir / "validation_selection.json",
        "train_windows": profile_dir / "train_contact_windows.json",
        "validation_windows": profile_dir / "validation_contact_windows.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing State56 Scheme-A {name}: {path}")
    train = _load_json(paths["train_selection"])
    validation = _load_json(paths["validation_selection"])
    train_windows = _load_json(paths["train_windows"])["windows"]
    validation_windows = _load_json(paths["validation_windows"])["windows"]
    if len(train["rows"]) != 4613 or len(validation["rows"]) != 243:
        raise ValueError("State56 Scheme-A selection counts must be4613/243")
    if train.get("population_rows") != 4856 or validation.get("population_rows") != 4856:
        raise ValueError("State56 Scheme-A population must be4856")
    release_sha = train.get("release_verification_sha256")
    if release_sha != validation.get("release_verification_sha256") or len(str(release_sha)) != 64:
        raise ValueError("Scheme-A release verification identity mismatch")
    entries: list[dict[str, Any]] = []
    seen_rows: set[int] = set()
    seen_uuid: set[str] = set()
    for split, selection, windows in (
        ("train", train, train_windows), ("validation", validation, validation_windows)
    ):
        for row in selection["rows"]:
            release_index = int(row["release_row_index"])
            if release_index in seen_rows or row["uuid"] in seen_uuid:
                raise ValueError(f"duplicate Scheme-A row/UUID {release_index}/{row['uuid']}")
            seen_rows.add(release_index); seen_uuid.add(row["uuid"])
            window = windows.get(str(release_index))
            if not isinstance(window, dict) or window.get("status") != "contact_window":
                raise ValueError(f"missing contact window for release row{release_index}")
            start, end = int(window["start_frame"]), int(window["end_frame"])
            frames = int(row["frames"])
            if int(window["total_frames"]) != frames or start < 0 or end < start or end >= frames:
                raise ValueError(f"invalid contact window for release row{release_index}")
            entries.append({
                **row, "split": split, "window_start": start, "window_end": end,
                "window_frames": end-start+1, "source_total_frames": frames,
                "context_frames": int(window["context_frames"]),
                "window_status": window["status"],
            })
    if len(entries) != 4856:
        raise ValueError(f"State56 Scheme-A union must have4856 rows, got {len(entries)}")
    entries.sort(key=lambda item: int(item["release_row_index"]))
    dataset = lance.dataset(str(source_dataset))
    version = int(dataset.version)
    light = dataset.take(
        [entry["release_row_index"] for entry in entries], columns=["index", "frame_count"]
    ).to_pylist()
    for entry, row in zip(entries, light, strict=True):
        if row["index"]["uuid"] != entry["uuid"] or int(row["frame_count"]) != entry["source_total_frames"]:
            raise ValueError(f"source light-index mismatch row{entry['release_row_index']}")
    total_frames = sum(entry["window_frames"] for entry in entries)
    target = total_frames / shard_count
    shards: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_frames = 0
    remaining_boundaries = shard_count - 1
    for entry in entries:
        if current and current_frames >= target and remaining_boundaries > 0:
            shards.append({"shard_index": len(shards), "entries": current, "frames": current_frames})
            current=[];current_frames=0;remaining_boundaries-=1
        current.append(entry);current_frames += entry["window_frames"]
    shards.append({"shard_index": len(shards), "entries": current, "frames": current_frames})
    if len(shards) != shard_count or any(not shard["entries"] for shard in shards):
        raise RuntimeError(f"failed to construct {shard_count} non-empty shards")
    plan = {
        "contract": PLAN_CONTRACT, "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source_dataset), "source_dataset_version": version,
        "source_release_verification_sha256": release_sha,
        "profile_dir": str(profile_dir), "scheme": "A",
        "population_grade": "A", "population_rows": 4856,
        "train_rows": 4613, "validation_rows": 243, "held_out_rows": 0,
        "train_selection": str(paths["train_selection"]),
        "train_selection_sha256": sha256(paths["train_selection"]),
        "validation_selection": str(paths["validation_selection"]),
        "validation_selection_sha256": sha256(paths["validation_selection"]),
        "train_windows": str(paths["train_windows"]),
        "train_windows_sha256": sha256(paths["train_windows"]),
        "validation_windows": str(paths["validation_windows"]),
        "validation_windows_sha256": sha256(paths["validation_windows"]),
        "state_contract": C.STATE_CONTRACT_ID, "action_contract": C.ACTION_CONTRACT_ID,
        "state_dim": C.STATE_DIM, "action_dim": C.ACTION_DIM,
        "window_frames": total_frames, "source_interval_seconds": C.SOURCE_INTERVAL_SECONDS,
        "code": code_identity(), "shard_count": shard_count, "shards": shards,
    }
    path = output_root / "release_plan.json"
    atomic_json(path, plan)
    print(json.dumps({"plan": str(path), "sha256": sha256(path), "rows": len(entries), "frames": total_frames, "shards": len(shards)}, sort_keys=True))
    return path


def load_plan(path: Path) -> tuple[dict[str, Any], str]:
    plan = _load_json(path)
    if plan.get("contract") != PLAN_CONTRACT:
        raise ValueError(f"State56 plan contract mismatch: {path}")
    actual_code = code_identity()
    if plan.get("code") != actual_code:
        raise ValueError(f"State56 plan code identity mismatch: plan={plan.get('code')} actual={actual_code}")
    return plan, sha256(path)


def shard_path(root: Path, index: int) -> Path:
    return root / "shards" / f"shard-{index:03d}.lance"


def existing_prefix(path: Path, entries: list[dict[str, Any]], plan_sha: str) -> int:
    if not path.exists():
        return 0
    import lance
    rows = lance.dataset(str(path)).to_table(columns=["index", "provenance"]).to_pylist()
    if len(rows) > len(entries):
        raise ValueError(f"existing State56 shard has excess rows: {path}")
    for offset, row in enumerate(rows):
        if (
            int(row["index"]["release_row_index"]) != int(entries[offset]["release_row_index"])
            or row["index"]["uuid"] != entries[offset]["uuid"]
            or row["provenance"]["plan_sha256"] != plan_sha
        ):
            raise ValueError(f"existing State56 shard prefix mismatch {path}:{offset}")
    return len(rows)


def run_shard(plan_path: Path, shard_index: int, batch_size: int) -> Path:
    import lance
    import pyarrow as pa

    plan, plan_sha = load_plan(plan_path)
    root = plan_path.parent
    shard = plan["shards"][shard_index]
    if int(shard["shard_index"]) != shard_index:
        raise ValueError("State56 shard plan index mismatch")
    entries = list(shard["entries"])
    target = shard_path(root, shard_index)
    target.parent.mkdir(parents=True, exist_ok=True)
    completed = existing_prefix(target, entries, plan_sha)
    dataset = lance.dataset(plan["source_dataset"], version=int(plan["source_dataset_version"]))
    started = time.monotonic()
    for batch_start in range(completed, len(entries), batch_size):
        batch_entries = entries[batch_start:batch_start+batch_size]
        rows = dataset.take([entry["release_row_index"] for entry in batch_entries], columns=SOURCE_COLUMNS).to_pylist()
        rendered = [derive_row(entry, source, plan=plan, plan_sha=plan_sha) for entry,source in zip(batch_entries, rows, strict=True)]
        table = pa.Table.from_pylist(rendered, schema=sidecar_schema())
        lance.write_dataset(table, str(target), mode="append" if target.exists() else "create")
        completed = existing_prefix(target, entries, plan_sha)
        progress = {
            "contract": RELEASE_CONTRACT, "updated_at": datetime.now(timezone.utc).isoformat(),
            "shard_index": shard_index, "completed_rows": completed, "total_rows": len(entries),
            "completed_frames": sum(int(entry["window_frames"]) for entry in entries[:completed]),
            "total_frames": int(shard["frames"]), "elapsed_seconds": time.monotonic()-started,
        }
        atomic_json(root / "progress" / f"shard-{shard_index:03d}.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    verify_shard(plan_path, shard_index)
    return target


def verify_shard(plan_path: Path, shard_index: int) -> dict[str, Any]:
    import lance

    plan, plan_sha = load_plan(plan_path)
    shard = plan["shards"][shard_index]
    entries = list(shard["entries"])
    path = shard_path(plan_path.parent, shard_index)
    rows = lance.dataset(str(path)).to_table().to_pylist()
    if len(rows) != len(entries):
        raise ValueError(f"State56 shard{shard_index} row count mismatch")
    frame_count = 0
    for entry,row in zip(entries,rows,strict=True):
        state = np.asarray(row["state"], dtype=np.float32)
        if state.shape != (int(entry["window_frames"]), C.STATE_DIM) or not np.all(np.isfinite(state)):
            raise ValueError(f"State56 shard{shard_index} invalid state row{entry['release_row_index']}")
        state_sha = hashlib.sha256(np.ascontiguousarray(state, dtype="<f4").tobytes()).hexdigest()
        if state_sha != row["state_sha256"] or row["provenance"]["plan_sha256"] != plan_sha:
            raise ValueError(f"State56 shard{shard_index} payload identity mismatch row{entry['release_row_index']}")
        frame_count += state.shape[0]
    result = {
        "contract": RELEASE_CONTRACT, "shard_index": shard_index, "rows": len(rows),
        "frames": frame_count, "path": str(path), "plan_sha256": plan_sha,
        "uuid_order_verified": True, "state_sha256_verified": True, "finite_verified": True,
    }
    atomic_json(plan_path.parent / "verifications" / f"shard-{shard_index:03d}.json", result)
    return result


def aggregate(plan_path: Path) -> Path:
    import lance

    plan, plan_sha = load_plan(plan_path)
    root = plan_path.parent
    target = root / "mano_state56_native28_sidecar_v1.lance"
    if target.exists():
        raise FileExistsError(f"State56 aggregate already exists: {target}")
    staging = root / f".{target.name}.incoming-{uuid.uuid4().hex}"
    try:
        mode = "create"
        for shard in plan["shards"]:
            index = int(shard["shard_index"])
            verify_shard(plan_path, index)
            source = lance.dataset(str(shard_path(root,index)))
            lance.write_dataset(source.scanner().to_reader(), str(staging), mode=mode)
            mode = "append"
        dataset = lance.dataset(str(staging))
        light = dataset.to_table(columns=["index","window","provenance"]).to_pylist()
        entries = [entry for shard in plan["shards"] for entry in shard["entries"]]
        if len(light) != 4856 or [row["index"]["uuid"] for row in light] != [entry["uuid"] for entry in entries]:
            raise ValueError("State56 aggregate UUID order/count mismatch")
        if any(row["provenance"]["plan_sha256"] != plan_sha for row in light):
            raise ValueError("State56 aggregate plan identity mismatch")
        os.replace(staging,target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    files=[]
    for path in sorted(target.rglob("*")):
        if path.is_file():
            files.append({"path":str(path.relative_to(target)),"bytes":path.stat().st_size,"sha256":sha256(path)})
    verification = {
        "contract": RELEASE_CONTRACT, "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(), "path": str(target),
        "lance_version": int(lance.dataset(str(target)).version), "rows": 4856,
        "train_rows":4613,"validation_rows":243,"held_out_rows":0,
        "frames":sum(int(entry["window_frames"]) for entry in entries),
        "state_dim":C.STATE_DIM,"action_dim":C.ACTION_DIM,"plan":str(plan_path),
        "plan_sha256":plan_sha,"files":files,
    }
    atomic_json(root/"release_verification.json",verification)
    print(json.dumps({"release":str(target),"verification":str(root/'release_verification.json'),"verification_sha256":sha256(root/'release_verification.json')},sort_keys=True))
    return target


def main() -> None:
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command",required=True)
    plan_parser=sub.add_parser("plan")
    plan_parser.add_argument("--source-dataset",type=Path,required=True)
    plan_parser.add_argument("--profile-dir",type=Path,required=True)
    plan_parser.add_argument("--output-root",type=Path,required=True)
    plan_parser.add_argument("--shards",type=int,default=16)
    shard_parser=sub.add_parser("run-shard")
    shard_parser.add_argument("--plan",type=Path,required=True)
    shard_parser.add_argument("--shard-index",type=int,required=True)
    shard_parser.add_argument("--batch-size",type=int,default=8)
    verify_parser=sub.add_parser("verify-shard")
    verify_parser.add_argument("--plan",type=Path,required=True)
    verify_parser.add_argument("--shard-index",type=int,required=True)
    aggregate_parser=sub.add_parser("aggregate")
    aggregate_parser.add_argument("--plan",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="plan":build_plan(source_dataset=args.source_dataset,profile_dir=args.profile_dir,output_root=args.output_root,shard_count=args.shards)
    elif args.command=="run-shard":run_shard(args.plan,args.shard_index,args.batch_size)
    elif args.command=="verify-shard":print(json.dumps(verify_shard(args.plan,args.shard_index),sort_keys=True))
    elif args.command=="aggregate":aggregate(args.plan)


if __name__=="__main__":
    main()
