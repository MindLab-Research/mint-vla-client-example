#!/usr/bin/env python3
"""Fail-closed adapter for the pinned ManoRL native28 MuJoCo model.

This module owns no ManoRL assets. It reads a clean, explicitly pinned checkout,
validates the28-joint/timing ABI, and exposes CPU-only snapshot FK used by the
State56 sidecar builder. Rendering and dynamics rollout are intentionally not
part of this data-derivation adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from scripts import mano_state56_contract as state56

HAND_DIM = 28
DT = 0.0025
NATIVE_SUBSTEPS = 2
SOURCE_INTERVAL_SECONDS = DT * NATIVE_SUBSTEPS
DEFAULT_MANORL_REPO_ROOT = Path("/vePFS-Mindverse/user/intern/wenxi/manorl-native-28d")
_RUNTIME: dict[str, Any] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments], text=True, stderr=subprocess.DEVNULL
    ).strip()


def runtime() -> dict[str, Any]:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    root = Path(os.environ.get("MANORL_REPO_ROOT", str(DEFAULT_MANORL_REPO_ROOT))).expanduser().resolve()
    required = (root / "sim" / "manorl" / "contracts.py", root / "sim" / "manorl" / "assets.py")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"authoritative ManoRL checkout is incomplete at {root}: {missing}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from sim.manorl import assets, contracts

    if (
        contracts.JOINT_DOF != HAND_DIM
        or tuple(contracts.JOINT_NAMES_28) != tuple(contracts.JOINT_NAMES)
        or contracts.PHYSICS_TIMESTEP != DT
        or contracts.PHYSICS_SUBSTEPS_PER_TARGET != NATIVE_SUBSTEPS
    ):
        raise RuntimeError("Client/ManoRL native28 timing or joint contract mismatch")
    commit = _git(root, "rev-parse", "HEAD")
    expected = os.environ.get("MANORL_EXPECTED_COMMIT")
    if not expected:
        raise RuntimeError("MANORL_EXPECTED_COMMIT is required")
    if _git(root, "rev-parse", f"{expected}^{{commit}}") != commit:
        raise RuntimeError(f"ManoRL checkout {commit} does not equal MANORL_EXPECTED_COMMIT={expected}")
    if commit != state56.EXPECTED_MANORL_COMMIT:
        raise RuntimeError(
            f"State56 contract pins ManoRL {state56.EXPECTED_MANORL_COMMIT}, runtime has {commit}"
        )
    status = _git(root, "status", "--porcelain", "--ignore-submodules=dirty")
    if status:
        raise RuntimeError(f"authoritative ManoRL checkout is dirty: {status.splitlines()[:8]}")
    _RUNTIME = {"root": root, "assets": assets, "contracts": contracts, "commit": commit}
    return _RUNTIME


def runtime_provenance(object_name: str | None = None) -> dict[str, Any]:
    value = runtime()
    root: Path = value["root"]
    assets = value["assets"]
    contracts = value["contracts"]
    all_assets_commit = os.environ.get("MANORL_ALL_ASSETS_COMMIT")
    if all_assets_commit != state56.EXPECTED_ALL_ASSETS_COMMIT:
        raise RuntimeError(
            "MANORL_ALL_ASSETS_COMMIT must equal the State56 pinned all-assets checkout"
        )
    result = {
        "repository": str(root),
        "commit": value["commit"],
        "contracts_sha256": _sha256(root / "sim" / "manorl" / "contracts.py"),
        "assets_code_sha256": _sha256(root / "sim" / "manorl" / "assets.py"),
        "joint_order": list(contracts.JOINT_NAMES_28),
        "physics_timestep": float(contracts.PHYSICS_TIMESTEP),
        "physics_substeps_per_target": int(contracts.PHYSICS_SUBSTEPS_PER_TARGET),
        "all_assets_commit": all_assets_commit,
        "all_assets_root": str(assets.ALL_ASSETS_ROOT),
    }
    if object_name is not None:
        validated = assets.validate_asset_manifest(object_name, hand_side="right")
        result["object"] = object_name
        result["curated_asset_source_commit"] = validated.get("source_commit")
        if (
            result["curated_asset_source_commit"]
            != state56.EXPECTED_CURATED_ASSET_SOURCE_COMMIT
        ):
            raise RuntimeError(f"asset manifest source commit mismatch for {object_name}")
    return result


@dataclass
class Native28Scene:
    object_name: str
    model: Any
    data: Any
    hand_qpos_addresses: np.ndarray
    hand_qvel_addresses: np.ndarray
    object_qpos_address: int
    object_qvel_address: int
    object_body_id: int
    joint_names: tuple[str, ...]


def object_body_id(model: Any, object_name: str) -> int:
    import mujoco

    body_name = runtime()["assets"].object_runtime(object_name).body_name
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if value < 0:
        raise ValueError(
            f"compiled object body missing for {object_name}: expected {body_name}"
        )
    return int(value)


def compile_scene(object_name: str) -> Native28Scene:
    import mujoco

    value = runtime()
    assets = value["assets"]
    contracts = value["contracts"]
    if object_name not in state56.OBJECT_COLLISION_BOXES:
        raise ValueError(f"object {object_name!r} is outside the State5611-object contract")
    runtime_provenance(object_name)
    _mujoco, model = assets.compile_model(
        contracts.ServoConfig(), object_type=object_name, visual_meshes=False, hand_side="right"
    )
    data = mujoco.MjData(model)
    object_spec = assets.object_runtime(object_name)
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, object_spec.free_joint_name)
    if object_joint < 0:
        raise ValueError(f"compiled object ABI missing for {object_name}")
    joint_names = tuple(contracts.JOINT_NAMES_28)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    if any(joint_id < 0 for joint_id in joint_ids):
        raise ValueError("compiled native28 hand joint ABI is incomplete")
    hand_qpos = np.asarray([int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids], dtype=np.int64)
    hand_qvel = np.asarray([int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids], dtype=np.int64)
    if not np.array_equal(hand_qpos, np.arange(HAND_DIM)):
        raise ValueError(f"compiled hand qpos addresses mismatch: {hand_qpos}")
    if not np.array_equal(hand_qvel, np.arange(HAND_DIM)):
        raise ValueError(f"compiled hand qvel addresses mismatch: {hand_qvel}")
    actuator_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) or ""
        for index in range(model.nu)
    )
    if actuator_names != joint_names:
        raise ValueError(f"compiled actuator order mismatch: {actuator_names}")
    if not np.isclose(model.opt.timestep, DT, rtol=0.0, atol=1e-15):
        raise ValueError(f"compiled timestep mismatch: {model.opt.timestep}")
    return Native28Scene(
        object_name=object_name,
        model=model,
        data=data,
        hand_qpos_addresses=hand_qpos,
        hand_qvel_addresses=hand_qvel,
        object_qpos_address=int(model.jnt_qposadr[object_joint]),
        object_qvel_address=int(model.jnt_dofadr[object_joint]),
        object_body_id=object_body_id(model, object_name),
        joint_names=joint_names,
    )


def set_snapshot(
    scene: Native28Scene,
    *,
    hand_qpos: np.ndarray,
    object_position: np.ndarray,
    object_quaternion_wxyz: np.ndarray,
    hand_qvel: np.ndarray | None = None,
    object_qvel: np.ndarray | None = None,
    target28: np.ndarray | None = None,
) -> None:
    import mujoco

    qpos = np.asarray(hand_qpos, dtype=np.float64)
    position = np.asarray(object_position, dtype=np.float64)
    quaternion = np.asarray(object_quaternion_wxyz, dtype=np.float64)
    if qpos.shape != (28,) or position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError(f"snapshot shapes must be28/3/4, got {qpos.shape}/{position.shape}/{quaternion.shape}")
    if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
        raise ValueError("snapshot qpos/object pose contains non-finite values")
    quaternion = quaternion / np.linalg.norm(quaternion)
    scene.data.qpos[scene.hand_qpos_addresses] = qpos
    address = scene.object_qpos_address
    scene.data.qpos[address : address + 3] = position
    scene.data.qpos[address + 3 : address + 7] = quaternion
    scene.data.qvel[:] = 0.0
    if hand_qvel is not None:
        velocity = np.asarray(hand_qvel, dtype=np.float64)
        if velocity.shape != (28,) or not np.all(np.isfinite(velocity)):
            raise ValueError("hand_qvel must be finite shape(28,)")
        scene.data.qvel[scene.hand_qvel_addresses] = velocity
    if object_qvel is not None:
        velocity = np.asarray(object_qvel, dtype=np.float64)
        if velocity.shape != (6,) or not np.all(np.isfinite(velocity)):
            raise ValueError("object_qvel must be finite shape(6,)")
        scene.data.qvel[scene.object_qvel_address : scene.object_qvel_address + 6] = velocity
    command = qpos if target28 is None else np.asarray(target28, dtype=np.float64)
    if command.shape != (28,) or not np.all(np.isfinite(command)):
        raise ValueError("target28 must be finite shape(28,)")
    scene.data.ctrl[:] = command
    mujoco.mj_forward(scene.model, scene.data)
    if not np.all(np.isfinite(scene.data.xpos)):
        raise FloatingPointError("native28 snapshot FK produced non-finite positions")


def fingertip_world(scene: Native28Scene) -> np.ndarray:
    return state56.fingertip_world_from_mujoco(scene.model, scene.data)


def compiled_collision_box(scene: Native28Scene) -> state56.ObjectCollisionBox:
    """Recompute the union collision AABB in object-body coordinates."""
    model = scene.model
    geom_ids = [
        index
        for index in range(model.ngeom)
        if int(model.geom_bodyid[index]) == scene.object_body_id
        and (int(model.geom_contype[index]) or int(model.geom_conaffinity[index]))
    ]
    if not geom_ids:
        raise ValueError(f"native28 model has no collision geoms for {scene.object_name}")
    points: list[np.ndarray] = []
    for geom_id in geom_ids:
        center = np.asarray(model.geom_aabb[geom_id][:3], dtype=np.float64)
        half = np.asarray(model.geom_aabb[geom_id][3:], dtype=np.float64)
        position = np.asarray(model.geom_pos[geom_id], dtype=np.float64)
        w, x, y, z = np.asarray(model.geom_quat[geom_id], dtype=np.float64)
        rotation = np.asarray(
            [
                [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
            ],
            dtype=np.float64,
        )
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            points.append(position + rotation @ (center + half * np.asarray(signs)))
    stacked = np.stack(points)
    low, high = stacked.min(axis=0), stacked.max(axis=0)
    return state56.ObjectCollisionBox(half_extents=(high-low)/2, local_center=(low+high)/2)
