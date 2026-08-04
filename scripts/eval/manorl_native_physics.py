#!/usr/bin/env python3
"""Thin Client adapter over the authoritative ManoRL 28D native model.

The production scheduler remains ``replay_mano_target_physics.py``. This module
only resolves a pinned ManoRL checkout, compiles its native ``MjModel``, and
exposes the small scene/step ABI already consumed by the historical replay.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

HAND_DIM = 28
DT = 0.0025
NATIVE_SUBSTEPS = 2
DEFAULT_MANORL_REPO_ROOT = Path(
    "/vePFS-Mindverse/user/intern/wenxi/manorl-native-28d"
)
_RUNTIME: dict[str, Any] | None = None
_CONTACT_IDS: dict[int, tuple[set[int], set[int], int]] = {}
_VISUAL_INVARIANCE: dict[int, dict[str, Any]] = {}
_STATE41_IDS: dict[int, "State41FeatureIds"] = {}

_STATE41_KEYPOINT_LINKS = (
    "thumb_cmc", "thumb_mcp", "thumb_ip",
    "index_mcp", "index_pip", "index_dip",
    "middle_mcp", "middle_pip", "middle_dip",
    "ring_mcp", "ring_pip", "ring_dip",
    "pinky_mcp", "pinky_pip", "pinky_dip",
    "palm",
)
_STATE41_FINGERTIP_LINKS = {
    "index": "index_dip",
    "thumb": "thumb_ip",
    "ring": "ring_dip",
    "middle": "middle_dip",
    "pinky": "pinky_dip",
}


@dataclass(frozen=True)
class State41FeatureIds:
    hand_geom_ids: frozenset[int]
    object_geom_ids: tuple[int, ...]
    geom_id_to_finger: dict[int, str]
    fingertip_measurement_geom_ids: tuple[int, ...]
    floor_geom_id: int


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


def _runtime() -> dict[str, Any]:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    root = Path(
        os.environ.get("MANORL_REPO_ROOT", str(DEFAULT_MANORL_REPO_ROOT))
    ).expanduser().resolve()
    required = (
        root / "sim" / "manorl" / "contracts.py",
        root / "sim" / "manorl" / "assets.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"authoritative ManoRL checkout is incomplete at {root}: {missing}"
        )
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
        raise RuntimeError("Client/ManoRL 28D timing or joint contract mismatch")
    commit = _git(root, "rev-parse", "HEAD")
    expected = os.environ.get("MANORL_EXPECTED_COMMIT")
    if expected and _git(root, "rev-parse", f"{expected}^{{commit}}") != commit:
        raise RuntimeError(
            f"ManoRL checkout {commit} does not equal MANORL_EXPECTED_COMMIT={expected}"
        )
    status = _git(root, "status", "--porcelain", "--ignore-submodules=dirty")
    if status:
        raise RuntimeError(f"authoritative ManoRL checkout is dirty: {status.splitlines()[:8]}")
    _RUNTIME = {
        "root": root,
        "assets": assets,
        "contracts": contracts,
        "commit": commit,
    }
    return _RUNTIME


def runtime_provenance(object_name: str | None = None) -> dict[str, Any]:
    runtime = _runtime()
    root: Path = runtime["root"]
    assets = runtime["assets"]
    contracts = runtime["contracts"]
    all_assets_commit = os.environ.get("MANORL_ALL_ASSETS_COMMIT")
    if not all_assets_commit or len(all_assets_commit) != 40:
        raise RuntimeError("MANORL_ALL_ASSETS_COMMIT must pin the transferred asset source")
    result = {
        "repository": str(root),
        "commit": runtime["commit"],
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
    return result


def servo_parameters() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    runtime = _runtime()
    contracts = runtime["contracts"]
    servo = contracts.ServoConfig()
    return (
        np.asarray(servo.kp, dtype=np.float64),
        np.asarray(servo.dampratio, dtype=np.float64),
        np.asarray(contracts.EFFORT, dtype=np.float64),
    )


def _legacy_visual_model(
    object_name: str, width: int, height: int, *,
    head_camera_preset: str | None = None,
):
    """Add the unchanged Client camera/floor/light contract to ManoRL physics."""
    runtime = _runtime()
    assets = runtime["assets"]
    contracts = runtime["contracts"]
    from export_mano_sim_video import (
        MjcfBuilder, _camera_xyaxes, _fmt, _scale_rgba,
        horizontal_to_vertical_fov,
    )
    from scripts.eval import mano_action_support as legacy
    from sim.manorl.native_trace_video import assert_visual_physics_invariant

    if head_camera_preset is None:
        head_camera_preset = legacy.DEFAULT_HEAD_CAMERA_PRESET
    head_camera = legacy.head_camera_config(head_camera_preset)
    mujoco, physics_model = assets.compile_model(
        contracts.ServoConfig(), object_type=object_name, visual_meshes=False,
        hand_side="right",
    )
    xml = assets.build_scene_xml(
        contracts.ServoConfig(), object_type=object_name, visual_meshes=True,
        hand_side="right",
    )
    root = ET.fromstring(xml)
    legacy_builder = MjcfBuilder(offwidth=width, offheight=height, show_debug_frames=False)
    legacy_root = legacy_builder.root
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    legacy_visual = legacy_root.find("visual")
    asset = root.find("asset")
    legacy_asset = legacy_root.find("asset")
    worldbody = root.find("worldbody")
    legacy_worldbody = legacy_root.find("worldbody")
    if any(node is None for node in (legacy_visual, asset, legacy_asset, worldbody, legacy_worldbody)):
        raise ValueError("ManoRL or legacy visual scene has incomplete top-level elements")
    visual.clear()
    visual.attrib.update(legacy_visual.attrib)
    for child in legacy_visual:
        visual.append(copy.deepcopy(child))
    for child in list(asset):
        old_floor_texture = child.tag == "texture" and child.get("name") == "floor_checker"
        skybox = child.tag == "texture" and child.get("type") == "skybox"
        old_floor_material = child.tag == "material" and child.get("name") in {"floor_checker", "floor_grid"}
        if old_floor_texture or skybox or old_floor_material:
            asset.remove(child)
    for child in legacy_asset:
        if (child.tag == "texture" and child.get("name") == "floor_checker") or (child.tag == "material" and child.get("name") == "floor_grid"):
            asset.append(copy.deepcopy(child))
    for light in list(worldbody.findall("light")):
        worldbody.remove(light)
    for light in legacy_worldbody.findall("light"):
        worldbody.append(copy.deepcopy(light))
    floor = next((geom for geom in worldbody.findall("geom") if geom.get("name") == "floor"), None)
    legacy_floor = next((geom for geom in legacy_worldbody.findall("geom") if geom.get("name") == "floor"), None)
    if floor is None or legacy_floor is None:
        raise ValueError("ManoRL or legacy visual scene has no floor geom")
    floor.set("material", legacy_floor.get("material", "floor_grid"))
    floor.set("size", legacy_floor.get("size", "3 3 0.02"))

    object_body_name = assets.object_runtime(object_name).body_name
    for body in worldbody.iter("body"):
        if body.get("name") == object_body_name:
            continue
        for geom in body.findall("geom"):
            if geom.get("group") == "2" and geom.get("rgba"):
                geom.set("rgba", _scale_rgba(geom.get("rgba", ""), 0.75))

    def add_camera(name: str, spec: dict[str, Any], parent: ET.Element):
        right, up = _camera_xyaxes(spec["position"], spec["target"])
        ET.SubElement(parent, "camera", {
            "name": name, "pos": _fmt(spec["position"]), "xyaxes": _fmt([*right, *up]),
            "fovy": f"{horizontal_to_vertical_fov(spec['horizontal_fov'], width, height):.9g}",
        })

    add_camera(legacy.HEAD_CAMERA_NAME, head_camera, worldbody)
    wrist = next((body for body in worldbody.iter("body") if body.get("name") == legacy.WRIST_CAMERA["parent_body"]), None)
    if wrist is None:
        raise ValueError(f"missing legacy wrist camera parent {legacy.WRIST_CAMERA['parent_body']}")
    add_camera(legacy.WRIST_CAMERA_NAME, legacy.WRIST_CAMERA, wrist)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    assets.validate_compiled_model(mujoco, model, contracts.ServoConfig(), object_type=object_name, hand_side="right")
    invariance = assert_visual_physics_invariant(mujoco, physics_model, model)
    invariance["legacy_visual_contract"] = {
        "head_camera_preset": head_camera_preset,
        "head_camera": head_camera, "wrist_camera": legacy.WRIST_CAMERA,
        "floor_texture": "floor_checker",
        "skybox": next((node.get("name") for node in legacy_asset if node.tag == "texture" and node.get("type") == "skybox"), None),
        "light_names": [node.get("name") for node in legacy_worldbody.findall("light")],
        "offscreen_size": [width, height],
    }
    return mujoco, model, invariance


def visual_invariance(model: Any) -> dict[str, Any]:
    try:
        return _VISUAL_INVARIANCE[id(model)]
    except KeyError as exc:
        raise ValueError("model was not compiled with the legacy visual contract") from exc


def object_body_id(model: Any, object_name: str) -> int:
    """Resolve compiled body identity from the authoritative object runtime."""
    import mujoco

    runtime = _runtime()
    body_name = runtime["assets"].object_runtime(object_name).body_name
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(
            f"compiled object body missing for {object_name}: expected {body_name}"
        )
    return int(body_id)


def make_scene(
    object_name: str,
    width: int,
    height: int,
    *,
    physics: bool = True,
    physics_timestep: float = DT,
    create_renderer: bool = False,
    head_camera_preset: str | None = None,
):
    """Return the historical scene tuple backed by ManoRL ``compile_model``."""
    if not physics:
        raise ValueError("the 28D branch only compiles the native physics model")
    if not np.isclose(physics_timestep, DT, rtol=0.0, atol=1e-15):
        raise ValueError(f"requested physics timestep {physics_timestep} != {DT}")
    runtime = _runtime()
    assets = runtime["assets"]
    contracts = runtime["contracts"]
    if create_renderer:
        mujoco, model, invariance = _legacy_visual_model(
            object_name, width, height,
            head_camera_preset=head_camera_preset,
        )
        _VISUAL_INVARIANCE[id(model)] = invariance
    else:
        mujoco, model = assets.compile_model(
            contracts.ServoConfig(), object_type=object_name,
            visual_meshes=False, hand_side="right",
        )
    data = mujoco.MjData(model)
    object_spec = assets.object_runtime(object_name)
    object_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, object_spec.free_joint_name
    )
    object_body = object_body_id(model, object_name)
    if object_joint < 0:
        raise ValueError(f"compiled object ABI missing for {object_name}")
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in contracts.JOINT_NAMES_28
    ]
    if any(joint_id < 0 for joint_id in joint_ids):
        raise ValueError("compiled 28D hand joint ABI is incomplete")
    hand_addresses = np.asarray(
        [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids], dtype=np.int64
    )
    hand_dof_addresses = np.asarray(
        [int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids], dtype=np.int64
    )
    if not np.array_equal(hand_addresses, np.arange(HAND_DIM)):
        raise ValueError(f"compiled hand qpos addresses mismatch: {hand_addresses}")
    actuator_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) or ""
        for index in range(model.nu)
    )
    if actuator_names != tuple(contracts.JOINT_NAMES_28):
        raise ValueError(f"compiled actuator order mismatch: {actuator_names}")
    from scripts.eval.mano_joint_limits import compiled_hand_joint_limits

    limits = compiled_hand_joint_limits(model, contracts.JOINT_NAMES_28, joint_ids)
    renderer = None
    if create_renderer:
        model.vis.global_.offwidth = int(width)
        model.vis.global_.offheight = int(height)
        renderer = mujoco.Renderer(model, height=height, width=width)
    return (
        None,
        model,
        data,
        renderer,
        int(model.jnt_qposadr[object_joint]),
        int(model.jnt_dofadr[object_joint]),
        hand_addresses,
        hand_dof_addresses,
        limits,
    )


def _contact_ids(model: Any, object_name: str) -> tuple[set[int], set[int], int]:
    key = id(model)
    cached = _CONTACT_IDS.get(key)
    if cached is not None:
        return cached
    import mujoco

    object_body = object_body_id(model, object_name)
    floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    object_geoms = {
        index
        for index in range(model.ngeom)
        if int(model.geom_bodyid[index]) == object_body
        and (int(model.geom_contype[index]) or int(model.geom_conaffinity[index]))
    }
    hand_geoms = {
        index
        for index in range(model.ngeom)
        if index != floor_geom
        and index not in object_geoms
        and int(model.geom_contype[index]) == 1
    }
    if floor_geom < 0 or not object_geoms or len(hand_geoms) != 16:
        raise ValueError(
            f"compiled contact partition mismatch: floor={floor_geom} "
            f"object={len(object_geoms)} hand={len(hand_geoms)}"
        )
    cached = hand_geoms, object_geoms, int(floor_geom)
    _CONTACT_IDS[key] = cached
    return cached


def resolve_state41_feature_ids(model: Any, object_name: str) -> State41FeatureIds:
    """Resolve contact and non-colliding measurement geometry for state41."""
    key = id(model)
    cached = _STATE41_IDS.get(key)
    if cached is not None:
        return cached
    import mujoco
    from scripts import mano_state41_contract as contract

    hand_geoms, object_geoms, floor_geom = _contact_ids(model, object_name)
    geom_id_to_finger: dict[int, str] = {}
    keypoint_bodies: set[str] = set()
    for geom_id in hand_geoms:
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name not in _STATE41_KEYPOINT_LINKS:
            raise ValueError(
                f"state41 hand collision geom has unexpected body {body_name!r}"
            )
        keypoint_bodies.add(body_name)
        for finger in contract.FINGER_NAMES:
            if body_name.startswith(finger):
                geom_id_to_finger[int(geom_id)] = finger
                break
    if keypoint_bodies != set(_STATE41_KEYPOINT_LINKS):
        raise ValueError(
            "state41 keypoint body mismatch: "
            f"missing={sorted(set(_STATE41_KEYPOINT_LINKS)-keypoint_bodies)} "
            f"extra={sorted(keypoint_bodies-set(_STATE41_KEYPOINT_LINKS))}"
        )

    tip_geoms: list[int] = []
    for finger in contract.FINGER_NAMES:
        body_name = _STATE41_FINGERTIP_LINKS[finger]
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"state41 missing fingertip body {body_name!r}")
        candidates = [
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) == body_id
            and int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
            and int(model.geom_contype[geom_id]) == 0
            and int(model.geom_conaffinity[geom_id]) == 0
        ]
        if len(candidates) != 1:
            names = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, value)
                for value in candidates
            ]
            raise ValueError(
                f"state41 expected one non-colliding sphere on {body_name}, got {names}"
            )
        tip_geoms.append(int(candidates[0]))
    result = State41FeatureIds(
        hand_geom_ids=frozenset(int(value) for value in hand_geoms),
        object_geom_ids=tuple(sorted(int(value) for value in object_geoms)),
        geom_id_to_finger=geom_id_to_finger,
        fingertip_measurement_geom_ids=tuple(tip_geoms),
        floor_geom_id=int(floor_geom),
    )
    _STATE41_IDS[key] = result
    return result


def state41_features_from_mujoco(
    model: Any,
    data: Any,
    object_name: str,
    *,
    feature_ids: State41FeatureIds | None = None,
) -> tuple[np.ndarray, np.ndarray, np.float32, list[dict[str, Any]]]:
    """Return simulated contact5, signed distance5, floor support and pairs."""
    import mujoco
    from scripts import mano_state41_contract as contract

    ids = feature_ids or resolve_state41_feature_ids(model, object_name)
    contacts = np.zeros(len(contract.FINGER_NAMES), dtype=np.float32)
    object_floor = False
    pair_records: list[dict[str, Any]] = []
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        first, second = int(contact.geom1), int(contact.geom2)
        if ids.floor_geom_id in (first, second) and (
            first in ids.object_geom_ids or second in ids.object_geom_ids
        ):
            object_floor = True
        if first in ids.hand_geom_ids and second in ids.object_geom_ids:
            hand_geom, object_geom, normal_sign = first, second, 1.0
        elif second in ids.hand_geom_ids and first in ids.object_geom_ids:
            hand_geom, object_geom, normal_sign = second, first, -1.0
        else:
            continue
        finger = ids.geom_id_to_finger.get(hand_geom)
        if finger is not None:
            contacts[contract.FINGER_NAMES.index(finger)] = 1.0
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, contact_index, force)
        normal = normal_sign * np.asarray(contact.frame[:3], dtype=np.float64)
        magnitude = abs(float(force[0]))
        force_world = normal * magnitude
        pair_records.append(
            {
                "finger": finger or "palm",
                "hand_geom": mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, hand_geom
                ) or str(hand_geom),
                "object_geom": mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, object_geom
                ) or str(object_geom),
                "position_world": np.asarray(contact.pos, dtype=np.float32).tolist(),
                "normal_force_world": force_world.astype(np.float32).tolist(),
                "normal_force_norm": np.float32(magnitude).item(),
                "signed_distance_m": np.float32(contact.dist).item(),
            }
        )

    signed = np.empty(len(ids.fingertip_measurement_geom_ids), dtype=np.float32)
    fromto = np.zeros(6, dtype=np.float64)
    for finger_index, tip_geom in enumerate(ids.fingertip_measurement_geom_ids):
        distances = [
            float(
                mujoco.mj_geomDistance(
                    model, data, tip_geom, object_geom, 1.0, fromto
                )
            )
            for object_geom in ids.object_geom_ids
        ]
        distance = min(distances)
        if not np.isfinite(distance) or distance >= 1.0:
            raise FloatingPointError(
                f"invalid state41 fingertip/object distance for geom {tip_geom}: {distance}"
            )
        signed[finger_index] = np.float32(distance)
    return contacts, signed, np.float32(object_floor), pair_records


def render_current_state(model: Any, data: Any, renderer: Any) -> tuple[np.ndarray, np.ndarray]:
    """Render Head/Wrist from the current native MjData without stepping."""
    from scripts.eval import mano_action_support as visual

    renderer.update_scene(data, camera=visual.HEAD_CAMERA_NAME)
    head = renderer.render().copy()
    renderer.update_scene(data, camera=visual.WRIST_CAMERA_NAME)
    wrist = renderer.render().copy()
    return head, wrist


def nearest_wrapped_position_target(
    current: np.ndarray, command: np.ndarray, limits: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the inherited nearest-Euler wrapping and compiled limit diagnostics."""
    from scripts.eval.mano_joint_limits import clip_hand_state

    current_array = np.asarray(current, dtype=np.float64)
    command_array = np.asarray(command, dtype=np.float64)
    if current_array.shape != (HAND_DIM,) or command_array.shape != (HAND_DIM,):
        raise ValueError(
            f"position target requires {(HAND_DIM,)} current/command, got "
            f"{current_array.shape}/{command_array.shape}"
        )
    target = current_array + command_array
    target[3:6] = current_array[3:6] + (
        target[3:6] - current_array[3:6] + np.pi
    ) % (2 * np.pi) - np.pi
    return clip_hand_state(target, limits)


def contact_types(model: Any, data: Any, object_name: str) -> set[str]:
    hand_geoms, object_geoms, floor_geom = _contact_ids(model, object_name)
    result: set[str] = set()
    for contact in data.contact[: data.ncon]:
        first, second = int(contact.geom1), int(contact.geom2)
        pair = {first, second}
        if (first in hand_geoms and second in object_geoms) or (
            second in hand_geoms and first in object_geoms
        ):
            result.add("hand_object")
        if floor_geom in pair and pair & object_geoms:
            result.add("object_floor")
        if floor_geom in pair and pair & hand_geoms:
            result.add("hand_floor")
    return result


def step_servo(
    *,
    model: Any,
    data: Any,
    target: np.ndarray,
    substeps: int,
    object_name: str | None = None,
) -> dict[str, Any]:
    """Write one absolute 28D target unchanged and advance native MuJoCo."""
    import mujoco

    target_array = np.asarray(target, dtype=np.float64)
    if model.nu != HAND_DIM or target_array.shape != (HAND_DIM,):
        raise ValueError("absolute target/model shape mismatch")
    if substeps != NATIVE_SUBSTEPS:
        raise ValueError(f"native target requires exactly {NATIVE_SUBSTEPS} substeps")
    started = float(data.time)
    data.ctrl[:] = target_array
    if not np.array_equal(np.asarray(data.ctrl), target_array):
        raise RuntimeError("data.ctrl assignment modified the source target")
    data.qfrc_applied[:] = 0
    max_ncon = 0
    max_force = 0.0
    max_actuator = 0.0
    contacts: set[str] = set()
    for _ in range(substeps):
        mujoco.mj_step(model, data)
        if object_name:
            contacts.update(contact_types(model, data, object_name))
        max_ncon = max(max_ncon, int(data.ncon))
        max_actuator = max(
            max_actuator, float(np.max(np.abs(data.actuator_force)))
        )
        for index in range(data.ncon):
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, index, force)
            max_force = max(max_force, float(np.linalg.norm(force[:3])))
    expected = started + model.opt.timestep * substeps
    if not np.isclose(data.time, expected, rtol=0.0, atol=1e-12):
        raise FloatingPointError(f"MuJoCo time mismatch: {data.time} != {expected}")
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise FloatingPointError("MuJoCo non-finite state")
    warnings = [
        {
            "index": index,
            "number": int(warning.number),
            "lastinfo": int(warning.lastinfo),
        }
        for index, warning in enumerate(data.warning)
        if int(warning.number)
    ]
    return {
        "max_ncon": max_ncon,
        "max_contact_force": max_force,
        "max_abs_actuator_force": max_actuator,
        "hand_object_contact": "hand_object" in contacts,
        "object_floor_contact": "object_floor" in contacts,
        "hand_floor_contact": "hand_floor" in contacts,
        "warnings": warnings,
        "command_assignment_exact": True,
    }
