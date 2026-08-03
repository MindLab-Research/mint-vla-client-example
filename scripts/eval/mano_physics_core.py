#!/usr/bin/env python3
"""Canonical MANO MuJoCo physics core shared by the sole Mode4 evaluator.

Parameters match the validated 200 Hz quality replay contract.
"""
from __future__ import annotations
from itertools import combinations
from pathlib import Path
import os, tempfile
import xml.etree.ElementTree as ET

import export_mano_sim_video as scene_export
import mujoco
import numpy as np
from scripts.eval import mano_action_support as chunk_helper
from scripts.eval.mano_joint_limits import compiled_hand_joint_limits, clip_hand_state

HAND_DIM = 26

DT = 0.0025
NATIVE_SUBSTEPS = 2
FRICTIONLOSS = 0.1
ARMATURE = 0.01
FLOOR_TOP_Z = -0.001
SELF_COLLISION_GROUPS = {
    "thumb": ("thumb_cmc", "thumb_mcp", "thumb_ip"),
    "index": ("index_mcp", "index_pip", "index_dip"),
    "middle": ("middle_mcp", "middle_pip", "middle_dip"),
    "ring": ("ring_mcp", "ring_pip", "ring_dip"),
    "pinky": ("pinky_mcp", "pinky_pip", "pinky_dip"),
}


def row_data_fps(row):
    override = os.environ.get("MANO_PHYSICS_FPS_OVERRIDE")
    if override:
        return float(override)
    values = [
        float(v)
        for v in (
            row.get("trajectory_metadata", {}).get("data_fps"),
            row.get("episode_metadata", {}).get("fps"),
        )
        if v is not None
    ]
    if not values or any(v <= 0 for v in values):
        raise ValueError("physics replay requires positive row fps metadata")
    if len(values) == 2 and not np.isclose(values[0], values[1]):
        raise ValueError(f"trajectory/episode fps disagree: {values}")
    return values[0]


def physics_substeps_for_row(row, requested=0):
    fps = row_data_fps(row)
    exact = 1.0 / (fps * DT)
    derived = int(round(exact))
    if derived <= 0 or not np.isclose(derived, exact, rtol=0, atol=1e-9):
        raise ValueError(f"row fps {fps} incompatible with dt={DT}")
    if derived % NATIVE_SUBSTEPS:
        raise ValueError(f"source frame needs {derived} steps, not whole native targets")
    if requested not in (0, derived):
        raise ValueError(f"requested {requested} substeps; row requires {derived}")
    return derived


def servo_parameters():
    kp = np.asarray([100.0] * 6 + [6.0, 4.0, 3.0, 3.0] * 5)
    dampratio = np.asarray([1.4] * 6 + [1.0] * 20)
    effort = np.asarray([5000.0] * 6 + [3.0, 2.0, 1.2, 1.2] * 5)
    return kp, dampratio, effort


def add_collision_geom(*, builder, body, collision, urdf_path, name, contact):
    geometry = collision.find("geometry")
    if geometry is None:
        raise ValueError(f"collision {name} has no geometry")
    attributes = {"name": name, **contact}
    mesh, box = geometry.find("mesh"), geometry.find("box")
    cylinder, sphere = geometry.find("cylinder"), geometry.find("sphere")
    if mesh is not None and "filename" in mesh.attrib:
        path = scene_export._resolve_mesh_path(urdf_path, mesh.attrib["filename"])
        attributes.update(type="mesh", mesh=builder.mesh_name(path, scene_export._mesh_scale(mesh)))
    elif box is not None and "size" in box.attrib:
        half = tuple(float(x) * 0.5 for x in box.attrib["size"].split())
        attributes.update(type="box", size=scene_export._fmt(half))
    elif cylinder is not None and {"radius", "length"} <= cylinder.attrib.keys():
        attributes.update(
            type="cylinder",
            size=scene_export._fmt(
                (float(cylinder.attrib["radius"]), float(cylinder.attrib["length"]) * 0.5)
            ),
        )
    elif sphere is not None and "radius" in sphere.attrib:
        attributes.update(type="sphere", size=sphere.attrib["radius"])
    else:
        raise ValueError(f"unsupported collision geometry for {name}")
    scene_export._geom_origin_attrib(collision, attributes)
    ET.SubElement(body, "geom", attributes)


def configure_physics(builder, *, hand_urdf, object_urdf, object_name, hand_joints):
    option = builder.root.find("option")
    if option is None:
        raise ValueError("scene missing option")
    option.attrib.clear()
    option.attrib.update(timestep=str(DT), gravity="0 0 -9.81", integrator="implicitfast")
    floor = next((g for g in builder.root.iter("geom") if g.attrib.get("name") == "floor"), None)
    if floor is None:
        raise ValueError("scene missing floor")
    floor.attrib.update(
        pos=f"0 0 {FLOOR_TOP_Z}", contype="4", conaffinity="1", condim="3", friction="1 0.01 0.001"
    )

    hand_root = ET.parse(hand_urdf).getroot()
    collision_count = 0
    for link in hand_root.findall("link"):
        body = builder.root.find(f".//body[@name='{link.attrib['name']}']")
        if body is None:
            raise ValueError(f"missing hand body {link.attrib['name']}")
        body.attrib["gravcomp"] = "1"
        for index, collision in enumerate(link.findall("collision")):
            add_collision_geom(
                builder=builder,
                body=body,
                collision=collision,
                urdf_path=hand_urdf,
                name=f"{link.attrib['name']}_collision_{index}",
                contact={
                    "contype": "1",
                    "conaffinity": "7",
                    "condim": "3",
                    "friction": "1 0.01 0.001",
                    "rgba": "0.88 0.58 0.46 0",
                },
            )
            collision_count += 1
    if not collision_count:
        raise ValueError("hand URDF has no collision geometry")

    object_body = builder.root.find(f".//body[@name='{object_name}_body']")
    object_link = ET.parse(object_urdf).getroot().find("link")
    if object_body is None or object_link is None:
        raise ValueError("object body/link missing")
    collisions = object_link.findall("collision")
    if not collisions:
        raise ValueError("object URDF has no collision geometry")
    for index, collision in enumerate(collisions):
        add_collision_geom(
            builder=builder,
            body=object_body,
            collision=collision,
            urdf_path=object_urdf,
            name=f"{object_name}_collision_{index}",
            contact={
                "contype": "2",
                "conaffinity": "5",
                "condim": "3",
                "friction": "0.9 0.01 0.001",
                "rgba": "0.8 0.18 0.16 0",
            },
        )
    src, dst = object_link.find("inertial"), object_body.find("inertial")
    if src is None or dst is None:
        raise ValueError("object inertial contract missing")
    origin, mass, inertia = src.find("origin"), src.find("mass"), src.find("inertia")
    if mass is None or inertia is None:
        raise ValueError("object mass/tensor missing")
    dst.attrib.clear()
    dst.attrib.update(
        pos=scene_export._fmt(scene_export._parse_xyz(origin, "xyz", (0, 0, 0))),
        mass=mass.attrib["value"],
        diaginertia=scene_export._fmt(
            tuple(float(inertia.attrib[n]) for n in ("ixx", "iyy", "izz"))
        ),
    )

    contact = builder.root.find("contact") or ET.SubElement(builder.root, "contact")
    body_names = {b.attrib.get("name") for b in builder.root.iter("body")}
    for group_name, group in SELF_COLLISION_GROUPS.items():
        if not set(group) <= body_names:
            raise ValueError(f"missing bodies for {group_name} excludes")
        for index, (first, second) in enumerate(combinations(group, 2)):
            ET.SubElement(
                contact, "exclude", name=f"{group_name}_internal_{index}", body1=first, body2=second
            )

    joints = {j.attrib.get("name"): j for j in builder.root.iter("joint")}
    for name in hand_joints:
        joints[name].attrib.update(
            damping="0", frictionloss=str(FRICTIONLOSS), armature=str(ARMATURE)
        )
    kp, dampratio, effort = servo_parameters()
    actuator = ET.SubElement(builder.root, "actuator")
    for name, k, d, e in zip(hand_joints, kp, dampratio, effort, strict=True):
        ET.SubElement(
            actuator,
            "position",
            name=f"servo_{name}",
            joint=name,
            kp=f"{k:.17g}",
            dampratio=f"{d:.17g}",
            inheritrange="1",
            forcelimited="true",
            forcerange=f"{-e:.17g} {e:.17g}",
        )


def make_scene(
    object_name, width, height, *, physics=False, physics_timestep=DT, create_renderer=True
):
    object_urdf = chunk_helper.DEFAULT_OBJECTS_URDF_DIR / f"{object_name}.urdf"
    builder, object_joint, hand_joints = chunk_helper.build_scene(
        hand_urdf=chunk_helper.DEFAULT_HAND_URDF,
        object_urdf=object_urdf,
        object_code=object_name,
        width=width,
        height=height,
    )
    if physics:
        configure_physics(
            builder,
            hand_urdf=chunk_helper.DEFAULT_HAND_URDF,
            object_urdf=object_urdf,
            object_name=object_name,
            hand_joints=hand_joints,
        )
    tmp = tempfile.TemporaryDirectory(prefix=f"mano_rollout_{object_name}_")
    xml = Path(tmp.name) / "scene.xml"
    builder.write(xml)
    model = mujoco.MjModel.from_xml_path(str(xml))
    if physics and not np.isclose(model.opt.timestep, physics_timestep):
        raise ValueError(f"compiled dt {model.opt.timestep} != requested {physics_timestep}")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width) if create_renderer else None
    oid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, object_joint)
    object_addr, object_dof_addr = int(model.jnt_qposadr[oid]), int(model.jnt_dofadr[oid])
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in hand_joints]
    hand_addrs = [int(model.jnt_qposadr[i]) for i in ids]
    hand_dof_addrs = [int(model.jnt_dofadr[i]) for i in ids]
    limits = compiled_hand_joint_limits(model, hand_joints, ids)
    if physics:
        kp, _, effort = servo_parameters()
        if model.nu != HAND_DIM:
            raise ValueError(f"expected {HAND_DIM} actuators, got {model.nu}")
        for aid, (jid, k, e) in enumerate(zip(ids, kp, effort, strict=True)):
            did = int(model.jnt_dofadr[jid])
            if (
                not np.isclose(model.dof_damping[did], 0)
                or not np.isclose(model.dof_armature[did], ARMATURE)
                or not np.isclose(model.dof_frictionloss[did], FRICTIONLOSS)
            ):
                raise ValueError("compiled joint dynamics mismatch")
            if not np.isclose(model.actuator_gainprm[aid, 0], k):
                raise ValueError("compiled actuator kp mismatch")
            if not model.actuator_forcelimited[aid] or not np.allclose(
                model.actuator_forcerange[aid], (-e, e)
            ):
                raise ValueError("compiled actuator effort mismatch")
    return (
        tmp,
        model,
        data,
        renderer,
        object_addr,
        object_dof_addr,
        hand_addrs,
        hand_dof_addrs,
        limits,
    )


def render_current_state(model, data, renderer):
    """Render an already-forwarded/integrated state without changing dynamics caches."""
    renderer.update_scene(data, camera=chunk_helper.HEAD_CAMERA_NAME)
    head = renderer.render().copy()
    renderer.update_scene(data, camera=chunk_helper.WRIST_CAMERA_NAME)
    return head, renderer.render().copy()


def nearest_wrapped_position_target(current, command, limits):
    target = np.asarray(current, dtype=np.float64) + np.asarray(command, dtype=np.float64)
    target[3:6] = current[3:6] + (target[3:6] - current[3:6] + np.pi) % (2 * np.pi) - np.pi
    return clip_hand_state(target, limits)


def contact_types(model, data, object_name):
    result = set()
    for contact in data.contact[: data.ncon]:
        names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or ""
            for g in (contact.geom1, contact.geom2)
        }
        has_object = any(n.startswith(f"{object_name}_collision_") for n in names)
        has_floor = "floor" in names
        has_hand = any("_collision_" in n and not n.startswith(f"{object_name}_") for n in names)
        if has_hand and has_object:
            result.add("hand_object")
        if has_object and has_floor:
            result.add("object_floor")
        if has_hand and has_floor:
            result.add("hand_floor")
    return result


def step_servo(*, model, data, target, substeps, object_name=None):
    if model.nu != HAND_DIM or np.asarray(target).shape != (HAND_DIM,):
        raise ValueError("servo target/model shape mismatch")
    started = float(data.time)
    data.ctrl[:] = np.asarray(target, dtype=np.float64)
    data.qfrc_applied[:] = 0
    max_ncon = 0
    max_force = 0.0
    max_actuator = 0.0
    contacts = set()
    for _ in range(substeps):
        mujoco.mj_step(model, data)
        if object_name:
            contacts.update(contact_types(model, data, object_name))
        max_ncon = max(max_ncon, int(data.ncon))
        max_actuator = max(max_actuator, float(np.max(np.abs(data.actuator_force))))
        for index in range(data.ncon):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            max_force = max(max_force, float(np.linalg.norm(force[:3])))
    if not np.isclose(data.time, started + model.opt.timestep * substeps, rtol=0, atol=1e-10):
        raise FloatingPointError("MuJoCo time mismatch")
    # mj_step leaves derived kinematics/contact data at its pre-integration
    # stage. Refresh so policy state, force extraction, and rendering describe
    # the same post-integration timestamp as data.qpos.
    mujoco.mj_forward(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise FloatingPointError("MuJoCo non-finite state")
    return {
        "max_ncon": max_ncon,
        "max_contact_force": max_force,
        "max_abs_actuator_force": max_actuator,
        "hand_object_contact": "hand_object" in contacts,
        "object_floor_contact": "object_floor" in contacts,
        "hand_floor_contact": "hand_floor" in contacts,
    }


# The 16 MANO keypoint link names that have collision geoms.
# palm is included for geom-ID resolution but ignored in the 5-finger state.
MANO_KEYPOINT_LINKS = (
    "thumb_cmc", "thumb_mcp", "thumb_ip",
    "index_mcp", "index_pip", "index_dip",
    "middle_mcp", "middle_pip", "middle_dip",
    "ring_mcp", "ring_pip", "ring_dip",
    "pinky_mcp", "pinky_pip", "pinky_dip",
    "palm",
)

def _link_to_finger(link_name: str) -> str | None:
    """Map a URDF link name to its finger, or None for palm/unknown."""
    for finger in ("index", "thumb", "middle", "ring", "pinky"):
        if link_name.startswith(finger):
            return finger
    return None


def resolve_keypoint_geom_ids(model, object_name):
    """Resolve the 16 keypoint collision geom IDs + object collision geom IDs once.

    Returns (keypoint_geom_ids: set[int], object_geom_ids: set[int],
             geom_id_to_finger: dict[int, str]).
    Raises if the keypoint count doesn't match MANO_KEYPOINT_LINKS.
    """
    keypoint_ids = set()
    object_ids = set()
    geom_id_to_finger = {}
    for gi in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        if name.startswith(f"{object_name}_collision_"):
            object_ids.add(gi)
            continue
        if "_collision_" not in name:
            continue
        link = name.split("_collision_")[0]
        if link in MANO_KEYPOINT_LINKS:
            keypoint_ids.add(gi)
            finger = _link_to_finger(link)
            if finger is not None:
                geom_id_to_finger[gi] = finger
    expected = len(MANO_KEYPOINT_LINKS)
    if len(keypoint_ids) != expected:
        raise ValueError(
            f"expected {expected} keypoint collision geoms, found {len(keypoint_ids)}: "
            f"{sorted(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g) for g in keypoint_ids)}"
        )
    if not object_ids:
        raise ValueError(f"no collision geoms found for object {object_name!r}")
    return keypoint_ids, object_ids, geom_id_to_finger


def finger_contacts_from_mujoco(model, data, object_name, *,
                                keypoint_geom_ids=None, object_geom_ids=None,
                                geom_id_to_finger=None):
    """Compute per-finger binary contacts from live MuJoCo contact pairs.

    v1 contract: binary contact = contact pair EXISTS between a keypoint geom and
    the object geom. NO force_norm threshold, NO mj_contactForce call.
    Palm geoms participate in 16-keypoint validation but are not output.
    Returns (5,) float32 array in FINGER_NAMES order: index/thumb/ring/middle/pinky.
    """
    from scripts.mano_state_contract import FINGER_NAMES
    if keypoint_geom_ids is None or object_geom_ids is None or geom_id_to_finger is None:
        keypoint_geom_ids, object_geom_ids, geom_id_to_finger = resolve_keypoint_geom_ids(
            model, object_name
        )
    contacts = np.zeros(len(FINGER_NAMES), dtype=np.float32)
    for ci in range(data.ncon):
        contact = data.contact[ci]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        # One geom must be object, the other must be a keypoint.
        if g1 in object_geom_ids and g2 in keypoint_geom_ids:
            hand_geom = g2
        elif g2 in object_geom_ids and g1 in keypoint_geom_ids:
            hand_geom = g1
        else:
            continue
        finger = geom_id_to_finger.get(hand_geom)
        if finger is not None:
            contacts[FINGER_NAMES.index(finger)] = 1.0
    return contacts


def finger_contact_and_force_from_mujoco(
    model, data, object_name, *, keypoint_geom_ids=None, object_geom_ids=None,
    geom_id_to_finger=None,
):
    """Return state54 binary contacts and log1p summed normal loads by finger."""
    from scripts.mano_state54_contract import FINGER_NAMES

    if keypoint_geom_ids is None or object_geom_ids is None or geom_id_to_finger is None:
        keypoint_geom_ids, object_geom_ids, geom_id_to_finger = resolve_keypoint_geom_ids(
            model, object_name
        )
    contacts = np.zeros(len(FINGER_NAMES), dtype=np.float32)
    loads = np.zeros(len(FINGER_NAMES), dtype=np.float64)
    force = np.zeros(6, dtype=np.float64)
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if g1 in object_geom_ids and g2 in keypoint_geom_ids:
            hand_geom = g2
        elif g2 in object_geom_ids and g1 in keypoint_geom_ids:
            hand_geom = g1
        else:
            continue
        finger = geom_id_to_finger.get(hand_geom)
        if finger is None:
            continue
        finger_index = FINGER_NAMES.index(finger)
        contacts[finger_index] = 1.0
        force[:] = 0.0
        mujoco.mj_contactForce(model, data, contact_index, force)
        normal_load = float(force[0])
        if not np.isfinite(normal_load) or normal_load < -1e-8:
            raise FloatingPointError(f"invalid MuJoCo normal contact load {normal_load}")
        loads[finger_index] += max(0.0, normal_load)
    return contacts, np.log1p(loads).astype(np.float32)
