from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from scripts.eval import manorl_native_physics as physics


MANORL_ROOT = Path(os.environ.get("MANORL_REPO_ROOT", physics.DEFAULT_MANORL_REPO_ROOT))
CUBE_VISUAL = (
    MANORL_ROOT
    / "assets/all_assets/Assets/sim/mano_assets/objects/cube1/cube1.obj"
)
pytestmark = pytest.mark.skipif(
    not CUBE_VISUAL.is_file(), reason="pinned ManoRL cube1 visual assets are unavailable"
)


@pytest.mark.parametrize("preset", ("current", "legacy"))
def test_legacy_visual_contract_adds_only_visual_elements(preset):
    import mujoco
    from export_mano_sim_video import horizontal_to_vertical_fov
    from scripts.eval import mano_action_support as legacy

    head_camera = legacy.head_camera_config(preset)
    _, model, invariance = physics._legacy_visual_model(
        "cube1", 640, 360, head_camera_preset=preset
    )

    assert invariance["verified"] is True
    assert invariance["joint_order_unchanged"] is True
    assert invariance["actuator_order_unchanged"] is True
    assert invariance["collision_geoms_unchanged"] is True
    assert invariance["body_inertials_unchanged"] is True
    assert invariance["solver_options_unchanged"] is True
    assert invariance["camera_count"] == 2
    assert invariance["light_count"] == 4
    assert invariance["legacy_visual_contract"]["skybox"] is None
    assert invariance["legacy_visual_contract"]["head_camera_preset"] == preset

    head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, legacy.HEAD_CAMERA_NAME)
    wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, legacy.WRIST_CAMERA_NAME)
    np.testing.assert_array_equal(model.cam_pos[head], head_camera["position"])
    np.testing.assert_array_equal(model.cam_pos[wrist], legacy.WRIST_CAMERA["position"])
    assert model.cam_fovy[head] == pytest.approx(
        horizontal_to_vertical_fov(head_camera["horizontal_fov"], 640, 360)
    )
    assert model.cam_fovy[wrist] == pytest.approx(
        horizontal_to_vertical_fov(legacy.WRIST_CAMERA["horizontal_fov"], 640, 360)
    )
    light_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_LIGHT, index)
        for index in range(model.nlight)
    ]
    assert light_names == ["key_light", "fill_light", "rim_light", "top_light"]


def test_object_body_resolution_uses_authoritative_runtime_name():
    import mujoco

    _, model, *_ = physics.make_scene(
        "iphone", 1, 1, physics=True, create_renderer=False
    )
    body_id = physics.object_body_id(model, "iphone")
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) == "iphone17"
    hand_geoms, object_geoms, floor_geom = physics._contact_ids(model, "iphone")
    assert len(hand_geoms) == 16
    assert object_geoms
    assert floor_geom >= 0


def test_head_camera_preset_registry_has_two_versions_and_current_default():
    from scripts.eval import mano_action_support as legacy

    assert tuple(legacy.HEAD_CAMERA_PRESETS) == ("current", "legacy")
    assert legacy.DEFAULT_HEAD_CAMERA_PRESET == "current"
    assert legacy.HEAD_CAMERA == legacy.head_camera_config("current")
    with pytest.raises(ValueError, match="unknown head camera preset"):
        legacy.head_camera_config("near_50deg")
