from __future__ import annotations
import numpy as np
from scripts import mano_state56_contract as C
from scripts.eval import manorl_native28_physics as P


def test_all_state56_objects_compile_and_match_geometry_contract():
    for object_name, expected in C.OBJECT_COLLISION_BOXES.items():
        scene=P.compile_scene(object_name)
        assert scene.model.nq==35 and scene.model.nv==34 and scene.model.nu==28
        assert np.array_equal(scene.hand_qpos_addresses,np.arange(28))
        assert np.array_equal(scene.hand_qvel_addresses,np.arange(28))
        actual=P.compiled_collision_box(scene)
        assert np.allclose(actual.local_center,expected.local_center,rtol=0,atol=2e-12),object_name
        assert np.allclose(actual.half_extents,expected.half_extents,rtol=0,atol=2e-12),object_name


def test_snapshot_fk_is_finite_and_preserves_hand_object_pose():
    scene=P.compile_scene('cube1')
    qpos=np.zeros(28,dtype=np.float64);qpos[:3]=[.1,-.2,.3];qpos[3:6]=[.2,-.1,.4]
    position=np.array([.02,.03,.5]);quaternion=np.array([.9238795325,0,0,.3826834324])
    P.set_snapshot(scene,hand_qpos=qpos,object_position=position,object_quaternion_wxyz=quaternion,target28=qpos)
    assert np.array_equal(scene.data.qpos[scene.hand_qpos_addresses],qpos)
    address=scene.object_qpos_address
    assert np.array_equal(scene.data.qpos[address:address+3],position)
    assert np.allclose(scene.data.qpos[address+3:address+7],quaternion/np.linalg.norm(quaternion),rtol=0,atol=1e-15)
    tips=P.fingertip_world(scene)
    assert tips.shape==(5,3) and np.isfinite(tips).all()
    feature=C.fingertips_in_collision_box_frame(tips,position,C.quaternion_wxyz_to_matrix(quaternion),'cube1')
    assert feature.shape==(5,3) and np.isfinite(feature).all()
