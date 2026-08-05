from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import pytest
from scripts import mano_state56_contract as C


def test_identites_and_layout():
    assert C.STATE_DIM == 56
    assert C.ACTION_DIM == 32
    assert C.ACTION_PHYSICAL_DIM == 28
    assert C.ACTION_DELTA_MASK == (3,-3,22,-4)
    assert C.FINGER_CONTACT_SLICE == slice(28,33)
    assert C.LIFT_HEIGHT_INDEX == 33
    assert C.FINGERTIP_OBJECT_SLICE == slice(34,49)
    assert C.FINGER_FORCE_SLICE == slice(49,54)
    assert C.RELATIVE_VERTICAL_VELOCITY_INDEX == 54
    assert C.MULTIFINGER_CONTACT_AGE_INDEX == 55
    assert set(C.OBJECT_COLLISION_BOXES) == {
        'banana','bowl','cube1','cube2','cylinder3','cylinder4','cylinder7',
        'iphone','largeclamp','mayonnaisebottle','powerdrill'
    }


def test_collision_box_center_and_faces_map_to_zero_and_unit():
    rotation=np.eye(3)
    for name,box in C.OBJECT_COLLISION_BOXES.items():
        center=box.local_center
        tips=np.repeat(center[None],5,axis=0)
        assert np.array_equal(C.fingertips_in_collision_box_frame(tips,np.zeros(3),rotation,name),np.zeros((5,3),dtype=np.float32))
        for axis in range(3):
            points=tips.copy();points[:,axis]+=box.half_extents[axis]
            expected=np.zeros((5,3),dtype=np.float32);expected[:,axis]=1
            assert np.allclose(C.fingertips_in_collision_box_frame(points,np.zeros(3),rotation,name),expected,rtol=0,atol=2e-7)


def test_quaternion_wxyz_matrix():
    assert np.allclose(C.quaternion_wxyz_to_matrix(np.array([1.,0,0,0])),np.eye(3),rtol=0,atol=1e-15)
    q=np.array([np.sqrt(.5),0,0,np.sqrt(.5)])
    expected=np.array([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
    assert np.allclose(C.quaternion_wxyz_to_matrix(q),expected,rtol=0,atol=3e-15)


def test_contact_force_aggregation_uses_all_pairs():
    frame={'finger_contacts':[1,1,0,0,0],'pair_count':3,'pairs':[
        {'finger':'index','normal_force_norm':2.0},
        {'finger':'index','normal_force_norm':3.0},
        {'finger':'thumb','normal_force_norm':4.0},
    ]}
    contacts,forces=C.aggregate_state41_contact_frame(frame)
    assert np.array_equal(contacts,np.array([1,1,0,0,0],dtype=np.float32))
    assert np.allclose(forces,np.log1p([5,4,0,0,0]),rtol=1e-7,atol=0)


def test_contact_pair_flag_disagreement_fails():
    with pytest.raises(ValueError,match='disagree'):
        C.aggregate_state41_contact_frame({'finger_contacts':[1,0,0,0,0],'pair_count':0,'pairs':[]})


def test_action_pad4_is_fail_closed():
    target=np.arange(28,dtype=np.float32)
    action=C.build_action32(target)
    assert np.array_equal(action[:28],target)
    assert np.array_equal(action[28:],np.zeros(4,dtype=np.float32))
    assert np.array_equal(C.extract_target28(action),target)
    action[31]=1
    with pytest.raises(ValueError,match='pad4'):
        C.extract_target28(action)


def test_window_temporal_semantics_and_layout():
    qpos=np.zeros((4,28),dtype=np.float32);qpos[:,2]=[.2,.2,.21,.21]
    contacts=np.array([[1,1,0,0,0],[1,1,0,0,0],[1,0,0,0,0],[1,1,0,0,0]],dtype=np.float32)
    force=np.zeros((4,5),dtype=np.float32);tips=np.zeros((4,5,3),dtype=np.float32)
    pos=np.zeros((4,3),dtype=np.float32);pos[:,2]=[.1,.11,.13,.13]
    state=C.build_state56_window_from_features(hand_qpos=qpos,finger_contacts=contacts,finger_log1p_force=force,fingertip_collision_box_xyz=tips,object_position_world=pos,window_start=1,window_end=3)
    assert state.shape==(3,56)
    assert np.allclose(state[:,33],np.array([.01,.03,.03],dtype=np.float32),rtol=0,atol=2e-8)
    assert state[0,54]==0
    assert np.isclose(state[1,54],2.0,rtol=0,atol=3e-6)
    assert np.isclose(state[2,54],0.0,rtol=0,atol=3e-6)
    assert np.array_equal(state[:,55],np.zeros(3,dtype=np.float32))


def test_state_assembly_has_exact_layout():
    state=C.build_state56(hand_qpos=np.arange(28),finger_contacts=np.ones(5),lift_height=2,fingertip_collision_box_xyz=np.arange(15).reshape(5,3),finger_log1p_force=np.arange(5),relative_vertical_velocity=3,multifinger_contact_age=.5)
    assert np.array_equal(state[:28],np.arange(28,dtype=np.float32))
    assert np.array_equal(state[28:33],np.ones(5,dtype=np.float32))
    assert state[33]==2
    assert np.array_equal(state[34:49],np.arange(15,dtype=np.float32))
    assert np.array_equal(state[49:54],np.arange(5,dtype=np.float32))
    assert state[54]==3 and state[55]==.5


def test_norm_authentication_requires_explicit_contract(tmp_path: Path):
    norm={'norm_stats':{'state':{'q01':[0]*56,'q99':[1]*56},'actions':{'q01':[0]*32,'q99':[1]*32}}}
    norm_path=tmp_path/'norm_stats.json';norm_path.write_text(json.dumps(norm))
    norm_sha=hashlib.sha256(norm_path.read_bytes()).hexdigest()
    audit={'zero_truncation':True,'overflow_count':0,'maximum_token_length':245,'norm_stats_sha256':norm_sha,'augmentation':{'zero_truncation':True,'overflow_count':0,'maximum_token_length':245,'seed':43,'state_noise_std':.05,'target_noise_std':0.0}}
    audit_path=tmp_path/'token_audit.json';audit_path.write_text(json.dumps(audit));audit_sha=hashlib.sha256(audit_path.read_bytes()).hexdigest()
    contract={'state_contract':C.STATE_CONTRACT_ID,'action_contract':C.ACTION_CONTRACT_ID,'state_dim':56,'action_dim':32,'action_horizon':10,'action_source':'urdf_target_absolute','action_physical_dim':28,'action_padding_dim':4,'action_delta_mask':[3,-3,22,-4],'norm_stats_sha256':norm_sha,'geometry_contract_sha256':C.GEOMETRY_CONTRACT_SHA256,'source_interval_seconds':.005,'contact_age_clip_seconds':1.0,'max_token_len':256,'train_trajectory_count':4613,'validation_trajectory_count':243,'held_out_trajectory_count':0,'token_audit':str(audit_path),'token_audit_sha256':audit_sha}
    contract_path=tmp_path/'data_contract.json';contract_path.write_text(json.dumps(contract))
    assert C.verify_locked_state56_norm_stats(tmp_path,expected_sha256=norm_sha,data_contract_path=contract_path)==(norm_path,norm_sha)
    with pytest.raises(ValueError,match='norm SHA mismatch'):
        C.verify_locked_state56_norm_stats(tmp_path,expected_sha256='0'*64,data_contract_path=contract_path)
