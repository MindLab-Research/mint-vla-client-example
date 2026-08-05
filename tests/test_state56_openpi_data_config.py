import numpy as np

import openpi.transforms as transforms
from scripts.openpi_profiles import ACTION_LORA_R16_STATE56_28DOF_MODEL, resolve_profile
import openpi_vla_smoke_lance_base as lance_base


def test_state56_data_config_uses_native28_delta_mask_and_physical_output():
    profile=resolve_profile(ACTION_LORA_R16_STATE56_28DOF_MODEL)
    model=lance_base._build_model_config(10,action_dim=32,base_model=profile.base_model)
    config=lance_base._make_data_config(
        model,None,action_source='urdf_target_absolute',
        delta_mask_segments=profile.delta_mask_segments,
        physical_action_dim=profile.physical_action_dim,
    )
    delta=config.data_transforms.inputs[-1]
    absolute=config.data_transforms.outputs[0]
    physical=config.data_transforms.outputs[-1]
    assert isinstance(delta,transforms.DeltaActions)
    assert isinstance(absolute,transforms.AbsoluteActions)
    expected=np.asarray([True]*3+[False]*3+[True]*22+[False]*4)
    assert np.array_equal(np.asarray(delta.mask),expected)
    assert np.array_equal(np.asarray(absolute.mask),expected)
    assert physical.physical_action_dim==28
    state=np.arange(56,dtype=np.float32)
    target=np.arange(32,dtype=np.float32)[None,:]
    residual=delta({'state':state,'actions':target.copy()})['actions'][0]
    assert np.array_equal(residual[:3],np.zeros(3,dtype=np.float32))
    assert np.array_equal(residual[3:6],target[0,3:6])
    assert np.array_equal(residual[6:28],np.zeros(22,dtype=np.float32))
    assert np.array_equal(residual[28:],target[0,28:])
    output=physical({'actions':np.zeros((10,32),dtype=np.float32)})['actions']
    assert output.shape==(10,28)


def test_legacy26_data_config_remains_explicit_and_distinct():
    profile=resolve_profile(ACTION_LORA_R16_STATE56_28DOF_MODEL)
    model=lance_base._build_model_config(10,action_dim=32,base_model=profile.base_model)
    legacy=lance_base._make_data_config(
        model,None,action_source='urdf_target_absolute',
        delta_mask_segments=(3,-3,20,-6),physical_action_dim=26,
    )
    assert legacy.data_transforms.outputs[-1].physical_action_dim==26
    assert not np.asarray(legacy.data_transforms.inputs[-1].mask)[26]
