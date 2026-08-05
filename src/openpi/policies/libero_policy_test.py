import numpy as np
import pytest

from openpi.policies import libero_policy
from openpi.training import config as training_config


def test_libero_outputs_preserves_legacy26_default():
    actions=np.arange(10*32,dtype=np.float32).reshape(10,32)
    result=libero_policy.LiberoOutputs()({'actions':actions})['actions']
    assert result.shape==(10,26)
    assert np.array_equal(result,actions[:,:26])


def test_libero_outputs_returns_native28_without_pad4():
    actions=np.arange(10*32,dtype=np.float32).reshape(10,32)
    result=libero_policy.LiberoOutputs(physical_action_dim=28)({'actions':actions})['actions']
    assert result.shape==(10,28)
    assert np.array_equal(result,actions[:,:28])


def test_libero_outputs_rejects_invalid_physical_width():
    with pytest.raises(ValueError,match='outside model action width'):
        libero_policy.LiberoOutputs(physical_action_dim=33)({'actions':np.zeros((10,32),dtype=np.float32)})


def test_state56_native28_config_is_distinct_and_exact():
    config=training_config.get_config('pi05_libero_state56_native28')
    assert config.model.state_dim==56
    assert config.model.action_dim==32
    assert config.model.action_horizon==10
    assert config.model.max_token_len==256
    assert config.model.discrete_state_input is True
    assert config.data.output_action_dim==28
    legacy=training_config.get_config('pi05_libero')
    assert legacy.data.output_action_dim==26
