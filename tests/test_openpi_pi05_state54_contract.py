import pytest

from mint_server.backend.openpi.openpi_pi05_worker import OpenPIPi05WorkerSession
from mint_server.backend.openpi.pi05_profiles import (
    PI05_ACTION_LORA_R16_STATE54_V1,
    PI05_ACTION_LORA_R16_V1,
)


def test_legacy_manifest_hash_is_byte_stable():
    assert PI05_ACTION_LORA_R16_V1.manifest_hash == "0e3acdc1a5be96819fd828eea494093447a966bb1c64fa264ac65a65e3712218"
    assert "state_dim" not in PI05_ACTION_LORA_R16_V1.manifest()
    assert "fail_on_token_truncation" not in PI05_ACTION_LORA_R16_V1.manifest()


def test_state54_profile_keeps_action_projection_width_and_fails_closed():
    profile = PI05_ACTION_LORA_R16_STATE54_V1
    kwargs = profile.pi0_config_kwargs()
    assert kwargs["state_dim"] == 54
    assert kwargs["action_dim"] == 32
    assert kwargs["max_token_len"] == 256
    assert kwargs["fail_on_token_truncation"] is True


def test_training_worker_refuses_prompt_overflow_instead_of_truncating():
    session = object.__new__(OpenPIPi05WorkerSession)
    session._max_token_len = 4
    with pytest.raises(ValueError, match="without truncation"):
        session._padded_prompt_arrays([1, 2, 3, 4, 5], [True] * 5)
    assert session._padded_prompt_arrays([1, 2], [True, True]) == (
        [1, 2, 0, 0],
        [True, True, False, False],
    )
