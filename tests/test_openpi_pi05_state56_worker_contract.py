import numpy as np
import pytest

from mint_server.backend.openpi.openpi_pi05_action_worker import OpenPIPi05ActionSession


def _action_session() -> OpenPIPi05ActionSession:
    session = OpenPIPi05ActionSession.__new__(OpenPIPi05ActionSession)
    session._state_dim = 56
    session._action_dim = 32
    session._max_token_len = 256
    session._camera_layout = ()
    return session


def _payload(width: int, token_count: int = 2) -> dict:
    return {
        "observation": {"chunks": [{"type": "encoded_text", "tokens": list(range(token_count))}]},
        "extra_inputs": {"state": {"shape": [width], "data": [0.25] * width}},
    }


def test_action_worker_accepts_exact_state56_action32_profile() -> None:
    observation = _action_session()._observation_numpy_from_payload(_payload(56))
    state = np.asarray(observation["state"], dtype=np.float32)
    assert state.shape == (56,)
    assert np.array_equal(state, np.full(56, 0.25, dtype=np.float32))


@pytest.mark.parametrize("width", [32, 54, 55, 57])
def test_action_worker_rejects_wrong_state56_width(width: int) -> None:
    with pytest.raises(ValueError, match=r"exact width 56"):
        _action_session()._observation_numpy_from_payload(_payload(width))


def test_state56_action_worker_keeps_max_token256_fail_closed() -> None:
    with pytest.raises(ValueError, match=r"exceeds max_token_len without truncation"):
        _action_session()._observation_numpy_from_payload(_payload(56, token_count=257))
