from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from mint_server.backend.core.model_registry import MODEL_CONFIGS
from scripts.openpi_profiles import ACTION_LORA_R16_STATE44_MODEL, LEGACY_L_LORA_MODEL
from scripts.train import openpi_vla_smoke_lance_base as base
from scripts.train import train_cube1_01_compare as compare
from scripts.train.prepare_mano_state44_profile import raw_pi05_token_length


def test_augmentation_diagnostics_preserve_state44_width_across_workers() -> None:
    aggregate = compare.AugmentationDiagnostics(44)
    worker = compare.AugmentationDiagnostics(44)
    aggregate.merge_from(worker)
    assert len(aggregate.summary(0.0)["bin_changed_fraction_by_dimension"]) == 44
    with pytest.raises(ValueError, match="state dims"):
        aggregate.merge_from(compare.AugmentationDiagnostics(32))


def test_legacy_non_discrete_config_keeps_shared_width_fallback() -> None:
    config = base._build_model_config(
        10, action_dim=7, base_model=LEGACY_L_LORA_MODEL
    )
    assert config.state_dim == config.action_dim == 7


def test_state44_model_config_keeps_action_projection_width_32() -> None:
    config = base._build_model_config(
        10,
        state_dim=44,
        action_dim=32,
        base_model=ACTION_LORA_R16_STATE44_MODEL,
    )
    assert config.state_dim == 44
    assert config.action_dim == 32
    assert config.action_horizon == 10
    assert config.max_token_len == 200
    assert config.fail_on_token_truncation is True
    with pytest.raises(ValueError, match="state_dim=44"):
        base._build_model_config(
            10,
            state_dim=32,
            action_dim=32,
            base_model=ACTION_LORA_R16_STATE44_MODEL,
        )


def test_raw_token_audit_matches_public_tokenizer_before_padding() -> None:
    from openpi.models.tokenizer import PaligemmaTokenizer

    tokenizer = PaligemmaTokenizer(max_len=200, fail_on_truncation=True)
    state = np.linspace(-0.9, 0.9, 44, dtype=np.float32)
    prompt = "pick_up cube1\nusing gesture 09"
    raw_length = raw_pi05_token_length(tokenizer, prompt, state)
    _tokens, mask = tokenizer.tokenize(prompt, state)
    assert raw_length == int(np.count_nonzero(mask))
    assert raw_length < 200


def test_state44_wire_payload_is_state44_action32() -> None:
    model_config = MODEL_CONFIGS[ACTION_LORA_R16_STATE44_MODEL]
    item = {
        "state": np.zeros(44, dtype=np.float32),
        "actions": np.zeros((10, 32), dtype=np.float32),
        "tokenized_prompt": np.arange(200, dtype=np.int32),
        "tokenized_prompt_mask": np.arange(200) < 170,
        "image": {
            camera: np.zeros((16, 16, 3), dtype=np.float32)
            for camera in model_config.camera_layout
        },
    }
    wire = base._pi05_datum_from_transformed(ACTION_LORA_R16_STATE44_MODEL, item)
    assert wire["observation"]["state"]["shape"] == [44]
    assert wire["supervision"]["actions"]["shape"] == [10, 32]
    assert len(wire["observation"]["model_input"]["chunks"][-1]["tokens"]) == 170
    with pytest.raises(ValueError, match="state shape"):
        base._pi05_datum_from_transformed(
            ACTION_LORA_R16_STATE44_MODEL, {**item, "state": np.zeros(32, dtype=np.float32)}
        )


class _State44TokenizePrompt:
    def __call__(self, data):
        state = np.asarray(data["state"], dtype=np.float32)
        output = dict(data)
        output.pop("prompt", None)
        output["tokenized_prompt"] = np.rint((state + 2.0) * 100).astype(np.int32)
        output["tokenized_prompt_mask"] = np.ones(44, dtype=bool)
        return output


class _State44Dataset:
    _state_dim = 44
    _state_contract = "state44"
    _extended_state = True
    _action_horizon = 10
    _action_source = "urdf_target_absolute"

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"index": index}

    def state44_surface_distances_for_sample(self, key, augmented_hand_qpos):
        assert key == 0
        assert np.asarray(augmented_hand_qpos).shape == (26,)
        return np.asarray([0.11, 0.12, 0.13, 0.14, 0.15], dtype=np.float32)


def _state44_prepared() -> compare.PreparedDatum:
    state = np.zeros(44, dtype=np.float32)
    actions = np.zeros((10, 32), dtype=np.float32)
    prefix = {"state": state, "prompt": "cube", "actions": actions}
    clean = {
        "observation": {
            "state": {"data": state.tolist(), "shape": [44]},
            "model_input": {
                "chunks": [
                    {"type": "image", "data": ["image"]},
                    {"type": "encoded_text", "tokens": [1, 2, 3]},
                ]
            },
        },
        "supervision": {
            "actions": {"data": actions.tolist(), "shape": [10, 32]}
        },
    }
    return compare.PreparedDatum(prefix, _State44TokenizePrompt(), (), clean)


def test_stateaug_recomputes_distance_but_keeps_rate_history_clean() -> None:
    state_q01 = -np.ones(44, dtype=np.float32)
    state_q99 = np.ones(44, dtype=np.float32)
    state_q01[32:37] = 0.0
    state_q99[32:37] = 0.2
    stats = {
        "state": SimpleNamespace(q01=state_q01, q99=state_q99, std=np.ones(44)),
        "actions": SimpleNamespace(
            q01=-np.ones(32), q99=np.ones(32), std=np.ones(32)
        ),
    }
    noise = np.zeros(44, dtype=np.float32)
    noise[:26] = 0.05
    dataset = _State44Dataset()
    with patch.object(compare, "_prepare_discrete_datum", return_value=_state44_prepared()):
        wire = compare.build_batch(
            dataset,
            object(),
            base_model=ACTION_LORA_R16_STATE44_MODEL,
            indices=[0],
            norm_stats=stats,
            state_noise_std=0.05,
            rng=np.random.default_rng(0),
            planned_requests=[(0, noise, None)],
            datum_cache=compare.DatumCache(1),
        )[0]
    state = np.asarray(wire["observation"]["state"]["data"], dtype=np.float32)
    np.testing.assert_allclose(state[:26], 0.05)
    expected_surface = (
        (np.asarray([0.11, 0.12, 0.13, 0.14, 0.15]) - state_q01[32:37])
        / (state_q99[32:37] - state_q01[32:37] + 1e-6)
        * 2.0
        - 1.0
    )
    np.testing.assert_allclose(state[32:37], expected_surface, atol=1e-6)
    np.testing.assert_array_equal(state[37:42], 0.0)
    assert state.shape == (44,)
