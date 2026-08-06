from __future__ import annotations

import hashlib
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from openpi.shared import normalize
from scripts.openpi_profiles import ACTION_LORA_R16_STATE45_MODEL, resolve_profile
from scripts.train import train_cube1_01_compare as compare


class _State45TokenizePrompt:
    def __call__(self, data):
        state = np.asarray(data["state"], dtype=np.float32)
        output = dict(data)
        output.pop("prompt", None)
        output["tokenized_prompt"] = np.rint((state + 2.0) * 100).astype(np.int32)
        output["tokenized_prompt_mask"] = np.ones(45, dtype=bool)
        return output


class _State45Dataset:
    _state_dim = 45
    _state_contract = "state45"
    _extended_state = True
    _action_horizon = 10
    _action_source = "urdf_target_absolute"

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"index": index}


def _prepared() -> compare.PreparedDatum:
    state = np.zeros(45, dtype=np.float32)
    actions = np.zeros((10, 32), dtype=np.float32)
    prefix = {
        "state": state,
        "prompt": (
            "pick up the cube1 using gesture 03, then place it back on the table"
        ),
        "actions": actions,
    }
    clean = {
        "observation": {
            "state": {"data": state.tolist(), "shape": [45]},
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
    return compare.PreparedDatum(prefix, _State45TokenizePrompt(), (), clean)


def _norm_stats() -> dict:
    state_q01 = -np.ones(45, dtype=np.float32)
    state_q99 = np.ones(45, dtype=np.float32)
    state_q01[28:33] = 0.0
    state_q99[28:33] = 1.0
    state_q01[39], state_q99[39] = 0.0, 1.0
    state_q01[42], state_q99[42] = 0.0, 1.0
    state_q01[44], state_q99[44] = 0.0, 2.0
    return {
        "state": normalize.NormStats(
            mean=np.zeros(45, dtype=np.float32),
            std=np.ones(45, dtype=np.float32),
            q01=state_q01,
            q99=state_q99,
        ),
        "actions": normalize.NormStats(
            mean=np.zeros(32, dtype=np.float32),
            std=np.ones(32, dtype=np.float32),
            q01=-np.ones(32, dtype=np.float32),
            q99=np.ones(32, dtype=np.float32),
        ),
    }


def test_state45_profile_is_45_by_32_and_fail_closed_at_224() -> None:
    profile = resolve_profile(ACTION_LORA_R16_STATE45_MODEL)
    assert profile.state_dim == 45
    assert profile.action_dim == 32
    assert profile.action_horizon == 10
    assert profile.max_tokens == 224
    assert profile.fail_on_token_truncation is True
    assert profile.delta_mask_segments == (3, -3, 22, -4)


def test_state45_stateaug_changes_only_qpos28() -> None:
    noise = np.full(45, 0.75, dtype=np.float32)
    noise[:28] = 0.05
    diagnostics = compare.AugmentationDiagnostics(45)
    with patch.object(compare, "_prepare_discrete_datum", return_value=_prepared()):
        wire = compare.build_batch(
            _State45Dataset(),
            object(),
            base_model=ACTION_LORA_R16_STATE45_MODEL,
            indices=[0],
            norm_stats=_norm_stats(),
            state_noise_std=0.1,
            rng=np.random.default_rng(0),
            planned_requests=[(0, noise, None)],
            datum_cache=compare.DatumCache(1),
            augmentation_diagnostics=diagnostics,
        )[0]
    state = np.asarray(wire["observation"]["state"]["data"], dtype=np.float32)
    np.testing.assert_allclose(state[:28], 0.05)
    np.testing.assert_array_equal(state[28:], 0.0)
    summary = diagnostics.summary(0.1, token_budget=224)
    changed = summary["bin_changed_fraction_by_dimension"]
    assert len(changed) == 45
    assert all(value > 0 for value in changed[:28])
    assert changed[28:] == [0.0] * 17


def test_state45_locked_norm_requires_semantic_ranges(tmp_path) -> None:
    stats = _norm_stats()
    normalize.save(tmp_path, stats)
    norm_path = tmp_path / "norm_stats.json"
    sha = hashlib.sha256(norm_path.read_bytes()).hexdigest()
    loaded, evidence = compare.load_or_compute_norm_stats(
        _State45Dataset(), tmp_path, expected_norm_sha256=sha
    )
    assert np.asarray(loaded["state"].mean).shape == (45,)
    assert evidence["sha256"] == sha

    bad_dir = tmp_path / "bad"
    bad = _norm_stats()
    bad["state"].q99[44] = 1.0
    normalize.save(bad_dir, bad)
    bad_path = bad_dir / "norm_stats.json"
    bad_sha = hashlib.sha256(bad_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="State45 semantic norm index 44"):
        compare.load_or_compute_norm_stats(
            _State45Dataset(), bad_dir, expected_norm_sha256=bad_sha
        )
