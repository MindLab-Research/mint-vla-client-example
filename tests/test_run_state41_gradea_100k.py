from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os
import subprocess


SCRIPT = Path(__file__).parents[1] / "scripts/remote/run_state41_gradea_100k.sh"
MODEL = "openpi/pi05-action-lora-r16-state41-28dof-finetune"


def _profile(tmp_path: Path) -> Path:
    norm_dir = tmp_path / "norm"
    norm_dir.mkdir()
    norm_path = norm_dir / "norm_stats.json"
    norm_path.write_text("{}\n")
    norm_sha = hashlib.sha256(norm_path.read_bytes()).hexdigest()
    contact = tmp_path / "train_contact_windows.json"
    contact.write_text("{}\n")
    selection = tmp_path / "train_selection.json"
    selection.write_text(json.dumps({
        "contract": "mano_state41_grade_a_selection_v1",
        "split": "train",
        "rows": [{"release_row_index": 7, "grade": "A"}],
    }) + "\n")
    report = {
        "contract": "mano_state41_grade_a_train_profile_v1",
        "status": "passed",
        "population": "grade_a",
        "population_rows": 4856,
        "train_rows": 1,
        "state_dim": 41,
        "action_dim": 32,
        "action_horizon": 10,
        "frame_window": "contact",
        "contact_context_frames": 100,
        "missing_contact_policy": "error",
        "language_conditioning": "gesture",
        "prompt_template": "pick up the {object} using gesture {gesture}",
        "model": MODEL,
        "norm_population": "train_only_contact_window",
        "delta_mask_segments": [3, -3, 22, -4],
        "sampling_default": {
            "strategy": "sqrt_tempered",
            "coverage_anchors_per_row": 8,
        },
        "dataset": str(tmp_path / "release.lance"),
        "train_selection_manifest": str(selection),
        "train_contact_window_manifest": str(contact),
        "norm": {"path": str(norm_path), "sha256": norm_sha},
        "train_uuid_sha256": "a" * 64,
        "validation_uuid_sha256": "b" * 64,
    }
    path = tmp_path / "profile_report.json"
    path.write_text(json.dumps(report) + "\n")
    return path


def test_print_config_exposes_locked_full_a_defaults(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    result = subprocess.run(
        ["bash", str(SCRIPT), str(profile)],
        cwd=SCRIPT.parents[2],
        env={**os.environ, "STATE41_GRADEA_PRINT_CONFIG": "1"},
        check=True,
        capture_output=True,
        text=True,
    )
    config = json.loads(result.stdout)
    assert config["steps"] == 100000
    assert config["batch_size"] == 64
    assert config["per_device_batch_size"] == 16
    assert config["expected_device_count"] == 4
    assert config["learning_rate"] == 5e-5
    assert config["state_noise_std"] == 0.1
    assert config["target_noise_std"] == 0.0
    assert config["sampling_strategy"] == "sqrt_tempered"
    assert config["coverage_anchors_per_row"] == 8
    assert config["checkpoint_every"] == 5000
    assert config["language_conditioning"] == "gesture"
    assert config["continuous_training"] is True
    assert config["interleaved_mode4"] is False


def test_launcher_rejects_non_gesture_profile(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    payload = json.loads(profile.read_text())
    payload["language_conditioning"] = "object_only"
    profile.write_text(json.dumps(payload) + "\n")
    result = subprocess.run(
        ["bash", str(SCRIPT), str(profile)],
        cwd=SCRIPT.parents[2],
        env={**os.environ, "STATE41_GRADEA_PRINT_CONFIG": "1"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "profile mismatch language_conditioning" in result.stderr
