from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_state45_training_launcher_is_explicit_and_profile_locked(tmp_path) -> None:
    norm = tmp_path / "norm" / "norm_stats.json"
    norm.parent.mkdir()
    norm.write_text("{}\n")
    selection = tmp_path / "train_selection.json"
    selection.write_text(json.dumps({
        "contract": "mano_state45_grade_a_selection_v1",
        "split": "train",
        "split_contract": "mano_state45_grade_a_object_gesture_split_v1",
        "rows": [{
            "release_row_index": 1,
            "object": "cube1",
            "gesture": "03",
            "grade": "A",
            "prompt": (
                "pick up the cube1 using gesture 03, then place it back on the table"
            ),
        }],
    }))
    windows = tmp_path / "train_contact_windows.json"
    windows.write_text("{}\n")
    report = {
        "contract": "mano_state45_grade_a_train_profile_v1",
        "status": "passed",
        "population": "grade_a",
        "population_rows": 4856,
        "train_rows": 1,
        "state_contract": "mano_state45_phase_native_sim_28d_v1",
        "source_state_contract": "mano_state41_native_sim_28d_v1",
        "state_dim": 45,
        "action_dim": 32,
        "action_horizon": 10,
        "frame_window": "contact",
        "contact_context_frames": 100,
        "missing_contact_policy": "error",
        "language_conditioning": "gesture",
        "prompt_template": (
            "pick up the {object} using gesture {gesture}, then place it back on the table"
        ),
        "model": "openpi/pi05-action-lora-r16-state45-phase-28dof-finetune",
        "profile_id": "pi05_action_lora_r16_state45_phase_28dof_v1",
        "norm_population": "train_only_contact_window",
        "fail_on_token_truncation": True,
        "delta_mask_segments": [3, -3, 22, -4],
        "max_token_len": 224,
        "token_audit": {"overflow_count": 0},
        "counterfactual_token_audit": {"overflow_count": 0, "max": 209},
        "dataset": "/immutable/formal.lance",
        "train_selection_manifest": str(selection),
        "train_contact_window_manifest": str(windows),
        "norm": {
            "path": str(norm),
            "sha256": hashlib.sha256(norm.read_bytes()).hexdigest(),
        },
    }
    report_path = tmp_path / "profile_report.json"
    report_path.write_text(json.dumps(report))
    env = {
        **os.environ,
        "STATE45_STEPS": "123",
        "STATE45_PRINT_CONFIG": "1",
    }
    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/remote/run_state45_gradea_train.sh"),
            str(report_path),
            "state45-test",
        ],
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )
    config = json.loads(completed.stdout)
    assert config["model"].endswith("state45-phase-28dof-finetune")
    assert config["steps"] == 123
    assert config["state_noise_std"] == 0.1


def test_state45_server_launcher_requires_norm_and_uses_own_cache_namespace(tmp_path) -> None:
    mint = tmp_path / "mint"
    openpi = tmp_path / "openpi"
    mint.mkdir()
    openpi.mkdir()
    norm = tmp_path / "norm_stats.json"
    norm.write_text("{}\n")
    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/remote/run_action_lora_server.sh"),
            "--runtime-root", str(tmp_path / "runtime"),
            "--port", "30539",
            "--gpus", "0",
            "--mint-root", str(mint),
            "--openpi-root", str(openpi),
            "--python-bin", "/bin/true",
            "--model", "openpi/pi05-action-lora-r16-state45-phase-28dof-finetune",
            "--norm-stats", str(norm),
            "--enable-jax-persistent-cache",
            "--print-config",
        ],
        text=True,
        capture_output=True,
        check=True,
        env={**os.environ, "PATH": "/usr/bin:/bin"},
    )
    assert "model=openpi/pi05-action-lora-r16-state45-phase-28dof-finetune" in completed.stderr
    assert "pi05_action_lora_r16_state45_phase_a800_1gpu" in completed.stderr


def test_state45_persistent_launcher_prints_done_timeout_contract(tmp_path) -> None:
    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/remote/run_state45_mode4_eval.sh"),
            "--lance-dataset", "/immutable/formal.lance",
            "--row", "7",
            "--norm-stats-dir", "/locked/norm",
            "--norm-sha-expected", "0" * 64,
            "--contact-window-manifest", "/locked/windows.json",
            "--model-path", "mint://sampler",
            "--owner-id", "owner",
            "--output-dir", str(tmp_path / "out"),
            "--print-config",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    config = json.loads(completed.stdout)
    assert config == {
        "action_dim": 32,
        "action_horizon": 10,
        "chunk_stride": 1,
        "frame_window": "persistent_task",
        "max_control_seconds": 15.0,
        "model": "openpi/pi05-action-lora-r16-state45-phase-28dof-finetune",
        "row": 7,
        "state_contract": "state45",
        "state_dim": 45,
        "video_mode": "none",
    }
