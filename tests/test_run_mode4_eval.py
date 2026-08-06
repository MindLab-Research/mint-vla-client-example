from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts/remote/run_mode4_eval.sh"
SERVER_LAUNCHER = REPO_ROOT / "scripts/remote/run_action_lora_server.sh"


def init_git_checkout(path: Path) -> str:
    path.mkdir()
    (path / "source.txt").write_text(path.name, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "source.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def base_fixture(root: Path) -> tuple[Path, Path, str, str, Path, Path, Path]:
    mint = root / "mint"
    openpi = root / "openpi"
    mint_commit = init_git_checkout(mint)
    openpi_commit = init_git_checkout(openpi)
    dataset = root / "dataset.lance"
    dataset.mkdir()
    (root / "dataset.contact_ctx100_error_v1.json").write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "dataset": str(dataset),
                "context_frames": 100,
                "missing_policy": "error",
                "windows": {
                    "2": {"row_index": 2, "start_frame": 1, "end_frame": 3},
                    "7": {"row_index": 7, "start_frame": 2, "end_frame": 4},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    norm = root / "norm"
    norm.mkdir()
    norm_stats = norm / "norm_stats.json"
    norm_stats.write_text('{"fixture": true}\n', encoding="utf-8")
    config = root / "remote.env"
    config.write_text(
        f"MINT_CODE_ROOT={mint}\n"
        f"MINT_OPENPI_ROOT={openpi}\n"
        f"VLA_CLIENT_RESULTS_ROOT={root / 'client-results'}\n"
        f"VLA_CLIENT_INFERENCE_ROOT={root / 'client-results' / 'inference'}\n",
        encoding="utf-8",
    )
    gesture = root / "gesture.index.json"
    gesture.write_text("{}\n", encoding="utf-8")
    return mint, openpi, mint_commit, openpi_commit, dataset, norm, config


class RunMode4EvalContractTests(unittest.TestCase):
    def run_launcher(self, root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(LAUNCHER), *extra],
            cwd=REPO_ROOT,
            env={"PATH": "/usr/bin:/bin", "VLA_CLIENT_CONFIG": str(root / "remote.env")},
            text=True,
            capture_output=True,
        )

    def common_args(
        self,
        dataset: Path,
        norm: Path,
        output: Path,
        *extra: str,
    ) -> list[str]:
        return [
            "--model-path",
            "mint://fixture/checkpoint",
            "--dataset",
            str(dataset),
            "--rows",
            "2,7,2",
            "--normalization-rows",
            "7,2,7",
            "--norm-stats-dir",
            str(norm),
            "--output-dir",
            str(output),
            "--owner-id",
            "owner-1",
            "--base-url",
            "http://127.0.0.1:30532",
            "--backend-commit",
            "0123456789abcdef0123456789abcdef01234567",
            "--model-commit",
            "abcdef0123456789abcdef0123456789abcdef01",
            "--allow-dirty-sources",
            *extra,
        ]

    def test_print_config_records_existing_endpoint_provenance_and_lists(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, mint_commit, openpi_commit, dataset, norm, config = base_fixture(root)
            output = root / "fresh-output"
            expected_norm_sha = hashlib.sha256((norm / "norm_stats.json").read_bytes()).hexdigest()
            completed = self.run_launcher(
                root,
                *self.common_args(
                    dataset,
                    norm,
                    output,
                    "--norm-sha-expected",
                    expected_norm_sha,
                    "--print-config",
                ),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(
                payload["endpoint"],
                {
                    "mode": "existing",
                    "label": None,
                    "base_url": "http://127.0.0.1:30532",
                    "source_verification": "operator_declared",
                    "reuse_server_info": None,
                },
            )
            self.assertEqual(payload["evaluation"]["row_indices"], [2, 7])
            self.assertEqual(payload["evaluation"]["normalization_row_indices"], [7, 2])
            self.assertEqual(payload["evaluation"]["video_mode"], "full")
            self.assertEqual(payload["evaluation"]["phase_gate"], {"mode": "off"})
            self.assertEqual(payload["evaluation"]["row_execution"], "lockstep")
            self.assertEqual(payload["evaluation"]["row_batch_size"], 4)
            self.assertEqual(payload["evaluation"]["frame_window"], "contact")
            self.assertEqual(
                payload["evaluation"]["contact_window_manifest"],
                str(root / "dataset.contact_ctx100_error_v1.json"),
            )
            self.assertEqual(payload["evaluation"]["dataset_reference_video_window"], "full")
            self.assertEqual(payload["evaluation"]["physics_comparison_video_window"], "contact")
            self.assertEqual(payload["provenance"]["backend_commit"], "0123456789abcdef0123456789abcdef01234567")
            self.assertEqual(payload["provenance"]["model_commit"], "abcdef0123456789abcdef0123456789abcdef01")
            self.assertIsNone(payload["provenance"]["backend_dirty"])
            self.assertIsNone(payload["dedicated_server"])
            self.assertFalse(output.exists(), "--print-config must not create output")
            self.assertEqual(payload["provenance"]["norm_stats_sha256"], expected_norm_sha)
            self.assertEqual(
                payload["provenance"]["norm_stats_sha256_expected"],
                expected_norm_sha,
            )
            release_manifest = REPO_ROOT / "config/datasets/mano_dataset_release.json"
            release_payload = json.loads(release_manifest.read_text())
            self.assertEqual(
                payload["provenance"]["dataset_release_id"],
                release_payload["release_id"],
            )
            self.assertEqual(
                payload["provenance"]["dataset_release_manifest_sha256"],
                hashlib.sha256(release_manifest.read_bytes()).hexdigest(),
            )

    def test_print_config_records_grasp_probe_phase_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, _, _, dataset, norm, _ = base_fixture(root)
            completed = self.run_launcher(
                root,
                *self.common_args(
                    dataset,
                    norm,
                    root / "output",
                    "--chunk-stride",
                    "1",
                    "--phase-gate",
                    "grasp-probe",
                    "--phase-gate-min-contact-count",
                    "4",
                    "--phase-gate-contact-persistence-frames",
                    "20",
                    "--phase-gate-probe-lift-mm",
                    "5",
                    "--phase-gate-retention-lift-mm",
                    "50",
                    "--print-config",
                ),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            gate = json.loads(completed.stdout)["evaluation"]["phase_gate"]
            self.assertEqual(gate["mode"], "grasp-probe")
            self.assertEqual(gate["min_contact_count"], 4)
            self.assertEqual(gate["contact_persistence_frames"], 20)
            self.assertEqual(gate["probe_lift_mm"], 5.0)
            self.assertEqual(gate["retention_lift_mm"], 50.0)
            self.assertTrue(gate["require_floor_clear"])

    def test_grasp_probe_requires_stride_one(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, _, _, dataset, norm, _ = base_fixture(root)
            completed = self.run_launcher(
                root,
                *self.common_args(
                    dataset,
                    norm,
                    root / "output",
                    "--phase-gate",
                    "grasp-probe",
                    "--print-config",
                ),
            )
            self.assertEqual(completed.returncode, 64)
            self.assertIn("requires --chunk-stride 1", completed.stderr)

    def test_state44_config_records_independent_state_and_action_widths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, _, _, dataset, norm, _ = base_fixture(root)
            completed = self.run_launcher(
                root,
                *self.common_args(
                    dataset,
                    norm,
                    root / "state44-output",
                    "--model",
                    "openpi/pi05-action-lora-r16-state44-finetune",
                    "--state-contract",
                    "state44",
                    "--print-config",
                ),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            evaluation = json.loads(completed.stdout)["evaluation"]
            self.assertEqual(evaluation["state_contract"], "state44")
            self.assertEqual(evaluation["state_dim"], 44)
            self.assertEqual(evaluation["action_dim"], 32)

    def test_state41_config_records_native_28dof_state_and_action_widths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, _, _, dataset, norm, _ = base_fixture(root)
            completed = self.run_launcher(
                root,
                *self.common_args(
                    dataset,
                    norm,
                    root / "state41-output",
                    "--model",
                    "openpi/pi05-action-lora-r16-state41-28dof-finetune",
                    "--state-contract",
                    "state41",
                    "--language-conditioning",
                    "object_only",
                    "--print-config",
                ),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            evaluation = json.loads(completed.stdout)["evaluation"]
            self.assertEqual(evaluation["state_contract"], "state41")
            self.assertEqual(evaluation["state_dim"], 41)
            self.assertEqual(evaluation["action_dim"], 32)
            self.assertEqual(evaluation["language_conditioning"], "object_only")

    def test_state41_row_residency_can_exceed_policy_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, _, _, dataset, norm, _ = base_fixture(root)
            completed = self.run_launcher(
                root,
                *self.common_args(
                    dataset,
                    norm,
                    root / "state41-residency-output",
                    "--model",
                    "openpi/pi05-action-lora-r16-state41-28dof-finetune",
                    "--state-contract",
                    "state41",
                    "--language-conditioning",
                    "object_only",
                    "--act-batch-size",
                    "4",
                    "--row-batch-size",
                    "16",
                    "--print-config",
                ),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            evaluation = json.loads(completed.stdout)["evaluation"]
            self.assertEqual(evaluation["act_batch_size"], 4)
            self.assertEqual(evaluation["row_batch_size"], 16)

    def test_state41_gesture_uses_formal_release_metadata_not_raw_index(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, _, _, dataset, norm, _ = base_fixture(root)
            completed = self.run_launcher(
                root,
                *self.common_args(
                    dataset,
                    norm,
                    root / "state41-gesture-output",
                    "--model",
                    "openpi/pi05-action-lora-r16-state41-28dof-finetune",
                    "--state-contract",
                    "state41",
                    "--language-conditioning",
                    "gesture",
                    "--gesture-index",
                    str(root / "missing-and-intentionally-unused.json"),
                    "--print-config",
                ),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            evaluation = json.loads(completed.stdout)["evaluation"]
            self.assertEqual(evaluation["language_conditioning"], "gesture")
            self.assertEqual(evaluation["language_source"], "formal_release_metadata")
            self.assertIsNone(evaluation["gesture_index"])

    def test_state44_model_and_contract_must_be_selected_together(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, _, _, dataset, norm, _ = base_fixture(root)
            completed = self.run_launcher(
                root,
                *self.common_args(
                    dataset,
                    norm,
                    root / "state44-output",
                    "--state-contract",
                    "state44",
                    "--print-config",
                ),
            )
            self.assertEqual(completed.returncode, 64)
            self.assertIn("state44 requires model", completed.stderr)

    def test_rejects_population_norm_sha_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, _, _, dataset, norm, _ = base_fixture(root)
            completed = self.run_launcher(
                root,
                *self.common_args(
                    dataset,
                    norm,
                    root / "output",
                    "--norm-sha-expected",
                    "0" * 64,
                    "--print-config",
                ),
            )
            self.assertEqual(completed.returncode, 64)
            self.assertIn("norm SHA mismatch", completed.stderr)

    def test_default_output_uses_client_inference_root_and_run_name(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, _, _, dataset, norm, _ = base_fixture(root)
            args = self.common_args(dataset, norm, root / "unused-explicit-output")
            output_index = args.index("--output-dir")
            del args[output_index : output_index + 2]
            completed = self.run_launcher(
                root,
                *args,
                "--run-name",
                "stateaug-row943-939",
                "--print-config",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            expected = root / "client-results" / "inference" / "stateaug-row943-939"
            self.assertEqual(payload["evaluation"]["output_dir"], str(expected))
            self.assertFalse(expected.exists(), "--print-config must not create default output")

    def test_run_name_and_output_dir_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, _, _, dataset, norm, _ = base_fixture(root)
            completed = self.run_launcher(
                root,
                *self.common_args(
                    dataset,
                    norm,
                    root / "explicit-output",
                    "--run-name",
                    "also-named",
                    "--print-config",
                ),
            )
            self.assertEqual(completed.returncode, 64)
            self.assertIn("--output-dir and --run-name are mutually exclusive", completed.stderr)

    def test_explicit_full_window_does_not_require_contact_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, _, _, dataset, norm, _ = base_fixture(root)
            (root / "dataset.contact_ctx100_error_v1.json").unlink()
            output = root / "full-stress-output"
            completed = self.run_launcher(
                root,
                *self.common_args(
                    dataset,
                    norm,
                    output,
                    "--frame-window",
                    "full",
                    "--print-config",
                ),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["evaluation"]["frame_window"], "full")
            self.assertIsNone(payload["evaluation"]["contact_window_manifest"])
            self.assertEqual(payload["evaluation"]["dataset_reference_video_window"], "full")
            self.assertEqual(payload["evaluation"]["physics_comparison_video_window"], "full")

    def test_print_config_accepts_video_none_without_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, _, _, dataset, norm, _ = base_fixture(root)
            output = root / "no-video-output"
            completed = self.run_launcher(
                root,
                *self.common_args(dataset, norm, output, "--video-mode", "none", "--print-config"),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["evaluation"]["video_mode"], "none")
            self.assertFalse(output.exists())

    def test_owned_server_uses_worktree_commits_and_disables_cache_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, mint_commit, openpi_commit, dataset, norm, _ = base_fixture(root)
            output = root / "owned-output"
            completed = self.run_launcher(
                root,
                *[
                    "--model-path",
                    "mint://fixture/checkpoint",
                    "--dataset",
                    str(dataset),
                    "--rows",
                    "2",
                    "--normalization-rows",
                    "2",
                    "--norm-stats-dir",
                    str(norm),
                    "--output-dir",
                    str(output),
                    "--owner-id",
                    "owner-1",
                    "--own-server",
                    "--server-runtime-root",
                    str(root / "runtime"),
                    "--server-port",
                    "30533",
                    "--server-gpus",
                    "0",
                    "--mint-root",
                    str(root / "mint"),
                    "--openpi-root",
                    str(root / "openpi"),
                    "--python-bin",
                    "/bin/true",
                    "--allow-dirty-sources",
                    "--print-config",
                ],
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["endpoint"]["mode"], "dedicated")
            self.assertEqual(
                payload["endpoint"]["source_verification"],
                "launcher_verified_worktrees",
            )
            self.assertEqual(payload["provenance"]["backend_commit"], mint_commit)
            self.assertEqual(payload["provenance"]["model_commit"], openpi_commit)
            self.assertFalse(
                payload["dedicated_server"]["jax_persistent_executable_cache"]
            )
            self.assertFalse(payload["dedicated_server"]["keep_server"])
            self.assertEqual(payload["dedicated_server"]["gpus"], [0])
            self.assertFalse(output.exists())

    def test_reuse_server_info_supplies_endpoint_commits_owner_and_session(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, _, _, dataset, norm, _ = base_fixture(root)
            fake_server = subprocess.Popen(["bash", "-c", "exec -a uvicorn sleep 60"])
            try:
                marker = root / "server.keepalive.json"
                marker.write_text(
                    json.dumps(
                        {
                            "status": "owned_running",
                            "pid": fake_server.pid,
                            "base_url": "http://127.0.0.1:30536",
                            "owner_id": "owner-from-marker",
                            "backend_commit": "0123456789abcdef0123456789abcdef01234567",
                            "model_commit": "abcdef0123456789abcdef0123456789abcdef01",
                            "action_session_id": "retained-session-123",
                            "model": "openpi/pi05-action-lora-r16-finetune",
                            "model_path": "mint://fixture/checkpoint",
                            "act_mode": "batch",
                            "act_batch_size": 4,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                output = root / "reused-output"
                completed = self.run_launcher(
                    root,
                    "--model-path",
                    "mint://fixture/checkpoint",
                    "--dataset",
                    str(dataset),
                    "--rows",
                    "2,7",
                    "--normalization-rows",
                    "7,2",
                    "--norm-stats-dir",
                    str(norm),
                    "--output-dir",
                    str(output),
                    "--reuse-server-info",
                    str(marker),
                    "--allow-dirty-sources",
                    "--print-config",
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                payload = json.loads(completed.stdout)
                self.assertEqual(payload["endpoint"]["mode"], "retained")
                self.assertEqual(
                    payload["endpoint"]["source_verification"],
                    "retained_action_session_marker",
                )
                self.assertEqual(payload["endpoint"]["reuse_server_info"], str(marker))
                self.assertEqual(payload["evaluation"]["owner_id"], "owner-from-marker")
                self.assertEqual(
                    payload["evaluation"]["action_session_id"], "retained-session-123"
                )
                self.assertIsNone(payload["dedicated_server"])
                self.assertFalse(output.exists())
            finally:
                fake_server.terminate()
                fake_server.wait(timeout=5)

    def test_reuse_server_info_requires_retained_action_session(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, _, _, dataset, norm, _ = base_fixture(root)
            marker = root / "server.keepalive.json"
            marker.write_text(
                json.dumps(
                    {
                        "status": "owned_running",
                        "pid": os.getpid(),
                        "base_url": "http://127.0.0.1:30536",
                        "owner_id": "owner-from-marker",
                        "backend_commit": "0123456789abcdef0123456789abcdef01234567",
                        "model_commit": "abcdef0123456789abcdef0123456789abcdef01",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            completed = self.run_launcher(
                root,
                "--model-path",
                "mint://fixture/checkpoint",
                "--dataset",
                str(dataset),
                "--rows",
                "2",
                "--normalization-rows",
                "2",
                "--norm-stats-dir",
                str(norm),
                "--output-dir",
                str(root / "out"),
                "--reuse-server-info",
                str(marker),
                "--allow-dirty-sources",
                "--print-config",
            )
            self.assertEqual(completed.returncode, 64)
            self.assertIn("invalid reuse server marker", completed.stderr)

    def test_keep_server_requires_owned_server(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, _, _, dataset, norm, _ = base_fixture(root)
            completed = self.run_launcher(
                root,
                *self.common_args(dataset, norm, root / "out", "--keep-server"),
            )
            self.assertEqual(completed.returncode, 64)
            self.assertIn("--keep-server requires --own-server", completed.stderr)

    def test_existing_endpoint_requires_declared_backend_and_model_commits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, _, _, dataset, norm, _ = base_fixture(root)
            args = self.common_args(dataset, norm, root / "out")
            args[args.index("--backend-commit") : args.index("--backend-commit") + 2] = []
            completed = self.run_launcher(root, *args)
            self.assertEqual(completed.returncode, 64)
            self.assertIn("--backend-commit is required", completed.stderr)

    def test_existing_endpoint_rejects_dedicated_server_options(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, _, _, dataset, norm, _ = base_fixture(root)
            completed = self.run_launcher(
                root,
                *self.common_args(dataset, norm, root / "out", "--server-gpus", "0"),
            )
            self.assertEqual(completed.returncode, 64)
            self.assertIn("dedicated-server options require --own-server", completed.stderr)

    def test_existing_output_is_rejected_without_explicit_override(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, _, _, dataset, norm, _ = base_fixture(root)
            output = root / "existing-output"
            output.mkdir()
            completed = self.run_launcher(
                root,
                *self.common_args(dataset, norm, output, "--print-config"),
            )
            self.assertEqual(completed.returncode, 64)
            self.assertIn("output already exists", completed.stderr)

    def test_dedicated_server_launcher_disables_persistent_cache_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "mint").mkdir()
            (root / "openpi").mkdir()
            norm = root / "norm"
            norm.mkdir()
            (norm / "norm_stats.json").write_text('{"fixture": true}\n', encoding="utf-8")
            completed = subprocess.run(
                [
                    "bash",
                    str(SERVER_LAUNCHER),
                    "--runtime-root",
                    str(root / "runtime"),
                    "--port",
                    "30539",
                    "--gpus",
                    "0",
                    "--mint-root",
                    str(root / "mint"),
                    "--openpi-root",
                    str(root / "openpi"),
                    "--python-bin",
                    "/bin/true",
                    "--norm-stats",
                    str(norm / "norm_stats.json"),
                    "--print-config",
                ],
                cwd=REPO_ROOT,
                env={"PATH": "/usr/bin:/bin"},
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("jax_persistent_executable_cache=0", completed.stderr)
            self.assertIn("jax_compilation_cache=disabled", completed.stderr)
            self.assertIn(
                "model=openpi/pi05-action-lora-r16-state41-28dof-finetune",
                completed.stderr,
            )
            self.assertIn("norm_stats_path=", completed.stderr)


if __name__ == "__main__":
    unittest.main()
