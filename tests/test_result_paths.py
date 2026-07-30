from __future__ import annotations

import os
from pathlib import Path
from unittest import mock
import unittest

from scripts.eval import result_paths


class ResultPathTests(unittest.TestCase):
    def test_default_results_root_is_inside_formal_client(self):
        with mock.patch.dict(
            os.environ,
            {"VLA_CLIENT_RESULTS_ROOT": "", "VLA_CLIENT_INFERENCE_ROOT": ""},
            clear=False,
        ):
            repo_root = Path(__file__).resolve().parents[1]
            self.assertEqual(result_paths.client_results_root(), repo_root / "results")
            self.assertEqual(
                result_paths.client_inference_root(), repo_root / "results" / "inference"
            )

    def test_inference_root_override_controls_generated_run_path(self):
        with mock.patch.dict(
            os.environ,
            {"VLA_CLIENT_INFERENCE_ROOT": "/tmp/client-inference-results"},
            clear=False,
        ):
            output = result_paths.default_inference_output_dir("mode4")
            self.assertEqual(output.parent, Path("/tmp/client-inference-results"))
            self.assertTrue(output.name.startswith("mode4_"))

    def test_mode_name_must_be_a_portable_component(self):
        with self.assertRaises(ValueError):
            result_paths.default_inference_output_dir("../mode4")


if __name__ == "__main__":
    unittest.main()
