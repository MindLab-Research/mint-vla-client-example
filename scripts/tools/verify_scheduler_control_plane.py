#!/usr/bin/env python3
"""Run the scheduler/control-plane local verification slate.

This is intentionally a local target rather than a CI entrypoint.  It captures
the component and contract tests that make PR #721 self-verifying.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCHEDULER_CONTROL_PLANE_TESTS = [
    "tests/component/control_plane/test_scheduler_component.py",
    "tests/test_stateless_control_plane_guardrails.py",
    "tests/test_issue_593_model_runtime_actor.py",
    "tests/test_issue_593_model_work_scheduler.py",
    "tests/test_issue_616_task_state_store.py",
    "tests/test_future_state_store.py",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the scheduler/control-plane local verification slate.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional pytest arguments. Use '-- -x -vv' to pass options.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    pytest_args = list(args.pytest_args)
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *SCHEDULER_CONTROL_PLANE_TESTS,
        "-q",
        *pytest_args,
    ]
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=_repo_root(), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
