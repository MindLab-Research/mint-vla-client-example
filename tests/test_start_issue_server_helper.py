from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _parse_env_lines(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key] = value
    return out


def test_start_issue_server_helper_scopes_control_plane_actor_names() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    helper = repo_root / "scripts" / "tools" / "start_issue_server.sh"

    env = os.environ.copy()
    env.update(
        {
            "ISSUE_SERVER_ROOT": str(repo_root),
            "ISSUE_STARTUP_PRINT_ENV": "1",
            "RAY_ADDRESS": "ray://192.168.39.87:10001",
            "MINT_RAY_CLIENT_ADDRESS": "ray://192.168.39.87:10001",
            "MINT_CODE_ROOT": "/vePFS-Mindverse/share/code/yiwen/mint-server-issue-416",
            "ISSUE_NAMESPACE": "mint_yiwen_issue_416_r9",
            "ISSUE_PORT": "10419",
            "ISSUE_LOG_FILE": "/tmp/mint_server_issue_416_r9.log",
            "ISSUE_SUPPORTED_MODELS": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "MINT_TOPOLOGY_CONFIG_PATH": "/vePFS-Mindverse/share/mint/dev/runtime/topology.yaml",
        }
    )

    proc = subprocess.run(
        ["bash", str(helper)],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    data = _parse_env_lines(proc.stdout)

    token = "mint_yiwen_issue_416_r9"
    assert data["RAY_ADDRESS"] == "ray://192.168.39.87:10001"
    assert data["MINT_RAY_CLIENT_ADDRESS"] == "ray://192.168.39.87:10001"
    assert (
        data["MINT_VLLM_CHILD_PYTHON_EXECUTABLE"]
        == "/vePFS-Mindverse/share/code/yiwen/mint-server-issue-416/scripts/vllm_worker_python.py"
    )
    assert (
        data["MINT_RAY_PY_MODULES_CSV"]
        == "/vePFS-Mindverse/share/code/yiwen/mint-server-issue-416/mint_server"
    )
    assert data["MINT_RAY_NAMESPACE"] == "mint_yiwen_issue_416_r9"
    assert data["MINT_DISABLE_MINT_ROUTE"] == "1"
    assert data["MINT_API_KEY"] == "dummy"
    assert data["MINT_TOPOLOGY_CONFIG_PATH"] == "/vePFS-Mindverse/share/mint/dev/runtime/topology.yaml"
    assert "MINT_MODEL_PLACEMENT_JSON" not in data
    assert "MINT_MEGATRON_MODEL_PLACEMENT_JSON" not in data
    assert "MINT_DENSE_MODEL_PLACEMENT_JSON" not in data
    assert "MINT_VLLM_MODEL_PLACEMENT_JSON" not in data

    assert token in data["MINT_MODEL_WORK_SCHEDULER_ACTOR_NAME"]
    assert token in data["MINT_TASK_STATE_STORE_ACTOR_NAME"]
    assert token in data["MINT_MAINTENANCE_CRON_ACTOR_NAME"]
    forbidden = {
        "mint_model_work_scheduler",
        "mint_task_state_store",
        "mint_maintenance_cron",
    }
    for key, value in data.items():
        if key.endswith("_ACTOR_NAME"):
            assert value not in forbidden
