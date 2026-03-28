from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEV_HEAD = REPO_ROOT / ".claude" / "skills" / "volcano-cluster" / "configs" / "mint-dev-head.yaml"
PROD_HEAD = REPO_ROOT / ".claude" / "skills" / "volcano-cluster" / "configs" / "mint-prod-head.yaml"
PROD_WORKER = REPO_ROOT / ".claude" / "skills" / "volcano-cluster" / "configs" / "mint-prod-worker.yaml"


def test_dev_head_keeps_dashboard_and_ray_client_enabled() -> None:
    text = DEV_HEAD.read_text(encoding="utf-8")

    assert 'HEAD_IP_PATH = "/vePFS-Mindverse/share/code/tinker-server/ray_head_ip.txt"' in text
    assert "include_dashboard=True" in text
    assert 'dashboard_host="0.0.0.0"' in text
    assert "dashboard_port=8265" in text
    assert "ray_client_server_port=10001" in text
    assert '_port_open("127.0.0.1", 6379)' in text
    assert "node.dead_processes()" in text


def test_prod_head_self_heals_without_dashboard_or_ray_client() -> None:
    text = PROD_HEAD.read_text(encoding="utf-8")

    assert 'HEAD_IP_PATH = "/vePFS-Mindverse/share/code/tinker-server-auth/ray_head_ip.txt"' in text
    assert "BACKOFF_INITIAL_S = 5" in text
    assert "BACKOFF_MAX_S = 60" in text
    assert "BACKOFF_RESET_AFTER_HEALTHY_CHECKS = 6" in text
    assert "include_dashboard=False" in text
    assert "dashboard_host=" not in text
    assert "dashboard_port=" not in text
    assert "ray_client_server_port=" not in text
    assert '_port_open("127.0.0.1", 6379)' in text
    assert "node.dead_processes()" in text
    assert "kill_all_processes(check_alive=False, allow_graceful=False, wait=True)" in text
    assert "restart_delay_s = min(BACKOFF_MAX_S, restart_delay_s * 2)" in text


def test_prod_worker_self_heals_with_backoff() -> None:
    text = PROD_WORKER.read_text(encoding="utf-8")

    assert 'HEAD_IP_PATH = Path("/vePFS-Mindverse/share/code/tinker-server-auth/ray_head_ip.txt")' in text
    assert "BACKOFF_INITIAL_S = 5" in text
    assert "BACKOFF_MAX_S = 60" in text
    assert "BACKOFF_RESET_AFTER_HEALTHY_CHECKS = 6" in text
    assert "def _read_head_ip() -> str:" in text
    assert "_port_open(head_ip, 6379)" in text
    assert "node.dead_processes()" in text
    assert "kill_all_processes(check_alive=False, allow_graceful=False, wait=True)" in text
    assert "restart_delay_s = min(BACKOFF_MAX_S, restart_delay_s * 2)" in text
    assert "sleep 300" not in text
