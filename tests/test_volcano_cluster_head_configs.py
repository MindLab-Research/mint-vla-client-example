from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEV_HEAD = REPO_ROOT / ".claude" / "skills" / "volcano-cluster" / "configs" / "mint-dev-head.yaml"
DEV_WORKER = REPO_ROOT / ".claude" / "skills" / "volcano-cluster" / "configs" / "mint-dev-worker.yaml"
PROD_HEAD = REPO_ROOT / ".claude" / "skills" / "volcano-cluster" / "configs" / "mint-prod-head.yaml"
PROD_WORKER = REPO_ROOT / ".claude" / "skills" / "volcano-cluster" / "configs" / "mint-prod-worker.yaml"


def test_dev_head_keeps_dashboard_and_ray_client_enabled() -> None:
    text = DEV_HEAD.read_text(encoding="utf-8")

    assert 'HEAD_IP_PATH = "/vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt"' in text
    assert 'TMP_ROOT = os.environ.get("MINT_TMP_ROOT", "/vePFS-Mindverse/share/mint-data/dev")' in text
    assert 'RAY_TMP_ROOT_REAL = f"{TMP_ROOT}/head"' in text
    assert 'RAY_TMP_LINK = "/tmp/mdh"' in text
    assert 'RAY_OS_TMPDIR = f"{RAY_TMP_LINK}/tmp"' in text
    assert 'RAY_XDG_CACHE_HOME = f"{RAY_TMP_LINK}/cache"' in text
    assert 'temp_dir=RAY_TEMP_DIR' in text
    assert 'object_spilling_directory=RAY_OBJECT_SPILLING_DIR' in text
    assert 'import shutil' in text
    assert 'shutil.rmtree(RAY_TMP_LINK)' in text
    assert 'os.symlink(RAY_TMP_ROOT_REAL, RAY_TMP_LINK)' in text
    assert 'os.path.realpath(RAY_TMP_LINK) != os.path.realpath(RAY_TMP_ROOT_REAL)' in text
    assert 'os.environ["TMPDIR"] = RAY_OS_TMPDIR' in text
    assert 'os.environ["XDG_CACHE_HOME"] = RAY_XDG_CACHE_HOME' in text
    assert 'RAY_RUNTIME_ENV_WORKING_DIR_CACHE_SIZE_GB' in text
    assert 'RAY_RUNTIME_ENV_PIP_CACHE_SIZE_GB' in text
    assert "include_dashboard=True" in text
    assert 'dashboard_host="0.0.0.0"' in text
    assert "dashboard_port=8265" in text
    assert "ray_client_server_port=10001" in text
    assert '_port_open("127.0.0.1", 6379)' in text
    assert "node.dead_processes()" in text
    assert 'MountPath: "/tos-mindverse"' in text
    assert 'Bucket: "tos-mindverse-dev"' in text
    assert 'MountPath: "/tos-mindverse-prod"' in text
    assert 'Bucket: "tos-mindverse"' in text
    assert 'Flavor: "ml.r3i.4xlarge"' in text


def test_dev_worker_uses_short_temp_paths() -> None:
    text = DEV_WORKER.read_text(encoding="utf-8")

    assert 'HEAD_IP_PATH = "/vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt"' in text
    assert 'TMP_ROOT = os.environ.get("MINT_TMP_ROOT", "/vePFS-Mindverse/share/mint-data/dev")' in text
    assert 'RAY_TMP_ROOT_BASE = f"{TMP_ROOT}/worker"' in text
    assert 'RAY_TMP_LINK = "/tmp/mdw"' in text
    assert 'ray_os_tmpdir = f"{RAY_TMP_LINK}/tmp"' in text
    assert 'ray_xdg_cache_home = f"{RAY_TMP_LINK}/cache"' in text
    assert "ip.replace('.', '-')" in text
    assert 'temp_dir=ray_temp_dir' in text
    assert 'object_spilling_directory=ray_object_spilling_dir' in text
    assert 'import shutil' in text
    assert 'shutil.rmtree(RAY_TMP_LINK)' in text
    assert 'os.symlink(ray_tmp_root_real, RAY_TMP_LINK)' in text
    assert 'os.path.realpath(RAY_TMP_LINK) != os.path.realpath(ray_tmp_root_real)' in text
    assert 'os.environ["TMPDIR"] = ray_os_tmpdir' in text
    assert 'os.environ["XDG_CACHE_HOME"] = ray_xdg_cache_home' in text
    assert 'RAY_RUNTIME_ENV_WORKING_DIR_CACHE_SIZE_GB' in text
    assert 'RAY_RUNTIME_ENV_PIP_CACHE_SIZE_GB' in text
    assert "node.dead_processes()" in text
    assert 'MountPath: "/tos-mindverse"' in text
    assert 'Bucket: "tos-mindverse-dev"' in text
    assert 'MountPath: "/tos-mindverse-prod"' in text
    assert 'Bucket: "tos-mindverse"' in text


def test_prod_head_self_heals_without_dashboard_or_ray_client() -> None:
    text = PROD_HEAD.read_text(encoding="utf-8")

    assert 'HEAD_IP_PATH = "/vePFS-Mindverse/share/mint/prod/ray/head-address/ray_head_ip.txt"' in text
    assert 'TMP_ROOT = os.environ.get("MINT_TMP_ROOT", "/vePFS-Mindverse/share/mint-data/prod")' in text
    assert 'RAY_TMP_ROOT_REAL = f"{TMP_ROOT}/head"' in text
    assert 'RAY_TMP_LINK = "/tmp/mph"' in text
    assert 'RAY_OS_TMPDIR = f"{RAY_TMP_LINK}/tmp"' in text
    assert 'RAY_XDG_CACHE_HOME = f"{RAY_TMP_LINK}/cache"' in text
    assert 'temp_dir=RAY_TEMP_DIR' in text
    assert 'object_spilling_directory=RAY_OBJECT_SPILLING_DIR' in text
    assert 'import shutil' in text
    assert 'shutil.rmtree(RAY_TMP_LINK)' in text
    assert 'os.symlink(RAY_TMP_ROOT_REAL, RAY_TMP_LINK)' in text
    assert 'os.path.realpath(RAY_TMP_LINK) != os.path.realpath(RAY_TMP_ROOT_REAL)' in text
    assert 'os.environ["TMPDIR"] = RAY_OS_TMPDIR' in text
    assert 'os.environ["XDG_CACHE_HOME"] = RAY_XDG_CACHE_HOME' in text
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

    assert 'HEAD_IP_PATH = Path("/vePFS-Mindverse/share/mint/prod/ray/head-address/ray_head_ip.txt")' in text
    assert 'TMP_ROOT = os.environ.get("MINT_TMP_ROOT", "/vePFS-Mindverse/share/mint-data/prod")' in text
    assert 'RAY_TMP_ROOT_BASE = f"{TMP_ROOT}/worker"' in text
    assert 'RAY_TMP_LINK = "/tmp/mpw"' in text
    assert 'temp_dir=ray_temp_dir' in text
    assert 'object_spilling_directory=ray_object_spilling_dir' in text
    assert 'import shutil' in text
    assert 'shutil.rmtree(RAY_TMP_LINK)' in text
    assert 'os.symlink(ray_tmp_root_real, RAY_TMP_LINK)' in text
    assert 'os.path.realpath(RAY_TMP_LINK) != os.path.realpath(ray_tmp_root_real)' in text
    assert 'os.environ["TMPDIR"] = ray_os_tmpdir' in text
    assert 'os.environ["XDG_CACHE_HOME"] = ray_xdg_cache_home' in text
    assert "BACKOFF_INITIAL_S = 5" in text
    assert "BACKOFF_MAX_S = 60" in text
    assert "BACKOFF_RESET_AFTER_HEALTHY_CHECKS = 6" in text
    assert "def _read_head_ip() -> str:" in text
    assert "_port_open(head_ip, 6379)" in text
    assert "node.dead_processes()" in text
    assert "kill_all_processes(check_alive=False, allow_graceful=False, wait=True)" in text
    assert "restart_delay_s = min(BACKOFF_MAX_S, restart_delay_s * 2)" in text
    assert "sleep 300" not in text
