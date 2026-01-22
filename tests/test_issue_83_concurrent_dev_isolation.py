import importlib
from pathlib import Path


def test_ray_namespace_env_roundtrip(monkeypatch):
    import tinker_server.config as cfg

    old = cfg.os.environ.get("TINKER_RAY_NAMESPACE")

    monkeypatch.setenv("TINKER_RAY_NAMESPACE", "ns_test_83")
    cfg2 = importlib.reload(cfg)
    assert cfg2.RAY_NAMESPACE == "ns_test_83"

    if old is None:
        monkeypatch.delenv("TINKER_RAY_NAMESPACE", raising=False)
    else:
        monkeypatch.setenv("TINKER_RAY_NAMESPACE", old)
    cfg3 = importlib.reload(cfg)
    assert cfg3.RAY_NAMESPACE == (old or "tinker")


def test_no_hardcoded_shared_code_root_in_worker_pythonpath():
    repo_root = Path(__file__).resolve().parents[1]

    for rel in (
        "tinker_server/backend/verl_inference.py",
        "tinker_server/backend/verl_training.py",
    ):
        txt = (repo_root / rel).read_text(encoding="utf-8")
        assert "/vePFS-Mindverse/share/code/tinker-server" not in txt


def test_no_hardcoded_tinker_namespace_in_backends():
    repo_root = Path(__file__).resolve().parents[1]

    for rel in (
        "tinker_server/backend/verl_inference.py",
        "tinker_server/backend/verl_training.py",
    ):
        txt = (repo_root / rel).read_text(encoding="utf-8")
        assert 'namespace="tinker"' not in txt
        assert "namespace='tinker'" not in txt

