import importlib
from pathlib import Path


def test_ray_namespace_env_roundtrip(monkeypatch):
    import mint_server.config as cfg

    old = cfg.os.environ.get("MINT_RAY_NAMESPACE")

    monkeypatch.setenv("MINT_RAY_NAMESPACE", "ns_test_83")
    cfg2 = importlib.reload(cfg)
    assert cfg2.RAY_NAMESPACE == "ns_test_83"

    if old is None:
        monkeypatch.delenv("MINT_RAY_NAMESPACE", raising=False)
    else:
        monkeypatch.setenv("MINT_RAY_NAMESPACE", old)
    cfg3 = importlib.reload(cfg)
    assert cfg3.RAY_NAMESPACE == (old or "mint")


def test_no_hardcoded_shared_code_root_in_worker_pythonpath():
    repo_root = Path(__file__).resolve().parents[1]

    for rel in (
        "mint_server/backend/verl_inference.py",
        "mint_server/backend/verl_training.py",
    ):
        txt = (repo_root / rel).read_text(encoding="utf-8")
        assert "/vePFS-Mindverse/share/code/mint-server" not in txt


def test_no_hardcoded_literal_namespace_in_backends():
    repo_root = Path(__file__).resolve().parents[1]

    for rel in (
        "mint_server/backend/verl_inference.py",
        "mint_server/backend/verl_training.py",
    ):
        txt = (repo_root / rel).read_text(encoding="utf-8")
        assert 'namespace="mint"' not in txt
        assert "namespace='mint'" not in txt


def test_detached_store_actors_use_mint_ray_namespace(monkeypatch):
    monkeypatch.setenv("MINT_RAY_NAMESPACE", "ns_shadow")
    monkeypatch.setenv("MINT_RAY_NAMESPACE", "ns_mint")

    import importlib

    task_state_store_mod = importlib.import_module("mint_server.backend.task_state_store")

    assert task_state_store_mod._ray_namespace() == "ns_mint"


def test_training_session_metadata_namespace_is_ray_namespace():
    repo_root = Path(__file__).resolve().parents[1]
    txt = (repo_root / "mint_server/routes/training.py").read_text(encoding="utf-8")
    assert "MINT_RAY_NAMESPACE" not in txt


def test_startup_reconciliation_does_not_guess_gpu_counts():
    repo_root = Path(__file__).resolve().parents[1]
    txt = (repo_root / "mint_server/app.py").read_text(encoding="utf-8")
    assert "num_gpus = 8" not in txt
    assert 'actor_type = ActorType.VLLM\n                            num_gpus = 1' not in txt
