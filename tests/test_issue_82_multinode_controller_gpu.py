from pathlib import Path


def test_issue_82_multinode_controller_does_not_reserve_extra_gpu() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "tinker_server/backend/multinode_inference.py"
    txt = src.read_text(encoding="utf-8")

    assert "controller_gpus = 0" in txt
    assert "total_required_gpus = worker_gpus" in txt

