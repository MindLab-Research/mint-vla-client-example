from pathlib import Path


def test_megatron_distributed_checkpoint_saves_and_loads_optimizer_state():
    repo_root = Path(__file__).resolve().parents[1]
    txt = (repo_root / "tinker_server/backend/megatron_distributed.py").read_text(encoding="utf-8")

    assert "def load_optimizer_state(" in txt
    assert "_optimizer.pt" in txt
    assert "torch.save(self._capture_optimizer_state(), optimizer_file)" in txt
    assert "load_optimizer_state.remote" in txt
    assert "optimizer_restored" in txt
