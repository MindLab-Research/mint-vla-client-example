from pathlib import Path


def test_save_weights_for_sampler_route_is_defined_once():
    repo_root = Path(__file__).resolve().parents[1]
    weights_txt = (repo_root / "tinker_server/routes/weights.py").read_text(encoding="utf-8")
    training_txt = (repo_root / "tinker_server/routes/training.py").read_text(encoding="utf-8")

    assert '@router.post("/save_weights_for_sampler"' not in weights_txt
    assert '@router.post("/save_weights_for_sampler"' in training_txt

