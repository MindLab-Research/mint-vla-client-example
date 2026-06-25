from __future__ import annotations

from pathlib import Path


def test_final_save_state_precedes_final_sampler_eval() -> None:
    path = Path(__file__).resolve().parents[2] / ".claude/skills/sanity-check/mint_rl_test_long.py"
    source = path.read_text(encoding="utf-8")

    training_complete = source.index('print("\\nRL training complete!")')
    final_save_state = source.index("save_state_future = training_client.save_state", training_complete)
    final_sampler_export = source.index("final_path = training_client.save_weights_for_sampler", training_complete)
    eval_start = source.index('print("Testing RL-only model (10-199 range):")', training_complete)

    assert final_save_state < final_sampler_export < eval_start
