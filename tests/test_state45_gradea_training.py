from __future__ import annotations

import pytest

from scripts.train.prepare_mano_state45_gradea_profile import (
    _counterfactual_token_audit,
)
from scripts.train.state45_gradea_contract import (
    CANONICAL_PROMPT_TEMPLATE,
    SPLIT_CONTRACT,
    canonical_release_full_task_prompt,
    split_grade_a_rows,
)


def _rows() -> list[dict]:
    return [
        {
            "release_row_index": index,
            "uuid": f"uuid-{index}",
            "object": "cube1",
            "gesture": "03",
            "grade": "A",
        }
        for index in range(20)
    ]


def test_full_task_prompt_uses_formal_object_and_gesture() -> None:
    assert canonical_release_full_task_prompt(
        {"object": "cube1", "gesture": "03"},
        {"object_names": ["cube1"]},
    ) == "pick up the cube1 using gesture 03, then place it back on the table"
    assert CANONICAL_PROMPT_TEMPLATE.endswith("then place it back on the table")


def test_state45_counterfactual_requires_budget_above_200() -> None:
    audit = _counterfactual_token_audit(
        object_names=["cube1"],
        max_tokens=224,
        gesture_ids=[0],
        bin_ids=[100],
    )
    assert audit["max"] == 209
    assert audit["overflow_count"] == 0
    assert audit["historical_200_overflow_count"] == 1
    assert audit["fits_historical_200"] is False
    with pytest.raises(RuntimeError, match="above max_token_len=200"):
        _counterfactual_token_audit(
            object_names=["cube1"],
            max_tokens=200,
            gesture_ids=[0],
            bin_ids=[100],
        )


def test_state45_split_keeps_state41_allocation_but_relabels_full_task() -> None:
    split = split_grade_a_rows(_rows(), validation_fraction=0.1, split_seed=42)
    assert split["contract"] == SPLIT_CONTRACT
    assert split["train_rows"] == 18
    assert split["validation_rows"] == 2
    for row in split["train"] + split["validation"]:
        assert row["prompt"] == (
            "pick up the cube1 using gesture 03, then place it back on the table"
        )
