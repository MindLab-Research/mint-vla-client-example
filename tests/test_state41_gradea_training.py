from __future__ import annotations

import copy

import pytest

from scripts.train.state41_gradea_contract import (
    canonical_release_gesture_prompt,
    selection_manifest,
    split_grade_a_rows,
)


def _rows(group_sizes: dict[tuple[str, str], int]) -> list[dict]:
    rows: list[dict] = []
    row_index = 0
    for (object_name, gesture), count in group_sizes.items():
        for ordinal in range(count):
            rows.append(
                {
                    "release_row_index": row_index,
                    "uuid": f"{object_name}-{gesture}-{ordinal:04d}",
                    "object": object_name,
                    "gesture": gesture,
                    "grade": "A",
                    "frames": 100 + ordinal,
                }
            )
            row_index += 1
    return rows


def test_canonical_release_gesture_prompt_uses_formal_metadata() -> None:
    assert canonical_release_gesture_prompt(
        {"object": "cube1", "gesture": "03"},
        {"object_names": ["cube1"]},
    ) == "pick up the cube1 using gesture 03"
    with pytest.raises(ValueError, match="two digits"):
        canonical_release_gesture_prompt({"object": "cube1", "gesture": "3"})
    with pytest.raises(ValueError, match="object mismatch"):
        canonical_release_gesture_prompt(
            {"object": "cube1", "gesture": "03"},
            {"object_names": ["banana"]},
        )


def test_grade_a_split_is_exact_deterministic_and_leak_free() -> None:
    rows = _rows({("cube1", "03"): 50, ("banana", "07"): 30, ("bowl", "01"): 20})
    first = split_grade_a_rows(rows, validation_fraction=0.05, split_seed=42)
    second = split_grade_a_rows(copy.deepcopy(rows), validation_fraction=0.05, split_seed=42)
    assert first == second
    assert first["population_rows"] == 100
    assert first["train_rows"] == 95
    assert first["validation_rows"] == 5
    assert {row["uuid"] for row in first["train"]}.isdisjoint(
        row["uuid"] for row in first["validation"]
    )
    assert all(row["prompt"].endswith(f"gesture {row['gesture']}") for row in first["train"])
    assert all(value["train_rows"] > 0 for value in first["strata"])
    assert all(value["validation_rows"] > 0 for value in first["strata"])


def test_grade_a_split_seed_changes_membership_not_contract() -> None:
    rows = _rows({("cube1", "03"): 80, ("banana", "07"): 40})
    first = split_grade_a_rows(rows, validation_fraction=0.1, split_seed=42)
    second = split_grade_a_rows(rows, validation_fraction=0.1, split_seed=43)
    assert first["validation_rows"] == second["validation_rows"] == 12
    assert first["validation_uuid_sha256"] != second["validation_uuid_sha256"]


def test_singleton_stratum_remains_train_only() -> None:
    rows = _rows({("rare", "01"): 1, ("cube1", "03"): 19})
    split = split_grade_a_rows(rows, validation_fraction=0.1, split_seed=42)
    rare = next(value for value in split["strata"] if value["object"] == "rare")
    assert rare == {
        "object": "rare",
        "gesture": "01",
        "population_rows": 1,
        "train_rows": 1,
        "validation_rows": 0,
    }


def test_grade_a_split_rejects_wrong_grade_and_duplicate_uuid() -> None:
    rows = _rows({("cube1", "03"): 20})
    rows[0]["grade"] = "B"
    with pytest.raises(ValueError, match="grade='B'"):
        split_grade_a_rows(rows)
    rows = _rows({("cube1", "03"): 20})
    rows[1]["uuid"] = rows[0]["uuid"]
    with pytest.raises(ValueError, match="duplicate Grade-A UUID"):
        split_grade_a_rows(rows)


def test_selection_manifest_preserves_split_hashes() -> None:
    rows = _rows({("cube1", "03"): 20})
    split = split_grade_a_rows(rows, validation_fraction=0.1, split_seed=42)
    manifest = selection_manifest(
        split,
        split_name="train",
        dataset="/release/state41.lance",
        release_verification_sha256="a" * 64,
    )
    assert manifest["selected_rows"] == 18
    assert manifest["row_index_sha256"] == split["train_row_index_sha256"]
    assert manifest["uuid_sha256"] == split["train_uuid_sha256"]
    assert all(row["grade"] == "A" for row in manifest["rows"])
