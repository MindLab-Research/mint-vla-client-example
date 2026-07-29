from __future__ import annotations

import pytest

from scripts.eval import replay_mano_target_physics as replay


def test_grade_boundaries_are_contractual():
    assert replay.grade_from_max_error(0.0) == "A"
    assert replay.grade_from_max_error(0.029999) == "A"
    assert replay.grade_from_max_error(0.03) == "B"
    assert replay.grade_from_max_error(0.079999) == "B"
    assert replay.grade_from_max_error(0.08) == "C"


def test_parse_rows_uses_end_exclusive_ranges_and_deduplicates():
    assert replay.parse_rows("2,4:7,5", 10) == [2, 4, 5, 6]
    with pytest.raises(ValueError, match="outside"):
        replay.parse_rows("10", 10)
    with pytest.raises(ValueError, match="invalid"):
        replay.parse_rows("7:7", 10)


def test_population_validation_rejects_missing_rows_and_uuid_aliases():
    entries = [
        {"uuid": "u0"},
        {"uuid": "u1"},
    ]
    valid = [
        {"row_index": 0, "status": "ok", "row_uuid": "u0", "object": "cube"},
        {"row_index": 1, "status": "ok", "row_uuid": "u1", "object": "cube"},
    ]
    replay.validate_record_population(valid, [0, 1], entries, "cube")
    with pytest.raises(ValueError, match="population mismatch"):
        replay.validate_record_population(valid[:1], [0, 1], entries, "cube")
    aliased = [dict(valid[0]), {**valid[1], "row_uuid": "u0"}]
    with pytest.raises(ValueError, match="UUID"):
        replay.validate_record_population(aliased, [0, 1], entries, "cube")
