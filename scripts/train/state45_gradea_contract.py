"""Pure language contract for phase-aware full-task State45 Grade-A rows."""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from scripts.mano_state45_contract import FULL_TASK_PROMPT_TEMPLATE


CANONICAL_PROMPT_TEMPLATE = FULL_TASK_PROMPT_TEMPLATE
_GESTURE_PATTERN = re.compile(r"^[0-9]{2}$")


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"State45 Grade-A row has invalid {key}={value!r}")
    return value.strip()


def canonical_release_full_task_prompt(
    index: Mapping[str, Any], trajectory_metadata: Mapping[str, Any] | None = None
) -> str:
    object_name = _required_string(index, "object")
    gesture = _required_string(index, "gesture")
    if not _GESTURE_PATTERN.fullmatch(gesture):
        raise ValueError(f"State45 Grade-A gesture must be two digits, got {gesture!r}")
    if trajectory_metadata is not None:
        names = trajectory_metadata.get("object_names") or []
        if (
            not isinstance(names, Sequence)
            or isinstance(names, (str, bytes))
            or len(names) != 1
            or not isinstance(names[0], str)
        ):
            raise ValueError(
                "State45 Grade-A full-task prompt requires exactly one trajectory object"
            )
        if names[0] != object_name:
            raise ValueError(
                "State45 Grade-A object mismatch between index and trajectory metadata: "
                f"{object_name!r} != {names[0]!r}"
            )
    return CANONICAL_PROMPT_TEMPLATE.format(object=object_name, gesture=gesture)

SPLIT_CONTRACT = "mano_state45_grade_a_object_gesture_split_v1"
SELECTION_CONTRACT = "mano_state45_grade_a_selection_v1"


def split_grade_a_rows(
    rows: Sequence[dict[str, Any]],
    *,
    validation_fraction: float = 0.05,
    split_seed: int = 42,
) -> dict[str, Any]:
    """Reuse the proven State41 allocation while replacing only task language."""
    from scripts.train import state41_gradea_contract as state41

    result = state41.split_grade_a_rows(
        rows,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
    )
    result["contract"] = SPLIT_CONTRACT
    for split_name in ("train", "validation"):
        for row in result[split_name]:
            row["prompt"] = canonical_release_full_task_prompt(row)
    return result


def selection_manifest(
    split: Mapping[str, Any],
    *,
    split_name: str,
    dataset: str,
    release_verification_sha256: str,
) -> dict[str, Any]:
    if split_name not in {"train", "validation"}:
        raise ValueError("split_name must be train or validation")
    rows = list(split[split_name])
    return {
        "contract": SELECTION_CONTRACT,
        "split": split_name,
        "dataset": dataset,
        "release_verification_sha256": release_verification_sha256,
        "split_contract": split["contract"],
        "split_seed": split["split_seed"],
        "validation_fraction": split["validation_fraction"],
        "population_rows": split["population_rows"],
        "selected_rows": len(rows),
        "row_index_sha256": split[f"{split_name}_row_index_sha256"],
        "uuid_sha256": split[f"{split_name}_uuid_sha256"],
        "rows": rows,
    }
