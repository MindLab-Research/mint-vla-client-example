"""Pure contracts for the full Grade-A state41/28DoF training population."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


CANONICAL_PROMPT_TEMPLATE = "pick up the {object} using gesture {gesture}"
SPLIT_CONTRACT = "mano_state41_grade_a_object_gesture_split_v1"
SELECTION_CONTRACT = "mano_state41_grade_a_selection_v1"
_GESTURE_PATTERN = re.compile(r"^[0-9]{2}$")


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"state41 Grade-A row has invalid {key}={value!r}")
    return value.strip()


def canonical_release_gesture_prompt(
    index: Mapping[str, Any], trajectory_metadata: Mapping[str, Any] | None = None
) -> str:
    """Build the canonical prompt from formal-release metadata, fail-closed."""
    object_name = _required_string(index, "object")
    gesture = _required_string(index, "gesture")
    if not _GESTURE_PATTERN.fullmatch(gesture):
        raise ValueError(f"state41 Grade-A gesture must be two digits, got {gesture!r}")
    if trajectory_metadata is not None:
        names = trajectory_metadata.get("object_names") or []
        if (
            not isinstance(names, Sequence)
            or isinstance(names, (str, bytes))
            or len(names) != 1
            or not isinstance(names[0], str)
        ):
            raise ValueError(
                "state41 Grade-A gesture prompt requires exactly one trajectory object"
            )
        if names[0] != object_name:
            raise ValueError(
                "state41 Grade-A object mismatch between index and trajectory metadata: "
                f"{object_name!r} != {names[0]!r}"
            )
    return CANONICAL_PROMPT_TEMPLATE.format(object=object_name, gesture=gesture)


def _split_score(seed: int, uuid: str) -> str:
    return hashlib.sha256(f"{int(seed)}:{uuid}".encode()).hexdigest()


def _group_score(seed: int, key: tuple[str, str]) -> str:
    return hashlib.sha256(f"{int(seed)}:{key[0]}:{key[1]}".encode()).hexdigest()


def _allocate_validation_counts(
    groups: Mapping[tuple[str, str], Sequence[dict[str, Any]]],
    *,
    validation_count: int,
    validation_fraction: float,
    seed: int,
) -> dict[tuple[str, str], int]:
    capacities = {key: max(0, len(rows) - 1) for key, rows in groups.items()}
    if validation_count > sum(capacities.values()):
        raise ValueError(
            "validation target cannot leave at least one training row per stratum: "
            f"target={validation_count} capacity={sum(capacities.values())}"
        )
    counts = {key: 0 for key in groups}
    eligible = [key for key, capacity in capacities.items() if capacity > 0]

    # Give every eligible object+gesture stratum one held-out row when the global
    # target permits it. Small strata otherwise remain train-only rather than
    # leaking their sole trajectory into validation.
    ranked_eligible = sorted(
        eligible,
        key=lambda key: (
            -(len(groups[key]) * validation_fraction),
            _group_score(seed, key),
            key,
        ),
    )
    for key in ranked_eligible[: min(validation_count, len(ranked_eligible))]:
        counts[key] = 1

    remaining = validation_count - sum(counts.values())
    while remaining:
        candidates = [key for key in groups if counts[key] < capacities[key]]
        if not candidates:
            raise RuntimeError("validation allocation exhausted all stratum capacity")
        key = min(
            candidates,
            key=lambda value: (
                -(len(groups[value]) * validation_fraction - counts[value]),
                _group_score(seed, value),
                value,
            ),
        )
        counts[key] += 1
        remaining -= 1
    return counts


def split_grade_a_rows(
    rows: Sequence[dict[str, Any]],
    *,
    validation_fraction: float = 0.05,
    split_seed: int = 42,
) -> dict[str, Any]:
    """Split Grade-A rows by UUID within object+gesture strata.

    The exact global validation count is round(N*fraction). Every stratum keeps
    at least one training row; singleton strata are train-only.
    """
    if not math.isfinite(validation_fraction) or not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be finite and in (0,1)")
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise TypeError("split_seed must be an integer")
    if not rows:
        raise ValueError("Grade-A split requires at least one row")

    normalized: list[dict[str, Any]] = []
    seen_uuids: set[str] = set()
    seen_release_rows: set[int] = set()
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        if row.get("grade") != "A":
            raise ValueError(f"Grade-A split received grade={row.get('grade')!r}")
        uuid = _required_string(row, "uuid")
        object_name = _required_string(row, "object")
        gesture = _required_string(row, "gesture")
        if not _GESTURE_PATTERN.fullmatch(gesture):
            raise ValueError(f"gesture must be two digits, got {gesture!r}")
        release_row = row.get("release_row_index")
        if isinstance(release_row, bool) or not isinstance(release_row, int) or release_row < 0:
            raise ValueError(f"invalid release_row_index={release_row!r}")
        if uuid in seen_uuids:
            raise ValueError(f"duplicate Grade-A UUID {uuid!r}")
        if release_row in seen_release_rows:
            raise ValueError(f"duplicate release row {release_row}")
        seen_uuids.add(uuid)
        seen_release_rows.add(release_row)
        row["uuid"] = uuid
        row["object"] = object_name
        row["gesture"] = gesture
        row["prompt"] = canonical_release_gesture_prompt(row)
        normalized.append(row)
        groups[(object_name, gesture)].append(row)

    validation_count = int(round(len(normalized) * validation_fraction))
    if validation_count <= 0 or validation_count >= len(normalized):
        raise ValueError(
            f"validation fraction produced invalid count {validation_count}/{len(normalized)}"
        )
    counts = _allocate_validation_counts(
        groups,
        validation_count=validation_count,
        validation_fraction=validation_fraction,
        seed=split_seed,
    )

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    strata: list[dict[str, Any]] = []
    for key in sorted(groups):
        ordered = sorted(
            groups[key],
            key=lambda row: (_split_score(split_seed, row["uuid"]), row["uuid"]),
        )
        count = counts[key]
        held_out = ordered[:count]
        retained = ordered[count:]
        if not retained:
            raise RuntimeError(f"stratum {key!r} has no training row")
        validation.extend(held_out)
        train.extend(retained)
        strata.append(
            {
                "object": key[0],
                "gesture": key[1],
                "population_rows": len(ordered),
                "train_rows": len(retained),
                "validation_rows": len(held_out),
            }
        )

    train.sort(key=lambda row: row["release_row_index"])
    validation.sort(key=lambda row: row["release_row_index"])
    if len(validation) != validation_count:
        raise RuntimeError(
            f"validation allocation mismatch {len(validation)} != {validation_count}"
        )
    train_uuids = {row["uuid"] for row in train}
    validation_uuids = {row["uuid"] for row in validation}
    if train_uuids & validation_uuids:
        raise RuntimeError("train/validation UUID leakage")
    if train_uuids | validation_uuids != seen_uuids:
        raise RuntimeError("train/validation split does not cover Grade-A population")

    return {
        "contract": SPLIT_CONTRACT,
        "split_seed": int(split_seed),
        "validation_fraction": float(validation_fraction),
        "population_rows": len(normalized),
        "train": train,
        "validation": validation,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_uuid_sha256": canonical_sha256([row["uuid"] for row in train]),
        "validation_uuid_sha256": canonical_sha256(
            [row["uuid"] for row in validation]
        ),
        "train_row_index_sha256": canonical_sha256(
            [row["release_row_index"] for row in train]
        ),
        "validation_row_index_sha256": canonical_sha256(
            [row["release_row_index"] for row in validation]
        ),
        "strata": strata,
    }


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
