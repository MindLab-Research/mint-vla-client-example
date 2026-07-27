"""Canonical gesture-language metadata for generated MANO trajectories.

The copied dataset index is the source of truth for semantic action labels:
``gesture`` is the action class, ``seed_uuid`` identifies one exact source
trajectory, and ``uuid`` identifies one generated replica row.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_GESTURE_INDEX_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/datasets/new_all_generated_mano.index.json"
)
_GESTURE_PATTERN = re.compile(r"^[0-9]{2}$")


@dataclass(frozen=True)
class GestureRecord:
    row_index: int
    uuid: str
    seed_uuid: str
    object_type: str
    gesture: str
    sequence_id: str
    trajectory_name: str
    total_frames: int


class GestureIndex:
    """Validated row/UUID index for semantic MANO gesture labels."""

    def __init__(self, path: Path, payload: dict[str, Any], sha256: str) -> None:
        self.path = path
        self.sha256 = sha256
        self.version = str(payload.get("version") or "")
        self.dataset_name = str(payload.get("dataset_name") or "")
        raw_entries = payload.get("entries")
        if self.version != "1.0":
            raise ValueError(f"unsupported gesture index version {self.version!r}: {path}")
        if self.dataset_name != "new_all_generated_mano.lance":
            raise ValueError(
                f"unexpected gesture index dataset {self.dataset_name!r}: {path}"
            )
        if not isinstance(raw_entries, list) or not raw_entries:
            raise ValueError(f"gesture index entries must be a non-empty list: {path}")

        records: list[GestureRecord] = []
        seen_uuids: set[str] = set()
        gesture_by_source: dict[tuple[str, str], str] = {}
        for expected_row, entry in enumerate(raw_entries):
            if not isinstance(entry, dict):
                raise ValueError(f"gesture index row {expected_row} is not an object")
            row_index = entry.get("row_index")
            if isinstance(row_index, bool) or row_index != expected_row:
                raise ValueError(
                    f"gesture index must be contiguous and ordered: "
                    f"position={expected_row}, row_index={row_index!r}"
                )
            gesture = entry.get("gesture")
            if not isinstance(gesture, str) or not _GESTURE_PATTERN.fullmatch(gesture):
                raise ValueError(
                    f"gesture index row {expected_row} has invalid gesture {gesture!r}"
                )
            if entry.get("action_id") != gesture:
                raise ValueError(
                    f"gesture/action_id mismatch at row {expected_row}: "
                    f"{gesture!r} != {entry.get('action_id')!r}"
                )
            uuid = entry.get("uuid")
            seed_uuid = entry.get("seed_uuid")
            object_type = entry.get("object_type")
            sequence_id = entry.get("sequence_id")
            trajectory_name = entry.get("trajectory_name")
            total_frames = entry.get("total_frames")
            for key, value in (
                ("uuid", uuid),
                ("seed_uuid", seed_uuid),
                ("object_type", object_type),
                ("sequence_id", sequence_id),
                ("trajectory_name", trajectory_name),
            ):
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"gesture index row {expected_row} has invalid {key}={value!r}"
                    )
            if uuid in seen_uuids:
                raise ValueError(f"duplicate generated uuid {uuid!r} in gesture index")
            seen_uuids.add(uuid)
            if isinstance(total_frames, bool) or not isinstance(total_frames, int) or total_frames <= 0:
                raise ValueError(
                    f"gesture index row {expected_row} has invalid total_frames={total_frames!r}"
                )
            source_key = (object_type, seed_uuid)
            existing = gesture_by_source.setdefault(source_key, gesture)
            if existing != gesture:
                raise ValueError(
                    f"exact source {source_key!r} aliases gestures {existing!r} and {gesture!r}"
                )
            records.append(
                GestureRecord(
                    row_index=row_index,
                    uuid=uuid,
                    seed_uuid=seed_uuid,
                    object_type=object_type,
                    gesture=gesture,
                    sequence_id=sequence_id,
                    trajectory_name=trajectory_name,
                    total_frames=total_frames,
                )
            )
        self._records = tuple(records)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_GESTURE_INDEX_PATH) -> "GestureIndex":
        resolved = Path(path).expanduser().resolve()
        raw = resolved.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"gesture index root must be an object: {resolved}")
        return cls(resolved, payload, hashlib.sha256(raw).hexdigest())

    def __len__(self) -> int:
        return len(self._records)

    def record_for(
        self,
        row_index: int,
        *,
        uuid: str,
        seed_uuid: str,
        object_type: str,
        total_frames: int | None = None,
    ) -> GestureRecord:
        if isinstance(row_index, bool) or not isinstance(row_index, int):
            raise TypeError(f"row_index must be an integer, got {row_index!r}")
        if not 0 <= row_index < len(self._records):
            raise IndexError(f"gesture row index out of range: {row_index}")
        record = self._records[row_index]
        observed = {
            "uuid": uuid,
            "seed_uuid": seed_uuid,
            "object_type": object_type,
        }
        expected = {
            "uuid": record.uuid,
            "seed_uuid": record.seed_uuid,
            "object_type": record.object_type,
        }
        mismatches = {
            key: {"expected": expected[key], "observed": observed[key]}
            for key in expected
            if observed[key] != expected[key]
        }
        if total_frames is not None and int(total_frames) != record.total_frames:
            mismatches["total_frames"] = {
                "expected": record.total_frames,
                "observed": int(total_frames),
            }
        if mismatches:
            raise ValueError(
                f"gesture index does not align with Lance row {row_index}: {mismatches}"
            )
        return record


def format_gesture_prompt(base_prompt: str, gesture: str) -> str:
    """Append the canonical semantic gesture class to an object task prompt."""
    if not isinstance(base_prompt, str) or not base_prompt.strip():
        raise ValueError("language prompt must be a non-empty string")
    if not isinstance(gesture, str) or not _GESTURE_PATTERN.fullmatch(gesture):
        raise ValueError(f"gesture must be a two-digit string, got {gesture!r}")
    return f"{base_prompt.strip()} using gesture {gesture}"
