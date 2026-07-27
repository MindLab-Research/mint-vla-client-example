"""Contact-centered trajectory window selection for MANO Lance episodes.

The dataset stores one ``contact`` value per source frame.  A contact value is
considered relevant when it contains a contact record whose ``object_name``
matches the episode's target object.  The default window is inclusive:

    start = max(0, first_matching_contact - context_frames)
    end   = min(total_frames - 1, last_matching_contact + context_frames)

The module is deliberately independent of OpenPI so it can be used by data
inspection, training, and inference clients.  A small JSON sidecar manifest can
cache the expensive contact-column scan without modifying the Lance dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_CONTACT_CONTEXT_FRAMES = 100
MANIFEST_VERSION = 1
CONTACT_COLUMNS = ("contact", "trajectory_metadata", "episode_metadata")


@dataclass(frozen=True)
class ContactWindow:
    """One trajectory's resolved, inclusive source-frame window."""

    row_index: int
    object_name: str
    total_frames: int
    first_contact_frame: int | None
    last_contact_frame: int | None
    start_frame: int
    end_frame: int
    contact_frame_count: int
    matching_contact_record_count: int
    contact_record_frame_count: int
    status: str
    context_frames: int = DEFAULT_CONTACT_CONTEXT_FRAMES

    @property
    def frame_count(self) -> int:
        return max(0, self.end_frame - self.start_frame + 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "object_name": self.object_name,
            "total_frames": self.total_frames,
            "first_contact_frame": self.first_contact_frame,
            "last_contact_frame": self.last_contact_frame,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frame_count": self.frame_count,
            "contact_frame_count": self.contact_frame_count,
            "matching_contact_record_count": self.matching_contact_record_count,
            "contact_record_frame_count": self.contact_record_frame_count,
            "status": self.status,
            "context_frames": self.context_frames,
        }


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def object_name_from_row(row: Mapping[str, Any]) -> str:
    metadata = _as_mapping(row.get("trajectory_metadata")) or {}
    names = metadata.get("object_names") or []
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)) and names:
        return str(names[0])
    objects = row.get("objects") or []
    if isinstance(objects, Sequence) and objects:
        first = _as_mapping(objects[0]) or {}
        name = first.get("object_name") or first.get("name")
        if name:
            return str(name)
    return "unknown"


def row_total_frames(row: Mapping[str, Any]) -> int:
    """Use the shortest available trajectory column to avoid out-of-bounds use."""
    lengths: list[int] = []
    for key in ("state", "actions", "image", "wrist_image"):
        value = row.get(key)
        if value is not None:
            try:
                lengths.append(len(value))
            except TypeError:
                pass
    objects = row.get("objects") or []
    if isinstance(objects, Sequence):
        for obj in objects:
            obj_map = _as_mapping(obj) or {}
            for key in ("pos", "rot_aa"):
                value = obj_map.get(key)
                if value is not None:
                    try:
                        lengths.append(len(value))
                    except TypeError:
                        pass
    episode_metadata = _as_mapping(row.get("episode_metadata")) or {}
    try:
        metadata_frames = int(episode_metadata.get("total_frames") or 0)
    except (TypeError, ValueError):
        metadata_frames = 0
    if metadata_frames > 0:
        lengths.append(metadata_frames)
    return min(lengths) if lengths else max(0, metadata_frames)


def _contact_records_for_frame(frame_value: Any) -> list[Mapping[str, Any]]:
    """Normalize the Lance list/struct representation for one frame."""
    if frame_value is None:
        return []
    if isinstance(frame_value, Mapping):
        return [frame_value]
    if isinstance(frame_value, Sequence) and not isinstance(frame_value, (str, bytes)):
        return [item for item in frame_value if isinstance(item, Mapping)]
    return []


def matching_contact_frames(
    row: Mapping[str, Any], *, object_name: str | None = None, total_frames: int | None = None
) -> tuple[list[int], int, int, str]:
    """Return matching frame indices and contact evidence counts.

    The fourth return value distinguishes missing contact data, no contact, and
    contact belonging only to another object.  It is intentionally explicit so
    callers can record why a full-trajectory fallback was selected.
    """
    if "contact" not in row or row.get("contact") is None:
        return [], 0, 0, "missing_contact_column"
    contact = row.get("contact")
    if not isinstance(contact, Sequence) or isinstance(contact, (str, bytes)):
        return [], 0, 0, "invalid_contact_column"
    limit = len(contact) if total_frames is None else min(len(contact), max(0, int(total_frames)))
    target = object_name or object_name_from_row(row)
    matching: list[int] = []
    contact_frame_count = 0
    matching_record_count = 0
    for frame_index in range(limit):
        records = _contact_records_for_frame(contact[frame_index])
        if records:
            contact_frame_count += 1
        frame_matches = 0
        for record in records:
            record_object = record.get("object_name")
            # If an episode has no known target, any contact record is useful.
            if not target or target == "unknown" or record_object is None or str(record_object) == target:
                frame_matches += 1
        if frame_matches:
            matching.append(frame_index)
            matching_record_count += frame_matches
    if matching:
        status = "contact_window"
    elif contact_frame_count:
        status = "contact_for_other_object"
    else:
        status = "no_contact_evidence"
    return matching, matching_record_count, contact_frame_count, status


def resolve_contact_window(
    *,
    row_index: int,
    object_name: str,
    total_frames: int,
    first_contact_frame: int | None,
    last_contact_frame: int | None,
    contact_frame_count: int = 0,
    matching_contact_record_count: int = 0,
    contact_record_frame_count: int = 0,
    evidence_status: str = "contact_window",
    context_frames: int = DEFAULT_CONTACT_CONTEXT_FRAMES,
    missing_policy: str = "full",
) -> ContactWindow | None:
    """Resolve an inclusive window, handling absent contact evidence explicitly.

    ``missing_policy`` is one of ``full`` (keep the entire trajectory and mark
    the reason), ``skip`` (return ``None``), or ``error`` (raise).  ``full`` is
    the default because it preserves training coverage for episodes that have
    no contact annotation, while the status remains visible in manifests and
    run metadata.
    """
    total = max(0, int(total_frames))
    context = max(0, int(context_frames))
    policy = str(missing_policy).strip().lower()
    if policy not in {"full", "skip", "error"}:
        raise ValueError(f"missing_policy must be full, skip, or error; got {missing_policy!r}")

    has_contact = (
        total > 0
        and first_contact_frame is not None
        and last_contact_frame is not None
        and 0 <= int(first_contact_frame) <= int(last_contact_frame) < total
    )
    if has_contact:
        start = max(0, int(first_contact_frame) - context)
        end = min(total - 1, int(last_contact_frame) + context)
        status = "contact_window"
    else:
        reason = evidence_status or "no_contact_evidence"
        if policy == "error":
            raise ValueError(
                f"row {row_index} ({object_name}) has no valid contact interval: {reason}"
            )
        if policy == "skip":
            return None
        start = 0
        end = total - 1
        status = f"full_fallback:{reason}"

    if total <= 0:
        start, end = 0, -1
        status = "empty_trajectory"
    return ContactWindow(
        row_index=int(row_index),
        object_name=str(object_name),
        total_frames=total,
        first_contact_frame=None if first_contact_frame is None else int(first_contact_frame),
        last_contact_frame=None if last_contact_frame is None else int(last_contact_frame),
        start_frame=start,
        end_frame=end,
        contact_frame_count=int(contact_frame_count),
        matching_contact_record_count=int(matching_contact_record_count),
        contact_record_frame_count=int(contact_record_frame_count),
        status=status,
        context_frames=context,
    )


def window_from_row(
    row: Mapping[str, Any],
    *,
    row_index: int = -1,
    total_frames: int | None = None,
    context_frames: int = DEFAULT_CONTACT_CONTEXT_FRAMES,
    missing_policy: str = "full",
) -> ContactWindow | None:
    total = row_total_frames(row) if total_frames is None else max(0, int(total_frames))
    object_name = object_name_from_row(row)
    frames, matching_records, contact_frame_count, evidence_status = matching_contact_frames(
        row, object_name=object_name, total_frames=total
    )
    first = frames[0] if frames else None
    last = frames[-1] if frames else None
    return resolve_contact_window(
        row_index=row_index,
        object_name=object_name,
        total_frames=total,
        first_contact_frame=first,
        last_contact_frame=last,
        contact_frame_count=len(frames),
        matching_contact_record_count=matching_records,
        contact_record_frame_count=contact_frame_count,
        evidence_status=evidence_status,
        context_frames=context_frames,
        missing_policy=missing_policy,
    )


def clamp_window(window: ContactWindow, total_frames: int) -> ContactWindow:
    """Clamp a manifest entry to the arrays actually loaded by a consumer."""
    total = max(0, int(total_frames))
    if total == window.total_frames:
        return window
    if total <= 0:
        return ContactWindow(
            row_index=window.row_index,
            object_name=window.object_name,
            total_frames=0,
            first_contact_frame=None,
            last_contact_frame=None,
            start_frame=0,
            end_frame=-1,
            contact_frame_count=window.contact_frame_count,
            matching_contact_record_count=window.matching_contact_record_count,
            contact_record_frame_count=window.contact_record_frame_count,
            status="empty_trajectory",
            context_frames=window.context_frames,
        )
    start = min(max(0, window.start_frame), total - 1)
    end = min(max(start, window.end_frame), total - 1)
    return ContactWindow(
        row_index=window.row_index,
        object_name=window.object_name,
        total_frames=total,
        first_contact_frame=window.first_contact_frame if window.first_contact_frame is None or window.first_contact_frame < total else None,
        last_contact_frame=window.last_contact_frame if window.last_contact_frame is None or window.last_contact_frame < total else None,
        start_frame=start,
        end_frame=end,
        contact_frame_count=window.contact_frame_count,
        matching_contact_record_count=window.matching_contact_record_count,
        contact_record_frame_count=window.contact_record_frame_count,
        status=window.status,
        context_frames=window.context_frames,
    )


def default_manifest_path(dataset_path: str | Path) -> Path:
    return Path(f"{dataset_path}.contact_windows.json")


def _manifest_entries(raw: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    values = raw.get("windows") or raw.get("trajectories") or {}
    if isinstance(values, Mapping):
        return {int(key): dict(value) for key, value in values.items()}
    if isinstance(values, Sequence):
        return {
            int(value["row_index"]): dict(value)
            for value in values
            if isinstance(value, Mapping) and "row_index" in value
        }
    raise ValueError("contact-window manifest windows must be an object or list")


def load_manifest(path: str | Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    manifest_path = Path(path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"contact-window manifest is not an object: {manifest_path}")
    return dict(raw), _manifest_entries(raw)


def write_manifest(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(destination.parent),
        prefix=f".{destination.name}.", suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(destination)


def _dataset_columns(dataset: Any) -> set[str]:
    schema = getattr(dataset, "schema", None)
    names = getattr(schema, "names", None)
    if names is not None:
        return {str(name) for name in names}
    try:
        return {str(field.name) for field in schema}
    except TypeError:
        return set()


def _scan_rows(
    dataset: Any,
    row_indices: Sequence[int],
    *,
    batch_rows: int,
    context_frames: int,
    missing_policy: str,
) -> dict[int, dict[str, Any]]:
    indices = [int(index) for index in row_indices]
    if not indices:
        return {}
    available = _dataset_columns(dataset)
    columns = [column for column in CONTACT_COLUMNS if column in available]
    if "episode_metadata" not in columns:
        # The resolver can still use the contact list and target metadata; this
        # branch is mainly for tiny synthetic test datasets.
        columns = [column for column in columns if column != "episode_metadata"]
    result: dict[int, dict[str, Any]] = {}
    step = max(1, int(batch_rows))
    for offset in range(0, len(indices), step):
        batch_indices = indices[offset : offset + step]
        rows = dataset.take(batch_indices, columns=columns).to_pylist() if columns else [{} for _ in batch_indices]
        if len(rows) != len(batch_indices):
            raise RuntimeError(
                f"Lance returned {len(rows)} rows for {len(batch_indices)} requested rows"
            )
        for row_index, row in zip(batch_indices, rows, strict=True):
            # Metadata-only scans intentionally avoid images/actions. Use the
            # episode metadata count; if unavailable, the row resolver can only
            # report an empty/unknown trajectory and callers should override it.
            metadata = _as_mapping(row.get("episode_metadata")) or {}
            total = int(metadata.get("total_frames") or 0)
            window = window_from_row(
                row,
                row_index=row_index,
                total_frames=total,
                context_frames=context_frames,
                missing_policy=missing_policy,
            )
            if window is not None:
                result[row_index] = window.as_dict()
    return result


def load_or_build_windows(
    dataset: Any,
    dataset_path: str | Path,
    row_indices: Iterable[int],
    *,
    manifest_path: str | Path | None = None,
    context_frames: int = DEFAULT_CONTACT_CONTEXT_FRAMES,
    missing_policy: str = "full",
    batch_rows: int = 256,
    cache: bool = True,
) -> dict[int, dict[str, Any]]:
    """Load requested windows, scanning only missing rows and optionally cache."""
    requested = list(dict.fromkeys(int(index) for index in row_indices))
    if not requested:
        return {}
    destination = Path(manifest_path) if manifest_path else default_manifest_path(dataset_path)
    raw: dict[str, Any] = {}
    entries: dict[int, dict[str, Any]] = {}
    if destination.exists():
        raw, entries = load_manifest(destination)
        try:
            row_count = int(dataset.count_rows())
        except Exception:
            row_count = None
        mismatches: list[str] = []
        if int(raw.get("manifest_version", MANIFEST_VERSION)) != MANIFEST_VERSION:
            mismatches.append("manifest_version")
        if raw.get("dataset") not in (None, str(dataset_path)):
            mismatches.append("dataset")
        if int(raw.get("context_frames", context_frames)) != int(context_frames):
            mismatches.append("context_frames")
        if str(raw.get("missing_policy", missing_policy)) != str(missing_policy):
            mismatches.append("missing_policy")
        if row_count is not None and raw.get("row_count") not in (None, row_count):
            mismatches.append("row_count")
        if mismatches:
            raise ValueError(
                f"contact-window manifest contract mismatch at {destination}: {', '.join(mismatches)}. "
                "Use a distinct manifest path or explicitly rebuild this manifest; the existing file was not modified."
            )
    missing = [index for index in requested if index not in entries]
    if missing:
        entries.update(
            _scan_rows(
                dataset,
                missing,
                batch_rows=batch_rows,
                context_frames=context_frames,
                missing_policy=missing_policy,
            )
        )
        if cache:
            try:
                row_count = int(dataset.count_rows())
            except Exception:
                row_count = None
            payload = {
                "manifest_version": MANIFEST_VERSION,
                "dataset": str(dataset_path),
                "row_count": row_count,
                "context_frames": int(context_frames),
                "missing_policy": str(missing_policy),
                "windows": {str(index): entries[index] for index in sorted(entries)},
            }
            write_manifest(destination, payload)
    unresolved = [index for index in requested if index not in entries]
    if unresolved and missing_policy == "error":
        raise ValueError(f"no contact windows resolved for rows: {unresolved}")
    return {index: entries[index] for index in requested if index in entries}


def window_from_manifest_entry(
    entry: Mapping[str, Any], *, total_frames: int, row_index: int | None = None
) -> ContactWindow:
    data = dict(entry)
    data.pop("frame_count", None)
    if row_index is not None:
        data["row_index"] = int(row_index)
    # Older/minimal manifests may not contain all evidence counters.
    data.setdefault("object_name", "unknown")
    data.setdefault("first_contact_frame", None)
    data.setdefault("last_contact_frame", None)
    data.setdefault("contact_frame_count", 0)
    data.setdefault("matching_contact_record_count", 0)
    data.setdefault("contact_record_frame_count", 0)
    data.setdefault("status", "manifest")
    data.setdefault("context_frames", DEFAULT_CONTACT_CONTEXT_FRAMES)
    data["total_frames"] = int(data.get("total_frames", total_frames))
    data["row_index"] = int(data.get("row_index", row_index if row_index is not None else -1))
    return clamp_window(ContactWindow(**data), total_frames)


def select_window(
    row: Mapping[str, Any],
    *,
    row_index: int,
    total_frames: int,
    mode: str = "contact",
    manifest_entry: Mapping[str, Any] | None = None,
    context_frames: int = DEFAULT_CONTACT_CONTEXT_FRAMES,
    missing_policy: str = "full",
) -> ContactWindow | None:
    """Resolve a consumer window with an explicit full-window diagnostic override."""
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "full":
        return resolve_contact_window(
            row_index=row_index,
            object_name=object_name_from_row(row),
            total_frames=total_frames,
            first_contact_frame=None,
            last_contact_frame=None,
            evidence_status="explicit_full_override",
            context_frames=context_frames,
            missing_policy="full",
        )
    if normalized_mode != "contact":
        raise ValueError(f"window mode must be contact or full, got {mode!r}")
    if manifest_entry is not None:
        return window_from_manifest_entry(manifest_entry, total_frames=total_frames, row_index=row_index)
    return window_from_row(
        row, row_index=row_index, total_frames=total_frames,
        context_frames=context_frames, missing_policy=missing_policy,
    )
