#!/usr/bin/env python3
"""Atomically promote MANO PD targets into an image-enriched Lance dataset.

The staged raw dataset must be an exact extension of the raw data embedded in
``final``: it adds only ``hands[].urdf_dof_target``.  This script never
projects image/blob fields and never rewrites their fragments.

Example (first inspect, then apply only after the JSON preflight is clean)::

    python scripts/tools/migrate_mano_target_column.py \
      --final /vePFS-Mindverse/user/intern/wenxi/results/datas/new_all_generated_mano_with_images.lance \
      --target /vePFS-Mindverse/user/intern/wenxi/results/datas/staging/new_all_generated_mano_with_target.lance \
      --expected-final-version 17 --dry-run

    python scripts/tools/migrate_mano_target_column.py \
      --final /vePFS-Mindverse/user/intern/wenxi/results/datas/new_all_generated_mano_with_images.lance \
      --target /vePFS-Mindverse/user/intern/wenxi/results/datas/staging/new_all_generated_mano_with_target.lance \
      --expected-final-version 17 --apply --report /tmp/mano-target-migration.json

Lance 8.0.0 can panic on nested dot projections.  The implementation always
reads complete ``index`` and ``hands`` structs instead.  Its intermediate
versions are reader-safe: first it adds a top-level ``hands_with_target``;
then it atomically swaps names while retaining ``hands_without_target``; last
it drops that backup metadata-only.  At no point is a latest version missing a
valid ``hands`` column.

``urdf_dof_target`` is preserved as provenance.  The script intentionally does
not modify ``state`` or the measured-next-state-delta ``actions`` contract.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa

try:  # Unit-tested helpers do not require pylance.
    import lance
    from lance.schema import LanceSchema
except ImportError:  # pragma: no cover - exercised only on non-runtime hosts
    lance = None
    LanceSchema = None


TARGET_FIELD = "urdf_dof_target"
STAGED_HANDS_COLUMN = "hands_with_target"
BACKUP_HANDS_COLUMN = "hands_without_target"
EXIT_OK = 0
EXIT_GUARD = 2
EXIT_RUNTIME = 3
# The legacy v17 `hands` files were written by an older Lance encoder.  Lance
# 8.0.0 mis-decodes multi-row slices from those files, while one-row reads are
# stable (the production loader also reads one episode row at a time).
LEGACY_HANDS_BATCH_ROWS = 1
TARGET_READER_BATCH_ROWS = 64


class MigrationError(RuntimeError):
    """A validated guard failure which must prevent a dataset write."""


@dataclass(frozen=True)
class SchemaPlan:
    """Validated nested-hand schema information used by the migration."""

    legacy_hand_fields: tuple[str, ...]
    target_hand_type: pa.DataType


def _require_lance() -> None:
    if lance is None or LanceSchema is None:
        raise MigrationError(
            "pylance is unavailable; run this script in the deployed Lance 8 runtime"
        )


def _combine(column: pa.Array | pa.ChunkedArray) -> pa.Array:
    return column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column


def _field_names(hands_type: pa.DataType) -> tuple[str, ...]:
    if not pa.types.is_list(hands_type) or not pa.types.is_struct(hands_type.value_type):
        raise MigrationError(f"hands must be list<struct<...>>, got {hands_type}")
    return tuple(field.name for field in hands_type.value_type)


def hands_field_names(schema: pa.Schema) -> tuple[str, ...]:
    """Return child names of the required top-level ``hands`` list struct."""

    try:
        hands = schema.field("hands")
    except KeyError as exc:
        raise MigrationError("dataset schema has no top-level hands column") from exc
    return _field_names(hands.type)


def validate_schema_extension(final_schema: pa.Schema, target_schema: pa.Schema) -> SchemaPlan:
    """Require the staged raw schema to add exactly one nested target child.

    The final dataset may contain unrelated top-level training fields, but its
    existing hands child schema must exactly match the target schema before the
    appended target field.
    """

    final_names = hands_field_names(final_schema)
    target_names = hands_field_names(target_schema)
    if TARGET_FIELD in final_names:
        raise MigrationError(f"final hands already contains {TARGET_FIELD}")
    if target_names != (*final_names, TARGET_FIELD):
        raise MigrationError(
            "target hands must equal final hands children plus exactly "
            f"{TARGET_FIELD}: final={final_names}, target={target_names}"
        )
    final_type = final_schema.field("hands").type.value_type
    target_type = target_schema.field("hands").type.value_type
    for name in final_names:
        if final_type.field(name).type != target_type.field(name).type:
            raise MigrationError(f"hands child type differs for {name}")
    target_field = target_type.field(TARGET_FIELD)
    _validate_target_type(target_field.type)
    return SchemaPlan(final_names, target_field.type)


def _validate_target_type(data_type: pa.DataType) -> None:
    if not pa.types.is_list(data_type):
        raise MigrationError(f"{TARGET_FIELD} must be a per-frame list, got {data_type}")
    item = data_type.value_type
    if not pa.types.is_fixed_size_list(item) or item.list_size != 26:
        raise MigrationError(
            f"{TARGET_FIELD} must be list<fixed_size_list<float>[26]>, got {data_type}"
        )
    if item.value_type != pa.float32():
        raise MigrationError(f"{TARGET_FIELD} scalar type must be float32, got {item.value_type}")


def _list_offsets(array: pa.Array) -> np.ndarray:
    if not pa.types.is_list(array.type):
        raise MigrationError(f"expected list array, got {array.type}")
    return np.asarray(array.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)


def _require_no_nulls(array: pa.Array, name: str) -> None:
    if array.null_count:
        raise MigrationError(f"{name} contains {array.null_count} null entries")


def validate_target_array(target_hands: pa.Array | pa.ChunkedArray) -> pa.Array:
    """Validate target values and return its ``urdf_dof_target`` child array.

    Target data must be a non-null per-frame 26-vector with finite floats.  It
    is deliberately validated without converting the surrounding hands struct
    to Python lists.
    """

    hands = _combine(target_hands)
    _require_no_nulls(hands, "target hands")
    names = _field_names(hands.type)
    if TARGET_FIELD not in names:
        raise MigrationError(f"target hands has no {TARGET_FIELD}")
    struct = hands.flatten()
    target = struct.field(TARGET_FIELD)
    _require_no_nulls(target, TARGET_FIELD)
    _validate_target_type(target.type)
    values = target.values
    _require_no_nulls(values, f"{TARGET_FIELD} frame vectors")
    scalar_values = values.values
    _require_no_nulls(scalar_values, f"{TARGET_FIELD} scalars")
    floats = np.asarray(scalar_values.to_numpy(zero_copy_only=False))
    if not np.isfinite(floats).all():
        raise MigrationError(f"{TARGET_FIELD} contains non-finite values")
    return target


def extend_hands_array(
    final_hands: pa.Array | pa.ChunkedArray,
    target_hands: pa.Array | pa.ChunkedArray,
    plan: SchemaPlan,
) -> pa.Array:
    """Build an extended final hands array after exact row/child validation."""

    final = _combine(final_hands)
    target = _combine(target_hands)
    _require_no_nulls(final, "final hands")
    _require_no_nulls(target, "target hands")
    if _field_names(final.type) != plan.legacy_hand_fields:
        raise MigrationError("final hands schema differs from preflight plan")
    if _field_names(target.type) != (*plan.legacy_hand_fields, TARGET_FIELD):
        raise MigrationError("target hands schema differs from preflight plan")
    if not np.array_equal(_list_offsets(final), _list_offsets(target)):
        raise MigrationError("final and target hands have different per-row hand counts")

    final_struct = final.flatten()
    target_struct = target.flatten()
    for name in plan.legacy_hand_fields:
        if not final_struct.field(name).equals(target_struct.field(name)):
            raise MigrationError(f"final and target values differ in hands.{name}")
    target_child = validate_target_array(target)
    child_fields = [final_struct.type.field(name) for name in plan.legacy_hand_fields]
    child_arrays = [final_struct.field(name) for name in plan.legacy_hand_fields]
    child_fields.append(target_struct.type.field(TARGET_FIELD))
    child_arrays.append(target_child)
    extended_struct = pa.StructArray.from_arrays(child_arrays, fields=child_fields)
    return pa.ListArray.from_arrays(final.offsets, extended_struct)


def build_swap_schema(schema: pa.Schema) -> pa.Schema:
    """Return schema that promotes staged hands while retaining old hands.

    This is a metadata-only name swap used after the staged top-level column
    has been fully validated.
    """

    names = tuple(schema.names)
    if STAGED_HANDS_COLUMN not in names or "hands" not in names:
        raise MigrationError("swap requires both hands and hands_with_target")
    if BACKUP_HANDS_COLUMN in names:
        raise MigrationError(f"temporary column {BACKUP_HANDS_COLUMN} already exists")
    fields: list[pa.Field] = []
    for field in schema:
        if field.name == "hands":
            fields.append(pa.field(BACKUP_HANDS_COLUMN, field.type, field.nullable, field.metadata))
        elif field.name == STAGED_HANDS_COLUMN:
            fields.append(pa.field("hands", field.type, field.nullable, field.metadata))
        else:
            fields.append(field)
    return pa.schema(fields, metadata=schema.metadata)


def validate_swap_schema(schema: pa.Schema) -> None:
    """Ensure the promoted schema remains reader-safe and target-bearing."""

    names = tuple(schema.names)
    if "hands" not in names or BACKUP_HANDS_COLUMN not in names:
        raise MigrationError("promoted schema must retain hands and hands_without_target")
    if STAGED_HANDS_COLUMN in names:
        raise MigrationError("promoted schema still exposes hands_with_target")
    hand_names = hands_field_names(schema)
    if TARGET_FIELD not in hand_names:
        raise MigrationError("promoted hands is missing urdf_dof_target")


def validate_drop_schema(schema: pa.Schema) -> None:
    """Ensure the final metadata-only cleanup preserved promoted hands."""

    if BACKUP_HANDS_COLUMN in schema.names:
        raise MigrationError("final schema still contains hands_without_target")
    if STAGED_HANDS_COLUMN in schema.names:
        raise MigrationError("final schema still contains hands_with_target")
    if TARGET_FIELD not in hands_field_names(schema):
        raise MigrationError("final hands is missing urdf_dof_target")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", type=Path, required=True, help="image-enriched final Lance dataset")
    parser.add_argument("--target", type=Path, required=True, help="staged raw Lance dataset with target")
    parser.add_argument(
        "--expected-final-version", type=int, required=True,
        help="latest final version observed during approval; mismatches abort",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="perform complete read-only preflight")
    mode.add_argument("--apply", action="store_true", help="permit versioned Lance writes")
    parser.add_argument(
        "--rollback-tag", default="pre_mano_target",
        help="stable tag assigned to the original final version (default: pre_mano_target)",
    )
    parser.add_argument("--batch-rows", type=int, default=64, help="preflight/validation batch row count")
    parser.add_argument("--report", type=Path, help="also write the JSON report to this path")
    args = parser.parse_args(argv)
    if args.expected_final_version < 0:
        parser.error("--expected-final-version must be non-negative")
    if args.batch_rows <= 0:
        parser.error("--batch-rows must be positive")
    if args.final == args.target:
        parser.error("--final and --target must be different datasets")
    return args


def _take(dataset: Any, indices: list[int], columns: list[str]) -> pa.Table:
    # Deliberately use complete nested structs: Lance 8 dot projections can panic.
    return dataset.take(indices, columns=columns)


def _list_value_lengths(array: pa.Array) -> np.ndarray:
    return np.diff(_list_offsets(array))


def _state_prefix_matches(state: pa.Array, urdf_dof: pa.Array) -> bool:
    state_values = state.values
    if not pa.types.is_fixed_size_list(state_values.type) or state_values.type.list_size != 32:
        raise MigrationError(f"final state must be list<fixed_size_list<float>[32]>, got {state.type}")
    dof_values = urdf_dof.values
    if not pa.types.is_fixed_size_list(dof_values.type) or dof_values.type.list_size != 26:
        raise MigrationError(f"hands.urdf_dof must be list<fixed_size_list<float>[26]>, got {urdf_dof.type}")
    q = np.asarray(dof_values.values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(-1, 26)
    s = np.asarray(state_values.values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(-1, 32)
    return bool(np.array_equal(q, s[:, :26]) and np.all(s[:, 26:] == 0))


def _require_training_schema(schema: pa.Schema) -> None:
    """Guard the final artifact type without materializing image/blob values."""

    required = {"image", "wrist_image", "state", "actions", "prompt", "episode_metadata"}
    missing = sorted(required.difference(schema.names))
    if missing:
        raise MigrationError(f"final training schema is missing required columns: {missing}")


def _full_preflight(final: Any, target: Any, batch_rows: int) -> tuple[SchemaPlan, dict[str, Any]]:
    _require_training_schema(final.schema)
    plan = validate_schema_extension(final.schema, target.schema)
    if final.count_rows() != target.count_rows():
        raise MigrationError(
            f"row count mismatch: final={final.count_rows()} target={target.count_rows()}"
        )
    row_count = int(final.count_rows())
    frame_count = 0
    legacy_batch_rows = min(batch_rows, LEGACY_HANDS_BATCH_ROWS)
    for start in range(0, row_count, legacy_batch_rows):
        stop = min(start + legacy_batch_rows, row_count)
        indices = list(range(start, stop))
        final_table = _take(final, indices, ["index", "hands", "state", "episode_metadata"])
        target_table = _take(target, indices, ["index", "hands", "trajectory_metadata"])
        final_index = _combine(final_table["index"])
        target_index = _combine(target_table["index"])
        if not final_index.equals(target_index):
            raise MigrationError(f"index UUID/order mismatch in rows [{start}, {stop})")
        final_hands = _combine(final_table["hands"])
        target_hands = _combine(target_table["hands"])
        extended = extend_hands_array(final_hands, target_hands, plan)
        del extended  # validates all hand children without retaining a second copy
        target_struct = target_hands.flatten()
        urdf_dof = target_struct.field("urdf_dof")
        state = _combine(final_table["state"])
        if not np.array_equal(_list_value_lengths(state), _list_value_lengths(urdf_dof)):
            raise MigrationError(f"final state and target DOF frame lengths differ in rows [{start}, {stop})")
        if not _state_prefix_matches(state, urdf_dof):
            raise MigrationError(f"final state prefix differs from target urdf_dof in rows [{start}, {stop})")
        final_meta = final_table["episode_metadata"].to_pylist()
        target_meta = target_table["trajectory_metadata"].to_pylist()
        lengths = _list_value_lengths(urdf_dof)
        for local, length in enumerate(lengths):
            if int(final_meta[local]["total_frames"]) != int(length):
                raise MigrationError(f"final episode_metadata frame mismatch at row {start + local}")
            if int(target_meta[local]["total_frames"]) != int(length):
                raise MigrationError(f"target trajectory_metadata frame mismatch at row {start + local}")
        frame_count += int(lengths.sum())
    return plan, {"row_count": row_count, "frame_count": frame_count}


def _validate_staged_column(staged: Any, target: Any, batch_rows: int) -> dict[str, Any]:
    """Validate the temporary top-level target-bearing hands column."""

    if STAGED_HANDS_COLUMN not in staged.schema.names:
        raise MigrationError("staged schema is missing hands_with_target")
    row_count = int(staged.count_rows())
    if row_count != int(target.count_rows()):
        raise MigrationError("staged and target row counts differ")
    frame_count = 0
    for start in range(0, row_count, batch_rows):
        stop = min(start + batch_rows, row_count)
        indices = list(range(start, stop))
        staged_table = _take(staged, indices, ["index", STAGED_HANDS_COLUMN])
        target_table = _take(target, indices, ["index", "hands"])
        if not _combine(staged_table["index"]).equals(_combine(target_table["index"])):
            raise MigrationError(f"staged index UUID/order mismatch in rows [{start}, {stop})")
        staged_hands = _combine(staged_table[STAGED_HANDS_COLUMN])
        target_hands = _combine(target_table["hands"])
        if not staged_hands.equals(target_hands):
            raise MigrationError(f"staged hands differs from target in rows [{start}, {stop})")
        frame_count += int(_list_value_lengths(staged_hands.flatten().field(TARGET_FIELD)).sum())
    return {"row_count": row_count, "frame_count": frame_count}


def _validate_final_target(final: Any, target: Any, plan: SchemaPlan, batch_rows: int) -> dict[str, Any]:
    """Validate promoted ``hands`` against staged target without image access."""

    if TARGET_FIELD not in hands_field_names(final.schema):
        raise MigrationError("final hands does not yet contain target")
    row_count = int(final.count_rows())
    frame_count = 0
    for start in range(0, row_count, batch_rows):
        stop = min(start + batch_rows, row_count)
        indices = list(range(start, stop))
        final_table = _take(final, indices, ["index", "hands", "state", "actions", "prompt", "episode_metadata"])
        target_table = _take(target, indices, ["index", "hands", "trajectory_metadata"])
        if not _combine(final_table["index"]).equals(_combine(target_table["index"])):
            raise MigrationError(f"promoted index UUID/order mismatch in rows [{start}, {stop})")
        final_hands = _combine(final_table["hands"])
        target_hands = _combine(target_table["hands"])
        # Compare all target-bearing hands values directly.  Existing training
        # fields are only schema/length checked here, never materialized as blobs.
        if not final_hands.equals(target_hands):
            raise MigrationError(f"promoted hands differs from staged target in rows [{start}, {stop})")
        state = _combine(final_table["state"])
        actions = _combine(final_table["actions"])
        if not np.array_equal(_list_value_lengths(state), _list_value_lengths(actions)):
            raise MigrationError(f"state/action frame lengths differ in rows [{start}, {stop})")
        if any(value is None for value in final_table["prompt"].to_pylist()):
            raise MigrationError(f"prompt has null at rows [{start}, {stop})")
        lengths = _list_value_lengths(state)
        if any(int(meta["total_frames"]) != int(length) for meta, length in zip(final_table["episode_metadata"].to_pylist(), lengths)):
            raise MigrationError(f"promoted episode metadata frame mismatch in rows [{start}, {stop})")
        frame_count += int(lengths.sum())
    return {"row_count": row_count, "frame_count": frame_count}


def _commit_version(base: Any, operation: Any, read_version: int, message: str) -> Any:
    """Commit with a message when supported by the deployed Lance runtime.

    Lance 8 supports ``commit_message``.  Retaining this narrow fallback makes
    the tool integration-testable with older local pylance builds without
    weakening the required optimistic read-version guard.
    """

    try:
        return lance.LanceDataset.commit(
            base, operation, read_version=read_version, commit_message=message
        )
    except TypeError as exc:
        if "commit_message" not in str(exc):
            raise
        return lance.LanceDataset.commit(base, operation, read_version=read_version)


def _fragment_row_counts(dataset: Any) -> list[int]:
    counts: list[int] = []
    for fragment in dataset.get_fragments():
        count = int(fragment.count_rows())
        if count <= 0:
            raise MigrationError(f"fragment {fragment.fragment_id} has no rows")
        counts.append(count)
    if sum(counts) != int(dataset.count_rows()):
        raise MigrationError("fragment row counts do not cover the final dataset")
    return counts


def _merge_target_column(final: Any, target: Any, plan: SchemaPlan, report: dict[str, Any]) -> Any:
    start = 0
    merged_fragments = []
    merged_schema = None
    staged_type = target.schema.field("hands").type
    staged_field = pa.field(STAGED_HANDS_COLUMN, staged_type)
    reader_schema = pa.schema([staged_field])

    for fragment, count in zip(final.get_fragments(), _fragment_row_counts(final)):
        stop = start + count
        indices = list(range(start, stop))
        # `merge_columns` aligns its reader positionally to this fragment.  Read
        # the fragment's index directly, then bind the already-validated target
        # hands in that exact UUID order.  Avoid re-decoding the legacy hands
        # here: Lance 8 cannot reliably read their multi-row list slices.
        fragment_index = fragment.to_table(columns=["index"])
        if fragment_index.num_rows != count:
            raise MigrationError(
                f"fragment {fragment.fragment_id} returned {fragment_index.num_rows} rows, expected {count}"
            )
        target_index = _take(target, indices, ["index"])
        if not _combine(fragment_index["index"]).equals(_combine(target_index["index"])):
            raise MigrationError(f"fragment {fragment.fragment_id} UUID/order mismatch in rows [{start}, {stop})")

        def target_batches():
            for batch_start in range(start, stop, TARGET_READER_BATCH_ROWS):
                batch_stop = min(batch_start + TARGET_READER_BATCH_ROWS, stop)
                target_table = _take(target, list(range(batch_start, batch_stop)), ["hands"])
                hands = _combine(target_table["hands"])
                if hands.type != staged_type:
                    raise MigrationError("target hands type changed while staging fragments")
                yield pa.RecordBatch.from_arrays([hands], schema=reader_schema)

        reader = pa.RecordBatchReader.from_batches(reader_schema, target_batches())
        new_fragment, fragment_schema = fragment.merge_columns(
            reader, reader_schema=reader_schema
        )
        merged_fragments.append(new_fragment)
        if merged_schema is None:
            merged_schema = fragment_schema
        elif merged_schema.to_pyarrow() != fragment_schema.to_pyarrow():
            raise MigrationError("fragments produced inconsistent merged schemas")
        report["fragments"].append(
            {"fragment_id": int(fragment.fragment_id), "row_start": start, "row_stop": stop}
        )
        start = stop
    if merged_schema is None:
        raise MigrationError("final dataset has no fragments")
    operation = lance.LanceOperation.Merge(merged_fragments, merged_schema)
    return _commit_version(
        final, operation, int(final.version), "stage MANO urdf_dof_target hands column"
    )


def _tag_original(final: Any, tag: str) -> None:
    tags = final.tags.list()
    if tag in tags:
        existing = int(tags[tag]["version"])
        if existing != int(final.version):
            raise MigrationError(f"rollback tag {tag!r} already points to version {existing}")
        return
    final.tags.create(tag, int(final.version))


def _validate_training_columns_unchanged(final: Any, rollback: Any, batch_rows: int) -> None:
    """Prove state/action/prompt/metadata stayed identical to the rollback version.

    Images are intentionally not read: the migration only adds/reprojects a
    hands column, while this helper verifies their required columns remain in
    the schema via ``_require_training_schema``.
    """

    _require_training_schema(final.schema)
    _require_training_schema(rollback.schema)
    row_count = int(final.count_rows())
    if row_count != int(rollback.count_rows()):
        raise MigrationError("rollback and final row counts differ")
    columns = ["index", "state", "actions", "prompt", "episode_metadata"]
    for start in range(0, row_count, batch_rows):
        stop = min(start + batch_rows, row_count)
        indices = list(range(start, stop))
        latest = _take(final, indices, columns)
        original = _take(rollback, indices, columns)
        for name in columns:
            if not _combine(latest[name]).equals(_combine(original[name])):
                raise MigrationError(f"training column {name} changed in rows [{start}, {stop})")


def _plan_from_target(target: Any) -> SchemaPlan:
    names = hands_field_names(target.schema)
    if not names or names[-1] != TARGET_FIELD or names.count(TARGET_FIELD) != 1:
        raise MigrationError(f"target hands must append exactly one {TARGET_FIELD} child")
    target_type = target.schema.field("hands").type.value_type.field(TARGET_FIELD).type
    _validate_target_type(target_type)
    return SchemaPlan(tuple(names[:-1]), target_type)


def _migration_phase(schema: pa.Schema) -> str:
    """Classify every reader-safe checkpoint so interrupted runs can resume."""

    names = tuple(schema.names)
    hand_has_target = TARGET_FIELD in hands_field_names(schema)
    has_staged = STAGED_HANDS_COLUMN in names
    has_backup = BACKUP_HANDS_COLUMN in names
    if not hand_has_target and not has_staged and not has_backup:
        return "initial"
    if not hand_has_target and has_staged and not has_backup:
        return "staged"
    if hand_has_target and not has_staged and has_backup:
        return "promoted"
    if hand_has_target and not has_staged and not has_backup:
        return "complete"
    raise MigrationError(
        "unrecognized migration checkpoint: "
        f"hands_has_target={hand_has_target}, staged={has_staged}, backup={has_backup}"
    )


def _rollback_version_from_tag(dataset: Any, tag: str) -> int:
    tags = dataset.tags.list()
    if tag not in tags:
        raise MigrationError(f"rollback tag {tag!r} is missing")
    return int(tags[tag]["version"])


def _validate_promoted_backup(
    promoted: Any, target: Any, plan: SchemaPlan, batch_rows: int
) -> dict[str, Any]:
    """Prove both sides of the name swap before dropping the old hands."""

    row_count = int(promoted.count_rows())
    frame_count = 0
    legacy_batch_rows = min(batch_rows, LEGACY_HANDS_BATCH_ROWS)
    for start in range(0, row_count, legacy_batch_rows):
        stop = min(start + legacy_batch_rows, row_count)
        indices = list(range(start, stop))
        promoted_table = _take(
            promoted, indices, ["index", "hands", BACKUP_HANDS_COLUMN]
        )
        target_table = _take(target, indices, ["index", "hands"])
        if not _combine(promoted_table["index"]).equals(_combine(target_table["index"])):
            raise MigrationError(f"promoted index mismatch in rows [{start}, {stop})")
        current = _combine(promoted_table["hands"])
        target_hands = _combine(target_table["hands"])
        if not current.equals(target_hands):
            raise MigrationError(f"promoted hands differs from target in rows [{start}, {stop})")
        backup = _combine(promoted_table[BACKUP_HANDS_COLUMN])
        rebuilt = extend_hands_array(backup, target_hands, plan)
        if not rebuilt.equals(target_hands):
            raise MigrationError(f"backup/name-swap mapping is invalid in rows [{start}, {stop})")
        frame_count += int(
            _list_value_lengths(target_hands.flatten().field(TARGET_FIELD)).sum()
        )
    return {"row_count": row_count, "frame_count": frame_count}


def _restore_after_failed_promotion(
    current: Any, restore_version: int, cause: BaseException
) -> MigrationError:
    """Restore a known reader-safe version if post-Project validation fails."""

    try:
        restored = _commit_version(
            current,
            lance.LanceOperation.Restore(int(restore_version)),
            int(current.version),
            "restore reader-safe MANO hands after failed promotion",
        )
    except BaseException as restore_exc:
        return MigrationError(
            "promotion validation failed and automatic restore also failed: "
            f"promotion={type(cause).__name__}: {cause}; "
            f"restore={type(restore_exc).__name__}: {restore_exc}"
        )
    return MigrationError(
        "promotion validation failed; latest was restored from "
        f"version {restore_version} as version {restored.version}: "
        f"{type(cause).__name__}: {cause}"
    )


def _write_report(report: dict[str, Any], path: Path | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    _require_lance()
    report: dict[str, Any] = {
        "final": str(args.final),
        "target": str(args.target),
        "mode": "apply" if args.apply else "dry-run",
        "expected_final_version": args.expected_final_version,
        "rollback_tag": args.rollback_tag,
        "status": "started",
        "fragments": [],
        "writes": [],
    }
    if not args.final.exists() or not args.target.exists():
        raise MigrationError("--final and --target must both exist")
    final = lance.dataset(str(args.final))
    target = lance.dataset(str(args.target))
    report["observed_final_version"] = int(final.version)
    report["target_version"] = int(target.version)
    if int(final.version) != args.expected_final_version:
        raise MigrationError(
            f"final latest version is {final.version}, expected {args.expected_final_version}"
        )

    phase = _migration_phase(final.schema)
    plan = _plan_from_target(target)
    report["observed_phase"] = phase
    report["legacy_hand_fields"] = list(plan.legacy_hand_fields)

    if phase == "complete":
        postflight = _validate_final_target(final, target, plan, args.batch_rows)
        rollback_version = _rollback_version_from_tag(final, args.rollback_tag)
        rollback = lance.dataset(str(args.final), version=rollback_version)
        _validate_training_columns_unchanged(final, rollback, args.batch_rows)
        report.update(
            {
                "status": "already-complete",
                "final_version": int(final.version),
                "rollback_version": rollback_version,
                "postflight": postflight,
            }
        )
        return EXIT_OK, report

    if phase in {"initial", "staged"}:
        checked_plan, preflight = _full_preflight(final, target, args.batch_rows)
        if checked_plan != plan:
            raise MigrationError("preflight and target-derived schema plans differ")
        report["preflight"] = preflight
    elif phase == "promoted":
        validate_swap_schema(final.schema)
        report["promoted_validation"] = _validate_promoted_backup(
            final, target, plan, args.batch_rows
        )

    if args.dry_run:
        report.update({"status": "dry-run-ok", "resume_from": phase})
        return EXIT_OK, report

    if phase == "initial":
        _tag_original(final, args.rollback_tag)
        report["writes"].append("tag")
        staged = _merge_target_column(final, target, plan, report)
        report["writes"].append("merge")
        if STAGED_HANDS_COLUMN not in staged.schema.names or "hands" not in staged.schema.names:
            raise MigrationError("merged version is missing a reader-safe hands/staged-hands pair")
        report["staged_version"] = int(staged.version)
    elif phase == "staged":
        _rollback_version_from_tag(final, args.rollback_tag)
        staged = final
        report["staged_version"] = int(staged.version)
    else:
        staged = None

    if phase in {"initial", "staged"}:
        report["staged_validation"] = _validate_staged_column(
            staged, target, args.batch_rows
        )
        promoted_schema = build_swap_schema(staged.schema)
        projected = _commit_version(
            staged,
            lance.LanceOperation.Project(LanceSchema.from_pyarrow(promoted_schema)),
            int(staged.version),
            "promote MANO target-bearing hands",
        )
        report["writes"].append("project")
        report["promoted_version"] = int(projected.version)
        try:
            validate_swap_schema(projected.schema)
            report["promoted_validation"] = _validate_promoted_backup(
                projected, target, plan, args.batch_rows
            )
        except BaseException as exc:
            raise _restore_after_failed_promotion(projected, int(staged.version), exc) from exc
    else:
        projected = final
        _rollback_version_from_tag(projected, args.rollback_tag)

    # Lance's high-level drop is itself an optimistic metadata commit against
    # this pinned dataset object.  If another writer wins, it fails loudly and
    # the promoted checkpoint remains a valid, resumable latest version.
    projected.drop_columns([BACKUP_HANDS_COLUMN])
    report["writes"].append("drop_columns")
    completed = lance.dataset(str(args.final))
    validate_drop_schema(completed.schema)
    postflight = _validate_final_target(completed, target, plan, args.batch_rows)
    rollback_version = _rollback_version_from_tag(completed, args.rollback_tag)
    rollback = lance.dataset(str(args.final), version=rollback_version)
    if TARGET_FIELD in hands_field_names(rollback.schema):
        raise MigrationError("rollback version unexpectedly contains urdf_dof_target")
    _validate_training_columns_unchanged(completed, rollback, args.batch_rows)
    report.update(
        {
            "status": "applied",
            "final_version": int(completed.version),
            "rollback_version": int(rollback.version),
            "postflight": postflight,
        }
    )
    return EXIT_OK, report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        code, report = run(args)
    except MigrationError as exc:
        report = {"status": "guard-failed", "error": str(exc)}
        _write_report(report, args.report)
        return EXIT_GUARD
    except Exception as exc:  # Preserve JSON evidence for unexpected Lance/runtime failures.
        report = {"status": "runtime-failed", "error_type": type(exc).__name__, "error": str(exc)}
        _write_report(report, args.report)
        return EXIT_RUNTIME
    _write_report(report, args.report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
