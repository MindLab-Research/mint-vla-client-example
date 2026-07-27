from __future__ import annotations

import math

import pyarrow as pa
import pytest

from scripts.tools import migrate_mano_target_column as migration


def _hand_types() -> tuple[pa.DataType, pa.DataType]:
    dof = pa.list_(pa.float32(), 26)
    legacy = pa.list_(pa.struct([pa.field("urdf_dof", pa.list_(dof))]))
    target = pa.list_(
        pa.struct(
            [
                pa.field("urdf_dof", pa.list_(dof)),
                pa.field("urdf_dof_target", pa.list_(dof)),
            ]
        )
    )
    return legacy, target


def _hands_arrays(target_values: list[float] | None = None) -> tuple[pa.Array, pa.Array, migration.SchemaPlan]:
    legacy_type, target_type = _hand_types()
    target_values = [0.25] * 26 if target_values is None else target_values
    final = pa.array(
        [[{"urdf_dof": [[0.0] * 26, [0.5] * 26]}]], type=legacy_type
    )
    target = pa.array(
        [[{
            "urdf_dof": [[0.0] * 26, [0.5] * 26],
            "urdf_dof_target": [target_values, target_values],
        }]],
        type=target_type,
    )
    plan = migration.validate_schema_extension(
        pa.schema([pa.field("hands", legacy_type)]),
        pa.schema([pa.field("hands", target_type)]),
    )
    return final, target, plan


def test_schema_extension_requires_only_target_child() -> None:
    legacy, target = _hand_types()
    plan = migration.validate_schema_extension(
        pa.schema([pa.field("hands", legacy), pa.field("image", pa.binary())]),
        pa.schema([pa.field("hands", target)]),
    )

    assert plan.legacy_hand_fields == ("urdf_dof",)
    assert plan.target_hand_type == target.value_type.field("urdf_dof_target").type

    extra = pa.list_(
        pa.struct(
            [
                pa.field("urdf_dof", pa.list_(pa.list_(pa.float32(), 26))),
                pa.field("unexpected", pa.int32()),
                pa.field("urdf_dof_target", pa.list_(pa.list_(pa.float32(), 26))),
            ]
        )
    )
    with pytest.raises(migration.MigrationError, match="plus exactly"):
        migration.validate_schema_extension(
            pa.schema([pa.field("hands", legacy)]), pa.schema([pa.field("hands", extra)])
        )


def test_extend_hands_preserves_legacy_values_and_adds_target() -> None:
    final, target, plan = _hands_arrays()

    extended = migration.extend_hands_array(final, target, plan)

    assert extended.type == target.type
    assert extended.equals(target)
    assert migration.validate_target_array(target).equals(target.flatten().field("urdf_dof_target"))


def test_target_validation_rejects_nonfinite_null_and_non_float32_values() -> None:
    _, nonfinite, _ = _hands_arrays([math.inf] * 26)
    with pytest.raises(migration.MigrationError, match="non-finite"):
        migration.validate_target_array(nonfinite)

    float64_target = pa.list_(
        pa.struct(
            [
                pa.field("urdf_dof", pa.list_(pa.list_(pa.float32(), 26))),
                pa.field("urdf_dof_target", pa.list_(pa.list_(pa.float64(), 26))),
            ]
        )
    )
    with pytest.raises(migration.MigrationError, match="float32"):
        migration.validate_schema_extension(
            pa.schema([pa.field("hands", _hand_types()[0])]),
            pa.schema([pa.field("hands", float64_target)]),
        )

    _, target_type = _hand_types()
    null_target = pa.array(
        [[{"urdf_dof": [[0.0] * 26], "urdf_dof_target": None}]], type=target_type
    )
    with pytest.raises(migration.MigrationError, match="null"):
        migration.validate_target_array(null_target)


def test_extend_hands_rejects_misaligned_legacy_values() -> None:
    final, target, plan = _hands_arrays()
    target_rows = target.to_pylist()
    target_rows[0][0]["urdf_dof"][0][0] = 9.0
    changed = pa.array(target_rows, type=target.type)

    with pytest.raises(migration.MigrationError, match="hands.urdf_dof"):
        migration.extend_hands_array(final, changed, plan)


def test_swap_schema_keeps_valid_hands_and_backup() -> None:
    legacy, target = _hand_types()
    schema = pa.schema(
        [
            pa.field("index", pa.struct([pa.field("uuid", pa.string())])),
            pa.field("hands", legacy),
            pa.field("image", pa.binary()),
            pa.field(migration.STAGED_HANDS_COLUMN, target),
        ]
    )

    swapped = migration.build_swap_schema(schema)

    assert swapped.names == ["index", migration.BACKUP_HANDS_COLUMN, "image", "hands"]
    assert migration.hands_field_names(swapped) == ("urdf_dof", "urdf_dof_target")
    migration.validate_swap_schema(swapped)
    with pytest.raises(migration.MigrationError, match="still contains"):
        migration.validate_drop_schema(swapped)


def test_migration_phase_recognizes_resumable_reader_safe_states() -> None:
    legacy, target = _hand_types()
    initial = pa.schema([pa.field("hands", legacy)])
    staged = pa.schema(
        [pa.field("hands", legacy), pa.field(migration.STAGED_HANDS_COLUMN, target)]
    )
    promoted = pa.schema(
        [pa.field(migration.BACKUP_HANDS_COLUMN, legacy), pa.field("hands", target)]
    )
    complete = pa.schema([pa.field("hands", target)])

    assert migration._migration_phase(initial) == "initial"
    assert migration._migration_phase(staged) == "staged"
    assert migration._migration_phase(promoted) == "promoted"
    assert migration._migration_phase(complete) == "complete"


def test_argument_guards_require_explicit_mode_and_valid_version() -> None:
    with pytest.raises(SystemExit):
        migration.parse_args([
            "--final", "/tmp/final.lance", "--target", "/tmp/target.lance",
            "--expected-final-version", "17",
        ])
    with pytest.raises(SystemExit):
        migration.parse_args([
            "--final", "/tmp/final.lance", "--target", "/tmp/target.lance",
            "--expected-final-version", "-1", "--dry-run",
        ])
    with pytest.raises(SystemExit):
        migration.parse_args([
            "--final", "/tmp/same.lance", "--target", "/tmp/same.lance",
            "--expected-final-version", "17", "--apply",
        ])

    args = migration.parse_args([
        "--final", "/tmp/final.lance", "--target", "/tmp/target.lance",
        "--expected-final-version", "17", "--dry-run",
    ])
    assert args.dry_run and not args.apply
    assert args.rollback_tag == "pre_mano_target"
