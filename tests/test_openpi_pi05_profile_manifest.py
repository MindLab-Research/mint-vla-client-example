from pathlib import Path

from mint_server.backend.openpi.pi05_profiles import (
    PI05_ACTION_LORA_R16_V1,
    PROFILE_MANIFEST_FILENAME,
    validate_profile_manifest,
    write_profile_manifest,
)


def test_profile_manifest_atomic_round_trip(tmp_path: Path) -> None:
    path = write_profile_manifest(tmp_path, PI05_ACTION_LORA_R16_V1)

    assert path == tmp_path / PROFILE_MANIFEST_FILENAME
    assert validate_profile_manifest(tmp_path, PI05_ACTION_LORA_R16_V1) == path
    assert list(tmp_path.glob(f".{PROFILE_MANIFEST_FILENAME}.*.tmp")) == []


def test_profile_manifest_rejects_missing_and_wrong_content(tmp_path: Path) -> None:
    try:
        validate_profile_manifest(tmp_path, PI05_ACTION_LORA_R16_V1)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing profile manifest must be rejected")

    path = write_profile_manifest(tmp_path, PI05_ACTION_LORA_R16_V1)
    path.write_text('{"profile_id":"wrong","manifest_hash":"wrong"}', encoding="utf-8")
    try:
        validate_profile_manifest(tmp_path, PI05_ACTION_LORA_R16_V1)
    except ValueError as exc:
        assert "mismatch" in str(exc)
    else:
        raise AssertionError("wrong profile manifest must be rejected")
