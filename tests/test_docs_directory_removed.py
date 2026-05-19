from __future__ import annotations

from pathlib import Path


def test_docs_directory_removed_runtime_notes_live_under_architecture_references() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    docs_dir = repo_root / "docs"
    references_dir = repo_root / ".claude" / "skills" / "architecture-design" / "references"

    assert not docs_dir.exists()
    assert (references_dir / "openai_compatible_sdk.md").exists()
    assert (references_dir / "issue_413_dense_session_state_fix.md").exists()
