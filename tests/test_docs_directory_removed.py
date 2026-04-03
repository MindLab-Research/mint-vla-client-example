from __future__ import annotations

from pathlib import Path


def test_docs_directory_contains_runtime_notes() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    docs_dir = repo_root / "docs"

    assert docs_dir.exists()
    assert (docs_dir / "openai_compatible_sdk.md").exists()
