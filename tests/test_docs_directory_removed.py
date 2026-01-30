from __future__ import annotations

from pathlib import Path


def test_docs_directory_removed() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert not (repo_root / "docs").exists()
