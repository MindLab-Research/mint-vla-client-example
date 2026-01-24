from pathlib import Path


def test_issue_24_sets_retry_after_header() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "tinker_server/routes/futures.py"
    txt = src.read_text(encoding="utf-8")

    assert 'response.headers["Retry-After"]' in txt
    assert "response.status_code = 408" in txt

