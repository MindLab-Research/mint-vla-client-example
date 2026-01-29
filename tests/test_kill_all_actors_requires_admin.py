from pathlib import Path


def test_kill_all_actors_is_admin_guarded():
    repo_root = Path(__file__).resolve().parents[1]
    txt = (repo_root / "tinker_server/routes/service.py").read_text(encoding="utf-8")
    assert "async def kill_all_actors(request: Request)" in txt
    assert "_require_admin(request)" in txt

