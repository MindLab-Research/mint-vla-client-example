from pathlib import Path


def test_actor_admin_endpoints_are_admin_guarded():
    repo_root = Path(__file__).resolve().parents[1]
    txt = (repo_root / "tinker_server/routes/internal.py").read_text(encoding="utf-8")
    assert "async def list_actors(" in txt
    assert "async def kill_actors(request: Request" in txt
    assert "require_admin(request)" in txt
