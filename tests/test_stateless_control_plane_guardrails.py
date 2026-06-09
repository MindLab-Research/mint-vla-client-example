from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _python_sources() -> list[Path]:
    roots = [REPO_ROOT / "mint_server", REPO_ROOT / "tests"]
    return sorted(path for root in roots for path in root.rglob("*.py"))


def test_initialize_execution_bindings_is_runtime_actor_only() -> None:
    callers: list[str] = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "initialize_execution_bindings":
                    callers.append(str(path.relative_to(REPO_ROOT)))
                elif isinstance(func, ast.Attribute) and func.attr == "initialize_execution_bindings":
                    callers.append(str(path.relative_to(REPO_ROOT)))

    assert callers == ["mint_server/backend/model_runtime_actor.py"]


def test_api_startup_does_not_start_local_manager_cleanup_loops() -> None:
    app_source = (REPO_ROOT / "mint_server" / "app.py").read_text()
    assert ".start_cleanup_task(" not in app_source

    manager_sources = [
        REPO_ROOT / "mint_server" / "backend" / "session_manager.py",
        REPO_ROOT / "mint_server" / "backend" / "training_session_manager.py",
    ]
    for path in manager_sources:
        source = path.read_text()
        assert "def start_cleanup_task" not in source
        assert "asyncio.create_task(self._cleanup_loop())" not in source
