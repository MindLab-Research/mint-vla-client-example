import ast
from pathlib import Path


def _get_module_docstring(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return ast.get_docstring(tree) or ""


def _get_class_method_docstring(path: Path, class_name: str, method_name: str) -> str | None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if isinstance(item, ast.AsyncFunctionDef | ast.FunctionDef) and item.name == method_name:
                return ast.get_docstring(item)
    return None


def test_issue_26_megatron_distributed_no_longer_contains_megatron_actor_pool() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "tinker_server/backend/megatron_distributed.py"
    txt = path.read_text(encoding="utf-8")

    assert "MegatronActorPool" not in txt
    assert "MegatronActorEntry" not in txt
    assert "get_megatron_actor_pool" not in txt
    assert "_megatron_actor_pool" not in txt


def test_issue_26_megatron_docstrings_do_not_reference_deprecated_worker_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    distributed = repo_root / "tinker_server/backend/megatron_distributed.py"
    distributed_doc = _get_module_docstring(distributed)
    assert "MegatronTrainingWorker" not in distributed_doc

    verl = repo_root / "tinker_server/backend/verl_training.py"
    doc = _get_class_method_docstring(verl, "VerlTrainingEngine", "create_training_session")
    assert doc is not None
    assert "MegatronTrainingWorker" not in doc
    assert "MegatronWorkerGroup" in doc
