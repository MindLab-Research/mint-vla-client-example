import ast
from pathlib import Path


def _has_chunks0_tokens_pattern(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        outer = node
        if not isinstance(outer.slice, ast.Constant) or outer.slice.value != "tokens":
            continue
        inner = outer.value
        if not isinstance(inner, ast.Subscript):
            continue
        if not isinstance(inner.value, ast.Name) or inner.value.id != "chunks":
            continue
        if not isinstance(inner.slice, ast.Constant) or inner.slice.value != 0:
            continue
        return True
    return False


def test_issue_81_megatron_backends_do_not_index_chunks0_tokens() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    targets = [
        repo_root / "tinker_server/backend/megatron_training.py",
        repo_root / "tinker_server/backend/megatron_distributed.py",
    ]

    offenders: list[str] = []
    for path in targets:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _has_chunks0_tokens_pattern(tree):
            offenders.append(str(path))

    assert not offenders, f"chunks[0]['tokens'] still present in: {offenders}"

