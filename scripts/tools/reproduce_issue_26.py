import ast
import sys
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


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]

    megatron_distributed = repo_root / "tinker_server/backend/megatron_distributed.py"
    distributed_txt = megatron_distributed.read_text(encoding="utf-8")
    if "MegatronActorPool" in distributed_txt:
        print(f"FAIL: {megatron_distributed} still contains MegatronActorPool", file=sys.stderr)
        return 1
    if "MegatronActorEntry" in distributed_txt:
        print(f"FAIL: {megatron_distributed} still contains MegatronActorEntry", file=sys.stderr)
        return 1
    if "get_megatron_actor_pool" in distributed_txt or "_megatron_actor_pool" in distributed_txt:
        print(f"FAIL: {megatron_distributed} still contains MegatronActorPool globals", file=sys.stderr)
        return 1

    module_doc = _get_module_docstring(megatron_distributed)
    if "MegatronTrainingWorker" in module_doc:
        print(
            "FAIL: megatron_distributed module docstring still mentions MegatronTrainingWorker",
            file=sys.stderr,
        )
        return 1

    verl_training = repo_root / "tinker_server/backend/verl_training.py"
    doc = _get_class_method_docstring(
        verl_training,
        class_name="VerlTrainingEngine",
        method_name="create_training_session",
    )
    if doc is None:
        print("FAIL: could not find VerlTrainingEngine.create_training_session", file=sys.stderr)
        return 1
    if "MegatronTrainingWorker" in doc:
        print(
            "FAIL: create_training_session docstring still mentions MegatronTrainingWorker",
            file=sys.stderr,
        )
        return 1
    if "MegatronWorkerGroup" not in doc:
        print(
            "FAIL: create_training_session docstring does not mention MegatronWorkerGroup",
            file=sys.stderr,
        )
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
