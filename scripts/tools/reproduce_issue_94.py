import ast
import sys
from pathlib import Path


def _call_forwards_logs_to_driver(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg == "log_to_driver":
            return True
        if kw.arg is not None:
            continue
        if not isinstance(kw.value, ast.Call):
            continue
        func = kw.value.func
        if isinstance(func, ast.Name) and func.id == "ray_log_to_driver_kwargs":
            return True
        if isinstance(func, ast.Attribute) and func.attr == "ray_log_to_driver_kwargs":
            return True
    return False


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    errors: list[str] = []

    for path in sorted((repo_root / "tinker_server").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "init":
                continue
            if not isinstance(node.func.value, ast.Name) or node.func.value.id != "ray":
                continue
            if not _call_forwards_logs_to_driver(node):
                errors.append(f"{path}:{node.lineno}: ray.init missing log-to-driver forwarding")

    if errors:
        print("FAIL:", file=sys.stderr)
        for e in errors:
            print(f"- {e}", file=sys.stderr)
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

