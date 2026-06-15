from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "mint_server" / "backend"

LAYER_BY_PACKAGE = {
    "contracts": 1,
    "core": 2,
    "ray_cluster": 3,
    "stores": 4,
    "scheduling": 5,
    "actors": 6,
    "sessions": 6,
    "inference": 6,
    "training": 6,
    "openpi": 6,
    "ops": 6,
}


def backend_module_for_path(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT).with_suffix("")
    return ".".join(rel.parts)


def package_for_backend_module(module: str) -> str | None:
    prefix = "mint_server.backend."
    if not module.startswith(prefix):
        return None
    remainder = module[len(prefix) :]
    if not remainder:
        return None
    package = remainder.split(".", 1)[0]
    if package not in LAYER_BY_PACKAGE:
        return None
    return package


def resolve_import(current_module: str, node: ast.ImportFrom) -> str | None:
    if node.level <= 0:
        return node.module
    parts = current_module.split(".")
    if node.level > len(parts):
        return None
    base = parts[: -node.level]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def iter_import_modules(path: Path, current_module: str):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            module = resolve_import(current_module, node)
            if module:
                yield node.lineno, module


def main() -> int:
    violations: list[str] = []
    contract_violations: list[str] = []

    for path in sorted(BACKEND_ROOT.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        current_module = backend_module_for_path(path)
        source_package = package_for_backend_module(current_module)
        if source_package is None:
            continue
        source_layer = LAYER_BY_PACKAGE.get(source_package)
        if source_layer is None:
            violations.append(f"{path}: unknown backend package {source_package!r}")
            continue

        for lineno, imported in iter_import_modules(path, current_module):
            target_package = package_for_backend_module(imported)
            if target_package is None:
                continue
            target_layer = LAYER_BY_PACKAGE.get(target_package)
            if target_layer is None:
                violations.append(f"{path}:{lineno}: imports unknown backend package {imported}")
                continue
            if source_package == "contracts" and target_package != "contracts":
                contract_violations.append(f"{path}:{lineno}: contracts imports {imported}")
            if source_layer < target_layer:
                violations.append(
                    f"{path}:{lineno}: layer {source_package}({source_layer}) imports "
                    f"{target_package}({target_layer}): {imported}"
                )

    if contract_violations:
        print("contracts import violations:", file=sys.stderr)
        print("\n".join(contract_violations), file=sys.stderr)
    if violations:
        print("backend layer violations:", file=sys.stderr)
        print("\n".join(violations), file=sys.stderr)
    return 1 if contract_violations or violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
