#!/opt/venv/bin/python3
from __future__ import annotations

import importlib.util
import os
import runpy
import sys
from pathlib import Path


def _load_repo_sitecustomize() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sitecustomize_path = repo_root / "sitecustomize.py"
    spec = importlib.util.spec_from_file_location("_mint_repo_sitecustomize", sitecustomize_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load repo sitecustomize from {sitecustomize_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _run_as_python(argv: list[str]) -> None:
    if not argv:
        raise RuntimeError("vllm worker wrapper requires a target script/module")

    while argv and argv[0].startswith("-") and argv[0] not in ("-m", "-c"):
        flag = argv.pop(0)
        if flag == "-B":
            sys.dont_write_bytecode = True
        elif flag in ("-W", "-X") and argv:
            argv.pop(0)

    if not argv:
        raise RuntimeError("vllm worker wrapper consumed only interpreter flags; no target remains")

    head, *tail = argv
    if head == "-m":
        if not tail:
            raise RuntimeError("python wrapper missing module name after -m")
        module_name, *module_args = tail
        sys.argv = [module_name, *module_args]
        runpy.run_module(module_name, run_name="__main__", alter_sys=True)
        return
    if head == "-c":
        if not tail:
            raise RuntimeError("python wrapper missing code string after -c")
        code, *code_args = tail
        sys.argv = ["-c", *code_args]
        globals_dict = {
            "__name__": "__main__",
            "__file__": "<string>",
            "__package__": None,
            "__cached__": None,
        }
        exec(compile(code, "<string>", "exec"), globals_dict)
        return

    script_path = os.fspath(Path(head).resolve())
    sys.argv = [script_path, *tail]
    runpy.run_path(script_path, run_name="__main__")


def main() -> None:
    _load_repo_sitecustomize()
    _run_as_python(sys.argv[1:])


if __name__ == "__main__":
    main()
