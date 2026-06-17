#!/usr/bin/env python3
"""Type check gate: hard-fail on dangerous errors, signal on the rest.

Usage:
    pyright --project pyrightconfig.json --outputjson \
        | python scripts/tools/type_check_gate.py --mode import   # blocking
        | python scripts/tools/type_check_gate.py --mode signal   # non-blocking

--mode import (hard gate):
    Fails on errors that indicate runtime bugs:
    - reportMissingImports: broken imports
    - reportUndefinedVariable: references to non-existent symbols
    - reportUnboundVariable: variables used before assignment
    - reportOptionalMemberAccess: .attr on potentially None (AttributeError)
    - reportOptionalSubscript: [x] on potentially None (TypeError)
    - reportOptionalCall: calling potentially None (TypeError)
    - reportOptionalOperand: operator on potentially None (TypeError)

--mode signal (informational):
    Prints a summary of all errors by rule and by area.
    Exits 0 regardless of error count. Used to track progress on develop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter


# Rules that indicate runtime bugs — broken imports, undefined variables,
# and Optional/None access without guards.
HARD_FAIL_RULES = frozenset({
    "reportMissingImports",
    "reportUndefinedVariable",
    "reportUnboundVariable",
    "reportOptionalMemberAccess",
    "reportOptionalSubscript",
    "reportOptionalCall",
    "reportOptionalOperand",
})


def _short_path(f: str) -> str:
    cwd = os.getcwd()
    if f.startswith(cwd + "/"):
        return f[len(cwd) + 1:]
    return f


def _group_dir(f: str) -> str:
    short = _short_path(f)
    parts = short.split("/")
    if len(parts) >= 3 and parts[0] in ("mint_server", "scripts"):
        return "/".join(parts[:2])
    if len(parts) >= 2 and parts[0] == "tests":
        return "tests"
    if len(parts) >= 1:
        return parts[0]
    return short


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["import", "signal"],
        required=True,
        help="import = hard gate (exit 1 on dangerous errors); signal = print summary, exit 0",
    )
    args = parser.parse_args()

    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        print("ERROR: Could not parse pyright JSON output from stdin", file=sys.stderr)
        return 1

    diags = data.get("generalDiagnostics", [])
    summary = data.get("summary", {})
    error_count = summary.get("errorCount", 0)

    if args.mode == "import":
        hard_errors = [d for d in diags if d.get("rule") in HARD_FAIL_RULES]
        if hard_errors:
            print(f"\n❌ Import gate failed: {len(hard_errors)} hard errors\n")
            for d in hard_errors[:50]:
                f = _short_path(d.get("file", "?"))
                line = d.get("range", {}).get("start", {}).get("line", 0) + 1
                col = d.get("range", {}).get("start", {}).get("character", 0) + 1
                rule = d.get("rule", "?")
                msg = d.get("message", "")
                print(f"  {f}:{line}:{col} [{rule}] {msg}")
            if len(hard_errors) > 50:
                print(f"  ... and {len(hard_errors) - 50} more")
            return 1
        else:
            print(f"\n✅ Import gate passed (0 hard errors, {error_count} total type errors in signal mode)")
            return 0

    # signal mode
    by_rule = Counter(d.get("rule", "?") for d in diags)
    by_dir = Counter()
    for d in diags:
        by_dir[_group_dir(d.get("file", ""))] += 1

    print(f"\n📊 Type check signal: {error_count} errors across {summary.get('filesAnalyzed', 0)} files\n")

    print("By rule:")
    for rule, count in by_rule.most_common(15):
        print(f"  {count:>5}  {rule}")

    print("\nBy area:")
    for d, count in by_dir.most_common(15):
        print(f"  {count:>5}  {d}")

    print(f"\nℹ️  Signal mode: exiting 0 ({error_count} errors to fix incrementally)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
