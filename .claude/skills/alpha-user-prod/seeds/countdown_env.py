"""
Countdown environment seed for MinT alpha-user demos.

Goal: capture the core environment semantics (prompt + validity checks) without
vendoring the entire countdown cookbook.

This module is intentionally small and self-contained:
- No HuggingFace datasets dependency.
- A small built-in task list is provided to bootstrap experiments.

The alpha-user agent can extend this module or generate new tasks, but should
preserve the validation semantics: "use each number exactly once" and evaluate
the arithmetic expression safely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CountdownTask:
    numbers: list[int]
    target: int


COUNTDOWN_TASKS: list[CountdownTask] = [
    CountdownTask(numbers=[1, 2, 3, 4], target=4),
    CountdownTask(numbers=[2, 3, 7, 8], target=24),
    CountdownTask(numbers=[1, 5, 6, 7], target=21),
    CountdownTask(numbers=[3, 3, 8, 8], target=24),
    CountdownTask(numbers=[2, 4, 6, 9], target=36),
]


def build_question(*, numbers: list[int], target: int) -> str:
    return (
        f"Using the numbers {numbers}, create an equation that equals {target}.\n"
        "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.\n"
        "Show your work in <think> </think> tags. Return the final equation and answer in <answer> </answer> tags,\n"
        "for example: <answer> (1 + 2) / 3 * 4 = 4 </answer>."
    )


def has_answer_tags(s: str) -> bool:
    return re.search(r"<answer>(.*?)</answer>", s, re.DOTALL) is not None


def _extract_equation(s: str) -> str | None:
    m = re.search(r"<answer>(.*?)</answer>", s, re.DOTALL)
    if m is None:
        return None
    eq = m.group(1).strip()
    if "=" in eq:
        eq = eq.split("=")[0].strip()
    return eq


def _extract_used_numbers(equation: str) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", equation)]


def _is_equation_charset_safe(equation: str) -> bool:
    return re.match(r"^[\d+\-*/().\s]+$", equation) is not None


def check_solution(*, response: str, numbers: list[int], target: int) -> bool:
    """
    Validate a response using Countdown rules:
    - Must contain <answer>...</answer>
    - Equation must use exactly the provided numbers (multiset match)
    - Allowed characters only
    - Equation evaluates to target
    """
    eq = _extract_equation(response)
    if eq is None:
        return False

    used = _extract_used_numbers(eq)
    if sorted(used) != sorted(numbers):
        return False

    if not _is_equation_charset_safe(eq):
        return False

    try:
        # Constrain eval: no builtins, no names.
        result = eval(eq, {"__builtins__": None}, {})
        return abs(float(result) - float(target)) < 1e-5
    except Exception:
        return False


def iter_bootstrap_tasks() -> Iterable[CountdownTask]:
    return list(COUNTDOWN_TASKS)

