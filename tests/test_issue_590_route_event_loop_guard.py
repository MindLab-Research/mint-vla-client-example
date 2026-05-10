from __future__ import annotations

import tokenize
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

SCAN_PATHS = (
    REPO_ROOT / "tinker_server/routes",
    REPO_ROOT / "tinker_server/health_checks.py",
    REPO_ROOT / "tinker_server/ray_cluster_health.py",
    REPO_ROOT / "tinker_server/ray_gcs_metrics.py",
    REPO_ROOT / "tinker_server/backend/api_work_queue_dispatch.py",
)

BLOCKING_PATTERNS = (
    "run_in_threadpool",
    "asyncio.to_thread",
    "ray.get(",
    "time.sleep(",
    "subprocess.run(",
    "subprocess.Popen(",
    "subprocess.check_output(",
    "init_ray(",
)

EXISTING_ROUTE_BLOCKING_DEBT = Counter(
    {
        ("tinker_server/routes/internal.py", "subprocess.Popen(", "proc = subprocess.Popen("): 1,
        (
            "tinker_server/routes/training.py",
            "asyncio.to_thread",
            "worker = await asyncio.to_thread(ray.get_actor, actor_name, namespace=namespace)",
        ): 1,
        ("tinker_server/routes/training.py", "asyncio.to_thread", "local_metadata = await asyncio.to_thread("): 1,
        ("tinker_server/routes/training.py", "asyncio.to_thread", "return await asyncio.to_thread("): 1,
        ("tinker_server/routes/training.py", "asyncio.to_thread", "tokenizer_metadata = await asyncio.to_thread("): 1,
        ("tinker_server/routes/weights.py", "time.sleep(", "time.sleep(0.25)"): 1,
        ("tinker_server/routes/weights.py", "subprocess.Popen(", "proc = subprocess.Popen("): 1,
    }
)


def test_routes_do_not_add_event_loop_blocking_primitives() -> None:
    violations: list[str] = []
    remaining_debt = EXISTING_ROUTE_BLOCKING_DEBT.copy()

    paths: list[Path] = []
    for scan_path in SCAN_PATHS:
        if scan_path.is_dir():
            paths.extend(sorted(scan_path.glob("*.py")))
        else:
            paths.append(scan_path)

    for path in paths:
        rel_path = path.relative_to(REPO_ROOT).as_posix()
        with path.open("rb") as fh:
            comment_starts = {
                tok.start[0]: tok.start[1] for tok in tokenize.tokenize(fh.readline) if tok.type == tokenize.COMMENT
            }
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            source = line[: comment_starts[lineno]] if lineno in comment_starts else line
            stripped = source.strip()
            for pattern in BLOCKING_PATTERNS:
                if pattern not in stripped:
                    continue
                debt_key = (rel_path, pattern, stripped)
                if remaining_debt[debt_key] > 0:
                    remaining_debt[debt_key] -= 1
                    continue
                violations.append(f"{rel_path}:{lineno}: {stripped}")

    assert violations == []
    assert +remaining_debt == Counter()
