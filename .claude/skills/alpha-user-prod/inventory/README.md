# alpha-user-prod inventory (directory)

This directory is the persistent memory of what this skill has tried and built.

Rules:
- Add a new demo as a new subdirectory under `demos/`.
- Each demo directory must include:
  - `INTENT.md`: user story, algorithm/scenario, why it matters, and what MinT primitives it exercises
  - `STATUS.md`: last known status, last run timestamp, pointers to artifact bundles, and next diagnostic step
- Prefer small, composable demos over monoliths.

Auto-maintenance:
- If `demos/` becomes large or repetitive, group by theme (training, inference, RL, preference, agents, long-context).
- Merge near-duplicates by keeping one canonical demo and moving variants into a `variants/` subdir.
- When a demo becomes obsolete, delete its directory. Git history is the archive.
