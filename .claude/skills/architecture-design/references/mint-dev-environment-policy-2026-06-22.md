# Mint Dev Environment Policy Update - 2026-06-22

## Design

Mint development now uses a dedicated driver node for API server and debugging
work. Project-level agent policy should state the invariants, while the
`mint-dev` skill owns the runnable startup and cleanup procedure.

## Decisions

- Keep the canonical policy in `CLAUDE.md`. `AGENTS.md` is a symlink to
  `CLAUDE.md`, so this also covers generic agents and Codex.
- Keep executable dev-server steps in `.claude/skills/mint-dev/SKILL.md`.
- Do not create a new memory file; the repository has no existing memory-file
  convention and a new entry point would be easy to miss.
- Treat `MINT_CODE_ROOT` as the canonical code-root variable. The current
  launcher and runtime use `MINT_CODE_ROOT`, not `MINT_ROOT_CODE`.
- Treat `MINT_TASK_STATE_STORE_DB_PATH=:memory:` as the task-store in-memory
  dev mode. It is intentionally non-persistent and should not be used for
  evidence that must survive a server restart.
- For dev operations, forbid Ray Client mode and use direct GCS attach from the
  dev driver. Ray Client caused GCS instability in the Mint usage pattern.
- Document `mint-dev-driver` / `192.168.42.106` as the dev driver, and keep Ray
  head usage limited to cluster services managed by `volcano-cluster`.

## Scope

This update changes documentation and runbook policy only. It does not remove
the compatibility code paths that still recognize Ray Client environment
variables.
