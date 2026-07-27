# Formal client repository migration — 2026-07-27

## Decision

`mint-vla-client-example` is the authoritative Git repository and the supported
GPU-host execution checkout. The previous split between a local canonical Git
repository and a non-Git execution copy is retired.

## Imported sources

- Reviewed client source snapshot: local `vla_mint` `main@4b62bd26df20df8779eb406d511f9db1170d9f22`.
- Execution evidence/source: `/vePFS-Mindverse/user/intern/wenxi/vla_mint-parallel-preprocess`, marker `4b62bd2`.
- The critical Mode4/training files in those two sources were byte-identical
  before import.
- One reviewed remote-only source file was retained:
  `scripts/remote/run_action_lora_server.sh`.

The target repository retained its initial Git commit
`2d9d23bbddad2e0ea0b710398b8d1154743b8c05`. The branch
`backup/pre-client-import-20260727` and tag
`backup-pre-client-import-20260727` point to that pre-import state.

## Deliberate exclusions

The migration did not import:

- `.memory/` or agent transcripts;
- private `config/remote.env` (a new ignored runtime-local copy was created);
- `.tmp*`, `.pytest_cache`, `__pycache__`, `*.pyc`, logs, videos, and results;
- `.vla_mint_commit` and `.isolation_origin` markers;
- ignored `archive/parent-vla-legacy/` content;
- the old rsync-based `scripts/remote/sync_to_server.sh` workflow.

## New source-of-truth workflow

1. Develop and review changes in a clone of
   `git@github.com:MindLab-Research/mint-vla-client-example.git`.
2. Push reviewed commits.
3. In the GPU-host checkout, update with `git pull --ff-only origin main`.
4. Run client training and Mode4 directly from
   `/vePFS-Mindverse/user/intern/wenxi/mint-vla-client-example`.
5. Keep MINT and OpenPI changes in their own repositories and branches.

After validation, the previous non-Git execution copy was retired in place at
`/vePFS-Mindverse/user/intern/wenxi/vla_mint-parallel-preprocess`. Pre-existing
idle shells still hold that working directory, so it is not renamed yet; it is
not a second source of truth or a destination for new work. The previous local
Git repository's full
history was preserved in
`/home/jay/vla/_archive/vla_mint-pre-formal-20260727.bundle`. It remains retired
in place until pre-existing shells using that working directory exit.
