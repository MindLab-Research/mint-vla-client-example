---
name: mintx
description: |
  Mint-only API extension design and implementation for interfaces that must stay separate from the Tinker-compatible surface.

  Use when adding or updating Mint-specific interfaces under `/api/v1/mint` on the server or under `mint.mint` on the client, especially for new training primitives, checkpoint utilities, or data structures that cannot be expressed by standard Tinker types.

  Hard rule: always keep this skill, the code, and the Mint-only API docs synchronized in the same change.
---

# MintX

MintX is the extension surface for features that must not change Tinker-compatible semantics.

## Principles

- Keep Tinker compatibility intact.
- Put server-only extensions under `/api/v1/mint`.
- Put client-only extensions under `mint.mint`.
- Do not add fields to Tinker data structures or change behavior of Tinker-compatible endpoints.
- Keep server primitives parsimonious and generic.
- Keep orchestration, training algorithms, and policy schedules on the client unless the server-side primitive is genuinely nontrivial.
- Prefer immutable checkpoint-producing operations over in-place mutation.
- When code changes, update `references/api.md` in the same change. Do not leave docs stale.

## Workflow

1. Confirm the feature cannot be expressed cleanly through the existing Tinker-compatible API.
2. Design the smallest Mint-only primitive that preserves Tinker compatibility.
3. Implement the server endpoint under `/api/v1/mint`.
4. Implement the client request/response helpers under `mint.mint`.
5. Update `references/api.md` so the documented request and response shapes match the code exactly.
6. Validate with local tests first, then with an integrated dev-server run.

## Current MintX APIs

Read `references/api.md` before editing Mint-only interfaces.

## Notes for SDPO and Distillation

- Keep EMA orchestration on the client.
- Prefer immutable checkpoint interpolation over server-owned EMA policy.
- Keep reverse-KL as a Mint-only primitive when it requires paired inputs or non-Tinker request shapes.
- Do not overload `mint.Datum`; define Mint-only data classes instead.
