---
name: tinker-official-reference
description: |
  Official Tinker API reference (types, methods, loss functions, data formats) packaged as a skill.

  Use for: answering questions about the Tinker contract, request/response types, loss functions,
  token/weights formats, and any client compatibility requirements.

  Triggers: "tinker api", "official reference", "types", "loss function", "datum format",
  "target_tokens", "loss_mask", "retrieve_future", "tinker sdk"
---

# Tinker official reference

Primary reference: section files under `references/upstream/` (docs pages and api-reference pages).

When updating routes or `tinker_server/models/types.py`, treat this reference as the source of truth for client-visible semantics.

## Upstream source and updates

Upstream documentation URL (single-file docs):
- `https://tinker-docs.thinkingmachines.ai/llms-full.txt`

Update section files by running:
- `python .claude/skills/tinker-official-reference/scripts/update_reference.py`
