# Checkpoint compatibility matrix (Tinker vs MinT)

This document describes deterministic behavior for the `download -> upload -> resume` loop.

## Checkpoint classes

- `training` (Tinker path segment: `weights/`)
  - Intended for exact training resume.
  - Must include optimizer artifacts.
- `sampler` (Tinker path segment: `sampler_weights/`)
  - Intended for inference/sampling only.
  - Must not include optimizer artifacts.

Canonical internal checkpoint paths:

- `mint://{training_run_id}/weights/{checkpoint_name}`
- `mint://{training_run_id}/sampler_weights/{checkpoint_name}`

External `tinker://...` payloads remain accepted for Tinker SDK compatibility and
are rewritten to `mint://...` by the API compatibility middleware.

Canonical checkpoint IDs (used by RestClient APIs):

- `weights/{checkpoint_name}`
- `sampler_weights/{checkpoint_name}`

## Matrix

| Checkpoint class | Download archive | Upload archive (`/api/v1/checkpoints/upload`) | Resume w/ optimizer (`optimizer=true`) | Resume w/o optimizer (`optimizer=false`) |
|---|---|---|---|---|
| `training` | Supported | Supported (requires optimizer artifacts) | Supported | Supported |
| `sampler` | Supported | Supported (requires no optimizer artifacts) | Rejected (4xx) | Supported |

Notes:

- Download protocol:
  - Tinker SDK expects `302` with `Location` (a direct-download URL) for `.../archive`.
  - Manual/browser clients can use the direct stream URL (`?direct=1`) or rely on the default streaming behavior when not using the Tinker SDK user-agent.
- Upload contract:
  - If an extracted `metadata.json` declares `checkpoint_type`, it must match the artifacts (optimizer present vs absent), otherwise upload fails with `400`.
  - If `metadata.json` does not declare a type, MinT infers `training` if optimizer artifacts exist, otherwise `sampler`.
