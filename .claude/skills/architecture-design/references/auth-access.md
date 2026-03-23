# Authentication and model access

Auth is enforced in `tinker_server/app.py` middleware for both `/api/v1/*` and `/internal/*` when either `TINKER_API_KEY` or `TINKER_TOKEN_SECRET_KEY` is set. If neither is set, legacy auth is disabled (dev mode), but forged gateway identity headers are still rejected unless internal gateway auth is configured correctly.

## API keys and identities

There are two authentication methods, checked in order:

1. Admin API key (`TINKER_API_KEY`)
   - Treated as a single "admin" key.
   - Compared with constant-time string comparison.
   - On match: request is marked privileged with `request.state.user_data = {"user_id": "admin"}`.

2. User tokens (`sk-mint-...` or legacy `sk-...`, encrypted)
   - Enabled when `TINKER_TOKEN_SECRET_KEY` is set.
   - Client sends an encrypted token that encodes user information.
   - Server decrypts it via `TokenEncryptor` and stores the decoded dict into `request.state.user_data`.

Headers supported:
- `X-API-Key: ...` (preferred)
- `Authorization: Bearer ...`
- `Authorization: sk-mint-...` (direct token)
- `Authorization: sk-...` (legacy direct token)

Gateway-forwarded trusted identity:
- `X-MinT-User-Id`
- `X-MinT-User-Role`
- `X-MinT-Apikey-Id`
- `X-MinT-Request-Id`
- `X-Internal-Token`

Trusted identity rules:
- `X-MinT-*` headers are only accepted when `INTERNAL_API_TOKEN` is configured and `X-Internal-Token` matches.
- `X-Request-Id` alone must not trigger trusted-header mode.
- Admin authorization should use `user_role == "admin"` (with `user_id == "admin"` kept only as legacy fallback).

Outbound identity header:
- When a validated request identity includes an API key id (`apikey_id` or legacy `key_id`), the server echoes it on responses as `X-MinT-Apikey-Id`.

Privilege boundary:
- `/api/v1/retrieve_future` hides detailed exception text unless the caller is privileged (admin role).

## Model access control

Model access control is centralized in `tinker_server/model_access_control.py` and is applied in routes like `POST /api/v1/create_sampling_session` based on `request.state.user_data`.
