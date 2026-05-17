# Authentication and model access

Auth is enforced in `tinker_server/app.py` middleware for both `/api/v1/*` and `/internal/*` when `INTERNAL_API_TOKEN` is configured. If it is not configured, the server runs in dev pass-through mode and stamps a local admin identity.

## API keys and identities

There is one production authentication method: trusted identity forwarded by the
platform with a matching internal token.

Gateway-forwarded trusted identity:
- `X-MinT-User-Id`
- `X-MinT-User-Role`
- `X-MinT-Apikey-Id`
- `X-MinT-Request-Id`
- `X-Internal-Token`

Trusted identity rules:
- `X-MinT-*` headers are only accepted when `INTERNAL_API_TOKEN` is configured and `X-Internal-Token` matches.
- `X-Request-Id` alone must not trigger trusted-header mode.
- Admin authorization uses trusted capabilities from headers when present, or
  the trusted `user_role == "admin"` role.

Outbound identity header:
- When a validated request identity includes an API key id (`apikey_id` or
  dev/test `key_id`), the server echoes it on responses as `X-MinT-Apikey-Id`.

Privilege boundary:
- `/api/v1/retrieve_future` hides detailed exception text unless the caller is privileged (admin role).

## Model access control

Model access control is centralized in `tinker_server/model_access_control.py` and is applied in routes like `POST /api/v1/create_sampling_session` based on `request.state.user_data`.
