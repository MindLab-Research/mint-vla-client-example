# Authentication and model access

Auth is enforced in `tinker_server/app.py` middleware when either `TINKER_API_KEY` or `TINKER_TOKEN_SECRET_KEY` is set. If neither is set, auth is disabled (dev mode).

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

Privilege boundary:
- `/api/v1/retrieve_future` hides detailed exception text unless the caller is privileged (admin).

## Model access control

Model access control is centralized in `tinker_server/model_access_control.py` and is applied in routes like `POST /api/v1/create_sampling_session` based on `request.state.user_data`.
