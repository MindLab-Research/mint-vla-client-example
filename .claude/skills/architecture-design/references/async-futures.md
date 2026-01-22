# Async futures (Tinker polling protocol)

Many endpoints return `{"request_id": "<uuid>"}` and do the work in the background (see `tinker_server/backend/future_store.py`).

`POST /api/v1/retrieve_future` behavior (`tinker_server/routes/futures.py`):
- HTTP 408: pending
- HTTP 200 with `{"error": ...}`: failed
- HTTP 200 with result payload: done
- HTTP 404: unknown `request_id` (FutureStore does not have it)

This contract is assumed by the SDK. Preserve it when adding endpoints that need background execution.

Note: retrieving a failed/done future deletes it from `FutureStore` (`future_store.cleanup(request_id)`), so `request_id` is single-use. A second `retrieve_future` for the same `request_id` returns HTTP 404.

## Why futures exist in Mint

Most work runs on Ray GPU actors and can exceed typical HTTP request lifetimes. The futures protocol keeps the HTTP surface stable and matches the Tinker client contract. Do not silently change status codes (for example, 408 to 202) or switch to streaming without updating the client contract.

## Failure mode: API restart loses futures

`FutureStore` is in-process. After an API server restart, previously issued `request_id` values are not retrievable and `retrieve_future` returns HTTP 404.
