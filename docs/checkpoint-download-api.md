# Checkpoint API

Internal endpoints for checkpoint management. Application gateway reverse-proxies these to serve the web UI.

## Authentication

All endpoints require `sk-` token. Users can only access their own checkpoints.

---

## List Checkpoints

```
GET /internal/v1/checkpoints
```

Returns all checkpoints belonging to the authenticated user.

### Response

```json
{
  "checkpoints": [
    {
      "checkpoint_id": "ckpt_abc123",
      "model_name": "Qwen/Qwen3-30B-A3B",
      "created_at": "2025-01-04T12:30:00Z",
      "type": "training",
      "size_bytes": 524288000
    },
    {
      "checkpoint_id": "ckpt_def456",
      "model_name": "Qwen/Qwen2.5-7B-Instruct",
      "created_at": "2025-01-03T09:15:00Z",
      "type": "inference",
      "size_bytes": 134217728
    }
  ]
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `checkpoint_id` | string | Unique checkpoint identifier |
| `model_name` | string | Base model name (e.g., `Qwen/Qwen3-30B-A3B`) |
| `created_at` | string | ISO 8601 timestamp |
| `type` | string | `training` (full state + optimizer) or `inference` (weights only) |
| `size_bytes` | integer | Total size of checkpoint directory |

---

## Download Checkpoint

```
GET /internal/v1/checkpoints/{checkpoint_id}/archive
```

Streams checkpoint as `tar.gz` archive.

### Response

**Headers:**
```
Content-Type: application/gzip
Content-Disposition: attachment; filename="{checkpoint_id}.tar.gz"
```

**Archive contents:**
```
{checkpoint_id}/
├── adapter_config.json
├── adapter_model.safetensors
└── metadata.json
```

---

## Errors

| Status | Description |
|--------|-------------|
| 401 | Invalid or missing token |
| 403 | Checkpoint belongs to another user |
| 404 | Checkpoint not found |

---

## Architecture

```
Browser
    │
    ▼
Application Gateway (reverse proxy)
    │
    ▼
Mint Server (/internal/v1/*)
    │
    ├── Auth: extract user_id from sk- token
    ├── List: query checkpoints where owner_id = user_id
    └── Download: stream tar.gz if owner_id matches
    │
    ▼
Checkpoint Storage (PFS)
```

---

## Implementation

```python
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import tarfile
import io
import os

router = APIRouter(prefix="/internal/v1")


class CheckpointInfo(BaseModel):
    checkpoint_id: str
    model_name: str
    created_at: str
    type: str  # "training" | "inference"
    size_bytes: int


class CheckpointsListResponse(BaseModel):
    checkpoints: list[CheckpointInfo]


@router.get("/checkpoints", response_model=CheckpointsListResponse)
async def list_checkpoints(request: Request):
    user_id = get_user_id(request)
    if user_id is None:
        raise HTTPException(401, "Authentication required")

    checkpoints = query_checkpoints_by_owner(user_id)
    return CheckpointsListResponse(checkpoints=checkpoints)


@router.get("/checkpoints/{checkpoint_id}/archive")
async def download_checkpoint(checkpoint_id: str, request: Request):
    user_id = get_user_id(request)
    if user_id is None:
        raise HTTPException(401, "Authentication required")

    checkpoint = get_checkpoint(checkpoint_id)
    if checkpoint is None:
        raise HTTPException(404, "Checkpoint not found")
    if checkpoint.owner_id != user_id:
        raise HTTPException(403, "Access denied")

    def stream_tar():
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            tar.add(checkpoint.path, arcname=checkpoint_id)
        buffer.seek(0)
        yield buffer.read()

    return StreamingResponse(
        stream_tar(),
        media_type="application/gzip",
        headers={"Content-Disposition": f"attachment; filename={checkpoint_id}.tar.gz"}
    )


def get_dir_size(path: str) -> int:
    """Calculate total size of directory."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total
```

---

## Storage Schema

One directory per user. Ownership implicit from path.

```
/checkpoints/
├── user_123/
│   ├── ckpt_abc123/
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── metadata.json
│   └── ckpt_def456/
│       └── ...
└── user_456/
    └── ...
```

`metadata.json` per checkpoint:

```json
{
  "model_name": "Qwen/Qwen3-30B-A3B",
  "created_at": "2025-01-04T12:30:00Z",
  "type": "training",
  "step": 100
}
```

List checkpoints = `os.listdir(f"/checkpoints/{user_id}")`

No index needed. Ownership check = path validation.
