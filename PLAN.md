# Integration Plan: Training + Per-Session Inference

## Goal

Integrate training from `lx_feature` branch while keeping per-session inference from `main`.

Architecture:
- Per-session inference: `SessionManager` → `VerlInferenceEngine` per sampling session
- Per-session training: `TrainingSessionManager` → `VerlTrainingEngine` per model
- Both active simultaneously (no mutually exclusive modes)

## Architecture

```
app.py lifespan
├── SessionManager (inference)
│   ├── session_1 → VerlInferenceEngine (lora_rank=32)
│   ├── session_2 → VerlInferenceEngine (lora_rank=64)
│   └── session_N → VerlInferenceEngine
│
└── TrainingSessionManager (training)
    ├── model_1 → TrainingSession → VerlTrainingEngine state
    ├── model_2 → TrainingSession → VerlTrainingEngine state
    └── model_N → TrainingSession
```

## Implementation Steps

### 1. Add training types to `types.py`

From lx_feature, add (without duplicates):
- `LoRAConfig`
- `CreateModelRequest`, `CreateModelResponse`
- `Datum`, `TensorData`
- `ForwardBackwardInput`, `ForwardBackwardRequest`
- `AdamParams`, `OptimStepRequest`, `OptimStepResponse`
- `TelemetryRequest`, `TelemetryResponse`

### 2. Create `training_session_manager.py`

New file for training session management (separate from inference SessionManager):

```python
class TrainingSession:
    model_id: str
    session_id: str
    model_seq_id: int
    base_model: str
    lora_config: LoRAConfig
    current_step: int
    # ... training state

class TrainingSessionManager:
    _sessions: dict[str, TrainingSession]

    def create_session(model_id, ...) -> TrainingSession
    def get_session(model_id) -> TrainingSession | None
    def delete_session(model_id) -> bool
    def list_sessions() -> list[TrainingSession]
```

### 3. Create `verl_training.py`

From lx_feature, adapt the training engine:
- Uses PyTorch + PEFT for LoRA
- Per-session model/optimizer state
- Methods: `create_training_session`, `forward_backward`, `optim_step`, `shutdown_session`

Key fix: Remove Chinese comments, use English throughout.

### 4. Create `routes/training.py`

Training endpoints:
- `POST /create_model` → create training session
- `POST /forward_backward` → forward + backward pass
- `POST /optim_step` → optimizer update
- `GET /models` → list training sessions
- `DELETE /models/{model_id}` → delete session

### 5. Update `app.py`

Initialize both managers:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inference manager (existing)
    inference_manager = SessionManager(...)
    service.session_manager = inference_manager
    sampling.session_manager = inference_manager
    await inference_manager.start_cleanup_task()

    # Training manager (new)
    training_manager = TrainingSessionManager()
    training_engine = VerlTrainingEngine()
    await training_engine.initialize()
    training.training_manager = training_manager
    training.training_engine = training_engine

    yield

    # Shutdown both
    await inference_manager.shutdown_all()
    await training_manager.shutdown_all(training_engine)
```

### 6. Register training routes

In `app.py`:
```python
from .routes import training
app.include_router(training.router, prefix="/api/v1", tags=["training"])
```

### 7. Add telemetry endpoint

In `service.py`:
```python
@router.post("/telemetry/send")
async def send_telemetry(request: TelemetryRequest) -> TelemetryResponse:
    return TelemetryResponse(status="accepted")
```

## Files Changed

New:
- `tinker_server/backend/training_session_manager.py`
- `tinker_server/backend/verl_training.py`
- `tinker_server/routes/training.py`

Modified:
- `tinker_server/models/types.py` (add training types)
- `tinker_server/app.py` (initialize both managers)
- `tinker_server/routes/service.py` (add telemetry endpoint)

Unchanged:
- `tinker_server/backend/session_manager.py` (inference sessions)
- `tinker_server/backend/verl_inference.py`
- `tinker_server/routes/sampling.py`

## Bug Fixes from lx_feature

1. ~~config.py boolean parsing~~ - Not needed (no ENABLE_TRAINING flag)
2. Duplicate `LossFnOutput` class - Only include once in types.py

## Testing

After implementation:
```bash
# Test inference (existing)
python scripts/test_client.py

# Test training (new)
python scripts/test_training_client.py
```
