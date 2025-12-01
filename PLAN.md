# Tinker-Server MVP: Inference-Only (Wrapping verl)

## Scope

Inference-only MVP wrapping verl's rollout infrastructure for scalability.

**In scope:**
- Service: boilerplate endpoints (healthz, create_session, create_sampling_session)
- Model: Qwen2.5-7B base model only (no LoRA training, but infrastructure ready)
- Sampling: wrap verl's `vLLMHttpServer` Ray actor
- Future: async operation polling

**Out of scope for MVP:**
- Training endpoints (forward_backward, optim_step)
- Weights management (save/load)
- Active LoRA training (but vLLM LoRA loading ready)

## Architecture

```
tinker-cookbook (client)
        |
        | HTTP JSON API
        v
tinker-server (FastAPI)
   |
   +-- SessionManager
   |       +-- session_id -> SamplingSession
   |
   +-- FutureStore
   |       +-- request_id -> result
   |
   +-- verl backend (Ray)
           +-- vLLMHttpServer (Ray actor)
                  +-- AsyncLLM engine
                  +-- Qwen2.5-7B loaded
```

## File Structure

```
tinker_server/
    __init__.py
    app.py                  # FastAPI application
    config.py               # Server configuration
    models/
        __init__.py
        types.py            # Pydantic models
    routes/
        __init__.py
        service.py          # GET /healthz, POST /create_session, /create_sampling_session
        sampling.py         # POST /asample
        futures.py          # POST /retrieve_future
    backend/
        __init__.py
        future_store.py     # request_id -> result storage
        verl_inference.py   # verl vLLMHttpServer wrapper
scripts/
    run_server.py
pyproject.toml
```

## Implementation Tasks

1. Create project structure and pyproject.toml
2. Implement Pydantic models (models/types.py)
3. Implement FutureStore (backend/future_store.py)
4. Implement VerlInferenceEngine wrapper (backend/verl_inference.py)
5. Implement service routes (routes/service.py)
6. Implement sampling routes (routes/sampling.py)
7. Implement futures routes (routes/futures.py)
8. Implement main app (app.py)
9. Test with curl and tinker-cookbook

## Dependencies

```toml
[project]
dependencies = [
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.22.0",
    "pydantic>=2.0.0",
    "ray>=2.10.0",
    "torch>=2.0.0",
    "transformers>=4.40.0",
    "vllm>=0.4.0",
    "omegaconf>=2.3.0",
]
```
