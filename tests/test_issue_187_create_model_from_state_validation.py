import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient


def test_issue_187_create_model_from_state_rejects_sampler_checkpoint_with_optimizer(
    tmp_path: Path,
) -> None:
    from tinker_server.routes import training as training_routes
    from tinker_server import checkpoints as checkpoints_module

    # Patch checkpoints root for this test module (resolver reads from checkpoints.py).
    training_routes.CHECKPOINTS_DIR = str(tmp_path)
    checkpoints_module.CHECKPOINTS_DIR = str(tmp_path)
    checkpoints_module.PERSISTENT_CHECKPOINTS_DIR = str(tmp_path)
    checkpoints_module.RUNTIME_CHECKPOINTS_DIR = str(tmp_path / "runtime")

    run_id = "run-187-cmfs"
    ckpt_name = "sampler-0001"
    ckpt_dir = tmp_path / "anonymous" / run_id / ckpt_name
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "adapter_model.safetensors").write_bytes(b"dummy-lora")
    (ckpt_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": ckpt_name,
                "owner_id": None,
                "model_id": run_id,
                "model_name": "Qwen/Qwen3-0.6B",
                "created_at": "2026-02-26T00:00:00Z",
                "step": 0,
                "checkpoint_type": "sampler",
                "optimizer_present": False,
                "backend": "dense",
                "type": "sampler",
            }
        ),
        encoding="utf-8",
    )

    app = FastAPI()

    @app.middleware("http")
    async def inject_user(request: Request, call_next):
        request.state.user_data = {"user_id": "anonymous"}
        return await call_next(request)

    app.include_router(training_routes.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/create_model_from_state",
        json={
            "session_id": "s",
            "model_seq_id": 0,
            "base_model": "Qwen/Qwen3-0.6B",
            "state_path": f"tinker://{run_id}/sampler_weights/{ckpt_name}",
            "lora_config": {"rank": 8},
            "load_optimizer": True,
        },
    )
    assert resp.status_code == 400, resp.text
    assert "checkpoint_type is not 'training'" in resp.text
