from fastapi import FastAPI
from fastapi.testclient import TestClient


def _invalid_training_datum() -> dict:
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": [11, 12]}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": [12, 13], "shape": [2], "dtype": "int64"},
            "logprobs": {"data": [-0.1, -0.2], "shape": [2], "dtype": "float32"},
            "advantages": {"data": [1.0, 1.0], "shape": [2], "dtype": "float32"},
        },
    }


def test_issue_429_forward_backward_rejects_missing_explicit_loss_mask_before_backend() -> None:
    from tinker_server.routes import training as training_routes

    app = FastAPI()
    app.include_router(training_routes.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/forward_backward",
        json={
            "model_id": "run-429",
            "forward_backward_input": {
                "data": [_invalid_training_datum()],
                "loss_fn": "importance_sampling",
            },
        },
    )

    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"] == "Item 0 missing loss_mask/mask/weights"


def test_issue_429_train_step_rejects_missing_explicit_loss_mask_before_backend() -> None:
    from tinker_server.routes import training as training_routes

    app = FastAPI()
    app.include_router(training_routes.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.post(
        "/api/v1/train_step",
        json={
            "model_id": "run-429",
            "adam_params": {"learning_rate": 1e-4},
            "forward_backward_input": {
                "data": [_invalid_training_datum()],
                "loss_fn": "importance_sampling",
            },
        },
    )

    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"] == "Item 0 missing loss_mask/mask/weights"
