import pytest
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


def _datum_with_mask_key(mask_key: str) -> dict:
    datum = _invalid_training_datum()
    datum["loss_fn_inputs"][mask_key] = {
        "data": [1.0, 1.0],
        "shape": [2],
        "dtype": "float32",
    }
    return datum


def _client_with_auth(router) -> TestClient:
    app = FastAPI()

    @app.middleware("http")
    async def _inject_user_data(request, call_next):
        request.state.user_data = {"user_id": "anonymous", "user_role": "user"}
        return await call_next(request)

    app.include_router(router, prefix="/api/v1")
    return TestClient(app)


def test_issue_429_forward_backward_rejects_missing_explicit_loss_mask_before_backend() -> None:
    from tinker_server.routes import training as training_routes

    client = _client_with_auth(training_routes.router)

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

    client = _client_with_auth(training_routes.router)

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


def test_issue_429_validation_accepts_weights_mask_and_loss_mask_aliases() -> None:
    from tinker_server.routes import training as training_routes

    for mask_key in ("weights", "mask", "loss_mask"):
        datum = training_routes.Datum.model_validate(_datum_with_mask_key(mask_key))
        training_routes._validate_training_batch_has_explicit_loss_masks_or_422([datum])


def test_issue_429_validation_reports_item_index() -> None:
    from tinker_server.routes import training as training_routes

    data = [
        training_routes.Datum.model_validate(_datum_with_mask_key("weights")),
        training_routes.Datum.model_validate(_invalid_training_datum()),
    ]

    with pytest.raises(training_routes.HTTPException) as exc_info:
        training_routes._validate_training_batch_has_explicit_loss_masks_or_422(data)

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "Item 1 missing loss_mask/mask/weights"
