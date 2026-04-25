import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_issue_417_weights_info_reads_training_checkpoint_metadata(tmp_path: Path, monkeypatch) -> None:
    from tinker_server.routes import weights as weights_routes

    ckpt = tmp_path / "ckpt_417"
    ckpt.mkdir()
    (ckpt / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "checkpoint_type": "training",
                "optimizer_present": True,
            }
        ),
        encoding="utf-8",
    )
    (ckpt / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 16,
                "target_modules": [
                    "q_a_proj",
                    "q_b_proj",
                    "kv_a_proj_with_mqa",
                    "kv_b_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                "base_model_name_or_path": "/metadata/model/should/win",
            }
        ),
        encoding="utf-8",
    )
    (ckpt / "adapter_model.safetensors").write_bytes(b"adapter")
    (ckpt / "mp_rank_00_optimizer.pt").write_bytes(b"optimizer")

    monkeypatch.setattr(
        weights_routes,
        "_resolve_mint_path",
        lambda mint_uri, *, user_id, is_admin=False: str(ckpt),
    )

    app = FastAPI()
    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    response = client.post("/api/v1/weights_info", json={"tinker_path": "tinker://run/weights/ckpt_417"})

    assert response.status_code == 200, response.text
    assert response.json() == {
        "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "is_lora": True,
        "lora_rank": 16,
        "train_unembed": False,
        "train_mlp": True,
        "train_attn": True,
    }


def test_issue_417_weights_info_rejects_sampler_checkpoint_for_training_resume(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tinker_server.routes import weights as weights_routes

    ckpt = tmp_path / "sampler_ckpt"
    ckpt.mkdir()
    (ckpt / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": "Qwen/Qwen3-0.6B",
                "checkpoint_type": "sampler",
                "optimizer_present": False,
            }
        ),
        encoding="utf-8",
    )
    (ckpt / "adapter_model.safetensors").write_bytes(b"adapter")

    monkeypatch.setattr(
        weights_routes,
        "_resolve_mint_path",
        lambda mint_uri, *, user_id, is_admin=False: str(ckpt),
    )

    app = FastAPI()
    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    response = client.post("/api/v1/weights_info", json={"tinker_path": "tinker://run/sampler_weights/ckpt"})

    assert response.status_code == 400, response.text
    assert "Sampler checkpoint cannot recreate a training client" in response.text


def test_issue_417_weights_info_accepts_optimizerless_training_adapter_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tinker_server.routes import weights as weights_routes

    ckpt = tmp_path / "adapter_only_training_ckpt"
    ckpt.mkdir()
    (ckpt / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "checkpoint_type": "training",
            }
        ),
        encoding="utf-8",
    )
    (ckpt / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 8,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                "base_model_name_or_path": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            }
        ),
        encoding="utf-8",
    )
    (ckpt / "mp_rank_00_adapter.pt").write_bytes(b"adapter")

    monkeypatch.setattr(
        weights_routes,
        "_resolve_mint_path",
        lambda mint_uri, *, user_id, is_admin=False: str(ckpt),
    )

    app = FastAPI()
    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    response = client.post("/api/v1/weights_info", json={"tinker_path": "tinker://run/weights/ckpt"})

    assert response.status_code == 200, response.text
    assert response.json() == {
        "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "is_lora": True,
        "lora_rank": 8,
        "train_unembed": False,
        "train_mlp": True,
        "train_attn": True,
    }


def test_issue_417_weights_info_rejects_lora_checkpoint_without_rank(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tinker_server.routes import weights as weights_routes

    ckpt = tmp_path / "malformed_lora_ckpt"
    ckpt.mkdir()
    (ckpt / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "checkpoint_type": "training",
                "optimizer_present": True,
            }
        ),
        encoding="utf-8",
    )
    (ckpt / "adapter_config.json").write_text(
        json.dumps(
            {
                "target_modules": ["q_a_proj", "gate_proj"],
                "base_model_name_or_path": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            }
        ),
        encoding="utf-8",
    )
    (ckpt / "adapter_model.safetensors").write_bytes(b"adapter")
    (ckpt / "mp_rank_00_optimizer.pt").write_bytes(b"optimizer")

    monkeypatch.setattr(
        weights_routes,
        "_resolve_mint_path",
        lambda mint_uri, *, user_id, is_admin=False: str(ckpt),
    )

    app = FastAPI()
    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    response = client.post("/api/v1/weights_info", json={"tinker_path": "tinker://run/weights/bad"})

    assert response.status_code == 400, response.text
    assert "Invalid or missing LoRA rank" in response.text


def test_issue_417_weights_info_rejects_malformed_adapter_config_as_400(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tinker_server.routes import weights as weights_routes

    ckpt = tmp_path / "malformed_json_ckpt"
    ckpt.mkdir()
    (ckpt / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "checkpoint_type": "training",
                "optimizer_present": True,
            }
        ),
        encoding="utf-8",
    )
    (ckpt / "adapter_config.json").write_text('{"r": 16,', encoding="utf-8")
    (ckpt / "adapter_model.safetensors").write_bytes(b"adapter")
    (ckpt / "mp_rank_00_optimizer.pt").write_bytes(b"optimizer")

    monkeypatch.setattr(
        weights_routes,
        "_resolve_mint_path",
        lambda mint_uri, *, user_id, is_admin=False: str(ckpt),
    )

    app = FastAPI()
    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    response = client.post("/api/v1/weights_info", json={"tinker_path": "tinker://run/weights/bad-json"})

    assert response.status_code == 400, response.text
    assert "Malformed adapter_config.json" in response.text


def test_issue_417_weights_info_rejects_non_object_metadata_as_400(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tinker_server.routes import weights as weights_routes

    ckpt = tmp_path / "non_object_metadata_ckpt"
    ckpt.mkdir()
    (ckpt / "metadata.json").write_text("[]", encoding="utf-8")
    (ckpt / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 16,
                "target_modules": ["q_a_proj", "gate_proj"],
                "base_model_name_or_path": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            }
        ),
        encoding="utf-8",
    )
    (ckpt / "adapter_model.safetensors").write_bytes(b"adapter")
    (ckpt / "mp_rank_00_optimizer.pt").write_bytes(b"optimizer")

    monkeypatch.setattr(
        weights_routes,
        "_resolve_mint_path",
        lambda mint_uri, *, user_id, is_admin=False: str(ckpt),
    )

    app = FastAPI()
    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    response = client.post("/api/v1/weights_info", json={"tinker_path": "tinker://run/weights/non-object"})

    assert response.status_code == 400, response.text
    assert "Invalid metadata.json" in response.text


def test_issue_417_weights_info_accepts_non_lora_training_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tinker_server.routes import weights as weights_routes

    ckpt = tmp_path / "openpi_training_ckpt"
    (ckpt / "1" / "params").mkdir(parents=True)
    (ckpt / "1" / "train_state").mkdir(parents=True)
    (ckpt / "1" / "assets" / "robot").mkdir(parents=True)
    (ckpt / "1" / "params" / "_METADATA").write_text("params", encoding="utf-8")
    (ckpt / "1" / "train_state" / "_METADATA").write_text("train", encoding="utf-8")
    (ckpt / "1" / "assets" / "robot" / "norm_stats.json").write_text("{}", encoding="utf-8")
    (ckpt / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": "openpi/pi0",
                "checkpoint_type": "training",
                "optimizer_present": True,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        weights_routes,
        "_resolve_mint_path",
        lambda mint_uri, *, user_id, is_admin=False: str(ckpt),
    )

    app = FastAPI()
    app.include_router(weights_routes.router, prefix="/api/v1")
    client = TestClient(app)

    response = client.post("/api/v1/weights_info", json={"tinker_path": "tinker://run/weights/openpi"})

    assert response.status_code == 200, response.text
    assert response.json() == {
        "base_model": "openpi/pi0",
        "is_lora": False,
        "lora_rank": None,
        "train_unembed": None,
        "train_mlp": None,
        "train_attn": None,
    }
