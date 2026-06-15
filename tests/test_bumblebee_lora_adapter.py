import json
from types import SimpleNamespace

import anyio
import pytest
import torch
from safetensors.torch import load_file, save_file

from mint_server.backend.training.bumblebee.bumblebee_lora import prepare_lora_adapter_for_vllm
from mint_server.models.types import CreateSamplingSessionRequest
from mint_server.routes import service as service_route


def _write_rank_sharded_adapter(adapter):
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 2,
                "lora_alpha": 4,
                "target_modules": ["q_proj"],
                "base_model_name_or_path": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    save_file(
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 4),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(1, 2),
        },
        str(adapter / "adapter_rank-00000.safetensors"),
    )
    save_file(
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.full((1, 2), 2.0),
        },
        str(adapter / "adapter_rank-00001.safetensors"),
    )
    manifest = {
        "format": "bumblebee_qwen3_moe_lora_rank_sharded_v1",
        "sharding_kind": "rank",
        "shards": [
            {
                "rank": 0,
                "file": "adapter_rank-00000.safetensors",
                "parallel_rank": {"tp": 0, "dp": 0, "cp": 0, "pp": 0},
                "tensors": [
                    {
                        "name": "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                        "placement": {"kind": "replicated", "canonical": {"tp": 0, "dp": 0, "cp": 0, "pp": 0}},
                    },
                    {
                        "name": "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                        "placement": {
                            "kind": "concat",
                            "axis": 0,
                            "index": 0,
                            "parts": 2,
                            "canonical": {"dp": 0, "cp": 0, "pp": 0},
                        },
                    },
                ],
            },
            {
                "rank": 1,
                "file": "adapter_rank-00001.safetensors",
                "parallel_rank": {"tp": 1, "dp": 0, "cp": 0, "pp": 0},
                "tensors": [
                    {
                        "name": "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                        "placement": {
                            "kind": "concat",
                            "axis": 0,
                            "index": 1,
                            "parts": 2,
                            "canonical": {"dp": 0, "cp": 0, "pp": 0},
                        },
                    },
                ],
            },
        ],
    }
    (adapter / "bumblebee_rank_sharded_adapter.json").write_text(
        json.dumps(manifest) + "\n",
        encoding="utf-8",
    )


def test_prepare_rank_sharded_bumblebee_lora_converts_to_cached_peft(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("MINT_BUMBLEBEE_LORA_CACHE_DIR", str(cache_root))
    adapter = tmp_path / "adapter"
    _write_rank_sharded_adapter(adapter)

    converted = prepare_lora_adapter_for_vllm(str(adapter))
    assert converted != str(adapter)
    converted_path = cache_root / next(cache_root.iterdir()).name
    assert converted == str(converted_path)
    assert (converted_path / "adapter_config.json").exists()

    tensors = load_file(str(converted_path / "adapter_model.safetensors"))
    assert torch.equal(
        tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"],
        torch.ones(2, 4),
    )
    assert torch.equal(
        tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"],
        torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
    )
    assert prepare_lora_adapter_for_vllm(str(adapter)) == converted


def test_create_sampling_session_converts_rank_sharded_bumblebee_lora(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("MINT_BUMBLEBEE_LORA_CACHE_DIR", str(cache_root))
    adapter = tmp_path / "adapter"
    _write_rank_sharded_adapter(adapter)

    import mint_server.backend.stores.sampling_session_store as sampling_store
    import mint_server.backend.stores.session_index_store as session_index
    import mint_server.gateway as gateway
    import mint_server.supported_models_gate as gate

    registered: dict[str, dict] = {}

    def _upsert_sampling_session(info: dict) -> None:
        registered[str(info["session_id"])] = dict(info)

    async def _allow(base_model: str, http_request=None):
        return base_model

    monkeypatch.setattr(service_route, "session_manager", None)
    monkeypatch.setattr(service_route, "can_access_model", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(service_route, "can_manage_system", lambda _request: True)
    monkeypatch.setattr(gateway, "upstream_for_model", lambda _model: None)
    monkeypatch.setattr(gate, "enforce_base_model_allowed", _allow)
    monkeypatch.setattr(sampling_store, "upsert_sampling_session", _upsert_sampling_session)
    monkeypatch.setattr(session_index, "add_sampler_to_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(session_index, "upsert_sampler_index", lambda *_args, **_kwargs: None)

    request = CreateSamplingSessionRequest(
        session_id="session",
        model_path=str(adapter),
        lora_rank=99,
    )
    http_request = SimpleNamespace(state=SimpleNamespace(user_data={"user_id": "admin"}), headers={})
    response = anyio.run(service_route.create_sampling_session, request, http_request)

    info = registered[response.sampling_session_id]
    converted_path = info["adapter_path"]
    assert converted_path != str(adapter)
    assert converted_path.startswith(str(cache_root))
    assert info["base_model"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert info["lora_rank"] == 2
    assert (cache_root / next(cache_root.iterdir()).name / "adapter_model.safetensors").exists()


def test_prepare_streamed_sharded_bumblebee_lora_fails_fast(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors.index.json").write_text(
        json.dumps({"format": "bumblebee_qwen3_moe_lora_peft_sharded_v1"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="intermediate size-sharded"):
        prepare_lora_adapter_for_vllm(str(adapter))
