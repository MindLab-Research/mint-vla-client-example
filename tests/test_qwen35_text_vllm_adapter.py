from __future__ import annotations

import json
import sys
import types

import pytest
import torch

from mint_server.backend import qwen35_text_vllm_adapter as adapter


def _raw_qwen35_config() -> dict[str, object]:
    return {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "tie_word_embeddings": False,
        "image_token_id": 248056,
        "video_token_id": 248057,
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": 248320,
            "hidden_size": 5120,
            "num_hidden_layers": 8,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "intermediate_size": 17408,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "head_dim": 256,
            "max_position_embeddings": 262144,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_num_value_heads": 48,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
            },
        },
    }


def test_qwen35_text_adapter_materializes_marked_text_only_config(tmp_path):
    model_dir = tmp_path / "qwen35"
    model_dir.mkdir()
    raw_config = _raw_qwen35_config()
    (model_dir / "config.json").write_text(json.dumps(raw_config), encoding="utf-8")

    config_dir = adapter.materialize_qwen35_text_vllm_config(
        str(model_dir),
        root_dir=str(tmp_path / "runtime"),
    )

    assert config_dir is not None
    config = json.loads((tmp_path / "runtime").glob("qwen35-text-vllm-config/*/config.json").__next__().read_text())
    assert config["model_type"] == "qwen3_next"
    assert config["architectures"] == [adapter.QWEN35_VLLM_ARCHITECTURE]
    assert config[adapter.QWEN35_TEXT_ONLY_SHIM_MARKER] is True
    assert config["mint_source_model_type"] == adapter.QWEN35_MODEL_TYPE
    assert config["mint_supported_modality"] == adapter.QWEN35_SUPPORTED_MODALITY
    assert config["num_experts"] == 0
    assert "image_token_id" not in config
    assert "video_token_id" not in config
    assert "mrope_section" not in config["rope_parameters"]
    assert "mrope_interleaved" not in config["rope_parameters"]


def test_qwen35_text_adapter_resolves_cached_repo_id(monkeypatch, tmp_path):
    cache_root = tmp_path / "hf-cache"
    repo_cache = cache_root / "models--Qwen--Qwen3.5-27B"
    snapshot = repo_cache / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (repo_cache / "refs").mkdir()
    (repo_cache / "refs" / "main").write_text("abc123\n", encoding="utf-8")
    (snapshot / "config.json").write_text(json.dumps(_raw_qwen35_config()), encoding="utf-8")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache_root))

    config_dir = adapter.materialize_qwen35_text_vllm_config(
        "Qwen/Qwen3.5-27B",
        root_dir=str(tmp_path / "runtime"),
    )

    assert config_dir is not None
    config = json.loads((tmp_path / "runtime").glob("qwen35-text-vllm-config/*/config.json").__next__().read_text())
    assert config["model_type"] == "qwen3_next"
    assert adapter.resolve_hf_config_dir("Qwen/Qwen3.5-27B") == str(snapshot)


def test_qwen35_text_adapter_ignores_non_qwen35_model(tmp_path):
    model_dir = tmp_path / "qwen3"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}),
        encoding="utf-8",
    )

    assert adapter.materialize_qwen35_text_vllm_config(str(model_dir)) is None


def test_qwen35_text_adapter_requires_real_qwen35_outer_config(tmp_path):
    model_dir = tmp_path / "qwen35_text_only_inner"
    model_dir.mkdir()
    raw_config = _raw_qwen35_config()
    raw_config["model_type"] = "qwen3_next"
    (model_dir / "config.json").write_text(json.dumps(raw_config), encoding="utf-8")

    assert adapter.is_qwen35_config(raw_config) is False
    assert adapter.materialize_qwen35_text_vllm_config(str(model_dir)) is None


def test_qwen35_text_adapter_rejects_malformed_qwen35_text_config(tmp_path):
    model_dir = tmp_path / "bad_qwen35"
    model_dir.mkdir()
    raw_config = _raw_qwen35_config()
    text_config = raw_config["text_config"]
    assert isinstance(text_config, dict)
    text_config.pop("linear_num_key_heads")
    (model_dir / "config.json").write_text(json.dumps(raw_config), encoding="utf-8")

    with pytest.raises(ValueError, match="linear_num_key_heads"):
        adapter.materialize_qwen35_text_vllm_config(str(model_dir))


def test_qwen35_text_adapter_rejects_unsupported_layer_types(tmp_path):
    model_dir = tmp_path / "bad_qwen35_layer_type"
    model_dir.mkdir()
    raw_config = _raw_qwen35_config()
    text_config = raw_config["text_config"]
    assert isinstance(text_config, dict)
    text_config["layer_types"] = ["linear_attention", "sliding_attention"] * 4
    (model_dir / "config.json").write_text(json.dumps(raw_config), encoding="utf-8")

    with pytest.raises(ValueError, match="sliding_attention"):
        adapter.materialize_qwen35_text_vllm_config(str(model_dir))


def test_qwen35_text_adapter_dtype_prefers_text_config_without_autoconfig(monkeypatch, tmp_path):
    model_dir = tmp_path / "qwen35"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "text_config": {"dtype": "bfloat16"},
            }
        ),
        encoding="utf-8",
    )

    fake_transformers = types.ModuleType("transformers")

    class _FailingAutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise AssertionError("AutoConfig should not be needed for config.json dtype")

    fake_transformers.AutoConfig = _FailingAutoConfig
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    assert adapter.infer_hf_torch_dtype_str(str(model_dir)) == "bfloat16"


def test_qwen35_text_adapter_weight_packing_fails_fast_on_bad_shapes():
    config = types.SimpleNamespace(
        linear_num_key_heads=2,
        linear_key_head_dim=2,
        linear_num_value_heads=4,
        linear_value_head_dim=2,
    )

    with pytest.raises(ValueError, match="in_proj_qkv weight shape"):
        adapter._pack_qwen35_qkv_z(config, torch.zeros((15, 3)), torch.zeros((8, 3)))

    with pytest.raises(ValueError, match="in_proj_a weight shape"):
        adapter._pack_qwen35_b_a(config, torch.zeros((4, 3)), torch.zeros((5, 3)))


def test_qwen35_text_adapter_weight_mapping_fails_on_incomplete_split_weights():
    config = types.SimpleNamespace(
        linear_num_key_heads=2,
        linear_key_head_dim=2,
        linear_num_value_heads=4,
        linear_value_head_dim=2,
    )
    weights = [
        (
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            torch.zeros((16, 3)),
        )
    ]

    with pytest.raises(ValueError, match="Incomplete Qwen3.5 linear attention split weights"):
        list(adapter._map_qwen35_text_weights(config, weights, inner_model=True))


def test_qwen35_text_adapter_weight_mapping_rejects_duplicate_split_weights():
    config = types.SimpleNamespace(
        linear_num_key_heads=2,
        linear_key_head_dim=2,
        linear_num_value_heads=4,
        linear_value_head_dim=2,
    )
    weights = [
        (
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            torch.zeros((16, 3)),
        ),
        (
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            torch.ones((16, 3)),
        ),
    ]

    with pytest.raises(ValueError, match="Duplicate Qwen3.5 linear attention split weight"):
        list(adapter._map_qwen35_text_weights(config, weights, inner_model=True))
