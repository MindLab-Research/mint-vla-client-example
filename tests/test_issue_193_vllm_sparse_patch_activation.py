import asyncio

import pytest

pytest.importorskip("verl")
pytest.importorskip("vllm")

from mint_server.backend import verl_inference
from mint_server.backend.verl_inference import _create_extended_server_class


def _make_server_instance():
    remote_cls = _create_extended_server_class()
    impl_cls = remote_cls.__ray_metadata__.modified_class
    return object.__new__(impl_cls)


def test_issue_193_sparse_ep_representative_state_dict_triggers_pack_moe_patch():
    server = _make_server_instance()
    calls: list[str] = []

    async def fake_ensure():
        calls.append("patched")

    server._ensure_pack_moe_patched = fake_ensure  # type: ignore[method-assign]
    original = verl_inference._mint_expected_num_experts_from_base_model
    verl_inference._mint_expected_num_experts_from_base_model = lambda _base_model: 8
    try:
        asyncio.run(
            server._maybe_ensure_pack_moe_patched_for_state_dict(
                {
                    "base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight": object(),
                    "base_model.model.model.layers.0.mlp.experts.4.gate_proj.lora_A.weight": object(),
                },
                base_model_name_or_path="fake-model",
            )
        )
    finally:
        verl_inference._mint_expected_num_experts_from_base_model = original

    assert calls == ["patched"]


def test_issue_193_attention_only_state_dict_does_not_trigger_pack_moe_patch():
    server = _make_server_instance()
    calls: list[str] = []

    async def fake_ensure():
        calls.append("patched")

    server._ensure_pack_moe_patched = fake_ensure  # type: ignore[method-assign]

    asyncio.run(
        server._maybe_ensure_pack_moe_patched_for_state_dict(
            {
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": object(),
            }
        )
    )

    assert calls == []


def test_issue_193_full_per_expert_state_dict_does_not_trigger_pack_moe_patch():
    server = _make_server_instance()
    calls: list[str] = []

    async def fake_ensure():
        calls.append("patched")

    server._ensure_pack_moe_patched = fake_ensure  # type: ignore[method-assign]

    original = verl_inference._mint_expected_num_experts_from_base_model
    verl_inference._mint_expected_num_experts_from_base_model = lambda _base_model: 2
    try:
        asyncio.run(
            server._maybe_ensure_pack_moe_patched_for_state_dict(
                {
                    "base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight": object(),
                    "base_model.model.model.layers.0.mlp.experts.1.gate_proj.lora_A.weight": object(),
                },
                base_model_name_or_path="fake-model",
            )
        )
    finally:
        verl_inference._mint_expected_num_experts_from_base_model = original

    assert calls == []
