from __future__ import annotations

import builtins
import asyncio
import importlib
import json
import os
import sys
import threading
import time
from types import SimpleNamespace
from typing import cast

import anyio
from fastapi import HTTPException, Request
import pytest
import ray

from mint_server.backend.contracts.control_plane_contracts import AppendWorkResult
from mint_server.backend.scheduling.model_work_admission import ModelWorkAdmissionResult
from mint_server.models.types import ComputeLogprobsRequest, ModelInput, SampleRequest, SamplingParams
from mint_server.backend.sglang_actor import (
    SGLangEngineActor,
    _apply_sglang_flashinfer_kernel_compatibility,
    _apply_sglang_import_boundary,
    normalize_sglang_generation_response,
    normalize_sglang_prompt_logprobs_response,
    normalize_sglang_prompt_topk_response,
)
from mint_server.backend.sglang_capabilities import (
    SGLANG_QWEN3_MOE_EXPERT_LORA_FEATURE,
    SGLangUnsupportedFeatureError,
    check_sglang_lora_adapter_support,
)
from mint_server.backend.sglang_engine import (
    PERSISTENT_NAMESPACE,
    GenerateResult,
    SGLangInferenceEngine,
    _assert_single_node_sglang_schedulable,
    _assert_sglang_node_ip_capacity,
    _default_sglang_engine_ready_wait_s,
    _sglang_pythonpath,
    _sglang_pycache_prefix,
    _rank_actor_name,
    _runtime_env,
    _sglang_actor_lora_defaults_for_config,
    _sglang_actor_memory_defaults_for_config,
    _single_node_pin_for_model,
    remove_sglang_session_from_existing_actor,
)
from mint_server.backend.core.model_registry import ModelConfig
from mint_server.backend.core.model_registry import get_model_config
from mint_server.routes import sampling as sampling_route


def _stub_single_node_gpu_capacity(monkeypatch, *, gpu_count: int = 8) -> None:
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.ray.nodes",
        lambda: [
            {
                "Alive": True,
                "NodeManagerAddress": "192.0.2.10",
                "Resources": {"GPU": gpu_count},
            }
        ],
    )


def test_importing_sglang_backend_modules_does_not_import_sglang(monkeypatch) -> None:
    sys.modules.pop("sglang", None)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sglang" or str(name).startswith("sglang."):
            raise AssertionError("SGLang must not be imported at Mint module import time")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    importlib.reload(importlib.import_module("mint_server.backend.sglang_actor"))
    importlib.reload(importlib.import_module("mint_server.backend.sglang_engine"))


def test_normalize_sglang_generation_response_extracts_token_ids_and_logprobs() -> None:
    normalized = normalize_sglang_generation_response(
        {
            "text": " hello",
            "output_ids": [11, "12"],
            "finish_reason": {"type": "length"},
            "meta_info": {
                "output_token_logprobs": [
                    [-0.1, 11, None],
                    {"logprob": "-0.2", "token_id": 12},
                ]
            },
        }
    )

    assert normalized["token_ids"] == [11, 12]
    assert normalized["text"] == " hello"
    assert normalized["logprobs"] == [-0.1, -0.2]
    assert normalized["stop_reason"] == "length"


def test_normalize_sglang_generation_response_rejects_missing_output_ids() -> None:
    with pytest.raises(RuntimeError, match="missing output_ids"):
        normalize_sglang_generation_response({"meta_info": {}})


def test_normalize_sglang_generation_response_rejects_logprob_length_mismatch() -> None:
    with pytest.raises(RuntimeError, match="output_token_logprobs length"):
        normalize_sglang_generation_response(
            {
                "output_ids": [1, 2],
                "meta_info": {"output_token_logprobs": [[-0.1, 1, None]]},
            }
        )


def test_normalize_sglang_prompt_logprobs_response_uses_real_phase4_shape() -> None:
    response = {
        "output_ids": [0, 1096],
        "meta_info": {
            "input_token_logprobs": [
                [None, 9707, None],
                [-11.805401802062988, 1879, None],
            ],
        },
    }

    assert normalize_sglang_prompt_logprobs_response(response, prompt_ids=[9707, 1879]) == [
        None,
        -11.805401802062988,
    ]


def test_normalize_sglang_prompt_logprobs_rejects_token_mismatch() -> None:
    with pytest.raises(RuntimeError, match="token id 999 does not match"):
        normalize_sglang_prompt_logprobs_response(
            {
                "meta_info": {
                    "input_token_logprobs": [
                        [None, 9707, None],
                        [-1.0, 999, None],
                    ],
                },
            },
            prompt_ids=[9707, 1879],
        )


def test_normalize_sglang_prompt_logprobs_rejects_first_token_mismatch() -> None:
    with pytest.raises(RuntimeError, match=r"input_token_logprobs\[0\] token id 999 does not match"):
        normalize_sglang_prompt_logprobs_response(
            {
                "meta_info": {
                    "input_token_logprobs": [
                        [None, 999, None],
                        [-1.0, 1879, None],
                    ],
                },
            },
            prompt_ids=[9707, 1879],
        )


def test_normalize_sglang_prompt_topk_response_uses_real_phase4_shape() -> None:
    response = {
        "meta_info": {
            "input_top_logprobs": [
                None,
                [
                    ["-5.016339302062988", "21806", "None"],
                    ["-5.172589302062988", "14582", "None"],
                    ["-5.547589302062988", "15846", "None"],
                ],
            ],
        },
    }

    assert normalize_sglang_prompt_topk_response(response, prompt_ids=[9707, 1879], k=2) == [
        None,
        [(21806, -5.016339302062988), (14582, -5.172589302062988)],
    ]


def test_phase4_sglang_prompt_logprobs_match_hf_bfloat16_reference_fixture() -> None:
    response = {
        "meta_info": {
            "input_token_logprobs": [
                [None, 9707, None],
                [-11.805401802062988, 1879, None],
            ],
            "input_top_logprobs": [
                None,
                [
                    [-5.016339302062988, 21806, None],
                    [-5.172589302062988, 14582, None],
                    [-5.547589302062988, 15846, None],
                ],
            ],
        },
    }

    prompt_logprobs = normalize_sglang_prompt_logprobs_response(response, prompt_ids=[9707, 1879])
    topk = normalize_sglang_prompt_topk_response(response, prompt_ids=[9707, 1879], k=3)

    hf_bfloat16_logprob_for_1879_after_9707 = -11.81741714477539
    hf_bfloat16_top3_after_9707 = [
        (21806, -5.043979644775391),
        (14582, -5.168979644775391),
        (15846, -5.543979644775391),
    ]

    assert prompt_logprobs[0] is None
    assert prompt_logprobs[1] == pytest.approx(hf_bfloat16_logprob_for_1879_after_9707, abs=0.05)
    assert topk[0] is None
    assert topk[1] is not None
    assert [token_id for token_id, _logprob in topk[1]] == [
        token_id for token_id, _logprob in hf_bfloat16_top3_after_9707
    ]
    for (_token_id, sglang_logprob), (_hf_token_id, hf_logprob) in zip(topk[1], hf_bfloat16_top3_after_9707):
        assert sglang_logprob == pytest.approx(hf_logprob, abs=0.05)


def test_sglang_actor_default_mem_fraction_can_follow_model_config(monkeypatch) -> None:
    captured = {}

    class _Engine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sglang.srt.entrypoints.engine" and "Engine" in fromlist:
            class _Module:
                Engine = _Engine

            return _Module()
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delenv("MINT_SGLANG_MEM_FRACTION_STATIC", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    actor = SGLangEngineActor(
        model_name="Qwen/Qwen3-0.6B",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_0_6b",
        max_model_len=512,
        default_mem_fraction_static=0.85,
    )
    actor.initialize()

    assert captured["mem_fraction_static"] == 0.85
    assert captured["disable_piecewise_cuda_graph"] is True
    assert captured["enable_lora"] is True
    assert captured["max_lora_rank"] == 64
    assert captured["max_loaded_loras"] == 8
    assert captured["max_loras_per_batch"] == 8
    assert captured["lora_backend"] == "triton"
    assert captured["lora_target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def test_sglang_actor_wraps_scheduler_entrypoint_for_spawned_child_compatibility(monkeypatch) -> None:
    captured = {}

    class _Engine:
        run_scheduler_process_func = staticmethod(lambda *args, **kwargs: None)

        def __init__(self, **kwargs):
            captured["engine_class"] = type(self)
            captured["kwargs"] = kwargs

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sglang.srt.entrypoints.engine" and "Engine" in fromlist:
            class _Module:
                Engine = _Engine

            return _Module()
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setenv("MINT_SGLANG_DISABLE_FLASHINFER_KERNELS", "1")
    monkeypatch.setattr(
        "mint_server.backend.sglang_actor._apply_sglang_flashinfer_kernel_compatibility",
        lambda **_kwargs: {"enabled": True},
    )
    monkeypatch.setattr(builtins, "__import__", fake_import)

    actor = SGLangEngineActor(
        model_name="Qwen/Qwen3-0.6B",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_0_6b",
        max_model_len=512,
    )
    actor.initialize()

    assert captured["engine_class"] is not _Engine
    live_actor_module = importlib.import_module("mint_server.backend.sglang_actor")
    assert (
        captured["engine_class"].run_scheduler_process_func
        is live_actor_module._run_sglang_scheduler_process_with_mint_compatibility
    )


def test_sglang_actor_disables_config_actor_hydration_for_engine_children(monkeypatch) -> None:
    captured = {}

    class _Engine:
        def __init__(self, **kwargs):
            captured["hydrate_during_init"] = os.environ.get("MINT_CONFIG_ACTOR_HYDRATE")
            captured["kwargs"] = kwargs

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sglang.srt.entrypoints.engine" and "Engine" in fromlist:
            class _Module:
                Engine = _Engine

            return _Module()
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setenv("MINT_CONFIG_ACTOR_HYDRATE", "1")
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(
        "mint_server.backend.sglang_actor._apply_sglang_flashinfer_kernel_compatibility",
        lambda **_kwargs: {"enabled": True},
    )

    actor = SGLangEngineActor(
        model_name="Qwen/Qwen3-0.6B",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_0_6b",
        max_model_len=512,
    )
    actor.initialize()

    assert captured["hydrate_during_init"] == "0"
    assert os.environ["MINT_CONFIG_ACTOR_HYDRATE"] == "1"


def test_sglang_flashinfer_kernel_compatibility_disables_direct_sgl_kernel_imports(monkeypatch) -> None:
    def is_flashinfer_available():
        return os.environ.get("SGLANG_IS_FLASHINFER_AVAILABLE") not in {"0", "false", "False"}

    is_flashinfer_available.cache_clear = lambda: None  # type: ignore[attr-defined]

    class _Common:
        pass

    _Common.is_flashinfer_available = is_flashinfer_available  # type: ignore[attr-defined]

    class _Elementwise:
        _has_flashinfer = True

    class _Sampling:
        _has_flashinfer = True

    modules = {
        "sglang.srt.utils.common": _Common,
        "sgl_kernel.elementwise": _Elementwise,
        "sgl_kernel.sampling": _Sampling,
    }

    monkeypatch.setenv("MINT_SGLANG_DISABLE_FLASHINFER_KERNELS", "1")
    monkeypatch.delenv("SGLANG_IS_FLASHINFER_AVAILABLE", raising=False)
    monkeypatch.setattr(
        "mint_server.backend.sglang_actor.importlib.import_module",
        lambda name: modules[name],
    )

    result = _apply_sglang_flashinfer_kernel_compatibility(import_direct_kernel_modules=True)

    assert result["enabled"] is True
    assert result["import_direct_kernel_modules"] is True
    assert os.environ["SGLANG_IS_FLASHINFER_AVAILABLE"] == "0"
    assert result["sglang_is_flashinfer_available"] is False
    assert _Elementwise._has_flashinfer is False
    assert _Sampling._has_flashinfer is False
    assert result["modules"]["sgl_kernel.elementwise"]["has_flashinfer_before"] is True
    assert result["modules"]["sgl_kernel.sampling"]["has_flashinfer_before"] is True


def test_sglang_flashinfer_actor_compatibility_does_not_import_direct_kernels(monkeypatch) -> None:
    imported: list[str] = []

    def fake_import_module(name: str):
        imported.append(name)
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setenv("MINT_SGLANG_DISABLE_FLASHINFER_KERNELS", "1")
    monkeypatch.delenv("SGLANG_IS_FLASHINFER_AVAILABLE", raising=False)
    monkeypatch.delitem(sys.modules, "sglang.srt.utils.common", raising=False)
    monkeypatch.delitem(sys.modules, "sgl_kernel.elementwise", raising=False)
    monkeypatch.delitem(sys.modules, "sgl_kernel.sampling", raising=False)
    monkeypatch.setattr("mint_server.backend.sglang_actor.importlib.import_module", fake_import_module)

    result = _apply_sglang_flashinfer_kernel_compatibility(import_direct_kernel_modules=False)

    assert result["enabled"] is True
    assert result["import_direct_kernel_modules"] is False
    assert os.environ["SGLANG_IS_FLASHINFER_AVAILABLE"] == "0"
    assert result["sglang_common"] == {"available": False, "skipped": "not_loaded"}
    assert imported == []
    assert result["modules"]["sgl_kernel.elementwise"] == {"available": False, "skipped": "not_loaded"}
    assert result["modules"]["sgl_kernel.sampling"] == {"available": False, "skipped": "not_loaded"}


def test_sglang_flashinfer_actor_compatibility_patches_already_loaded_direct_kernels(monkeypatch) -> None:
    class _Elementwise:
        _has_flashinfer = True

    class _Sampling:
        _has_flashinfer = True

    monkeypatch.setenv("MINT_SGLANG_DISABLE_FLASHINFER_KERNELS", "1")
    monkeypatch.setitem(sys.modules, "sgl_kernel.elementwise", _Elementwise)
    monkeypatch.setitem(sys.modules, "sgl_kernel.sampling", _Sampling)
    monkeypatch.setattr(
        "mint_server.backend.sglang_actor.importlib.import_module",
        lambda name: (_ for _ in ()).throw(AssertionError(f"unexpected import: {name}")),
    )

    result = _apply_sglang_flashinfer_kernel_compatibility(import_direct_kernel_modules=False)

    assert _Elementwise._has_flashinfer is False
    assert _Sampling._has_flashinfer is False
    assert result["modules"]["sgl_kernel.elementwise"]["has_flashinfer_before"] is True
    assert result["modules"]["sgl_kernel.sampling"]["has_flashinfer_before"] is True


def test_sglang_runtime_env_uses_explicit_worker_venv_python(monkeypatch) -> None:
    monkeypatch.setattr("mint_server.backend.sglang_engine.PFS_PYTHONPATH", "/runtime/site-packages")
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.actor_runtime_env_vars",
        lambda *, pythonpath, extra=None: {"PYTHONPATH": pythonpath, **dict(extra or {})},
    )
    monkeypatch.setattr("mint_server.backend.sglang_engine.otel_env_vars", lambda: {})
    monkeypatch.setenv("MINT_SGLANG_PYTHONPATH", "/runtime/sglang-overlay")
    monkeypatch.setenv(
        "MINT_SGLANG_PY_EXECUTABLE",
        "/vePFS-Mindverse/share/mint/dev/runtime/sglang-overlays/sglang-0.5.12.post1-cu129-venv/bin/python",
    )
    monkeypatch.setenv("MINT_SGLANG_DISABLE_FLASHINFER_KERNELS", "1")

    runtime_env = _runtime_env()

    assert (
        runtime_env["py_executable"]
        == "/vePFS-Mindverse/share/mint/dev/runtime/sglang-overlays/sglang-0.5.12.post1-cu129-venv/bin/python"
    )
    assert runtime_env["env_vars"]["MINT_SGLANG_PY_EXECUTABLE"] == runtime_env["py_executable"]
    assert runtime_env["env_vars"]["MINT_SGLANG_DISABLE_FLASHINFER_KERNELS"] == "1"
    assert runtime_env["env_vars"]["PYTHONPATH"] == "/runtime/sglang-overlay:/runtime/site-packages"
    assert runtime_env["env_vars"]["PYTHONPYCACHEPREFIX"].startswith("/tmp/mint_sglang_pycache/")
    assert runtime_env["env_vars"]["MINT_SGLANG_PYCACHE_PREFIX"] == runtime_env["env_vars"]["PYTHONPYCACHEPREFIX"]


def test_sglang_pythonpath_with_overlay_excludes_training_source_trees(monkeypatch) -> None:
    class _Layout:
        site_packages = "/runtime/gpu_rl/site-packages"

    monkeypatch.setattr("mint_server.backend.sglang_engine.PFS_PYTHONPATH", "/wide/runtime")
    monkeypatch.setattr("mint_server.backend.sglang_engine.PFS_RUNTIME_ENV_ROOT", "/runtime")
    monkeypatch.setattr("mint_server.backend.sglang_engine.MINT_CODE_ROOT", "/code/mint")
    monkeypatch.setattr("mint_server.backend.sglang_engine.PFS_HF_MODULES_PATH", "/hf/modules")
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.runtime_env_layout",
        lambda env_root, *, tier: _Layout(),
    )
    monkeypatch.setenv("MINT_SGLANG_PYTHONPATH", "/runtime/sglang-overlay")

    pythonpath = _sglang_pythonpath()

    assert pythonpath == "/runtime/sglang-overlay:/runtime/gpu_rl/site-packages:/code/mint:/hf/modules"
    assert "Megatron-LM" not in pythonpath
    assert "/src/verl" not in pythonpath
    assert "/src/vllm" not in pythonpath


def test_sglang_actor_import_boundary_drops_inherited_training_source_trees(monkeypatch) -> None:
    original_sys_path = list(sys.path)
    try:
        sys.path[:] = [
            "/driver",
            "/runtime/src/Megatron-LM",
            "/runtime/src/Megatron-Bridge/src",
            "/runtime/src/verl",
            "/runtime/src/vllm",
            "/runtime/gpu_rl/site-packages",
        ]
        monkeypatch.setenv(
            "PYTHONPATH",
            os.pathsep.join(
                [
                    "/runtime/src/Megatron-LM",
                    "/runtime/src/Megatron-Bridge",
                    "/runtime/src/verl",
                    "/runtime/src/vllm",
                    "/runtime/gpu_rl/site-packages",
                ]
            ),
        )
        monkeypatch.setenv("MINT_SGLANG_PYTHONPATH", "/runtime/sglang-overlay")

        result = _apply_sglang_import_boundary()

        assert result["removed_sys_path"] == 4
        assert result["removed_pythonpath"] == 4
        assert sys.path[:3] == [
            "/runtime/sglang-overlay",
            "/driver",
            "/runtime/gpu_rl/site-packages",
        ]
        assert "Megatron-LM" not in os.environ["PYTHONPATH"]
        assert "/src/verl" not in os.environ["PYTHONPATH"]
        assert "/src/vllm" not in os.environ["PYTHONPATH"]
        assert os.environ["PYTHONPATH"] == "/runtime/gpu_rl/site-packages"
    finally:
        sys.path[:] = original_sys_path


def test_sglang_runtime_env_respects_explicit_pycache_prefix(monkeypatch) -> None:
    monkeypatch.setattr("mint_server.backend.sglang_engine.PFS_PYTHONPATH", "/runtime/site-packages")
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.actor_runtime_env_vars",
        lambda *, pythonpath, extra=None: {"PYTHONPATH": pythonpath, **dict(extra or {})},
    )
    monkeypatch.setattr("mint_server.backend.sglang_engine.otel_env_vars", lambda: {})
    monkeypatch.setenv("MINT_SGLANG_PYCACHE_PREFIX", "/local/sglang-pycache")

    runtime_env = _runtime_env()

    assert _sglang_pycache_prefix() == "/local/sglang-pycache"
    assert runtime_env["env_vars"]["MINT_SGLANG_PYCACHE_PREFIX"] == "/local/sglang-pycache"
    assert runtime_env["env_vars"]["PYTHONPYCACHEPREFIX"] == "/local/sglang-pycache"


def test_sglang_runtime_env_can_disable_pycache_prefix(monkeypatch) -> None:
    monkeypatch.setattr("mint_server.backend.sglang_engine.PFS_PYTHONPATH", "/runtime/site-packages")
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.actor_runtime_env_vars",
        lambda *, pythonpath, extra=None: {"PYTHONPATH": pythonpath, **dict(extra or {})},
    )
    monkeypatch.setattr("mint_server.backend.sglang_engine.otel_env_vars", lambda: {})
    monkeypatch.setenv("MINT_SGLANG_PYCACHE_PREFIX", "")

    runtime_env = _runtime_env()

    assert _sglang_pycache_prefix() is None
    assert runtime_env["env_vars"]["MINT_SGLANG_PYCACHE_PREFIX"] == ""
    assert "PYTHONPYCACHEPREFIX" not in runtime_env["env_vars"]


def test_sglang_actor_lora_engine_kwargs_are_configurable(monkeypatch) -> None:
    captured = {}

    class _Engine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sglang.srt.entrypoints.engine" and "Engine" in fromlist:
            class _Module:
                Engine = _Engine

            return _Module()
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setenv("MINT_SGLANG_MAX_LORA_RANK", "16")
    monkeypatch.setenv("MINT_SGLANG_MAX_LOADED_LORAS", "4")
    monkeypatch.setenv("MINT_SGLANG_MAX_LORAS_PER_BATCH", "2")
    monkeypatch.setenv("MINT_SGLANG_LORA_BACKEND", "csgmv")
    monkeypatch.setenv("MINT_SGLANG_LORA_TARGET_MODULES", "q_proj,v_proj")

    actor = SGLangEngineActor(
        model_name="Qwen/Qwen3-0.6B",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_0_6b",
        max_model_len=512,
        default_max_lora_rank=32,
        default_max_loaded_loras=6,
    )
    actor.initialize()

    assert captured["enable_lora"] is True
    assert captured["max_lora_rank"] == 16
    assert captured["max_loaded_loras"] == 4
    assert captured["max_loras_per_batch"] == 2
    assert captured["lora_backend"] == "csgmv"
    assert captured["lora_target_modules"] == ["q_proj", "v_proj"]


def test_sglang_actor_lora_engine_kwargs_use_constructor_defaults(monkeypatch) -> None:
    captured = {}

    class _Engine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sglang.srt.entrypoints.engine" and "Engine" in fromlist:
            class _Module:
                Engine = _Engine

            return _Module()
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delenv("MINT_SGLANG_MAX_LOADED_LORAS", raising=False)
    monkeypatch.delenv("MINT_SGLANG_MAX_LORAS_PER_BATCH", raising=False)
    monkeypatch.delenv("MINT_SGLANG_LORA_TARGET_MODULES", raising=False)

    actor = SGLangEngineActor(
        model_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_235b_a22b_instruct_2507",
        max_model_len=512,
        default_max_lora_rank=64,
        default_max_loaded_loras=1,
        default_max_loras_per_batch=1,
        default_lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    )
    actor.initialize()

    assert captured["enable_lora"] is True
    assert captured["max_lora_rank"] == 64
    assert captured["max_loaded_loras"] == 1
    assert captured["max_loras_per_batch"] == 1
    assert captured["lora_target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_sglang_actor_import_failure_logs_actionable_fields(monkeypatch, caplog) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sglang.srt.entrypoints.engine" and "Engine" in fromlist:
            raise ModuleNotFoundError("No module named 'sglang'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    caplog.set_level("WARNING", logger="mint_server.backend.sglang_actor")

    actor = SGLangEngineActor(
        model_name="Qwen/Qwen3-0.6B",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_0_6b",
        max_model_len=512,
    )

    with pytest.raises(RuntimeError, match="SGLang import failed"):
        actor.initialize()

    assert actor.health()["ready"] is False
    assert "ModuleNotFoundError" in str(actor.health()["last_error"])
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "SGLang import failed backend=sglang" in messages
    assert "actor=mint_sglang_qwen3_0_6b" in messages
    assert "model=Qwen/Qwen3-0.6B" in messages
    assert "error_type=ModuleNotFoundError" in messages


def test_sglang_actor_engine_init_failure_logs_nonsecret_config(monkeypatch, caplog) -> None:
    class _Engine:
        def __init__(self, **_kwargs):
            raise RuntimeError("synthetic init failure")

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sglang.srt.entrypoints.engine" and "Engine" in fromlist:
            class _Module:
                Engine = _Engine

            return _Module()
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setenv("MINT_SGLANG_MAX_RUNNING_REQUESTS", "7")
    caplog.set_level("WARNING", logger="mint_server.backend.sglang_actor")

    actor = SGLangEngineActor(
        model_name="Qwen/Qwen3-0.6B",
        model_path="/tmp/secret-ish/model",
        actor_name="mint_sglang_qwen3_0_6b",
        max_model_len=512,
        dtype="bfloat16",
    )

    with pytest.raises(RuntimeError, match="SGLang engine initialization failed"):
        actor.initialize()

    assert actor.health()["ready"] is False
    assert "RuntimeError" in str(actor.health()["last_error"])
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "SGLang engine initialization failed backend=sglang" in messages
    assert "actor=mint_sglang_qwen3_0_6b" in messages
    assert "model=Qwen/Qwen3-0.6B" in messages
    assert "error_type=RuntimeError" in messages
    assert "'context_length': 512" in messages
    assert "'max_running_requests': 7" in messages
    assert "'dtype': 'bfloat16'" in messages
    assert "model_path" not in messages
    assert "/tmp/secret-ish/model" not in messages


def test_sglang_actor_lora_defaults_follow_model_config() -> None:
    cfg = ModelConfig(
        num_parameters=1.0,
        is_moe=False,
        inference_tp=1,
        inference_dp=1,
        max_model_len=512,
        max_loras=3,
        max_lora_rank=16,
    )

    assert _sglang_actor_lora_defaults_for_config(cfg) == {
        "default_enable_lora": True,
        "default_max_lora_rank": 16,
        "default_max_loaded_loras": 3,
    }


def test_sglang_actor_lora_defaults_use_sglang_specific_model_config() -> None:
    cfg = ModelConfig(
        num_parameters=235.0,
        is_moe=True,
        inference_tp=16,
        inference_dp=1,
        max_model_len=512,
        max_loras=8,
        max_lora_rank=64,
        sglang_max_loaded_loras=1,
        sglang_max_lora_rank=64,
        sglang_lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    assert _sglang_actor_lora_defaults_for_config(cfg) == {
        "default_enable_lora": True,
        "default_max_lora_rank": 64,
        "default_max_loaded_loras": 1,
        "default_max_loras_per_batch": 1,
        "default_lora_target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),
    }


def test_qwen3_235b_sglang_lora_defaults_enable_moe_mlp_with_low_adapter_capacity(monkeypatch) -> None:
    monkeypatch.delenv("MINT_MODEL_CONFIG_OVERRIDES_JSON", raising=False)
    cfg = get_model_config("Qwen/Qwen3-235B-A22B-Instruct-2507")

    assert cfg.max_loras == 8
    assert cfg.sglang_max_loaded_loras == 1
    assert cfg.sglang_max_lora_rank == 64
    assert cfg.sglang_lora_target_modules == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def test_sglang_actor_lora_defaults_respect_disabled_model_config() -> None:
    cfg = ModelConfig(
        num_parameters=1.0,
        is_moe=False,
        inference_tp=1,
        inference_dp=1,
        max_model_len=512,
        max_loras=0,
        max_lora_rank=16,
    )

    assert _sglang_actor_lora_defaults_for_config(cfg) == {
        "default_enable_lora": False,
        "default_max_lora_rank": 16,
        "default_max_loaded_loras": 1,
    }


def test_sglang_actor_memory_defaults_follow_model_config_gpu_utilization() -> None:
    cfg = ModelConfig(
        num_parameters=30.0,
        is_moe=True,
        inference_tp=4,
        inference_dp=1,
        max_model_len=512,
        gpu_memory_utilization=0.85,
    )

    assert _sglang_actor_memory_defaults_for_config(cfg) == {"default_mem_fraction_static": 0.85}


def test_sglang_actor_memory_defaults_fall_back_to_phase_0_probe_value() -> None:
    cfg = ModelConfig(
        num_parameters=1.0,
        is_moe=False,
        inference_tp=1,
        inference_dp=1,
        max_model_len=512,
    )

    assert _sglang_actor_memory_defaults_for_config(cfg) == {"default_mem_fraction_static": 0.4}


def test_sglang_engine_forwards_lora_session_id_to_actor(monkeypatch) -> None:
    engine = SGLangInferenceEngine(
        model_name="Qwen/Qwen3-0.6B",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_0_6b",
        tensor_parallel_size=1,
        max_model_len=16,
    )

    async def fake_initialize():
        engine._initialized = True

    async def fake_dispatch():
        return {"token_ids": [9], "logprobs": [-0.9], "stop_reason": "length"}

    monkeypatch.setattr(engine, "initialize", fake_initialize)
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.ray_get_with_model_actor_supervisor_keepalive",
        lambda *args, **kwargs: fake_dispatch(),
    )

    seen = {}

    class _Remote:
        def remote(self, **kwargs):
            seen.update(kwargs)
            return object()

    class _Server:
        generate_base = _Remote()

    engine.server = _Server()

    async def run() -> None:
        result = await engine.generate(
            sampling_session_id="session-with-lora",
            prompt_ids=[1, 2],
            request_id="req",
            max_tokens=1,
        )
        assert result.token_ids == [9]
        assert seen["sampling_session_id"] == "session-with-lora"

    asyncio.run(run())


def test_sglang_engine_checks_context_length_before_initializing(monkeypatch) -> None:
    engine = SGLangInferenceEngine(
        model_name="Qwen/Qwen3-0.6B",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_0_6b",
        tensor_parallel_size=1,
        max_model_len=2,
    )

    async def should_not_initialize():
        raise AssertionError("Context length rejection should happen before engine initialization")

    monkeypatch.setattr(engine, "initialize", should_not_initialize)

    async def run() -> None:
        with pytest.raises(ValueError, match="exceeds max_model_len"):
            await engine.generate(
                sampling_session_id="__base__",
                prompt_ids=[1, 2],
                request_id="req",
                max_tokens=1,
            )

    asyncio.run(run())


def test_sglang_engine_builds_base_generate_result_with_fake_server(monkeypatch) -> None:
    engine = SGLangInferenceEngine(
        model_name="Qwen/Qwen3-0.6B",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_0_6b",
        tensor_parallel_size=1,
        max_model_len=8,
    )

    async def fake_initialize():
        engine._initialized = True

    async def fake_dispatch():
        return {"token_ids": [7, 8], "logprobs": [-0.7, -0.8], "stop_reason": "length", "_timing_total_s": 0.5}

    monkeypatch.setattr(engine, "initialize", fake_initialize)
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.ray_get_with_model_actor_supervisor_keepalive",
        lambda *args, **kwargs: fake_dispatch(),
    )

    class _Remote:
        def remote(self, **kwargs):
            assert kwargs["prompt_ids"] == [1, 2]
            assert kwargs["max_tokens"] == 2
            assert kwargs["temperature"] == 0.0
            assert kwargs["top_p"] == 1.0
            assert kwargs["logprobs"] is True
            assert kwargs["sampling_session_id"] == "__base__"
            return object()

    class _Server:
        generate_base = _Remote()

    engine.server = _Server()

    async def run() -> None:
        result = await engine.generate(
            sampling_session_id="__base__",
            prompt_ids=[1, 2],
            request_id="req",
            max_tokens=2,
            temperature=0.0,
            top_p=1.0,
        )
        assert result.token_ids == [7, 8]
        assert result.logprobs == [-0.7, -0.8]
        assert result.stop_reason == "length"
        assert result.timing_total_s == 0.5

    asyncio.run(run())


def test_sglang_engine_generate_timeout_aborts_actor_request(monkeypatch, caplog) -> None:
    engine = SGLangInferenceEngine(
        model_name="Qwen/Qwen3-0.6B",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_0_6b",
        tensor_parallel_size=1,
        max_model_len=8,
    )

    async def fake_initialize():
        engine._initialized = True

    async def fake_wait(_ref, **kwargs):
        assert kwargs["timeout_s"] == 0.01
        raise asyncio.TimeoutError("synthetic timeout")

    aborts: list[str] = []

    class _Remote:
        def __init__(self, fn):
            self._fn = fn

        def remote(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

    class _Server:
        generate_base = _Remote(lambda **_kwargs: object())
        abort = _Remote(lambda request_id: aborts.append(request_id) or True)

    async def fake_async_get(ref, *, timeout_s=None):
        assert timeout_s == 5
        return ref

    monkeypatch.setenv("MINT_SGLANG_GENERATE_TIMEOUT_S", "0.01")
    monkeypatch.setattr(engine, "initialize", fake_initialize)
    monkeypatch.setattr("mint_server.backend.sglang_engine.async_get_ray_ref", fake_async_get)
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.ray_get_with_model_actor_supervisor_keepalive",
        fake_wait,
    )
    engine.server = _Server()
    caplog.set_level("WARNING", logger="mint_server.backend.sglang_engine")

    async def run() -> None:
        with pytest.raises(RuntimeError, match="SGLang generate timed out"):
            await engine.generate(
                sampling_session_id="__base__",
                prompt_ids=[1, 2],
                request_id="req-timeout",
                max_tokens=1,
            )
        assert aborts == ["req-timeout"]

    asyncio.run(run())
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "SGLang request timed out backend=sglang" in messages
    assert "model=Qwen/Qwen3-0.6B" in messages
    assert "actor=mint_sglang_qwen3_0_6b" in messages
    assert "request_id=req-timeout" in messages
    assert "timeout_config=MINT_SGLANG_GENERATE_TIMEOUT_S" in messages


def test_sglang_engine_timeout_log_identifies_generic_timeout_fallback(monkeypatch, caplog) -> None:
    engine = SGLangInferenceEngine(
        model_name="Qwen/Qwen3-0.6B",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_0_6b",
        tensor_parallel_size=1,
        max_model_len=8,
    )

    async def fake_initialize():
        engine._initialized = True

    async def fake_wait(_ref, **kwargs):
        assert kwargs["timeout_s"] == 0.02
        raise asyncio.TimeoutError("synthetic timeout")

    aborts: list[str] = []

    class _Remote:
        def __init__(self, fn):
            self._fn = fn

        def remote(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

    class _Server:
        generate_base = _Remote(lambda **_kwargs: object())
        abort = _Remote(lambda request_id: aborts.append(request_id) or True)

    async def fake_async_get(ref, *, timeout_s=None):
        return ref

    monkeypatch.delenv("MINT_SGLANG_GENERATE_TIMEOUT_S", raising=False)
    monkeypatch.setenv("MINT_SGLANG_REQUEST_TIMEOUT_S", "0.02")
    monkeypatch.setattr(engine, "initialize", fake_initialize)
    monkeypatch.setattr("mint_server.backend.sglang_engine.async_get_ray_ref", fake_async_get)
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.ray_get_with_model_actor_supervisor_keepalive",
        fake_wait,
    )
    engine.server = _Server()
    caplog.set_level("WARNING", logger="mint_server.backend.sglang_engine")

    async def run() -> None:
        with pytest.raises(RuntimeError, match="SGLang generate timed out"):
            await engine.generate(
                sampling_session_id="__base__",
                prompt_ids=[1, 2],
                request_id="req-timeout-generic",
                max_tokens=1,
            )
        assert aborts == ["req-timeout-generic"]

    asyncio.run(run())
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "request_id=req-timeout-generic" in messages
    assert "timeout_config=MINT_SGLANG_REQUEST_TIMEOUT_S" in messages


def test_sglang_engine_generate_slow_request_logs_backend_fields(monkeypatch, caplog) -> None:
    engine = SGLangInferenceEngine(
        model_name="Qwen/Qwen3-0.6B",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_0_6b",
        tensor_parallel_size=1,
        max_model_len=8,
    )

    async def fake_initialize():
        engine._initialized = True

    async def fake_wait(_ref, **_kwargs):
        return {"token_ids": [7], "logprobs": [-0.7], "stop_reason": "length"}

    class _Remote:
        def remote(self, **_kwargs):
            return object()

    class _Server:
        generate_base = _Remote()

    monkeypatch.setenv("MINT_SGLANG_SLOW_REQUEST_LOG_S", "0.000001")
    monkeypatch.setattr(engine, "initialize", fake_initialize)
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.ray_get_with_model_actor_supervisor_keepalive",
        fake_wait,
    )
    engine.server = _Server()
    caplog.set_level("WARNING", logger="mint_server.backend.sglang_engine")

    async def run() -> None:
        await engine.generate(
            sampling_session_id="__base__",
            prompt_ids=[1, 2],
            request_id="req-slow",
            max_tokens=1,
        )

    asyncio.run(run())
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "SGLang request slow backend=sglang" in messages
    assert "actor=mint_sglang_qwen3_0_6b" in messages
    assert "request_id=req-slow" in messages


def test_sglang_engine_computes_prompt_logprobs_with_fake_server(monkeypatch) -> None:
    engine = SGLangInferenceEngine(
        model_name="Qwen/Qwen3-0.6B",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_0_6b",
        tensor_parallel_size=1,
        max_model_len=8,
    )

    async def fake_initialize():
        engine._initialized = True

    async def fake_dispatch():
        return [None, -1.25]

    monkeypatch.setattr(engine, "initialize", fake_initialize)
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.ray_get_with_model_actor_supervisor_keepalive",
        lambda *args, **kwargs: fake_dispatch(),
    )

    seen = {}

    class _Remote:
        def remote(self, **kwargs):
            seen.update(kwargs)
            return object()

    class _Server:
        compute_prompt_logprobs = _Remote()

    engine.server = _Server()

    async def run() -> None:
        out = await engine.compute_logprobs(
            sampling_session_id="__base__",
            prompt_ids=[1, 2],
            request_id="req-lp",
        )
        assert out == [None, -1.25]
        assert seen == {
            "prompt_ids": [1, 2],
            "request_id": "req-lp",
            "sampling_session_id": "__base__",
        }

    asyncio.run(run())


def test_sglang_engine_compute_logprobs_uses_backend_timeout(monkeypatch) -> None:
    engine = SGLangInferenceEngine(
        model_name="Qwen/Qwen3-0.6B",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_0_6b",
        tensor_parallel_size=1,
        max_model_len=8,
    )

    async def fake_initialize():
        engine._initialized = True

    seen = {}

    async def fake_wait(_ref, **kwargs):
        seen.update(kwargs)
        return [None, -1.25]

    class _Remote:
        def remote(self, **_kwargs):
            return object()

    class _Server:
        compute_prompt_logprobs = _Remote()

    monkeypatch.setenv("MINT_SGLANG_LOGPROBS_TIMEOUT_S", "12.5")
    monkeypatch.setattr(engine, "initialize", fake_initialize)
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.ray_get_with_model_actor_supervisor_keepalive",
        fake_wait,
    )
    engine.server = _Server()

    async def run() -> None:
        assert await engine.compute_logprobs(
            sampling_session_id="__base__",
            prompt_ids=[1, 2],
            request_id="req-lp-timeout",
        ) == [None, -1.25]
        assert seen["timeout_s"] == 12.5
        assert seen["request_id"] == "req-lp-timeout"

    asyncio.run(run())


def test_sglang_engine_computes_prompt_topk_with_fake_server(monkeypatch) -> None:
    engine = SGLangInferenceEngine(
        model_name="Qwen/Qwen3-0.6B",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_0_6b",
        tensor_parallel_size=1,
        max_model_len=8,
    )

    async def fake_initialize():
        engine._initialized = True

    async def fake_dispatch():
        return [None, [(9, -0.5), (8, -0.7)]]

    monkeypatch.setattr(engine, "initialize", fake_initialize)
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.ray_get_with_model_actor_supervisor_keepalive",
        lambda *args, **kwargs: fake_dispatch(),
    )

    seen = {}

    class _Remote:
        def remote(self, **kwargs):
            seen.update(kwargs)
            return object()

    class _Server:
        compute_prompt_topk = _Remote()

    engine.server = _Server()

    async def run() -> None:
        out = await engine.compute_topk(
            sampling_session_id="session-with-lora",
            prompt_ids=[1, 2],
            request_id="req-topk",
            k=2,
        )
        assert out == [None, [(9, -0.5), (8, -0.7)]]
        assert seen == {
            "prompt_ids": [1, 2],
            "request_id": "req-topk",
            "k": 2,
            "sampling_session_id": "session-with-lora",
        }

    asyncio.run(run())


def test_sglang_engine_rejects_generate_without_fake_server(monkeypatch) -> None:
    engine = SGLangInferenceEngine(
        model_name="Qwen/Qwen3-0.6B",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_0_6b",
        tensor_parallel_size=1,
        max_model_len=8,
    )

    async def fake_initialize():
        engine._initialized = True

    monkeypatch.setattr(engine, "initialize", fake_initialize)

    async def run() -> None:
        with pytest.raises(RuntimeError, match="not available"):
            await engine.generate(
                sampling_session_id="__base__",
                prompt_ids=[1],
                request_id="req",
                max_tokens=1,
            )

    asyncio.run(run())


def _capture_sglang_actor_launch(
    monkeypatch,
    *,
    tensor_parallel_size: int = 1,
    gpu_count: int = 8,
    pinned_node_ip: str | None = None,
    pinned_node_id: str = "node-pinned",
) -> dict[str, object]:
    engine = SGLangInferenceEngine(
        model_name="Qwen/Qwen3-0.6B",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_0_6b",
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=8,
    )

    captured: dict[str, object] = {}
    published: list[object] = []
    capacity_checks: list[dict[str, object]] = []

    class _RemoteMethod:
        def __init__(self, value):
            self._value = value

        def remote(self, *args, **kwargs):
            return self._value

    class _FakeActor:
        initialize = _RemoteMethod({"ready": True})

    class _RemoteClass:
        def options(self, **options):
            captured["options"] = options
            return self

        def remote(self, **kwargs):
            captured["constructor_kwargs"] = kwargs
            return _FakeActor()

    def fake_remote(**remote_kwargs):
        captured["remote_kwargs"] = remote_kwargs

        def _wrap(_cls):
            return _RemoteClass()

        return _wrap

    async def fake_async_get(value, *, timeout_s=None):
        return value

    monkeypatch.setattr("mint_server.backend.sglang_engine.ray.is_initialized", lambda: True)
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.ray.get_actor",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("missing")),
    )
    monkeypatch.setattr("mint_server.backend.sglang_engine.ray.remote", fake_remote)
    monkeypatch.setattr("mint_server.backend.sglang_engine.async_get_ray_ref", fake_async_get)
    monkeypatch.setattr("mint_server.backend.sglang_engine._runtime_env", lambda: {"env_vars": {}})
    monkeypatch.setattr("mint_server.backend.sglang_engine.publish_backend_model_actor", lambda launch: published.append(launch))
    if pinned_node_ip is None:
        monkeypatch.delenv("MINT_SGLANG_MODEL_PLACEMENT_JSON", raising=False)
        monkeypatch.delenv("MINT_MODEL_PLACEMENT_JSON", raising=False)
        _stub_single_node_gpu_capacity(monkeypatch, gpu_count=gpu_count)
    else:
        monkeypatch.setenv(
            "MINT_SGLANG_MODEL_PLACEMENT_JSON",
            json.dumps(
                {
                    "Qwen/Qwen3-0.6B": {
                        "replica": 0,
                        "node_ip": pinned_node_ip,
                        "gpu_count": tensor_parallel_size,
                    }
                }
            ),
        )
        monkeypatch.setenv("MINT_MODEL_PLACEMENT_JSON", "{}")
        monkeypatch.setattr(
            "mint_server.backend.sglang_engine.ray.nodes",
            lambda: [
                {
                    "Alive": True,
                    "NodeID": pinned_node_id,
                    "NodeManagerAddress": pinned_node_ip,
                    "Resources": {"GPU": gpu_count},
                }
            ],
        )

        def fake_assert_node_ip_capacity(**kwargs):
            capacity_checks.append(kwargs)

        class _FakeNodeAffinitySchedulingStrategy:
            def __init__(self, node_id, *, soft):
                self.node_id = node_id
                self.soft = soft

        monkeypatch.setattr("mint_server.backend.sglang_engine.assert_node_ip_capacity", fake_assert_node_ip_capacity)
        monkeypatch.setattr(
            "mint_server.backend.sglang_engine.NodeAffinitySchedulingStrategy",
            _FakeNodeAffinitySchedulingStrategy,
        )

    asyncio.run(engine.initialize())
    captured["published"] = published
    captured["capacity_checks"] = capacity_checks
    return captured


class _SGLangFakeRemoteMethod:
    def __init__(self, value):
        self._value = value

    def remote(self, *args, **kwargs):
        return self._value


class _SGLangExistingActor:
    def __init__(self, *, ready):
        self.is_ready = _SGLangFakeRemoteMethod(ready)
        self.initialize = _SGLangFakeRemoteMethod({"ready": True})


def test_sglang_engine_reuses_existing_ready_actor(monkeypatch) -> None:
    existing = _SGLangExistingActor(ready=True)
    published = []

    monkeypatch.setattr("mint_server.backend.sglang_engine.ray.is_initialized", lambda: True)
    monkeypatch.setattr("mint_server.backend.sglang_engine.ray.get_actor", lambda *args, **kwargs: existing)

    async def fake_async_get(value, **_kwargs):
        return value

    monkeypatch.setattr("mint_server.backend.sglang_engine.async_get_ray_ref", fake_async_get)
    monkeypatch.setattr("mint_server.backend.sglang_engine.publish_backend_model_actor", lambda launch: published.append(launch))
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.ray.remote",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ready existing actor should be reused")),
    )

    engine = SGLangInferenceEngine(
        model_name="Qwen/Qwen3-0.6B",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_0_6b",
        tensor_parallel_size=1,
        max_model_len=8,
    )

    asyncio.run(engine.initialize())

    assert engine.server is existing
    assert engine._initialized is True
    assert len(published) == 1
    assert published[0].actor_name == "mint_sglang_qwen3_0_6b"


def test_sglang_engine_recreates_existing_unready_actor(monkeypatch) -> None:
    existing = _SGLangExistingActor(ready=False)
    killed = []
    invalidated = []
    captured: dict[str, object] = {}

    class _RemoteClass:
        def options(self, **options):
            captured["options"] = options
            return self

        def remote(self, **kwargs):
            captured["constructor_kwargs"] = kwargs
            return _SGLangExistingActor(ready=True)

    def fake_remote(**remote_kwargs):
        captured["remote_kwargs"] = remote_kwargs

        def _wrap(_cls):
            return _RemoteClass()

        return _wrap

    monkeypatch.setattr("mint_server.backend.sglang_engine.ray.is_initialized", lambda: True)
    monkeypatch.setattr("mint_server.backend.sglang_engine.ray.get_actor", lambda *args, **kwargs: existing)

    async def fake_async_get(value, **_kwargs):
        return value

    monkeypatch.setattr("mint_server.backend.sglang_engine.async_get_ray_ref", fake_async_get)
    monkeypatch.setattr("mint_server.backend.sglang_engine.ray.remote", fake_remote)
    monkeypatch.setattr("mint_server.backend.sglang_engine._runtime_env", lambda: {"env_vars": {}})
    monkeypatch.setattr("mint_server.backend.sglang_engine.publish_backend_model_actor", lambda _launch: None)
    _stub_single_node_gpu_capacity(monkeypatch)
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine._invalidate_model_session_loras",
        lambda model: invalidated.append(model),
    )
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.ray_kill.kill",
        lambda actor, **kwargs: killed.append((actor, kwargs)),
    )

    engine = SGLangInferenceEngine(
        model_name="Qwen/Qwen3-0.6B",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_0_6b",
        tensor_parallel_size=1,
        max_model_len=8,
    )

    asyncio.run(engine.initialize())

    assert invalidated == ["Qwen/Qwen3-0.6B"]
    assert killed == [
        (
            existing,
            {
                "reason": "sglang_engine_not_ready",
                "actor_name": "mint_sglang_qwen3_0_6b",
                "namespace": PERSISTENT_NAMESPACE,
                "no_restart": True,
                "verify_absent": True,
            },
        )
    ]
    assert captured["remote_kwargs"]["num_gpus"] == 1
    assert captured["options"]["name"] == "mint_sglang_qwen3_0_6b"


def test_sglang_engine_verifies_actor_absent_after_init_failure(monkeypatch) -> None:
    killed = []

    class _RemoteMethod:
        def remote(self):
            return object()

    class _FakeActor:
        initialize = _RemoteMethod()

    class _RemoteClass:
        def options(self, **_options):
            return self

        def remote(self, **_kwargs):
            return _FakeActor()

    def fake_remote(**_remote_kwargs):
        def _wrap(_cls):
            return _RemoteClass()

        return _wrap

    async def fake_async_get(_value, **_kwargs):
        raise RuntimeError("init failed")

    monkeypatch.setattr("mint_server.backend.sglang_engine.ray.is_initialized", lambda: True)
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.ray.get_actor",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("missing")),
    )
    monkeypatch.setattr("mint_server.backend.sglang_engine.ray.remote", fake_remote)
    monkeypatch.setattr("mint_server.backend.sglang_engine.async_get_ray_ref", fake_async_get)
    monkeypatch.setattr("mint_server.backend.sglang_engine._runtime_env", lambda: {"env_vars": {}})
    _stub_single_node_gpu_capacity(monkeypatch)
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.ray_kill.kill",
        lambda actor, **kwargs: killed.append((actor, kwargs)),
    )

    engine = SGLangInferenceEngine(
        model_name="Qwen/Qwen3-0.6B",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_0_6b",
        tensor_parallel_size=1,
        max_model_len=8,
    )

    with pytest.raises(RuntimeError, match="init failed"):
        asyncio.run(engine.initialize())

    assert engine.server is None
    assert killed
    assert killed[0][1] == {
        "reason": "sglang_init_failed",
        "actor_name": "mint_sglang_qwen3_0_6b",
        "namespace": PERSISTENT_NAMESPACE,
        "no_restart": True,
        "verify_absent": True,
    }


def test_sglang_engine_reuses_existing_actor_when_readiness_probe_times_out(monkeypatch, caplog) -> None:
    existing = _SGLangExistingActor(ready=object())
    published = []

    async def raise_timeout(_value, **_kwargs):
        raise ray.exceptions.GetTimeoutError()

    monkeypatch.setattr("mint_server.backend.sglang_engine.ray.is_initialized", lambda: True)
    monkeypatch.setattr("mint_server.backend.sglang_engine.ray.get_actor", lambda *args, **kwargs: existing)
    monkeypatch.setattr("mint_server.backend.sglang_engine.async_get_ray_ref", raise_timeout)
    monkeypatch.setattr("mint_server.backend.sglang_engine.publish_backend_model_actor", lambda launch: published.append(launch))
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.ray.remote",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("timeout existing actor should be reused")),
    )
    caplog.set_level("WARNING", logger="mint_server.backend.sglang_engine")

    engine = SGLangInferenceEngine(
        model_name="Qwen/Qwen3-0.6B",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_0_6b",
        tensor_parallel_size=1,
        max_model_len=8,
    )

    asyncio.run(engine.initialize())

    assert engine.server is existing
    assert engine._initialized is True
    assert len(published) == 1
    assert "readiness timed out; assuming busy and reusing" in "\n".join(
        record.getMessage() for record in caplog.records
    )


def test_sglang_engine_actor_launch_defaults_to_concurrent_actor(monkeypatch) -> None:
    monkeypatch.delenv("MINT_SGLANG_ACTOR_MAX_CONCURRENCY", raising=False)

    captured = _capture_sglang_actor_launch(monkeypatch)

    # Default >1 so the actor serves concurrent sampling requests and SGLang
    # batches them internally; adapter mutations stay safe via the actor's
    # writer-preferring RW lock.
    assert captured["remote_kwargs"]["max_concurrency"] == 64
    assert captured["options"]["name"] == "mint_sglang_qwen3_0_6b"


def test_sglang_engine_actor_launch_allows_explicit_concurrency_override(monkeypatch) -> None:
    monkeypatch.setenv("MINT_SGLANG_ACTOR_MAX_CONCURRENCY", "2")

    captured = _capture_sglang_actor_launch(monkeypatch)

    assert captured["remote_kwargs"]["max_concurrency"] == 2
    assert captured["options"]["name"] == "mint_sglang_qwen3_0_6b"


def test_sglang_engine_actor_launch_uses_single_node_tp_when_node_has_capacity(monkeypatch) -> None:
    captured = _capture_sglang_actor_launch(monkeypatch, tensor_parallel_size=4, gpu_count=8)

    assert captured["remote_kwargs"]["num_gpus"] == 4
    assert captured["constructor_kwargs"]["tp_size"] == 4
    assert captured["constructor_kwargs"]["model_name"] == "Qwen/Qwen3-0.6B"
    assert "resources" not in captured["options"]
    assert "scheduling_strategy" not in captured["options"]
    published = captured["published"]
    assert len(published) == 1
    assert published[0].num_gpus == 4


def test_sglang_engine_actor_launch_uses_pinned_node_affinity_when_configured(monkeypatch) -> None:
    captured = _capture_sglang_actor_launch(
        monkeypatch,
        tensor_parallel_size=4,
        gpu_count=8,
        pinned_node_ip="192.0.2.44",
        pinned_node_id="node-44",
    )

    assert captured["remote_kwargs"]["num_gpus"] == 4
    assert captured["constructor_kwargs"]["tp_size"] == 4
    assert captured["capacity_checks"] == [
        {
            "required_gpus_by_node_ip": {"192.0.2.44": 4},
            "context": "single_node_sglang model='Qwen/Qwen3-0.6B' actor='mint_sglang_qwen3_0_6b'_pin",
        }
    ]
    assert captured["options"]["resources"] == {"node:192.0.2.44": 0.001}
    strategy = captured["options"]["scheduling_strategy"]
    assert strategy.node_id == "node-44"
    assert strategy.soft is False
    published = captured["published"]
    assert len(published) == 1
    assert published[0].num_gpus == 4


def test_sglang_engine_ready_wait_scales_for_large_moe_models() -> None:
    cfg = ModelConfig(
        num_parameters=30.0,
        is_moe=True,
        inference_tp=4,
        inference_dp=1,
        max_model_len=32768,
    )

    assert _default_sglang_engine_ready_wait_s(cfg, total_gpus=4) == 1800.0


def test_sglang_engine_ready_wait_scales_for_235b_models() -> None:
    cfg = ModelConfig(
        num_parameters=235.0,
        is_moe=True,
        inference_tp=16,
        inference_dp=1,
        max_model_len=32768,
    )

    assert _default_sglang_engine_ready_wait_s(cfg, total_gpus=16) == 3600.0


def test_sglang_engine_ready_wait_preserves_small_model_default() -> None:
    cfg = ModelConfig(
        num_parameters=0.6,
        is_moe=False,
        inference_tp=1,
        inference_dp=1,
        max_model_len=32768,
    )

    assert _default_sglang_engine_ready_wait_s(cfg, total_gpus=1) == 900.0


def test_sglang_engine_actor_launch_uses_scaled_ready_wait_for_30b(monkeypatch) -> None:
    captured_timeouts: list[float | None] = []
    cfg = ModelConfig(
        num_parameters=30.0,
        is_moe=True,
        inference_tp=4,
        inference_dp=1,
        max_model_len=32768,
    )

    engine = SGLangInferenceEngine(
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_30b_a3b_instruct_2507",
        tensor_parallel_size=4,
        max_model_len=32768,
    )
    engine.config = cfg

    class _RemoteMethod:
        def __init__(self, value):
            self._value = value

        def remote(self, *args, **kwargs):
            return self._value

    class _FakeActor:
        initialize = _RemoteMethod({"ready": True})

    class _RemoteClass:
        def options(self, **_options):
            return self

        def remote(self, **_kwargs):
            return _FakeActor()

    def fake_remote(**_remote_kwargs):
        def _wrap(_cls):
            return _RemoteClass()

        return _wrap

    async def fake_async_get(value, *, timeout_s=None):
        captured_timeouts.append(timeout_s)
        return value

    monkeypatch.delenv("MINT_SGLANG_ENGINE_READY_WAIT_S", raising=False)
    monkeypatch.setattr("mint_server.backend.sglang_engine.ray.is_initialized", lambda: True)
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.ray.get_actor",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("missing")),
    )
    monkeypatch.setattr("mint_server.backend.sglang_engine.ray.remote", fake_remote)
    monkeypatch.setattr("mint_server.backend.sglang_engine.async_get_ray_ref", fake_async_get)
    monkeypatch.setattr("mint_server.backend.sglang_engine._runtime_env", lambda: {"env_vars": {}})
    monkeypatch.setattr("mint_server.backend.sglang_engine.publish_backend_model_actor", lambda _launch: None)
    _stub_single_node_gpu_capacity(monkeypatch, gpu_count=8)

    asyncio.run(engine.initialize())

    assert captured_timeouts == [1800.0]


def test_sglang_engine_actor_launch_respects_explicit_ready_wait_override(monkeypatch) -> None:
    captured_timeouts: list[float | None] = []

    class _RemoteMethod:
        def __init__(self, value):
            self._value = value

        def remote(self, *args, **kwargs):
            return self._value

    class _FakeActor:
        initialize = _RemoteMethod({"ready": True})

    class _RemoteClass:
        def options(self, **_options):
            return self

        def remote(self, **_kwargs):
            return _FakeActor()

    def fake_remote(**_remote_kwargs):
        def _wrap(_cls):
            return _RemoteClass()

        return _wrap

    async def fake_async_get(value, *, timeout_s=None):
        captured_timeouts.append(timeout_s)
        return value

    monkeypatch.setenv("MINT_SGLANG_ENGINE_READY_WAIT_S", "42")
    monkeypatch.setattr("mint_server.backend.sglang_engine.ray.is_initialized", lambda: True)
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.ray.get_actor",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("missing")),
    )
    monkeypatch.setattr("mint_server.backend.sglang_engine.ray.remote", fake_remote)
    monkeypatch.setattr("mint_server.backend.sglang_engine.async_get_ray_ref", fake_async_get)
    monkeypatch.setattr("mint_server.backend.sglang_engine._runtime_env", lambda: {"env_vars": {}})
    monkeypatch.setattr("mint_server.backend.sglang_engine.publish_backend_model_actor", lambda _launch: None)
    _stub_single_node_gpu_capacity(monkeypatch)

    engine = SGLangInferenceEngine(
        model_name="Qwen/Qwen3-0.6B",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_0_6b",
        tensor_parallel_size=1,
        max_model_len=8,
    )

    asyncio.run(engine.initialize())

    assert captured_timeouts == [42.0]


def test_sglang_capacity_check_passes_through_when_node_has_room(monkeypatch) -> None:
    capacity_calls: list[dict[str, object]] = []

    def fake_assert_node_ip_capacity(**kwargs):
        capacity_calls.append(kwargs)

    monkeypatch.setattr("mint_server.backend.sglang_engine.assert_node_ip_capacity", fake_assert_node_ip_capacity)

    _assert_sglang_node_ip_capacity(
        model_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
        required_gpus_by_node_ip={"192.0.2.10": 8, "192.0.2.11": 8},
        context="multinode_sglang model='Qwen/Qwen3-235B-A22B-Instruct-2507'",
    )

    # Capacity is asserted exactly once; the sampler is placed on its own GPUs
    # with no eviction of any same-model trainer.
    assert capacity_calls == [
        {
            "required_gpus_by_node_ip": {"192.0.2.10": 8, "192.0.2.11": 8},
            "context": "multinode_sglang model='Qwen/Qwen3-235B-A22B-Instruct-2507'",
        }
    ]


def test_sglang_capacity_block_fails_closed_without_reclaiming_trainer(monkeypatch) -> None:
    capacity_calls: list[dict[str, object]] = []

    def fake_assert_node_ip_capacity(**kwargs):
        capacity_calls.append(kwargs)
        raise RuntimeError("pinned node capacity check failed: training placement group blocks node")

    # No reclaim helper exists anymore; assert we never kill an actor either.
    def fail_if_killed(*args, **kwargs):
        raise AssertionError("SGLang launch must not kill any actor on capacity block")

    monkeypatch.setattr("mint_server.backend.sglang_engine.assert_node_ip_capacity", fake_assert_node_ip_capacity)
    monkeypatch.setattr("mint_server.backend.sglang_engine.ray_kill.kill", fail_if_killed)

    with pytest.raises(RuntimeError) as excinfo:
        _assert_sglang_node_ip_capacity(
            model_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            required_gpus_by_node_ip={"192.0.2.10": 8, "192.0.2.11": 8},
            context="multinode_sglang model='Qwen/Qwen3-235B-A22B-Instruct-2507'",
        )

    message = str(excinfo.value)
    assert "insufficient GPU capacity" in message
    assert "does not preempt live training actors" in message
    # Capacity is checked once and the block is surfaced immediately (no retry-after-kill).
    assert len(capacity_calls) == 1


def test_sglang_engine_actor_launch_uses_explicit_multinode_rank_actors(monkeypatch) -> None:
    cfg = ModelConfig(
        num_parameters=235.0,
        is_moe=True,
        inference_tp=16,
        inference_pp=1,
        inference_dp=1,
        max_model_len=4096,
        serving_backend="sglang",
    )
    engine = SGLangInferenceEngine(
        model_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_235b_a22b_instruct_2507",
        max_model_len=4096,
    )
    monkeypatch.setattr("mint_server.backend.sglang_engine.get_model_config", lambda _model: cfg)
    engine.config = cfg
    engine.tensor_parallel_size = 16
    monkeypatch.setenv(
        "MINT_SGLANG_MODEL_PLACEMENT_JSON",
        json.dumps(
            {
                "Qwen/Qwen3-235B-A22B-Instruct-2507": [
                    {"replica": 0, "node_ip": "192.0.2.10", "gpu_count": 8},
                    {"replica": 0, "node_ip": "192.0.2.11", "gpu_count": 8},
                ]
            }
        ),
    )
    monkeypatch.setenv("MINT_MODEL_PLACEMENT_JSON", "{}")
    monkeypatch.setenv("MINT_SGLANG_DIST_PORT", "29991")
    monkeypatch.setattr("mint_server.backend.sglang_engine.ray.is_initialized", lambda: True)
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.ray.get_actor",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("missing")),
    )
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.ray.nodes",
        lambda: [
            {
                "Alive": True,
                "NodeID": "node-10",
                "NodeManagerAddress": "192.0.2.10",
                "Resources": {"GPU": 8},
            },
            {
                "Alive": True,
                "NodeID": "node-11",
                "NodeManagerAddress": "192.0.2.11",
                "Resources": {"GPU": 8},
            },
        ],
    )

    launched: list[dict[str, object]] = []
    capacity_checks: list[dict[str, object]] = []
    published: list[object] = []

    class _RemoteMethod:
        def __init__(self, value):
            self._value = value

        def remote(self, *args, **kwargs):
            return self._value

    class _FakeActor:
        def __init__(self, name: str):
            self.name = name
            self.initialize = _RemoteMethod({"ready": True})
            self.is_ready = _RemoteMethod(True)

    class _RemoteClass:
        def __init__(self, remote_kwargs):
            self.remote_kwargs = remote_kwargs
            self.options_kwargs: dict[str, object] = {}

        def options(self, **options):
            self.options_kwargs = options
            return self

        def remote(self, **kwargs):
            actor_name = str(kwargs["actor_name"])
            launched.append(
                {
                    "remote_kwargs": self.remote_kwargs,
                    "options": self.options_kwargs,
                    "constructor_kwargs": kwargs,
                }
            )
            return _FakeActor(actor_name)

    def fake_remote(**remote_kwargs):
        def _wrap(_cls):
            return _RemoteClass(remote_kwargs)

        return _wrap

    async def fake_async_get(value, *, timeout_s=None):
        return value

    class _FakeNodeAffinitySchedulingStrategy:
        def __init__(self, node_id, *, soft):
            self.node_id = node_id
            self.soft = soft

    monkeypatch.setattr("mint_server.backend.sglang_engine.ray.remote", fake_remote)
    monkeypatch.setattr("mint_server.backend.sglang_engine.async_get_ray_ref", fake_async_get)
    monkeypatch.setattr("mint_server.backend.sglang_engine._runtime_env", lambda: {"env_vars": {"BASE": "1"}})
    monkeypatch.setattr("mint_server.backend.sglang_engine.assert_node_ip_capacity", lambda **kwargs: capacity_checks.append(kwargs))
    monkeypatch.setattr("mint_server.backend.sglang_engine.publish_backend_model_actor", lambda launch: published.append(launch))
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.NodeAffinitySchedulingStrategy",
        _FakeNodeAffinitySchedulingStrategy,
    )

    asyncio.run(engine.initialize())

    assert len(launched) == 2
    assert capacity_checks == [
        {
            "required_gpus_by_node_ip": {"192.0.2.10": 8, "192.0.2.11": 8},
            "context": "multinode_sglang model='Qwen/Qwen3-235B-A22B-Instruct-2507' actor='mint_sglang_qwen3_235b_a22b_instruct_2507'",
        }
    ]
    rank0 = launched[0]
    rank1 = launched[1]
    assert rank0["remote_kwargs"]["num_gpus"] == 8
    assert rank1["remote_kwargs"]["num_gpus"] == 8
    assert rank0["options"]["name"] == "mint_sglang_qwen3_235b_a22b_instruct_2507"
    assert rank1["options"]["name"] == "mint_sglang_qwen3_235b_a22b_instruct_2507_rank1"
    assert rank0["options"]["resources"] == {"node:192.0.2.10": 0.001}
    assert rank1["options"]["resources"] == {"node:192.0.2.11": 0.001}
    assert rank0["options"]["scheduling_strategy"].node_id == "node-10"
    assert rank1["options"]["scheduling_strategy"].node_id == "node-11"
    assert rank0["options"]["runtime_env"]["env_vars"]["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] == "0"
    assert rank1["options"]["runtime_env"]["env_vars"]["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] == "0"
    assert rank0["constructor_kwargs"]["tp_size"] == 16
    assert rank1["constructor_kwargs"]["tp_size"] == 16
    assert rank0["constructor_kwargs"]["engine_kwargs"] == {
        "nnodes": 2,
        "node_rank": 0,
        "dist_init_addr": "192.0.2.10:29991",
    }
    assert rank1["constructor_kwargs"]["engine_kwargs"] == {
        "nnodes": 2,
        "node_rank": 1,
        "dist_init_addr": "192.0.2.10:29991",
    }
    assert engine.server is not None
    assert engine.server.name == "mint_sglang_qwen3_235b_a22b_instruct_2507"
    assert [name for name, _actor in engine._rank_servers] == ["mint_sglang_qwen3_235b_a22b_instruct_2507_rank1"]
    assert len(published) == 1
    assert published[0].num_gpus == 16
    assert published[0].metadata["placement_mode"] == "multinode"
    assert published[0].metadata["nnodes"] == 2
    assert published[0].metadata["node_ips"] == ["192.0.2.10", "192.0.2.11"]
    assert published[0].metadata["rank_actor_names"] == [
        "mint_sglang_qwen3_235b_a22b_instruct_2507",
        "mint_sglang_qwen3_235b_a22b_instruct_2507_rank1",
    ]


def test_sglang_rank_actor_name_keeps_rank0_as_canonical_actor_name() -> None:
    assert _rank_actor_name("mint_sglang_model", 0) == "mint_sglang_model"
    assert _rank_actor_name("mint_sglang_model", 2) == "mint_sglang_model_rank2"


@pytest.mark.parametrize(
    ("inference_pp", "inference_dp"),
    [
        (2, 1),
        (1, 2),
    ],
)
def test_sglang_engine_rejects_dp_or_pp_before_actor_launch(
    monkeypatch,
    inference_pp: int,
    inference_dp: int,
) -> None:
    cfg = ModelConfig(
        num_parameters=1.0,
        is_moe=False,
        inference_tp=1,
        inference_pp=inference_pp,
        inference_dp=inference_dp,
        max_model_len=512,
        serving_backend="sglang",
    )
    monkeypatch.setattr("mint_server.backend.sglang_engine.get_model_config", lambda _model: cfg)
    monkeypatch.setattr("mint_server.backend.sglang_engine.ray.is_initialized", lambda: True)
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.ray.get_actor",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("missing")),
    )

    def fail_remote(*_args, **_kwargs):
        raise AssertionError("SGLang unsupported DP/PP should fail before actor launch")

    monkeypatch.setattr("mint_server.backend.sglang_engine.ray.remote", fail_remote)

    engine = SGLangInferenceEngine(
        model_name="Qwen/Qwen3-0.6B",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_0_6b",
        max_model_len=8,
    )

    with pytest.raises(RuntimeError, match="single-node TP only"):
        asyncio.run(engine.initialize())


def test_sglang_engine_rejects_tp_larger_than_any_alive_node_before_actor_launch(monkeypatch) -> None:
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.ray.nodes",
        lambda: [
            {
                "Alive": True,
                "NodeManagerAddress": "192.0.2.10",
                "Resources": {"GPU": 8},
            },
            {
                "Alive": True,
                "NodeManagerAddress": "192.0.2.11",
                "Resources": {"GPU": 8},
            },
        ],
    )

    with pytest.raises(RuntimeError, match="required_gpus=16 exceeds max_alive_node_gpus=8"):
        _assert_single_node_sglang_schedulable(
            required_gpus=16,
            context="single_node_sglang model='Qwen/Qwen3-235B-A22B-Instruct-2507'",
        )


def test_sglang_engine_initialize_rejects_cross_node_tp_before_actor_launch(monkeypatch) -> None:
    cfg = ModelConfig(
        num_parameters=235.0,
        is_moe=True,
        inference_tp=16,
        inference_pp=1,
        inference_dp=1,
        max_model_len=4096,
        serving_backend="sglang",
    )
    monkeypatch.setattr("mint_server.backend.sglang_engine.get_model_config", lambda _model: cfg)
    monkeypatch.setattr("mint_server.backend.sglang_engine.ray.is_initialized", lambda: True)
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.ray.get_actor",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("missing")),
    )
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.ray.nodes",
        lambda: [
            {
                "Alive": True,
                "NodeManagerAddress": "192.0.2.10",
                "Resources": {"GPU": 8},
            },
            {
                "Alive": True,
                "NodeManagerAddress": "192.0.2.11",
                "Resources": {"GPU": 8},
            },
        ],
    )
    monkeypatch.delenv("MINT_SGLANG_MODEL_PLACEMENT_JSON", raising=False)
    monkeypatch.delenv("MINT_MODEL_PLACEMENT_JSON", raising=False)

    def fail_remote(*_args, **_kwargs):
        raise AssertionError("SGLang TP cannot span nodes before Phase 6 placement support")

    monkeypatch.setattr("mint_server.backend.sglang_engine.ray.remote", fail_remote)

    engine = SGLangInferenceEngine(
        model_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_235b_a22b_instruct_2507",
        max_model_len=4096,
    )

    with pytest.raises(RuntimeError, match="required_gpus=16 exceeds max_alive_node_gpus=8"):
        asyncio.run(engine.initialize())


def test_sglang_single_node_pin_reads_only_whitelisted_env(monkeypatch) -> None:
    placement = json.dumps(
        {
            "Qwen/Qwen3-0.6B": {
                "node_ip": "192.0.2.55",
                "gpu_count": 1,
            }
        }
    )
    monkeypatch.setenv("MINT_SGLANG_MODEL_PLACEMENT_JSON", placement)
    monkeypatch.setenv("MINT_MODEL_PLACEMENT_JSON", "{}")

    class _Slice:
        node_ip = "192.0.2.55"

    class _Placement:
        slices = (_Slice(),)
        total_gpus = 1

    seen: list[tuple[str | None, str]] = []

    def fake_parse_model_gpu_placement(**kwargs):
        seen.append((kwargs["raw_json"], kwargs["env_var_name"]))
        return _Placement()

    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.parse_model_gpu_placement",
        fake_parse_model_gpu_placement,
    )

    real_open = builtins.open

    def guarded_open(path, *args, **kwargs):
        if str(path) == "/proc/self/environ":
            raise AssertionError("SGLang placement lookup must not read process environ dumps")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)

    assert _single_node_pin_for_model("Qwen/Qwen3-0.6B", "mint_sglang_qwen3_0_6b", 1) == "192.0.2.55"
    assert seen == [(placement, "MINT_SGLANG_MODEL_PLACEMENT_JSON")]


def test_sglang_single_node_pin_prefers_sglang_specific_placement(monkeypatch) -> None:
    sglang_placement = '{"Qwen/Test":{"node_ip":"192.0.2.10","gpu_count":1}}'
    generic_placement = '{"Qwen/Test":{"node_ip":"192.0.2.20","gpu_count":1}}'
    monkeypatch.setenv("MINT_SGLANG_MODEL_PLACEMENT_JSON", sglang_placement)
    monkeypatch.setenv("MINT_MODEL_PLACEMENT_JSON", generic_placement)

    class _Slice:
        node_ip = "192.0.2.10"

    class _Placement:
        slices = (_Slice(),)
        total_gpus = 1

    seen: list[tuple[str | None, str]] = []

    def fake_parse_model_gpu_placement(**kwargs):
        seen.append((kwargs["raw_json"], kwargs["env_var_name"]))
        if kwargs["env_var_name"] == "MINT_SGLANG_MODEL_PLACEMENT_JSON":
            return _Placement()
        raise AssertionError("generic placement should not be parsed when SGLang-specific placement matches")

    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.parse_model_gpu_placement",
        fake_parse_model_gpu_placement,
    )

    assert _single_node_pin_for_model("Qwen/Test", "mint_sglang_qwen_test", 1) == "192.0.2.10"
    assert seen == [(sglang_placement, "MINT_SGLANG_MODEL_PLACEMENT_JSON")]


def test_sglang_single_node_pin_falls_back_to_generic_model_placement(monkeypatch) -> None:
    sglang_placement = '{"Other/Model":{"node_ip":"192.0.2.10","gpu_count":1}}'
    generic_placement = '{"Qwen/Test":{"node_ip":"192.0.2.20","gpu_count":1}}'
    monkeypatch.setenv("MINT_SGLANG_MODEL_PLACEMENT_JSON", sglang_placement)
    monkeypatch.setenv("MINT_MODEL_PLACEMENT_JSON", generic_placement)

    class _Slice:
        node_ip = "192.0.2.20"

    class _Placement:
        slices = (_Slice(),)
        total_gpus = 1

    seen: list[tuple[str | None, str]] = []

    def fake_parse_model_gpu_placement(**kwargs):
        seen.append((kwargs["raw_json"], kwargs["env_var_name"]))
        if kwargs["env_var_name"] == "MINT_SGLANG_MODEL_PLACEMENT_JSON":
            return None
        return _Placement()

    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.parse_model_gpu_placement",
        fake_parse_model_gpu_placement,
    )

    assert _single_node_pin_for_model("Qwen/Test", "mint_sglang_qwen_test", 1) == "192.0.2.20"
    assert seen == [
        (sglang_placement, "MINT_SGLANG_MODEL_PLACEMENT_JSON"),
        (generic_placement, "MINT_MODEL_PLACEMENT_JSON"),
    ]


def test_sglang_single_node_pin_rejects_multi_slice_placement(monkeypatch) -> None:
    placement = '{"Qwen/Test":[{"node_ip":"192.0.2.10","gpu_count":1},{"node_ip":"192.0.2.20","gpu_count":1}]}'
    monkeypatch.setenv("MINT_SGLANG_MODEL_PLACEMENT_JSON", placement)
    monkeypatch.setenv("MINT_MODEL_PLACEMENT_JSON", "{}")

    fake_placement = SimpleNamespace(
        slices=(
            SimpleNamespace(node_ip="192.0.2.10"),
            SimpleNamespace(node_ip="192.0.2.20"),
        ),
        total_gpus=2,
    )

    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.parse_model_gpu_placement",
        lambda **_kwargs: fake_placement,
    )

    with pytest.raises(RuntimeError, match="single-node SGLang requires exactly 1 placement slice"):
        _single_node_pin_for_model("Qwen/Test", "mint_sglang_qwen_test", 2)


def test_sglang_single_node_pin_rejects_gpu_count_mismatch(monkeypatch) -> None:
    placement = '{"Qwen/Test":{"node_ip":"192.0.2.10","gpu_count":2}}'
    monkeypatch.setenv("MINT_SGLANG_MODEL_PLACEMENT_JSON", placement)
    monkeypatch.setenv("MINT_MODEL_PLACEMENT_JSON", "{}")

    fake_placement = SimpleNamespace(
        slices=(SimpleNamespace(node_ip="192.0.2.10"),),
        total_gpus=2,
    )

    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.parse_model_gpu_placement",
        lambda **_kwargs: fake_placement,
    )

    with pytest.raises(RuntimeError, match="placement GPU count mismatch, need 4 GPUs, got 2"):
        _single_node_pin_for_model("Qwen/Test", "mint_sglang_qwen_test", 4)


def _write_fake_peft_adapter(path) -> None:
    path.mkdir()
    (path / "adapter_model.safetensors").write_bytes(b"stub")
    (path / "adapter_config.json").write_text('{"peft_type":"LORA","r":8}', encoding="utf-8")


def _write_fake_safetensors_peft_adapter(path, tensors: dict) -> None:
    from safetensors.torch import save_file

    path.mkdir()
    save_file(tensors, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text('{"peft_type":"LORA","r":16}', encoding="utf-8")


class _FakeSGLangLoadResult:
    success = True


class _FakeSGLangEngine:
    def __init__(self, *, fail_load: bool = False, fail_unload: bool = False) -> None:
        self.fail_load = bool(fail_load)
        self.fail_unload = bool(fail_unload)
        self.load_calls: list[tuple[str, str]] = []
        self.unload_calls: list[str] = []
        self.generate_calls: list[dict] = []
        self.shutdown_called = False

    def load_lora_adapter(self, name: str, path: str):
        self.load_calls.append((name, path))
        if self.fail_load:
            raise RuntimeError("synthetic load failure")
        return _FakeSGLangLoadResult()

    def unload_lora_adapter(self, name: str):
        self.unload_calls.append(name)
        if self.fail_unload:
            raise RuntimeError("synthetic unload failure")
        return _FakeSGLangLoadResult()

    def generate(self, **kwargs):
        self.generate_calls.append(dict(kwargs))
        prompt_ids = [int(x) for x in kwargs.get("input_ids", [])]
        input_token_logprobs = [
            [None, token_id, None] if i == 0 else [-1.0 - (i / 10.0), token_id, None]
            for i, token_id in enumerate(prompt_ids)
        ]
        input_top_logprobs = [
            None if i == 0 else [["-0.5", "9", "None"], ["-0.7", "8", "None"]]
            for i, _token_id in enumerate(prompt_ids)
        ]
        return {
            "output_ids": [42],
            "finish_reason": {"type": "length"},
            "meta_info": {
                "input_token_logprobs": input_token_logprobs,
                "input_top_logprobs": input_top_logprobs,
                "output_token_logprobs": [[-0.42, 42, None]],
            },
        }

    def shutdown(self):
        self.shutdown_called = True


class _AbortableFakeSGLangEngine(_FakeSGLangEngine):
    def __init__(self) -> None:
        super().__init__()
        self.abort_calls: list[str] = []

    def abort(self, request_id: str) -> None:
        self.abort_calls.append(str(request_id))


def _ready_sglang_actor(fake_engine: _FakeSGLangEngine) -> SGLangEngineActor:
    actor = SGLangEngineActor(
        model_name="Qwen/Qwen3-0.6B",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_0_6b",
        max_model_len=64,
    )
    actor._engine = fake_engine
    actor._ready = True
    return actor


def test_sglang_actor_abort_calls_engine_abort_when_available() -> None:
    fake_engine = _AbortableFakeSGLangEngine()
    actor = _ready_sglang_actor(fake_engine)

    assert actor.abort("req-abort") is True
    assert fake_engine.abort_calls == ["req-abort"]


def test_sglang_actor_abort_returns_false_when_engine_has_no_abort() -> None:
    actor = _ready_sglang_actor(_FakeSGLangEngine())

    assert actor.abort("req-no-abort") is False


def test_sglang_actor_observability_binding_reports_gpu_and_adapter_state(monkeypatch, tmp_path) -> None:
    import mint_server.backend.ray_cluster.gpu_binding_helpers as gpu_binding_helpers

    class _RayContext:
        def get_node_id(self):
            return "node-a"

    fake_ray = SimpleNamespace(
        get_runtime_context=lambda: _RayContext(),
        util=SimpleNamespace(get_node_ip_address=lambda: "192.0.2.75"),
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(
        gpu_binding_helpers,
        "gpu_bindings_from_ray_gpu_ids",
        lambda *, hostname, node_id=None, rank=None: [
            {
                "hostname": hostname,
                "node_id": node_id,
                "ray_gpu_id": "0",
                "gpu_index": 0,
                "gpu_uuid": "GPU-test-0",
            }
        ],
    )
    adapter_dir = tmp_path / "adapter"
    _write_fake_peft_adapter(adapter_dir)
    actor = _ready_sglang_actor(_FakeSGLangEngine())
    actor.add_lora_for_session_from_path(sampling_session_id="session-a", lora_path=str(adapter_dir))

    binding = actor.get_observability_binding()

    assert binding["node_id"] == "node-a"
    assert binding["node_ip"] == "192.0.2.75"
    assert binding["gpu_indices"] == [0]
    assert binding["gpu_bindings"][0]["gpu_uuid"] == "GPU-test-0"
    assert binding["active_sessions"] == 1
    assert binding["loaded_adapter_count"] == 1
    assert binding["ready_count"] == 1


def test_sglang_actor_inventory_metadata_preserves_node_ip() -> None:
    from mint_server.backend.actors.model_actor_inventory import _normalize_actor_observability_payload

    metadata = _normalize_actor_observability_payload(
        {
            "hostname": "worker-a",
            "node_id": "node-a",
            "node_ip": " 192.0.2.75 ",
            "gpu_indices": ["4"],
            "gpu_bindings": [
                {
                    "hostname": "worker-a",
                    "node_id": "node-a",
                    "node_ip": " 192.0.2.75 ",
                    "ray_gpu_id": "0",
                    "gpu_index": "4",
                    "gpu_uuid": "GPU-test-4",
                }
            ],
            "ready_count": "1",
        }
    )

    assert metadata is not None
    assert metadata["node_ip"] == "192.0.2.75"
    assert metadata["gpu_bindings"][0]["node_ip"] == "192.0.2.75"
    assert metadata["ready_count"] == 1


def test_sglang_actor_reports_rss_bytes_for_internal_inventory(monkeypatch) -> None:
    actor = _ready_sglang_actor(_FakeSGLangEngine())

    class _FakeStatm:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return "10 3 0 0 0 0 0\n"

    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: _FakeStatm())
    monkeypatch.setattr(os, "sysconf", lambda name: 4096 if name == "SC_PAGE_SIZE" else 1)

    assert actor.get_rss_bytes() == 3 * 4096


def test_sglang_actor_loads_lora_by_path_and_selects_adapter_by_session(tmp_path) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_fake_peft_adapter(adapter_dir)
    fake_engine = _FakeSGLangEngine()
    actor = _ready_sglang_actor(fake_engine)

    base = actor.generate_base(
        sampling_session_id="__base__",
        prompt_ids=[1],
        request_id="base",
        max_tokens=1,
    )
    assert base["token_ids"] == [42]
    assert "lora_path" not in fake_engine.generate_calls[-1]

    loaded = actor.add_lora_for_session_from_path(
        sampling_session_id="session-a",
        lora_path=str(adapter_dir),
    )
    assert loaded["loaded"] is True
    assert loaded["reused"] is False
    assert fake_engine.load_calls == [(loaded["adapter_name"], str(adapter_dir.resolve()))]

    out = actor.generate_base(
        sampling_session_id="session-a",
        prompt_ids=[1, 2],
        request_id="lora",
        max_tokens=1,
    )
    assert out["token_ids"] == [42]
    assert fake_engine.generate_calls[-1]["lora_path"] == loaded["adapter_name"]


def test_sglang_actor_dedupes_same_adapter_path_and_refcounts_remove(tmp_path) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_fake_peft_adapter(adapter_dir)
    fake_engine = _FakeSGLangEngine()
    actor = _ready_sglang_actor(fake_engine)

    first = actor.add_lora_for_session_from_path(sampling_session_id="session-a", lora_path=str(adapter_dir))
    second = actor.add_lora_for_session_from_path(sampling_session_id="session-b", lora_path=str(adapter_dir))

    assert first["adapter_name"] == second["adapter_name"]
    assert first["lora_int_id"] == second["lora_int_id"]
    assert len(fake_engine.load_calls) == 1

    removed_first = actor.remove_session("session-a")
    assert removed_first["removed"] is True
    assert removed_first["unloaded"] is False
    assert fake_engine.unload_calls == []

    removed_second = actor.remove_session("session-b")
    assert removed_second["removed"] is True
    assert removed_second["unloaded"] is True
    assert fake_engine.unload_calls == [first["adapter_name"]]


def test_sglang_actor_preserves_lora_state_when_last_unref_unload_fails(tmp_path) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_fake_peft_adapter(adapter_dir)
    fake_engine = _FakeSGLangEngine(fail_unload=True)
    actor = _ready_sglang_actor(fake_engine)

    loaded = actor.add_lora_for_session_from_path(sampling_session_id="session-a", lora_path=str(adapter_dir))

    with pytest.raises(RuntimeError, match="synthetic unload failure"):
        actor.remove_session("session-a")

    health = actor.health()
    assert health["loaded_adapter_count"] == 1
    assert health["session_adapter_count"] == 1
    actor.generate_base(sampling_session_id="session-a", prompt_ids=[1], request_id="retryable", max_tokens=1)
    assert fake_engine.generate_calls[-1]["lora_path"] == loaded["adapter_name"]

    fake_engine.fail_unload = False
    removed = actor.remove_session("session-a")
    assert removed["removed"] is True
    assert removed["unloaded"] is True
    assert actor.health()["loaded_adapter_count"] == 0
    assert actor.health()["session_adapter_count"] == 0


def test_sglang_actor_keeps_distinct_adapter_names_for_distinct_paths(tmp_path) -> None:
    adapter_a = tmp_path / "adapter-a"
    adapter_b = tmp_path / "adapter-b"
    _write_fake_peft_adapter(adapter_a)
    _write_fake_peft_adapter(adapter_b)
    fake_engine = _FakeSGLangEngine()
    actor = _ready_sglang_actor(fake_engine)

    first = actor.add_lora_for_session_from_path(sampling_session_id="session-a", lora_path=str(adapter_a))
    second = actor.add_lora_for_session_from_path(sampling_session_id="session-b", lora_path=str(adapter_b))

    assert first["adapter_name"] != second["adapter_name"]
    assert len(fake_engine.load_calls) == 2
    actor.generate_base(sampling_session_id="session-a", prompt_ids=[1], request_id="a", max_tokens=1)
    assert fake_engine.generate_calls[-1]["lora_path"] == first["adapter_name"]
    actor.generate_base(sampling_session_id="session-b", prompt_ids=[1], request_id="b", max_tokens=1)
    assert fake_engine.generate_calls[-1]["lora_path"] == second["adapter_name"]


def test_sglang_actor_rejects_missing_adapter_files(tmp_path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"peft_type":"LORA","r":8}', encoding="utf-8")
    actor = _ready_sglang_actor(_FakeSGLangEngine())

    with pytest.raises(RuntimeError, match="adapter_model\\.safetensors"):
        actor.add_lora_for_session_from_path(sampling_session_id="session-a", lora_path=str(adapter_dir))


def test_sglang_actor_rolls_back_registry_on_load_failure(tmp_path) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_fake_peft_adapter(adapter_dir)
    actor = _ready_sglang_actor(_FakeSGLangEngine(fail_load=True))

    with pytest.raises(RuntimeError, match="synthetic load failure"):
        actor.add_lora_for_session_from_path(sampling_session_id="session-a", lora_path=str(adapter_dir))

    health = actor.health()
    assert health["loaded_adapter_count"] == 0
    assert health["session_adapter_count"] == 0


def test_sglang_actor_allows_qwen3_moe_expert_mlp_lora_to_reach_backend_load(tmp_path) -> None:
    import torch

    model_dir = tmp_path / "qwen3-moe"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_moe",
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "moe_intermediate_size": 768,
            }
        ),
        encoding="utf-8",
    )
    adapter_dir = tmp_path / "adapter"
    _write_fake_safetensors_peft_adapter(
        adapter_dir,
        {
            "base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight": torch.zeros(16, 2048),
            "base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_B.weight": torch.zeros(768, 16),
        },
    )
    fake_engine = _FakeSGLangEngine()
    actor = _ready_sglang_actor(fake_engine)
    actor.model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    actor.model_path = str(model_dir)

    loaded = actor.add_lora_for_session_from_path(sampling_session_id="session-moe", lora_path=str(adapter_dir))

    assert loaded["loaded"] is True
    assert fake_engine.load_calls == [(loaded["adapter_name"], str(adapter_dir.resolve()))]
    assert actor.health()["loaded_adapter_count"] == 1
    assert actor.health()["session_adapter_count"] == 1


def test_sglang_capability_check_supports_qwen3_moe_expert_mlp_lora(tmp_path) -> None:
    import torch

    model_dir = tmp_path / "qwen3-moe"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_moe",
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "moe_intermediate_size": 768,
            }
        ),
        encoding="utf-8",
    )
    adapter_dir = tmp_path / "adapter"
    _write_fake_safetensors_peft_adapter(
        adapter_dir,
        {
            "base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight": torch.zeros(16, 2048),
            "base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_B.weight": torch.zeros(768, 16),
        },
    )

    decision = check_sglang_lora_adapter_support(
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        model_path=str(model_dir),
        adapter_path=str(adapter_dir),
    )

    assert decision.supported is True
    assert decision.backend == "sglang"
    assert decision.feature == SGLANG_QWEN3_MOE_EXPERT_LORA_FEATURE
    assert decision.evidence["model_type"] == "qwen3_moe"
    assert decision.evidence["example_shape"] == (16, 2048)
    assert decision.reason is None


def test_sglang_facade_initializes_for_qwen3_moe_expert_mlp_lora(tmp_path, monkeypatch) -> None:
    import torch

    model_dir = tmp_path / "qwen3-moe"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_moe",
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "moe_intermediate_size": 768,
            }
        ),
        encoding="utf-8",
    )
    adapter_dir = tmp_path / "adapter"
    _write_fake_safetensors_peft_adapter(
        adapter_dir,
        {
            "base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight": torch.zeros(16, 2048),
            "base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_B.weight": torch.zeros(768, 16),
        },
    )
    engine = SGLangInferenceEngine(
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        model_path=str(model_dir),
        actor_name="mint_sglang_qwen3_30b_a3b_instruct_2507",
        tensor_parallel_size=4,
        max_model_len=512,
    )

    initialized = False

    async def initialize() -> None:
        nonlocal initialized
        initialized = True

    monkeypatch.setattr(engine, "initialize", initialize)

    class _FakeRemote:
        def remote(self, **kwargs):
            assert kwargs["sampling_session_id"] == "session-moe"
            assert kwargs["lora_path"] == str(adapter_dir)
            return {
                "sampling_session_id": "session-moe",
                "adapter_name": "adapter-moe",
                "adapter_path": str(adapter_dir),
                "lora_int_id": 1,
                "loaded": True,
                "reused": False,
            }

    actor = SimpleNamespace(add_lora_for_session_from_path=_FakeRemote())
    engine.server = actor

    async def fake_ray_get(ref, **_kwargs):
        return ref

    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.ray_get_with_model_actor_supervisor_keepalive",
        fake_ray_get,
    )

    async def run() -> None:
        result = await engine.add_lora_for_session_from_path(
            sampling_session_id="session-moe",
            lora_path=str(adapter_dir),
        )
        assert result == 1

    anyio.run(run)
    assert initialized is True


def test_sglang_actor_allows_dense_lora_safetensors_adapter(tmp_path) -> None:
    import torch

    model_dir = tmp_path / "qwen3-dense"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "hidden_size": 1024, "intermediate_size": 3072}),
        encoding="utf-8",
    )
    adapter_dir = tmp_path / "adapter"
    _write_fake_safetensors_peft_adapter(
        adapter_dir,
        {
            "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight": torch.zeros(16, 1024),
            "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight": torch.zeros(3072, 16),
        },
    )
    fake_engine = _FakeSGLangEngine()
    actor = _ready_sglang_actor(fake_engine)
    actor.model_path = str(model_dir)

    loaded = actor.add_lora_for_session_from_path(sampling_session_id="session-dense", lora_path=str(adapter_dir))

    assert loaded["loaded"] is True
    assert fake_engine.load_calls == [(loaded["adapter_name"], str(adapter_dir.resolve()))]


def test_sglang_actor_computes_prompt_logprobs_and_topk() -> None:
    fake_engine = _FakeSGLangEngine()
    actor = _ready_sglang_actor(fake_engine)

    assert actor.compute_prompt_logprobs(
        sampling_session_id="__base__",
        prompt_ids=[1, 2],
        request_id="lp",
    ) == [None, -1.1]
    assert fake_engine.generate_calls[-1]["return_logprob"] is True
    assert fake_engine.generate_calls[-1]["top_logprobs_num"] == 0

    assert actor.compute_prompt_topk(
        sampling_session_id="__base__",
        prompt_ids=[1, 2],
        request_id="topk",
        k=1,
    ) == [None, [(9, -0.5)]]
    assert fake_engine.generate_calls[-1]["return_logprob"] is True
    assert fake_engine.generate_calls[-1]["top_logprobs_num"] == 1


def test_sglang_actor_generate_can_return_prompt_logprob_fields() -> None:
    fake_engine = _FakeSGLangEngine()
    actor = _ready_sglang_actor(fake_engine)

    out = actor.generate_base(
        sampling_session_id="__base__",
        prompt_ids=[1, 2],
        request_id="req",
        max_tokens=1,
        prompt_logprobs=True,
        topk_prompt_logprobs=2,
    )

    assert out["prompt_logprobs"] == [None, -1.1]
    assert out["topk_prompt_logprobs"] == [None, [(9, -0.5), (8, -0.7)]]
    assert fake_engine.generate_calls[-1]["rid"] == "req"


def test_sglang_actor_shutdown_clears_actor_local_lora_state(tmp_path) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_fake_peft_adapter(adapter_dir)
    fake_engine = _FakeSGLangEngine()
    actor = _ready_sglang_actor(fake_engine)
    actor.add_lora_for_session_from_path(sampling_session_id="session-a", lora_path=str(adapter_dir))

    health = actor.shutdown()

    assert fake_engine.shutdown_called is True
    assert health["ready"] is False
    assert health["loaded_adapter_count"] == 0
    assert health["session_adapter_count"] == 0
    with pytest.raises(RuntimeError, match="not initialized"):
        actor.generate_base(sampling_session_id="session-a", prompt_ids=[1], request_id="after-shutdown", max_tokens=1)


def test_sglang_backend_session_id_uses_base_sentinel_for_base_session() -> None:
    snapshot = sampling_route.SamplingSessionSnapshot(
        session_id="sglang-base",
        uses_multi_lora=True,
        uses_base_model=True,
        base_model="Qwen/Qwen3-0.6B",
        lora_rank=0,
        adapter_path=None,
        lora_loaded=False,
        lora_int_id=None,
        metadata_version=1,
    )

    assert sampling_route._backend_sampling_session_id(
        engine_backend="sglang",
        session_id="sglang-base",
        snapshot=snapshot,
    ) == "__base__"
    assert sampling_route._backend_sampling_session_id(
        engine_backend="vllm",
        session_id="vllm-base",
        snapshot=snapshot,
    ) == "vllm-base"


def test_sglang_backend_session_id_preserves_lora_session_id() -> None:
    snapshot = sampling_route.SamplingSessionSnapshot(
        session_id="sglang-lora",
        uses_multi_lora=True,
        uses_base_model=False,
        base_model="Qwen/Qwen3-0.6B",
        lora_rank=8,
        adapter_path="/tmp/adapter",
        lora_loaded=False,
        lora_int_id=None,
        metadata_version=1,
    )

    assert sampling_route._backend_sampling_session_id(
        engine_backend="sglang",
        session_id="sglang-lora",
        snapshot=snapshot,
    ) == "sglang-lora"


def test_sglang_force_backend_confirm_reloads_even_when_session_marked_loaded(monkeypatch) -> None:
    class _SessionManager:
        def __init__(self) -> None:
            self.marked: list[tuple[str, bool, int | None]] = []

        def is_multi_lora_session(self, _session_id: str) -> bool:
            return True

        def get_engine(self, _session_id: str):
            return None

        def get_session_base_model(self, _session_id: str) -> str:
            return "Qwen/Qwen3-0.6B"

        def get_session_lora_rank(self, _session_id: str) -> int:
            return 8

        def get_session_adapter_path(self, _session_id: str):
            return "/tmp/fake-adapter"

        def is_session_lora_loaded(self, _session_id: str) -> bool:
            return True

        def get_session_lora_int_id(self, _session_id: str):
            return 1

        def is_base_model_session(self, _session_id: str) -> bool:
            return False

        def get_session_metadata_version(self, _session_id: str) -> int:
            return 1

        def mark_session_lora_loaded(self, session_id: str, loaded: bool = True, *, lora_int_id: int | None = None):
            self.marked.append((session_id, bool(loaded), lora_int_id))

    class _Engine:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def add_lora_for_session_from_path(self, *, sampling_session_id: str, lora_path: str) -> int:
            self.calls.append((sampling_session_id, lora_path))
            return 1

    manager = _SessionManager()
    engine = _Engine()
    monkeypatch.setattr(sampling_route, "session_manager", manager)

    async def run() -> None:
        snapshot = sampling_route.SamplingSessionSnapshot(
            session_id="sglang-lora",
            uses_multi_lora=True,
            uses_base_model=False,
            base_model="Qwen/Qwen3-0.6B",
            lora_rank=8,
            adapter_path="/tmp/fake-adapter",
            lora_loaded=True,
            lora_int_id=1,
            metadata_version=1,
        )
        await sampling_route._ensure_session_lora_loaded(
            engine,
            "sglang-lora",
            snapshot=snapshot,
            force_backend_confirm=True,
        )

    anyio.run(run)
    assert engine.calls == [("sglang-lora", "/tmp/fake-adapter")]
    assert manager.marked == [("sglang-lora", True, 1)]


@pytest.mark.parametrize(
    ("returned_lora_int_id", "error_pattern"),
    [
        (None, "returned no lora_int_id"),
        ("not-an-int", "returned invalid lora_int_id"),
    ],
)
def test_sglang_ensure_lora_loaded_rejects_invalid_lora_int_id(
    monkeypatch,
    returned_lora_int_id,
    error_pattern: str,
) -> None:
    class _SessionManager:
        def __init__(self) -> None:
            self.marked: list[tuple[str, bool, int | None]] = []

        def is_multi_lora_session(self, _session_id: str) -> bool:
            return True

        def get_engine(self, _session_id: str):
            return None

        def get_session_base_model(self, _session_id: str) -> str:
            return "Qwen/Qwen3-0.6B"

        def get_session_lora_rank(self, _session_id: str) -> int:
            return 8

        def get_session_adapter_path(self, _session_id: str):
            return "/tmp/fake-adapter"

        def is_session_lora_loaded(self, _session_id: str) -> bool:
            return False

        def get_session_lora_int_id(self, _session_id: str):
            return None

        def is_base_model_session(self, _session_id: str) -> bool:
            return False

        def get_session_metadata_version(self, _session_id: str) -> int:
            return 1

        def mark_session_lora_loaded(self, session_id: str, loaded: bool = True, *, lora_int_id: int | None = None):
            self.marked.append((session_id, bool(loaded), lora_int_id))

    class _Engine:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def add_lora_for_session_from_path(self, *, sampling_session_id: str, lora_path: str):
            self.calls.append((sampling_session_id, lora_path))
            return returned_lora_int_id

    manager = _SessionManager()
    engine = _Engine()
    monkeypatch.setattr(sampling_route, "session_manager", manager)

    async def run() -> None:
        snapshot = sampling_route.SamplingSessionSnapshot(
            session_id="sglang-lora",
            uses_multi_lora=True,
            uses_base_model=False,
            base_model="Qwen/Qwen3-0.6B",
            lora_rank=8,
            adapter_path="/tmp/fake-adapter",
            lora_loaded=False,
            lora_int_id=None,
            metadata_version=1,
        )
        with pytest.raises(RuntimeError, match=error_pattern):
            await sampling_route._ensure_session_lora_loaded(engine, "sglang-lora", snapshot=snapshot)

    anyio.run(run)
    assert engine.calls == [("sglang-lora", "/tmp/fake-adapter")]
    assert manager.marked == []


def test_sglang_ensure_lora_loaded_preflights_backend_capability_before_load(monkeypatch) -> None:
    class _SessionManager:
        def __init__(self) -> None:
            self.marked: list[tuple[str, bool, int | None]] = []

        def is_multi_lora_session(self, _session_id: str) -> bool:
            return True

        def get_engine(self, _session_id: str):
            return None

        def get_session_base_model(self, _session_id: str) -> str:
            return "Qwen/Qwen3-30B-A3B-Instruct-2507"

        def get_session_lora_rank(self, _session_id: str) -> int:
            return 8

        def get_session_adapter_path(self, _session_id: str):
            return "/tmp/fake-adapter"

        def is_session_lora_loaded(self, _session_id: str) -> bool:
            return False

        def get_session_lora_int_id(self, _session_id: str):
            return None

        def is_base_model_session(self, _session_id: str) -> bool:
            return False

        def get_session_metadata_version(self, _session_id: str) -> int:
            return 1

        def mark_session_lora_loaded(self, session_id: str, loaded: bool = True, *, lora_int_id: int | None = None):
            self.marked.append((session_id, bool(loaded), lora_int_id))

    class _Engine:
        def __init__(self) -> None:
            self.validated: list[str] = []
            self.calls: list[tuple[str, str]] = []

        def validate_lora_adapter_supported(self, lora_path: str) -> None:
            self.validated.append(str(lora_path))
            raise SGLangUnsupportedFeatureError("unsupported capability")

        async def add_lora_for_session_from_path(self, *, sampling_session_id: str, lora_path: str):
            self.calls.append((sampling_session_id, lora_path))
            return 1

    manager = _SessionManager()
    engine = _Engine()
    monkeypatch.setattr(sampling_route, "session_manager", manager)

    async def run() -> None:
        snapshot = sampling_route.SamplingSessionSnapshot(
            session_id="sglang-lora",
            uses_multi_lora=True,
            uses_base_model=False,
            base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
            lora_rank=8,
            adapter_path="/tmp/fake-adapter",
            lora_loaded=False,
            lora_int_id=None,
            metadata_version=1,
        )
        with pytest.raises(SGLangUnsupportedFeatureError, match="unsupported capability"):
            await sampling_route._ensure_session_lora_loaded(engine, "sglang-lora", snapshot=snapshot)

    anyio.run(run)
    assert engine.validated == ["/tmp/fake-adapter"]
    assert engine.calls == []
    assert manager.marked == []


def test_remove_sglang_session_from_existing_actor_uses_named_actor_without_cold_start(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class _RemoteMethod:
        def remote(self, session_id: str):
            calls["session_id"] = session_id
            return {"removed": True}

    class _Actor:
        remove_session = _RemoteMethod()

    async def fake_async_get(ref, *, timeout_s=None):
        calls["timeout_s"] = timeout_s
        return ref

    def fake_get_actor(name: str, *, namespace: str):
        calls["actor_name"] = name
        calls["namespace"] = namespace
        return _Actor()

    monkeypatch.setattr("mint_server.backend.sglang_engine.ray.is_initialized", lambda: True)
    monkeypatch.setattr("mint_server.backend.sglang_engine.ray.get_actor", fake_get_actor)
    monkeypatch.setattr("mint_server.backend.sglang_engine.async_get_ray_ref", fake_async_get)

    async def run() -> None:
        removed = await remove_sglang_session_from_existing_actor(
            "Qwen/Qwen3-0.6B",
            "sglang-lora-session",
            timeout_s=7,
        )
        assert removed is True

    asyncio.run(run())
    assert calls == {
        "actor_name": "mint_sglang_qwen3_0_6b",
        "namespace": "mint",
        "session_id": "sglang-lora-session",
        "timeout_s": 7,
    }

    monkeypatch.setattr("mint_server.backend.sglang_engine.ray.is_initialized", lambda: False)
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.ray.get_actor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not cold-start lookup")),
    )

    async def run_missing_ray() -> None:
        removed = await remove_sglang_session_from_existing_actor("Qwen/Qwen3-0.6B", "sglang-lora-session")
        assert removed is False

    asyncio.run(run_missing_ray())


def test_sglang_end_session_removes_unowned_detached_actor_session(monkeypatch, tmp_path) -> None:
    from mint_server.backend.sessions.session_manager import SessionManager

    adapter_dir = tmp_path / "adapter"
    _write_fake_peft_adapter(adapter_dir)
    session_id = "sglang-unowned-session"
    removed: list[tuple[str, str]] = []
    deleted: list[str] = []

    detached_info = {
        "session_id": session_id,
        "base_model": "Qwen/Qwen3-0.6B",
        "uses_base_model": False,
        "lora_rank": 8,
        "adapter_path": str(adapter_dir),
        "lora_loaded": True,
        "lora_int_id": 1,
        "metadata_version": 1,
        "last_activity": 123.0,
    }

    async def fake_remove_existing_actor(model_name: str, sampling_session_id: str) -> bool:
        removed.append((model_name, sampling_session_id))
        return True

    async def fake_drop_lora_load_lock(_session_id: str) -> None:
        return None

    monkeypatch.setattr(
        "mint_server.backend.stores.sampling_session_store.get_sampling_session_info",
        lambda requested: detached_info if requested == session_id else None,
    )
    monkeypatch.setattr(
        "mint_server.backend.stores.sampling_session_store.delete_sampling_session",
        lambda requested: deleted.append(requested),
    )
    monkeypatch.setattr(
        "mint_server.backend.core.model_registry.get_model_config",
        lambda _model: SimpleNamespace(serving_backend="sglang"),
    )
    monkeypatch.setattr("mint_server.backend.sglang_engine.get_cached_sglang_engine_for_model", lambda _model: None)
    monkeypatch.setattr(
        "mint_server.backend.sglang_engine.remove_sglang_session_from_existing_actor",
        fake_remove_existing_actor,
    )
    monkeypatch.setattr("mint_server.routes.sampling._drop_lora_load_lock", fake_drop_lora_load_lock)

    manager = SessionManager()

    async def run() -> None:
        assert await manager.end_session(session_id) is True

    asyncio.run(run())
    assert removed == [("Qwen/Qwen3-0.6B", session_id)]
    assert deleted == [session_id]
    assert session_id not in manager.list_sessions()


def test_sglang_compute_logprobs_enqueues_model_work(monkeypatch) -> None:
    class _SessionManager:
        def is_multi_lora_session(self, _session_id: str) -> bool:
            return True

        def get_engine(self, _session_id: str):
            return None

        def get_session_base_model(self, _session_id: str) -> str:
            return "Qwen/Qwen3-0.6B"

        def get_session_lora_rank(self, _session_id: str) -> int:
            return 0

        def get_session_adapter_path(self, _session_id: str):
            return None

        def is_session_lora_loaded(self, _session_id: str) -> bool:
            return False

        def get_session_lora_int_id(self, _session_id: str):
            return None

        def is_base_model_session(self, _session_id: str) -> bool:
            return True

        def get_session_metadata_version(self, _session_id: str) -> int:
            return 1

    calls: list[dict] = []

    async def capture_enqueue(**kwargs):
        calls.append(kwargs)

    snapshot = sampling_route.SamplingSessionSnapshot(
        session_id="sglang-base-session",
        uses_multi_lora=True,
        uses_base_model=True,
        base_model="Qwen/Qwen3-0.6B",
        lora_rank=0,
        adapter_path=None,
        lora_loaded=False,
        lora_int_id=None,
        metadata_version=1,
    )

    async def _get_snapshot(_session_id: str):
        return snapshot

    monkeypatch.setattr(sampling_route, "session_manager", _SessionManager())
    monkeypatch.setattr(sampling_route, "_async_get_http_sampling_snapshot", _get_snapshot)
    monkeypatch.setattr("mint_server.backend.scheduling.model_work_admission.enqueue_model_work", capture_enqueue)
    monkeypatch.setattr("mint_server.routes.sampling.enqueue_model_work", capture_enqueue)
    monkeypatch.setattr(sampling_route, "record_sampling_admission_metric", lambda **_kwargs: None)
    monkeypatch.setattr(
        "mint_server.backend.core.model_registry.get_model_config",
        lambda _model: SimpleNamespace(max_model_len=512, serving_backend="sglang"),
    )
    monkeypatch.setattr(
        "mint_server.backend.scheduling.model_work_scheduler.model_work_scheduler",
        SimpleNamespace(),
    )
    monkeypatch.setattr(sampling_route, "task_futures", SimpleNamespace())

    request = ComputeLogprobsRequest(
        sampling_session_id="sglang-base-session",
        seq_id=0,
        sequence=ModelInput.from_ints([1, 2, 3]),
    )
    http_request = SimpleNamespace(state=SimpleNamespace(user_data=None), headers={})

    async def run() -> None:
        result = await sampling_route.compute_logprobs(request, cast(Request, http_request))
        assert result.request_id
        assert calls
        assert calls[0]["op"] == "sampling.compute_logprobs"
        assert calls[0]["domain_key"] == "sglang:Qwen/Qwen3-0.6B"
        assert calls[0]["queued_meta"]["backend"] == "sglang"

    anyio.run(run)


def test_sglang_do_compute_logprobs_external_cancel_aborts_backend_request(monkeypatch) -> None:
    session_id = "sglang-base-session"
    request_id = "req-sglang-logprobs-cancel"
    aborts: list[str] = []

    class _SessionManager:
        def __init__(self) -> None:
            self.inflight: list[tuple[str, int]] = []

        def mark_session_inflight(self, session_id: str, delta: int) -> None:
            self.inflight.append((session_id, delta))

        def is_multi_lora_session(self, _session_id: str) -> bool:
            return True

        def get_session_base_model(self, _session_id: str) -> str:
            return "Qwen/Qwen3-0.6B"

    class _TaskFutures:
        def __init__(self) -> None:
            self.failed: list[tuple[str, str]] = []

        async def async_get_status(self, _request_id: str):
            return sampling_route.FutureStatus.FAILED

        async def async_fail(self, request_id: str, error: str) -> None:
            self.failed.append((request_id, error))

    class _Supervisor:
        def __init__(self) -> None:
            self.inflight: list[tuple[str, int]] = []

        def mark_inflight(self, actor_name: str, delta: int) -> None:
            self.inflight.append((actor_name, delta))

    class _Engine:
        actor_name = "mint_sglang_qwen3_0_6b"

        async def compute_logprobs(self, **_kwargs):
            await asyncio.sleep(3600)

        async def abort_request(self, request_id: str) -> bool:
            aborts.append(request_id)
            return True

    snapshot = sampling_route.SamplingSessionSnapshot(
        session_id=session_id,
        uses_multi_lora=True,
        uses_base_model=True,
        base_model="Qwen/Qwen3-0.6B",
        lora_rank=0,
        adapter_path=None,
        lora_loaded=False,
        lora_int_id=None,
        metadata_version=1,
    )
    manager = _SessionManager()
    futures = _TaskFutures()
    supervisor = _Supervisor()
    engine = _Engine()

    async def _get_sglang_engine(**_kwargs):
        return engine, "sglang"

    async def _restore_noop(_session_id: str) -> None:
        return None

    monkeypatch.delenv("MINT_SAMPLE_AWAIT_TIMEOUT_S", raising=False)
    monkeypatch.setattr(sampling_route, "session_manager", manager)
    monkeypatch.setattr(sampling_route, "task_futures", futures)
    monkeypatch.setattr(sampling_route, "_get_sampling_snapshot", lambda _session_id: snapshot)
    monkeypatch.setattr(sampling_route, "_restore_local_sampling_session_if_needed", _restore_noop)
    monkeypatch.setattr(sampling_route, "_get_engine_for_sampling_session", _get_sglang_engine)
    monkeypatch.setattr(sampling_route, "_resolve_billing_model", lambda _session_id: "Qwen/Qwen3-0.6B")
    monkeypatch.setattr(
        "mint_server.backend.core.model_registry.get_model_config",
        lambda _model: SimpleNamespace(max_model_len=512, serving_backend="sglang"),
    )
    monkeypatch.setattr(
        "mint_server.backend.actors.model_actor_supervisor.get_model_actor_supervisor",
        lambda: supervisor,
    )

    request = ComputeLogprobsRequest(
        sampling_session_id=session_id,
        seq_id=0,
        sequence=ModelInput.from_ints([1, 2, 3]),
    )

    anyio.run(
        sampling_route._do_compute_logprobs,
        request_id,
        request,
        None,
    )

    assert aborts == [request_id]
    assert futures.failed and futures.failed[-1][0] == request_id
    assert "future_status=failed" in futures.failed[-1][1]
    assert manager.inflight == [(session_id, 1), (session_id, -1)]
    assert supervisor.inflight == [("mint_sglang_qwen3_0_6b", 1), ("mint_sglang_qwen3_0_6b", -1)]


def test_sglang_asample_enqueues_backend_metadata(monkeypatch) -> None:
    class _SessionManager:
        def is_multi_lora_session(self, _session_id: str) -> bool:
            return True

        def get_engine(self, _session_id: str):
            return None

        def get_session_base_model(self, _session_id: str) -> str:
            return "Qwen/Qwen3-0.6B"

        def get_session_lora_rank(self, _session_id: str) -> int:
            return 0

        def get_session_adapter_path(self, _session_id: str):
            return None

        def is_session_lora_loaded(self, _session_id: str) -> bool:
            return False

        def get_session_lora_int_id(self, _session_id: str):
            return None

        def is_base_model_session(self, _session_id: str) -> bool:
            return True

        def get_session_metadata_version(self, _session_id: str) -> int:
            return 1

    calls: list[dict] = []

    async def capture_enqueue(**kwargs):
        calls.append(kwargs)
        return ModelWorkAdmissionResult(
            request_id=str(kwargs["request_id"]),
            scheduler_result=AppendWorkResult(ok=True, request_id=str(kwargs["request_id"])),
        )

    snapshot = sampling_route.SamplingSessionSnapshot(
        session_id="sglang-base-session",
        uses_multi_lora=True,
        uses_base_model=True,
        base_model="Qwen/Qwen3-0.6B",
        lora_rank=0,
        adapter_path=None,
        lora_loaded=False,
        lora_int_id=None,
        metadata_version=1,
    )

    async def _get_snapshot(_session_id: str):
        return snapshot

    monkeypatch.setattr(sampling_route, "session_manager", _SessionManager())
    monkeypatch.setattr(sampling_route, "_async_get_http_sampling_snapshot", _get_snapshot)
    monkeypatch.setattr("mint_server.routes.sampling.enqueue_model_work", capture_enqueue)
    monkeypatch.setattr(sampling_route, "record_sampling_admission_metric", lambda **_kwargs: None)
    monkeypatch.setattr(
        "mint_server.backend.core.model_registry.get_model_config",
        lambda _model: SimpleNamespace(max_model_len=512, serving_backend="sglang"),
    )

    request = SampleRequest(
        sampling_session_id="sglang-base-session",
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=2),
    )
    http_request = SimpleNamespace(state=SimpleNamespace(user_data=None), headers={})

    async def run() -> None:
        result = await sampling_route.asample(request, cast(Request, http_request))
        assert result.request_id
        assert calls
        assert calls[0]["op"] == "sampling.asample"
        assert calls[0]["domain_key"] == "sglang:Qwen/Qwen3-0.6B"
        assert calls[0]["queued_meta"]["backend"] == "sglang"
        assert calls[0]["queued_meta"]["domain_key"] == "sglang:Qwen/Qwen3-0.6B"

    anyio.run(run)


def test_sglang_do_sample_emits_backend_execution_span_attributes(monkeypatch) -> None:
    class _SessionManager:
        def __init__(self) -> None:
            self.inflight: list[tuple[str, int]] = []

        def mark_session_inflight(self, session_id: str, delta: int) -> None:
            self.inflight.append((session_id, delta))

        def is_multi_lora_session(self, _session_id: str) -> bool:
            return True

        def get_engine(self, _session_id: str):
            return None

        def get_session_base_model(self, _session_id: str) -> str:
            return "Qwen/Qwen3-0.6B"

        def get_session_lora_rank(self, _session_id: str) -> int:
            return 0

        def get_session_adapter_path(self, _session_id: str):
            return None

        def is_session_lora_loaded(self, _session_id: str) -> bool:
            return False

        def get_session_lora_int_id(self, _session_id: str):
            return None

        def is_base_model_session(self, _session_id: str) -> bool:
            return True

        def get_session_metadata_version(self, _session_id: str) -> int:
            return 1

    class _TaskFutures:
        def __init__(self) -> None:
            self.resolved: list[tuple[str, dict]] = []
            self.failed: list[tuple[str, str]] = []
            self.meta_updates: list[tuple[str, dict]] = []

        async def async_update_meta(self, request_id: str, meta: dict | None = None) -> None:
            self.meta_updates.append((request_id, dict(meta or {})))

        async def async_get_status(self, _request_id: str):
            raise KeyError("unknown request")

        async def async_get_meta(self, _request_id: str):
            return {}

        async def async_resolve(self, request_id: str, response: dict, **_kwargs) -> None:
            self.resolved.append((request_id, response))

        async def async_fail(self, request_id: str, error: str) -> None:
            self.failed.append((request_id, error))

    class _Supervisor:
        def __init__(self) -> None:
            self.inflight: list[tuple[str, int]] = []

        def mark_inflight(self, actor_name: str, delta: int) -> None:
            self.inflight.append((actor_name, delta))

    class _Engine:
        actor_name = "mint_sglang_qwen3_0_6b"

        async def generate(self, **_kwargs):
            return SimpleNamespace(token_ids=[7], logprobs=[-0.7], stop_reason="length")

        async def compute_logprobs(self, **_kwargs):
            return [None, -1.25, -1.5]

        async def compute_topk(self, **_kwargs):
            return [None, [(9, -0.5)], [(8, -0.7)]]

        async def abort_request(self, _request_id: str) -> bool:
            return True

    spans: list[dict[str, object]] = []

    async def _capture_span(span_name, action, **kwargs):
        spans.append(
            {
                "span_name": span_name,
                "component": kwargs.get("component"),
                "op": kwargs.get("op"),
                "request_id": kwargs.get("request_id"),
                "attributes": dict(kwargs.get("attributes") or {}),
            }
        )
        return await action()

    manager = _SessionManager()
    futures = _TaskFutures()
    supervisor = _Supervisor()
    engine = _Engine()

    async def _get_sglang_engine(**_kwargs):
        return engine, "sglang"

    monkeypatch.setattr(sampling_route, "session_manager", manager)
    monkeypatch.setattr(sampling_route, "task_futures", futures)
    monkeypatch.setattr(sampling_route, "run_async_with_otel_span", _capture_span)
    monkeypatch.setattr(sampling_route, "_get_engine_for_sampling_session", _get_sglang_engine)
    monkeypatch.setattr(
        "mint_server.backend.actors.model_actor_supervisor.get_model_actor_supervisor",
        lambda: supervisor,
    )
    monkeypatch.setattr(
        "mint_server.backend.core.model_registry.get_model_config",
        lambda _model: SimpleNamespace(max_model_len=512, serving_backend="sglang"),
    )

    request = SampleRequest(
        sampling_session_id="sglang-base-session",
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=2),
        prompt_logprobs=True,
        topk_prompt_logprobs=1,
    )

    async def run() -> None:
        await sampling_route._do_sample(
            request_id="req-sglang-span",
            request=request,
            user_id=None,
        )

    anyio.run(run)

    assert futures.failed == []
    assert futures.resolved
    assert manager.inflight == [("sglang-base-session", 1), ("sglang-base-session", -1)]
    assert supervisor.inflight == [("mint_sglang_qwen3_0_6b", 1), ("mint_sglang_qwen3_0_6b", -1)]

    by_name = {str(span["span_name"]): span for span in spans}
    generate_attrs = by_name["sampling.generate"]["attributes"]
    assert generate_attrs["backend"] == "sglang"
    assert generate_attrs["base_model"] == "Qwen/Qwen3-0.6B"
    assert generate_attrs["actor_name"] == "mint_sglang_qwen3_0_6b"
    assert generate_attrs["sampling_session_id"] == "sglang-base-session"
    assert generate_attrs["prompt_tokens"] == 3
    assert generate_attrs["max_tokens"] == 2
    assert generate_attrs["num_samples"] == 1
    assert generate_attrs["lora_rank"] == 0
    assert generate_attrs["uses_base_model"] is True

    logprob_attrs = by_name["sampling.compute_prompt_logprobs"]["attributes"]
    assert logprob_attrs["backend"] == "sglang"
    assert logprob_attrs["actor_name"] == "mint_sglang_qwen3_0_6b"
    assert logprob_attrs["prompt_tokens"] == 3

    topk_attrs = by_name["sampling.compute_prompt_topk"]["attributes"]
    assert topk_attrs["backend"] == "sglang"
    assert topk_attrs["actor_name"] == "mint_sglang_qwen3_0_6b"
    assert topk_attrs["topk"] == 1


def test_sampling_await_timeout_aborts_and_raises_runtime_error_after_task_cancel(monkeypatch) -> None:
    aborts: list[str] = []

    class _Engine:
        async def abort_request(self, request_id: str) -> bool:
            aborts.append(request_id)
            return True

    class _TaskFutures:
        async def async_get_status(self, _request_id: str):
            return sampling_route.FutureStatus.PENDING

    async def never_finishes():
        await asyncio.sleep(3600)

    monkeypatch.setenv("MINT_SAMPLE_AWAIT_TIMEOUT_S", "0.01")
    monkeypatch.setattr(sampling_route, "task_futures", _TaskFutures())

    async def run() -> None:
        with pytest.raises(RuntimeError, match="timed out in _await_with_external_fail_abort"):
            await sampling_route._await_with_external_fail_abort(
                engine=_Engine(),
                request_id="req-timeout-cleanup",
                awaitable=never_finishes(),
            )

    asyncio.run(run())
    assert aborts == ["req-timeout-cleanup"]


def test_sampling_await_external_cancel_aborts_engine_subrequest_id(monkeypatch) -> None:
    aborts: list[str] = []

    class _Engine:
        async def abort_request(self, request_id: str) -> bool:
            aborts.append(request_id)
            return True

    class _TaskFutures:
        async def async_get_status(self, _request_id: str):
            return sampling_route.FutureStatus.FAILED

    async def never_finishes():
        await asyncio.sleep(3600)

    monkeypatch.delenv("MINT_SAMPLE_AWAIT_TIMEOUT_S", raising=False)
    monkeypatch.setattr(sampling_route, "task_futures", _TaskFutures())

    async def run() -> None:
        with pytest.raises(RuntimeError, match="engine_request_id=req-main_prompt_logprobs"):
            await sampling_route._await_with_external_fail_abort_for_engine_request(
                engine=_Engine(),
                future_request_id="req-main",
                engine_request_id="req-main_prompt_logprobs",
                awaitable=never_finishes(),
            )

    asyncio.run(run())
    assert aborts == ["req-main_prompt_logprobs"]


# --- SGLang actor concurrency (writer-preferring RW lock) -------------------


def test_rwlock_allows_concurrent_readers() -> None:
    from mint_server.backend.sglang_actor import _WriterPreferringRWLock

    lock = _WriterPreferringRWLock()
    barrier = threading.Barrier(3, timeout=5.0)
    overlap_reached = []

    def reader() -> None:
        with lock.read_locked():
            # All three readers must be inside the lock simultaneously; the
            # barrier only trips if the lock did not serialize them.
            idx = barrier.wait()
            overlap_reached.append(idx)

    threads = [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert sorted(overlap_reached) == [0, 1, 2]


def test_rwlock_writer_excludes_readers_and_drains_inflight() -> None:
    from mint_server.backend.sglang_actor import _WriterPreferringRWLock

    lock = _WriterPreferringRWLock()
    events: list[str] = []
    reader_in = threading.Event()
    reader_release = threading.Event()
    writer_acquired = threading.Event()

    def reader() -> None:
        with lock.read_locked():
            events.append("reader_enter")
            reader_in.set()
            reader_release.wait(timeout=5.0)
            events.append("reader_exit")

    def writer() -> None:
        reader_in.wait(timeout=5.0)
        # Reader holds the read lock; the writer must block until it exits.
        with lock.write_locked():
            events.append("writer_enter")
            writer_acquired.set()

    rt = threading.Thread(target=reader)
    wt = threading.Thread(target=writer)
    rt.start()
    wt.start()

    assert reader_in.wait(timeout=5.0)
    # Writer cannot acquire while the reader is still inside.
    assert not writer_acquired.wait(timeout=0.2)
    reader_release.set()
    rt.join(timeout=5.0)
    wt.join(timeout=5.0)

    assert events == ["reader_enter", "reader_exit", "writer_enter"]


def test_rwlock_waiting_writer_blocks_new_readers() -> None:
    from mint_server.backend.sglang_actor import _WriterPreferringRWLock

    lock = _WriterPreferringRWLock()
    order: list[str] = []
    first_reader_in = threading.Event()
    first_reader_release = threading.Event()
    writer_waiting = threading.Event()

    def first_reader() -> None:
        with lock.read_locked():
            first_reader_in.set()
            first_reader_release.wait(timeout=5.0)

    def writer() -> None:
        first_reader_in.wait(timeout=5.0)
        writer_waiting.set()
        with lock.write_locked():
            order.append("writer")

    def late_reader() -> None:
        # Starts after a writer is queued; writer-preference must let the
        # writer go first even though a reader is already active.
        writer_waiting.wait(timeout=5.0)
        time.sleep(0.05)
        with lock.read_locked():
            order.append("late_reader")

    threads = [
        threading.Thread(target=first_reader),
        threading.Thread(target=writer),
        threading.Thread(target=late_reader),
    ]
    for t in threads:
        t.start()
    assert first_reader_in.wait(timeout=5.0)
    assert writer_waiting.wait(timeout=5.0)
    time.sleep(0.1)
    first_reader_release.set()
    for t in threads:
        t.join(timeout=5.0)

    assert order == ["writer", "late_reader"]


def test_sglang_actor_generate_runs_concurrently() -> None:
    # generate_base takes the read lock, so concurrent generate calls overlap
    # inside the actor instead of serializing (SGLang batches them internally).
    concurrency = 4
    barrier = threading.Barrier(concurrency, timeout=5.0)

    class _ConcurrentEngine(_FakeSGLangEngine):
        def generate(self, **kwargs):
            barrier.wait()  # deadlocks/timeouts if calls are serialized
            return super().generate(**kwargs)

    actor = _ready_sglang_actor(_ConcurrentEngine())
    results: list[dict] = []

    def call() -> None:
        results.append(
            actor.generate_base(
                prompt_ids=[1, 2, 3],
                request_id="req",
                max_tokens=1,
            )
        )

    threads = [threading.Thread(target=call) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert len(results) == concurrency
    assert all(not t.is_alive() for t in threads)


def test_sglang_actor_add_lora_replace_does_not_deadlock(monkeypatch) -> None:
    # add_lora_for_session_from_path holds the write lock and internally drops
    # the prior session via _remove_session_locked. The RW write lock is not
    # reentrant, so this must use the locked helper (not the public method).
    import mint_server.backend.sglang_actor as sglang_actor

    fake_engine = _FakeSGLangEngine()
    actor = _ready_sglang_actor(fake_engine)

    actor._session_adapters["sess"] = "adapter_old"
    actor._adapter_paths["adapter_old"] = "/canonical/old"
    actor._path_to_adapter_name["/canonical/old"] = "adapter_old"
    actor._adapter_refcounts["adapter_old"] = 1

    monkeypatch.setattr(actor, "_load_lora_adapter", lambda **_: None)
    monkeypatch.setattr(sglang_actor, "canonical_peft_adapter_path", lambda p: f"/canonical/{p}")
    monkeypatch.setattr(sglang_actor, "validate_sglang_lora_adapter_supported", lambda **_: None)
    monkeypatch.setattr(sglang_actor, "_adapter_name_for_path", lambda p: f"adapter_{p.rsplit('/', 1)[-1]}")

    finished = threading.Event()

    def run() -> None:
        actor.add_lora_for_session_from_path(sampling_session_id="sess", lora_path="new")
        finished.set()

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout=5.0)

    assert finished.is_set(), "add_lora_for_session_from_path deadlocked on the write lock"
    assert "adapter_old" in fake_engine.unload_calls
    assert actor._session_adapters["sess"] == "adapter_new"


def test_sglang_engine_generate_many_dispatches_concurrently(monkeypatch) -> None:
    # generate_many issues all N samples concurrently (asyncio.gather) rather
    # than awaiting them one at a time, and preserves input order in the result.
    engine = SGLangInferenceEngine(
        model_name="Qwen/Qwen3-0.6B",
        model_path="/tmp/model",
        actor_name="mint_sglang_qwen3_0_6b",
        tensor_parallel_size=1,
        max_model_len=16,
    )

    num_samples = 4
    in_flight = 0
    max_in_flight = 0
    release = asyncio.Event()

    async def fake_generate(*, request_id: str, **kwargs):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        if in_flight >= num_samples:
            # Only unblocks once every call is simultaneously in flight, which
            # is impossible under sequential dispatch.
            release.set()
        await asyncio.wait_for(release.wait(), timeout=5.0)
        in_flight -= 1
        return GenerateResult(token_ids=[int(request_id.rsplit("_", 1)[-1])], logprobs=None, stop_reason="length")

    monkeypatch.setattr(engine, "generate", fake_generate)

    async def run() -> None:
        results = await engine.generate_many(
            sampling_session_id="sess",
            prompt_ids=[1, 2],
            request_id="req",
            num_samples=num_samples,
            max_tokens=1,
        )
        # Order preserved: result i corresponds to request_id req_i.
        assert [r.token_ids[0] for r in results] == list(range(num_samples))

    asyncio.run(run())
    assert max_in_flight == num_samples
