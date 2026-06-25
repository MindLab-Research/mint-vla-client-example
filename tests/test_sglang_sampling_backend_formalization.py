from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import anyio

from mint_server.backend.contracts.control_plane_contracts import AppendWorkResult
from mint_server.backend.scheduling.model_work_admission import ModelWorkAdmissionResult
from mint_server.backend.actors.model_actor_inventory import (
    ActorEntry,
    ActorType,
    ModelActorInventory,
    _backend_for_entry,
    _role_for_entry,
)
from mint_server.backend.actors.model_actor_launchers import (
    BUMBLEBEE_RUNTIME_ENV_KEYS,
    MODEL_RUNTIME_LAUNCHER_ENV_KEYS,
    SGLANG_RUNTIME_ENV_KEYS,
    bumblebee_env_for_spec,
    default_model_actor_launcher_registry,
    launcher_process_env,
    placement_env_for_spec,
    sglang_env_for_spec,
)
from mint_server.backend.actors.model_actor_supervisor import (
    ModelActorSpec,
    _spec_for_scheduler_domain_from_env,
    domain_key_for_sampling_base_model,
    domain_key_for_vllm_base_model,
)
from mint_server.backend.core.model_registry import MODEL_CONFIGS, ModelConfig
from mint_server.backend.sampling_backend import (
    actor_name_for_sampling_base_model,
    base_model_from_sampling_domain_key,
    normalize_sampling_backend,
    sampling_backend_from_domain_key,
)
from mint_server.models.types import ComputeLogprobsRequest, ModelInput, SampleRequest, SamplingParams
from mint_server.config.runtime_config import CONFIG_ACTOR_ENV_EXCLUDED_KEYS, SNAPSHOT_CONFIG_ENV_KEYS
from mint_server.routes import sampling as sampling_routes


def _admitted_model_work(request_id: str) -> ModelWorkAdmissionResult:
    return ModelWorkAdmissionResult(
        request_id=str(request_id),
        scheduler_result=AppendWorkResult(ok=True, request_id=str(request_id)),
    )


def test_model_config_defaults_to_vllm_serving_backend() -> None:
    assert MODEL_CONFIGS["Qwen/Qwen3-0.6B"].serving_backend == "vllm"

    cfg = ModelConfig(
        num_parameters=1.0,
        is_moe=False,
        inference_tp=1,
        inference_dp=1,
        max_model_len=1024,
    )
    assert cfg.serving_backend == "vllm"


def test_model_config_can_represent_sglang_serving_backend() -> None:
    cfg = ModelConfig(
        num_parameters=1.0,
        is_moe=False,
        inference_tp=1,
        inference_dp=1,
        max_model_len=1024,
        serving_backend="sglang",
    )
    assert cfg.serving_backend == "sglang"


def test_sampling_domain_helpers_keep_vllm_backward_compatible() -> None:
    assert domain_key_for_sampling_base_model("Qwen/A") == "vllm:Qwen/A"
    assert domain_key_for_vllm_base_model("Qwen/A") == "vllm:Qwen/A"
    assert sampling_backend_from_domain_key("vllm:Qwen/A") == "vllm"
    assert base_model_from_sampling_domain_key("vllm:Qwen/A") == "Qwen/A"


def test_sampling_domain_helpers_support_sglang_without_vllm_disguise() -> None:
    assert domain_key_for_sampling_base_model("Qwen/A", backend="sglang") == "sglang:Qwen/A"
    assert sampling_backend_from_domain_key("sglang:Qwen/A") == "sglang"
    assert base_model_from_sampling_domain_key("sglang:Qwen/A") == "Qwen/A"


def test_sampling_actor_names_are_backend_specific() -> None:
    assert actor_name_for_sampling_base_model("Qwen/Qwen3-0.6B") == "mint_vllm_qwen3_0_6b"
    assert actor_name_for_sampling_base_model("Qwen/Qwen3-0.6B", backend="sglang") == "mint_sglang_qwen3_0_6b"


def test_normalize_sampling_backend_rejects_unknown_backend() -> None:
    try:
        normalize_sampling_backend("not-a-backend")
    except ValueError as e:
        assert "unsupported sampling serving backend" in str(e)
    else:
        raise AssertionError("unknown sampling backend should fail loudly")


def test_actor_type_sglang_is_inference_not_vllm() -> None:
    entry = ActorEntry(actor_name="mint_sglang_qwen3_0_6b", actor_type=ActorType.SGLANG, num_gpus=1)

    assert entry.actor_type is ActorType.SGLANG
    assert _role_for_entry(entry) == "inference"
    assert _backend_for_entry(entry) == "sglang"


def test_sglang_inventory_record_is_backend_specific_inference_actor() -> None:
    inventory = ModelActorInventory()
    inventory.clear(kill_actors=False)
    try:
        inventory.register(
            actor_name="mint_sglang_qwen3_0_6b",
            actor_type=ActorType.SGLANG,
            num_gpus=1,
            base_model="Qwen/Qwen3-0.6B",
            metadata={"serving_backend": "sglang"},
        )
        inventory.mark_ready("mint_sglang_qwen3_0_6b")

        records = inventory.list_actors(refresh_metadata=False, actor_type=ActorType.SGLANG)

        assert len(records) == 1
        assert records[0]["actor_type"] == "sglang"
        assert records[0]["backend"] == "sglang"
        assert records[0]["role"] == "inference"
        assert records[0]["base_model"] == "Qwen/Qwen3-0.6B"
    finally:
        inventory.clear(kill_actors=False)


def test_supervisor_autocreates_sglang_spec_after_launcher_exists() -> None:
    spec = _spec_for_scheduler_domain_from_env("sglang:Qwen/Qwen3-0.6B")

    assert spec is not None
    assert spec.domain_key == "sglang:Qwen/Qwen3-0.6B"
    assert spec.base_model == "Qwen/Qwen3-0.6B"
    assert spec.launcher_key == "sglang"
    assert spec.normalized_actor_name().startswith("mint_model_runtime_sglang_")


def test_launcher_registry_knows_sglang_launcher() -> None:
    registry = default_model_actor_launcher_registry()

    assert registry.resolve("sglang") is registry.resolve("vllm")


def test_sglang_placement_env_uses_sglang_specific_key_not_vllm() -> None:
    env = placement_env_for_spec(
        ModelActorSpec(
            domain_key="sglang:Qwen/Qwen3-0.6B",
            base_model="Qwen/Qwen3-0.6B",
            node_pin="192.0.2.10",
            gpu_count=1,
        )
    )

    assert "MINT_SGLANG_MODEL_PLACEMENT_JSON" in env
    assert "MINT_VLLM_MODEL_PLACEMENT_JSON" not in env
    assert "Qwen/Qwen3-0.6B" in env["MINT_SGLANG_MODEL_PLACEMENT_JSON"]


def test_bumblebee_placement_env_uses_bumblebee_specific_key_not_vllm() -> None:
    env = placement_env_for_spec(
        ModelActorSpec(
            domain_key="bumblebee:mint_megatron_qwen3_30b_a3b_instruct_2507",
            base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
            node_pin="192.0.2.20",
            gpu_count=4,
        )
    )

    assert "MINT_BUMBLEBEE_MODEL_PLACEMENT_JSON" in env
    assert "MINT_VLLM_MODEL_PLACEMENT_JSON" not in env
    assert "Qwen/Qwen3-30B-A3B-Instruct-2507" in env["MINT_BUMBLEBEE_MODEL_PLACEMENT_JSON"]


def test_sglang_runtime_env_forwards_only_sglang_specs(monkeypatch) -> None:
    monkeypatch.setenv("MINT_SGLANG_PYTHONPATH", "/tmp/sglang")
    monkeypatch.setenv("MINT_SGLANG_PY_EXECUTABLE", "/tmp/sglang-venv/bin/python")
    monkeypatch.setenv("MINT_SGLANG_ACTOR_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("MINT_SGLANG_MODEL_PLACEMENT_JSON", '{"Qwen/Test":{"node_ip":"10.0.0.7","gpu_count":1}}')
    monkeypatch.setenv("MINT_SGLANG_MODEL_RUNTIME_MAX_CLAIM", "3")
    monkeypatch.setenv("MINT_SGLANG_MODEL_RUNTIME_TOKEN_BUDGET", "2048")
    monkeypatch.setenv("MINT_SGLANG_MAX_LORA_RANK", "16")
    monkeypatch.setenv("MINT_SGLANG_LORA_TARGET_MODULES", "q_proj,v_proj")
    monkeypatch.setenv("MINT_SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH", "1")
    monkeypatch.setenv("MINT_SGLANG_DISABLE_FLASHINFER_KERNELS", "1")
    monkeypatch.setenv("MINT_SGLANG_GENERATE_TIMEOUT_S", "600")
    monkeypatch.setenv("MINT_SGLANG_SLOW_REQUEST_LOG_S", "30")
    monkeypatch.setenv("MINT_SGLANG_PYCACHE_PREFIX", "/tmp/sglang-pycache")

    assert sglang_env_for_spec(ModelActorSpec(domain_key="vllm:Qwen/Qwen3-0.6B")) == {}
    env = sglang_env_for_spec(ModelActorSpec(domain_key="sglang:Qwen/Qwen3-0.6B"))

    assert env["MINT_SGLANG_PYTHONPATH"] == "/tmp/sglang"
    assert env["MINT_SGLANG_PY_EXECUTABLE"] == "/tmp/sglang-venv/bin/python"
    assert env["MINT_SGLANG_ACTOR_MAX_CONCURRENCY"] == "1"
    assert env["MINT_SGLANG_MODEL_PLACEMENT_JSON"] == '{"Qwen/Test":{"node_ip":"10.0.0.7","gpu_count":1}}'
    assert "MINT_SGLANG_MODEL_RUNTIME_MAX_CLAIM" not in env
    assert "MINT_SGLANG_MODEL_RUNTIME_TOKEN_BUDGET" not in env
    assert env["MINT_SGLANG_MAX_LORA_RANK"] == "16"
    assert env["MINT_SGLANG_LORA_TARGET_MODULES"] == "q_proj,v_proj"
    assert env["MINT_SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH"] == "1"
    assert env["MINT_SGLANG_DISABLE_FLASHINFER_KERNELS"] == "1"
    assert env["MINT_SGLANG_GENERATE_TIMEOUT_S"] == "600"
    assert env["MINT_SGLANG_SLOW_REQUEST_LOG_S"] == "30"
    assert env["MINT_SGLANG_PYCACHE_PREFIX"] == "/tmp/sglang-pycache"


def test_bumblebee_runtime_env_forwards_only_bumblebee_specs(monkeypatch) -> None:
    monkeypatch.setenv("MINT_BUMBLEBEE_REPO_PATH", "/tmp/bumblebee")
    monkeypatch.setenv("MINT_BUMBLEBEE_MEGATRON_LM_PATH", "/tmp/Megatron-LM")
    monkeypatch.setenv("MINT_BUMBLEBEE_ATTENTION_BACKEND", "unfused")
    monkeypatch.setenv("MINT_BUMBLEBEE_MODEL_PLACEMENT_JSON", '{"Qwen/Test":{"node_ip":"10.0.0.7","gpu_count":4}}')
    monkeypatch.setenv("MINT_MODEL_PLACEMENT_JSON", '{"Qwen/Test":{"node_ip":"10.0.0.8","gpu_count":4}}')
    monkeypatch.setenv("MINT_BUMBLEBEE_MODEL_RUNTIME_TOKEN_BUDGET", "4096")

    assert bumblebee_env_for_spec(ModelActorSpec(domain_key="megatron:Qwen/Test")) == {}
    env = bumblebee_env_for_spec(ModelActorSpec(domain_key="bumblebee:Qwen/Test"))

    assert env["MINT_BUMBLEBEE_REPO_PATH"] == "/tmp/bumblebee"
    assert env["MINT_BUMBLEBEE_MEGATRON_LM_PATH"] == "/tmp/Megatron-LM"
    assert env["MINT_BUMBLEBEE_ATTENTION_BACKEND"] == "unfused"
    assert "MINT_BUMBLEBEE_MODEL_PLACEMENT_JSON" not in env
    assert "MINT_MODEL_PLACEMENT_JSON" not in env
    assert "MINT_BUMBLEBEE_MODEL_RUNTIME_TOKEN_BUDGET" not in env


def test_launcher_process_env_includes_bumblebee_runtime_env(monkeypatch) -> None:
    monkeypatch.setenv("MINT_BUMBLEBEE_REPO_PATH", "/tmp/bumblebee")
    monkeypatch.setenv("MINT_BUMBLEBEE_MODEL_RUNTIME_TOKEN_BUDGET", "4096")

    env = launcher_process_env()

    assert env["MINT_BUMBLEBEE_REPO_PATH"] == "/tmp/bumblebee"
    assert env["MINT_BUMBLEBEE_MODEL_RUNTIME_TOKEN_BUDGET"] == "4096"


def test_bumblebee_env_boundary_lists_stay_in_sync() -> None:
    from mint_server.backend.training.bumblebee.bumblebee_distributed import (
        BUMBLEBEE_RUNTIME_ENV_PASSTHROUGH_KEYS,
    )

    runtime_keys = set(BUMBLEBEE_RUNTIME_ENV_KEYS)
    passthrough_keys = set(BUMBLEBEE_RUNTIME_ENV_PASSTHROUGH_KEYS)
    placement_keys = {"MINT_BUMBLEBEE_MODEL_PLACEMENT_JSON", "MINT_MODEL_PLACEMENT_JSON"}

    assert runtime_keys
    assert passthrough_keys - placement_keys <= runtime_keys
    assert "MINT_BUMBLEBEE_REPO_PATH" in runtime_keys
    assert "MINT_BUMBLEBEE_MEGATRON_LM_PATH" in runtime_keys
    assert "MINT_BUMBLEBEE_MODEL_PLACEMENT_JSON" not in runtime_keys
    assert "MINT_MODEL_PLACEMENT_JSON" not in runtime_keys
    assert "MINT_BUMBLEBEE_MODEL_RUNTIME_TOKEN_BUDGET" not in runtime_keys


def test_sglang_env_boundary_lists_stay_in_sync() -> None:
    runtime_keys = set(SGLANG_RUNTIME_ENV_KEYS)
    launcher_process_keys = set(MODEL_RUNTIME_LAUNCHER_ENV_KEYS)

    assert runtime_keys
    assert all(key.startswith("MINT_SGLANG_") for key in runtime_keys)
    assert "MINT_SGLANG_MODEL_PLACEMENT_JSON" in runtime_keys
    assert "MINT_SGLANG_PY_EXECUTABLE" in runtime_keys
    assert "MINT_SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH" in runtime_keys
    assert "MINT_SGLANG_DISABLE_FLASHINFER_KERNELS" in runtime_keys
    assert "MINT_SGLANG_PYCACHE_PREFIX" in runtime_keys
    assert "MINT_SGLANG_MODEL_RUNTIME_MAX_CLAIM" not in runtime_keys
    assert "MINT_SGLANG_MODEL_RUNTIME_TOKEN_BUDGET" not in runtime_keys
    assert "MINT_SGLANG_ADMIN_SHUTDOWN_TIMEOUT_S" not in runtime_keys

    model_runtime_sglang_keys = {
        "MINT_SGLANG_MODEL_RUNTIME_MAX_CLAIM",
        "MINT_SGLANG_MODEL_RUNTIME_TOKEN_BUDGET",
    }
    assert model_runtime_sglang_keys <= launcher_process_keys

    snapshot_runtime_keys = runtime_keys - {"MINT_SGLANG_MODEL_PLACEMENT_JSON"}
    assert snapshot_runtime_keys <= SNAPSHOT_CONFIG_ENV_KEYS
    assert "MINT_SGLANG_MODEL_PLACEMENT_JSON" in CONFIG_ACTOR_ENV_EXCLUDED_KEYS
    assert "MINT_SGLANG_MODEL_PLACEMENT_JSON" not in SNAPSHOT_CONFIG_ENV_KEYS
    assert model_runtime_sglang_keys <= SNAPSHOT_CONFIG_ENV_KEYS
    assert "MINT_SGLANG_ADMIN_SHUTDOWN_TIMEOUT_S" in SNAPSHOT_CONFIG_ENV_KEYS


def test_sampling_route_import_does_not_import_external_sglang_package() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import importlib
import sys

real_import = builtins.__import__


def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "sglang" or str(name).startswith("sglang."):
        raise AssertionError(f"unexpected external SGLang import: {name}")
    return real_import(name, globals, locals, fromlist, level)


builtins.__import__ = guarded_import
importlib.import_module("mint_server.routes.sampling")
loaded = sorted(name for name in sys.modules if name == "sglang" or name.startswith("sglang."))
assert not loaded, loaded
print("ok")
'''

    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert proc.stdout.strip() == "ok"


def test_sampling_route_keeps_vllm_domain_by_default(monkeypatch) -> None:
    cfg = ModelConfig(
        num_parameters=1.0,
        is_moe=False,
        inference_tp=1,
        inference_dp=1,
        max_model_len=1024,
        serving_backend="vllm",
    )
    monkeypatch.setattr("mint_server.backend.core.model_registry.get_model_config", lambda _model: cfg)

    assert sampling_routes._model_work_domain_key("Qwen/Qwen3-0.6B") == "vllm:Qwen/Qwen3-0.6B"


def test_asample_enqueue_keeps_vllm_backend_by_default(monkeypatch) -> None:
    cfg = ModelConfig(
        num_parameters=1.0,
        is_moe=False,
        inference_tp=1,
        inference_dp=1,
        max_model_len=1024,
        serving_backend="vllm",
    )
    monkeypatch.setattr("mint_server.backend.core.model_registry.get_model_config", lambda _model: cfg)
    snapshot = sampling_routes.SamplingSessionSnapshot(
        session_id="session-vllm-default",
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

    calls: list[dict] = []

    async def _enqueue(**kwargs):
        calls.append(dict(kwargs))
        return _admitted_model_work(str(kwargs["request_id"]))

    monkeypatch.setattr(sampling_routes, "_async_get_http_sampling_snapshot", _get_snapshot)
    monkeypatch.setattr(sampling_routes, "enqueue_model_work", _enqueue)
    monkeypatch.setattr(sampling_routes, "record_sampling_admission_metric", lambda **_kwargs: None)

    req = SampleRequest(
        sampling_session_id="session-vllm-default",
        num_samples=2,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=5),
    )
    http_request = SimpleNamespace(state=SimpleNamespace(user_data=None), headers={})

    out = anyio.run(sampling_routes.asample, req, http_request)

    assert out.request_id
    assert len(calls) == 1
    call = calls[0]
    assert call["domain_key"] == "vllm:Qwen/Qwen3-0.6B"
    assert call["affinity_group"] == "base:Qwen/Qwen3-0.6B"
    assert call["ordering_key"] == "session:session-vllm-default"
    assert call["token_cost"] == 16
    assert call["queued_meta"]["backend"] == "vllm"
    assert call["queued_meta"]["domain_key"] == "vllm:Qwen/Qwen3-0.6B"
    assert call["queued_meta"]["token_cost"] == 16
    assert call["trace_kwargs"]["backend"] == "vllm"
    assert call["trace_kwargs"]["domain_key"] == "vllm:Qwen/Qwen3-0.6B"


def test_compute_logprobs_enqueue_keeps_vllm_backend_by_default(monkeypatch) -> None:
    cfg = ModelConfig(
        num_parameters=1.0,
        is_moe=False,
        inference_tp=1,
        inference_dp=1,
        max_model_len=1024,
        serving_backend="vllm",
    )
    monkeypatch.setattr("mint_server.backend.core.model_registry.get_model_config", lambda _model: cfg)
    snapshot = sampling_routes.SamplingSessionSnapshot(
        session_id="session-vllm-default",
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

    calls: list[dict] = []

    async def _enqueue(**kwargs):
        calls.append(dict(kwargs))
        return _admitted_model_work(str(kwargs["request_id"]))

    monkeypatch.setattr(sampling_routes, "_async_get_http_sampling_snapshot", _get_snapshot)
    monkeypatch.setattr("mint_server.backend.scheduling.model_work_admission.enqueue_model_work", _enqueue)
    monkeypatch.setattr(sampling_routes, "record_sampling_admission_metric", lambda **_kwargs: None)
    monkeypatch.setattr("mint_server.backend.scheduling.model_work_scheduler.model_work_scheduler", SimpleNamespace())
    monkeypatch.setattr(sampling_routes, "task_futures", SimpleNamespace())

    req = ComputeLogprobsRequest(
        sampling_session_id="session-vllm-default",
        seq_id=0,
        sequence=ModelInput.from_ints([1, 2, 3]),
    )
    http_request = SimpleNamespace(state=SimpleNamespace(user_data=None), headers={})

    out = anyio.run(sampling_routes.compute_logprobs, req, http_request)

    assert out.request_id
    assert len(calls) == 1
    call = calls[0]
    assert call["op"] == "sampling.compute_logprobs"
    assert call["domain_key"] == "vllm:Qwen/Qwen3-0.6B"
    assert call["affinity_group"] == "base:Qwen/Qwen3-0.6B"
    assert call["ordering_key"] == "session:session-vllm-default"
    assert call["token_cost"] == 3
    assert call["queued_meta"]["backend"] == "vllm"
    assert call["queued_meta"]["domain_key"] == "vllm:Qwen/Qwen3-0.6B"
    assert call["trace_kwargs"]["backend"] == "vllm"
    assert call["trace_kwargs"]["domain_key"] == "vllm:Qwen/Qwen3-0.6B"


def test_sampling_route_uses_sglang_domain_when_model_config_selects_sglang(monkeypatch) -> None:
    cfg = ModelConfig(
        num_parameters=1.0,
        is_moe=False,
        inference_tp=1,
        inference_dp=1,
        max_model_len=1024,
        serving_backend="sglang",
    )
    monkeypatch.setattr("mint_server.backend.core.model_registry.get_model_config", lambda _model: cfg)

    assert sampling_routes._model_work_domain_key("Qwen/Qwen3-0.6B") == "sglang:Qwen/Qwen3-0.6B"
