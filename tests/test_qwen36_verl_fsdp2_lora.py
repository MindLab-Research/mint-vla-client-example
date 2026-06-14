from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from mint_server.backend.core.model_registry import get_model_config
from mint_server.backend.qwen36_verl_fsdp2_lora import (
    QWEN36_MODEL_ID,
    QWEN36_VERL_FSDP2_LORA_BACKEND,
    is_qwen36_model,
    qwen36_model_path_override,
)
from mint_server.backend.training.verl.verl_training import _uses_verl_fsdp2_lora_backend
from mint_server.routes.training import (
    _build_create_scheduler_extra,
    _build_training_scheduler_extra,
    _infer_training_backend_for_base_model,
    _supports_control_plane_tokenizer_metadata,
    _training_model_work_domain_key,
)


def _load_repo_sitecustomize():
    path = Path(__file__).resolve().parents[1] / "sitecustomize.py"
    spec = importlib.util.spec_from_file_location("mint_qwen36_sitecustomize", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qwen36_registry_selects_verl_fsdp2_lora_backend() -> None:
    cfg = get_model_config(QWEN36_MODEL_ID)

    assert cfg.training_backend == QWEN36_VERL_FSDP2_LORA_BACKEND
    assert cfg.is_moe is False
    assert cfg.inference_tp == 4
    assert _infer_training_backend_for_base_model(QWEN36_MODEL_ID) == QWEN36_VERL_FSDP2_LORA_BACKEND
    assert _supports_control_plane_tokenizer_metadata(QWEN36_VERL_FSDP2_LORA_BACKEND) is True


def test_qwen36_backend_helpers_fail_closed_to_verl_fsdp2_lora() -> None:
    assert is_qwen36_model("Qwen/Qwen3.6-27B")
    assert is_qwen36_model("/models/qwen3.6-27b")
    assert not is_qwen36_model("Qwen/Qwen3.5-27B")
    assert _uses_verl_fsdp2_lora_backend(QWEN36_MODEL_ID) is True
    assert _uses_verl_fsdp2_lora_backend("Qwen/Qwen3.5-27B") is False


def test_qwen36_model_path_override_prefers_env(monkeypatch) -> None:
    monkeypatch.delenv("MINT_QWEN36_MODEL_PATH", raising=False)
    monkeypatch.delenv("QWEN36_MODEL_PATH", raising=False)
    assert qwen36_model_path_override(QWEN36_MODEL_ID) is None
    assert qwen36_model_path_override("Qwen/Qwen3.5-27B") is None

    monkeypatch.setenv("MINT_QWEN36_MODEL_PATH", "/models/mint-qwen36")
    monkeypatch.setenv("QWEN36_MODEL_PATH", "/models/legacy-qwen36")
    assert qwen36_model_path_override(QWEN36_MODEL_ID) == "/models/mint-qwen36"

    monkeypatch.delenv("MINT_QWEN36_MODEL_PATH", raising=False)
    assert qwen36_model_path_override(QWEN36_MODEL_ID) == "/models/legacy-qwen36"


def test_qwen36_scheduler_domains_use_dedicated_backend_lane(monkeypatch) -> None:
    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")
    session = SimpleNamespace(
        backend=QWEN36_VERL_FSDP2_LORA_BACKEND,
        base_model=QWEN36_MODEL_ID,
    )

    assert (
        _training_model_work_domain_key(
            backend=QWEN36_VERL_FSDP2_LORA_BACKEND,
            base_model=QWEN36_MODEL_ID,
            model_id="run-q36",
        )
        == f"{QWEN36_VERL_FSDP2_LORA_BACKEND}:{QWEN36_MODEL_ID}"
    )

    training_extra = _build_training_scheduler_extra(
        session=session,
        model_id="run-q36",
        training_op="forward_backward",
    )
    assert training_extra["scheduler_enabled"] is True
    assert training_extra["scheduler_domain"] == f"{QWEN36_VERL_FSDP2_LORA_BACKEND}:{QWEN36_MODEL_ID}"
    assert training_extra["scheduler_session_key"] == "run-q36"

    create_extra = _build_create_scheduler_extra(
        base_model=QWEN36_MODEL_ID,
        model_id="run-q36",
        training_op="create_model",
    )
    assert create_extra["backend"] == QWEN36_VERL_FSDP2_LORA_BACKEND
    assert create_extra["scheduler_domain"] == f"{QWEN36_VERL_FSDP2_LORA_BACKEND}:{QWEN36_MODEL_ID}"


def test_sitecustomize_qwen36_patch_gate(monkeypatch) -> None:
    calls: list[str] = []
    fake_patch_module = types.ModuleType("mint_server.backend.qwen36_verl_fsdp2_lora")
    fake_patch_module.install_qwen36_verl_fsdp2_lora_patches = lambda: calls.append("install")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mint_server.backend.qwen36_verl_fsdp2_lora", fake_patch_module)

    monkeypatch.delenv("MINT_QWEN36_VERL_FSDP2_LORA_PATCHES", raising=False)
    module = _load_repo_sitecustomize()
    module._apply_qwen36_verl_fsdp2_lora_patches()
    assert calls == []

    monkeypatch.setenv("MINT_QWEN36_VERL_FSDP2_LORA_PATCHES", "1")
    module._apply_qwen36_verl_fsdp2_lora_patches()
    assert calls == ["install"]
