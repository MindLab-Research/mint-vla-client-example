from __future__ import annotations

from pathlib import Path
import re

from tinker_server.config import otel_env_vars


def test_issue_290_otel_env_vars_include_app_key_and_skip_empty(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "mint")
    monkeypatch.setenv("MINT_APMPLUS_APP_KEY", "secret-key")
    # Empty headers should be treated as unset and not forwarded.
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "")

    out = otel_env_vars()

    assert out["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://collector:4317"
    assert out["OTEL_SERVICE_NAME"] == "mint"
    assert out["MINT_APMPLUS_APP_KEY"] == "secret-key"
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in out


def test_issue_290_otel_env_vars_supports_legacy_apmplus_alias(monkeypatch):
    monkeypatch.delenv("MINT_APMPLUS_APP_KEY", raising=False)
    monkeypatch.setenv("OTEL_APMPLUS_APP_KEY", "legacy-secret-key")

    out = otel_env_vars()

    assert out["MINT_APMPLUS_APP_KEY"] == "legacy-secret-key"


def test_issue_290_all_actor_runtime_env_call_otel_env_vars():
    repo_root = Path(__file__).resolve().parents[1]
    required = {
        "tinker_server/backend/multi_lora_engine.py": 1,
        "tinker_server/backend/multinode_inference.py": 1,
        "tinker_server/backend/megatron_distributed.py": 2,
        "tinker_server/backend/dense_trainer.py": 1,
        "tinker_server/backend/verl_inference.py": 1,
        "tinker_server/backend/api_work_queue.py": 1,
        "tinker_server/backend/future_store.py": 1,
        "tinker_server/backend/capacity_manager.py": 1,
        "tinker_server/backend/gateway_session_store.py": 1,
        "tinker_server/backend/sampling_session_store.py": 1,
        "tinker_server/backend/session_index_store.py": 1,
        "tinker_server/backend/training_session_store.py": 1,
    }

    for rel_path, min_count in required.items():
        text = (repo_root / rel_path).read_text(encoding="utf-8")
        got = text.count("otel_env_vars()")
        assert got >= min_count, f"{rel_path} should contain >= {min_count} otel_env_vars() calls, got {got}"
        assert re.search(
            r"^\s*from\s+[.\w]+config\s+import\s+.*\botel_env_vars\b",
            text,
            flags=re.MULTILINE,
        ), f"{rel_path} must import otel_env_vars to avoid NameError at runtime"
