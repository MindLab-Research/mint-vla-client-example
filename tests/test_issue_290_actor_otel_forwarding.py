from __future__ import annotations

from pathlib import Path
import re

from mint_server.config import otel_env_vars


def test_issue_290_otel_env_vars_forward_endpoint_and_api_key_header(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "x-api-key=secret-key")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_INSECURE", "false")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "mint")
    monkeypatch.setenv("MINT_DEPLOYMENT_ENV", "prod")
    monkeypatch.setenv("MINT_CLUSTER_ID", "volcano")

    out = otel_env_vars()

    assert out["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://collector:4317"
    assert out["OTEL_EXPORTER_OTLP_HEADERS"] == "x-api-key=secret-key"
    assert out["OTEL_EXPORTER_OTLP_INSECURE"] == "false"
    assert out["OTEL_SERVICE_NAME"] == "mint"
    assert out["MINT_DEPLOYMENT_ENV"] == "prod"
    assert out["MINT_CLUSTER_ID"] == "volcano"


def test_issue_290_otel_env_vars_skip_empty_header(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "")

    out = otel_env_vars()

    assert out["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://collector:4317"
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in out


def test_issue_290_otel_env_vars_do_not_forward_unlisted_keys(monkeypatch):
    monkeypatch.setenv("OTEL_UNLISTED_KEY", "value")

    out = otel_env_vars()

    assert "OTEL_UNLISTED_KEY" not in out


def test_issue_290_all_actor_runtime_env_call_otel_env_vars():
    repo_root = Path(__file__).resolve().parents[1]
    required = {
        "mint_server/backend/inference/multi_lora_engine.py": 1,
        "mint_server/backend/inference/multinode_inference.py": 1,
        "mint_server/backend/training/megatron/megatron_distributed.py": 2,
        "mint_server/backend/training/dense/dense_trainer.py": 1,
        "mint_server/backend/training/verl/verl_inference.py": 1,
        "mint_server/backend/actors/model_engine_host.py": 1,
        "mint_server/backend/scheduling/model_work_scheduler.py": 1,
        "mint_server/backend/observability/node_metrics_daemon.py": 1,
        "mint_server/backend/stores/task_state_store.py": 1,
        "mint_server/backend/core/config_actor.py": 1,
    }

    for rel_path, min_count in required.items():
        text = (repo_root / rel_path).read_text(encoding="utf-8")
        got = text.count("otel_env_vars()")
        assert got >= min_count, f"{rel_path} should contain >= {min_count} otel_env_vars() calls, got {got}"
        assert re.search(
            r"^\s*from\s+[.\w]+config\s+import\s+(?:[^\n]*\botel_env_vars\b|\([\s\S]*?\botel_env_vars\b)",
            text,
            flags=re.MULTILINE,
        ), f"{rel_path} must import otel_env_vars to avoid NameError at runtime"
