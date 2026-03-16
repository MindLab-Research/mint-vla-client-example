from __future__ import annotations

import logging
from pathlib import Path

import tinker_server.logging_context as logging_context


def test_issue_301_configure_logging_fallback_without_structlog(monkeypatch):
    # Keep OTEL disabled in this unit test.
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    saved_structlog = logging_context.structlog
    saved_warned = logging_context._STRUCTLOG_WARNED
    try:
        logging_context.structlog = None
        logging_context._STRUCTLOG_WARNED = False
        logging_context.configure_logging()
        root_logger = logging.getLogger()
        assert root_logger.handlers, "fallback logging should install handlers"
    finally:
        logging_context.structlog = saved_structlog
        logging_context._STRUCTLOG_WARNED = saved_warned


def test_issue_301_actor_entrypoints_call_observability_init():
    repo_root = Path(__file__).resolve().parents[1]
    required = {
        "tinker_server/backend/verl_inference.py": 1,
        "tinker_server/backend/multinode_inference.py": 1,
        "tinker_server/backend/megatron_distributed.py": 2,
        "tinker_server/backend/verl_training.py": 1,
        "tinker_server/backend/api_work_queue.py": 1,
        "tinker_server/backend/capacity_manager.py": 1,
        "tinker_server/backend/future_store.py": 1,
        "tinker_server/backend/session_index_store.py": 1,
        "tinker_server/backend/training_session_store.py": 1,
        "tinker_server/backend/gateway_session_store.py": 1,
    }

    for rel_path, min_count in required.items():
        text = (repo_root / rel_path).read_text(encoding="utf-8")
        got = text.count("init_actor_observability()")
        assert got >= min_count, f"{rel_path} should contain >= {min_count} init_actor_observability() calls, got {got}"


def test_issue_301_sampling_actor_entrypoints_use_traceparent_span_decorator():
    repo_root = Path(__file__).resolve().parents[1]
    required = {
        "tinker_server/backend/verl_inference.py": 5,
        "tinker_server/backend/multinode_inference.py": 3,
    }

    for rel_path, min_count in required.items():
        text = (repo_root / rel_path).read_text(encoding="utf-8")
        got = text.count("@traced_async_from_traceparent(")
        assert got >= min_count, f"{rel_path} should contain >= {min_count} traced actor entrypoints, got {got}"
