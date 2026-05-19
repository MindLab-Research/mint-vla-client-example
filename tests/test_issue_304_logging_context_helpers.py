from __future__ import annotations

import asyncio
import logging
import sys
import types

import mint_server.logging_context as logging_context


def test_issue_304_bind_request_trace_context_restores_previous_values():
    prev_request_id = logging_context.get_request_id()
    prev_trace_id = logging_context.get_trace_id()
    prev_identity = logging_context.get_request_identity_context()
    try:
        logging_context.set_request_id("req-old")
        logging_context.set_trace_id("a" * 32)
        with logging_context.bind_request_trace_context(
            request_id="req-new",
            trace_id="b" * 32,
            apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
            gateway_request_id="gw-123",
        ):
            assert logging_context.get_request_id() == "req-new"
            assert logging_context.get_trace_id() == "b" * 32
            assert logging_context.get_request_identity_context()["apikey_id"] == "bbbbbbbbbbbbbbbbbbbbbbbb"
            assert logging_context.get_request_identity_context()["gateway_request_id"] == "gw-123"
        assert logging_context.get_request_id() == "req-old"
        assert logging_context.get_trace_id() == "a" * 32
        assert logging_context.get_request_identity_context() == prev_identity
    finally:
        logging_context.set_request_id(prev_request_id)
        logging_context.set_trace_id(prev_trace_id)
        logging_context.request_identity_var.set(prev_identity or None)


def test_issue_304_log_with_bound_context_does_not_leak_context():
    prev_request_id = logging_context.get_request_id()
    prev_trace_id = logging_context.get_trace_id()
    try:
        logging_context.set_request_id("req-parent")
        logging_context.set_trace_id("c" * 32)
        logger = logging.getLogger("tests.test_issue_304")
        logger.addHandler(logging.NullHandler())

        logging_context.log_with_bound_context(
            logger,
            "msg",
            request_id="req-child",
            trace_id="d" * 32,
        )

        assert logging_context.get_request_id() == "req-parent"
        assert logging_context.get_trace_id() == "c" * 32
    finally:
        logging_context.set_request_id(prev_request_id)
        logging_context.set_trace_id(prev_trace_id)


def test_issue_304_run_async_with_otel_span_without_tracer_executes_action(monkeypatch):
    saved_tracer = logging_context._TRACER
    try:
        monkeypatch.setattr(logging_context, "_TRACER", None)

        async def _run() -> int:
            return 7

        out = asyncio.run(
            logging_context.run_async_with_otel_span(
                "test.span",
                _run,
                component="test",
                request_id="rid",
            )
        )
        assert out == 7
    finally:
        logging_context._TRACER = saved_tracer


def test_issue_304_extract_otel_context_from_traceparent_accepts_valid_header(monkeypatch):
    traceparent = "00-" + ("a" * 32) + "-" + ("1" * 16) + "-01"

    class _DummyPropagator:
        def extract(self, carrier):
            assert carrier == {"traceparent": traceparent}
            return {"ctx": "ok"}

    otel_mod = types.ModuleType("opentelemetry")
    otel_mod.__path__ = []
    trace_mod = types.ModuleType("opentelemetry.trace")
    trace_mod.__path__ = []
    propagation_mod = types.ModuleType("opentelemetry.trace.propagation")
    propagation_mod.__path__ = []
    tracecontext_mod = types.ModuleType("opentelemetry.trace.propagation.tracecontext")
    tracecontext_mod.TraceContextTextMapPropagator = _DummyPropagator

    monkeypatch.setitem(sys.modules, "opentelemetry", otel_mod)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", trace_mod)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace.propagation", propagation_mod)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace.propagation.tracecontext", tracecontext_mod)

    ctx = logging_context.extract_otel_context_from_traceparent(traceparent)
    assert ctx is not None


def test_issue_304_extract_otel_context_from_traceparent_rejects_blank():
    assert logging_context.extract_otel_context_from_traceparent("") is None


def test_issue_304_start_as_current_span_from_traceparent_restores_previous_trace_id(monkeypatch):
    saved_tracer = logging_context._TRACER
    prev_trace_id = logging_context.get_trace_id()
    try:
        monkeypatch.setattr(logging_context, "_TRACER", None)
        logging_context.set_trace_id("b" * 32)
        with logging_context.start_as_current_span_from_traceparent(
            "test.span",
            traceparent="00-" + ("a" * 32) + "-" + ("1" * 16) + "-01",
        ):
            assert logging_context.get_trace_id() == "a" * 32
        assert logging_context.get_trace_id() == "b" * 32
    finally:
        logging_context._TRACER = saved_tracer
        logging_context.set_trace_id(prev_trace_id)


def test_issue_304_traced_async_from_traceparent_binds_context(monkeypatch):
    saved_tracer = logging_context._TRACER
    prev_request_id = logging_context.get_request_id()
    prev_trace_id = logging_context.get_trace_id()
    try:
        monkeypatch.setattr(logging_context, "_TRACER", None)

        @logging_context.traced_async_from_traceparent(
            "test.decorated",
            component="test",
            op="test.decorated",
            request_id_arg="request_id",
        )
        async def _decorated(*, request_id: str, traceparent: str | None = None):
            return logging_context.get_request_id(), logging_context.get_trace_id()

        request_id, trace_id = asyncio.run(
            _decorated(
                request_id="req-123",
                traceparent="00-" + ("a" * 32) + "-" + ("1" * 16) + "-01",
            )
        )
        assert request_id == "req-123"
        assert trace_id == "a" * 32
    finally:
        logging_context._TRACER = saved_tracer
        logging_context.set_request_id(prev_request_id)
        logging_context.set_trace_id(prev_trace_id)
