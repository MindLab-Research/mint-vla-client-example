from __future__ import annotations

import logging
import sys
import types

import pytest

from mint_server.config.config import _hydrate_otel_env
from mint_server.config.config_file import load_mint_config_file
import mint_server.observability.logging_context as logging_context


def test_config_file_otel_section_loads(tmp_path):
    p = tmp_path / "otel.toml"
    p.write_text(
        "\n".join(
            [
                "[otel]",
                'endpoint = "otel.macaron.xin:4317"',
                'api_key = "secret-key"',
                "insecure = false",
                'headers = "x-api-key=header-key,foo=bar"',
                "metric_export_interval_ms = 5000",
                'deployment_env = "prod"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_mint_config_file(p)

    assert cfg.otel.endpoint == "otel.macaron.xin:4317"
    assert cfg.otel.api_key == "secret-key"
    assert cfg.otel.insecure is False
    assert cfg.otel.headers == "x-api-key=header-key,foo=bar"
    assert cfg.otel.metric_export_interval_ms == 5000
    assert cfg.otel.deployment_env == "prod"


def test_config_file_otel_defaults_empty_section(tmp_path):
    p = tmp_path / "empty.toml"
    p.write_text("[server]\nport = 8000\n", encoding="utf-8")

    cfg = load_mint_config_file(p)

    assert cfg.otel.endpoint is None
    assert cfg.otel.api_key is None
    assert cfg.otel.insecure is None
    assert cfg.otel.headers is None
    assert cfg.otel.metric_export_interval_ms is None
    assert cfg.otel.deployment_env is None


def test_config_file_otel_unknown_key_fails(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("[otel]\nunknown = true\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        load_mint_config_file(p)

    assert "Config validation failed" in str(exc.value)


def test_hydrate_otel_env_from_config_file(tmp_path):
    p = tmp_path / "otel.toml"
    p.write_text(
        "\n".join(
            [
                "[otel]",
                'endpoint = "otel.macaron.xin:4317"',
                'api_key = " secret-key "',
                "insecure = false",
                "metric_export_interval_ms = 7000",
                'deployment_env = " prod "',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_mint_config_file(p)
    environ: dict[str, str] = {}

    _hydrate_otel_env(environ, cfg)

    assert environ == {
        "OTEL_EXPORTER_OTLP_HEADERS": "x-api-key=secret-key",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "otel.macaron.xin:4317",
        "OTEL_EXPORTER_OTLP_INSECURE": "false",
        "OTEL_METRIC_EXPORT_INTERVAL_MS": "7000",
        "MINT_DEPLOYMENT_ENV": "prod",
    }


def test_hydrate_otel_env_does_not_override_existing_env(tmp_path):
    p = tmp_path / "otel.toml"
    p.write_text(
        "\n".join(
            [
                "[otel]",
                'endpoint = "file-endpoint:4317"',
                'api_key = "file-key"',
                "insecure = false",
                "metric_export_interval_ms = 7000",
                'deployment_env = "file-prod"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_mint_config_file(p)
    environ = {
        "OTEL_EXPORTER_OTLP_HEADERS": "x-api-key=env-key",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "env-endpoint:4317",
        "OTEL_EXPORTER_OTLP_INSECURE": "true",
        "OTEL_METRIC_EXPORT_INTERVAL_MS": "3000",
        "MINT_DEPLOYMENT_ENV": "env-prod",
    }

    _hydrate_otel_env(environ, cfg)

    assert environ == {
        "OTEL_EXPORTER_OTLP_HEADERS": "x-api-key=env-key",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "env-endpoint:4317",
        "OTEL_EXPORTER_OTLP_INSECURE": "true",
        "OTEL_METRIC_EXPORT_INTERVAL_MS": "3000",
        "MINT_DEPLOYMENT_ENV": "env-prod",
    }


def test_hydrate_otel_env_headers_used_when_api_key_absent(tmp_path):
    p = tmp_path / "otel.toml"
    p.write_text("[otel]\nheaders = 'x-api-key=header-key,foo=bar'\n", encoding="utf-8")
    cfg = load_mint_config_file(p)
    environ: dict[str, str] = {}

    _hydrate_otel_env(environ, cfg)

    assert environ["OTEL_EXPORTER_OTLP_HEADERS"] == "x-api-key=header-key,foo=bar"


def test_hydrate_otel_env_none_config_safe():
    environ = {"OTEL_EXPORTER_OTLP_ENDPOINT": "env-endpoint:4317"}

    _hydrate_otel_env(environ, None)

    assert environ == {"OTEL_EXPORTER_OTLP_ENDPOINT": "env-endpoint:4317"}


def test_gate_no_x_api_key_skips_otel(monkeypatch, caplog):
    _reset_otel_state(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "otel.macaron.xin:4317")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "foo=bar")

    with caplog.at_level(logging.INFO):
        logging_context._configure_opentelemetry(logging.getLogger("test.otel"))

    assert not logging_context.is_otel_enabled()
    assert "[otel] no api key configured; skipping OTLP export" in caplog.text


def test_gate_with_x_api_key_enables_otel(monkeypatch):
    _reset_otel_state(monkeypatch)
    calls: list[dict[str, object]] = []
    _install_fake_opentelemetry(monkeypatch, calls)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "otel.macaron.xin:4317")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "x-api-key=secret-key")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_INSECURE", "false")

    logging_context._configure_opentelemetry(logging.getLogger("test.otel"))

    assert logging_context.is_otel_enabled()
    exporter_calls = [call for call in calls if call["kind"] in {"span", "metric", "log"}]
    assert exporter_calls
    assert all(call["endpoint"] == "otel.macaron.xin:4317" for call in exporter_calls)
    assert all(call["headers"] == {"x-api-key": "secret-key"} for call in exporter_calls)
    assert all(call["insecure"] is False for call in exporter_calls)


def _reset_otel_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(logging_context, "_OTEL_ENABLED", False)
    monkeypatch.setattr(logging_context, "_OTEL_INITIALIZED", False)
    monkeypatch.setattr(logging_context, "_OTEL_LOG_HANDLER_ATTACHED", False)
    monkeypatch.setattr(logging_context, "_OTEL_RESOURCE_LOGGED", False)
    monkeypatch.setattr(logging_context, "_API_PROCESS_OBSERVABLES_REGISTERED", False)
    monkeypatch.setattr(logging_context, "_TRACER", None)


def _install_fake_opentelemetry(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[dict[str, object]],
) -> None:
    def module(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    opentelemetry = module("opentelemetry")
    metrics = module("opentelemetry.metrics")
    trace = module("opentelemetry.trace")
    logs = module("opentelemetry._logs")
    exporter = module("opentelemetry.exporter")
    exporter_otlp = module("opentelemetry.exporter.otlp")
    exporter_otlp_proto = module("opentelemetry.exporter.otlp.proto")
    exporter_otlp_proto_grpc = module("opentelemetry.exporter.otlp.proto.grpc")
    grpc_log_exporter = module("opentelemetry.exporter.otlp.proto.grpc._log_exporter")
    grpc_metric_exporter = module("opentelemetry.exporter.otlp.proto.grpc.metric_exporter")
    grpc_trace_exporter = module("opentelemetry.exporter.otlp.proto.grpc.trace_exporter")
    sdk = module("opentelemetry.sdk")
    sdk_logs = module("opentelemetry.sdk._logs")
    sdk_logs_export = module("opentelemetry.sdk._logs.export")
    sdk_metrics = module("opentelemetry.sdk.metrics")
    sdk_metrics_export = module("opentelemetry.sdk.metrics.export")
    sdk_resources = module("opentelemetry.sdk.resources")
    sdk_trace = module("opentelemetry.sdk.trace")
    sdk_trace_export = module("opentelemetry.sdk.trace.export")

    setattr(opentelemetry, "metrics", metrics)
    setattr(opentelemetry, "trace", trace)
    setattr(opentelemetry, "exporter", exporter)
    setattr(opentelemetry, "sdk", sdk)
    setattr(exporter, "otlp", exporter_otlp)
    setattr(exporter_otlp, "proto", exporter_otlp_proto)
    setattr(exporter_otlp_proto, "grpc", exporter_otlp_proto_grpc)
    setattr(exporter_otlp_proto_grpc, "_log_exporter", grpc_log_exporter)
    setattr(exporter_otlp_proto_grpc, "metric_exporter", grpc_metric_exporter)
    setattr(exporter_otlp_proto_grpc, "trace_exporter", grpc_trace_exporter)
    setattr(sdk, "_logs", sdk_logs)
    setattr(sdk, "metrics", sdk_metrics)
    setattr(sdk, "resources", sdk_resources)
    setattr(sdk, "trace", sdk_trace)
    setattr(sdk_logs, "export", sdk_logs_export)
    setattr(sdk_metrics, "export", sdk_metrics_export)
    setattr(sdk_trace, "export", sdk_trace_export)

    class _Exporter:
        kind = "unknown"

        def __init__(self, *, endpoint: str, headers: dict[str, str] | None, insecure: bool):
            calls.append(
                {
                    "kind": self.kind,
                    "endpoint": endpoint,
                    "headers": headers,
                    "insecure": insecure,
                }
            )

    class _SpanExporter(_Exporter):
        kind = "span"

    class _MetricExporter(_Exporter):
        kind = "metric"

    class _LogExporter(_Exporter):
        kind = "log"

    class _TracerProvider:
        def __init__(self, *, resource):
            self.resource = resource

        def add_span_processor(self, processor):
            self.processor = processor

    class _MeterProvider:
        def __init__(self, *, metric_readers, resource):
            self.metric_readers = metric_readers
            self.resource = resource

    class _LoggerProvider:
        def __init__(self, *, resource):
            self.resource = resource

        def add_log_record_processor(self, processor):
            self.processor = processor

    class _Processor:
        def __init__(self, exporter):
            self.exporter = exporter

    class _MetricReader:
        def __init__(self, *, exporter, export_interval_millis: int):
            self.exporter = exporter
            self.export_interval_millis = export_interval_millis

    class _Resource:
        def __init__(self, *, attributes):
            self.attributes = attributes

    class _Meter:
        def create_counter(self, *_args, **_kwargs):
            return object()

        def create_histogram(self, *_args, **_kwargs):
            return object()

        def create_observable_gauge(self, *_args, **_kwargs):
            return object()

    class _LoggingHandler(logging.Handler):
        def __init__(self, *, level: int, logger_provider):
            super().__init__(level=level)
            self.logger_provider = logger_provider

    class _Observation:
        def __init__(self, value, attributes):
            self.value = value
            self.attributes = attributes

    setattr(grpc_trace_exporter, "OTLPSpanExporter", _SpanExporter)
    setattr(grpc_metric_exporter, "OTLPMetricExporter", _MetricExporter)
    setattr(grpc_log_exporter, "OTLPLogExporter", _LogExporter)
    setattr(sdk_trace, "TracerProvider", _TracerProvider)
    setattr(sdk_trace_export, "BatchSpanProcessor", _Processor)
    setattr(sdk_metrics, "MeterProvider", _MeterProvider)
    setattr(sdk_metrics_export, "PeriodicExportingMetricReader", _MetricReader)
    setattr(sdk_logs, "LoggerProvider", _LoggerProvider)
    setattr(sdk_logs, "LoggingHandler", _LoggingHandler)
    setattr(sdk_logs_export, "BatchLogRecordProcessor", _Processor)
    setattr(sdk_resources, "Resource", _Resource)
    setattr(metrics, "set_meter_provider", lambda provider: calls.append({"kind": "set_meter_provider", "provider": provider}))
    setattr(metrics, "get_meter", lambda _name: _Meter())
    setattr(metrics, "Observation", _Observation)
    setattr(trace, "set_tracer_provider", lambda provider: calls.append({"kind": "set_tracer_provider", "provider": provider}))
    setattr(trace, "get_tracer", lambda _name: object())
    setattr(logs, "set_logger_provider", lambda provider: calls.append({"kind": "set_logger_provider", "provider": provider}))
