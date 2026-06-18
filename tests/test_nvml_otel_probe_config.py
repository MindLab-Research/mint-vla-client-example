from __future__ import annotations

from scripts.tools import start_nvml_otel_probe


def test_nvml_otel_probe_skips_export_without_x_api_key(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "otel.macaron.xin:4317")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "foo=bar")

    assert start_nvml_otel_probe._otel_disabled_reason() == "OTEL_EXPORTER_OTLP_HEADERS missing x-api-key"
    assert start_nvml_otel_probe._configure_otel() is None


def test_nvml_otel_probe_skips_export_without_endpoint(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "x-api-key=secret-key")

    assert start_nvml_otel_probe._otel_disabled_reason() == "OTEL_EXPORTER_OTLP_ENDPOINT is not set"
    assert start_nvml_otel_probe._configure_otel() is None
