"""Request context propagation and structured logging configuration.

Environment Variables:
    MINT_LOG_FILE: Log file path (default: /tmp/tinker_server.log)
    MINT_LOG_MAX_BYTES: Max log file size before rotation (default: 100MB)
    MINT_LOG_BACKUP_COUNT: Number of backup files to keep (default: 5)
    OTEL_EXPORTER_OTLP_ENDPOINT: Collector or APM endpoint (default: disabled)
    OTEL_EXPORTER_OTLP_HEADERS: OTLP headers (e.g. "x-byteapm-appkey=xxx")
    OTEL_EXPORTER_OTLP_INSECURE: grpc insecure transport (default: true)
    OTEL_SERVICE_NAME: service.name resource attribute (default: mint)
    OTEL_METRIC_EXPORT_INTERVAL_MS: metrics export interval (default: 60000)
    OTEL_LOG_LEVEL: OTLP log handler level (default: INFO)
    MINT_APMPLUS_APP_KEY: optional shortcut for x-byteapm-appkey header
"""

from __future__ import annotations

import contextvars
import logging
import logging.handlers
import os
import re
import sys
import threading
import uuid
from typing import Any

try:
    import structlog
except Exception:
    structlog = None

# Context variable to store current request_id
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
# Context variable to store current trace_id
trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)

_HEX_CHARS = frozenset("0123456789abcdef")
_OTEL_ENABLED = False
_OTEL_INITIALIZED = False
_OTEL_LOG_HANDLER_ATTACHED = False
_STRUCTLOG_WARNED = False
_HTTP_REQUEST_COUNTER: Any | None = None
_HTTP_DURATION_HISTOGRAM: Any | None = None
_HTTP_ERROR_COUNTER: Any | None = None
_TRACER: Any | None = None
_OP_PREFIX_RE = re.compile(r"^\[([A-Za-z0-9_.:-]+)\]")
_ACTOR_OBS_INITIALIZED = False
_ACTOR_OBS_LOCK = threading.Lock()


def set_request_id(request_id: str | None) -> None:
    """Set the request_id for the current context."""
    request_id_var.set(request_id)


def get_request_id() -> str | None:
    """Get the request_id from the current context."""
    return request_id_var.get()


def _normalize_trace_id(trace_id: str | None) -> str | None:
    if not trace_id:
        return None
    normalized = str(trace_id).strip().lower().replace("-", "")
    if len(normalized) != 32:
        return None
    if any(ch not in _HEX_CHARS for ch in normalized):
        return None
    if normalized == "0" * 32:
        return None
    return normalized


def set_trace_id(trace_id: str | None) -> None:
    """Set the trace_id for the current context."""
    trace_id_var.set(_normalize_trace_id(trace_id))


def get_trace_id() -> str | None:
    """Get the trace_id from the current context."""
    return _normalize_trace_id(trace_id_var.get())


def generate_trace_id() -> str:
    """Generate an OpenTelemetry-compatible trace_id (32 lowercase hex chars)."""
    return uuid.uuid4().hex


def _get_current_otel_trace_id() -> str | None:
    """Return current OTel trace_id if a valid span context is active."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span is None:
            return None
        span_context = span.get_span_context()
        trace_id_int = getattr(span_context, "trace_id", 0)
        if not isinstance(trace_id_int, int) or trace_id_int == 0:
            return None
        return _normalize_trace_id(f"{trace_id_int:032x}")
    except Exception:
        return None


def extract_trace_id_from_traceparent(traceparent: str | None) -> str | None:
    """Extract trace_id from W3C traceparent header."""
    if not traceparent:
        return None
    parts = str(traceparent).strip().split("-")
    if len(parts) != 4:
        return None
    return _normalize_trace_id(parts[1])


def ensure_trace_id(preferred_trace_id: str | None = None) -> str:
    """Ensure trace_id is available in context.

    Priority: preferred -> existing context -> current OTel span -> generated.
    """
    trace_id = _normalize_trace_id(preferred_trace_id)
    if trace_id is None:
        trace_id = get_trace_id()
    if trace_id is None:
        trace_id = _get_current_otel_trace_id()
    if trace_id is None:
        trace_id = generate_trace_id()
    set_trace_id(trace_id)
    return trace_id


def add_request_id(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Structlog processor to add request_id to all log events."""
    request_id = get_request_id()
    event_dict["request_id"] = request_id if request_id else "-"
    return event_dict


def add_trace_id(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Structlog processor to add trace_id to all log events."""
    trace_id = get_trace_id() or _get_current_otel_trace_id()
    event_dict["trace_id"] = trace_id if trace_id else "-"
    return event_dict


def add_component(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Structlog processor to add component field for easier incident filtering."""
    if isinstance(event_dict.get("component"), str) and event_dict["component"]:
        return event_dict
    name = str(event_dict.get("logger") or getattr(logger, "name", "") or "")
    if name.startswith("tinker_server."):
        name = name[len("tinker_server.") :]
    event_dict["component"] = name if name else "unknown"
    return event_dict


def add_operation(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Infer `op` from common `[op] ...` log prefixes when caller didn't provide one."""
    if isinstance(event_dict.get("op"), str) and event_dict["op"]:
        return event_dict
    event = event_dict.get("event")
    if isinstance(event, str):
        match = _OP_PREFIX_RE.match(event.strip())
        if match:
            event_dict["op"] = match.group(1)
    return event_dict


def classify_failure_reason(error: Exception) -> str:
    """Best-effort failure categorization for incident-first logs."""
    typ = type(error).__name__.lower()
    msg = str(error).lower()
    text = f"{typ} {msg}"
    if "timeout" in text:
        return "timeout"
    if "cancel" in text:
        return "canceled"
    if "oom" in text or "out of memory" in text or "resource_exhausted" in text:
        return "resource_exhausted"
    if "permission" in text or "forbidden" in text or "access denied" in text:
        return "permission_denied"
    if "not found" in text:
        return "not_found"
    if "validation" in text or "invalid" in text:
        return "validation"
    return "internal_error"


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}


def _parse_headers(raw: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not raw:
        return out
    for pair in str(raw).split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            out[key] = value
    return out


def _configure_opentelemetry(root_logger: logging.Logger) -> None:
    """Configure OTLP trace/metric/log export (APMPlus or collector)."""
    global _OTEL_ENABLED, _OTEL_INITIALIZED, _OTEL_LOG_HANDLER_ATTACHED
    global _HTTP_REQUEST_COUNTER, _HTTP_DURATION_HISTOGRAM, _HTTP_ERROR_COUNTER, _TRACER

    if _OTEL_INITIALIZED:
        return

    endpoint = (os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip()
    if not endpoint:
        return

    try:
        from opentelemetry import metrics, trace
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception as e:
        print(f"Warning: OpenTelemetry dependencies unavailable: {e}", file=sys.stderr)
        return

    service_name = (os.getenv("OTEL_SERVICE_NAME") or "mint").strip()
    resource = Resource.create({"service.name": service_name})
    headers = _parse_headers(os.getenv("OTEL_EXPORTER_OTLP_HEADERS"))
    app_key = (os.getenv("MINT_APMPLUS_APP_KEY") or "").strip()
    if app_key and "x-byteapm-appkey" not in headers:
        headers["x-byteapm-appkey"] = app_key
    insecure = _parse_bool_env("OTEL_EXPORTER_OTLP_INSECURE", default=True)

    try:
        # 1) Traces
        span_exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers or None, insecure=insecure)
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)
        _TRACER = trace.get_tracer("mint.http")

        # 2) Metrics
        export_interval_ms = int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL_MS", "60000"))
        metric_exporter = OTLPMetricExporter(endpoint=endpoint, headers=headers or None, insecure=insecure)
        metric_reader = PeriodicExportingMetricReader(
            exporter=metric_exporter,
            export_interval_millis=max(1000, export_interval_ms),
        )
        meter_provider = MeterProvider(metric_readers=[metric_reader], resource=resource)
        metrics.set_meter_provider(meter_provider)
        meter = metrics.get_meter("mint.http")
        _HTTP_REQUEST_COUNTER = meter.create_counter(
            "mint_http_server_requests_total",
            unit="{request}",
            description="Total HTTP requests handled by mint",
        )
        _HTTP_ERROR_COUNTER = meter.create_counter(
            "mint_http_server_errors_total",
            unit="{error}",
            description="Total HTTP 5xx responses handled by mint",
        )
        _HTTP_DURATION_HISTOGRAM = meter.create_histogram(
            "mint_http_server_request_duration_ms",
            unit="ms",
            description="HTTP request duration in milliseconds",
        )

        # 3) Logs
        log_exporter = OTLPLogExporter(endpoint=endpoint, headers=headers or None, insecure=insecure)
        logger_provider = LoggerProvider(resource=resource)
        logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
        set_logger_provider(logger_provider)
        if not _OTEL_LOG_HANDLER_ATTACHED:
            level_name = (os.getenv("OTEL_LOG_LEVEL") or "INFO").upper()
            level = getattr(logging, level_name, logging.INFO)
            root_logger.addHandler(LoggingHandler(level=level, logger_provider=logger_provider))
            _OTEL_LOG_HANDLER_ATTACHED = True

        # Optional logging instrumentation: do not fail if package is absent.
        try:
            from opentelemetry.instrumentation.logging import LoggingInstrumentor

            LoggingInstrumentor().instrument(set_logging_format=False)
        except Exception:
            pass

        _OTEL_ENABLED = True
        _OTEL_INITIALIZED = True
    except Exception as e:
        print(f"Warning: Failed to configure OpenTelemetry exporters: {e}", file=sys.stderr)


def get_otel_tracer() -> Any | None:
    return _TRACER


def is_otel_enabled() -> bool:
    return bool(_OTEL_ENABLED)


def record_http_server_metrics(*, method: str, route: str, status_code: int, duration_ms: float) -> None:
    if not _OTEL_ENABLED:
        return
    attrs = {
        "http.method": str(method),
        "http.route": str(route),
        "http.status_code": int(status_code),
    }
    try:
        if _HTTP_REQUEST_COUNTER is not None:
            _HTTP_REQUEST_COUNTER.add(1, attributes=attrs)
        if status_code >= 500 and _HTTP_ERROR_COUNTER is not None:
            _HTTP_ERROR_COUNTER.add(1, attributes=attrs)
        if _HTTP_DURATION_HISTOGRAM is not None:
            _HTTP_DURATION_HISTOGRAM.record(float(duration_ms), attributes=attrs)
    except Exception:
        # Metrics are best-effort and must never break request handling.
        pass


def _configure_stdlib_logging(
    *,
    root_logger: logging.Logger,
    log_file: str,
    log_max_bytes: int,
    log_backup_count: int,
) -> None:
    """Configure plain stdlib logging (fallback when structlog is unavailable)."""
    root_logger.setLevel(logging.DEBUG)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    fmt = "%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s trace_id=%(trace_id)s %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S%z"

    class _ContextFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if not hasattr(record, "request_id"):
                record.request_id = get_request_id() or "-"
            if not hasattr(record, "trace_id"):
                record.trace_id = get_trace_id() or _get_current_otel_trace_id() or "-"
            return True

    context_filter = _ContextFilter()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(context_filter)
    console_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    root_logger.addHandler(console_handler)

    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=log_max_bytes,
            backupCount=log_backup_count,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.addFilter(context_filter)
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Failed to configure file logging to {log_file}: {e}", file=sys.stderr)


def configure_logging() -> None:
    """Configure structlog + stdlib logging and optional OTLP exporters."""
    # Environment variables
    log_file = os.getenv("MINT_LOG_FILE", "/tmp/tinker_server.log")
    log_max_bytes = int(os.getenv("MINT_LOG_MAX_BYTES", str(100 * 1024 * 1024)))  # 100MB
    log_backup_count = int(os.getenv("MINT_LOG_BACKUP_COUNT", "5"))

    root_logger = logging.getLogger()

    global _STRUCTLOG_WARNED
    if structlog is None:
        _configure_stdlib_logging(
            root_logger=root_logger,
            log_file=log_file,
            log_max_bytes=log_max_bytes,
            log_backup_count=log_backup_count,
        )
        if not _STRUCTLOG_WARNED:
            print(
                "Warning: structlog unavailable; using stdlib logging fallback",
                file=sys.stderr,
            )
            _STRUCTLOG_WARNED = True
        _configure_opentelemetry(root_logger)
        return

    # Configure structlog processors (shared for all outputs)
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        add_request_id,
        add_trace_id,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        add_component,
        add_operation,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Configure structlog
    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    root_logger.setLevel(logging.DEBUG)  # Capture all levels, handlers will filter
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(colors=True),
            ],
        )
    )
    root_logger.addHandler(console_handler)

    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=log_max_bytes,
            backupCount=log_backup_count,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=shared_processors,
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(),
                ],
            )
        )
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Failed to configure file logging to {log_file}: {e}", file=sys.stderr)
    _configure_opentelemetry(root_logger)


def init_actor_observability() -> None:
    """Initialize logging + OTEL inside Ray actor processes (best-effort, idempotent)."""
    global _ACTOR_OBS_INITIALIZED
    if _ACTOR_OBS_INITIALIZED:
        return
    with _ACTOR_OBS_LOCK:
        if _ACTOR_OBS_INITIALIZED:
            return
        init_state = "ok"
        try:
            configure_logging()
        except Exception as e:
            init_state = "degraded"
            try:
                logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
                )
            except Exception:
                pass
            print(
                f"Warning: actor observability init failed: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
        logger = logging.getLogger(__name__)
        logger.info(
            "[actor_observability] init=%s structlog_available=%s otel_enabled=%s tracer_set=%s endpoint_set=%s headers_set=%s app_key_set=%s",
            init_state,
            structlog is not None,
            is_otel_enabled(),
            get_otel_tracer() is not None,
            bool((os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip()),
            bool((os.getenv("OTEL_EXPORTER_OTLP_HEADERS") or "").strip()),
            bool((os.getenv("MINT_APMPLUS_APP_KEY") or "").strip()),
        )
        _ACTOR_OBS_INITIALIZED = True
