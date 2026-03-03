"""Request context propagation and structured logging configuration.

Environment Variables:
    MINT_LOG_FILE: Log file path (default: /tmp/tinker_server.log)
    MINT_LOG_MAX_BYTES: Max log file size before rotation (default: 10MB)
    MINT_LOG_BACKUP_COUNT: Number of backup files to keep (default: 5)
    OTEL_EXPORTER_OTLP_ENDPOINT: OpenTelemetry collector endpoint (e.g., http://localhost:4318)
    OTEL_EXPORTER_OTLP_HEADERS: OpenTelemetry headers for auth (e.g., "api-key=xxx")
    OTEL_SERVICE_NAME: Service name for traces (default: mint-core)
"""

from __future__ import annotations

import contextvars
import logging
import logging.handlers
import os
import sys
import uuid
from typing import Any

import structlog

# Context variable to store current request_id
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
# Context variable to store current trace_id
trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)

_HEX_CHARS = frozenset("0123456789abcdef")


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


def configure_logging() -> None:
    """Configure structlog with multiple outputs: console (INFO), file (DEBUG), OTel (DEBUG, optional).

    Output targets:
    1. Console: Human-readable text format, INFO level
    2. File: JSON format, DEBUG level, with rotation
    3. OTel: JSON format, DEBUG level (only if OTEL_EXPORTER_OTLP_ENDPOINT is set)
    """
    # Environment variables
    log_file = os.getenv("MINT_LOG_FILE", "/tmp/tinker_server.log")
    log_max_bytes = int(os.getenv("MINT_LOG_MAX_BYTES", str(10 * 1024 * 1024)))  # 10MB
    log_backup_count = int(os.getenv("MINT_LOG_BACKUP_COUNT", "5"))
    otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
    otel_service_name = os.getenv("OTEL_SERVICE_NAME", "mint-core")

    # Configure structlog processors (shared for all outputs)
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        add_request_id,
        add_trace_id,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
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

    # Configure stdlib logging
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels, handlers will filter

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 1. Console handler (INFO level, human-readable)
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

    # 2. File handler (DEBUG level, JSON)
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
        # If file logging fails, log to console but don't crash
        print(f"Warning: Failed to configure file logging to {log_file}: {e}", file=sys.stderr)

    # 3. OTel handler (DEBUG level, JSON, optional)
    if otel_endpoint:
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            # Parse headers if provided
            headers_dict = None
            if otel_headers:
                headers_dict = {}
                for pair in otel_headers.split(","):
                    if "=" in pair:
                        key, value = pair.split("=", 1)
                        headers_dict[key.strip()] = value.strip()

            # Configure OTel tracer
            resource = Resource.create({"service.name": otel_service_name})
            tracer_provider = TracerProvider(resource=resource)

            otlp_exporter = OTLPSpanExporter(
                endpoint=otel_endpoint,
                headers=headers_dict,
            )
            tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            trace.set_tracer_provider(tracer_provider)

            # Create OTel log handler (logs as JSON to OTel)
            # Note: OpenTelemetry Python SDK doesn't have native log export yet,
            # so we'll create a custom handler that sends logs as span events
            class OTelLogHandler(logging.Handler):
                def __init__(self):
                    super().__init__()
                    self.tracer = trace.get_tracer(__name__)

                def emit(self, record: logging.LogRecord) -> None:
                    try:
                        # Get current span or create a new one
                        current_span = trace.get_current_span()
                        if current_span and current_span.is_recording():
                            # Add log as span event
                            attributes = {
                                "log.level": record.levelname,
                                "log.logger": record.name,
                                "log.message": self.format(record),
                            }
                            if hasattr(record, "request_id"):
                                attributes["request_id"] = record.request_id
                            if hasattr(record, "trace_id"):
                                attributes["trace_id"] = record.trace_id
                            else:
                                trace_id = get_trace_id() or _get_current_otel_trace_id()
                                if trace_id:
                                    attributes["trace_id"] = trace_id
                            current_span.add_event(record.getMessage(), attributes=attributes)
                    except Exception:
                        self.handleError(record)

            otel_handler = OTelLogHandler()
            otel_handler.setLevel(logging.DEBUG)
            otel_handler.setFormatter(
                structlog.stdlib.ProcessorFormatter(
                    foreign_pre_chain=shared_processors,
                    processors=[
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        structlog.processors.JSONRenderer(),
                    ],
                )
            )
            root_logger.addHandler(otel_handler)

        except Exception as e:
            # If OTel setup fails, log warning but continue
            print(f"Warning: Failed to configure OpenTelemetry logging: {e}", file=sys.stderr)
