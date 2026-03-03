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
from typing import Any

import structlog

# Context variable to store current request_id
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)


def set_request_id(request_id: str | None) -> None:
    """Set the request_id for the current context."""
    request_id_var.set(request_id)


def get_request_id() -> str | None:
    """Get the request_id from the current context."""
    return request_id_var.get()


def add_request_id(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Structlog processor to add request_id to all log events."""
    request_id = get_request_id()
    event_dict["request_id"] = request_id if request_id else "-"
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
