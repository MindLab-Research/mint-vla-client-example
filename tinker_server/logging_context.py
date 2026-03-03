"""Request context propagation for distributed tracing."""

import contextvars
import logging
from typing import Any

# Context variable to store current request_id
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)


def set_request_id(request_id: str | None) -> None:
    """Set the request_id for the current context."""
    request_id_var.set(request_id)


def get_request_id() -> str | None:
    """Get the request_id from the current context."""
    return request_id_var.get()


class RequestIDFilter(logging.Filter):
    """Logging filter that injects request_id into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        request_id = get_request_id()
        record.request_id = request_id if request_id else "-"
        return True


class RequestIDFormatter(logging.Formatter):
    """Logging formatter that includes request_id in output."""

    def __init__(self, fmt: str | None = None, datefmt: str | None = None):
        if fmt is None:
            fmt = "[%(asctime)s] [%(levelname)s] [request_id=%(request_id)s] %(name)s: %(message)s"
        super().__init__(fmt=fmt, datefmt=datefmt)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger with request_id support."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add new handler with request_id support
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.addFilter(RequestIDFilter())
    handler.setFormatter(RequestIDFormatter())
    root_logger.addHandler(handler)
