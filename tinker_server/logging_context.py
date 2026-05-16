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
    OTEL_APMPLUS_APP_KEY: legacy alias for MINT_APMPLUS_APP_KEY
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import logging
import logging.handlers
import os
import re
import socket
import sys
import threading
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

try:
    import structlog
except Exception:
    structlog = None

# Context variable to store current request_id
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
# Context variable to store current trace_id
trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)
request_identity_var: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "request_identity",
    default=None,
)

_UNSET = object()
_REQUEST_IDENTITY_KEYS = (
    "user_id",
    "user_role",
    "account_id",
    "apikey_id",
    "gateway_request_id",
    "gateway_session_id",
)

_HEX_CHARS = frozenset("0123456789abcdef")
_OTEL_ENABLED = False
_OTEL_INITIALIZED = False
_OTEL_LOG_HANDLER_ATTACHED = False
_STRUCTLOG_WARNED = False
_HTTP_REQUEST_COUNTER: Any | None = None
_HTTP_DURATION_HISTOGRAM: Any | None = None
_HTTP_ERROR_COUNTER: Any | None = None
_SAMPLING_ADMISSION_COUNTER: Any | None = None
_FUTURE_STORE_TIMEOUT_COUNTER: Any | None = None
_VLLM_ACTOR_REQUEST_COUNTER: Any | None = None
_VLLM_ACTOR_REQUEST_DURATION_HISTOGRAM: Any | None = None
_TRAINING_OPERATION_COUNTER: Any | None = None
_TRAINING_OPERATION_DURATION_HISTOGRAM: Any | None = None
_MEGATRON_SESSION_SWITCH_COUNTER: Any | None = None
_MEGATRON_SESSION_SWITCH_DURATION_COUNTER: Any | None = None
_SCHEDULER_DECISION_COUNTER: Any | None = None
_SCHEDULER_SWITCH_COUNTER: Any | None = None
_SCHEDULER_QUEUE_WAIT_HISTOGRAM: Any | None = None
_SCHEDULER_READY_SESSIONS_HISTOGRAM: Any | None = None
_SCHEDULER_CHOSEN_QUEUE_DEPTH_HISTOGRAM: Any | None = None
_TRACER: Any | None = None
_OP_PREFIX_RE = re.compile(r"^\[([A-Za-z0-9_.:-]+)\]")
_ACTOR_OBS_INITIALIZED = False
_ACTOR_OBS_LOCK = threading.Lock()
_T = TypeVar("_T")


def _detect_hostname() -> str:
    try:
        name = socket.gethostname()
        if isinstance(name, str) and name.strip():
            return name.strip()
    except Exception:
        pass
    return "unknown-host"


_HOSTNAME = _detect_hostname()


def _coerce_file_line(pathname: object, lineno: object) -> tuple[str, int, str]:
    path = str(pathname).strip() if isinstance(pathname, str) and pathname.strip() else "<unknown>"
    try:
        line = int(lineno) if lineno is not None else 0
    except Exception:
        line = 0
    if line <= 0:
        line = 1
    return path, line, f"{path}:{line}"


class _ContextEnrichmentFilter(logging.Filter):
    """Inject request/trace/host/callsite fields for all handlers (including OTLP)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = get_request_id() or "-"
        if not hasattr(record, "trace_id"):
            record.trace_id = get_trace_id() or _get_current_otel_trace_id() or "-"
        identity = get_request_identity_context()
        for key in _REQUEST_IDENTITY_KEYS:
            if not hasattr(record, key):
                record.__dict__[key] = identity.get(key) or "-"

        current_hostname = getattr(record, "hostname", None)
        if not isinstance(current_hostname, str) or not current_hostname.strip():
            record.hostname = _HOSTNAME

        current_file_line = getattr(record, "file_line", None)
        if not isinstance(current_file_line, str) or not current_file_line.strip() or current_file_line.endswith(":0"):
            path, line, file_line = _coerce_file_line(getattr(record, "pathname", None), getattr(record, "lineno", None))
            record.pathname = path
            record.lineno = line
            record.file_line = file_line
            if path != "<unknown>":
                record.filename = os.path.basename(path)
        return True


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


def get_request_identity_context() -> dict[str, str]:
    raw = request_identity_var.get()
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if isinstance(k, str) and isinstance(v, str) and v}


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


def get_current_traceparent() -> str | None:
    """Return current W3C traceparent from active OTel span context."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span is None:
            return None
        span_context = span.get_span_context()
        if not bool(getattr(span_context, "is_valid", False)):
            return None
        trace_id = int(getattr(span_context, "trace_id", 0))
        span_id = int(getattr(span_context, "span_id", 0))
        if trace_id == 0 or span_id == 0:
            return None
        flags = int(getattr(span_context, "trace_flags", 1))
        return f"00-{trace_id:032x}-{span_id:016x}-{flags:02x}"
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


def restore_trace_id_from_traceparent(traceparent: str | None) -> str | None:
    """Restore trace_id context from W3C traceparent header."""
    trace_id = extract_trace_id_from_traceparent(traceparent)
    set_trace_id(trace_id)
    return trace_id


def extract_otel_context_from_traceparent(traceparent: str | None) -> Any | None:
    """Best-effort extract of an OTel parent context from W3C traceparent."""
    if not isinstance(traceparent, str) or not traceparent.strip():
        return None
    try:
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

        return TraceContextTextMapPropagator().extract({"traceparent": traceparent.strip()})
    except Exception:
        return None


@contextmanager
def start_as_current_span(
    span_name: str,
    *,
    component: str | None = None,
    op: str | None = None,
    request_id: str | None = None,
    attributes: dict[str, Any] | None = None,
    kind: Any | None = None,
    context: Any | None = None,
) -> Iterator[Any | None]:
    """Start an OTel span in the current process context."""
    tracer = get_otel_tracer()
    if tracer is None:
        yield None
        return

    try:
        from opentelemetry.trace import SpanKind, Status, StatusCode
    except Exception:
        yield None
        return

    span_kind = SpanKind.INTERNAL if kind is None else kind
    with tracer.start_as_current_span(str(span_name), kind=span_kind, context=context) as span:
        if component:
            span.set_attribute("component", str(component))
        if op:
            span.set_attribute("op", str(op))
        if request_id:
            span.set_attribute("request_id", str(request_id))
        if attributes:
            for key, value in attributes.items():
                if value is None:
                    continue
                try:
                    span.set_attribute(str(key), value)
                except Exception:
                    span.set_attribute(str(key), str(value))
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


@contextmanager
def start_as_current_span_from_traceparent(
    span_name: str,
    *,
    traceparent: str | None = None,
    component: str | None = None,
    op: str | None = None,
    request_id: str | None = None,
    attributes: dict[str, Any] | None = None,
    kind: Any | None = None,
) -> Iterator[Any | None]:
    """Start an OTel span from a propagated traceparent and bind request/trace IDs for logs."""
    trace_id = extract_trace_id_from_traceparent(traceparent)
    context = extract_otel_context_from_traceparent(traceparent)
    with bind_request_trace_context(request_id=request_id, trace_id=trace_id):
        with start_as_current_span(
            span_name,
            component=component,
            op=op,
            request_id=request_id,
            attributes=attributes,
            kind=kind,
            context=context,
        ) as span:
            yield span


def traced_async_from_traceparent(
    span_name: str,
    *,
    component: str | None = None,
    op: str | None = None,
    request_id_arg: str | None = None,
    traceparent_arg: str = "traceparent",
    attributes_builder: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
) -> Callable[[Callable[..., Awaitable[_T]]], Callable[..., Awaitable[_T]]]:
    """Decorator for async actor methods that receive a propagated traceparent."""

    def _decorator(fn: Callable[..., Awaitable[_T]]) -> Callable[..., Awaitable[_T]]:
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        async def _wrapped(*args: Any, **kwargs: Any) -> _T:
            bound = signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            traceparent = arguments.get(traceparent_arg)
            request_id = arguments.get(request_id_arg) if request_id_arg else None
            attributes: dict[str, Any] | None = None
            if attributes_builder is not None:
                try:
                    attributes = attributes_builder(arguments)
                except Exception:
                    attributes = None
            with start_as_current_span_from_traceparent(
                span_name,
                traceparent=traceparent if isinstance(traceparent, str) else None,
                component=component,
                op=op,
                request_id=str(request_id) if request_id is not None else None,
                attributes=attributes,
            ):
                return await fn(*args, **kwargs)

        return _wrapped

    return _decorator


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


def _normalize_context_id(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized == "-":
        return None
    return normalized


@contextmanager
def bind_request_trace_context(
    *,
    request_id: str | None = None,
    trace_id: str | None = None,
    user_id: str | None | object = _UNSET,
    user_role: str | None | object = _UNSET,
    account_id: str | None | object = _UNSET,
    apikey_id: str | None | object = _UNSET,
    gateway_request_id: str | None | object = _UNSET,
    gateway_session_id: str | None | object = _UNSET,
) -> Iterator[None]:
    """Temporarily bind request/trace IDs and restore previous context on exit."""
    prev_request_id = get_request_id()
    prev_trace_id = get_trace_id()
    prev_identity = get_request_identity_context()
    set_request_id(_normalize_context_id(request_id))
    set_trace_id(_normalize_context_id(trace_id))
    next_identity = dict(prev_identity)
    for key, value in (
        ("user_id", user_id),
        ("user_role", user_role),
        ("account_id", account_id),
        ("apikey_id", apikey_id),
        ("gateway_request_id", gateway_request_id),
        ("gateway_session_id", gateway_session_id),
    ):
        if value is _UNSET:
            continue
        normalized = _normalize_context_id(None if value is None else str(value))
        if normalized is None:
            next_identity.pop(key, None)
        else:
            next_identity[key] = normalized
    request_identity_var.set(next_identity)
    try:
        yield
    finally:
        set_request_id(prev_request_id)
        set_trace_id(prev_trace_id)
        request_identity_var.set(prev_identity or None)


def log_with_bound_context(
    logger: logging.Logger,
    message: str,
    *,
    request_id: str | None = None,
    trace_id: str | None = None,
    level: int = logging.INFO,
) -> None:
    """Emit one log line with explicit request/trace context."""
    with bind_request_trace_context(request_id=request_id, trace_id=trace_id):
        logger.log(int(level), message)


async def run_async_with_otel_span(
    span_name: str,
    action: Callable[[], Awaitable[_T]],
    *,
    component: str | None = None,
    op: str | None = None,
    request_id: str | None = None,
    attributes: dict[str, Any] | None = None,
    context: Any | None = None,
) -> _T:
    """Run async action inside an OTEL span when tracer is available."""
    tracer = get_otel_tracer()
    if tracer is None:
        return await action()

    try:
        from opentelemetry.trace import SpanKind, Status, StatusCode
    except Exception:
        return await action()

    with tracer.start_as_current_span(str(span_name), kind=SpanKind.INTERNAL, context=context) as span:
        if component:
            span.set_attribute("component", str(component))
        if op:
            span.set_attribute("op", str(op))
        if request_id:
            span.set_attribute("request_id", str(request_id))
        if attributes:
            for key, value in attributes.items():
                if value is None:
                    continue
                try:
                    span.set_attribute(str(key), value)
                except Exception:
                    span.set_attribute(str(key), str(value))
        try:
            return await action()
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


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


def add_request_identity_fields(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Structlog processor to add request identity fields to all log events."""
    identity = get_request_identity_context()
    for key in _REQUEST_IDENTITY_KEYS:
        event_dict[key] = identity.get(key) or "-"
    return event_dict


def add_hostname(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Structlog processor to add hostname to all log events."""
    hostname = event_dict.get("hostname")
    if not isinstance(hostname, str) or not hostname.strip():
        event_dict["hostname"] = _HOSTNAME
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
    if (
        "dense_input_contract_violation" in text
        or "contract_violation" in text
        or "out_of_range" in text
        or "len_mismatch" in text
        or "non_finite" in text
    ):
        return "input_contract"
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
    global _HTTP_REQUEST_COUNTER, _HTTP_DURATION_HISTOGRAM, _HTTP_ERROR_COUNTER
    global _SAMPLING_ADMISSION_COUNTER, _FUTURE_STORE_TIMEOUT_COUNTER
    global _VLLM_ACTOR_REQUEST_COUNTER, _VLLM_ACTOR_REQUEST_DURATION_HISTOGRAM
    global _TRAINING_OPERATION_COUNTER, _TRAINING_OPERATION_DURATION_HISTOGRAM
    global _MEGATRON_SESSION_SWITCH_COUNTER, _MEGATRON_SESSION_SWITCH_DURATION_COUNTER
    global _SCHEDULER_DECISION_COUNTER, _SCHEDULER_SWITCH_COUNTER, _SCHEDULER_QUEUE_WAIT_HISTOGRAM
    global _SCHEDULER_READY_SESSIONS_HISTOGRAM, _SCHEDULER_CHOSEN_QUEUE_DEPTH_HISTOGRAM, _TRACER

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
    # Use explicit resource attributes only; avoid default detector payloads.
    resource = Resource(attributes={"service.name": service_name})
    headers = _parse_headers(os.getenv("OTEL_EXPORTER_OTLP_HEADERS"))
    app_key = (
        os.getenv("MINT_APMPLUS_APP_KEY")
        or os.getenv("OTEL_APMPLUS_APP_KEY")
        or ""
    ).strip()
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
        _SAMPLING_ADMISSION_COUNTER = meter.create_counter(
            "mint_sampling_admission_total",
            unit="{decision}",
            description="Sampling admission decisions observed by mint",
        )
        _FUTURE_STORE_TIMEOUT_COUNTER = meter.create_counter(
            "mint_task_state_futures_timeout_events_total",
            unit="{timeout}",
            description="task state future queue and execution timeout events observed by mint",
        )
        _VLLM_ACTOR_REQUEST_COUNTER = meter.create_counter(
            "mint_vllm_actor_requests_total",
            unit="{request}",
            description="Successful vLLM actor requests observed by mint",
        )
        _VLLM_ACTOR_REQUEST_DURATION_HISTOGRAM = meter.create_histogram(
            "mint_vllm_actor_request_duration_s",
            unit="s",
            description="Successful vLLM actor request duration in seconds",
        )
        _TRAINING_OPERATION_COUNTER = meter.create_counter(
            "mint_training_operations_total",
            unit="{operation}",
            description="Training operations observed by mint",
        )
        _TRAINING_OPERATION_DURATION_HISTOGRAM = meter.create_histogram(
            "mint_training_operation_duration_s",
            unit="s",
            description="Training operation duration in seconds",
        )
        _MEGATRON_SESSION_SWITCH_COUNTER = meter.create_counter(
            "mint_megatron_session_switch_total",
            unit="{switch}",
            description="Megatron session-switch events observed by mint",
        )
        _MEGATRON_SESSION_SWITCH_DURATION_COUNTER = meter.create_counter(
            "mint_megatron_session_switch_duration_s_total",
            unit="s",
            description="Megatron session-switch duration totals observed by mint",
        )
        _SCHEDULER_DECISION_COUNTER = meter.create_counter(
            "mint_scheduler_decision_total",
            unit="{decision}",
            description="Queue scheduling decisions observed by mint",
        )
        _SCHEDULER_SWITCH_COUNTER = meter.create_counter(
            "mint_scheduler_switch_total",
            unit="{switch}",
            description="Queue scheduling session switches observed by mint",
        )
        _SCHEDULER_QUEUE_WAIT_HISTOGRAM = meter.create_histogram(
            "mint_scheduler_queue_wait_s",
            unit="s",
            description="Queue wait time from enqueue to dequeue in seconds",
        )
        _SCHEDULER_READY_SESSIONS_HISTOGRAM = meter.create_histogram(
            "mint_scheduler_ready_sessions",
            unit="{session}",
            description="Ready session count seen at scheduler decision time",
        )
        _SCHEDULER_CHOSEN_QUEUE_DEPTH_HISTOGRAM = meter.create_histogram(
            "mint_scheduler_chosen_queue_depth",
            unit="{request}",
            description="Chosen session queue depth seen at scheduler decision time",
        )

        # 3) Logs
        log_exporter = OTLPLogExporter(endpoint=endpoint, headers=headers or None, insecure=insecure)
        logger_provider = LoggerProvider(resource=resource)
        logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
        set_logger_provider(logger_provider)
        if not _OTEL_LOG_HANDLER_ATTACHED:
            level_name = (os.getenv("OTEL_LOG_LEVEL") or "INFO").upper()
            level = getattr(logging, level_name, logging.INFO)
            otel_handler = LoggingHandler(level=level, logger_provider=logger_provider)
            otel_handler.addFilter(_ContextEnrichmentFilter())
            root_logger.addHandler(otel_handler)
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


def record_sampling_admission_metric(
    *,
    route: str,
    decision: str,
    reason: str,
    scope: str | None = None,
) -> None:
    if not _OTEL_ENABLED:
        return
    attrs: dict[str, str] = {
        "route": str(route),
        "decision": str(decision),
        "reason": str(reason),
    }
    if isinstance(scope, str) and scope.strip():
        attrs["scope"] = scope.strip()
    try:
        if _SAMPLING_ADMISSION_COUNTER is not None:
            _SAMPLING_ADMISSION_COUNTER.add(1, attributes=attrs)
    except Exception:
        pass


def record_task_state_futures_timeout_metric(*, kind: str, op: str | None = None) -> None:
    if not _OTEL_ENABLED:
        return
    attrs: dict[str, str] = {"kind": str(kind)}
    if isinstance(op, str) and op.strip():
        attrs["op"] = op.strip()
    try:
        if _FUTURE_STORE_TIMEOUT_COUNTER is not None:
            _FUTURE_STORE_TIMEOUT_COUNTER.add(1, attributes=attrs)
    except Exception:
        pass


def _record_current_span_event(event_name: str, attrs: dict[str, object]) -> None:
    if not _OTEL_ENABLED:
        return
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        span_ctx = span.get_span_context() if span is not None else None
        if span_ctx is None or not bool(getattr(span_ctx, "is_valid", False)):
            return
        span.add_event(str(event_name), attributes=attrs)
    except Exception:
        pass


def record_span_event_otel(event_name: str, *, attributes: dict[str, object] | None = None) -> None:
    _record_current_span_event(str(event_name), dict(attributes or {}))


def record_vllm_actor_latency_otel(
    *,
    actor_name: str | None,
    base_model: str,
    op: str,
    status: str,
    duration_s: float,
) -> None:
    if not _OTEL_ENABLED:
        return
    attrs: dict[str, str] = {
        "actor_name": str(actor_name or "unknown"),
        "base_model": str(base_model),
        "op": str(op),
    }
    try:
        if str(status) != "ok":
            _record_current_span_event(
                "mint.vllm_actor_request.failure",
                {**attrs, "status": str(status), "duration_s": float(duration_s)},
            )
            return
        if _VLLM_ACTOR_REQUEST_COUNTER is not None:
            _VLLM_ACTOR_REQUEST_COUNTER.add(1, attributes=attrs)
        if _VLLM_ACTOR_REQUEST_DURATION_HISTOGRAM is not None:
            _VLLM_ACTOR_REQUEST_DURATION_HISTOGRAM.record(float(duration_s), attributes=attrs)
    except Exception:
        pass


def record_megatron_session_switch_otel(
    *,
    base_model: str,
    session_state: str,
    count: int,
    durations_s: dict[str, float],
) -> None:
    if not _OTEL_ENABLED:
        return
    attrs = {
        "base_model": str(base_model),
        "session_state": str(session_state),
    }
    try:
        if _MEGATRON_SESSION_SWITCH_COUNTER is not None and int(count) > 0:
            _MEGATRON_SESSION_SWITCH_COUNTER.add(int(count), attributes=attrs)
        if _MEGATRON_SESSION_SWITCH_DURATION_COUNTER is not None:
            for phase, total_s in durations_s.items():
                total = max(0.0, float(total_s))
                if total <= 0.0:
                    continue
                _MEGATRON_SESSION_SWITCH_DURATION_COUNTER.add(total, attributes={**attrs, "phase": str(phase)})
    except Exception:
        pass


def record_scheduler_decision_otel(
    *,
    op: str,
    backend: str,
    queue_kind: str,
    reason: str,
    queue_wait_s: float,
    switched: bool,
    ready_sessions: int | None = None,
    chosen_queue_depth: int | None = None,
) -> None:
    if not _OTEL_ENABLED:
        return
    attrs = {
        "op": str(op),
        "backend": str(backend),
        "queue_kind": str(queue_kind),
        "reason": str(reason),
    }
    try:
        if _SCHEDULER_DECISION_COUNTER is not None:
            _SCHEDULER_DECISION_COUNTER.add(1, attributes=attrs)
        if bool(switched) and _SCHEDULER_SWITCH_COUNTER is not None:
            _SCHEDULER_SWITCH_COUNTER.add(1, attributes=attrs)
        if _SCHEDULER_QUEUE_WAIT_HISTOGRAM is not None:
            _SCHEDULER_QUEUE_WAIT_HISTOGRAM.record(max(0.0, float(queue_wait_s)), attributes=attrs)
        if ready_sessions is not None and _SCHEDULER_READY_SESSIONS_HISTOGRAM is not None:
            _SCHEDULER_READY_SESSIONS_HISTOGRAM.record(max(0, int(ready_sessions)), attributes=attrs)
        if chosen_queue_depth is not None and _SCHEDULER_CHOSEN_QUEUE_DEPTH_HISTOGRAM is not None:
            _SCHEDULER_CHOSEN_QUEUE_DEPTH_HISTOGRAM.record(max(0, int(chosen_queue_depth)), attributes=attrs)
    except Exception:
        pass


def record_training_operation_latency_otel(
    *,
    base_model: str,
    backend: str,
    op: str,
    status: str,
    failure_class: str | None = None,
    duration_s: float,
) -> None:
    if not _OTEL_ENABLED:
        return
    attrs: dict[str, str] = {
        "base_model": str(base_model),
        "backend": str(backend),
        "op": str(op),
        "status": str(status),
        "failure_class": str(failure_class or "none"),
    }
    try:
        if _TRAINING_OPERATION_COUNTER is not None:
            _TRAINING_OPERATION_COUNTER.add(1, attributes=attrs)
        if _TRAINING_OPERATION_DURATION_HISTOGRAM is not None:
            _TRAINING_OPERATION_DURATION_HISTOGRAM.record(float(duration_s), attributes=attrs)
        if str(status) != "ok":
            _record_current_span_event(
                "mint.training_operation.failure",
                {**attrs, "status": str(status), "duration_s": float(duration_s)},
            )
    except Exception:
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

    fmt = (
        "%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s trace_id=%(trace_id)s "
        "user_id=%(user_id)s user_role=%(user_role)s account_id=%(account_id)s apikey_id=%(apikey_id)s "
        "gateway_request_id=%(gateway_request_id)s gateway_session_id=%(gateway_session_id)s %(message)s"
    )
    datefmt = "%Y-%m-%dT%H:%M:%S%z"

    context_filter = _ContextEnrichmentFilter()

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
        add_request_identity_fields,
        add_hostname,
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
    context_filter = _ContextEnrichmentFilter()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(context_filter)
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
        file_handler.addFilter(context_filter)
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
            bool(
                (os.getenv("MINT_APMPLUS_APP_KEY") or os.getenv("OTEL_APMPLUS_APP_KEY") or "").strip()
            ),
        )
        _ACTOR_OBS_INITIALIZED = True
