from __future__ import annotations

import importlib
import logging
import traceback
from typing import Any

from mint_server.backend.openpi.openpi_fast_runtime import (
    OpenPIFastRuntimeSpec,
    OpenPIFastWorkerError,
    OpenPIFastWorkerProtocolError,
    OpenPIFastWorkerRemoteError,
)


logger = logging.getLogger(__name__)


class OpenPIDirectWorkerClient:
    """In-process OpenPI worker-session adapter for Mint Ray actors.

    The old OpenPI path spawned a Python subprocess inside a Ray actor and used
    stdout JSON lines as a private RPC channel. This client keeps the same small
    `request(op, payload)` shape for callers, but executes worker session code
    directly in the Ray actor process so Ray owns exceptions and lifecycle.
    """

    def __init__(
        self,
        *,
        spec: OpenPIFastRuntimeSpec,
        module: Any,
    ) -> None:
        self._spec = spec
        self._module = module
        self._session: Any | None = None
        self._closed = False

    @classmethod
    async def start(cls, spec: OpenPIFastRuntimeSpec) -> "OpenPIDirectWorkerClient":
        module = importlib.import_module(spec.worker_module)
        return cls(spec=spec, module=module)

    def timeout_for(self, op: str) -> float:
        if op == "create_session":
            return self._spec.create_session_timeout_s
        if op == "save_weights":
            return self._spec.save_weights_timeout_s
        if op == "load_weights":
            return self._spec.load_weights_timeout_s
        return self._spec.request_timeout_s

    async def request(
        self,
        op: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        _ = timeout_s
        if self._closed:
            raise OpenPIFastWorkerProtocolError("direct OpenPI worker client is closed")

        try:
            result = self._request_sync(op, dict(payload or {}))
        except OpenPIFastWorkerError:
            raise
        except Exception as exc:
            raise OpenPIFastWorkerRemoteError(
                error_type=type(exc).__name__,
                message=str(exc),
                traceback_text=traceback.format_exc(),
            ) from exc

        if result is None:
            return {}
        if not isinstance(result, dict):
            raise OpenPIFastWorkerProtocolError(
                f"direct OpenPI worker payload must be a dict, got {type(result).__name__}"
            )
        return result

    def _request_sync(self, op: str, payload: dict[str, Any]) -> Any:
        if op == "create_session":
            return self._create_session(payload)

        dispatch = getattr(self._module, "_dispatch", None)
        if not callable(dispatch):
            raise OpenPIFastWorkerProtocolError(
                f"OpenPI worker module {self._spec.worker_module!r} does not expose _dispatch"
            )
        result = dispatch(self._session, op, payload)
        if not isinstance(result, tuple) or len(result) != 2:
            raise OpenPIFastWorkerProtocolError(
                f"OpenPI worker module {self._spec.worker_module!r} returned invalid dispatch result"
            )
        response, next_state = result
        if isinstance(next_state, bool):
            if next_state:
                self._session = None
                self._closed = True
        else:
            self._session = next_state
        return response

    def _create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._session is not None:
            dispatch = getattr(self._module, "_dispatch", None)
            if callable(dispatch):
                response, next_session = dispatch(self._session, "create_session", payload)
                self._session = next_session
                return response
            raise ValueError("OpenPI direct worker session is already initialized")

        session_cls = self._session_class()
        self._session = session_cls(payload)
        create = getattr(self._session, "create_session", None)
        if callable(create):
            response = create()
        else:
            response = {"ready": True}
        if not isinstance(response, dict):
            raise OpenPIFastWorkerProtocolError(
                f"OpenPI create_session returned non-dict payload: {type(response).__name__}"
            )
        return response

    def _session_class(self) -> type[Any]:
        class_names = (
            "OpenPIFastWorkerSession",
            "OpenPIPi05WorkerSession",
            "OpenPIFastActionSession",
            "OpenPIPi05ActionSession",
        )
        for name in class_names:
            value = getattr(self._module, name, None)
            if isinstance(value, type):
                return value
        raise OpenPIFastWorkerProtocolError(
            f"OpenPI worker module {self._spec.worker_module!r} does not expose a known session class"
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        session = self._session
        self._session = None
        if session is None:
            return
        shutdown = getattr(session, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception as exc:
                logger.warning(
                    "OpenPI direct worker shutdown failed for %s: %s: %s",
                    self._spec.worker_module,
                    type(exc).__name__,
                    exc,
                )
