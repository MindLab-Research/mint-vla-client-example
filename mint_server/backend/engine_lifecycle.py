from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Any, Awaitable, Callable

from .engine_adapter import EngineHealth, EngineHealthStatus, EngineObservability
from .execution_context import ExecutionContext

LifecycleFactory = Callable[[], Awaitable[ExecutionContext]]


class ExecutionContextEngineLifecycle:
    """Lifecycle view over the runtime-local execution bindings."""

    def __init__(
        self,
        context_factory: LifecycleFactory,
        *,
        refresh_context_factory: LifecycleFactory | None = None,
    ) -> None:
        self._context_factory = context_factory
        self._refresh_context_factory = refresh_context_factory or context_factory
        self._ready: bool | None = None
        self._unhealthy_reason: str | None = None
        self._restart_count = 0
        self._last_error: str | None = None

    async def is_ready(self) -> bool:
        try:
            context = await self._context_factory()
            ready = await self._context_ready(context)
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._ready = False
            return False
        self._ready = bool(ready)
        if self._ready:
            self._unhealthy_reason = None
        return bool(ready)

    async def health(self) -> EngineHealth:
        ready = await self.is_ready()
        if ready:
            return EngineHealth(
                status=EngineHealthStatus.READY,
                restart_count=self._restart_count,
                last_error=self._last_error,
            )
        return EngineHealth(
            status=EngineHealthStatus.UNHEALTHY,
            reason=self._unhealthy_reason,
            restart_count=self._restart_count,
            last_error=self._last_error,
        )

    async def get_observability_binding(self) -> EngineObservability:
        context = await self._context_factory()
        payload: dict[str, Any] = {}
        for source in self._candidate_sources(context):
            observed = await self._call_optional(source, "get_observability_binding")
            if isinstance(observed, EngineObservability):
                return observed
            if isinstance(observed, dict):
                payload.update(observed)
        return EngineObservability.from_wire(payload) if payload else EngineObservability()

    async def mark_unhealthy(self, reason: str) -> None:
        self._ready = False
        self._unhealthy_reason = str(reason)
        context = await self._context_factory()
        for source in self._candidate_sources(context):
            marker = getattr(source, "mark_unhealthy", None)
            if not callable(marker):
                continue
            out = marker(str(reason))
            if inspect.isawaitable(out):
                await out

    async def restart(self) -> None:
        self._restart_count += 1
        self._ready = False
        context = await self._context_factory()
        restarted = False
        for source in self._restart_sources(context):
            if await self._restart_source(source):
                restarted = True
        if not restarted:
            # Recreate the runtime-local bindings when no backend exposes a narrower restart hook.
            new_context = await self._refresh_context_factory()
            self._ready = await self._context_ready(new_context)
            return
        self._ready = await self._context_ready(context)

    async def _context_ready(self, context: ExecutionContext) -> bool:
        probes = []
        for source in self._candidate_sources(context):
            probes.extend((source, method) for method in ("is_ready", "is_engine_ready"))
        if not probes:
            return True
        saw_probe = False
        for source, method in probes:
            result = await self._call_optional(source, method)
            if result is None:
                continue
            saw_probe = True
            if not bool(result):
                return False
        return True if saw_probe else True

    def _candidate_sources(self, context: ExecutionContext) -> tuple[Any, ...]:
        sources: list[Any] = []
        for attr in ("inference_manager", "train_engine", "action_manager", "multi_model_manager"):
            source = getattr(context, attr, None)
            if source is not None and source is not False and source not in sources:
                sources.append(source)
        for manager in list(sources):
            for source in self._nested_sources(manager):
                if source is not None and source not in sources:
                    sources.append(source)
        return tuple(sources)

    def _nested_sources(self, source: Any) -> tuple[Any, ...]:
        nested: list[Any] = []
        for getter_name in ("get_multi_model_manager", "get_multi_lora_engine"):
            getter = getattr(source, getter_name, None)
            if not callable(getter):
                continue
            try:
                value = getter()
            except Exception:
                continue
            if value is not None and value not in nested:
                nested.append(value)
        shared = getattr(source, "_shared_engine", None)
        if shared is not None and shared not in nested:
            nested.append(shared)
        sessions = getattr(source, "_sessions", None)
        if isinstance(sessions, dict):
            for info in sessions.values():
                engine = getattr(info, "engine", None)
                if engine is not None and engine not in nested:
                    nested.append(engine)
        for attr in ("_text_engine", "_openpi_fast_engine", "_openpi_pi05_engine"):
            engine = getattr(source, attr, None)
            if engine is not None and engine not in nested:
                nested.append(engine)
        list_models = getattr(source, "list_models", None)
        get_engine_if_exists = getattr(source, "get_engine_if_exists", None)
        if callable(list_models) and callable(get_engine_if_exists):
            try:
                model_names = list_models()
            except Exception:
                model_names = ()
            if not isinstance(model_names, Iterable) or isinstance(model_names, (str, bytes)):
                model_names = ()
            for model_name in model_names or ():
                try:
                    engine = get_engine_if_exists(model_name)
                except Exception:
                    continue
                if engine is not None and engine not in nested:
                    nested.append(engine)
        return tuple(nested)

    def _restart_sources(self, context: ExecutionContext) -> tuple[Any, ...]:
        sources = list(self._candidate_sources(context))
        return tuple(
            sorted(
                sources,
                key=lambda source: 0
                if any(callable(getattr(source, method, None)) for method in ("restart", "restart_engine"))
                else 1,
            )
        )

    async def _restart_source(self, source: Any) -> bool:
        for method in ("restart", "restart_engine"):
            hook = getattr(source, method, None)
            if not callable(hook):
                continue
            out = hook()
            if inspect.isawaitable(out):
                await out
            return True
        shutdown_all = getattr(source, "shutdown_all", None)
        if callable(shutdown_all):
            out = shutdown_all()
            if inspect.isawaitable(out):
                await out
            return True
        shutdown = getattr(source, "shutdown", None)
        if callable(shutdown):
            out = shutdown()
            if inspect.isawaitable(out):
                await out
            return True
        return False

    async def _call_optional(self, source: Any, method: str) -> Any:
        hook = getattr(source, method, None)
        if not callable(hook):
            return None
        out = hook()
        if inspect.isawaitable(out):
            out = await out
        return out


__all__ = ["ExecutionContextEngineLifecycle"]
