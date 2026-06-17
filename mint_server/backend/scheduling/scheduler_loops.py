from __future__ import annotations

import asyncio
import contextlib
import structlog
from collections.abc import Awaitable, Callable
from typing import Any

logger = structlog.get_logger(__name__)


class BackgroundLoopSupervisor:
    def __init__(
        self,
        *,
        assignment_loop: Callable[[], Awaitable[None]],
        owner_heartbeat_loop: Callable[[], Awaitable[None]],
        reaper_loop: Callable[[], Awaitable[None]],
        use_task_state_store: bool,
        assignment_interval_s: float,
        owner_heartbeat_interval_s: float,
        reaper_interval_s: float,
    ) -> None:
        self._assignment_loop = assignment_loop
        self._owner_heartbeat_loop = owner_heartbeat_loop
        self._reaper_loop = reaper_loop
        self._use_task_state_store = bool(use_task_state_store)
        self.assignment_interval_s = float(assignment_interval_s)
        self.owner_heartbeat_interval_s = float(owner_heartbeat_interval_s)
        self.reaper_interval_s = float(reaper_interval_s)
        self._manager_task: asyncio.Task | None = None
        self._tasks: dict[str, asyncio.Task] = {}
        self._start_deferred: set[str] = set()
        self._shutdown = False

    @property
    def manager_task(self) -> asyncio.Task | None:
        return self._manager_task

    @manager_task.setter
    def manager_task(self, value: asyncio.Task | None) -> None:
        self._manager_task = value

    @property
    def tasks(self) -> dict[str, asyncio.Task]:
        return self._tasks

    @property
    def start_deferred(self) -> set[str]:
        return self._start_deferred

    @start_deferred.setter
    def start_deferred(self, value: set[str]) -> None:
        self._start_deferred = value

    @property
    def shutdown(self) -> bool:
        return self._shutdown

    @shutdown.setter
    def shutdown(self, value: bool) -> None:
        self._shutdown = bool(value)

    @property
    def assignment_task(self) -> asyncio.Task | None:
        return self._tasks.get("assignment")

    @assignment_task.setter
    def assignment_task(self, value: asyncio.Task | None) -> None:
        if value is None:
            self._tasks.pop("assignment", None)
        else:
            self._tasks["assignment"] = value

    @property
    def owner_heartbeat_task(self) -> asyncio.Task | None:
        return self._tasks.get("owner_heartbeat")

    @owner_heartbeat_task.setter
    def owner_heartbeat_task(self, value: asyncio.Task | None) -> None:
        if value is None:
            self._tasks.pop("owner_heartbeat", None)
        else:
            self._tasks["owner_heartbeat"] = value

    @property
    def reaper_task(self) -> asyncio.Task | None:
        return self._tasks.get("reaper")

    @reaper_task.setter
    def reaper_task(self, value: asyncio.Task | None) -> None:
        if value is None:
            self._tasks.pop("reaper", None)
        else:
            self._tasks["reaper"] = value

    def desired_names(self) -> list[str]:
        names: list[str] = []
        if self.assignment_interval_s > 0:
            names.append("assignment")
        if self._use_task_state_store and self.owner_heartbeat_interval_s > 0:
            names.append("owner_heartbeat")
        if self._use_task_state_store and self.reaper_interval_s > 0:
            names.append("reaper")
        return names

    def running(self, name: str) -> bool:
        task = self._tasks.get(name)
        if task is not None:
            return not task.done()
        manager = self._manager_task
        return (
            manager is not None
            and not manager.done()
            and name in self.desired_names()
            and name not in self._start_deferred
        )

    async def _manager(self) -> None:
        try:
            async with asyncio.TaskGroup() as task_group:
                if "assignment" in self.desired_names():
                    self._tasks["assignment"] = task_group.create_task(
                        self._assignment_loop(),
                        name="model-work-scheduler-assignment-loop",
                    )
                if "owner_heartbeat" in self.desired_names():
                    self._tasks["owner_heartbeat"] = task_group.create_task(
                        self._owner_heartbeat_loop(),
                        name="model-work-scheduler-owner-heartbeat-loop",
                    )
                if "reaper" in self.desired_names():
                    self._tasks["reaper"] = task_group.create_task(
                        self._reaper_loop(),
                        name="model-work-scheduler-reaper-loop",
                    )
        finally:
            self._tasks.clear()

    def ensure_started(self) -> None:
        if self._shutdown:
            return
        desired = self.desired_names()
        if not desired:
            self._start_deferred.clear()
            return
        if self._manager_task is not None and not self._manager_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._start_deferred = set(desired)
            logger.debug(
                "[model_work_scheduler] background loop start deferred; no running event loop names=%s",
                sorted(desired),
            )
            return
        self._start_deferred.clear()
        self._manager_task = loop.create_task(
            self._manager(),
            name="model-work-scheduler-background-loop-manager",
        )

    async def shutdown_loops(self) -> dict[str, Any]:
        self._shutdown = True
        task = self._manager_task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._manager_task = None
        self._tasks.clear()
        return {"ok": True}

    def stats_snapshot(self) -> dict[str, Any]:
        return {
            "background_loop_manager_running": self._manager_task is not None and not self._manager_task.done(),
            "background_loop_names": [
                name
                for name in ("assignment", "owner_heartbeat", "reaper")
                if self.running(name)
            ],
            "background_loop_start_deferred": sorted(self._start_deferred),
            "assignment_loop_interval_s": self.assignment_interval_s,
            "assignment_loop_running": self.running("assignment"),
            "owner_heartbeat_interval_s": self.owner_heartbeat_interval_s,
            "owner_heartbeat_running": self.running("owner_heartbeat"),
            "reaper_loop_interval_s": self.reaper_interval_s,
            "reaper_loop_running": self.running("reaper"),
        }
