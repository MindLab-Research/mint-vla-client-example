from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class BlockPoint:
    entered: asyncio.Event
    release: asyncio.Event


class FaultController:
    def __init__(self) -> None:
        self._blocks: dict[str, BlockPoint] = {}
        self._errors: dict[str, BaseException | Callable[..., BaseException]] = {}

    def block(self, name: str) -> BlockPoint:
        point = BlockPoint(entered=asyncio.Event(), release=asyncio.Event())
        self._blocks[str(name)] = point
        return point

    def fail_next(self, name: str, error: BaseException | Callable[..., BaseException]) -> None:
        self._errors[str(name)] = error

    async def before_call(self, name: str, **kwargs: Any) -> None:
        name = str(name)
        point = self._blocks.get(name)
        if point is not None:
            point.entered.set()
            await point.release.wait()
            self._blocks.pop(name, None)
        error = self._errors.pop(name, None)
        if error is None:
            return
        if callable(error):
            raise error(**kwargs)
        raise error
