from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

TokenBudgetProvider = Callable[[], Awaitable[int | None]]


class TokenBudgetController:
    def __init__(self, *, refresh_interval_s: float) -> None:
        self.refresh_interval_s = float(refresh_interval_s)
        self.budget: int | None = None
        self.capacity_tokens: int | None = None
        self.ratio: float | None = None
        self.updated_at: float | None = None
        self.error: str | None = None

    async def refresh(
        self,
        provider: TokenBudgetProvider,
        *,
        actor_name: str,
        domain_key: str,
        logger: Any,
    ) -> int | None:
        now = time.time()
        if self.budget is not None and self.updated_at is not None:
            if now - float(self.updated_at) < self.refresh_interval_s:
                return int(self.budget)
        try:
            budget = await provider()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            logger.debug(
                "[model_runtime] dynamic token budget refresh failed actor=%s domain=%s error=%s",
                actor_name,
                domain_key,
                self.error,
            )
            return self.budget
        if budget is None:
            return self.budget
        budget_i = max(1, int(budget))
        self.budget = budget_i
        self.updated_at = now
        self.error = None
        return budget_i

    def budget_from_capacity(self, capacity: int, ratio: float) -> int:
        self.capacity_tokens = int(capacity)
        self.ratio = float(ratio)
        return max(1, int(float(capacity) * float(ratio)))

    def snapshot(self) -> dict[str, Any]:
        return {
            "dynamic_token_budget": self.budget,
            "dynamic_token_capacity_tokens": self.capacity_tokens,
            "dynamic_token_budget_ratio": self.ratio,
            "dynamic_token_budget_updated_at": self.updated_at,
            "dynamic_token_budget_error": self.error,
        }
