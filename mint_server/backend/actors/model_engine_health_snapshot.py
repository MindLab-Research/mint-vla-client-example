from __future__ import annotations

import time
import traceback
from typing import Any


class EngineHealthSnapshot:
    def __init__(self) -> None:
        self.last_claimed_at: float | None = None
        self.last_completed_at: float | None = None
        self.last_renewed_at: float | None = None
        self.max_renew_rpc_latency_s = 0.0
        self.consecutive_renew_failures = 0
        self.last_renew_deadline_slack_s: float | None = None
        self.last_error: str | None = None
        self.last_error_traceback: str | None = None

    def record_claimed(self) -> None:
        self.last_claimed_at = time.time()

    def record_completed(self) -> None:
        self.last_completed_at = time.time()

    def record_success(self) -> None:
        self.record_completed()
        self.clear_error()

    def record_failure(self, error: str) -> None:
        self.last_error = str(error)
        self.last_error_traceback = None
        self.record_completed()

    def record_exception(self, exc: BaseException) -> None:
        self.last_error = f"{type(exc).__name__}: {exc}"
        self.last_error_traceback = traceback.format_exc()

    def clear_error(self) -> None:
        self.last_error = None
        self.last_error_traceback = None

    def clear_transient_scheduler_error(self) -> None:
        error = str(self.last_error or "").lower()
        if not error:
            return
        if "modelworkschedulerconflicterror" in error or "consumer_id mismatch" in error:
            self.clear_error()

    def record_renew_result(self, *, ok: bool, latency_s: float, lease_ttl_s: float) -> None:
        latency = max(0.0, float(latency_s))
        self.max_renew_rpc_latency_s = max(self.max_renew_rpc_latency_s, latency)
        if ok:
            self.last_renewed_at = time.time()
            self.consecutive_renew_failures = 0
            self.last_renew_deadline_slack_s = max(0.0, float(lease_ttl_s) - latency)
            return
        self.consecutive_renew_failures += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "last_claimed_at": self.last_claimed_at,
            "last_completed_at": self.last_completed_at,
            "last_renewed_at": self.last_renewed_at,
            "max_renew_rpc_latency_s": self.max_renew_rpc_latency_s,
            "consecutive_renew_failures": self.consecutive_renew_failures,
            "last_renew_deadline_slack_s": self.last_renew_deadline_slack_s,
            "last_error": self.last_error,
            "last_error_traceback": self.last_error_traceback,
        }
