from __future__ import annotations


class SchedulerCounters:
    def __init__(self) -> None:
        self.completed = 0
        self.failed = 0
        self.requeued = 0
        self.stale_dropped = 0
        self.appended = 0
        self.assigned = 0
        self.claimed = 0
        self.reaper_recovered = 0
        self.reaper_scanned = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "appended": self.appended,
            "assigned": self.assigned,
            "claimed": self.claimed,
            "completed": self.completed,
            "failed": self.failed,
            "requeued": self.requeued,
            "stale_dropped": self.stale_dropped,
            "reaper_scanned": self.reaper_scanned,
            "reaper_recovered": self.reaper_recovered,
        }
