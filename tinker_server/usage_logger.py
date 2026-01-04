"""Usage logging for API calls - tracks user operations and token counts."""

import json
import logging
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .config import config

logger = logging.getLogger(__name__)

OperationType = Literal["forward_backward", "sample_prefill", "sample_generation"]


@dataclass
class UsageLog:
    """Single usage log entry."""

    user_id: str
    operation_type: OperationType
    model_name: str
    token_count: int
    session_id: str
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)


class UsageLogger:
    """File-based usage logger with daily rotation.

    Logs are stored as JSONL files in the configured directory:
        {usage_log_dir}/usage_{date}.jsonl

    Thread-safe for concurrent writes.
    """

    def __init__(self, log_dir: str | None = None):
        self.log_dir = Path(log_dir or config.usage_log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        logger.info(f"UsageLogger initialized: {self.log_dir}")

    def _get_log_file(self, date: str | None = None) -> Path:
        """Get log file path for a given date (default: today)."""
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_dir / f"usage_{date}.jsonl"

    def log(
        self,
        user_id: str,
        operation_type: OperationType,
        model_name: str,
        token_count: int,
        session_id: str,
        request_id: str | None = None,
    ) -> UsageLog:
        """Log a usage event.

        Args:
            user_id: User ID from decrypted token
            operation_type: Type of operation (sample, forward_backward, compute_logprobs)
            model_name: Model name used
            token_count: Number of tokens processed
            session_id: Training/inference session ID
            request_id: Optional request ID (auto-generated if not provided)

        Returns:
            The created UsageLog entry
        """
        entry = UsageLog(
            user_id=user_id,
            operation_type=operation_type,
            model_name=model_name,
            token_count=token_count,
            session_id=session_id,
            request_id=request_id or str(uuid.uuid4()),
        )

        log_file = self._get_log_file()
        with self._lock:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

        logger.debug(
            f"Usage logged: user={user_id}, op={operation_type}, "
            f"model={model_name}, tokens={token_count}"
        )
        return entry

    def query_logs(
        self,
        since: datetime | None = None,
        user_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict], int, bool]:
        """Query usage logs with optional filters.

        Args:
            since: Only return logs after this timestamp
            user_id: Filter by user ID
            limit: Maximum number of logs to return
            offset: Number of logs to skip

        Returns:
            Tuple of (logs, total_count, has_more)
        """
        all_logs = []

        # Read all log files (sorted by date)
        log_files = sorted(self.log_dir.glob("usage_*.jsonl"))

        for log_file in log_files:
            # Extract date from filename to skip old files
            if since:
                file_date_str = log_file.stem.replace("usage_", "")
                try:
                    file_date = datetime.strptime(file_date_str, "%Y-%m-%d").replace(
                        tzinfo=timezone.utc
                    )
                    # Skip files that are entirely before 'since'
                    if file_date.date() < since.date():
                        continue
                except ValueError:
                    pass

            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)

                        # Apply filters
                        if user_id and entry.get("user_id") != user_id:
                            continue

                        if since:
                            entry_time = datetime.fromisoformat(
                                entry["timestamp"].replace("Z", "+00:00")
                            )
                            if entry_time < since:
                                continue

                        all_logs.append(entry)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in {log_file}: {line[:50]}...")

        # Sort by timestamp descending (newest first)
        all_logs.sort(key=lambda x: x["timestamp"], reverse=True)

        total_count = len(all_logs)
        paginated = all_logs[offset : offset + limit]
        has_more = offset + limit < total_count

        return paginated, total_count, has_more

    def get_user_summary(self, user_id: str) -> dict:
        """Get usage summary for a specific user.

        Args:
            user_id: User ID to summarize

        Returns:
            Summary dict with total_tokens and operation_counts
        """
        logs, _, _ = self.query_logs(user_id=user_id, limit=1_000_000)

        total_tokens = 0
        operation_counts: dict[str, int] = {}

        for log in logs:
            total_tokens += log.get("token_count", 0)
            op_type = log.get("operation_type", "unknown")
            operation_counts[op_type] = operation_counts.get(op_type, 0) + 1

        return {
            "total_tokens": total_tokens,
            "operation_counts": operation_counts,
        }


# Global singleton instance
_usage_logger: UsageLogger | None = None


def get_usage_logger() -> UsageLogger:
    """Get or create the global UsageLogger instance."""
    global _usage_logger
    if _usage_logger is None:
        _usage_logger = UsageLogger()
    return _usage_logger
