from __future__ import annotations

from typing import Any, Callable


class BillingOutbox:
    """Billing outbox operations backed by the task-store hot KV helper."""

    def __init__(
        self,
        hot_kv: Any,
        *,
        inc_metric: Callable[[str, int], None],
        metrics_snapshot: Callable[[], dict[str, Any]],
    ) -> None:
        self._hot_kv = hot_kv
        self._inc_metric = inc_metric
        self._metrics_snapshot = metrics_snapshot

    def append(
        self,
        *,
        observations: list[dict[str, Any]],
        source: str = "unknown",
        now: float | None = None,
    ) -> dict[str, Any]:
        normalized = [dict(item) for item in observations if isinstance(item, dict)]
        if not normalized:
            return {
                "ok": True,
                "source": str(source),
                "inserted": 0,
                "duplicate": 0,
                "conflicts": 0,
                "errors": [],
            }
        out = self._hot_kv.append_billing_outbox(observations=normalized, source=source, now=now)
        inserted = int(out.get("inserted") or 0)
        conflicts = int(out.get("conflicts") or 0)
        errors = out.get("errors") if isinstance(out.get("errors"), list) else []
        if inserted:
            self._inc_metric("event_inserted", inserted)
        if conflicts:
            self._inc_metric("outbox_conflict", conflicts)
        if errors:
            self._inc_metric("write_error", len(errors))
        return out

    def append_after_terminal_success(
        self,
        *,
        observations: list[dict[str, Any]] | None,
        source: str,
        now: float,
    ) -> dict[str, Any]:
        normalized = [dict(item) for item in (observations or []) if isinstance(item, dict)]
        if not normalized:
            return {}
        try:
            billing_result = self.append(
                observations=normalized,
                source=source,
                now=now,
            )
            if not bool(billing_result.get("ok")):
                return {"billing_status": "dropped", "billing_error": billing_result}
            inserted = int(billing_result.get("inserted") or 0)
            if inserted > 0:
                return {
                    "billing_status": "outboxed",
                    "billing_observation_count": inserted,
                }
            return {}
        except Exception as e:
            self._inc_metric("write_error", 1)
            return {"billing_status": "dropped", "billing_error": f"{type(e).__name__}: {e}"}

    def claim(
        self,
        *,
        claim_id: str,
        limit: int = 100,
        lease_ttl_s: float = 60.0,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        return self._hot_kv.claim_billing_outbox(
            claim_id=str(claim_id),
            limit=int(limit),
            lease_ttl_s=float(lease_ttl_s),
            now=now,
        )

    def delete_claim(
        self,
        *,
        claim_id: str,
        outbox_ids: list[int],
    ) -> dict[str, Any]:
        return self._hot_kv.delete_billing_outbox_claim(
            claim_id=str(claim_id),
            outbox_ids=[int(value) for value in outbox_ids],
        )

    def mark_claim_failed(
        self,
        *,
        claim_id: str,
        outbox_ids: list[int],
        permanent: bool,
        error: str,
        now: float | None = None,
    ) -> dict[str, Any]:
        return self._hot_kv.mark_billing_outbox_claim_failed(
            claim_id=str(claim_id),
            outbox_ids=[int(value) for value in outbox_ids],
            permanent=bool(permanent),
            error=str(error),
            now=now,
        )

    def stats(self, *, now: float | None = None) -> dict[str, Any]:
        stats = self._hot_kv.billing_outbox_stats(now=now)
        stats["metrics"] = self._metrics_snapshot()
        return stats
