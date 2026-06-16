from __future__ import annotations

import os
from typing import Any, Protocol

from mint_server.config import config as server_config


class SamplingAdmissionWorkItem(Protocol):
    @property
    def request_id(self) -> str: ...

    @property
    def op(self) -> str: ...

    @property
    def domain_key(self) -> str: ...

    @property
    def throttle_principal(self) -> str | None: ...

    @property
    def apikey_id(self) -> str | None: ...

    @property
    def user_id(self) -> str | None: ...

    @property
    def token_cost(self) -> int: ...

    @property
    def extra(self) -> dict[str, Any]: ...

def sampling_inflight_admission_mode() -> str:
    raw = (
        os.environ.get("MINT_SAMPLING_INFLIGHT_ADMISSION_MODE")
        or str(getattr(server_config, "sampling_inflight_admission_mode", "observe"))
    )
    mode = str(raw).strip().lower()
    if mode in {"0", "false", "disabled", "disable"}:
        return "off"
    if mode in {"1", "true", "enabled", "enable"}:
        return "enforce"
    if mode in {"off", "observe", "enforce"}:
        return mode
    return "observe"


def sampling_inflight_limit(name: str, config_attr: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        raw = getattr(server_config, config_attr, default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


class AdmissionAccounting:
    def __init__(self) -> None:
        self._inflight_by_domain: dict[str, int] = {}
        self._inflight_by_principal_domain: dict[tuple[str, str], int] = {}
        self._inflight_tokens_by_domain: dict[str, int] = {}
        self._inflight_tokens_by_principal_domain: dict[tuple[str, str], int] = {}
        self._principal_domain_by_request_id: dict[str, tuple[str, str]] = {}
        self._token_cost_by_request_id: dict[str, int] = {}
        self._would_reject: dict[tuple[str, str], int] = {}
        self._reject: dict[tuple[str, str], int] = {}

    @property
    def inflight_by_domain(self) -> dict[str, int]:
        return self._inflight_by_domain

    @inflight_by_domain.setter
    def inflight_by_domain(self, value: dict[str, int]) -> None:
        self._inflight_by_domain = value

    @property
    def inflight_by_principal_domain(self) -> dict[tuple[str, str], int]:
        return self._inflight_by_principal_domain

    @inflight_by_principal_domain.setter
    def inflight_by_principal_domain(self, value: dict[tuple[str, str], int]) -> None:
        self._inflight_by_principal_domain = value

    @property
    def inflight_tokens_by_domain(self) -> dict[str, int]:
        return self._inflight_tokens_by_domain

    @inflight_tokens_by_domain.setter
    def inflight_tokens_by_domain(self, value: dict[str, int]) -> None:
        self._inflight_tokens_by_domain = value

    @property
    def inflight_tokens_by_principal_domain(self) -> dict[tuple[str, str], int]:
        return self._inflight_tokens_by_principal_domain

    @inflight_tokens_by_principal_domain.setter
    def inflight_tokens_by_principal_domain(self, value: dict[tuple[str, str], int]) -> None:
        self._inflight_tokens_by_principal_domain = value

    @property
    def principal_domain_by_request_id(self) -> dict[str, tuple[str, str]]:
        return self._principal_domain_by_request_id

    @principal_domain_by_request_id.setter
    def principal_domain_by_request_id(self, value: dict[str, tuple[str, str]]) -> None:
        self._principal_domain_by_request_id = value

    @property
    def token_cost_by_request_id(self) -> dict[str, int]:
        return self._token_cost_by_request_id

    @token_cost_by_request_id.setter
    def token_cost_by_request_id(self, value: dict[str, int]) -> None:
        self._token_cost_by_request_id = value

    @property
    def would_reject(self) -> dict[tuple[str, str], int]:
        return self._would_reject

    @would_reject.setter
    def would_reject(self, value: dict[tuple[str, str], int]) -> None:
        self._would_reject = value

    @property
    def reject(self) -> dict[tuple[str, str], int]:
        return self._reject

    @reject.setter
    def reject(self, value: dict[tuple[str, str], int]) -> None:
        self._reject = value

    def is_sampling_inflight_work(self, item: SamplingAdmissionWorkItem) -> bool:
        return str(item.op) == "sampling.asample"

    def principal(self, item: SamplingAdmissionWorkItem) -> str:
        extra_principal = item.extra.get("sampling_admission_principal")
        for value in (extra_principal, item.throttle_principal, item.apikey_id, item.user_id):
            text = str(value or "").strip()
            if text:
                return text
        return "anonymous"

    def track_locked(self, item: SamplingAdmissionWorkItem) -> None:
        if not self.is_sampling_inflight_work(item):
            return
        request_id = str(item.request_id)
        if request_id in self._principal_domain_by_request_id:
            return
        domain_key = str(item.domain_key)
        principal = self.principal(item)
        key = (principal, domain_key)
        self._principal_domain_by_request_id[request_id] = key
        token_cost = max(1, int(item.token_cost))
        self._token_cost_by_request_id[request_id] = token_cost
        self._inflight_by_domain[domain_key] = self._inflight_by_domain.get(domain_key, 0) + 1
        self._inflight_by_principal_domain[key] = self._inflight_by_principal_domain.get(key, 0) + 1
        self._inflight_tokens_by_domain[domain_key] = (
            self._inflight_tokens_by_domain.get(domain_key, 0) + token_cost
        )
        self._inflight_tokens_by_principal_domain[key] = (
            self._inflight_tokens_by_principal_domain.get(key, 0) + token_cost
        )

    def untrack_locked(self, request_id: str) -> None:
        key = self._principal_domain_by_request_id.pop(str(request_id), None)
        if key is None:
            return
        principal, domain_key = key
        token_cost = max(1, int(self._token_cost_by_request_id.pop(str(request_id), 1)))
        current_domain = self._inflight_by_domain.get(domain_key, 0) - 1
        if current_domain > 0:
            self._inflight_by_domain[domain_key] = current_domain
        else:
            self._inflight_by_domain.pop(domain_key, None)
        current_principal = self._inflight_by_principal_domain.get((principal, domain_key), 0) - 1
        if current_principal > 0:
            self._inflight_by_principal_domain[(principal, domain_key)] = current_principal
        else:
            self._inflight_by_principal_domain.pop((principal, domain_key), None)
        current_domain_tokens = self._inflight_tokens_by_domain.get(domain_key, 0) - token_cost
        if current_domain_tokens > 0:
            self._inflight_tokens_by_domain[domain_key] = current_domain_tokens
        else:
            self._inflight_tokens_by_domain.pop(domain_key, None)
        current_principal_tokens = (
            self._inflight_tokens_by_principal_domain.get((principal, domain_key), 0) - token_cost
        )
        if current_principal_tokens > 0:
            self._inflight_tokens_by_principal_domain[(principal, domain_key)] = current_principal_tokens
        else:
            self._inflight_tokens_by_principal_domain.pop((principal, domain_key), None)

    def limit_decision_locked(self, item: SamplingAdmissionWorkItem) -> dict[str, Any]:
        mode = sampling_inflight_admission_mode()
        if mode == "off" or not self.is_sampling_inflight_work(item):
            return {"ok": True, "mode": mode}
        domain_key = str(item.domain_key)
        principal = self.principal(item)
        principal_key = (principal, domain_key)
        principal_current = int(self._inflight_by_principal_domain.get(principal_key, 0))
        domain_current = int(self._inflight_by_domain.get(domain_key, 0))
        token_cost = max(1, int(item.token_cost))
        principal_token_current = int(self._inflight_tokens_by_principal_domain.get(principal_key, 0))
        domain_token_current = int(self._inflight_tokens_by_domain.get(domain_key, 0))
        principal_limit = sampling_inflight_limit(
            "MINT_SAMPLING_MAX_INFLIGHT_PER_PRINCIPAL_DOMAIN",
            "sampling_max_inflight_per_principal_domain",
            1024,
        )
        domain_limit = sampling_inflight_limit(
            "MINT_SAMPLING_MAX_INFLIGHT_PER_DOMAIN",
            "sampling_max_inflight_per_domain",
            10240,
        )
        principal_token_limit = sampling_inflight_limit(
            "MINT_SAMPLING_MAX_INFLIGHT_TOKENS_PER_PRINCIPAL_DOMAIN",
            "sampling_max_inflight_tokens_per_principal_domain",
            0,
        )
        domain_token_limit = sampling_inflight_limit(
            "MINT_SAMPLING_MAX_INFLIGHT_TOKENS_PER_DOMAIN",
            "sampling_max_inflight_tokens_per_domain",
            0,
        )
        reason: str | None = None
        current = 0
        limit = 0
        if principal_limit > 0 and principal_current >= principal_limit:
            reason = "principal_domain_inflight_limit_exceeded"
            current = principal_current
            limit = principal_limit
        elif domain_limit > 0 and domain_current >= domain_limit:
            reason = "domain_inflight_limit_exceeded"
            current = domain_current
            limit = domain_limit
        elif principal_token_limit > 0 and principal_token_current + token_cost > principal_token_limit:
            reason = "principal_domain_token_budget_exceeded"
            current = principal_token_current + token_cost
            limit = principal_token_limit
        elif domain_token_limit > 0 and domain_token_current + token_cost > domain_token_limit:
            reason = "domain_token_budget_exceeded"
            current = domain_token_current + token_cost
            limit = domain_token_limit
        if reason is None:
            return {
                "ok": True,
                "mode": mode,
                "domain_key": domain_key,
                "principal": principal,
                "principal_current": principal_current,
                "principal_limit": principal_limit,
                "domain_current": domain_current,
                "domain_limit": domain_limit,
                "token_cost": token_cost,
                "principal_token_current": principal_token_current,
                "principal_token_limit": principal_token_limit,
                "domain_token_current": domain_token_current,
                "domain_token_limit": domain_token_limit,
            }
        counter_key = (domain_key, reason)
        if mode == "observe":
            self._would_reject[counter_key] = self._would_reject.get(counter_key, 0) + 1
            return {
                "ok": True,
                "mode": mode,
                "would_reject": True,
                "reason": reason,
                "domain_key": domain_key,
                "principal": principal,
                "current": current,
                "limit": limit,
                "principal_current": principal_current,
                "principal_limit": principal_limit,
                "domain_current": domain_current,
                "domain_limit": domain_limit,
                "token_cost": token_cost,
                "principal_token_current": principal_token_current,
                "principal_token_limit": principal_token_limit,
                "domain_token_current": domain_token_current,
                "domain_token_limit": domain_token_limit,
            }
        self._reject[counter_key] = self._reject.get(counter_key, 0) + 1
        return {
            "ok": False,
            "mode": mode,
            "reason": reason,
            "domain_key": domain_key,
            "principal": principal,
            "current": current,
            "limit": limit,
            "principal_current": principal_current,
            "principal_limit": principal_limit,
            "domain_current": domain_current,
            "domain_limit": domain_limit,
            "token_cost": token_cost,
            "principal_token_current": principal_token_current,
            "principal_token_limit": principal_token_limit,
            "domain_token_current": domain_token_current,
            "domain_token_limit": domain_token_limit,
            "retry_after_s": 5,
        }

    def inflight_snapshot(self) -> dict[str, Any]:
        principal_domain_max_by_domain: dict[str, int] = {}
        active_principals_by_domain: dict[str, int] = {}
        principal_domain_token_max_by_domain: dict[str, int] = {}
        for (_principal, domain_key), count in self._inflight_by_principal_domain.items():
            principal_domain_max_by_domain[domain_key] = max(
                principal_domain_max_by_domain.get(domain_key, 0),
                int(count),
            )
            active_principals_by_domain[domain_key] = active_principals_by_domain.get(domain_key, 0) + 1
        for (_principal, domain_key), tokens in self._inflight_tokens_by_principal_domain.items():
            principal_domain_token_max_by_domain[domain_key] = max(
                principal_domain_token_max_by_domain.get(domain_key, 0),
                int(tokens),
            )
        return {
            "mode": sampling_inflight_admission_mode(),
            "per_principal_domain_limit": sampling_inflight_limit(
                "MINT_SAMPLING_MAX_INFLIGHT_PER_PRINCIPAL_DOMAIN",
                "sampling_max_inflight_per_principal_domain",
                1024,
            ),
            "per_domain_limit": sampling_inflight_limit(
                "MINT_SAMPLING_MAX_INFLIGHT_PER_DOMAIN",
                "sampling_max_inflight_per_domain",
                10240,
            ),
            "per_principal_domain_token_limit": sampling_inflight_limit(
                "MINT_SAMPLING_MAX_INFLIGHT_TOKENS_PER_PRINCIPAL_DOMAIN",
                "sampling_max_inflight_tokens_per_principal_domain",
                0,
            ),
            "per_domain_token_limit": sampling_inflight_limit(
                "MINT_SAMPLING_MAX_INFLIGHT_TOKENS_PER_DOMAIN",
                "sampling_max_inflight_tokens_per_domain",
                0,
            ),
            "by_domain": dict(sorted(self._inflight_by_domain.items())),
            "principal_domain_max_by_domain": dict(sorted(principal_domain_max_by_domain.items())),
            "active_principals_by_domain": dict(sorted(active_principals_by_domain.items())),
            "tokens_by_domain": dict(sorted(self._inflight_tokens_by_domain.items())),
            "principal_domain_token_max_by_domain": dict(
                sorted(principal_domain_token_max_by_domain.items())
            ),
        }

    def admission_counters_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "would_reject": self._counter_records(self._would_reject),
            "reject": self._counter_records(self._reject),
        }

    @staticmethod
    def _counter_records(counters: dict[tuple[str, str], int]) -> list[dict[str, Any]]:
        return [
            {"domain_key": domain_key, "reason": reason, "count": int(count)}
            for (domain_key, reason), count in sorted(counters.items())
        ]
