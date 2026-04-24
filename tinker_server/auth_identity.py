"""Helpers for request identity, legacy role fallback, and capability checks."""

from __future__ import annotations

from fastapi import Request


def get_user_data(request: Request) -> dict | None:
    user_data = getattr(request.state, "user_data", None)
    return user_data if isinstance(user_data, dict) else None


def get_user_id(request: Request) -> str | None:
    user_data = get_user_data(request)
    if user_data:
        user_id = user_data.get("user_id")
        return str(user_id) if user_id is not None else None
    return None


def get_user_role_from_user_data(user_data: dict | None) -> str | None:
    if not isinstance(user_data, dict):
        return None
    raw_role = str(user_data.get("user_role") or "").strip().lower()
    if raw_role:
        return raw_role
    user_id = str(user_data.get("user_id") or "").strip()
    if user_id == "admin" or bool(user_data.get("is_admin")):
        return "admin"
    if user_id:
        return "user"
    return None


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _caps_from_headers(user_data: dict | None) -> bool:
    if not isinstance(user_data, dict):
        return False
    return _coerce_bool(user_data.get("caps_from_headers"))


def _legacy_caps_for_role(role: str | None) -> tuple[bool, bool, bool, bool]:
    normalized = str(role or "").strip().lower()
    if normalized == "admin":
        return True, True, True, True
    if normalized == "internal":
        return True, True, False, False
    if normalized == "user":
        return True, False, False, False
    return False, False, False, False


def _cap_from_user_data(user_data: dict | None, key: str, index: int) -> bool:
    if not isinstance(user_data, dict):
        return False
    if key == "cap_write":
        return bool(
            get_user_role_from_user_data(user_data)
            or str(user_data.get("user_id") or "").strip()
            or str(user_data.get("apikey_id") or user_data.get("key_id") or "").strip()
        )
    if _caps_from_headers(user_data):
        return _coerce_bool(user_data.get(key))
    return _legacy_caps_for_role(get_user_role_from_user_data(user_data))[index]


def can_write_user_data(user_data: dict | None) -> bool:
    return _cap_from_user_data(user_data, "cap_write", 0)


def can_view_internal_errors_user_data(user_data: dict | None) -> bool:
    return _cap_from_user_data(user_data, "cap_view_internal_errors", 1)


def can_bypass_ownership_user_data(user_data: dict | None) -> bool:
    return _cap_from_user_data(user_data, "cap_bypass_ownership", 2)


def can_manage_system_user_data(user_data: dict | None) -> bool:
    return _cap_from_user_data(user_data, "cap_manage_system", 3)


def can_write(request: Request) -> bool:
    return can_write_user_data(get_user_data(request))


def can_view_internal_errors(request: Request) -> bool:
    return can_view_internal_errors_user_data(get_user_data(request))


def can_bypass_ownership(request: Request) -> bool:
    return can_bypass_ownership_user_data(get_user_data(request))


def can_manage_system(request: Request) -> bool:
    return can_manage_system_user_data(get_user_data(request))


def get_apikey_id(request: Request) -> str | None:
    user_data = get_user_data(request)
    if user_data:
        apikey_id = user_data.get("apikey_id") or user_data.get("key_id")
        if apikey_id is not None:
            value = str(apikey_id).strip()
            if value:
                return value
    return None


def get_account_id(request: Request) -> str | None:
    user_data = get_user_data(request)
    if user_data:
        account_id = user_data.get("account_id")
        if account_id is not None:
            value = str(account_id).strip()
            if value:
                return value
    return None


def get_gateway_request_id(request: Request) -> str | None:
    user_data = get_user_data(request)
    if user_data:
        request_id = user_data.get("request_id")
        if request_id is not None:
            value = str(request_id).strip()
            if value:
                return value
    return None


def get_gateway_session_id(request: Request) -> str | None:
    user_data = get_user_data(request)
    if user_data:
        session_id = user_data.get("session_id")
        if session_id is not None:
            value = str(session_id).strip()
            if value:
                return value
    return None


def get_request_observability_context(request: Request) -> dict[str, str]:
    out: dict[str, str] = {}
    user_id = get_user_id(request)
    if user_id:
        out["user_id"] = user_id
    user_role = get_user_role_from_user_data(get_user_data(request))
    if user_role:
        out["user_role"] = user_role
    account_id = get_account_id(request)
    if account_id:
        out["account_id"] = account_id
    apikey_id = get_apikey_id(request)
    if apikey_id:
        out["apikey_id"] = apikey_id
    gateway_request_id = get_gateway_request_id(request)
    if gateway_request_id:
        out["gateway_request_id"] = gateway_request_id
    gateway_session_id = get_gateway_session_id(request)
    if gateway_session_id:
        out["gateway_session_id"] = gateway_session_id
    return out


def can_access_restricted_models_user_data(user_data: dict | None) -> bool:
    role = get_user_role_from_user_data(user_data)
    return role == "admin"

