"""Gateway-forwarded auth context extraction and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from fastapi import HTTPException, Request

_OBJECT_ID_RE = re.compile(r"^[0-9a-fA-F]{24}$")
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class GatewayAuthContext:
    user_id: str
    user_role: str
    account_id: str
    apikey_id: str
    request_id: str
    session_id: str = ""
    cap_write: bool = False
    cap_view_internal_errors: bool = False
    cap_bypass_ownership: bool = False
    cap_manage_system: bool = False
    caps_from_headers: bool = False


_USER_ID_HEADERS = ("x-mint-user-id",)
_USER_ROLE_HEADERS = ("x-mint-user-role",)
_ACCOUNT_ID_HEADERS = ("x-mint-account-id",)
_APIKEY_ID_HEADERS = ("x-mint-apikey-id",)
_REQUEST_ID_HEADERS = ("x-mint-request-id",)
_SESSION_ID_HEADERS = ("x-mint-session-id",)
_INTERNAL_TOKEN_HEADERS = ("x-internal-token",)
_CAP_WRITE_HEADERS = ("x-mint-cap-write",)
_CAP_VIEW_INTERNAL_ERRORS_HEADERS = ("x-mint-cap-view-internal-errors",)
_CAP_BYPASS_OWNERSHIP_HEADERS = ("x-mint-cap-bypass-ownership",)
_CAP_MANAGE_SYSTEM_HEADERS = ("x-mint-cap-manage-system",)
_SUPPORTED_USER_ROLES = {"user", "admin", "internal"}


def _canonical_headers(headers: dict[str, str]) -> dict[str, str]:
    return {str(k).lower(): str(v) for k, v in headers.items()}


def _get_first_header(headers: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = (headers.get(name) or "").strip()
        if value:
            return value
    return ""


def _has_any_header(headers: dict[str, str], names: tuple[str, ...]) -> bool:
    for name in names:
        if name in headers:
            return True
    return False


def _require_header(headers: dict[str, str], names: tuple[str, ...], field_name: str) -> str:
    value = _get_first_header(headers, names)
    if not value:
        raise HTTPException(status_code=400, detail=f"Missing required header: {field_name}")
    return value


def _validate_object_id(value: str, field_name: str) -> str:
    if not _OBJECT_ID_RE.match(value):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name}: must be a 24-character hex ObjectId",
        )
    return value.lower()


def _validate_internal_token(headers: dict[str, str], internal_api_token: str) -> None:
    if not internal_api_token:
        raise HTTPException(status_code=503, detail="Gateway auth is not configured on this server")
    token = _get_first_header(headers, _INTERNAL_TOKEN_HEADERS)
    if token != internal_api_token:
        raise HTTPException(status_code=403, detail="Invalid or missing internal token")


def _validate_user_role(value: str) -> str:
    role = str(value or "").strip().lower()
    if role not in _SUPPORTED_USER_ROLES:
        raise HTTPException(status_code=400, detail="Invalid X-MinT-User-Role")
    return role


def _parse_bool_header(headers: dict[str, str], names: tuple[str, ...], field_name: str) -> bool:
    raw = _get_first_header(headers, names)
    if raw == "":
        return False
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise HTTPException(status_code=400, detail=f"Invalid {field_name}: expected boolean")


def _extract_cap_headers(headers: dict[str, str]) -> tuple[bool, bool, bool, bool, bool]:
    cap_names = (
        _CAP_WRITE_HEADERS,
        _CAP_VIEW_INTERNAL_ERRORS_HEADERS,
        _CAP_BYPASS_OWNERSHIP_HEADERS,
        _CAP_MANAGE_SYSTEM_HEADERS,
    )
    caps_from_headers = any(_has_any_header(headers, names) for names in cap_names)
    if not caps_from_headers:
        return False, False, False, False, False
    return (
        True,
        True,
        _parse_bool_header(
            headers,
            _CAP_VIEW_INTERNAL_ERRORS_HEADERS,
            "X-MinT-Cap-View-Internal-Errors",
        ),
        _parse_bool_header(headers, _CAP_BYPASS_OWNERSHIP_HEADERS, "X-MinT-Cap-Bypass-Ownership"),
        _parse_bool_header(headers, _CAP_MANAGE_SYSTEM_HEADERS, "X-MinT-Cap-Manage-System"),
    )


def has_gateway_auth_headers(headers: dict[str, str]) -> bool:
    headers = _canonical_headers(headers)
    return bool(
        _get_first_header(headers, _USER_ID_HEADERS)
        or _get_first_header(headers, _USER_ROLE_HEADERS)
        or _get_first_header(headers, _ACCOUNT_ID_HEADERS)
        or _get_first_header(headers, _APIKEY_ID_HEADERS)
        or _has_any_header(headers, _CAP_WRITE_HEADERS)
        or _has_any_header(headers, _CAP_VIEW_INTERNAL_ERRORS_HEADERS)
        or _has_any_header(headers, _CAP_BYPASS_OWNERSHIP_HEADERS)
        or _has_any_header(headers, _CAP_MANAGE_SYSTEM_HEADERS)
    )


def extract_gateway_auth_context_from_headers(
    headers: dict[str, str],
    *,
    internal_api_token: str = "",
) -> GatewayAuthContext:
    headers = _canonical_headers(headers)
    _validate_internal_token(headers, internal_api_token)

    user_id = _validate_object_id(
        _require_header(headers, _USER_ID_HEADERS, "X-MinT-User-Id"),
        "X-MinT-User-Id",
    )
    caps_from_headers, cap_write, cap_view_internal_errors, cap_bypass_ownership, cap_manage_system = _extract_cap_headers(headers)
    role_value = _get_first_header(headers, _USER_ROLE_HEADERS)
    if caps_from_headers:
        user_role = _validate_user_role(role_value) if role_value else "user"
    else:
        user_role = _validate_user_role(
            _require_header(headers, _USER_ROLE_HEADERS, "X-MinT-User-Role")
        )
    account_id_raw = _get_first_header(headers, _ACCOUNT_ID_HEADERS) or user_id
    account_id = _validate_object_id(account_id_raw, "X-MinT-Account-Id")
    apikey_id = _validate_object_id(
        _require_header(headers, _APIKEY_ID_HEADERS, "X-MinT-Apikey-Id"),
        "X-MinT-Apikey-Id",
    )
    request_id = _require_header(headers, _REQUEST_ID_HEADERS, "X-MinT-Request-Id")
    session_id = _get_first_header(headers, _SESSION_ID_HEADERS)
    return GatewayAuthContext(
        user_id=user_id,
        user_role=user_role,
        account_id=account_id,
        apikey_id=apikey_id,
        request_id=request_id,
        session_id=session_id,
        cap_write=cap_write,
        cap_view_internal_errors=cap_view_internal_errors,
        cap_bypass_ownership=cap_bypass_ownership,
        cap_manage_system=cap_manage_system,
        caps_from_headers=caps_from_headers,
    )


def extract_gateway_auth_context(request: Request, *, internal_api_token: str = "") -> GatewayAuthContext:
    return extract_gateway_auth_context_from_headers(
        dict(request.headers),
        internal_api_token=internal_api_token,
    )


def build_billing_auth_context(
    request: Request,
    *,
    fallback_request_id: str | None = None,
) -> GatewayAuthContext | None:
    ctx = getattr(request.state, "gateway_auth", None)
    if isinstance(ctx, GatewayAuthContext):
        request_id = (ctx.request_id or "").strip() or str(fallback_request_id or "").strip()
        if not request_id:
            return None
        if request_id == ctx.request_id:
            return ctx
        return GatewayAuthContext(
            user_id=ctx.user_id,
            user_role=ctx.user_role,
            account_id=ctx.account_id,
            apikey_id=ctx.apikey_id,
            request_id=request_id,
            session_id=ctx.session_id,
            cap_write=ctx.cap_write,
            cap_view_internal_errors=ctx.cap_view_internal_errors,
            cap_bypass_ownership=ctx.cap_bypass_ownership,
            cap_manage_system=ctx.cap_manage_system,
            caps_from_headers=ctx.caps_from_headers,
        )

    user_data = getattr(request.state, "user_data", None)
    if not isinstance(user_data, dict):
        return None

    user_id = str(user_data.get("user_id") or "").strip()
    apikey_id = str(user_data.get("apikey_id") or user_data.get("key_id") or "").strip()
    account_id = str(user_data.get("account_id") or user_id).strip()
    request_id = (
        str(user_data.get("request_id") or "").strip()
        or str(fallback_request_id or "").strip()
        or _get_first_header(_canonical_headers(dict(request.headers)), _REQUEST_ID_HEADERS)
    )
    session_id = str(user_data.get("session_id") or "").strip()

    if not user_id or not apikey_id or not request_id:
        return None
    try:
        user_role = str(user_data.get("user_role") or "").strip().lower()
        if not user_role:
            user_role = "admin" if user_id == "admin" or bool(user_data.get("is_admin")) else "user"
        return GatewayAuthContext(
            user_id=_validate_object_id(user_id, "user_id"),
            user_role=_validate_user_role(user_role),
            account_id=_validate_object_id(account_id, "account_id"),
            apikey_id=_validate_object_id(apikey_id, "apikey_id"),
            request_id=request_id,
            session_id=session_id,
            cap_write=True,
            cap_view_internal_errors=bool(user_data.get("cap_view_internal_errors")),
            cap_bypass_ownership=bool(user_data.get("cap_bypass_ownership")),
            cap_manage_system=bool(user_data.get("cap_manage_system")),
            caps_from_headers=bool(user_data.get("caps_from_headers")),
        )
    except HTTPException:
        return None


def get_gateway_auth_context(request: Request) -> GatewayAuthContext | None:
    ctx = getattr(request.state, "gateway_auth", None)
    if isinstance(ctx, GatewayAuthContext):
        return ctx
    return None
