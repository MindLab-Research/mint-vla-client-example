from types import SimpleNamespace

from tinker_server.auth_identity import (
    can_bypass_ownership_user_data,
    can_manage_system_user_data,
    can_view_internal_errors_user_data,
    can_write_user_data,
)
from tinker_server.gateway_auth import GatewayAuthContext, build_billing_auth_context, extract_gateway_auth_context_from_headers, has_gateway_auth_headers
from tinker_server.routes import futures as futures_route


def test_extract_gateway_auth_context_accepts_explicit_caps_without_role():
    ctx = extract_gateway_auth_context_from_headers(
        {
            "X-MinT-User-Id": "aaaaaaaaaaaaaaaaaaaaaaaa",
            "X-MinT-Apikey-Id": "bbbbbbbbbbbbbbbbbbbbbbbb",
            "X-MinT-Request-Id": "req-123",
            "X-MinT-Cap-Write": "true",
            "X-MinT-Cap-View-Internal-Errors": "false",
            "X-MinT-Cap-Bypass-Ownership": "true",
            "X-MinT-Cap-Manage-System": "false",
            "X-Internal-Token": "secret",
        },
        internal_api_token="secret",
    )

    assert ctx.user_role == "user"
    assert ctx.caps_from_headers is True
    assert ctx.cap_write is True
    assert ctx.cap_view_internal_errors is False
    assert ctx.cap_bypass_ownership is True
    assert ctx.cap_manage_system is False


def test_extract_gateway_auth_context_defaults_write_true_when_other_caps_present():
    ctx = extract_gateway_auth_context_from_headers(
        {
            "X-MinT-User-Id": "aaaaaaaaaaaaaaaaaaaaaaaa",
            "X-MinT-Apikey-Id": "bbbbbbbbbbbbbbbbbbbbbbbb",
            "X-MinT-Request-Id": "req-123",
            "X-MinT-Cap-View-Internal-Errors": "false",
            "X-MinT-Cap-Bypass-Ownership": "false",
            "X-MinT-Cap-Manage-System": "false",
            "X-Internal-Token": "secret",
        },
        internal_api_token="secret",
    )

    assert ctx.caps_from_headers is True
    assert ctx.cap_write is True


def test_has_gateway_auth_headers_detects_cap_headers():
    assert has_gateway_auth_headers({"X-MinT-Cap-Write": "true"}) is True


def test_legacy_internal_role_keeps_write_and_internal_error_visibility():
    user_data = {"user_role": "internal", "caps_from_headers": False}

    assert can_write_user_data(user_data) is True
    assert can_view_internal_errors_user_data(user_data) is True
    assert can_bypass_ownership_user_data(user_data) is False
    assert can_manage_system_user_data(user_data) is False


def test_explicit_caps_still_override_admin_for_non_write_permissions():
    user_data = {
        "user_role": "admin",
        "caps_from_headers": True,
        "cap_write": False,
        "cap_view_internal_errors": False,
        "cap_bypass_ownership": False,
        "cap_manage_system": False,
    }

    assert can_write_user_data(user_data) is True
    assert can_view_internal_errors_user_data(user_data) is False
    assert can_bypass_ownership_user_data(user_data) is False
    assert can_manage_system_user_data(user_data) is False


def test_caps_from_headers_default_write_true_when_missing():
    user_data = {
        "user_role": "user",
        "caps_from_headers": True,
        "cap_view_internal_errors": False,
        "cap_bypass_ownership": False,
        "cap_manage_system": False,
    }

    assert can_write_user_data(user_data) is True


def test_build_billing_auth_context_preserves_caps_from_gateway_state():
    request = SimpleNamespace(
        state=SimpleNamespace(
            gateway_auth=GatewayAuthContext(
                user_id="aaaaaaaaaaaaaaaaaaaaaaaa",
                user_role="user",
                account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
                apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
                request_id="req-1",
                cap_write=True,
                cap_view_internal_errors=True,
                cap_bypass_ownership=False,
                cap_manage_system=False,
                caps_from_headers=True,
            )
        ),
        headers={},
    )

    ctx = build_billing_auth_context(request, fallback_request_id="req-2")

    assert ctx is not None
    assert ctx.cap_write is True
    assert ctx.cap_view_internal_errors is True
    assert ctx.caps_from_headers is True


def test_failed_payload_masks_without_internal_error_cap():
    request = SimpleNamespace(
        state=SimpleNamespace(
            user_data={
                "user_role": "user",
                "caps_from_headers": True,
                "cap_view_internal_errors": False,
            }
        )
    )

    payload = futures_route._failed_payload("secret backend detail", request)

    assert payload["error"] in {"secret backend detail", futures_route.GENERIC_ERROR_MESSAGE}
