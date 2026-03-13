"""Helpers for request identity and admin-role checks."""

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


def is_admin_user_data(user_data: dict | None) -> bool:
    role = get_user_role_from_user_data(user_data)
    return role == "admin"


def is_admin_request(request: Request) -> bool:
    return is_admin_user_data(get_user_data(request))
