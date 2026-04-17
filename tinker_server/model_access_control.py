"""Product-level model catalog gating.

`RESTRICTED_MODELS` stays hidden from regular Mint platform API users.
Privileged callers still bypass that product gate through an explicit policy helper.
"""

from typing import Optional

from .auth_identity import can_access_restricted_models_user_data

RESTRICTED_MODELS = {
    "moonshotai/Kimi-K2-Instruct",
    "moonshotai/Kimi-K2-Thinking",
}


def is_mint_platform_user(user_data: Optional[dict]) -> bool:
    """Return whether the caller should receive the restricted model catalog."""
    if user_data is None:
        return False

    if can_access_restricted_models_user_data(user_data):
        return False

    user_id = user_data.get("user_id")
    return user_id is not None


def can_access_model(model_name: str, user_data: Optional[dict]) -> bool:
    """Return whether this caller may use `model_name`."""
    if can_access_restricted_models_user_data(user_data):
        return True

    if not is_mint_platform_user(user_data):
        return True

    if model_name in RESTRICTED_MODELS:
        return False

    return True


def get_access_denied_error(model_name: str) -> str:
    """Get error message for denied model access.

    Args:
        model_name: Model name that was denied

    Returns:
        Error message string
    """
    return (
        f"Access denied: Model '{model_name}' is not available for MINT platform users. "
        f"Please contact support for access to enterprise models."
    )
