"""Model access control for restricting certain models to specific user types.

MINT platform users (identified by sk- tokens) have restricted access to certain models.
Admin users (using hardcoded API keys) have full access to all models.
"""

from typing import Optional

from .auth_identity import is_admin_user_data

# Models that are restricted from MINT platform users
RESTRICTED_MODELS = {
    "moonshotai/Kimi-K2-Instruct",
    "moonshotai/Kimi-K2-Thinking",
}


def is_mint_platform_user(user_data: Optional[dict]) -> bool:
    """Check if user is from MINT platform (sk- token user).

    Args:
        user_data: User data from auth middleware (contains user_id)

    Returns:
        True if user is from MINT platform, False if admin or unauthenticated
    """
    if user_data is None:
        return False

    if is_admin_user_data(user_data):
        return False

    # Any other user_id means it's a MINT platform user (sk- token)
    user_id = user_data.get("user_id")
    return user_id is not None


def can_access_model(model_name: str, user_data: Optional[dict]) -> bool:
    """Check if user can access the specified model.

    Args:
        model_name: Model name to check access for
        user_data: User data from auth middleware

    Returns:
        True if user can access the model, False otherwise
    """
    # Admin users can access all models
    if not is_mint_platform_user(user_data):
        return True

    # MINT platform users cannot access restricted models
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
