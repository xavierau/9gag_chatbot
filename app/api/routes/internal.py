"""Internal API endpoints for MCP server access.

These endpoints are protected by an internal secret and should only be
accessible to MCP servers running on the same infrastructure.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.dependencies import DbSession
from app.infrastructure.oauth.exceptions import NoTokenError, TokenRevokedError
from app.infrastructure.oauth.google_token_service import GoogleTokenService

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Response Models
# =============================================================================


class TokenResponse(BaseModel):
    """Response model for internal token endpoint."""

    access_token: str
    email: str | None = None


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    error_code: str


# =============================================================================
# Dependencies
# =============================================================================


def verify_internal_secret(
    x_internal_secret: str = Header(..., alias="X-Internal-Secret"),
) -> str:
    """Verify the internal API secret header.

    Args:
        x_internal_secret: The secret header value

    Returns:
        The verified secret

    Raises:
        HTTPException: If secret is missing or invalid
    """
    if not settings.internal_api_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Internal API not configured",
        )

    if x_internal_secret != settings.internal_api_secret:
        logger.warning("Invalid internal API secret attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal secret",
        )

    return x_internal_secret


InternalSecret = Annotated[str, Depends(verify_internal_secret)]


def get_token_service(db: DbSession) -> GoogleTokenService:
    """Dependency to get GoogleTokenService instance."""
    return GoogleTokenService(
        db=db,
        client_id=settings.google_oauth_client_id,
        client_secret=settings.google_oauth_client_secret,
        redirect_uri=settings.google_oauth_redirect_uri,
        encryption_key=settings.google_oauth_encryption_key,
    )


TokenService = Annotated[GoogleTokenService, Depends(get_token_service)]


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/oauth/google/token", response_model=TokenResponse)
async def get_google_token(
    user_id: str = Query(..., description="User ID to get token for"),
    _secret: InternalSecret = None,
    token_service: TokenService = None,
) -> TokenResponse:
    """Get a valid Google OAuth access token for a user.

    This endpoint is used by MCP servers to obtain access tokens
    for making Google API calls on behalf of users.

    The access token will be automatically refreshed if it's expired
    or about to expire.

    Args:
        user_id: The user's unique identifier
        _secret: Internal API secret (validated by dependency)

    Returns:
        TokenResponse with access_token and email

    Raises:
        HTTPException: 404 if no token found, 403 if token revoked
    """
    try:
        access_token = await token_service.get_valid_access_token(user_id)

        # Get email for context
        status_data = await token_service.get_token_status(user_id)
        email = status_data.get("google_email") if status_data else None

        return TokenResponse(
            access_token=access_token,
            email=email,
        )

    except NoTokenError:
        logger.info(f"No token found for user {user_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "No Google account linked",
                "error_code": "NO_TOKEN",
            },
        )

    except TokenRevokedError:
        logger.info(f"Token revoked for user {user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "Google account access revoked",
                "error_code": "TOKEN_REVOKED",
            },
        )

    except Exception as e:
        logger.error(f"Failed to get token for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to retrieve token",
                "error_code": "TOKEN_ERROR",
            },
        )


@router.get("/oauth/google/status")
async def get_google_status(
    user_id: str = Query(..., description="User ID to check"),
    _secret: InternalSecret = None,
    token_service: TokenService = None,
) -> dict:
    """Check if a user has a valid Google OAuth token.

    This endpoint is used by MCP servers to check if they can
    make Google API calls for a user before attempting to do so.

    Args:
        user_id: The user's unique identifier
        _secret: Internal API secret (validated by dependency)

    Returns:
        Dict with 'linked' boolean and optional account info
    """
    status_data = await token_service.get_token_status(user_id)

    if not status_data:
        return {
            "linked": False,
            "email": None,
        }

    return {
        "linked": True,
        "email": status_data.get("google_email"),
        "scopes": status_data.get("scopes"),
    }
