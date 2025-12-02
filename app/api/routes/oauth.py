"""OAuth flow endpoints for Google Calendar and Gmail integration.

This module provides endpoints for:
- Initiating OAuth flow (generating authorization URL)
- Handling OAuth callback from Google
- Revoking OAuth tokens
- Checking OAuth status
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.dependencies import DbSession
from app.infrastructure.oauth.exceptions import InvalidStateError
from app.infrastructure.oauth.google_token_service import GoogleTokenService

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================


class AuthorizeRequest(BaseModel):
    """Request model for initiating OAuth flow."""

    user_id: str
    whatsapp_phone: str


class AuthorizeResponse(BaseModel):
    """Response with OAuth authorization URL."""

    authorization_url: str
    state: str


class TokenStatusResponse(BaseModel):
    """Response model for token status."""

    linked: bool
    google_email: str | None = None
    google_name: str | None = None
    scopes: list[str] | None = None


class RevokeRequest(BaseModel):
    """Request model for revoking tokens."""

    user_id: str


class RevokeResponse(BaseModel):
    """Response model for token revocation."""

    success: bool
    message: str


# =============================================================================
# Dependencies
# =============================================================================


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


@router.get("/google/authorize", response_model=AuthorizeResponse)
async def authorize_google(
    user_id: str = Query(..., description="User ID (WhatsApp phone number)"),
    whatsapp_phone: str = Query(..., description="WhatsApp phone number for callback"),
    token_service: TokenService = None,
) -> AuthorizeResponse:
    """Generate Google OAuth authorization URL.

    This endpoint creates a state token for CSRF protection and returns
    the Google OAuth URL that the user should be redirected to.

    Args:
        user_id: The user's unique identifier (WhatsApp phone)
        whatsapp_phone: Phone number for WhatsApp callback after OAuth

    Returns:
        AuthorizeResponse with authorization URL and state token
    """
    if not settings.google_oauth_client_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth is not configured",
        )

    # Create state for CSRF protection
    state = await token_service.create_oauth_state(
        user_id=user_id,
        whatsapp_phone=whatsapp_phone,
    )

    # Build authorization URL
    authorization_url = token_service.build_authorization_url(state)

    logger.info(f"Generated OAuth authorization URL for user {user_id}")

    return AuthorizeResponse(
        authorization_url=authorization_url,
        state=state,
    )


@router.get("/google/callback")
async def oauth_callback(
    code: str = Query(None, description="Authorization code from Google"),
    state: str = Query(None, description="State parameter for CSRF validation"),
    error: str = Query(None, description="Error from Google OAuth"),
    token_service: TokenService = None,
) -> HTMLResponse:
    """Handle Google OAuth callback.

    This endpoint is called by Google after the user authorizes the application.
    It exchanges the authorization code for tokens and stores them encrypted.

    Args:
        code: Authorization code from Google
        state: State parameter for CSRF validation
        error: Error message if OAuth failed

    Returns:
        HTML page with success/error message and WhatsApp deep link
    """
    # Handle OAuth errors
    if error:
        logger.warning(f"OAuth error: {error}")
        return _build_error_html(
            title="Authorization Failed",
            message=f"Google authorization was denied: {error}",
        )

    if not code or not state:
        return _build_error_html(
            title="Invalid Request",
            message="Missing authorization code or state parameter",
        )

    # Validate state
    state_data = await token_service.validate_and_consume_state(state)
    if not state_data:
        logger.warning(f"Invalid or expired OAuth state: {state[:10]}...")
        return _build_error_html(
            title="Session Expired",
            message="The authorization session has expired. Please try again.",
        )

    user_id = state_data["user_id"]
    whatsapp_phone = state_data["whatsapp_phone"]

    try:
        # Exchange code for tokens
        result = await token_service.exchange_code_for_tokens(
            code=code,
            user_id=user_id,
        )

        logger.info(f"Successfully linked Google account for user {user_id}")

        # TODO: Send WhatsApp confirmation message here
        # await whatsapp_service.send_message(
        #     phone=whatsapp_phone,
        #     message=f"Your Google account ({result['google_email']}) has been linked successfully!"
        # )

        return _build_success_html(
            google_email=result.get("google_email", ""),
            google_name=result.get("google_name", ""),
            whatsapp_phone=whatsapp_phone,
        )

    except Exception as e:
        logger.error(f"OAuth token exchange failed for user {user_id}: {e}")
        return _build_error_html(
            title="Authorization Failed",
            message="Failed to complete authorization. Please try again.",
        )


@router.get("/google/status", response_model=TokenStatusResponse)
async def get_token_status(
    user_id: str = Query(..., description="User ID to check"),
    token_service: TokenService = None,
) -> TokenStatusResponse:
    """Check if a user has linked their Google account.

    Args:
        user_id: The user's unique identifier

    Returns:
        TokenStatusResponse with linked status and account info
    """
    status_data = await token_service.get_token_status(user_id)

    if not status_data:
        return TokenStatusResponse(linked=False)

    return TokenStatusResponse(
        linked=True,
        google_email=status_data.get("google_email"),
        google_name=status_data.get("google_name"),
        scopes=status_data.get("scopes"),
    )


@router.post("/google/revoke", response_model=RevokeResponse)
async def revoke_tokens(
    request: RevokeRequest,
    token_service: TokenService = None,
) -> RevokeResponse:
    """Revoke a user's Google OAuth tokens.

    This will unlink the user's Google account from the application.

    Args:
        request: RevokeRequest with user_id

    Returns:
        RevokeResponse indicating success or failure
    """
    success = await token_service.revoke_tokens(request.user_id)

    if success:
        logger.info(f"Revoked OAuth tokens for user {request.user_id}")
        return RevokeResponse(
            success=True,
            message="Google account has been unlinked successfully",
        )
    else:
        return RevokeResponse(
            success=False,
            message="No linked Google account found",
        )


# =============================================================================
# HTML Response Helpers
# =============================================================================


def _build_success_html(
    google_email: str,
    google_name: str,
    whatsapp_phone: str,
) -> HTMLResponse:
    """Build success HTML page with WhatsApp deep link."""
    # WhatsApp deep link (opens WhatsApp app)
    whatsapp_link = f"https://wa.me/{whatsapp_phone.lstrip('+')}"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Authorization Successful</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 600px;
                margin: 50px auto;
                padding: 20px;
                text-align: center;
                background: #f5f5f5;
            }}
            .card {{
                background: white;
                border-radius: 12px;
                padding: 40px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .success-icon {{
                font-size: 64px;
                margin-bottom: 20px;
            }}
            h1 {{
                color: #1a73e8;
                margin-bottom: 10px;
            }}
            .email {{
                color: #5f6368;
                font-size: 18px;
                margin-bottom: 30px;
            }}
            .button {{
                display: inline-block;
                background: #25D366;
                color: white;
                padding: 15px 30px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: 600;
                font-size: 16px;
            }}
            .button:hover {{
                background: #128C7E;
            }}
            .info {{
                color: #5f6368;
                font-size: 14px;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <div class="success-icon">✅</div>
            <h1>Google Account Linked!</h1>
            <p class="email">{google_email or 'Your Google account'}</p>
            <p>You can now use Google Calendar and Gmail features through the chatbot.</p>
            <a href="{whatsapp_link}" class="button">Return to WhatsApp</a>
            <p class="info">You can close this window and return to WhatsApp.</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


def _build_error_html(title: str, message: str) -> HTMLResponse:
    """Build error HTML page."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{title}</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 600px;
                margin: 50px auto;
                padding: 20px;
                text-align: center;
                background: #f5f5f5;
            }}
            .card {{
                background: white;
                border-radius: 12px;
                padding: 40px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .error-icon {{
                font-size: 64px;
                margin-bottom: 20px;
            }}
            h1 {{
                color: #d93025;
                margin-bottom: 10px;
            }}
            .message {{
                color: #5f6368;
                font-size: 16px;
                margin-bottom: 30px;
            }}
            .info {{
                color: #5f6368;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <div class="error-icon">❌</div>
            <h1>{title}</h1>
            <p class="message">{message}</p>
            <p class="info">Please close this window and try again from WhatsApp.</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html, status_code=400)
