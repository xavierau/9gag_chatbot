"""Google OAuth token service for token exchange, refresh, and management.

This service handles the complete OAuth token lifecycle:
- Exchanging authorization codes for tokens
- Refreshing expired access tokens
- Storing encrypted tokens in the database
- Revoking tokens
"""

import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.oauth.encryption import TokenEncryption
from app.infrastructure.oauth.exceptions import (
    NoTokenError,
    TokenExpiredError,
    TokenRefreshError,
    TokenRevokedError,
)
from app.infrastructure.oauth.repository import OAuthRepository

logger = logging.getLogger(__name__)

# Google OAuth endpoints
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"

# OAuth scopes for Calendar and Gmail
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]


class GoogleTokenService:
    """Service for managing Google OAuth tokens.

    This service coordinates between the OAuth repository (database)
    and the token encryption service to securely manage OAuth tokens.
    """

    def __init__(
        self,
        db: AsyncSession,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        encryption_key: str,
    ):
        """Initialize the token service.

        Args:
            db: SQLAlchemy async session
            client_id: Google OAuth client ID
            client_secret: Google OAuth client secret
            redirect_uri: OAuth callback URL
            encryption_key: Fernet encryption key for tokens
        """
        self.repository = OAuthRepository(db)
        self.encryption = TokenEncryption(encryption_key)
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

    def build_authorization_url(self, state: str) -> str:
        """Build the Google OAuth authorization URL.

        Args:
            state: CSRF protection state parameter

        Returns:
            Full authorization URL to redirect the user to
        """
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": " ".join(GOOGLE_SCOPES),
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        }
        return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"

    async def exchange_code_for_tokens(
        self,
        code: str,
        user_id: str,
    ) -> dict:
        """Exchange authorization code for access and refresh tokens.

        Args:
            code: Authorization code from Google callback
            user_id: User ID to associate tokens with

        Returns:
            Dict with google_email, google_name, and scopes

        Raises:
            TokenRefreshError: If code exchange fails
        """
        async with httpx.AsyncClient() as client:
            # Exchange code for tokens
            response = await client.post(
                GOOGLE_TOKEN_URL,
                data={
                    "code": code,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "redirect_uri": self.redirect_uri,
                    "grant_type": "authorization_code",
                },
            )

            if response.status_code != 200:
                logger.error(f"Token exchange failed: {response.text}")
                raise TokenRefreshError(user_id, f"Token exchange failed: {response.status_code}")

            token_data = response.json()
            access_token = token_data["access_token"]
            refresh_token = token_data.get("refresh_token")
            expires_in = token_data.get("expires_in", 3600)
            scopes = token_data.get("scope", "").split()

            if not refresh_token:
                raise TokenRefreshError(user_id, "No refresh token returned")

            # Get user info
            userinfo_response = await client.get(
                GOOGLE_USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}"},
            )

            google_email = None
            google_name = None
            if userinfo_response.status_code == 200:
                userinfo = userinfo_response.json()
                google_email = userinfo.get("email")
                google_name = userinfo.get("name")

            # Calculate expiry
            token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

            # Encrypt tokens
            access_token_encrypted = self.encryption.encrypt(access_token)
            refresh_token_encrypted = self.encryption.encrypt(refresh_token)

            # Store in database
            await self.repository.upsert_token(
                user_id=user_id,
                access_token_encrypted=access_token_encrypted,
                refresh_token_encrypted=refresh_token_encrypted,
                token_expiry=token_expiry,
                scopes=scopes,
                google_email=google_email,
                google_name=google_name,
            )

            logger.info(f"Successfully stored OAuth tokens for user {user_id}")

            return {
                "google_email": google_email,
                "google_name": google_name,
                "scopes": scopes,
            }

    async def get_valid_access_token(self, user_id: str) -> str:
        """Get a valid access token, refreshing if necessary.

        This is the main method for obtaining an access token.
        It will automatically refresh expired tokens.

        Args:
            user_id: User ID to get token for

        Returns:
            Valid (decrypted) access token

        Raises:
            NoTokenError: If user has no linked Google account
            TokenRevokedError: If token was revoked
            TokenExpiredError: If token is expired and refresh failed
        """
        token = await self.repository.get_token_by_user_id(user_id)

        if not token:
            raise NoTokenError(user_id)

        if token.is_revoked:
            raise TokenRevokedError(user_id)

        # Check if token needs refresh (expires within 5 minutes)
        now = datetime.now(timezone.utc)
        refresh_threshold = now + timedelta(minutes=5)

        if token.token_expiry <= refresh_threshold:
            # Token needs refresh
            try:
                await self._refresh_access_token(user_id, token)
                # Re-fetch token after refresh
                token = await self.repository.get_token_by_user_id(user_id)
                if not token:
                    raise TokenExpiredError(user_id, "Token not found after refresh")
            except TokenRefreshError as e:
                # If refresh fails with invalid_grant, token is revoked
                if "invalid_grant" in str(e):
                    await self.repository.revoke_token(user_id)
                    raise TokenRevokedError(user_id)
                raise TokenExpiredError(user_id, str(e))

        # Decrypt and return access token
        return self.encryption.decrypt(token.access_token_encrypted)

    async def _refresh_access_token(
        self,
        user_id: str,
        token,
    ) -> None:
        """Refresh an expired access token using the refresh token.

        Args:
            user_id: User ID
            token: GoogleOAuthTokenORM record

        Raises:
            TokenRefreshError: If refresh fails
        """
        refresh_token = self.encryption.decrypt(token.refresh_token_encrypted)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                GOOGLE_TOKEN_URL,
                data={
                    "refresh_token": refresh_token,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "grant_type": "refresh_token",
                },
            )

            if response.status_code != 200:
                error_data = response.json() if response.content else {}
                error = error_data.get("error", "unknown")
                logger.error(f"Token refresh failed for user {user_id}: {error}")
                raise TokenRefreshError(user_id, error)

            token_data = response.json()
            new_access_token = token_data["access_token"]
            expires_in = token_data.get("expires_in", 3600)
            token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

            # Encrypt and store new access token
            access_token_encrypted = self.encryption.encrypt(new_access_token)
            await self.repository.update_access_token(
                user_id=user_id,
                access_token_encrypted=access_token_encrypted,
                token_expiry=token_expiry,
            )

            logger.info(f"Successfully refreshed access token for user {user_id}")

    async def revoke_tokens(self, user_id: str) -> bool:
        """Revoke user's Google OAuth tokens.

        This will:
        1. Call Google's revoke endpoint
        2. Mark the token as revoked in the database

        Args:
            user_id: User ID to revoke tokens for

        Returns:
            True if revocation succeeded, False if no token found
        """
        token = await self.repository.get_token_by_user_id(user_id)
        if not token:
            return False

        # Try to revoke with Google
        try:
            access_token = self.encryption.decrypt(token.access_token_encrypted)
            async with httpx.AsyncClient() as client:
                await client.post(
                    GOOGLE_REVOKE_URL,
                    params={"token": access_token},
                )
        except Exception as e:
            # Log but don't fail - we still want to mark as revoked locally
            logger.warning(f"Failed to revoke token with Google for user {user_id}: {e}")

        # Mark as revoked in database
        await self.repository.revoke_token(user_id)
        logger.info(f"Revoked OAuth tokens for user {user_id}")
        return True

    async def get_token_status(self, user_id: str) -> dict | None:
        """Get the status of a user's OAuth token.

        Args:
            user_id: User ID to check

        Returns:
            Dict with token status info, or None if no token
        """
        token = await self.repository.get_token_by_user_id(user_id)
        if not token:
            return None

        return {
            "linked": True,
            "google_email": token.google_email,
            "google_name": token.google_name,
            "scopes": token.scopes,
            "is_revoked": token.is_revoked,
            "created_at": token.created_at.isoformat(),
            "updated_at": token.updated_at.isoformat(),
        }

    async def create_oauth_state(
        self,
        user_id: str,
        whatsapp_phone: str,
    ) -> str:
        """Create an OAuth state for CSRF protection.

        Args:
            user_id: User ID initiating OAuth
            whatsapp_phone: User's WhatsApp phone number

        Returns:
            The state token string
        """
        state_record = await self.repository.create_state(
            user_id=user_id,
            whatsapp_phone=whatsapp_phone,
        )
        return state_record.state

    async def validate_and_consume_state(self, state: str) -> dict | None:
        """Validate and consume an OAuth state (single-use).

        Args:
            state: The state token to validate

        Returns:
            Dict with user_id and whatsapp_phone if valid, None otherwise
        """
        state_record = await self.repository.get_state(state)
        if not state_record:
            return None

        result = {
            "user_id": state_record.user_id,
            "whatsapp_phone": state_record.whatsapp_phone,
        }

        # Delete state (single-use)
        await self.repository.delete_state(state)

        return result
