"""Repository for OAuth database operations."""

import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database.models import GoogleOAuthTokenORM, OAuthStateORM


class OAuthRepository:
    """Repository for OAuth token and state database operations.

    This class handles all database interactions for OAuth tokens
    and state management. It does NOT handle encryption - that's
    the responsibility of the service layer.
    """

    def __init__(self, db: AsyncSession):
        """Initialize the repository.

        Args:
            db: SQLAlchemy async session
        """
        self.db = db

    # =========================================================================
    # OAuth State Operations (CSRF protection)
    # =========================================================================

    async def create_state(
        self,
        user_id: str,
        whatsapp_phone: str,
        expiry_minutes: int = 10,
    ) -> OAuthStateORM:
        """Create a new OAuth state for CSRF protection.

        Args:
            user_id: The user ID initiating OAuth
            whatsapp_phone: User's WhatsApp phone number
            expiry_minutes: Minutes until state expires (default: 10)

        Returns:
            The created OAuthStateORM record
        """
        state = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)

        oauth_state = OAuthStateORM(
            state=state,
            user_id=user_id,
            whatsapp_phone=whatsapp_phone,
            created_at=now,
            expires_at=now + timedelta(minutes=expiry_minutes),
        )

        self.db.add(oauth_state)
        await self.db.flush()
        return oauth_state

    async def get_state(self, state: str) -> OAuthStateORM | None:
        """Get an OAuth state record by state value.

        Args:
            state: The state token to look up

        Returns:
            The OAuthStateORM if found and not expired, None otherwise
        """
        result = await self.db.execute(
            select(OAuthStateORM).where(
                OAuthStateORM.state == state,
                OAuthStateORM.expires_at > datetime.now(timezone.utc),
            )
        )
        return result.scalar_one_or_none()

    async def delete_state(self, state: str) -> bool:
        """Delete an OAuth state record (single-use).

        Args:
            state: The state token to delete

        Returns:
            True if deleted, False if not found
        """
        result = await self.db.execute(
            delete(OAuthStateORM).where(OAuthStateORM.state == state)
        )
        return result.rowcount > 0

    async def cleanup_expired_states(self) -> int:
        """Delete all expired OAuth states.

        Returns:
            Number of states deleted
        """
        result = await self.db.execute(
            delete(OAuthStateORM).where(
                OAuthStateORM.expires_at <= datetime.now(timezone.utc)
            )
        )
        return result.rowcount

    # =========================================================================
    # OAuth Token Operations
    # =========================================================================

    async def get_token_by_user_id(self, user_id: str) -> GoogleOAuthTokenORM | None:
        """Get OAuth token by user ID.

        Args:
            user_id: The user ID to look up

        Returns:
            The GoogleOAuthTokenORM if found, None otherwise
        """
        result = await self.db.execute(
            select(GoogleOAuthTokenORM).where(
                GoogleOAuthTokenORM.user_id == user_id,
                GoogleOAuthTokenORM.is_revoked == False,  # noqa: E712
            )
        )
        return result.scalar_one_or_none()

    async def upsert_token(
        self,
        user_id: str,
        access_token_encrypted: str,
        refresh_token_encrypted: str,
        token_expiry: datetime,
        scopes: list[str],
        google_email: str | None = None,
        google_name: str | None = None,
    ) -> GoogleOAuthTokenORM:
        """Create or update OAuth token for a user.

        If a token exists for the user, it will be updated.
        If no token exists, a new one will be created.

        Args:
            user_id: The user ID
            access_token_encrypted: Encrypted access token
            refresh_token_encrypted: Encrypted refresh token
            token_expiry: When the access token expires
            scopes: List of OAuth scopes granted
            google_email: User's Google email (optional)
            google_name: User's Google display name (optional)

        Returns:
            The created or updated GoogleOAuthTokenORM
        """
        existing = await self.db.execute(
            select(GoogleOAuthTokenORM).where(GoogleOAuthTokenORM.user_id == user_id)
        )
        token = existing.scalar_one_or_none()

        now = datetime.now(timezone.utc)

        if token:
            # Update existing token
            token.access_token_encrypted = access_token_encrypted
            token.refresh_token_encrypted = refresh_token_encrypted
            token.token_expiry = token_expiry
            token.scopes = scopes
            token.google_email = google_email
            token.google_name = google_name
            token.updated_at = now
            token.is_revoked = False
            token.revoked_at = None
        else:
            # Create new token
            token = GoogleOAuthTokenORM(
                user_id=user_id,
                access_token_encrypted=access_token_encrypted,
                refresh_token_encrypted=refresh_token_encrypted,
                token_expiry=token_expiry,
                scopes=scopes,
                google_email=google_email,
                google_name=google_name,
                created_at=now,
                updated_at=now,
                is_revoked=False,
            )
            self.db.add(token)

        await self.db.flush()
        return token

    async def update_access_token(
        self,
        user_id: str,
        access_token_encrypted: str,
        token_expiry: datetime,
    ) -> GoogleOAuthTokenORM | None:
        """Update only the access token (after refresh).

        Args:
            user_id: The user ID
            access_token_encrypted: New encrypted access token
            token_expiry: New expiry time

        Returns:
            The updated token, or None if not found
        """
        result = await self.db.execute(
            select(GoogleOAuthTokenORM).where(GoogleOAuthTokenORM.user_id == user_id)
        )
        token = result.scalar_one_or_none()

        if token:
            token.access_token_encrypted = access_token_encrypted
            token.token_expiry = token_expiry
            token.updated_at = datetime.now(timezone.utc)
            await self.db.flush()

        return token

    async def revoke_token(self, user_id: str) -> bool:
        """Mark a token as revoked.

        Args:
            user_id: The user ID whose token to revoke

        Returns:
            True if token was found and revoked, False otherwise
        """
        result = await self.db.execute(
            select(GoogleOAuthTokenORM).where(GoogleOAuthTokenORM.user_id == user_id)
        )
        token = result.scalar_one_or_none()

        if token:
            now = datetime.now(timezone.utc)
            token.is_revoked = True
            token.revoked_at = now
            token.updated_at = now
            await self.db.flush()
            return True

        return False

    async def delete_token(self, user_id: str) -> bool:
        """Permanently delete a token.

        Args:
            user_id: The user ID whose token to delete

        Returns:
            True if deleted, False if not found
        """
        result = await self.db.execute(
            delete(GoogleOAuthTokenORM).where(GoogleOAuthTokenORM.user_id == user_id)
        )
        return result.rowcount > 0

    async def has_valid_token(self, user_id: str) -> bool:
        """Check if a user has a valid (non-revoked) token.

        Args:
            user_id: The user ID to check

        Returns:
            True if user has a valid token
        """
        token = await self.get_token_by_user_id(user_id)
        return token is not None
