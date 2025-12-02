"""Custom exceptions for OAuth operations."""


class OAuthError(Exception):
    """Base exception for OAuth-related errors."""

    pass


class NoTokenError(OAuthError):
    """Raised when no OAuth token exists for a user."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        super().__init__(f"No Google OAuth token found for user: {user_id}")


class TokenExpiredError(OAuthError):
    """Raised when the OAuth token is expired and cannot be refreshed."""

    def __init__(self, user_id: str, reason: str = "Token expired"):
        self.user_id = user_id
        self.reason = reason
        super().__init__(f"OAuth token expired for user {user_id}: {reason}")


class TokenRevokedError(OAuthError):
    """Raised when the OAuth token has been revoked by the user."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        super().__init__(f"OAuth token revoked for user: {user_id}")


class InvalidStateError(OAuthError):
    """Raised when the OAuth state parameter is invalid or expired."""

    def __init__(self, state: str):
        self.state = state
        super().__init__(f"Invalid or expired OAuth state: {state}")


class TokenRefreshError(OAuthError):
    """Raised when token refresh fails."""

    def __init__(self, user_id: str, reason: str):
        self.user_id = user_id
        self.reason = reason
        super().__init__(f"Failed to refresh token for user {user_id}: {reason}")
