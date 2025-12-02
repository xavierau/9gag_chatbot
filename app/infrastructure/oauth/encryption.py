"""Token encryption service using Fernet (AES-128-CBC).

This module provides symmetric encryption for OAuth tokens at rest.
The encryption key must be provided via the GOOGLE_OAUTH_ENCRYPTION_KEY
environment variable.

To generate a new key:
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""

from cryptography.fernet import Fernet, InvalidToken


class TokenEncryption:
    """Handles encryption/decryption of OAuth tokens using Fernet.

    Fernet uses AES-128-CBC with HMAC for authentication, providing
    both confidentiality and integrity of the encrypted data.

    Usage:
        encryption = TokenEncryption(key="your-fernet-key")
        encrypted = encryption.encrypt("my-secret-token")
        decrypted = encryption.decrypt(encrypted)
    """

    def __init__(self, key: str):
        """Initialize the encryption service.

        Args:
            key: A Fernet-compatible key (base64-encoded 32 bytes).
                 Generate with: Fernet.generate_key()
        """
        self.fernet = Fernet(key.encode())

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a plaintext string.

        Args:
            plaintext: The string to encrypt (e.g., OAuth access_token)

        Returns:
            Base64-encoded encrypted string
        """
        return self.fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt an encrypted string.

        Args:
            ciphertext: Base64-encoded encrypted string

        Returns:
            Decrypted plaintext string

        Raises:
            InvalidToken: If the ciphertext is invalid or tampered with
        """
        return self.fernet.decrypt(ciphertext.encode()).decode()

    def is_valid_ciphertext(self, ciphertext: str) -> bool:
        """Check if a ciphertext can be decrypted.

        Args:
            ciphertext: The encrypted string to validate

        Returns:
            True if the ciphertext is valid, False otherwise
        """
        try:
            self.decrypt(ciphertext)
            return True
        except (InvalidToken, Exception):
            return False
