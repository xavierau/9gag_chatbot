"""WhatsApp messaging service for sending notifications.

This service provides methods for sending WhatsApp messages,
primarily used for OAuth flow notifications.
"""

import logging

import httpx

logger = logging.getLogger(__name__)


class WhatsAppService:
    """Service for sending WhatsApp messages.

    This service wraps the WhatsApp Business API for sending
    text messages to users.
    """

    def __init__(self, api_url: str, api_token: str):
        """Initialize the WhatsApp service.

        Args:
            api_url: Base URL for WhatsApp Business API
            api_token: Bearer token for authentication
        """
        self.api_url = api_url.rstrip("/")
        self.api_token = api_token

    async def send_message(self, phone: str, message: str) -> bool:
        """Send a text message to a WhatsApp user.

        Args:
            phone: Phone number in international format (e.g., +85291234567)
            message: Text message to send

        Returns:
            True if message was sent successfully, False otherwise
        """
        if not self.api_url or not self.api_token:
            logger.warning("WhatsApp service not configured, skipping message send")
            return False

        # Normalize phone number (remove + prefix if present)
        phone_normalized = phone.lstrip("+")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_url}/messages",
                    headers={
                        "Authorization": f"Bearer {self.api_token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "messaging_product": "whatsapp",
                        "recipient_type": "individual",
                        "to": phone_normalized,
                        "type": "text",
                        "text": {"body": message},
                    },
                    timeout=30.0,
                )

                if response.status_code == 200:
                    logger.info(f"Successfully sent WhatsApp message to {phone_normalized}")
                    return True
                else:
                    logger.error(
                        f"Failed to send WhatsApp message: {response.status_code} - {response.text}"
                    )
                    return False

        except httpx.TimeoutException:
            logger.error(f"Timeout sending WhatsApp message to {phone_normalized}")
            return False
        except Exception as e:
            logger.error(f"Error sending WhatsApp message: {e}")
            return False

    async def send_oauth_success_notification(
        self,
        phone: str,
        google_email: str,
    ) -> bool:
        """Send OAuth success notification.

        Args:
            phone: User's WhatsApp phone number
            google_email: Linked Google account email

        Returns:
            True if message was sent successfully
        """
        message = (
            f"Your Google account ({google_email}) has been linked successfully! "
            f"You can now use Google Calendar and Gmail features through the chatbot."
        )
        return await self.send_message(phone, message)

    async def send_oauth_revoked_notification(self, phone: str) -> bool:
        """Send OAuth revoked notification.

        Args:
            phone: User's WhatsApp phone number

        Returns:
            True if message was sent successfully
        """
        message = (
            "Your Google account has been unlinked. "
            "To use Google Calendar and Gmail features again, "
            "please link your account again."
        )
        return await self.send_message(phone, message)
