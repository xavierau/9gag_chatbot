"""DSPy configuration with Langfuse observability.

This module configures DSPy with the Gemini LM and integrates Langfuse
for LLM observability via OpenTelemetry instrumentation.
"""

import base64
import logging
import os

import dspy

from app.core.config import settings

logger = logging.getLogger(__name__)

_lm: dspy.LM | None = None
_configured: bool = False
_langfuse_configured: bool = False


def _configure_langfuse() -> None:
    """Configure Langfuse observability via OpenTelemetry.

    Sets up the OpenTelemetry exporter to send traces to Langfuse
    and instruments DSPy for automatic tracing of all LLM calls.
    """
    global _langfuse_configured

    if _langfuse_configured:
        return

    # Check if Langfuse is enabled and configured
    if not settings.langfuse_enabled:
        logger.info("Langfuse observability is disabled")
        return

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning(
            "Langfuse keys not configured. Set LANGFUSE_PUBLIC_KEY and "
            "LANGFUSE_SECRET_KEY to enable observability."
        )
        return

    try:
        # Configure OTEL environment variables for Langfuse
        langfuse_auth = base64.b64encode(
            f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()
        ).decode()

        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = (
            f"{settings.langfuse_host}/api/public/otel"
        )
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"

        # Initialize OpenTelemetry tracer provider
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # Create and configure tracer provider
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(tracer_provider)

        # Instrument DSPy for automatic tracing
        from openinference.instrumentation.dspy import DSPyInstrumentor

        DSPyInstrumentor().instrument()

        _langfuse_configured = True
        logger.info(
            f"Langfuse observability configured successfully "
            f"(host: {settings.langfuse_host})"
        )

    except ImportError as e:
        logger.warning(
            f"Langfuse dependencies not installed: {e}. "
            "Install with: pip install langfuse openinference-instrumentation-dspy "
            "opentelemetry-sdk opentelemetry-exporter-otlp"
        )
    except Exception as e:
        logger.error(f"Failed to configure Langfuse: {e}", exc_info=True)


def get_tracer():
    """Get the OpenTelemetry tracer for custom spans.

    Returns:
        The tracer instance if Langfuse is configured, None otherwise.
    """
    if not _langfuse_configured:
        return None

    try:
        from opentelemetry import trace

        return trace.get_tracer("chatbot.agent")
    except Exception:
        return None


def configure_dspy() -> dspy.LM:
    """Configure DSPy with the Gemini LM. Must be called from main thread at startup.

    Also configures Langfuse observability if enabled and properly configured.
    """
    global _lm, _configured
    if _configured:
        return _lm  # type: ignore

    # Configure Langfuse first (before DSPy to ensure instrumentation is ready)
    _configure_langfuse()

    _lm = dspy.LM(
        f"gemini/{settings.llm_model}",
        api_key=settings.google_api_key,
    )
    dspy.settings.configure(lm=_lm)
    _configured = True

    logger.info(f"DSPy configured with model: gemini/{settings.llm_model}")
    return _lm


def get_lm() -> dspy.LM:
    """Get the configured LM. Must be called after configure_dspy() has been called."""
    if _lm is None:
        raise RuntimeError(
            "DSPy not configured. Call configure_dspy() from the main thread at startup."
        )
    return _lm
