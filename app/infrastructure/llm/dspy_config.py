import dspy

from app.core.config import settings

_lm: dspy.LM | None = None
_configured: bool = False


def configure_dspy() -> dspy.LM:
    """Configure DSPy with the Gemini LM. Must be called from main thread at startup."""
    global _lm, _configured
    if _configured:
        return _lm  # type: ignore

    _lm = dspy.LM(
        f"gemini/{settings.llm_model}",
        api_key=settings.google_api_key,
    )
    dspy.settings.configure(lm=_lm)
    _configured = True
    return _lm


def get_lm() -> dspy.LM:
    """Get the configured LM. Must be called after configure_dspy() has been called."""
    if _lm is None:
        raise RuntimeError(
            "DSPy not configured. Call configure_dspy() from the main thread at startup."
        )
    return _lm
