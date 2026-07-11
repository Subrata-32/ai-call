import os


def normalize_provider(provider: str | None, fallback: str = "openai") -> str:
    value = (provider or "").strip().lower()
    if value in {"openai", "deepgram"}:
        return value
    return fallback


def resolve_speech_provider(config_value: str | None, env_value: str | None, fallback: str = "deepgram") -> str:
    requested = normalize_provider(config_value or env_value, fallback)
    if requested == "deepgram":
        return "deepgram" if os.getenv("DEEPGRAM_API_KEY", "").strip() else "openai"
    if requested == "openai":
        return "openai" if os.getenv("OPENAI_API_KEY", "").strip() else fallback
    return fallback
