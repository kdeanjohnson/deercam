import os

from ..config import settings
from .base import AntlerCheck, BuckAge, DeerAnalysis, ImagePart, MatchDecision, NameIdeas, Rack, VisionProvider, WildlifeCheck

# label: shown in setup. key: the Settings field holding the API key. model: default for the buck work
# (antler double-check, matching). fast: default for the first look at every crop and name ideas, where
# volume is high and a cheaper model does fine. None when there's no safe default to suggest (the setup
# page lists the provider's models instead).
PROVIDERS = {
    "grok": {"label": "Grok (xAI)", "base_url": "https://api.x.ai/v1", "key": "xai_api_key",
             "model": os.environ.get("GROK_MODEL") or "grok-4.7",
             "fast": os.environ.get("GROK_FAST_MODEL") or "grok-4.3"},
    "claude": {"label": "Claude (Anthropic)", "base_url": None, "key": "anthropic_api_key",
               "model": os.environ.get("CLAUDE_MODEL") or "claude-opus-5-5",
               "fast": os.environ.get("CLAUDE_FAST_MODEL") or "claude-haiku-5-5"},
    "openai": {"label": "OpenAI", "base_url": "https://api.openai.com/v1", "key": "openai_api_key",
               "model": None, "fast": None},
    "gemini": {"label": "Google Gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "key": "gemini_api_key", "model": None, "fast": None},
    "custom": {"label": "Other (OpenAI-compatible)", "base_url": None, "key": "custom_api_key",
               "model": None, "fast": None},
}


class SetupError(RuntimeError):
    """The vision provider isn't configured well enough to call."""


def get_provider(name: str | None = None, fast: bool = False, api_key: str | None = None,
                 model: str | None = None, base_url: str | None = None) -> VisionProvider:
    """fast=True picks the cheaper model, for the first look at every crop and quick calls from the UI.
    The keyword overrides let the setup page try values before saving them."""
    name = (name or settings.vision_provider).lower()
    spec = PROVIDERS.get(name)
    if spec is None:
        raise SetupError(f"Unknown vision provider {name!r}; use one of {', '.join(PROVIDERS)}")
    key = api_key or getattr(settings, spec["key"])
    if not key and name != "custom":  # local OpenAI-compatible servers often don't need one
        raise SetupError(f"No API key for {spec['label']}; add one on the Settings page")
    if model is None:
        same = name == settings.vision_provider
        main = (same and settings.vision_model) or spec["model"]
        model = ((same and settings.vision_fast_model) or spec["fast"] or main) if fast else main
    if not model:
        raise SetupError(f"Pick a model for {spec['label']} on the Settings page")
    if name == "claude":
        from .claude import ClaudeProvider
        return ClaudeProvider(key, model)
    from .openai_compat import OpenAICompatProvider
    url = base_url or spec["base_url"] or settings.custom_base_url
    if not url:
        raise SetupError("Set the server URL for the OpenAI-compatible provider")
    return OpenAICompatProvider(name, url, key or "none", model)


def list_models(name: str, api_key: str | None = None, base_url: str | None = None) -> list[str]:
    return get_provider(name, api_key=api_key, base_url=base_url, model="-").models()


__all__ = ["AntlerCheck", "DeerAnalysis", "ImagePart", "MatchDecision", "NameIdeas", "PROVIDERS", "Rack",
           "SetupError", "VisionProvider", "WildlifeCheck", "get_provider", "list_models"]
