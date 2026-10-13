"""
runtime.py — per-request LLM credentials and model selection ("bring your
own key").

The whole pipeline used to read a single process-wide ``GEMINI_API_KEY`` at
import time. That is wrong for a hosted, multi-user app: one shared key is a
cost/quota liability and a security problem, and it makes "bring your own
key" impossible.

Instead, the active provider + key + model live in a ``ContextVar``. The
Streamlit UI sets it at the top of each script run from that session's
state; every agent downstream reads it through :func:`get_settings`. A
``ContextVar`` is the right tool here because:

  * Streamlit runs each browser session's script in its own thread, and a
    ``ContextVar`` is isolated per thread/async-context — so two users with
    two different keys never see each other's credentials.
  * LangGraph's ``invoke`` runs the graph synchronously in that same thread,
    so the value set at the top of the run is visible to every node without
    threading it through the graph state (where it would risk being logged
    or serialised).

Nothing is ever written to disk or logged. The key exists only in memory for
the duration of a request.

For the command-line entry points and the offline test-suite (which has no
Streamlit session), :func:`get_settings` falls back to environment
variables, preserving the original local-dev workflow.
"""

from __future__ import annotations

import contextvars
import os
from dataclasses import dataclass

# Providers we know how to talk to. "openai_compatible" covers any endpoint
# that speaks the OpenAI Chat Completions API (Groq, OpenRouter, Together,
# Mistral, a local Ollama/llama.cpp server, …) via a custom base URL.
CHAT_PROVIDERS = ("gemini", "openai", "anthropic", "openai_compatible")
EMBED_PROVIDERS = ("auto", "local", "gemini", "openai")

# A sensible, current default chat model per provider. gemini-2.5-flash is
# being retired (Oct 2026), so the Gemini default points at the maintained
# "latest" flash alias rather than a pinned version that will 404.
DEFAULT_CHAT_MODELS = {
    "gemini": "gemini-flash-latest",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
    "openai_compatible": "llama-3.3-70b-versatile",
}


@dataclass(frozen=True)
class LLMSettings:
    """One session's choice of model + credentials. Immutable so it can be
    read freely across the pipeline without anyone mutating it mid-run."""

    chat_provider: str
    chat_model: str
    chat_api_key: str
    chat_base_url: str | None = None  # only for openai_compatible

    # Embeddings are decoupled from chat so a user can bring, say, a Groq key
    # for reasoning and still index code for free with a local model.
    embed_provider: str = "auto"      # resolved against the chat provider
    embed_model: str = ""
    embed_api_key: str = ""           # defaults to chat_api_key when empty
    embed_base_url: str | None = None

    def resolved_embedding(self) -> "EmbedResolution":
        """Turn the user's (possibly "auto") embedding choice into a concrete
        provider + model + key. Kept here so the UI, the vector store cache
        key, and the embedding call all agree on exactly one answer."""
        provider = self.embed_provider or "auto"

        if provider == "auto":
            if self.chat_provider == "gemini":
                provider = "gemini"
            elif self.chat_provider == "openai":
                provider = "openai"
            else:
                # Groq / Anthropic / OpenRouter etc. have no first-class
                # embedding endpoint we rely on — index locally, no key, no
                # quota. This is also the safest default for an unknown
                # openai_compatible base URL.
                provider = "local"

        if provider == "local":
            return EmbedResolution("local", "all-MiniLM-L6-v2", "", None)

        if provider == "gemini":
            key = self.embed_api_key or (self.chat_api_key if self.chat_provider == "gemini" else "")
            return EmbedResolution("gemini", self.embed_model or "gemini-embedding-001", key, None)

        # openai
        key = self.embed_api_key or (self.chat_api_key if self.chat_provider == "openai" else "")
        base = self.embed_base_url or (self.chat_base_url if self.chat_provider == "openai" else None)
        return EmbedResolution("openai", self.embed_model or "text-embedding-3-small", key, base)


@dataclass(frozen=True)
class EmbedResolution:
    provider: str          # "local" | "gemini" | "openai"
    model: str
    api_key: str
    base_url: str | None


_current: contextvars.ContextVar[LLMSettings | None] = contextvars.ContextVar(
    "aurafix_llm_settings", default=None
)


def set_settings(settings: LLMSettings) -> contextvars.Token:
    """Install this request's settings. Returns a token the caller can use to
    reset to the previous value (not usually needed in Streamlit, where each
    run starts fresh, but useful in tests)."""
    return _current.set(settings)


def reset_settings(token: contextvars.Token) -> None:
    _current.reset(token)


def clear_settings() -> None:
    """Drop any settings bound to the current context. Important in a server
    that reuses worker threads across sessions: the UI calls this at the top of
    every run so a previous session's key can never linger on a reused thread
    and be picked up by a different user."""
    _current.set(None)


def has_settings() -> bool:
    return _current.get() is not None


def get_settings() -> LLMSettings:
    """The active settings, or a fallback built from environment variables for
    CLI / test use. Raises only when neither a session nor an env key exists,
    with a message that tells the user exactly what to do."""
    settings = _current.get()
    if settings is not None:
        return settings

    env = _settings_from_env()
    if env is None:
        raise RuntimeError(
            "No LLM credentials available. In the app, paste an API key in the "
            "sidebar. For the CLI/tests, set GEMINI_API_KEY (or OPENAI_API_KEY / "
            "ANTHROPIC_API_KEY) in your environment or .env file."
        )
    return env


def _settings_from_env() -> LLMSettings | None:
    """Build settings from environment variables, in priority order. Lets the
    CLI demo and the manual ``python -m ...`` module tests keep working with a
    single key in .env, exactly as before BYOK existed."""
    gemini = _clean(os.environ.get("GEMINI_API_KEY"), "your-key-here")
    if gemini:
        return LLMSettings(
            chat_provider="gemini",
            chat_model=os.environ.get("GEMINI_MODEL") or DEFAULT_CHAT_MODELS["gemini"],
            chat_api_key=gemini,
        )

    openai = _clean(os.environ.get("OPENAI_API_KEY"), "")
    if openai:
        return LLMSettings(
            chat_provider="openai",
            chat_model=os.environ.get("OPENAI_MODEL") or DEFAULT_CHAT_MODELS["openai"],
            chat_api_key=openai,
            chat_base_url=os.environ.get("OPENAI_BASE_URL") or None,
        )

    anthropic = _clean(os.environ.get("ANTHROPIC_API_KEY"), "")
    if anthropic:
        return LLMSettings(
            chat_provider="anthropic",
            chat_model=os.environ.get("ANTHROPIC_MODEL") or DEFAULT_CHAT_MODELS["anthropic"],
            chat_api_key=anthropic,
        )

    return None


def _clean(value: str | None, placeholder: str) -> str:
    """Treat empty strings and the shipped placeholders as 'unset'."""
    if not value:
        return ""
    value = value.strip()
    if placeholder and value == placeholder:
        return ""
    return value
