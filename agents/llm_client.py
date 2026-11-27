"""
Thin, provider-agnostic wrapper around whatever LLM the current request is
configured to use.

Every agent imports ``call_llm`` / ``call_llm_json`` from here instead of
talking to a vendor SDK directly. That gives us one place to:
  - route to the active provider (Gemini / OpenAI / Anthropic / any
    OpenAI-compatible endpoint) based on the per-session settings in
    ``agents.runtime`` — this is what makes "bring your own key" work;
  - enforce JSON-only output when we need structured data;
  - keep retry/backoff for transient rate-limit / server errors in one spot.

The actual network calls live in ``agents.providers``; the credentials and
model choice live in ``agents.runtime``. Nothing here reads an API key at
import time, so the app boots fine with no key configured and only needs one
when a user actually runs the pipeline.
"""

from __future__ import annotations

import json
import re

from agents import providers
from agents.runtime import get_settings


class ProviderError(RuntimeError):
    """A model-provider failure with a plain-language, key-free message that
    is safe to show users (bad key, quota, unknown model, overload)."""


# (markers in the raw vendor error, message for the user). Checked in order.
_FRIENDLY = [
    (("API_KEY_INVALID", "API key not valid", "invalid x-api-key", "Incorrect API key", "401",
      "key was rejected"),
     "The API key was rejected. Check that it is correct and still active."),
    (("RESOURCE_EXHAUSTED", "429", "rate limit", "quota"),
     "This key has hit its rate limit or daily quota. Wait a little, or switch to another model or key."),
    (("NOT_FOUND", "404", "is not found", "does not exist", "model not found"),
     "The model was not found. Check the model name on the Setup page."),
    (("PERMISSION_DENIED", "403", "access denied"),
     "This key does not have access to that model."),
    (("UNAVAILABLE", "503", "high demand", "overloaded", "529"),
     "The model provider is overloaded right now. Try again in a minute."),
]


def friendly_error(raw: str) -> str | None:
    low = (raw or "").lower()
    for markers, message in _FRIENDLY:
        for m in markers:
            # Bare status codes must stand alone ("429", not "line 4291").
            hit = re.search(rf"\b{m}\b", low) if m.isdigit() else m.lower() in low
            if hit:
                return message
    return None


def _scrub(text: str, *secrets: str) -> str:
    """Remove any API-key values from a string before it can be shown or
    logged. Defence in depth — providers already avoid echoing keys, but a
    stray vendor error could include one."""
    for s in secrets:
        if s:
            text = text.replace(s, "***")
    return text


def call_llm(
    prompt: str,
    system_instruction: str | None = None,
    model: str | None = None,
    json_mode: bool = False,
    max_retries: int = 4,
) -> str:
    """
    Send a prompt to the active provider and return the text response.

    Args:
        prompt: the user-facing prompt/content.
        system_instruction: optional system-level instruction (role, rules).
        model: optional per-call model override. Defaults to the session's
            configured model. Kept for callers that want a cheaper/faster
            model for a specific step.
        json_mode: if True, ask the model for raw JSON (no markdown fences).
        max_retries: transient-failure retries with exponential backoff.

    Returns:
        The model's text output. With json_mode=True this is still a JSON
        *string* the caller must json.loads() — kept explicit so callers
        handle parse errors deliberately (see call_llm_json).
    """
    settings = get_settings()
    if model:
        from dataclasses import replace
        settings = replace(settings, chat_model=model)

    try:
        return providers.chat(settings, prompt, system_instruction, json_mode, max_retries=max_retries)
    except Exception as e:  # noqa: BLE001
        raw = _scrub(str(e), settings.chat_api_key, settings.embed_api_key)
        message = friendly_error(raw)
        if message:
            raise ProviderError(message) from None
        raise RuntimeError(raw[:500]) from None


def call_llm_json(
    prompt: str,
    system_instruction: str | None = None,
    model: str | None = None,
    max_json_retries: int = 2,
) -> dict:
    """
    Call the LLM and parse a JSON object back.

    Even with JSON mode on, a model can occasionally emit malformed JSON
    (an unescaped quote in a free-text field) or, on providers without a
    native JSON mode, wrap the object in ``` fences. We strip fences and, if
    parsing still fails, regenerate (a fresh attempt usually fixes a one-off
    glitch) before giving up.

    Raises json.JSONDecodeError if every attempt fails, so callers can tell a
    transient glitch apart from a persistently bad prompt.
    """
    last_error: json.JSONDecodeError | None = None
    last_raw = ""

    for attempt in range(max_json_retries + 1):
        raw = call_llm(prompt, system_instruction=system_instruction, model=model, json_mode=True)
        last_raw = raw
        try:
            return json.loads(_strip_code_fences(raw))
        except json.JSONDecodeError as e:
            last_error = e
            if attempt < max_json_retries:
                continue

    preview = (last_raw or "")[:500]
    print(f"[llm_client] giving up after {max_json_retries + 1} attempts. "
          f"Last raw output (first 500 chars):\n{preview}")
    raise last_error


def _strip_code_fences(text: str) -> str:
    """Remove a leading/trailing ```json … ``` fence if a provider added one
    despite being asked for raw JSON."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()
