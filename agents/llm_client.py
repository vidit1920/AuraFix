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

from agents import providers
from agents.runtime import get_settings


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
        raise RuntimeError(_scrub(str(e), settings.chat_api_key, settings.embed_api_key)) from None


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
