"""
Thin wrapper around the Gemini API.

Every agent in this project imports `call_llm` from here instead of
constructing its own client. That gives us one place to:
  - swap models (flash vs flash-lite) per call
  - add retry/backoff for free-tier rate limits
  - enforce JSON-only output when we need structured data
  - log/debug every prompt sent during development

Nothing else in the codebase should import google.genai directly.
"""

import os
import time
import json
import httpx
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY")
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

if not API_KEY or API_KEY == "your-key-here":
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Copy .env.example to .env and add your "
        "free key from https://aistudio.google.com/apikey"
    )

_client = genai.Client(api_key=API_KEY)


def call_llm(
    prompt: str,
    system_instruction: str | None = None,
    model: str = DEFAULT_MODEL,
    json_mode: bool = False,
    max_retries: int = 4,
) -> str:
    """
    Send a prompt to Gemini and return the text response.

    Args:
        prompt: the user-facing prompt/content.
        system_instruction: optional system-level instruction (role, rules).
        model: which Gemini model to use. Defaults to GEMINI_MODEL from .env.
        json_mode: if True, asks the model to return raw JSON (no markdown
                   fences) — use this for every agent that returns structured
                   data, which is most of them.
        max_retries: retries transient failures (429 rate limits, 503
                     server overload, 500/504 errors) with exponential
                     backoff (1s, 2s, 4s, 8s) instead of crashing. These
                     happen periodically on the free tier, especially
                     during peak load — they're not bugs in this code.

    Returns:
        The model's text output as a string. If json_mode=True, this is a
        JSON string you still need to json.loads() yourself — kept explicit
        on purpose so callers handle parse errors deliberately.
    """
    config_kwargs = {}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    last_error = None
    for attempt in range(max_retries):
        try:
            response = _client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            return response.text
        except Exception as e:
            last_error = e
            error_str = str(e)

            # Network-level transient failures: the request never got a
            # response at all (timeout, connection drop, DNS hiccup).
            # These have no HTTP status code to pattern-match on, so we
            # check the exception type directly. A timeout is never "your
            # request was malformed" — it's always worth retrying.
            is_network_transient = isinstance(
                e, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError, httpx.NetworkError)
            )

            # Retry on transient failures: 429 (rate limit), 503
            # (server overloaded/unavailable), 500 (internal error),
            # 504 (gateway timeout). These are Google's infrastructure
            # having a temporary issue, not a problem with our request,
            # so retrying with backoff is the right response.
            #
            # Deliberately NOT retrying on 400 (bad request) or 401/403
            # (auth) — those are real errors that won't fix themselves,
            # and retrying them just delays a failure that's going to
            # happen anyway.
            is_transient = is_network_transient or any(
                code in error_str
                for code in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL", "504"]
            )
            if is_transient and attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(f"[llm_client] transient error ({error_str[:80]}...), retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise

    raise last_error


def call_llm_json(
    prompt: str,
    system_instruction: str | None = None,
    model: str = DEFAULT_MODEL,
    max_json_retries: int = 2,
) -> dict:
    """
    Convenience wrapper for the common case: call the LLM and parse JSON back.

    Occasionally a model produces malformed JSON even with
    response_mime_type="application/json" set — usually because a
    free-form text field (like a `reasoning` array) contains an
    unescaped quote or control character that breaks JSON structure.
    This is typically a one-off generation glitch, not a systematic
    problem, so retrying the generation itself (not just re-parsing
    the same broken text) is the right response — a fresh attempt
    usually produces valid JSON.

    Raises json.JSONDecodeError if every attempt fails — callers
    should catch this rather than have it fail silently, since it can
    also indicate a genuine, persistent problem with the prompt.
    """
    last_error = None
    last_raw = None

    for attempt in range(max_json_retries + 1):
        raw = call_llm(prompt, system_instruction=system_instruction, model=model, json_mode=True)
        last_raw = raw
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            last_error = e
            if attempt < max_json_retries:
                print(f"[llm_client] model returned malformed JSON ({e}), retrying generation...")
                continue

    # Every attempt failed — print a bounded preview of the last raw
    # output so the failure is debuggable rather than just "bad JSON",
    # without dumping a potentially huge response to the console.
    preview = (last_raw or "")[:500]
    print(f"[llm_client] giving up after {max_json_retries + 1} attempts. Last raw output (first 500 chars):\n{preview}")
    raise last_error
