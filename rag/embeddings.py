"""
embeddings.py — turns code chunks into vectors for RAG retrieval.

Kept separate from the chat path (agents/llm_client.py) because embeddings
are a different job with a different model. Which backend runs is decided per
session by the user's settings (see agents.runtime.LLMSettings.resolved_embedding):

  * ``local``  — a key-free ONNX model (all-MiniLM-L6-v2). The default when
                 the chat provider has no embedding endpoint we use (Groq,
                 Anthropic, OpenRouter, …), so indexing never burns the user's
                 chat quota or needs a second key.
  * ``gemini`` — gemini-embedding-001, when the user brought a Gemini key.
  * ``openai`` — text-embedding-3-small, when the user brought an OpenAI key.

No API key is read at import time; the active one comes from the request
context when these functions are called.
"""

from __future__ import annotations

from agents import providers
from agents.runtime import get_settings


def _resolution():
    return get_settings().resolved_embedding()


def embed_text(text: str) -> list[float]:
    """Embed a single string, returns its vector."""
    return providers.embed(_resolution(), [text])[0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple strings in one call where the backend supports it."""
    if not texts:
        return []
    return providers.embed(_resolution(), texts)


if __name__ == "__main__":
    # Quick manual test — run with: python -m rag.embeddings
    # Uses whatever provider your environment is configured for (see
    # agents.runtime._settings_from_env), defaulting to local embeddings when
    # the chat provider has none.
    vec = embed_text("def authenticate(username, password):")
    print(f"Embedding dimension: {len(vec)}")
    print(f"First 5 values: {vec[:5]}")
