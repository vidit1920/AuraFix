"""
embeddings.py — wraps Gemini's embedding API.

Kept separate from llm_client.py because embeddings use a different
underlying model (an embedding model, not a generative one) and a
different API method (embed_content, not generate_content). Mixing
the two into one client would make llm_client.py do two unrelated jobs.
"""

import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY")
EMBEDDING_MODEL = "gemini-embedding-001"

if not API_KEY or API_KEY == "your-key-here":
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Copy .env.example to .env and add your "
        "free key from https://aistudio.google.com/apikey"
    )

_client = genai.Client(api_key=API_KEY)


def embed_text(text: str) -> list[float]:
    """Embeds a single string, returns its vector."""
    result = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return result.embeddings[0].values


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embeds multiple strings in one API call where possible.

    Falls back to one-by-one calls if the SDK version in use doesn't
    support batch embedding cleanly — better to be slow than to crash
    the whole indexing run on a single malformed batch request.
    """
    try:
        result = _client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
        )
        return [e.values for e in result.embeddings]
    except Exception:
        return [embed_text(t) for t in texts]


if __name__ == "__main__":
    # Quick manual test — run with: python -m rag.embeddings
    vec = embed_text("def authenticate(username, password):")
    print(f"Embedding dimension: {len(vec)}")
    print(f"First 5 values: {vec[:5]}")
