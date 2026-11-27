"""
providers.py — the one place that actually talks to an LLM vendor.

``llm_client.call_llm`` and ``rag.embeddings`` both route through here, so
adding or swapping a provider is a change to this file alone. Four chat
backends are supported:

  * ``gemini``            — Google Gemini via the google-genai SDK.
  * ``openai``            — OpenAI's Chat Completions API.
  * ``openai_compatible`` — anything that speaks the same API at a custom
                            base URL (Groq, OpenRouter, Together, Mistral,
                            a local Ollama/llama.cpp server, …).
  * ``anthropic``         — Anthropic's Messages API.

and three embedding backends (``gemini``, ``openai``, and a key-free
``local`` ONNX model). HTTP is done with httpx (already a dependency) rather
than pulling in each vendor's SDK, which keeps the install small and the
attack surface smaller.

Design rules enforced here:
  * API keys are used, never logged. Error text surfaced to the UI is
    scrubbed of anything key-shaped by the caller.
  * Transient infrastructure failures (429/500/503/504, network drops) are
    retried with exponential backoff; real errors (400/401/403) are not.
"""

from __future__ import annotations

import time

import httpx

from agents.runtime import EmbedResolution, LLMSettings

# LLM calls can legitimately take a while (large prompts, slow free tiers).
_HTTP_TIMEOUT = httpx.Timeout(120.0, connect=15.0)

# Anthropic requires an explicit output cap, and the Patch Agent may rewrite a
# whole file, so keep it generous.
_ANTHROPIC_MAX_TOKENS = 8192


class LLMError(RuntimeError):
    """Raised for a non-retryable provider error, with vendor detail attached
    so the caller can decide how much to show the user."""


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------
def chat(
    settings: LLMSettings,
    prompt: str,
    system_instruction: str | None,
    json_mode: bool,
    max_retries: int = 4,
) -> str:
    """Send one prompt, return the model's text. Retries transient failures."""
    provider = settings.chat_provider
    if provider == "gemini":
        call = lambda: _gemini_chat(settings, prompt, system_instruction, json_mode)
    elif provider in ("openai", "openai_compatible"):
        call = lambda: _openai_chat(settings, prompt, system_instruction, json_mode)
    elif provider == "anthropic":
        call = lambda: _anthropic_chat(settings, prompt, system_instruction, json_mode)
    else:
        raise LLMError(f"Unknown chat provider: {provider!r}")

    return _with_retries(call, max_retries)


def embed(resolution: EmbedResolution, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with the resolved embedding backend."""
    if resolution.provider == "local":
        return _local_embed(texts)
    if resolution.provider == "sfr":
        return _sfr_embed(texts)
    if resolution.provider == "gemini":
        return _gemini_embed(resolution, texts)
    if resolution.provider == "openai":
        return _openai_embed(resolution, texts)
    raise LLMError(f"Unknown embedding provider: {resolution.provider!r}")


# --------------------------------------------------------------------------
# Retry wrapper
# --------------------------------------------------------------------------
def _with_retries(call, max_retries: int) -> str:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return call()
        except LLMError:
            raise  # non-retryable by construction
        except Exception as e:  # noqa: BLE001
            last_error = e
            if _is_transient(e) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s, 8s
                continue
            raise
    raise last_error  # pragma: no cover


def _is_transient(e: Exception) -> bool:
    if isinstance(e, (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError, httpx.RemoteProtocolError)):
        return True
    if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code in (429, 500, 502, 503, 504)
    text = str(e)
    return any(code in text for code in ("429", "RESOURCE_EXHAUSTED", "500", "502", "503", "504", "UNAVAILABLE", "INTERNAL"))


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------
_gemini_clients: dict[str, object] = {}


def _gemini_client(api_key: str):
    client = _gemini_clients.get(api_key)
    if client is None:
        from google import genai
        client = genai.Client(api_key=api_key)
        _gemini_clients[api_key] = client
    return client


def _gemini_chat(settings: LLMSettings, prompt: str, system_instruction: str | None, json_mode: bool) -> str:
    from google.genai import types

    config_kwargs: dict = {}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"
    config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    response = _gemini_client(settings.chat_api_key).models.generate_content(
        model=settings.chat_model,
        contents=prompt,
        config=config,
    )
    return response.text or ""


def _gemini_embed(resolution: EmbedResolution, texts: list[str]) -> list[list[float]]:
    client = _gemini_client(resolution.api_key)
    try:
        result = client.models.embed_content(model=resolution.model, contents=texts)
        return [e.values for e in result.embeddings]
    except Exception:
        # Some SDK/model combinations reject batched input; fall back to
        # one-at-a-time rather than failing the whole index build.
        out = []
        for t in texts:
            r = client.models.embed_content(model=resolution.model, contents=t)
            out.append(r.embeddings[0].values)
        return out


# --------------------------------------------------------------------------
# OpenAI / OpenAI-compatible
# --------------------------------------------------------------------------
def _openai_base(settings: LLMSettings) -> str:
    if settings.chat_provider == "openai":
        return (settings.chat_base_url or "https://api.openai.com/v1").rstrip("/")
    base = (settings.chat_base_url or "").rstrip("/")
    if not base:
        raise LLMError("An OpenAI-compatible provider needs a Base URL (e.g. https://api.groq.com/openai/v1).")
    return base


def _openai_chat(settings: LLMSettings, prompt: str, system_instruction: str | None, json_mode: bool) -> str:
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    body: dict = {"model": settings.chat_model, "messages": messages}
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    headers = {"Authorization": f"Bearer {settings.chat_api_key}"}
    try:
        data = _post_json(f"{_openai_base(settings)}/chat/completions", headers, body)
    except httpx.HTTPStatusError as e:
        # json_object isn't universal across OpenAI-compatible servers; retry
        # once without it before giving up, since the prompts already ask for
        # JSON explicitly.
        if json_mode and e.response.status_code == 400:
            body.pop("response_format", None)
            try:
                data = _post_json(f"{_openai_base(settings)}/chat/completions", headers, body)
            except httpx.HTTPStatusError as e2:
                raise _as_llm_error(e2)
        else:
            raise _as_llm_error(e)
    return data["choices"][0]["message"]["content"] or ""


def _openai_embed(resolution: EmbedResolution, texts: list[str]) -> list[list[float]]:
    base = (resolution.base_url or "https://api.openai.com/v1").rstrip("/")
    headers = {"Authorization": f"Bearer {resolution.api_key}"}
    data = _post_json(f"{base}/embeddings", headers, {"model": resolution.model, "input": texts})
    return [row["embedding"] for row in data["data"]]


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------
def _anthropic_chat(settings: LLMSettings, prompt: str, system_instruction: str | None, json_mode: bool) -> str:
    system = system_instruction or ""
    if json_mode:
        system = (system + "\n\nRespond with a single valid JSON object only. "
                  "No prose, no explanation, no markdown code fences.").strip()

    body: dict = {
        "model": settings.chat_model,
        "max_tokens": _ANTHROPIC_MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system

    headers = {"x-api-key": settings.chat_api_key, "anthropic-version": "2023-06-01"}
    try:
        data = _post_json("https://api.anthropic.com/v1/messages", headers, body)
    except httpx.HTTPStatusError as e:
        raise _as_llm_error(e)

    parts = [blk.get("text", "") for blk in data.get("content", []) if blk.get("type") == "text"]
    return "".join(parts)


# --------------------------------------------------------------------------
# Local embeddings (no API key, no quota)
# --------------------------------------------------------------------------
_local_ef = None


def _local_embed(texts: list[str]) -> list[list[float]]:
    """ChromaDB's built-in ONNX model (all-MiniLM-L6-v2). The model is
    downloaded and cached on first use, then runs entirely offline — which is
    why it is the default for providers without an embedding endpoint."""
    global _local_ef
    if _local_ef is None:
        from chromadb.utils import embedding_functions
        _local_ef = embedding_functions.DefaultEmbeddingFunction()
    vectors = _local_ef(texts)
    # The ONNX model yields numpy float32 scalars; ChromaDB (and downstream
    # JSON) want plain Python floats, so cast explicitly.
    return [[float(x) for x in v] for v in vectors]


# --------------------------------------------------------------------------
# SFR-Embedding-Code-400M_R (local code-retrieval model, no key)
# --------------------------------------------------------------------------
# The model ships custom architecture code (trust_remote_code), which lives in
# a second repo (Alibaba-NLP/new-impl). Both are pinned to exact commits so the
# code that runs here can never change underneath us.
_SFR_REVISION = "cb950dc80d677c6fdc00f56c8ddd20ca2642c59e"
_SFR_CODE_REVISION = "40ced75c3017eb27626c9d4ea981bde21a2662f4"
# Chunks are ~800 characters (~250 tokens); 1024 tokens leaves headroom while
# keeping CPU memory far below the model's 8192-token maximum.
_SFR_MAX_SEQ = 1024
_sfr_model = None


def _sfr_load():
    global _sfr_model
    if _sfr_model is None:
        from agents.runtime import SFR_MODEL_ID, sfr_available
        if not sfr_available():
            raise LLMError(
                "The SFR code embedding model needs extra packages. Install them with "
                "`pip install -r requirements-sfr.txt`, or choose another embedding option."
            )
        import torch
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            SFR_MODEL_ID,
            revision=_SFR_REVISION,
            trust_remote_code=True,
            device="cpu",
            config_kwargs={"code_revision": _SFR_CODE_REVISION},
            # float32: the checkpoint is bf16, which is far slower on most CPUs.
            model_kwargs={"code_revision": _SFR_CODE_REVISION, "dtype": torch.float32},
        )
        model.max_seq_length = _SFR_MAX_SEQ
        _sfr_rebuild_buffers(model)
        _sfr_model = model
    return _sfr_model


def _sfr_rebuild_buffers(model) -> None:
    """Recompute the model's non-persistent buffers.

    transformers 5.x builds models on the meta device and only materialises
    weights from the checkpoint, so buffers the custom architecture computes
    in __init__ (position_ids, rotary inv_freq and its cos/sin cache) are left
    as uninitialised memory. Without this the first encode() fails with an
    out-of-bounds index, or silently yields garbage vectors. These are the same
    formulas the model's own __init__ uses (Alibaba-NLP/new-impl modeling.py)."""
    import torch
    for m in model.modules():
        if type(m).__name__.endswith("RotaryEmbedding") and hasattr(m, "_set_cos_sin_cache"):
            inv_freq = 1.0 / (m.base ** (torch.arange(0, m.dim, 2, dtype=torch.float32) / m.dim))
            m.register_buffer("inv_freq", inv_freq, persistent=False)
            seq = m.max_position_embeddings
            if getattr(m, "scaling_factor", None):  # NTK-scaled variant
                seq = m.max_position_embeddings * m.scaling_factor
            m._set_cos_sin_cache(seq, inv_freq.device, torch.float32)
        pos = getattr(m, "position_ids", None)
        if isinstance(pos, torch.Tensor) and pos.dim() == 1:
            m.register_buffer("position_ids", torch.arange(pos.size(0)), persistent=False)


def _sfr_embed(texts: list[str]) -> list[list[float]]:
    """Embed with SFR-Embedding-Code (CLS pooling, L2-normalised, 1024-dim).
    The model card uses no query/document prefix, so both are encoded as-is.
    Downloaded (~870 MB) and cached by Hugging Face on first use."""
    vectors = _sfr_load().encode(texts, batch_size=8, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
    return [[float(x) for x in v] for v in vectors]


# --------------------------------------------------------------------------
# HTTP helper
# --------------------------------------------------------------------------
def _post_json(url: str, headers: dict, body: dict) -> dict:
    headers = {**headers, "Content-Type": "application/json"}
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        resp = client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        return resp.json()


def _as_llm_error(e: httpx.HTTPStatusError) -> LLMError:
    """Turn a 4xx into a non-retryable LLMError with a short, key-free reason
    pulled from the vendor's error body when possible."""
    status = e.response.status_code
    detail = ""
    try:
        payload = e.response.json()
        detail = (payload.get("error", {}) or {}).get("message") or payload.get("message") or ""
    except Exception:  # noqa: BLE001
        detail = ""
    hint = {
        400: "the request was rejected (check the model name)",
        401: "the API key was rejected (check it is correct and active)",
        403: "access denied for this key or model",
        404: "model not found for this provider",
    }.get(status, f"HTTP {status}")
    return LLMError(f"Provider error: {hint}. {detail}".strip())
