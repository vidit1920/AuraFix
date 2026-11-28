"""
test_providers.py — quota, rate-limit and timeout handling in the LLM layer.

Error texts mirror real Gemini free-tier responses (captured 2026-11-28):
the quota is counted per model *per Google Cloud project*, and the error's
quotaId says whether it is a per-day or per-minute limit. A daily quota must
fail fast with a clear message; a per-minute limit should be waited out; a
timeout must not be retried 4x (that is how a "Contacting the provider"
spinner ran for minutes).
"""

import httpx
import pytest

import agents.providers as providers
from agents.llm_client import friendly_error
from agents.providers import ProviderTimeout, classify_error

DAILY = ("429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota. "
         "* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, "
         "limit: 20, model: gemini-3.5-flash', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'quotaId': "
         "'GenerateRequestsPerDayPerProjectPerModel', 'quotaValue': '20'}]}}")
PER_MINUTE = ("429 RESOURCE_EXHAUSTED. {'error': {'message': 'Quota exceeded for metric: "
              "generate_content_free_tier_requests, limit: 10. Please retry in 37.2s.', 'details': "
              "[{'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel'}, {'retryDelay': '37s'}]}}")
NO_QUOTA = ("429 RESOURCE_EXHAUSTED. {'error': {'message': 'Quota exceeded for metric: "
            "generate_content_free_tier_requests, limit: 0, model: gemini-3.8-flash'}}")


def _http_error(status, body="", headers=None):
    req = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    return httpx.HTTPStatusError("err", request=req,
                                 response=httpx.Response(status, request=req, text=body, headers=headers or {}))


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
def test_classifies_gemini_quota_kinds():
    assert classify_error(RuntimeError(DAILY))[0] == "daily_quota"
    kind, wait = classify_error(RuntimeError(PER_MINUTE))
    assert kind == "per_minute" and 37 < wait < 40
    assert classify_error(RuntimeError(NO_QUOTA))[0] == "no_quota"
    assert classify_error(RuntimeError("503 UNAVAILABLE. The model is overloaded."))[0] == "transient"
    assert classify_error(RuntimeError("400 INVALID_ARGUMENT"))[0] == "fatal"


def test_classifies_http_rate_limits():
    kind, wait = classify_error(_http_error(429, '{"error": {"type": "rate_limit"}}', {"retry-after": "12"}))
    assert kind == "per_minute" and wait == 12
    assert classify_error(_http_error(429, '{"error": {"code": "insufficient_quota"}}'))[0] == "daily_quota"
    assert classify_error(_http_error(502))[0] == "transient"


# --------------------------------------------------------------------------
# Retry policy
# --------------------------------------------------------------------------
def _flaky(errors, result="OK"):
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        if errors:
            raise errors.pop(0)
        return result
    return call, calls


def test_daily_quota_fails_fast(monkeypatch):
    slept = []
    monkeypatch.setattr(providers.time, "sleep", slept.append)
    call, calls = _flaky([RuntimeError(DAILY)] * 4)
    with pytest.raises(RuntimeError):
        providers._with_retries(call, max_retries=4)
    assert calls["n"] == 1 and slept == []


def test_per_minute_limit_waits_as_asked_then_succeeds(monkeypatch):
    slept = []
    monkeypatch.setattr(providers.time, "sleep", slept.append)
    call, calls = _flaky([RuntimeError(PER_MINUTE)])
    assert providers._with_retries(call, max_retries=4) == "OK"
    assert calls["n"] == 2 and 37 < slept[0] < 40


def test_per_minute_wait_is_capped(monkeypatch):
    slept = []
    monkeypatch.setattr(providers.time, "sleep", slept.append)
    call, _ = _flaky([RuntimeError(PER_MINUTE.replace("37.2s", "900s"))])
    providers._with_retries(call, max_retries=2)
    assert slept == [providers._MAX_RATE_LIMIT_WAIT_S]


def test_timeout_is_not_retried(monkeypatch):
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    call, calls = _flaky([httpx.ReadTimeout("slow")] * 4)
    with pytest.raises(ProviderTimeout, match="within 30 seconds"):
        providers._with_retries(call, max_retries=4, timeout_s=30)
    assert calls["n"] == 1


def test_gemini_call_carries_a_timeout(monkeypatch):
    seen = {}

    class FakeModels:
        def generate_content(self, model, contents, config):
            seen["timeout"] = config.http_options.timeout
            return type("R", (), {"text": "OK"})()

    monkeypatch.setattr(providers, "_gemini_client", lambda key: type("C", (), {"models": FakeModels()})())
    settings = providers.LLMSettings(chat_provider="gemini", chat_model="gemini-flash-lite-latest", chat_api_key="k")
    assert providers.chat(settings, "hi", None, False, timeout=30) == "OK"
    assert seen["timeout"] == 30_000


# --------------------------------------------------------------------------
# User-facing messages
# --------------------------------------------------------------------------
def test_messages_say_which_quota_and_what_to_do():
    daily = friendly_error(DAILY, model="gemini-3.5-flash", provider="gemini")
    assert "daily free quota for gemini-3.5-flash" in daily
    assert "per Google Cloud project" in daily and "gemini-flash-lite-latest" in daily

    assert "no free quota" in friendly_error(NO_QUOTA, model="gemini-3.8-flash", provider="gemini")
    assert "per minute" in friendly_error(PER_MINUTE, model="m", provider="gemini")
    slow = friendly_error("timeout: no answer within 90 seconds", model="gemini-3.5-flash", provider="gemini",
                          exc=ProviderTimeout("timeout: no answer within 90 seconds"))
    assert "didn't answer in time" in slow and "gemini-flash-lite-latest" in slow

    err = _http_error(429, '{"error": {"code": "insufficient_quota"}}')
    openai = friendly_error(str(err), model="gpt-4o-mini", provider="openai", exc=err)
    assert "billing" in openai


def test_default_gemini_model_is_flash_lite():
    from agents.runtime import DEFAULT_CHAT_MODELS
    assert DEFAULT_CHAT_MODELS["gemini"] == "gemini-flash-lite-latest"
