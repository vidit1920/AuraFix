"""
ratelimit.py — a small, thread-safe, in-process rate limiter.

A pipeline run is expensive: it clones a repo, builds an embedding index, and
makes several LLM calls. Even though each user brings their own LLM key, the
*server* still does real work (disk, CPU, subprocess test runs), so we cap how
often any one identity can trigger it, plus a global ceiling so a crowd can't
exhaust the box.

Implementation is a sliding-window log per (bucket, identity): a deque of
request timestamps, pruned to the window on each check. This is exact (no
token-bucket drift), cheap at these volumes, and needs no external store —
appropriate for a single Render instance. If the app is ever scaled to
multiple instances, swap this for a shared store (Redis) behind the same
interface.

Identity is the Firebase uid when signed in, else the client IP. Limits come
from the environment with conservative defaults.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque, OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class Limit:
    max_events: int
    window_seconds: int


def _limit_from_env(name: str, default_max: int, default_window: int) -> Limit:
    try:
        return Limit(
            max_events=int(os.environ.get(f"{name}_MAX", default_max)),
            window_seconds=int(os.environ.get(f"{name}_WINDOW", default_window)),
        )
    except ValueError:
        return Limit(default_max, default_window)


# Defaults: 5 runs / 10 min and 40 runs / day per identity; 40 chat msgs /
# 10 min; and a global ceiling of 60 runs / 10 min across all users.
RUN_LIMITS = (
    _limit_from_env("AURAFIX_RL_RUN_SHORT", 5, 600),
    _limit_from_env("AURAFIX_RL_RUN_DAY", 40, 86400),
)
CHAT_LIMITS = (_limit_from_env("AURAFIX_RL_CHAT", 40, 600),)
GLOBAL_RUN_LIMIT = _limit_from_env("AURAFIX_RL_GLOBAL_RUN", 60, 600)

# Cap how many identities we track so the limiter itself can't grow without
# bound under churn; evict least-recently-used.
_MAX_IDENTITIES = 5000


class _Windows:
    """Sliding-window counters for one named bucket across many identities."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._log: "OrderedDict[str, deque[float]]" = OrderedDict()

    def exceeds(self, identity: str, limits: tuple[Limit, ...], now: float) -> bool:
        """True if recording one more event now would break any limit. Does
        not record anything (a pure peek)."""
        longest = max(l.window_seconds for l in limits)
        with self._lock:
            dq = self._log.get(identity)
            if dq is None:
                return False
            self._log.move_to_end(identity)
            cutoff = now - longest
            while dq and dq[0] < cutoff:
                dq.popleft()
            for limit in limits:
                window_start = now - limit.window_seconds
                count = sum(1 for ts in dq if ts >= window_start)
                if count >= limit.max_events:
                    return True
            return False

    def record(self, identity: str, now: float) -> None:
        with self._lock:
            dq = self._log.get(identity)
            if dq is None:
                dq = deque()
                self._log[identity] = dq
            self._log.move_to_end(identity)
            dq.append(now)
            while len(self._log) > _MAX_IDENTITIES:
                self._log.popitem(last=False)

    def hit(self, identity: str, limits: tuple[Limit, ...], now: float) -> bool:
        """Peek-and-record atomically for a single bucket. Returns True if
        allowed (and recorded)."""
        if self.exceeds(identity, limits, now):
            return False
        self.record(identity, now)
        return True

    def retry_after(self, identity: str, limits: tuple[Limit, ...], now: float) -> int:
        """Seconds until the tightest exceeded limit frees up (for messaging)."""
        with self._lock:
            dq = self._log.get(identity)
            if not dq:
                return 0
            waits = []
            for limit in limits:
                window_start = now - limit.window_seconds
                in_window = [ts for ts in dq if ts >= window_start]
                if len(in_window) >= limit.max_events:
                    waits.append(int(in_window[0] + limit.window_seconds - now) + 1)
            return max(waits) if waits else 0


_runs = _Windows()
_chats = _Windows()
_global_runs = _Windows()


@dataclass(frozen=True)
class Decision:
    allowed: bool
    retry_after: int = 0
    reason: str = ""


def check_run(identity: str) -> Decision:
    """Gate a pipeline run. Checks the global ceiling first, then per-identity
    limits. Nothing is recorded unless the run is allowed."""
    now = time.time()
    # Peek the global ceiling first (no record), so a per-identity rejection
    # doesn't consume a global slot.
    if _global_runs.exceeds("__global__", (GLOBAL_RUN_LIMIT,), now):
        wait = _global_runs.retry_after("__global__", (GLOBAL_RUN_LIMIT,), now)
        return Decision(False, wait, "The service is busy right now. Please try again shortly.")

    if not _runs.hit(identity, RUN_LIMITS, now):
        wait = _runs.retry_after(identity, RUN_LIMITS, now)
        return Decision(False, wait, "You've run AuraFix several times recently. Please wait a bit before the next run.")

    # Per-identity run allowed — now consume a global slot too.
    _global_runs.record("__global__", now)
    return Decision(True)


def check_chat(identity: str) -> Decision:
    now = time.time()
    if not _chats.hit(identity, CHAT_LIMITS, now):
        wait = _chats.retry_after(identity, CHAT_LIMITS, now)
        return Decision(False, wait, "Too many messages in a short time. Please wait a moment.")
    return Decision(True)
