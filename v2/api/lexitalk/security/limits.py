"""Per-client rate limiting and a global daily budget.

In-memory, so it is correct for a single process (the HF Space deployment).
For multiple replicas, swap these for a Redis-backed implementation; the
interface is intentionally tiny.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from datetime import UTC, datetime


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: float = 60.0, clock=time.monotonic) -> None:
        self.limit = limit
        self.window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        now = self._clock()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False, max(1, int(self.window - (now - q[0])) + 1)
            q.append(now)
            if len(self._hits) > 10_000:  # bound memory under key-spraying
                for k in [k for k, v in self._hits.items() if not v][:5_000]:
                    del self._hits[k]
            return True, 0


class DailyBudget:
    def __init__(self, limit: int, today=lambda: datetime.now(UTC).date()) -> None:
        self.limit = limit
        self._today = today
        self._day = today()
        self._used = 0
        self._lock = threading.Lock()

    def consume(self) -> bool:
        with self._lock:
            d = self._today()
            if d != self._day:
                self._day, self._used = d, 0
            if self._used >= self.limit:
                return False
            self._used += 1
            return True

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self._used)
