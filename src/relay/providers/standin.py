"""Local stand-in for measuring simulated model behavior."""

from collections import deque
from threading import Lock
from time import monotonic, sleep


class StandInError(Exception):
    """Base error raised by the local stand-in provider."""

    status_code = 500


class RateLimitError(StandInError):
    """Raised when a simulated rate or daily limit is exceeded."""

    status_code = 429


class OverloadedError(StandInError):
    """Raised when the stand-in is handling too many concurrent requests."""

    status_code = 503


class StandInProvider:
    def __init__(
        self,
        *,
        ttft_ms=806,
        output_tokens_per_second=775,
        requests_per_minute=30,
        requests_per_day=1000,
        tokens_per_minute=8000,
        tokens_per_day=200000,
        max_concurrency=5,
        extra_load_delay_ms=100,
        fail_every_n_requests=0,
    ):
        self.ttft_ms = ttft_ms
        self.output_tokens_per_second = output_tokens_per_second
        self.requests_per_minute = requests_per_minute
        self.requests_per_day = requests_per_day
        self.tokens_per_minute = tokens_per_minute
        self.tokens_per_day = tokens_per_day
        self.max_concurrency = max_concurrency
        self.extra_load_delay_ms = extra_load_delay_ms
        self.fail_every_n_requests = fail_every_n_requests

        self._lock = Lock()
        self._request_times = deque()
        self._daily_request_times = deque()
        self._token_events = deque()
        self._daily_token_events = deque()
        self._active_requests = 0
        self._request_number = 0

    def stream(
        self,
        *,
        input_tokens=974,
        output_tokens=319,
        chunk_tokens=20,
    ):
        """Yield simulated text chunks, one chunk at a time."""
        now = monotonic()
        total_tokens = input_tokens + output_tokens

        with self._lock:
            self._remove_expired_events(now)

            if len(self._request_times) >= self.requests_per_minute:
                raise RateLimitError("Simulated requests-per-minute limit reached.")

            if len(self._daily_request_times) >= self.requests_per_day:
                raise RateLimitError("Simulated requests-per-day limit reached.")

            minute_tokens = sum(count for _, count in self._token_events)
            if minute_tokens + total_tokens > self.tokens_per_minute:
                raise RateLimitError("Simulated tokens-per-minute limit reached.")

            daily_tokens = sum(count for _, count in self._daily_token_events)
            if daily_tokens + total_tokens > self.tokens_per_day:
                raise RateLimitError("Simulated tokens-per-day limit reached.")

            if self._active_requests >= self.max_concurrency:
                raise OverloadedError("Stand-in is at its concurrency limit.")

            self._request_number += 1
            request_number = self._request_number
            self._request_times.append(now)
            self._daily_request_times.append(now)
            self._token_events.append((now, total_tokens))
            self._daily_token_events.append((now, total_tokens))

            self._active_requests += 1
            active_requests = self._active_requests

        try:
            if (
                self.fail_every_n_requests
                and request_number % self.fail_every_n_requests == 0
            ):
                raise OverloadedError("Simulated provider failure.")

            load_delay = max(0, active_requests - 1) * self.extra_load_delay_ms
            sleep((self.ttft_ms + load_delay) / 1000)

            remaining_tokens = output_tokens
            chunk_number = 1

            while remaining_tokens > 0:
                current_chunk_size = min(chunk_tokens, remaining_tokens)
                yield {
                    "content": (
                        f"[simulated chunk {chunk_number}; "
                        f"{current_chunk_size} tokens]"
                    ),
                    "tokens_in": input_tokens if chunk_number == 1 else 0,
                    "tokens_out": current_chunk_size,
                }

                remaining_tokens -= current_chunk_size
                chunk_number += 1

                if remaining_tokens > 0:
                    sleep(current_chunk_size / self.output_tokens_per_second)

        finally:
            with self._lock:
                self._active_requests -= 1

    def _remove_expired_events(self, now):
        minute_cutoff = now - 60
        day_cutoff = now - 86400

        while self._request_times and self._request_times[0] <= minute_cutoff:
            self._request_times.popleft()

        while (
            self._daily_request_times
            and self._daily_request_times[0] <= day_cutoff
        ):
            self._daily_request_times.popleft()

        while self._token_events and self._token_events[0][0] <= minute_cutoff:
            self._token_events.popleft()

        while (
            self._daily_token_events
            and self._daily_token_events[0][0] <= day_cutoff
        ):
            self._daily_token_events.popleft()