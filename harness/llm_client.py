import asyncio
import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from harness.budget import BudgetLedger

LOG = logging.getLogger("harness")

Transport = Callable[[str, dict[str, str], bytes, float], tuple[int, bytes]]


class LLMUnavailable(RuntimeError):
    pass


class LLMError(RuntimeError):

    def __init__(self, message: str, *, transient: bool = False):
        super().__init__(message)
        self.transient = transient


class NotSent(OSError):
    pass


@dataclass(frozen=True)
class ChatResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    seconds: float
    finish_reason: str | None


def urllib_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
    except urllib.error.URLError as error:
        raise NotSent(str(error.reason)) from error


def estimate_tokens(text: str) -> int:
    return len(text) // 3 + 1


class LLMClient:
    def __init__(
        self, base_url: str, api_key: str, ledger: BudgetLedger, *,
        transport: Transport = urllib_transport, max_consecutive_failures: int = 3, outage_limit: float = 300.0,
        failure_pause: float = 2.0,
        rate_limit_retries: int = 4, rate_limit_pause: float = 2.0,
    ):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.ledger = ledger
        self.transport = transport
        self.max_consecutive_failures = max_consecutive_failures
        self.rate_limit_retries = rate_limit_retries
        self.rate_limit_pause = rate_limit_pause
        self.outage_limit = outage_limit
        self.failure_pause = failure_pause
        self._failures = 0
        self._outage_started = 0.0
        self._refused = ""

    async def chat(
        self, model: str, messages: list[dict[str, str]], *, max_tokens: int, timeout: float, label: str = "",
    ) -> ChatResult:
        deadline = time.monotonic() + timeout
        for attempt in range(self.rate_limit_retries + 1):
            result = await self._send(model, messages, max_tokens=max_tokens,
                                      timeout=max(1.0, deadline - time.monotonic()), label=label)
            if result is not None:
                return result
            pause = self.rate_limit_pause + attempt
            if attempt == self.rate_limit_retries or time.monotonic() + pause >= deadline - 1:
                break
            LOG.info("LLM %s rate limited; retrying in %.0fs", label, pause)
            await asyncio.sleep(pause)
        self._failed()
        raise LLMError("HTTP 429: rate limited", transient=True)

    def _failed(self) -> None:
        if self._failures == 0:
            self._outage_started = time.monotonic()
        self._failures += 1

    async def _send(
        self, model: str, messages: list[dict[str, str]], *, max_tokens: int, timeout: float, label: str,
    ) -> ChatResult | None:
        if self._refused:
            raise LLMUnavailable(self._refused)
        if self._failures >= self.max_consecutive_failures:
            down = time.monotonic() - self._outage_started
            if down >= self.outage_limit:
                raise LLMUnavailable(f"{self._failures} consecutive proxy failures over {down:.0f}s")
        if self._failures:
            await asyncio.sleep(min(self.failure_pause * 2 ** (self._failures - 1), 15.0))
        prompt_estimate = sum(estimate_tokens(message["content"]) + 8 for message in messages) + 16
        reservation = self.ledger.reserve(prompt_estimate, max_tokens)

        body = json.dumps({
            "model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
        }).encode("utf-8")
        started = time.monotonic()
        try:
            status, raw = await asyncio.to_thread(self.transport, self.url, self._headers, body, timeout)
        except BaseException as error:
            if isinstance(error, NotSent):
                self.ledger.release(reservation)
            else:
                self.ledger.charge_unknown(reservation)
            self._failed()
            if isinstance(error, (OSError, TimeoutError)):
                raise LLMError(f"transport failure: {type(error).__name__}: {error}", transient=True) from error
            raise
        seconds = time.monotonic() - started

        if status == 429:
            self.ledger.release(reservation)
            return None
        if status != 200:
            self.ledger.charge_unknown(reservation, input_only=True, input_tokens=prompt_estimate)
            detail = raw[:300].decode("utf-8", "replace")
            if status in (401, 402, 403):
                self._refused = f"proxy refused with HTTP {status}: {detail}"
                raise LLMUnavailable(self._refused)
            if status >= 500:
                self._failed()
            raise LLMError(f"HTTP {status}: {detail}", transient=status >= 500)

        try:
            data = json.loads(raw)
            usage = data["usage"]
            prompt_tokens, completion_tokens = int(usage["prompt_tokens"]), int(usage["completion_tokens"])
        except (ValueError, KeyError, TypeError) as error:
            self.ledger.charge_unknown(reservation)
            self._failed()
            raise LLMError(f"response without usable usage: {error}") from error
        cost = self.ledger.settle(reservation, prompt_tokens, completion_tokens) / 10
        self._failures = 0
        choice = (data.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        LOG.info(
            "LLM %s model=%s in=%d out=%d cost=%.1f %.1fs finish=%s",
            label, model, prompt_tokens, completion_tokens, cost, seconds, choice.get("finish_reason"),
        )
        return ChatResult(content, prompt_tokens, completion_tokens, cost, seconds, choice.get("finish_reason"))
