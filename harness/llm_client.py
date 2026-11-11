import asyncio
import json
import logging
import socket
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from email.utils import parsedate_to_datetime

from harness.budget import BudgetLedger

LOG = logging.getLogger("harness")

Transport = Callable[[str, dict[str, str], bytes, float], tuple[int, bytes] | tuple[int, bytes, float]]


class LLMUnavailable(RuntimeError):
    pass


class LLMError(RuntimeError):

    def __init__(self, message: str, *, transient: bool = False, retry_after: float = 0):
        super().__init__(message)
        self.transient = transient
        self.retry_after = retry_after


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


def urllib_transport(url: str, headers: dict[str, str], body: bytes, timeout: float):
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        if error.code == 429 and (value := error.headers.get("Retry-After")):
            try:
                delay = float(value)
            except ValueError:
                try:
                    delay = parsedate_to_datetime(value).timestamp() - time.time()
                except (ValueError, TypeError, OverflowError):
                    delay = 0
            if 0 < delay <= 3600:
                return error.code, error.read(), delay
        return error.code, error.read()
    except urllib.error.URLError as error:
        if isinstance(error.reason, (ConnectionRefusedError, socket.gaierror, ssl.SSLCertVerificationError)):
            raise NotSent(str(error.reason)) from error
        raise OSError(str(error.reason)) from error


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
        self._cooldown_until = 0.0

    async def chat(
        self, model: str, messages: list[dict[str, str]], *, max_tokens: int, timeout: float, label: str = "",
        min_tokens: int | None = None,
    ) -> ChatResult:
        deadline = time.monotonic() + timeout
        cooldown = self._cooldown_until - time.monotonic()
        if cooldown > 0:
            raise LLMError("proxy Retry-After cooldown", transient=True, retry_after=cooldown)
        for attempt in range(self.rate_limit_retries + 1):
            result = await self._send(model, messages, max_tokens=max_tokens,
                                      timeout=max(0.01, deadline - time.monotonic()), label=label,
                                      min_tokens=min_tokens)
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
        min_tokens: int | None = None,
    ) -> ChatResult | None:
        deadline = time.monotonic() + timeout
        cooldown = self._cooldown_until - time.monotonic()
        if cooldown > 0:
            raise LLMError("proxy Retry-After cooldown", transient=True, retry_after=cooldown)
        if self._refused:
            raise LLMUnavailable(self._refused)
        if self._failures >= self.max_consecutive_failures:
            down = time.monotonic() - self._outage_started
            if down >= self.outage_limit:
                raise LLMUnavailable(f"{self._failures} consecutive proxy failures over {down:.0f}s")
        if self._failures:
            pause = min(self.failure_pause * 2 ** min(self._failures - 1, 10), 15.0)
            if pause >= timeout:
                raise LLMError("proxy backoff exceeds attempt deadline", transient=True)
            await asyncio.sleep(pause)
        timeout = deadline - time.monotonic()
        if timeout <= 0:
            raise LLMError("request deadline reached before send", transient=True)
        prompt_estimate = sum(estimate_tokens(message["content"]) + 8 for message in messages) + 16
        reservation, max_tokens = self.ledger.reserve_up_to(
            prompt_estimate, max_tokens, min_output_tokens=max_tokens if min_tokens is None else min_tokens,
        )

        body = json.dumps({
            "model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
        }).encode("utf-8")
        started = time.monotonic()
        cancelled = False

        def account(cost: int, status: str, known: bool) -> None:
            LOG.info("LLM_USAGE %s", json.dumps({"task": label, "model": model, "cost": cost / 10,
                     "status": status, "known": known, "max_tokens": max_tokens,
                     "seconds": round(time.monotonic() - started, 3)}))

        try:
            request = asyncio.create_task(asyncio.to_thread(self.transport, self.url, self._headers, body, timeout))
            try:
                received = await asyncio.shield(request)
            except asyncio.CancelledError:
                cancelled = True
                received = await request
            status, raw = received[:2]
        except BaseException as error:
            if isinstance(error, NotSent):
                self.ledger.release(reservation)
                account(0, "not_sent", False)
            else:
                cost = self.ledger.charge_unknown(reservation)
                account(cost, type(error).__name__, False)
            self._failed()
            if isinstance(error, (OSError, TimeoutError)):
                raise LLMError(f"transport failure: {type(error).__name__}: {error}", transient=True) from error
            raise
        seconds = time.monotonic() - started

        if status == 429:
            self.ledger.release(reservation)
            account(0, "429", False)
            if cancelled:
                raise asyncio.CancelledError
            if len(received) == 3 and received[2] > 0:
                self._cooldown_until = max(self._cooldown_until, time.monotonic() + received[2])
                raise LLMError("HTTP 429: respect Retry-After", transient=True, retry_after=received[2])
            return None
        if status != 200:
            cost = self.ledger.charge_unknown(reservation, input_only=True, input_tokens=prompt_estimate)
            account(cost, str(status), False)
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
            prompt_tokens, completion_tokens = usage["prompt_tokens"], usage["completion_tokens"]
            if any(type(value) is not int or value < 0 for value in (prompt_tokens, completion_tokens)):
                raise ValueError("invalid token counts")
        except (ValueError, KeyError, TypeError) as error:
            cost = self.ledger.charge_unknown(reservation)
            account(cost, "invalid_usage", False)
            self._failed()
            raise LLMError(f"response without usable usage: {error}") from error
        cost_tenths = self.ledger.settle(reservation, prompt_tokens, completion_tokens)
        account(cost_tenths, "200", True)
        cost = cost_tenths / 10
        self._failures = 0
        if cancelled:
            raise asyncio.CancelledError
        choice = (data.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        if not isinstance(content, str):
            raise LLMError("response content must be text")
        LOG.info(
            "LLM %s model=%s in=%d out=%d cost=%.1f %.1fs finish=%s",
            label, model, prompt_tokens, completion_tokens, cost, seconds, choice.get("finish_reason"),
        )
        return ChatResult(content, prompt_tokens, completion_tokens, cost, seconds, choice.get("finish_reason"))
