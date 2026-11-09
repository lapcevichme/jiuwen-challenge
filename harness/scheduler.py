import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

from harness.contracts import SolveResult, SolveStatus, TaskSpec
from harness.task_io import write_answer

LOG = logging.getLogger("harness")


class FallbackUnavailable(RuntimeError):
    pass


class FallbackSolver(Protocol):
    async def __call__(
        self, task: TaskSpec, previous: SolveResult, *, attempt: int, timeout: float,
    ) -> SolveResult: ...


@dataclass(frozen=True)
class ScheduleLimits:
    max_fallback_attempts: int = 2
    fallback_timeout: float = 45.0
    fallback_reserve: float = 60.0

    def __post_init__(self) -> None:
        if type(self.max_fallback_attempts) is not int or self.max_fallback_attempts < 0:
            raise ValueError("max_fallback_attempts must be a non-negative integer")
        if not math.isfinite(self.fallback_timeout) or self.fallback_timeout <= 0:
            raise ValueError("fallback_timeout must be finite and positive")
        if not math.isfinite(self.fallback_reserve) or self.fallback_reserve < 0:
            raise ValueError("fallback_reserve must be finite and non-negative")


def _record(task: TaskSpec, result: SolveResult, phase: str) -> None:
    if result.status is SolveStatus.SOLVED:
        try:
            write_answer(task, result)
        except (OSError, ValueError) as error:
            result = SolveResult.error(f"cannot save answer: {error}")
    LOG.info("%r %s %s %r", task.task_id, result.status.value, phase, result.detail[:512])


async def run_tasks(
    tasks: Iterable[TaskSpec],
    deterministic: Callable[[TaskSpec], Awaitable[SolveResult]],
    *,
    deadline: float,
    fallback: FallbackSolver | None = None,
    limits: ScheduleLimits = ScheduleLimits(),
) -> None:
    pending: list[tuple[TaskSpec, SolveResult]] = []
    for task in tasks:
        if time.monotonic() >= deadline:
            LOG.warning("deterministic deadline reached")
            return
        try:
            result = await deterministic(task)
            if not isinstance(result, SolveResult):
                raise TypeError("deterministic solver must return SolveResult")
        except Exception as error:
            result = SolveResult.error(f"{type(error).__name__}: {error}")
        _record(task, result, "deterministic")
        if result.status is SolveStatus.UNSUPPORTED:
            pending.append((task, result))

    if fallback is None:
        return

    cutoff = deadline - limits.fallback_reserve
    for task, previous in sorted(pending, key=lambda entry: -entry[0].reward):
        for attempt in range(1, limits.max_fallback_attempts + 1):
            timeout = min(limits.fallback_timeout, cutoff - time.monotonic())
            if timeout <= 0:
                LOG.warning("fallback deadline reached")
                return
            try:
                async with asyncio.timeout(timeout):
                    result = await fallback(task, previous, attempt=attempt, timeout=timeout)
                if not isinstance(result, SolveResult):
                    raise TypeError("fallback solver must return SolveResult")
            except FallbackUnavailable as error:
                LOG.warning("fallback stopped: %r", str(error)[:512])
                return
            except TimeoutError:
                previous = SolveResult.error("fallback attempt timed out")
                _record(task, previous, f"fallback #{attempt}")
                continue
            except Exception as error:
                result = SolveResult.error(f"{type(error).__name__}: {error}")
            _record(task, result, f"fallback #{attempt}")
            if result.status is not SolveStatus.UNSUPPORTED:
                break
            previous = result
