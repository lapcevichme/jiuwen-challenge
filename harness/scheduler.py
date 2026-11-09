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
    last_resort_reserve: float = 75.0

    def __post_init__(self) -> None:
        if type(self.max_fallback_attempts) is not int or self.max_fallback_attempts < 0:
            raise ValueError("max_fallback_attempts must be a non-negative integer")
        if not math.isfinite(self.fallback_timeout) or self.fallback_timeout <= 0:
            raise ValueError("fallback_timeout must be finite and positive")
        if not math.isfinite(self.fallback_reserve) or self.fallback_reserve < 0:
            raise ValueError("fallback_reserve must be finite and non-negative")
        if not math.isfinite(self.last_resort_reserve) or self.last_resort_reserve < 0:
            raise ValueError("last_resort_reserve must be finite and non-negative")


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
    last_resort: Callable[[TaskSpec], Awaitable[SolveResult]] | None = None,
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

    pending.sort(key=lambda entry: -entry[0].reward)
    unresolved = [task for task, _ in pending]
    if fallback is not None:
        unresolved = await _fallback_pass(pending, fallback, deadline=deadline, limits=limits)
    if last_resort is not None:
        await _last_resort_pass(unresolved, last_resort, deadline=deadline, limits=limits)


async def _fallback_pass(
    pending: list[tuple[TaskSpec, SolveResult]], fallback: FallbackSolver, *, deadline: float, limits: ScheduleLimits,
) -> list[TaskSpec]:
    cutoff = deadline - limits.fallback_reserve
    unresolved: list[TaskSpec] = []
    for index, (task, previous) in enumerate(pending):
        final = False
        for attempt in range(1, limits.max_fallback_attempts + 1):
            timeout = min(limits.fallback_timeout, cutoff - time.monotonic())
            if timeout <= 0:
                LOG.warning("fallback deadline reached")
                return unresolved + [task for task, _ in pending[index:]]
            try:
                async with asyncio.timeout(timeout):
                    result = await fallback(task, previous, attempt=attempt, timeout=timeout)
                if not isinstance(result, SolveResult):
                    raise TypeError("fallback solver must return SolveResult")
            except FallbackUnavailable as error:
                LOG.warning("fallback stopped: %r", str(error)[:512])
                return unresolved + [task for task, _ in pending[index:]]
            except TimeoutError:
                previous = SolveResult.error("fallback attempt timed out")
                _record(task, previous, f"fallback #{attempt}")
                continue
            except Exception as error:
                result = SolveResult.error(f"{type(error).__name__}: {error}")
            _record(task, result, f"fallback #{attempt}")
            if result.status is not SolveStatus.UNSUPPORTED:
                final = result.status in (SolveStatus.SOLVED, SolveStatus.UNSOLVABLE)
                break
            previous = result
        if not final:
            unresolved.append(task)
    return unresolved


async def _last_resort_pass(
    tasks: list[TaskSpec], last_resort: Callable[[TaskSpec], Awaitable[SolveResult]], *,
    deadline: float, limits: ScheduleLimits,
) -> None:
    for task in tasks:
        if deadline - time.monotonic() < limits.last_resort_reserve:
            LOG.warning("last-resort deadline reached")
            return
        try:
            result = await last_resort(task)
            if not isinstance(result, SolveResult):
                raise TypeError("last-resort solver must return SolveResult")
        except Exception as error:
            result = SolveResult.error(f"{type(error).__name__}: {error}")
        _record(task, result, "last resort")
