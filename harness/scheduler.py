import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

from harness.budget import BudgetExhausted
from harness.contracts import SolveResult, SolveStatus, TaskSpec
from harness.task_io import write_answer

LOG = logging.getLogger("harness")


class FallbackUnavailable(RuntimeError):
    pass


class FallbackRetryLater(RuntimeError):

    def __init__(self, message: str, *, retry_after: float = 0):
        super().__init__(message)
        self.retry_after = retry_after


class FallbackSolver(Protocol):

    async def __call__(
        self, task: TaskSpec, previous: SolveResult, *, attempt: int, timeout: float,
    ) -> SolveResult: ...


@dataclass(frozen=True)
class ScheduleLimits:
    max_fallback_attempts: int = 2
    fallback_timeout: float = 45.0
    strong_timeout: float | None = None
    fallback_reserve: float = 60.0
    last_resort_reserve: float = 75.0
    fallback_concurrency: int = 1
    max_requeues: int = 2

    def __post_init__(self) -> None:
        if type(self.max_fallback_attempts) is not int or self.max_fallback_attempts < 0:
            raise ValueError("max_fallback_attempts must be a non-negative integer")
        if not math.isfinite(self.fallback_timeout) or self.fallback_timeout <= 0:
            raise ValueError("fallback_timeout must be finite and positive")
        if self.strong_timeout is not None and (not math.isfinite(self.strong_timeout) or self.strong_timeout <= 0):
            raise ValueError("strong_timeout must be finite and positive")
        if not math.isfinite(self.fallback_reserve) or self.fallback_reserve < 0:
            raise ValueError("fallback_reserve must be finite and non-negative")
        if not math.isfinite(self.last_resort_reserve) or self.last_resort_reserve < 0:
            raise ValueError("last_resort_reserve must be finite and non-negative")
        if type(self.fallback_concurrency) is not int or not 1 <= self.fallback_concurrency <= 5:
            raise ValueError("fallback_concurrency must be an integer from 1 to 5")
        if type(self.max_requeues) is not int or self.max_requeues < 0:
            raise ValueError("max_requeues must be a non-negative integer")


def _record(task: TaskSpec, result: SolveResult, phase: str) -> SolveResult:
    if result.status is SolveStatus.SOLVED:
        try:
            write_answer(task, result)
        except (OSError, ValueError) as error:
            result = SolveResult.error(f"cannot save answer: {error}")
    LOG.info("%r %s %s %r", task.task_id, result.status.value, phase, result.detail[:512])
    return result


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
        result = _record(task, result, "deterministic")
        if result.status in (SolveStatus.UNSUPPORTED, SolveStatus.ERROR):
            pending.append((task, result))

    pending.sort(key=lambda entry: -entry[0].reward)
    unresolved = [task for task, _ in pending]
    if fallback is not None:
        unresolved = await _fallback_pass(pending, fallback, deadline=deadline, limits=limits)
    if last_resort is not None:
        await _last_resort_pass(unresolved, last_resort, deadline=deadline, limits=limits)


def _priority(fallback: FallbackSolver, task: TaskSpec, attempt: int) -> float:
    estimate = getattr(fallback, "estimate", None)
    if estimate is None:
        return task.reward
    try:
        cost = float(estimate(task, attempt))
    except Exception as error:
        LOG.warning("%r cost estimate failed: %s: %s", task.task_id, type(error).__name__, error)
        return task.reward
    return task.reward / cost if math.isfinite(cost) and cost >= 1 else task.reward


async def _fallback_pass(
    pending: list[tuple[TaskSpec, SolveResult]], fallback: FallbackSolver, *, deadline: float, limits: ScheduleLimits,
) -> list[TaskSpec]:
    cutoff = deadline - limits.fallback_reserve
    final: set[int] = set()
    stopped = False
    not_before = 0.0

    async def attempt_one(task: TaskSpec, previous: SolveResult, attempt: int,
                          retry: list[tuple[TaskSpec, SolveResult]], deferred, requeue) -> None:
        nonlocal stopped, not_before
        pause = not_before - time.monotonic()
        if pause > 0:
            if not_before >= cutoff:
                stopped = True
                return
            await asyncio.sleep(pause)
        maximum = limits.strong_timeout if attempt >= 2 and limits.strong_timeout is not None else limits.fallback_timeout
        timeout = min(maximum, cutoff - time.monotonic())
        if timeout <= 0:
            LOG.warning("fallback deadline reached")
            stopped = True
            return
        try:
            async with asyncio.timeout(timeout):
                result = await fallback(task, previous, attempt=attempt, timeout=timeout)
            if not isinstance(result, SolveResult):
                raise TypeError("fallback solver must return SolveResult")
        except BudgetExhausted as error:
            LOG.info("%r fallback #%d skipped by budget: %s", task.task_id, attempt, error)
            if error.temporary:
                deferred.append((task, previous))
            return
        except FallbackUnavailable as error:
            if not stopped:
                LOG.warning("fallback stopped: %r", str(error)[:512])
            stopped = True
            return
        except (FallbackRetryLater, TimeoutError) as error:
            detail = str(error) if isinstance(error, FallbackRetryLater) else "fallback attempt timed out"
            if isinstance(error, FallbackRetryLater) and error.retry_after > 0:
                not_before = max(not_before, time.monotonic() + error.retry_after)
            if requeue(task, previous):
                LOG.info("%r requeued after fallback #%d: %s", task.task_id, attempt, detail[:200])
                return
            result = SolveResult.error(detail)
            _record(task, result, f"fallback #{attempt}")
            retry.append((task, result))
            return
        except Exception as error:
            result = SolveResult.error(f"{type(error).__name__}: {error}")
        result = _record(task, result, f"fallback #{attempt}")
        if result.status in (SolveStatus.UNSUPPORTED, SolveStatus.ERROR):
            retry.append((task, result))
        elif result.status in (SolveStatus.SOLVED, SolveStatus.UNSOLVABLE):
            final.add(id(task))

    active = list(pending)
    for attempt in range(1, limits.max_fallback_attempts + 1):
        active.sort(key=lambda entry: -_priority(fallback, entry[0], attempt))
        retry: list[tuple[TaskSpec, SolveResult]] = []
        deferred: list[tuple[TaskSpec, SolveResult]] = []
        order = list(active)
        position = 0
        requeued: dict[int, int] = {}

        def requeue(task: TaskSpec, previous: SolveResult) -> bool:
            if requeued.get(id(task), 0) >= limits.max_requeues:
                return False
            requeued[id(task)] = requeued.get(id(task), 0) + 1
            order.append((task, previous))
            return True

        async def worker() -> None:
            nonlocal position
            while not stopped and position < len(order):
                task, previous = order[position]
                position += 1
                await attempt_one(task, previous, attempt, retry, deferred, requeue)

        await asyncio.gather(*(worker() for _ in range(limits.fallback_concurrency)))
        for task, previous in deferred:
            if stopped:
                break
            await attempt_one(task, previous, attempt, retry, [], requeue)
        await worker()
        if stopped:
            break
        active = retry
    return [task for task, _ in pending if id(task) not in final]


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
