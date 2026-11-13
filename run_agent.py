import argparse
import asyncio
import logging
import os
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

from harness.contracts import Solver, TaskSpec
from harness.config import number
from harness.task_io import load_task, resolve_inside

LOG = logging.getLogger("harness")
DEFAULT_BUDGET = 100_000
DEFAULT_CONCURRENCY = 3


def build_registry() -> dict[str, Solver]:
    if os.environ.get("HARNESS_DETERMINISTIC") == "0":
        return {}
    from harness.solvers import coding, data_analysis, math, search

    return {"math": math.solve, "data_analysis": data_analysis.solve, "search": search.solve,
            "coding": coding.precheck}


def build_last_resort() -> dict[str, Solver]:
    if os.environ.get("HARNESS_DETERMINISTIC") == "0":
        return {}
    from harness.solvers import coding

    return {"coding": coding.solve}


def _tasks(root: Path) -> Iterator[TaskSpec]:
    for entry in sorted(root.iterdir()):
        try:
            task_dir = resolve_inside(root, entry.name)
            if task_dir != entry:
                raise ValueError("task directory must not be an alias")
            if not task_dir.is_dir() or not resolve_inside(task_dir, "task.json").is_file():
                continue
            yield load_task(task_dir)
        except (OSError, ValueError, RuntimeError) as error:
            LOG.error("%r INVALID_TASK %s", entry.name, error)


def _configure_sdk() -> None:
    cache_root = Path(tempfile.gettempdir()) / "jiuwen-cache"
    for name, directory in (
        ("XDG_CACHE_HOME", cache_root),
        ("MPLCONFIGDIR", cache_root / "matplotlib"),
        ("HF_HOME", cache_root / "huggingface"),
    ):
        os.environ.setdefault(name, str(directory))
    from openjiuwen.core.common.logging.log_config import configure_log_config

    configure_log_config({
        "level": "WARNING",
        "output": ["console"],
        "interface_output": ["console"],
        "performance_output": ["console"],
        "log_path": str(cache_root / "logs"),
        "propagate": False,
    })


def build_fallback():
    base_url, api_key = os.environ.get("LLM_BASE_URL"), os.environ.get("LLM_API_KEY")
    if not base_url or not api_key or os.environ.get("HARNESS_LLM") == "0":
        LOG.warning("LLM fallback disabled")
        return None, None
    from harness.budget import BudgetLedger
    from harness.llm_client import LLMClient
    from harness.solvers.llm import LLMFallback, Models

    try:
        budget = float(os.environ.get("BUDGET") or DEFAULT_BUDGET)
        ledger = BudgetLedger(budget)
    except ValueError:
        LOG.warning("invalid BUDGET %r; using %d", os.environ.get("BUDGET"), DEFAULT_BUDGET)
        ledger = BudgetLedger(DEFAULT_BUDGET)
    try:
        models = Models(
            fast=os.environ.get("LLM_MODEL") or Models.fast,
            strong=os.environ.get("LLM_STRONG_MODEL") or Models.strong,
            fast_max_tokens=number("LLM_FAST_MAX_TOKENS", Models.fast_max_tokens, minimum=128, maximum=16000),
            strong_max_tokens=number("LLM_STRONG_MAX_TOKENS", Models.strong_max_tokens, minimum=128, maximum=16000),
        )
    except ValueError:
        LOG.warning("unapproved LLM model override; using competition defaults")
        models = Models()
    trust = os.environ.get("LLM_TRUST_FAST_UNSOLVABLE", "1") != "0"
    from harness.agent import JiuwenChat

    chat = JiuwenChat(LLMClient(base_url, api_key, ledger), default_model=models.fast)
    return LLMFallback(chat, models, trust_fast_unsolvable=trust,
                       adaptive_output=os.environ.get("LLM_ADAPTIVE_OUTPUT", "1") != "0"), ledger


def build_limits() -> "ScheduleLimits":
    from harness.scheduler import ScheduleLimits

    return ScheduleLimits(
        fallback_concurrency=number("LLM_CONCURRENCY", DEFAULT_CONCURRENCY, minimum=1, maximum=5),
        fallback_timeout=number("LLM_FAST_TIMEOUT", 45.0, minimum=1, maximum=600),
        strong_timeout=number("LLM_STRONG_TIMEOUT", 120.0, minimum=1, maximum=600),
    )


async def _run(root: Path, deadline: float) -> None:
    from harness.agent import ContestAgent
    from harness.runtime import sandbox_status
    from harness.workflow import SolverWorkflow

    LOG.info("task sandbox: %s", sandbox_status())
    workflow = SolverWorkflow(build_registry())
    rescue = build_last_resort()
    last_resort = SolverWorkflow(rescue).solve if rescue else None
    fallback, ledger = build_fallback()
    agent = ContestAgent(workflow.solve, fallback=fallback, last_resort=last_resort, limits=build_limits())
    try:
        await agent.invoke({"tasks": _tasks(root), "deadline": deadline})
    finally:
        if ledger is not None:
            LOG.info("%s", ledger.summary())


def main(argv: list[str] | None = None) -> int:
    deadline = time.monotonic() + 3600
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.tasks_dir.is_dir():
        parser.error("--tasks-dir must be an existing directory")
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    LOG.setLevel(logging.INFO)
    sys.dont_write_bytecode = True
    _configure_sdk()
    asyncio.run(_run(args.tasks_dir.resolve(), deadline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
