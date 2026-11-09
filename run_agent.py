import argparse
import asyncio
import importlib.util
import logging
import os
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

from harness.contracts import Solver, TaskSpec
from harness.scheduler import run_tasks
from harness.task_io import load_task, resolve_inside

LOG = logging.getLogger("harness")
DEFAULT_BUDGET = 100_000


def build_registry() -> dict[str, Solver]:
    if os.environ.get("HARNESS_DETERMINISTIC") == "0":
        return {}
    from harness.solvers import data_analysis, math, search

    registry = {"math": math.solve, "data_analysis": data_analysis.solve, "search": search.solve}
    if importlib.util.find_spec("harness.solvers.coding") is not None:
        from harness.solvers.coding import solve

        registry["coding"] = solve
    return registry


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
    models = Models(
        fast=os.environ.get("LLM_MODEL") or Models.fast,
        strong=os.environ.get("LLM_STRONG_MODEL") or Models.strong,
    )
    return LLMFallback(LLMClient(base_url, api_key, ledger), models), ledger


async def _run(root: Path, deadline: float) -> None:
    from harness.workflow import SolverWorkflow

    workflow = SolverWorkflow(build_registry())
    fallback, ledger = build_fallback()
    try:
        await run_tasks(_tasks(root), workflow.solve, deadline=deadline, fallback=fallback)
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
