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


def build_registry() -> dict[str, Solver]:
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


async def _run(root: Path, deadline: float) -> None:
    from harness.workflow import SolverWorkflow

    workflow = SolverWorkflow(build_registry())
    await run_tasks(_tasks(root), workflow.solve, deadline=deadline)


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
