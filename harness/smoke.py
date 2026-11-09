import asyncio
import json
import tempfile
from pathlib import Path

import run_agent
from harness.task_io import load_task


def main() -> None:
    run_agent._configure_sdk()
    from harness.workflow import SolverWorkflow

    with tempfile.TemporaryDirectory(prefix="jiuwen-smoke-") as root:
        task_dir = Path(root) / "math_smoke"
        task_dir.mkdir()
        spec = {
            "task_id": "math_smoke", "task_type": "math", "reward": 1, "check": "exact_int",
            "instruction": "A warehouse starts with 10 boxes. It ships 3 boxes per day for 2 days. "
                           "How many boxes does it have at the end?",
        }
        (task_dir / "task.json").write_text(json.dumps(spec), encoding="utf-8")
        result = asyncio.run(SolverWorkflow(run_agent.build_registry()).solve(load_task(task_dir)))
    if result.answer != "4":
        raise SystemExit(f"smoke check failed: {result}")
    print("smoke check ok:", sorted(run_agent.build_registry()))


if __name__ == "__main__":
    main()
