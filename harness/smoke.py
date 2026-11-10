import asyncio
import json
import tempfile
from pathlib import Path

import run_agent
from harness.runtime import run_python
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
        outside = Path(root) / "outside.txt"
        outside.write_text("unchanged", encoding="utf-8")
        script = task_dir / "smoke.py"
        script.write_text(
            "from pathlib import Path\n"
            "try:\n    Path('../outside.txt').write_text('escaped')\n"
            "except PermissionError:\n    print('isolated')\n",
            encoding="utf-8",
        )
        child = run_python(script, cwd=task_dir, timeout=10)
        if not child.passed or child.output.strip() != "isolated" or outside.read_text() != "unchanged":
            raise SystemExit(f"task sandbox smoke check failed: {child.output}")
    if result.answer != "4":
        raise SystemExit(f"smoke check failed: {result}")
    print("smoke check ok:", sorted(run_agent.build_registry()))


if __name__ == "__main__":
    main()
