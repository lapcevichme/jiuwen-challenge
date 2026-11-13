import asyncio
import json
import tempfile
from pathlib import Path

import run_agent
from harness.runtime import run_python, sandbox_status
from harness.task_io import load_task


def _check_model_path() -> None:
    from harness.agent import JiuwenChat
    from harness.budget import BudgetLedger
    from harness.llm_client import LLMClient

    def transport(url, headers, body, timeout):
        return 200, json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                                "usage": {"prompt_tokens": 20, "completion_tokens": 3}}).encode()

    ledger = BudgetLedger(100, safety_fraction=0)
    chat = JiuwenChat(LLMClient("http://proxy.invalid/v1", "smoke", ledger, transport=transport))
    reply = asyncio.run(chat.chat("qwen3.8-27b-noreason", [{"role": "user", "content": "ping"}],
                                  max_tokens=8, timeout=5))
    if reply.content != "ok" or ledger.spent != 20 + 10 * 3 or ledger.reserved:
        raise SystemExit(f"model path smoke check failed: {reply}, {ledger.summary()}")


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
        status = sandbox_status()
        child = run_python(script, cwd=task_dir, timeout=10)
        if not child.passed:
            raise SystemExit(f"task Python does not run: {child.output}")
        if "landlock abi" in status and (child.output.strip() != "isolated" or outside.read_text() != "unchanged"):
            raise SystemExit(f"task sandbox smoke check failed ({status}): {child.output}")
    if result.answer != "4":
        raise SystemExit(f"smoke check failed: {result}")
    _check_model_path()
    print("smoke check ok:", sorted(run_agent.build_registry()), "| task sandbox:", status)


if __name__ == "__main__":
    main()
