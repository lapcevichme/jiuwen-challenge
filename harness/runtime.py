import math
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import Thread

from harness.contracts import TaskSpec
from harness.task_io import resolve_inside

_OUTPUT_LIMIT = 64 * 1024


@dataclass(frozen=True)
class TestResult:
    returncode: int | None
    output: str
    timed_out: bool

    @property
    def passed(self) -> bool:
        return not self.timed_out and self.returncode == 0


def _stop_test(process: subprocess.Popen, env: dict[str, str]) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        system_root = env.get("SystemRoot")
        if system_root:
            try:
                subprocess.run(
                    [str(Path(system_root) / "System32/taskkill.exe"), "/F", "/T", "/PID", str(process.pid)],
                    env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
        process.kill()
    process.wait()


def run_python_tests(task: TaskSpec, script: str | Path, *, timeout: float) -> TestResult:
    if task.workspace is None:
        raise ValueError("the task has no workspace")
    test_path = resolve_inside(task.workspace, script)
    if not test_path.is_file() or test_path.suffix != ".py":
        raise ValueError("expected an existing Python test script inside workspace")
    return run_python(test_path, cwd=task.workspace, timeout=timeout)


def run_python(script: Path, *, cwd: Path, timeout: float) -> TestResult:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    test_path = script

    env = {"PYTHONIOENCODING": "utf-8"}
    if os.name == "nt" and "SystemRoot" in os.environ:
        env["SystemRoot"] = os.environ["SystemRoot"]
    with tempfile.TemporaryDirectory(prefix="jiuwen-test-") as scratch:
        env.update(TMP=scratch, TEMP=scratch, TMPDIR=scratch)
        process = subprocess.Popen(
            [sys.executable, "-B", "-s", str(test_path)],
            cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        output = bytearray()

        def drain_output() -> None:
            with process.stdout as pipe:
                while chunk := pipe.read1(8192):
                    output.extend(chunk[:_OUTPUT_LIMIT + 1 - len(output)])

        reader = Thread(target=drain_output, daemon=True)
        reader.start()
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = None
        finally:
            _stop_test(process, env)
            reader.join(timeout=1)
        decoded = output[:_OUTPUT_LIMIT].decode("utf-8", errors="replace")
        if len(output) > _OUTPUT_LIMIT:
            decoded += "\n[test output truncated]"
        return TestResult(returncode, decoded, timed_out)
