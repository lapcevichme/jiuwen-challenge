import json
import math
import os
import tempfile
from pathlib import Path, PureWindowsPath

from harness.contracts import SolveResult, SolveStatus, TaskSpec


def resolve_inside(root: Path, relative_path: str | Path) -> Path:
    path = Path(relative_path)
    windows_path = PureWindowsPath(relative_path)
    if (
        not path.parts
        or path.anchor
        or windows_path.anchor
        or ".." in path.parts
        or ".." in windows_path.parts
    ):
        raise ValueError("expected a relative path inside the task")
    root = root.resolve(strict=True)
    resolved = (root / path).resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("path escapes its root")
    return resolved


def load_task(task_dir: Path) -> TaskSpec:
    spec_path = resolve_inside(task_dir, "task.json")
    with spec_path.open(encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("task.json must contain an object")
    for key in ("task_id", "task_type", "instruction"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"task.json requires a non-empty {key}")
    reward = data.get("reward")
    if (
        type(reward) not in (int, float)
        or reward < 0
        or (isinstance(reward, float) and not math.isfinite(reward))
    ):
        raise ValueError("reward must be a finite non-negative number")
    check = data.get("check")
    if check is not None and not isinstance(check, str):
        raise ValueError("check must be a string")
    workspace = resolve_inside(task_dir, "workspace")
    if workspace.exists() and not workspace.is_dir():
        raise ValueError("workspace must be a directory")
    return TaskSpec(
        task_id=data["task_id"],
        task_type=data["task_type"],
        instruction=data["instruction"],
        reward=reward,
        task_dir=task_dir.resolve(),
        workspace=workspace if workspace.is_dir() else None,
        check=check,
    )


def write_answer(task: TaskSpec, result: SolveResult) -> None:
    if result.status is not SolveStatus.SOLVED or result.answer is None:
        raise ValueError("only a solved task can have an answer")
    answer = result.answer.strip()
    if task.task_type == "coding" and answer != "DONE":
        raise ValueError("a solved coding task must return DONE")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=task.task_dir,
            prefix=".answer-", delete=False,
        ) as file:
            temp_path = Path(file.name)
            file.write(answer)
        temp_path.chmod(0o644)
        os.replace(temp_path, task.task_dir / "answer.txt")
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
