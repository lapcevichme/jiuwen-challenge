import ast
import json
import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

from harness.contracts import SolveResult, TaskSpec
from harness.llm_client import LLMClient, LLMError, LLMUnavailable
from harness.runtime import run_python, run_python_tests
from harness.scheduler import FallbackUnavailable
from harness.solvers.coding import _digest, _locate, _write_atomic
from harness.task_io import resolve_inside

TRAJECTORY_ENV = "HARNESS_TRAJECTORIES"
_TRACE_LOCK = threading.Lock()


def trace(event: dict) -> None:
    directory = os.environ.get(TRAJECTORY_ENV)
    if not directory:
        return
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        line = json.dumps({"time": round(time.time(), 3), **event}, ensure_ascii=False)
        with _TRACE_LOCK, (Path(directory) / "trajectory.jsonl").open("a", encoding="utf-8") as file:
            file.write(line + "\n")
    except OSError:
        pass


SCRIPT_NAME = "llm_solution.py"
SCRIPT_TIMEOUT = 10.0
CONTEXT_CHARS = 4000

_BOILERPLATE = re.compile(
    r"\s*(?:Compute it with the bash tool[^.]*\.|Use bash with python3 to compute it\.|"
    r"Use the file tools or bash[^.]*\.|Work only inside the current directory\.|When you are done, write.*$)",
    re.IGNORECASE | re.DOTALL,
)
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

SCRIPT_SYSTEM = (
    "If the task or its files lack something needed for an exact answer (a quantity given only vaguely such as "
    "'some' or 'several', a missing file, column, record or fact), reply with only the word UNSOLVABLE. "
    "Otherwise write a minimal Python 3 script (standard library only, no comments) that solves the task: input "
    "files are in the current directory; handle every sentence in order and apply rounding exactly as stated; "
    "print only the final answer. Output only code."
)
CODING_SYSTEM = (
    "Fix the bugs in the module so that the tests pass; do not change the tests. "
    "Output only the corrected top-level functions or classes as Python code, complete definitions, no comments."
)


@dataclass(frozen=True)
class Models:
    fast: str = "qwen3.8-27b-noreason"
    strong: str = "qwen3.8-27b"
    fast_max_tokens: int = 800
    strong_max_tokens: int = 4000


def question(instruction: str) -> str:
    return _BOILERPLATE.sub("", instruction).strip()


def extract_code(content: str) -> str:
    fenced = _FENCE.findall(content)
    if fenced:
        return max(fenced, key=len).strip()
    return re.sub(r"^```\w*\s*|```\s*$", "", content.strip(), flags=re.MULTILINE).strip()


def workspace_context(workspace: Path | None) -> str:
    if workspace is None:
        return ""
    files = sorted(path for path in workspace.rglob("*") if path.is_file())
    names = [str(path.relative_to(workspace)) for path in files]
    parts = [f"Files ({len(names)}): " + ", ".join(names[:60]) + (", ..." if len(names) > 60 else "")]
    kinds: set[str] = set()
    for path, name in zip(files, names):
        suffix = path.suffix.lower()
        try:
            if suffix in (".csv", ".tsv"):
                with path.open(encoding="utf-8-sig", errors="replace") as file:
                    head = [next(file, "").rstrip("\n") for _ in range(3)]
                with path.open(encoding="utf-8", errors="replace") as file:
                    rows = sum(1 for _ in file) - 1
                parts.append(f"{name} ({rows} data rows), first lines:\n" + "\n".join(head))
            elif suffix in (".md", ".txt", ".json", ".yaml", ".yml"):
                kind = path.parent.as_posix() + "/" + path.stem.split("_")[0] + suffix
                if kind not in kinds and len(kinds) < 3:
                    kinds.add(kind)
                    parts.append(f"Example {name}:\n" + path.read_text(encoding="utf-8", errors="replace")[:400])
        except OSError:
            continue
    return "\n\n".join(parts)[:CONTEXT_CHARS]


def normalize_answer(output: str, check: str | None) -> str | None:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return None
    answer = lines[-1]
    if len(answer) > 200:
        return None
    if check == "exact_int":
        match = re.fullmatch(r"(-?[0-9]+)(?:\.0+)?", answer.replace(",", ""))
        return match[1] if match else None
    return answer


def _scratch_copy(task: TaskSpec, scratch: Path) -> Path:
    root = scratch / "workspace"
    if task.workspace is not None:
        shutil.copytree(task.workspace, root, symlinks=True)
    else:
        root.mkdir()
    return root


def _splice(module: str, patch: str) -> str | None:
    try:
        tree, new = ast.parse(module), ast.parse(patch)
    except SyntaxError:
        return None
    lines = module.splitlines(keepends=True)
    patch_lines = patch.splitlines(keepends=True)
    spans = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
            spans[node.name] = (start - 1, node.end_lineno)
    replacements, imports, extra = [], [], []
    for node in new.body:
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        text = "".join(patch_lines[start - 1:node.end_lineno]).rstrip() + "\n"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in spans:
                replacements.append((spans[node.name], text))
            else:
                extra.append(text)
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and text not in module:
            imports.append(text)
    if not replacements and not extra:
        return None
    for (start, end), text in sorted(replacements, reverse=True):
        lines[start:end] = [text]
    result = "".join(imports) + "".join(lines)
    if extra:
        result = result.rstrip() + "\n\n\n" + "\n\n".join(extra)
    try:
        ast.parse(result)
    except SyntaxError:
        return None
    return result


_DUNDER_GAMING = {"__eq__", "__ne__", "__hash__", "__bool__", "__lt__", "__le__", "__gt__", "__ge__",
                  "__contains__", "__len__", "__getattr__", "__getattribute__", "__instancecheck__"}


def _games_tests(original: str, patched: str) -> str | None:
    def shape(source: str) -> tuple[set[str], set[str]]:
        tree = ast.parse(source)
        classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        dunders = {node.name for node in ast.walk(tree)
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _DUNDER_GAMING}
        return classes, dunders
    old_classes, old_dunders = shape(original)
    new_classes, new_dunders = shape(patched)
    added = sorted((new_classes - old_classes) | (new_dunders - old_dunders))
    return f"fix adds {', '.join(added)}" if added else None


def _failure_summary(output: str, limit: int = 1500) -> str:
    keep = [
        line for line in output.splitlines()
        if line.startswith(("FAIL:", "ERROR:", "Ran ", "FAILED", "OK")) or "Error" in line or "assert" in line
    ]
    return "\n".join(keep)[:limit]


class LLMFallback:
    def __init__(self, client: LLMClient, models: Models = Models()):
        self.client = client
        self.models = models
        self._event: dict = {}

    async def __call__(self, task: TaskSpec, previous: SolveResult, *, attempt: int, timeout: float) -> SolveResult:
        strong = attempt >= 2
        model = self.models.strong if strong else self.models.fast
        max_tokens = self.models.strong_max_tokens if strong else self.models.fast_max_tokens
        feedback = previous.detail[:400] if attempt >= 2 else ""
        self._event = {"task": task.task_id, "type": task.task_type, "attempt": attempt, "model": model}
        try:
            if task.task_type == "coding":
                result = await self._coding(task, model, max_tokens, timeout, feedback)
            else:
                result = await self._script(task, model, max_tokens, timeout, feedback, final=strong)
        except LLMUnavailable as error:
            trace({**self._event, "status": "UNAVAILABLE", "detail": str(error)[:300]})
            raise FallbackUnavailable(str(error)) from error
        except LLMError as error:
            result = SolveResult.unsupported(f"LLM request failed: {error}")
        trace({**self._event, "status": result.status.value, "answer": result.answer, "detail": result.detail[:500]})
        return result

    async def _chat(self, model: str, system: str, prompt: str, max_tokens: int, timeout: float, label: str):
        reply = await self.client.chat(
            model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_tokens=max_tokens, timeout=timeout, label=label,
        )
        self._event.update(prompt=prompt, reply=reply.content, cost=reply.cost, input_tokens=reply.prompt_tokens,
                           output_tokens=reply.completion_tokens, seconds=round(reply.seconds, 2))
        return reply

    async def _script(self, task, model, max_tokens, timeout, feedback, *, final) -> SolveResult:
        prompt = question(task.instruction)
        context = workspace_context(task.workspace)
        if context:
            prompt += "\n\n" + context
        if feedback:
            prompt += f"\n\nA previous attempt failed: {feedback}"
        reply = await self._chat(model, SCRIPT_SYSTEM, prompt, max_tokens, max(1.0, timeout - SCRIPT_TIMEOUT),
                                 task.task_id)
        if reply.content.strip().strip("`").strip().upper().rstrip(".") == "UNSOLVABLE":
            if final:
                return SolveResult.unsolvable(f"{model} finds information missing for an exact answer")
            return SolveResult.unsupported(f"{model} replied UNSOLVABLE")
        code = extract_code(reply.content)
        if not code:
            return SolveResult.unsupported(f"{model} returned no code")
        with tempfile.TemporaryDirectory(prefix="jiuwen-llm-") as scratch:
            root = _scratch_copy(task, Path(scratch))
            script = Path(scratch) / SCRIPT_NAME
            script.write_text(code, encoding="utf-8")
            result = run_python(script, cwd=root, timeout=SCRIPT_TIMEOUT)
        self._event["script_output"] = result.output[-500:]
        if not result.passed:
            reason = "timed out" if result.timed_out else _failure_summary(result.output, 300) or "failed"
            return SolveResult.unsupported(f"{model} script {reason}")
        if result.output.strip().splitlines()[-1:] == ["UNSOLVABLE"]:
            if final:
                return SolveResult.unsolvable(f"{model} script reports the data cannot answer the question")
            return SolveResult.unsupported(f"{model} script printed UNSOLVABLE")
        answer = normalize_answer(result.output, task.check)
        if answer is None:
            return SolveResult.unsupported(f"{model} script output is not a valid answer: {result.output[-200:]!r}")
        return SolveResult.solved(answer, f"{model} script, cost {reply.cost:.1f} tokens")

    async def _coding(self, task, model, max_tokens, timeout, feedback) -> SolveResult:
        task = task if task.workspace is not None else replace(task, workspace=task.task_dir)
        located = _locate(task, task.workspace)
        if isinstance(located, SolveResult):
            return located
        script, target = located
        target_path = resolve_inside(task.workspace, target)
        script_path = resolve_inside(task.workspace, script)
        original = target_path.read_text(encoding="utf-8")
        tests = script_path.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="jiuwen-llm-") as scratch:
            root = _scratch_copy(task, Path(scratch))
            baseline = run_python_tests(replace(task, workspace=root), script, timeout=SCRIPT_TIMEOUT)
            if baseline.passed:
                return SolveResult.solved("DONE", "original tests already pass")
            prompt = (
                f"Module {target}:\n{original}\n\nTests {script}:\n{tests}\n\n"
                f"Failures:\n{_failure_summary(baseline.output)}"
            )
            if feedback:
                prompt += f"\n\nA previous fix failed: {feedback}"
            reply = await self._chat(model, CODING_SYSTEM, prompt, max_tokens,
                                     max(1.0, timeout - 2 * SCRIPT_TIMEOUT), task.task_id)
            patched = _splice(original, extract_code(reply.content))
            if patched is None:
                return SolveResult.unsupported(f"{model} reply has no usable definitions")
            gaming = _games_tests(original, patched)
            if gaming:
                return SolveResult.unsupported(f"{model} {gaming}; rejected as test gaming")
            resolve_inside(root, target).write_text(patched, encoding="utf-8")
            check = run_python_tests(replace(task, workspace=root), script, timeout=SCRIPT_TIMEOUT)
            if not check.passed:
                return SolveResult.unsupported(f"{model} fix fails tests: {_failure_summary(check.output, 300)}")

        digest = _digest(script_path)
        _write_atomic(target_path, patched.encode("utf-8"))
        verified = run_python_tests(task, script, timeout=SCRIPT_TIMEOUT)
        if not verified.passed or _digest(script_path) != digest:
            _write_atomic(target_path, original.encode("utf-8"))
            return SolveResult.unsupported(f"{model} fix did not verify in place; restored")
        return SolveResult.solved("DONE", f"{model} fix, cost {reply.cost:.1f} tokens")
