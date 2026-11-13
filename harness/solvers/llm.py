import ast
import codecs
import asyncio
import contextvars
import json
import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

from harness.budget import BudgetExhausted
from harness.config import number
from harness.contracts import SolveResult, TaskSpec
from harness.llm_client import LLMClient, LLMError, LLMUnavailable
from harness.runtime import run_python, run_python_tests
from harness.scheduler import FallbackRetryLater, FallbackUnavailable
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
_EVENT: contextvars.ContextVar[dict] = contextvars.ContextVar("llm_event")
_DEADLINE: contextvars.ContextVar[float] = contextvars.ContextVar("llm_deadline")
CONTEXT_CHARS = 4000
FULL_CONTEXT_CHARS = number("LLM_FULL_CONTEXT_CHARS", 6000, maximum=100000)
TEXT_SUFFIXES = (".csv", ".tsv", ".md", ".txt", ".json", ".yaml", ".yml")
CSV = (".csv", ".tsv")

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
MATH_SYSTEM = SCRIPT_SYSTEM + (
    " For inverse problems, verify the candidate by substituting it back into the original conditions."
)
CODING_SYSTEM = (
    "Fix the module according to the task and its documented contract. Visible tests are examples, not the "
    "complete specification. Inspect EVERY function, including those whose tests pass. For each, infer its "
    "contract from the task, name, docstring and tests. Mentally try boundary values, repeated values, missing "
    "items and empty inputs. Fix independent logic errors wherever the implementation contradicts the contract. "
    "Preserve existing behavior where the contract is ambiguous. Preserve the API; do not "
    "hardcode test inputs or change tests. Output only complete corrected top-level definitions and any "
    "necessary imports or module constants as Python code. Keep unchanged definitions out of the reply."
)


EXPECTED_OUTPUT = {
    1: {"math": 50, "data_analysis": 85, "coding": 145, "search": 235},
    2: {"math": 500, "data_analysis": 180, "coding": 500, "search": 330},
}
CHARS_PER_INPUT_TOKEN = 3.0


@dataclass(frozen=True)
class Models:
    fast: str = "qwen3.8-27b-noreason"
    strong: str = "qwen3.8-27b"
    fast_max_tokens: int = 800
    strong_max_tokens: int = 4000

    def __post_init__(self):
        allowed = {"qwen3.8-27b", "qwen3.8-27b-noreason", "qwen3.6-35b-a3b", "qwen3.6-35b-a3b-noreason"}
        if self.fast not in allowed or self.strong not in allowed:
            raise ValueError("LLM_MODEL and LLM_STRONG_MODEL must be competition-approved models")


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
    files = sorted(path for path in workspace.rglob("*") if not path.is_symlink() and path.is_file()
                   and path.resolve().is_relative_to(workspace))
    names = [str(path.relative_to(workspace)) for path in files]
    parts = [f"Files ({len(names)}): " + ", ".join(names[:60]) + (", ..." if len(names) > 60 else "")]
    try:
        prose = [path for path in files if path.suffix.lower() in TEXT_SUFFIXES and path.suffix.lower() not in CSV]
        full = (len(files) <= 60 and all(path.suffix.lower() in TEXT_SUFFIXES for path in files)
                and sum(path.stat().st_size for path in prose) <= FULL_CONTEXT_CHARS)
    except OSError:
        full = False
    kinds: set[str] = set()
    for path, name in zip(files, names):
        suffix = path.suffix.lower()
        try:
            if suffix in CSV:
                with path.open(encoding="utf-8-sig", errors="replace") as file:
                    head = [next(file, "").rstrip("\n") for _ in range(3)]
                with path.open(encoding="utf-8", errors="replace") as file:
                    rows = sum(1 for _ in file) - 1
                parts.append(f"{name} ({rows} data rows), first lines:\n" + "\n".join(head))
            elif suffix in TEXT_SUFFIXES and full:
                parts.append(f"--- {name}\n" + path.read_text(encoding="utf-8-sig", errors="replace").strip())
            elif suffix in TEXT_SUFFIXES:
                kind = path.parent.as_posix() + "/" + path.stem.split("_")[0] + suffix
                if kind not in kinds and len(kinds) < 3:
                    kinds.add(kind)
                    with path.open(encoding="utf-8", errors="replace") as file:
                        parts.append(f"Example {name}:\n" + file.read(400))
        except OSError:
            continue
    return "\n\n".join(parts)[:CONTEXT_CHARS + (FULL_CONTEXT_CHARS if full else 0)]


_NO_VALUE = {"none", "null", "nan", "undefined", "n/a", "[]", "{}", "''", '""'}
_LOOKUP_FAILURE = re.compile(
    r"(?:(?:person|company|employer|record|entity|value|answer|file|data|match|field|column|key)\s+)?not found[.!]?|"
    r"no (?:matching )?(?:person|records?|entities|values?|answers?|files?|data|matches|results?)(?: found)?[.!]?",
    re.IGNORECASE,
)
_SEARCH_LITERAL = re.compile(r"[^\W_][\w .’'&/-]{0,119}")
_TABLE_SEPARATOR = re.compile(r"\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*")


def normalize_answer(output: str, check: str | None) -> str | None:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return None
    answer = lines[-1]
    if len(answer) > 200 or answer.casefold() in _NO_VALUE or _LOOKUP_FAILURE.fullmatch(answer):
        return None
    if check == "exact_int":
        match = re.fullmatch(r"([+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+))(?:\.0+)?", answer)
        return str(int(match[1].replace(",", ""))) if match else None
    return answer


def _grounded_search_answer(reply: str, context: str) -> str | None:
    text = reply.strip()
    if not _SEARCH_LITERAL.fullmatch(text) or normalize_answer(text, "exact_str") is None:
        return None
    lines = context.splitlines()
    for index, line in enumerate(lines[1:], 1):
        if line.startswith(("--- ", "Example ")):
            continue
        line = line.strip()
        if line.startswith("|"):
            if index + 1 < len(lines) and _TABLE_SEPARATOR.fullmatch(lines[index + 1]):
                continue
            values = line.split("|")[1:]
        else:
            _, separator, value = line.partition(":")
            values = [value] if separator else []
        for value in values:
            literal = value.strip(" *`")
            if literal.casefold() == text.casefold():
                return literal
    return None


def _scratch_copy(task: TaskSpec, scratch: Path) -> Path:
    root = scratch / "workspace"
    if task.workspace is not None:
        shutil.copytree(task.workspace, root, symlinks=True)
    else:
        root.mkdir()
    return root


async def _execute(function, *args, **kwargs):
    kwargs["timeout"] = max(0.01, min(kwargs["timeout"], _DEADLINE.get() - time.monotonic()))
    operation = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(operation)
    except asyncio.CancelledError:
        await operation
        raise


def _splice(module: str, patch: str) -> str | None:
    try:
        tree, new = ast.parse(module), ast.parse(patch)
    except SyntaxError:
        return None
    lines = module.splitlines(keepends=True)
    patch_lines = patch.splitlines(keepends=True)
    def names(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return [node.name]
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        return [target.id for target in targets] if all(isinstance(target, ast.Name) for target in targets) else []

    spans = {}
    for node in tree.body:
        for name in names(node):
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            spans[name] = (start - 1, node.end_lineno)
    replacements, imports, futures, globals_, extra = [], [], [], [], []
    changed = set()
    for node in new.body:
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        text = "".join(patch_lines[start - 1:node.end_lineno]).rstrip() + "\n"
        identifiers = names(node)
        if identifiers:
            if changed.intersection(identifiers):
                return None
            changed.update(identifiers)
            existing = {spans[name] for name in identifiers if name in spans}
            if len(existing) > 1:
                return None
            if existing:
                span = existing.pop()
                if any(span == old for old, _ in replacements):
                    return None
                replacements.append((span, text))
            else:
                (globals_ if isinstance(node, (ast.Assign, ast.AnnAssign)) else extra).append(text)
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and text not in module:
            (futures if isinstance(node, ast.ImportFrom) and node.module == "__future__" else imports).append(text)
    if not replacements and not extra and not globals_ and not imports and not futures:
        return None
    for (start, end), text in sorted(replacements, reverse=True):
        lines[start:end] = [text]
    result = "".join(lines)
    if extra:
        result = result.rstrip() + "\n\n\n" + "\n\n".join(extra)
    try:
        updated = ast.parse(result)
        insert = 0
        for index, node in enumerate(updated.body):
            if ((index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                 and isinstance(node.value.value, str))
                    or isinstance(node, ast.ImportFrom) and node.module == "__future__"):
                insert = node.end_lineno
            else:
                break
        lines = result.splitlines(keepends=True)
        if insert and not lines[insert - 1].endswith("\n"):
            lines[insert - 1] += "\n"
        lines[insert:insert] = futures + imports
        result = "".join(lines)
        if globals_:
            setup_end = 0
            for index, node in enumerate(ast.parse(result).body):
                if (isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign))
                        or index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str)):
                    setup_end = node.end_lineno
                else:
                    break
            lines = result.splitlines(keepends=True)
            if setup_end and not lines[setup_end - 1].endswith("\n"):
                lines[setup_end - 1] += "\n"
            lines[setup_end:setup_end] = globals_
        result = "".join(lines)
        compile(result, "<patched-module>", "exec")
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


_FILE_MARKER = re.compile(r"^#\s*file:\s*([^\r\n]+?)\s*$", re.MULTILINE | re.IGNORECASE)


def _coding_sections(content: str, targets: list[str]) -> dict[str, str]:
    if len(targets) == 1:
        return {targets[0]: extract_code(content)}
    code = re.sub(r"^```(?:python|py)?[ \t]*$", "", content, flags=re.MULTILINE)
    parts = _FILE_MARKER.split(code)
    sections = {}
    for name, code in zip(parts[1::2], parts[2::2]):
        if name not in targets or name in sections:
            return {}
        sections[name] = code.strip()
    return sections


def _failure_summary(output: str, limit: int = 1500) -> str:
    keep = [
        line for line in output.splitlines()
        if line.startswith(("FAIL:", "ERROR:", "Ran ", "FAILED", "OK")) or "Error" in line or "assert" in line
    ]
    return "\n".join(keep)[:limit]


class LLMFallback:
    def __init__(self, client: LLMClient, models: Models = Models(), *, trust_fast_unsolvable: bool = True,
                 adaptive_output: bool = True):
        self.client = client
        self.models = models
        self.trust_fast_unsolvable = trust_fast_unsolvable
        self.adaptive_output = adaptive_output
        self._previous_code: dict[Path, str] = {}
        self._contexts: dict[Path, str] = {}

    def _context(self, task: TaskSpec) -> str:
        if task.task_dir not in self._contexts:
            self._contexts[task.task_dir] = workspace_context(task.workspace)
        return self._contexts[task.task_dir]

    def _feedback(self, task: TaskSpec, feedback: str) -> str:
        if not feedback:
            return ""
        code = self._previous_code.get(task.task_dir, "")
        return (f"\n\nPrevious attempt failed: {feedback}"
                + (f"\nPrevious code (may be incomplete):\n{code}" if code else ""))

    def estimate(self, task: TaskSpec, attempt: int) -> float:
        if task.task_type == "coding":
            directory = task.workspace or task.task_dir
            chars = len(task.instruction) + sum(path.stat().st_size for path in directory.rglob("*.py")
                        if not path.is_symlink() and path.is_file() and path.resolve().is_relative_to(directory))
        else:
            chars = len(task.instruction) + len(self._context(task))
        output = EXPECTED_OUTPUT[min(attempt, 2)].get(task.task_type, 300)
        return (len(SCRIPT_SYSTEM) + chars) / CHARS_PER_INPUT_TOKEN / 10 + output

    async def __call__(self, task: TaskSpec, previous: SolveResult, *, attempt: int, timeout: float) -> SolveResult:
        strong = attempt >= 2
        model = self.models.strong if strong else self.models.fast
        max_tokens = self.models.strong_max_tokens if strong else self.models.fast_max_tokens
        feedback = previous.detail[:400] if attempt >= 2 else ""
        _EVENT.set({"task": task.task_id, "type": task.task_type, "attempt": attempt, "model": model})
        _DEADLINE.set(time.monotonic() + timeout)
        try:
            if task.task_type == "coding":
                result = await self._coding(task, model, max_tokens, timeout, feedback)
            else:
                result = await self._script(task, model, max_tokens, timeout, feedback, final=strong)
        except BudgetExhausted as error:
            trace({**_EVENT.get(), "status": "BUDGET_SKIPPED", "detail": str(error)})
            raise
        except LLMUnavailable as error:
            trace({**_EVENT.get(), "status": "UNAVAILABLE", "detail": str(error)[:300]})
            raise FallbackUnavailable(str(error)) from error
        except LLMError as error:
            if error.transient:
                trace({**_EVENT.get(), "status": "RETRY_LATER", "detail": str(error)[:300]})
                raise FallbackRetryLater(str(error), retry_after=error.retry_after) from error
            result = SolveResult.unsupported(f"LLM request failed: {error}")
        trace({**_EVENT.get(), "status": result.status.value, "answer": result.answer, "detail": result.detail[:500]})
        if result.answer is not None or strong:
            self._previous_code.pop(task.task_dir, None)
        return result

    async def _chat(self, model: str, system: str, prompt: str, max_tokens: int, timeout: float, label: str):
        reserve = 2 * SCRIPT_TIMEOUT if _EVENT.get()["type"] == "coding" else SCRIPT_TIMEOUT
        timeout = min(timeout, _DEADLINE.get() - time.monotonic() - reserve)
        if timeout <= 0:
            raise LLMError("no time left for generation and verification", transient=True)
        reply = await self.client.chat(
            model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_tokens=max_tokens, timeout=timeout, label=label,
            min_tokens=min(max_tokens, 256 if model.endswith("-noreason") else 768) if self.adaptive_output else None,
        )
        _EVENT.get().update(prompt=prompt, reply=reply.content, cost=reply.cost, input_tokens=reply.prompt_tokens,
                           output_tokens=reply.completion_tokens, seconds=round(reply.seconds, 2))
        return reply

    async def _script(self, task, model, max_tokens, timeout, feedback, *, final) -> SolveResult:
        prompt = question(task.instruction)
        context = self._context(task)
        if context:
            prompt += "\n\n" + context
        prompt += self._feedback(task, feedback)
        system = MATH_SYSTEM if task.task_type == "math" else SCRIPT_SYSTEM
        reply = await self._chat(model, system, prompt, max_tokens, max(1.0, timeout - SCRIPT_TIMEOUT), task.task_id)
        self._previous_code[task.task_dir] = extract_code(reply.content)[:6000]
        if reply.finish_reason == "length":
            return SolveResult.unsupported(f"{model} output truncated before completion")
        if reply.content.strip().strip("`").strip().upper().rstrip(".") == "UNSOLVABLE":
            if final or (self.trust_fast_unsolvable and task.task_type == "math" and task.workspace is None):
                return SolveResult.unsolvable(f"{model} finds information missing for an exact answer")
            return SolveResult.unsupported(f"{model} replied UNSOLVABLE")
        if task.task_type == "search" and task.check == "exact_str":
            answer = _grounded_search_answer(reply.content, context)
            if answer is not None:
                return SolveResult.solved(answer, f"{model} returned a value shown in the search context")
        code = extract_code(reply.content)
        if not code:
            return SolveResult.unsupported(f"{model} returned no code")
        with tempfile.TemporaryDirectory(prefix="jiuwen-llm-") as scratch:
            root = _scratch_copy(task, Path(scratch))
            script = Path(scratch) / SCRIPT_NAME
            script.write_text(code, encoding="utf-8")
            result = await _execute(run_python, script, cwd=root, timeout=SCRIPT_TIMEOUT)
        _EVENT.get()["script_output"] = result.output[-500:]
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
        script, targets = located
        paths = {name: resolve_inside(task.workspace, name) for name in targets}
        originals = {name: path.read_bytes() for name, path in paths.items()}
        sources = {name: data.decode("utf-8-sig") for name, data in originals.items()}
        script_path = resolve_inside(task.workspace, script)
        tests = script_path.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="jiuwen-llm-") as scratch:
            root = _scratch_copy(task, Path(scratch))
            baseline = await _execute(run_python_tests, replace(task, workspace=root), script,
                                      timeout=SCRIPT_TIMEOUT)
            if baseline.passed:
                return SolveResult.solved("DONE", "original tests already pass")
            modules = "\n\n".join(f"Module {name}:\n{source}" for name, source in sources.items())
            prompt = (
                f"Task:\n{question(task.instruction)}\n\n{modules}\n\nTests {script}:\n{tests}\n\n"
                f"Failures:\n{_failure_summary(baseline.output)}"
            )
            if len(targets) > 1:
                prompt += ("\nGroup corrected definitions by module. Start each group with a line "
                           "'# file: <exact filename>' followed by code. Omit unchanged modules.")
            prompt += self._feedback(task, feedback)
            reply = await self._chat(model, CODING_SYSTEM, prompt, max_tokens,
                                     max(1.0, timeout - 2 * SCRIPT_TIMEOUT), task.task_id)
            self._previous_code[task.task_dir] = reply.content[:6000]
            if reply.finish_reason == "length":
                return SolveResult.unsupported(f"{model} repair truncated before completion")
            sections = _coding_sections(reply.content, targets)
            if not sections:
                return SolveResult.unsupported("expected '# file: <exact filename>' for each changed module")
            patches = {}
            for name, code in sections.items():
                patched = _splice(sources[name], code)
                if patched is None:
                    return SolveResult.unsupported(f"{model} supplied no usable definitions for {name}")
                gaming = _games_tests(sources[name], patched)
                if gaming:
                    return SolveResult.unsupported(f"{model} {gaming}; rejected as test gaming")
                encoding = "utf-8-sig" if originals[name].startswith(codecs.BOM_UTF8) else "utf-8"
                patches[name] = patched.encode(encoding)
                resolve_inside(root, name).write_bytes(patches[name])
            check = await _execute(run_python_tests, replace(task, workspace=root), script,
                                   timeout=SCRIPT_TIMEOUT)
            if not check.passed:
                return SolveResult.unsupported(f"{model} fix fails tests: {_failure_summary(check.output, 300)}")

        written = []

        def restore():
            for name in reversed(written):
                _write_atomic(paths[name], originals[name])
            written.clear()

        def commit():
            digest = _digest(script_path)
            accepted = False
            try:
                for name, data in patches.items():
                    _write_atomic(paths[name], data)
                    written.append(name)
                verified = run_python_tests(task, script, timeout=SCRIPT_TIMEOUT)
                accepted = verified.passed and _digest(script_path) == digest
                return accepted
            finally:
                if not accepted:
                    restore()

        operation = asyncio.create_task(asyncio.to_thread(commit))
        try:
            accepted = await asyncio.shield(operation)
        except asyncio.CancelledError:
            await operation
            restore()
            raise
        if not accepted:
            return SolveResult.unsupported(f"{model} fix did not verify in place; restored")
        return SolveResult.solved("DONE", f"{model} fix, cost {reply.cost:.1f} tokens")
