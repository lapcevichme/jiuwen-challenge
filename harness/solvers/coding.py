import ast
import codecs
import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from queue import SimpleQueue

from harness.contracts import SolveResult, TaskSpec
from harness.runtime import TestResult, run_python, run_python_tests
from harness.solvers import shape_repairs
from harness.solvers.defaults import isolated_defaults
from harness.task_io import resolve_inside

TIME_LIMIT = 60.0
PRECHECK_TIMEOUT = 2.0
MAX_ROUNDS = 8
MAX_PAIR_RUNS = 400
MAX_PROBES = 40
RIVAL_WINDOW = 1
LOW_PLAUSIBILITY = 3
WORKERS = max(1, min(4, os.cpu_count() or 1))

_SCRIPT = re.compile(r"\bRun `(?:python3?\s+)?(?P<script>[\w./-]+\.py)`")
_MODULE_NAME = r"[\w./-]+\.py"
_TARGET = re.compile(
    rf"\bFix (?:the )?bugs? in (?P<targets>`?{_MODULE_NAME}`?"
    rf"(?:(?:,\s*(?:and\s+)?|\s+and\s+)`?{_MODULE_NAME}`?)*)", re.IGNORECASE,
)
_FAILURE = re.compile(r"^(?:FAIL|ERROR): (\S+) \(([^)\s]*)\)\s*$", re.MULTILINE)
_RAN = re.compile(r"^Ran [0-9]+ tests? in ", re.MULTILINE)

Failures = frozenset[str] | None

_ARITH_FAMILIES = (
    (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow),
    (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift),
)
_ARITH_PARTNERS = {
    ast.Add: {ast.Sub}, ast.Sub: {ast.Add}, ast.Mult: {ast.Div}, ast.Div: {ast.Mult, ast.FloorDiv},
    ast.FloorDiv: {ast.Div, ast.Mod}, ast.Mod: {ast.FloorDiv}, ast.Pow: {ast.Mult},
    ast.BitAnd: {ast.BitOr}, ast.BitOr: {ast.BitAnd}, ast.LShift: {ast.RShift}, ast.RShift: {ast.LShift},
}
_COMPARE_FAMILIES = (
    (ast.Lt, ast.LtE, ast.Gt, ast.GtE), (ast.Eq, ast.NotEq), (ast.In, ast.NotIn), (ast.Is, ast.IsNot),
)
_COMPARE_PARTNERS = {
    ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE, ast.GtE: ast.LtE, ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq, ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is,
}
_BUILTIN_FAMILIES = (("min", "max"), ("any", "all"), ("sorted", "reversed"))
_METHOD_FAMILIES = (
    ("lower", "upper"), ("startswith", "endswith"), ("find", "rfind"), ("index", "rindex"),
    ("split", "rsplit"), ("strip", "lstrip", "rstrip"), ("keys", "values", "items"),
)
_UNWRAP = {"sorted", "reversed", "list", "set", "tuple", "abs", "int", "float", "str", "round"}
_STRINGS = ("", " ", ",", ", ", "\n")
_DELETABLE = (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Expr, ast.If, ast.Break, ast.Continue)
_LOOSE = (ast.BinOp, ast.BoolOp, ast.Compare, ast.UnaryOp, ast.IfExp, ast.Lambda, ast.NamedExpr)


@dataclass(frozen=True)
class _Edit:
    start: int
    end: int
    text: bytes
    priority: int
    unit: str
    label: str


def _failures(result: TestResult) -> Failures:
    if result.passed:
        return frozenset()
    if result.timed_out or not _RAN.search(result.output):
        return None
    found = frozenset(f"{name} ({where})" for name, where in _FAILURE.findall(result.output))
    return found or None


def _better(new: Failures, old: Failures) -> bool:
    if new is None:
        return False
    return old is None or new < old


def _rank(failures: Failures) -> float:
    return float("inf") if failures is None else len(failures)


def _apply(source: bytes, edits: Iterable[_Edit]) -> bytes:
    for edit in sorted(edits, key=lambda item: item.start, reverse=True):
        source = source[:edit.start] + edit.text + source[edit.end:]
    return source


def _disjoint(edits: Iterable[_Edit]) -> bool:
    spans = sorted((edit.start, edit.end) for edit in edits)
    return all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))


def _with(node: ast.AST, **changes) -> ast.AST:
    clone = copy.copy(node)
    for name, value in changes.items():
        setattr(clone, name, value)
    return clone


def _shifts(node: ast.expr) -> Iterator[ast.expr]:
    unit_offset = isinstance(node, ast.BinOp) and isinstance(node.right, ast.Constant) and node.right.value == 1
    for op in (ast.Add, ast.Sub):
        cancels = unit_offset and isinstance(node.op, ast.Sub if op is ast.Add else ast.Add)
        if not cancels:
            yield ast.BinOp(node, op(), ast.Constant(1))


_PRECEDENCE = (
    (ast.Add, ast.Sub), (ast.Mult, ast.Div, ast.FloorDiv, ast.Mod), (ast.BitAnd,), (ast.BitOr,), (ast.BitXor,),
    (ast.LShift, ast.RShift), (ast.Pow,),
)


def _same_precedence(original: ast.AST, replacement: ast.AST) -> bool:
    if isinstance(original, ast.Compare) and isinstance(replacement, ast.Compare):
        return True
    if isinstance(original, ast.BinOp) and isinstance(replacement, ast.BinOp):
        return any(isinstance(original.op, group) and isinstance(replacement.op, group) for group in _PRECEDENCE)
    return False


def _text(node: ast.AST, original: ast.AST) -> bytes:
    text = ast.unparse(node)
    negative = isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and node.value < 0
    if (isinstance(node, _LOOSE) and not _same_precedence(original, node)) or negative:
        text = f"({text})"
    return text.encode("utf-8")


def _operator_swaps(op: ast.AST) -> Iterator[tuple[ast.AST, int]]:
    kind = type(op)
    for family in _ARITH_FAMILIES:
        if kind in family:
            for other in family:
                if other is not kind:
                    yield other(), 0 if other in _ARITH_PARTNERS.get(kind, ()) else 1


def _variants(node: ast.AST, parent: ast.AST | None) -> Iterator[tuple[ast.AST | bytes, int]]:
    if isinstance(node, ast.BinOp):
        identity = isinstance(node.right, ast.Constant) and node.right.value == 1
        for op, priority in _operator_swaps(node.op):
            if not (identity and isinstance(op, (ast.Mult, ast.Div, ast.FloorDiv, ast.Pow))):
                yield _with(node, op=op), priority
        if isinstance(node.op, (ast.Add, ast.Sub)) and isinstance(node.right, ast.Constant):
            yield node.left, 2
        if isinstance(node.op, ast.Add) and isinstance(node.left, ast.Constant):
            yield node.right, 2
        if isinstance(node.op, (ast.Sub, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
            yield _with(node, left=node.right, right=node.left), 3
    elif isinstance(node, ast.AugAssign):
        for op, priority in _operator_swaps(node.op):
            yield _with(node, op=op), priority
    elif isinstance(node, ast.Compare):
        for index, op in enumerate(node.ops):
            for family in _COMPARE_FAMILIES:
                if type(op) in family:
                    for other in family:
                        if other is not type(op):
                            ops = list(node.ops)
                            ops[index] = other()
                            yield _with(node, ops=ops), 0 if _COMPARE_PARTNERS[type(op)] is other else 1
    elif isinstance(node, ast.BoolOp):
        yield _with(node, op=ast.Or() if isinstance(node.op, ast.And) else ast.And()), 0
    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            yield node.operand, 1
        elif isinstance(node.op, ast.USub):
            yield node.operand, 2
    elif isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool):
            yield ast.Constant(not value), 0
        elif isinstance(value, int):
            for other in dict.fromkeys((value + 1, value - 1, 0, 1)):
                if other != value:
                    yield ast.Constant(other), 2
        elif isinstance(value, str) and value in _STRINGS:
            for other in _STRINGS:
                if other != value:
                    yield ast.Constant(other), 2
    elif isinstance(node, ast.Call):
        yield from _call_variants(node)
    elif isinstance(node, ast.Subscript):
        yield from _subscript_variants(node)

    if isinstance(parent, (ast.If, ast.While, ast.IfExp, ast.Assert)) and node is parent.test:
        if not isinstance(node, (ast.Compare, ast.UnaryOp, ast.BoolOp)):
            yield ast.UnaryOp(ast.Not(), node), 3
    if isinstance(node, _DELETABLE) and not isinstance(parent, ast.Module):
        yield b"pass", 3


def _call_variants(node: ast.Call) -> Iterator[tuple[ast.AST, int]]:
    func = node.func
    if isinstance(func, ast.Name):
        for family in _BUILTIN_FAMILIES:
            if func.id in family:
                for other in family:
                    if other != func.id:
                        yield _with(node, func=ast.Name(other, ast.Load())), 0 if func.id != "all" and func.id != "any" else 1
        if func.id in _UNWRAP and len(node.args) == 1 and not node.keywords:
            yield node.args[0], 3
        if func.id == "range":
            for index, arg in enumerate(node.args):
                if not isinstance(arg, ast.Constant):
                    for shifted in _shifts(arg):
                        args = list(node.args)
                        args[index] = shifted
                        yield _with(node, args=args), 2
    elif isinstance(func, ast.Attribute):
        for family in _METHOD_FAMILIES:
            if func.attr in family:
                for other in family:
                    if other != func.attr:
                        yield _with(node, func=_with(func, attr=other)), 0 if len(family) == 2 else 1
    name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
    if name in ("sorted", "sort"):
        keywords = [keyword for keyword in node.keywords if keyword.arg != "reverse"]
        if len(keywords) == len(node.keywords):
            keywords.append(ast.keyword("reverse", ast.Constant(True)))
        yield _with(node, keywords=keywords), 1


def _subscript_variants(node: ast.Subscript) -> Iterator[tuple[ast.AST, int]]:
    index = node.slice
    if isinstance(index, ast.Slice):
        for field in ("lower", "upper"):
            bound = getattr(index, field)
            if bound is not None and not isinstance(bound, ast.Constant):
                for shifted in _shifts(bound):
                    yield _with(node, slice=_with(index, **{field: shifted})), 2
        if index.step is None and (index.lower is None) != (index.upper is None):
            yield _with(node, slice=_with(index, lower=index.upper, upper=index.lower)), 2
        if index.step is not None:
            yield _with(node, slice=_with(index, step=None)), 2
        else:
            yield _with(node, slice=_with(index, step=ast.UnaryOp(ast.USub(), ast.Constant(1)))), 3
    elif not isinstance(index, (ast.Constant, ast.Tuple)):
        for shifted in _shifts(index):
            yield _with(node, slice=shifted), 2


def _name_swaps(unit: ast.AST) -> Iterator[tuple[ast.Name, ast.AST, int]]:
    for function in ast.walk(unit):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = function.args
        local = {arg.arg for arg in arguments.posonlyargs + arguments.args + arguments.kwonlyargs}
        local |= {
            node.id for node in ast.walk(function)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        for node in ast.walk(function):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in local:
                for other in sorted(local - {node.id}):
                    yield node, ast.Name(other, ast.Load()), 3


def _walk(node: ast.AST, parent: ast.AST | None = None) -> Iterator[tuple[ast.AST, ast.AST | None]]:
    yield node, parent
    for child in ast.iter_child_nodes(node):
        if not isinstance(child, ast.JoinedStr):
            yield from _walk(child, node)


def _units(tree: ast.Module) -> dict[str, list[ast.stmt]]:
    units: dict[str, list[ast.stmt]] = {}
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            units[statement.name] = [statement]
        elif not isinstance(statement, (ast.Import, ast.ImportFrom)) and not (
            isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
        ):
            units.setdefault("<module>", []).append(statement)
    return units


def _edits(source: bytes, units: dict[str, list[ast.stmt]], names: Iterable[str]) -> list[_Edit]:
    starts = [0]
    for line in source.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))

    def span(node: ast.AST) -> tuple[int, int] | None:
        if getattr(node, "end_col_offset", None) is None:
            return None
        return starts[node.lineno - 1] + node.col_offset, starts[node.end_lineno - 1] + node.end_col_offset

    seen: set[tuple[int, int, bytes]] = set()
    edits: list[_Edit] = []

    def add(node: ast.AST, replacement: ast.AST | bytes, priority: int, unit: str) -> None:
        position = span(node)
        if position is None:
            return
        start, end = position
        text = replacement if isinstance(replacement, bytes) else _text(replacement, node)
        original = source[start:end]
        if text == original or (start, end, text) in seen:
            return
        seen.add((start, end, text))
        label = f"line {node.lineno}: {original.decode('utf-8', 'replace')} -> {text.decode('utf-8')}"
        edits.append(_Edit(start, end, text, priority, unit, label))

    for unit in names:
        for top in units[unit]:
            for node, parent in _walk(top):
                if node is top and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if not isinstance(getattr(node, "ctx", ast.Load()), ast.Load):
                    continue
                for replacement, priority in _variants(node, parent):
                    add(node, replacement, priority, unit)
            for node, replacement, priority in _name_swaps(top):
                add(node, replacement, priority, unit)
    edits.sort(key=lambda edit: edit.priority)
    return edits


def _suspects(test_source: bytes, module: str, failures: Failures, units: dict[str, list[ast.stmt]]) -> set[str]:
    everything = set(units)
    if not failures:
        return everything
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return everything
    aliases: set[str] = set()
    imported: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases |= {alias.asname or alias.name for alias in node.names if alias.name == module}
        elif isinstance(node, ast.ImportFrom) and node.module == module:
            imported |= {alias.asname or alias.name: alias.name for alias in node.names}
    failing = {failure.split(" ", 1)[0] for failure in failures}
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in failing:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id in aliases:
                    used.add(sub.attr)
                elif isinstance(sub, ast.Name) and sub.id in imported:
                    used.add(imported[sub.id])
    used &= everything
    if not used:
        return everything
    stack = list(used)
    while stack:
        references = {
            node.id for statement in units[stack.pop()] for node in ast.walk(statement) if isinstance(node, ast.Name)
        }
        for name in references & everything - used:
            used.add(name)
            stack.append(name)
    if "<module>" in units:
        used.add("<module>")
    return used


class _Bench:
    def __init__(self, task: TaskSpec, target: str, script: str, scratch: Path, deadline: float):
        self.task, self.target, self.script, self.deadline = task, target, script, deadline
        self.timeout = 10.0
        self.cache: dict[bytes, Failures] = {}
        self.runs = 0
        self.slots: SimpleQueue[Path] = SimpleQueue()
        for index in range(WORKERS):
            copy_root = scratch / f"ws{index}"
            shutil.copytree(task.workspace, copy_root, symlinks=True)
            self.slots.put(copy_root)
        self.pool = ThreadPoolExecutor(WORKERS)

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)

    def _run(self, source: bytes) -> Failures:
        if time.monotonic() >= self.deadline:
            return None
        try:
            compile(source, self.target, "exec")
        except (SyntaxError, ValueError):
            return None
        root = self.slots.get()
        try:
            resolve_inside(root, self.target).write_bytes(source)
            result = run_python_tests(replace(self.task, workspace=root), self.script, timeout=self.timeout)
            self.runs += 1
            return _failures(result)
        finally:
            self.slots.put(root)

    def evaluate(self, sources: list[bytes]) -> list[Failures]:
        missing = [source for source in dict.fromkeys(sources) if source not in self.cache]
        for source, failures in zip(missing, self.pool.map(self._run, missing)):
            if time.monotonic() < self.deadline:
                self.cache[source] = failures
        return [self.cache.get(source) for source in sources]

    def calibrate(self, source: bytes) -> Failures:
        started = time.monotonic()
        failures = self.evaluate([source])[0]
        self.timeout = min(10.0, max(2.0, 5 * (time.monotonic() - started) + 1))
        return failures


def _search(bench: _Bench, source: bytes, test_source: bytes, module: str) -> tuple[bytes, list[str]] | None:
    failures = bench.calibrate(source)
    applied: list[str] = []
    for _ in range(MAX_ROUNDS):
        if failures == frozenset():
            return source, applied
        if time.monotonic() >= bench.deadline:
            return None
        try:
            units = _units(ast.parse(source))
        except SyntaxError:
            return None
        suspects = _suspects(test_source, module, failures, units)
        scopes = [suspects] if suspects == set(units) else [suspects, set(units)]
        step = None
        for scope in scopes:
            edits = _edits(source, units, sorted(scope))
            step = _best_single(bench, source, failures, edits)
            if step is not None:
                chosen, outcome, improving = step
                _check_unambiguous(bench, source, chosen, improving, test_source, module)
                step = chosen, outcome
                break
            if scope is suspects:
                step = _best_pair(bench, source, failures, edits)
                if step is not None:
                    break
        if step is None:
            return None
        edits, failures = step
        source = _apply(source, edits)
        applied += [edit.label for edit in edits]
    return (source, applied) if failures == frozenset() else None


def _best_single(bench: _Bench, source: bytes, failures: Failures, edits: list[_Edit]):
    outcomes = bench.evaluate([_apply(source, [edit]) for edit in edits])
    improving = [(edit, outcome) for edit, outcome in zip(edits, outcomes) if _better(outcome, failures)]
    if not improving:
        return None
    best_edit, best = min(improving, key=lambda item: (_rank(item[1]), item[0].priority))
    per_unit: dict[str, tuple[_Edit, Failures]] = {}
    for edit, outcome in improving:
        current = per_unit.get(edit.unit)
        if current is None or (_rank(outcome), edit.priority) < (_rank(current[1]), current[0].priority):
            per_unit[edit.unit] = (edit, outcome)
    if len(per_unit) > 1:
        combined = [edit for edit, _ in per_unit.values()]
        if _disjoint(combined):
            outcome = bench.evaluate([_apply(source, combined)])[0]
            if _better(outcome, failures) and _rank(outcome) < _rank(best):
                return combined, outcome, improving
    return [best_edit], best, improving


class Ambiguous(Exception):
    pass


_PROBE_DRIVER = r"""
import json, signal, sys, types
data = json.load(sys.stdin)
def _alarm(*_):
    raise TimeoutError
if hasattr(signal, "setitimer"):
    signal.signal(signal.SIGALRM, _alarm)
    def timer(seconds):
        signal.setitimer(signal.ITIMER_REAL, seconds)
else:
    def timer(seconds):
        pass
rows = []
for source in data["sources"]:
    module = types.ModuleType(data["module"])
    module.__file__ = data["file"]
    sys.modules[data["module"]] = module
    try:
        timer(2)
        exec(compile(source, data["file"], "exec"), module.__dict__)
        timer(0)
        function = getattr(module, data["function"])
    except BaseException as error:
        timer(0)
        rows.append(["LOAD:" + type(error).__name__])
        continue
    row = []
    for args in data["probes"]:
        try:
            timer(0.5)
            row.append(repr(function(*eval(args))))
        except BaseException as error:
            row.append("EXC:" + type(error).__name__)
        finally:
            timer(0)
    rows.append(row)
print(json.dumps(rows))
"""


def _alternatives(value) -> list:
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, int):
        return [value + 1, value - 1, 0, -value, 3 * value + 7]
    if isinstance(value, float):
        return [value + 1.5, -value, 0.0]
    if isinstance(value, str):
        words = value.split()
        return [value.upper(), value.lower(), value.title(), "  " + "   ".join(words) + " ", value[::-1], "",
                value + " Extra words", " ".join(reversed(words))]
    if isinstance(value, (list, tuple)):
        kind = type(value)
        items = list(value)
        variants = [items[::-1], sorted(items, key=repr), [], items + items[:1], items[1:], items * 2]
        if items and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in items):
            variants += [sorted(items), sorted(items, reverse=True), [-item for item in items],
                         items + [max(items) + 5], [min(items) - 3] + items]
        return [kind(variant) for variant in variants]
    return []


def _literal_calls(tree: ast.Module, module: str, function: str) -> list[tuple]:
    aliases = {alias.asname or alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
               for alias in node.names if alias.name == module}
    direct = {alias.asname or alias.name for node in ast.walk(tree)
              if isinstance(node, ast.ImportFrom) and node.module == module
              for alias in node.names if alias.name == function}
    seeds = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or node.keywords:
            continue
        target = node.func
        if (isinstance(target, ast.Attribute) and target.attr == function and isinstance(target.value, ast.Name)
                and target.value.id in aliases) or (isinstance(target, ast.Name) and target.id in direct):
            try:
                seeds.append(tuple(ast.literal_eval(arg) for arg in node.args))
            except (ValueError, SyntaxError, TypeError):
                continue
    return seeds


def _probes(test_source: bytes, module: str, function: str) -> list[str]:
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return []
    probes: list[str] = []
    for seed in _literal_calls(tree, module, function):
        for index, value in enumerate(seed):
            for alternative in _alternatives(value):
                text = repr(seed[:index] + (alternative,) + seed[index + 1:])
                if text not in probes:
                    probes.append(text)
    return probes[:MAX_PROBES]


def _behaviour(bench: "_Bench", sources: list[bytes], function: str, module: str, probes: list[str]):
    timeout = min(bench.deadline - time.monotonic(), max(5.0, 0.6 * len(probes) * len(sources) + 2))
    if timeout <= 0:
        return None
    root = bench.slots.get()
    try:
        payload = json.dumps({"sources": [source.decode("utf-8", "replace") for source in sources],
                              "module": module, "file": str(resolve_inside(root, bench.target)),
                              "function": function, "probes": probes})
        with tempfile.TemporaryDirectory(prefix="jiuwen-probe-") as scratch:
            driver = Path(scratch) / "probe.py"
            driver.write_text(_PROBE_DRIVER, encoding="utf-8")
            completed = run_python(driver, cwd=root, timeout=timeout, writable=False, input_data=payload)
        return json.loads(completed.output) if completed.passed else None
    except (OSError, ValueError):
        return None
    finally:
        bench.slots.put(root)


def _check_unambiguous(bench, source: bytes, chosen: list[_Edit], improving, test_source: bytes, module: str):
    outcome_of = {edit: outcome for edit, outcome in improving}
    for edit in chosen:
        if edit.priority >= LOW_PLAUSIBILITY:
            raise Ambiguous(f"low-plausibility repair {edit.label}")
        rivals = [other for other, outcome in improving if other is not edit and other.unit == edit.unit
                  and outcome == outcome_of[edit] and other.priority <= edit.priority + RIVAL_WINDOW]
        if not rivals:
            continue
        probes = _probes(test_source, module, edit.unit)
        if not probes:
            continue
        candidates = [edit] + sorted(rivals, key=lambda other: other.priority)[:15]
        rows = _behaviour(bench, [_apply(source, [item]) for item in candidates], edit.unit, module, probes)
        if not isinstance(rows, list) or len(rows) != len(candidates):
            raise Ambiguous(f"could not compare {edit.label} against alternative repairs")
        differing = [other.label for other, row in zip(candidates[1:], rows[1:]) if row != rows[0]]
        if differing:
            raise Ambiguous(f"{edit.label} vs {differing[0]} (+{len(differing) - 1} more) disagree on probes")


def _best_pair(bench: _Bench, source: bytes, failures: Failures, edits: list[_Edit]):
    by_unit: dict[str, list[_Edit]] = {}
    for edit in edits:
        by_unit.setdefault(edit.unit, []).append(edit)
    pairs = [
        pair for unit_edits in by_unit.values() for pair in combinations(unit_edits, 2) if _disjoint(pair)
    ]
    pairs.sort(key=lambda pair: pair[0].priority + pair[1].priority)
    chunk = WORKERS * 8
    for offset in range(0, min(len(pairs), MAX_PAIR_RUNS), chunk):
        if time.monotonic() >= bench.deadline:
            return None
        batch = pairs[offset:offset + chunk]
        outcomes = bench.evaluate([_apply(source, pair) for pair in batch])
        improving = [(list(pair), outcome) for pair, outcome in zip(batch, outcomes) if _better(outcome, failures)]
        if improving:
            return min(improving, key=lambda item: _rank(item[1]))
    return None


def _locate(task: TaskSpec, root: Path) -> tuple[str, list[str]] | SolveResult:
    script_match = _SCRIPT.search(task.instruction)
    target_match = _TARGET.search(task.instruction)
    script = script_match["script"] if script_match else None
    targets = re.findall(_MODULE_NAME, target_match["targets"]) if target_match else []
    if script is None:
        tests = sorted(path.name for path in root.glob("test_*.py"))
        if len(tests) != 1:
            return SolveResult.unsupported("cannot identify the test script")
        script = tests[0]
    if not targets:
        modules = sorted(
            path.name for path in root.glob("*.py") if path.name != script and not path.name.startswith("test_")
        )
        if len(modules) != 1:
            return SolveResult.unsupported("cannot identify the module to fix")
        targets = modules
    for name in [script, *targets]:
        if not resolve_inside(root, name).is_file():
            return SolveResult.unsolvable(f"{name!r} named by the task does not exist")
    paths = {resolve_inside(root, name) for name in targets}
    if len(paths) != len(targets) or any(root / name != resolve_inside(root, name) for name in targets):
        return SolveResult.unsupported("module list contains an alias")
    if resolve_inside(root, script) in paths or any(p.name.startswith("test_") or p.stem.endswith("_test") for p in paths):
        return SolveResult.unsupported("the module to fix is a test script")
    return script, targets


def _contradiction(test_source: bytes, module_source: bytes, module: str) -> str | None:
    try:
        tests, tree = ast.parse(test_source), ast.parse(module_source)
    except SyntaxError:
        return None
    functions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and not any(isinstance(sub, (ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom))
                             for sub in ast.walk(node))}
    aliases = {alias.asname or alias.name for node in ast.walk(tests) if isinstance(node, ast.Import)
               for alias in node.names if alias.name == module}
    seen: dict[str, tuple[str, object]] = {}
    for node in ast.walk(tests):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "assertEqual" and len(node.args) == 2):
            continue
        call, expected = node.args
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name) and call.func.value.id in aliases
                and call.func.attr in functions and not call.keywords):
            continue
        try:
            arguments = tuple(ast.literal_eval(arg) for arg in call.args)
            value = ast.literal_eval(expected)
        except (ValueError, SyntaxError, TypeError):
            continue
        key = f"{call.func.attr}{arguments!r}"
        if key in seen and seen[key][1] != value:
            return f"{key} is expected to equal both {seen[key][1]!r} and {value!r}"
        seen.setdefault(key, (call.func.attr, value))
    return None


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-")
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(data)
        shutil.copymode(path, temp_name)
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def precheck(task: TaskSpec) -> SolveResult:
    task = task if task.workspace is not None else replace(task, workspace=task.task_dir)
    try:
        located = _locate(task, task.workspace)
        if isinstance(located, SolveResult):
            return located
        script, targets = located
        test_source = resolve_inside(task.workspace, script).read_bytes()
        for target in targets:
            source = resolve_inside(task.workspace, target).read_bytes()
            contradiction = _contradiction(test_source, source, ".".join(Path(target).with_suffix("").parts))
            if contradiction:
                return SolveResult.unsolvable(f"the tests contradict each other: {contradiction}")
    except (OSError, ValueError) as error:
        return SolveResult.error(f"cannot prepare coding task: {error}")
    with tempfile.TemporaryDirectory(prefix="jiuwen-precheck-") as scratch:
        root = Path(scratch) / "workspace"
        shutil.copytree(task.workspace, root, symlinks=True)
        result = run_python_tests(replace(task, workspace=root), script, timeout=PRECHECK_TIMEOUT)
        if not result.passed:
            patches = {}
            for target in targets:
                path = resolve_inside(task.workspace, target)
                original = path.read_bytes()
                candidate = isolated_defaults(original)
                module = ".".join(Path(target).with_suffix("").parts)
                shaped = shape_repairs.candidate(candidate or original, test_source, module)
                candidate = shaped or candidate
                if candidate is not None:
                    (root / target).write_bytes(candidate)
                    patches[path] = (original, candidate)
            if patches and run_python_tests(replace(task, workspace=root), script, timeout=PRECHECK_TIMEOUT).passed:
                accepted = False
                written = []
                try:
                    for path, (original, candidate) in patches.items():
                        _write_atomic(path, candidate)
                        written.append((path, original))
                    check = run_python_tests(task, script, timeout=PRECHECK_TIMEOUT)
                    accepted = check.passed and resolve_inside(task.workspace, script).read_bytes() == test_source
                    if accepted:
                        return SolveResult.solved("DONE", "local source repair; original tests pass in place")
                finally:
                    if not accepted:
                        for path, original in reversed(written):
                            _write_atomic(path, original)
    if result.passed:
        return SolveResult.solved("DONE", "original tests already pass")
    return SolveResult.unsupported("precheck timed out; needs longer verification or repair" if result.timed_out
                                   else "original tests fail; repair needed")


def solve(task: TaskSpec) -> SolveResult:
    deadline = time.monotonic() + TIME_LIMIT
    task = task if task.workspace is not None else replace(task, workspace=task.task_dir)
    try:
        located = _locate(task, task.workspace)
        if isinstance(located, SolveResult):
            return located
        script, targets = located
        script_path = resolve_inside(task.workspace, script)
        test_source = script_path.read_bytes()
        script_digest = _digest(script_path)
        sources = {name: resolve_inside(task.workspace, name).read_bytes() for name in targets}
    except (OSError, ValueError) as error:
        return SolveResult.error(f"cannot prepare coding task: {error}")

    for name, original in sources.items():
        module = ".".join(Path(name).with_suffix("").parts)
        if contradiction := _contradiction(test_source, original, module):
            return SolveResult.unsolvable(f"the tests contradict each other: {contradiction}")
    runs = 0
    ambiguous = None
    for index, (target, original) in enumerate(sources.items()):
        now = time.monotonic()
        if now >= deadline:
            break
        cutoff = now + (deadline - now) / (len(sources) - index)
        bom = codecs.BOM_UTF8 if original.startswith(codecs.BOM_UTF8) else b""
        body = original[len(bom):]
        module = ".".join(Path(target).with_suffix("").parts)
        try:
            with tempfile.TemporaryDirectory(prefix="jiuwen-coding-") as scratch:
                bench = _Bench(task, target, script, Path(scratch), cutoff)
                try:
                    found = _search(bench, body, test_source, module)
                except Ambiguous as error:
                    ambiguous = str(error)
                    found = None
                finally:
                    bench.close()
                    runs += bench.runs
        except OSError as error:
            return SolveResult.error(f"cannot evaluate repairs: {error}")
        if found is not None:
            target_path = resolve_inside(task.workspace, target)
            return _verify_repair(task, target_path, original, bom, found, script, script_digest, runs)
    if ambiguous:
        return SolveResult.unsupported(f"ambiguous repair: {ambiguous}")
    return SolveResult.unsupported(f"no verified repair within limits after {runs} test runs")


def _verify_repair(task, target_path, original, bom, found, script, script_digest, runs):
    repaired, applied = found
    accepted = False
    try:
        if bom + repaired != original:
            _write_atomic(target_path, bom + repaired)
        result = run_python_tests(task, script, timeout=10.0)
        if not result.passed or _digest(resolve_inside(task.workspace, script)) != script_digest:
            return SolveResult.unsupported("repair did not pass the original tests in place; restored")
        accepted = True
    except (OSError, ValueError) as error:
        return SolveResult.error(f"cannot apply repair: {error}")
    finally:
        if not accepted and bom + repaired != original:
            _write_atomic(target_path, original)
    detail = "; ".join(applied) if applied else "original tests already pass"
    return SolveResult.solved("DONE", f"{detail} ({runs} test runs)")
