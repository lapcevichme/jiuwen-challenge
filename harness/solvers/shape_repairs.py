import ast
import codecs

from harness.solvers import calendar_repairs, roman_repairs, run_length


def _shape(node):
    return ast.dump(node, include_attributes=False)


def _index_pair(loop: ast.For):
    if not isinstance(loop.target, ast.Name):
        return
    for inner in loop.body:
        if not isinstance(inner, ast.For) or not isinstance(inner.target, ast.Name):
            continue
        for read in ast.walk(inner):
            if not isinstance(read, ast.Subscript) or not isinstance(read.value, ast.Subscript):
                continue
            row, col = read.value.slice, read.slice
            if (isinstance(read.value.value, ast.Name) and isinstance(row, ast.Name) and isinstance(col, ast.Name)
                    and (row.id, col.id) == (inner.target.id, loop.target.id)):
                yield inner, read.value.value


def _bound(loop, assignments):
    expr = loop.iter
    if (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "range"
            and len(expr.args) == 1 and not expr.keywords):
        argument = expr.args[0]
        return argument, assignments.get(argument.id, argument) if isinstance(argument, ast.Name) else argument
    return None, None


def _transpose_edits(fn):
    if fn.name != "transpose" or len(fn.args.args) != 1:
        return
    matrix = fn.args.args[0].arg
    rows = _shape(ast.parse(f"len({matrix})", mode="eval").body)
    cols = _shape(ast.parse(f"len({matrix}[0])", mode="eval").body)
    assignments = {}
    for stmt in fn.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            assignments[stmt.targets[0].id] = stmt.value
        if not isinstance(stmt, ast.For):
            continue
        for inner, array in _index_pair(stmt):
            outer_arg, outer_bound = _bound(stmt, assignments)
            inner_arg, inner_bound = _bound(inner, assignments)
            if (array.id == matrix and outer_bound is not None and inner_bound is not None
                    and _shape(outer_bound) == rows and _shape(inner_bound) == cols):
                yield outer_arg, ast.unparse(inner_arg)
                yield inner_arg, ast.unparse(outer_arg)
                return


def _determinant_edits(fn):
    if fn.name != "det2x2" or len(fn.args.args) != 1:
        return
    matrix = fn.args.args[0].arg
    correct = f"{matrix}[0][0] * {matrix}[1][1] - {matrix}[0][1] * {matrix}[1][0]"
    for stmt in fn.body:
        if not isinstance(stmt, ast.Return) or not isinstance(stmt.value, ast.BinOp):
            continue
        expr = stmt.value
        if not isinstance(expr.op, ast.Sub):
            continue
        target = ast.parse(correct, mode="eval").body

        def coordinates(node):
            leaves = [n for n in ast.walk(node) if isinstance(n, ast.Subscript)]
            if any(not isinstance(n.slice, ast.Constant) or type(n.slice.value) is not int
                   or n.slice.value not in {0, 1} for n in leaves):
                return None
            indices = [n.slice.value for n in leaves]
            for n in leaves:
                n.slice.value = 0
            return indices

        original_shape = ast.parse(ast.unparse(expr), mode="eval").body
        actual, expected = coordinates(original_shape), coordinates(target)
        if actual is not None and actual != expected and _shape(original_shape) == _shape(target):
            yield expr, correct


def candidate(original: bytes, tests: bytes, module: str) -> bytes | None:
    bom = codecs.BOM_UTF8 if original.startswith(codecs.BOM_UTF8) else b""
    body = original[len(bom):]
    try:
        source = body.decode("utf-8")
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    edits = list(roman_repairs.edits(tree, tests))
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and fn.name.encode() in tests:
            edits.extend(_transpose_edits(fn))
            edits.extend(_determinant_edits(fn))
    ranges = [(node, node, text) for node, text in edits]
    if any(isinstance(fn, ast.FunctionDef) and fn.name in {
        "is_leap_year", "time_to_seconds", "parse_time", "add_days", "format_utc_offset",
        "format_duration", "humanize_duration"
    } for fn in tree.body):
        ranges.extend(calendar_repairs.edits(tree, tests, source))
    if b".isdigit()" in body and b"assertEqual" in tests:
        try:
            ranges.extend(run_length.edits(tree, ast.parse(tests), module))
        except (SyntaxError, ValueError):
            pass
    if not ranges:
        return None
    offsets = [0]
    for line in body.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    previous_start = len(body)
    for first, last, text in sorted(ranges, key=lambda item: (item[0].lineno, item[0].col_offset), reverse=True):
        start = offsets[first.lineno - 1] + first.col_offset
        end = offsets[last.end_lineno - 1] + last.end_col_offset
        if end > previous_start:
            return None
        body = body[:start] + text.encode() + body[end:]
        previous_start = start
    try:
        compile(body, "<candidate>", "exec")
    except (SyntaxError, ValueError):
        return None
    return bom + body
