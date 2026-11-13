import ast
import codecs


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
        for index in ast.walk(target):
            if isinstance(index, ast.Subscript) and isinstance(index.slice, ast.Constant):
                index.slice.value ^= 1
                match = _shape(expr) == _shape(target)
                index.slice.value ^= 1
                if match:
                    yield expr, correct
                    return


def candidate(original: bytes, tests: bytes) -> bytes | None:
    bom = codecs.BOM_UTF8 if original.startswith(codecs.BOM_UTF8) else b""
    body = original[len(bom):]
    try:
        tree = ast.parse(body)
    except (SyntaxError, ValueError):
        return None
    edits = []
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and fn.name.encode() in tests:
            edits.extend(_transpose_edits(fn))
            edits.extend(_determinant_edits(fn))
    if not edits:
        return None
    offsets = [0]
    for line in body.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    for node, text in sorted(edits, key=lambda pair: (pair[0].lineno, pair[0].col_offset), reverse=True):
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        body = body[:start] + text.encode() + body[end:]
    return bom + body
