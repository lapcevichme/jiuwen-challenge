import ast

_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_PAIRS = {"IV": 4, "IX": 9, "XL": 40, "XC": 90, "CD": 400, "CM": 900}
_DIGITS = sorted((_VALUES | _PAIRS).items(), key=lambda item: -item[1])


def _literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def _shape(node):
    return ast.dump(node)


def _tables(block, fn):
    arrays = {n.targets[0].id: (n.value, _literal(n.value)) for n in block
              if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
              and isinstance(n.value, ast.List)}
    for first, (nums, values) in arrays.items():
        for second, (syms, symbols) in arrays.items():
            if not (isinstance(values, list) and isinstance(symbols, list)
                    and 7 <= len(values) == len(symbols) < len(_DIGITS)
                    and all(type(v) is int for v in values) and all(isinstance(s, str) for s in symbols)):
                continue
            observed = list(zip(symbols, values))
            if ([pair for pair in _DIGITS if pair in observed] != observed
                    or not _VALUES.items() <= set(observed)):
                continue
            accesses = {name: set() for name in (first, second)}
            for sub in ast.walk(fn):
                if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name) and sub.value.id in accesses:
                    accesses[sub.value.id].add(_shape(sub.slice))
            if accesses[first] & accesses[second]:
                yield nums, repr([v for _, v in _DIGITS])
                yield syms, repr([k for k, _ in _DIGITS])
                return


def _progress(fn):
    for loop in ast.walk(fn):
        if not isinstance(loop, ast.While) or not isinstance(loop.test, ast.Compare):
            continue
        test = loop.test
        if not (len(test.ops) == 1 and isinstance(test.left, ast.Name)
                and isinstance(test.ops[0], (ast.Gt, ast.GtE))):
            continue
        value = test.comparators[0]
        if not isinstance(value, (ast.Name, ast.Constant, ast.Subscript)):
            continue
        decrements = [n for n in loop.body if isinstance(n, ast.AugAssign) and isinstance(n.op, ast.Sub)
                      and isinstance(n.target, ast.Name) and n.target.id == test.left.id]
        if len(decrements) != 1:
            continue
        step = decrements[0].value
        if _literal(step) == 0 and (not isinstance(value, ast.Constant)
                                    or type(value.value) is int and value.value > 0):
            yield step, ast.unparse(value)
        if isinstance(test.ops[0], ast.Gt) and _shape(step) == _shape(value):
            yield test, f"{test.left.id} >= {ast.unparse(value)}"


def edits(tree, tests):
    if not any(isinstance(n, ast.Dict) and _literal(n) == _VALUES for n in ast.walk(tree)):
        return
    if not any(repr(k).encode() in tests or f'"{k}"'.encode() in tests for k in _PAIRS):
        return
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        if fn.name in {"to_roman", "int_to_roman"}:
            yield from _tables([*tree.body, *fn.body], fn)
            yield from _progress(fn)
        if fn.name in {"from_roman", "roman_to_int"}:
            for stmt in tree.body:
                if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                    continue
                mapping = _literal(stmt.value)
                if (isinstance(mapping, dict) and 2 <= len(mapping) < 6
                        and all(k in _PAIRS and _PAIRS[k] == v for k, v in mapping.items())
                        and any(isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                                and n.value.id == stmt.targets[0].id for n in ast.walk(fn))):
                    yield stmt.value, repr(_PAIRS)
            for loop in ast.walk(fn):
                if not isinstance(loop, ast.While) or not isinstance(loop.test, ast.Compare):
                    continue
                index = loop.test.left
                if not isinstance(index, ast.Name) or len(loop.body) != 1 or not isinstance(loop.body[0], ast.If):
                    continue
                branch = loop.body[0]
                if (len(branch.orelse) == 1 and isinstance(branch.orelse[0], ast.AugAssign)
                        and isinstance(branch.orelse[0].value, ast.Subscript)
                        and any(isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name)
                                and n.target.id == index.id and isinstance(n.op, ast.Add) and _literal(n.value) == 2
                                for n in branch.body)):
                    stmt = branch.orelse[0]
                    yield stmt, ast.unparse(stmt) + "\n" + " " * stmt.col_offset + index.id + " += 1"
